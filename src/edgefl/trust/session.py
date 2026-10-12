"""Phase E round composition. Only this simulator coordinator joins attack and defense."""

import copy
import math
import pickle
import time
from dataclasses import asdict, replace
from edgefl.config import workspace_path
from edgefl.contracts.records import Partition, PartitionRef, ClientUpdate, require_compatible
from edgefl.contracts.interfaces import AggregationRequest, ScoringRequest
from edgefl.contracts.phase_e import SubmissionDisposition
from edgefl.learning import checkpoints
from edgefl.learning.federated import BaselineSession, FederatedRound
from edgefl.learning.aggregation import BaselineAggregator
from edgefl.learning.metrics import magnitude
from edgefl.learning.storage import reference, checked
from edgefl.data.storage import write_json
from edgefl.reproducibility import derive_seed
from edgefl.attacks.trainer import AttackTrainer
from edgefl.attacks.diagnostics import summarize as truth_summary
from edgefl.trust.scoring import TrustedClientScorer
from edgefl.trust.reputation import ReputationStore
from edgefl.trust.aggregation import FixedWeightAggregator, validate_weights


def validate_submission(root, update, expected, parent, count):
    if not isinstance(update, ClientUpdate):
        raise ValueError("Invalid submission record")
    if update.client_id != expected.client.client_id or update.round_id != expected.round_id or update.parent_model != expected.parent_model:
        raise ValueError("Unexpected client/round/parent")
    require_compatible(expected.compatibility, update.compatibility)
    if type(update.registered_training_count) is not int or update.registered_training_count != count:
        raise ValueError("Unregistered training count")
    if not math.isfinite(update.training_seconds) or update.training_seconds < 0 or type(update.transmitted_bytes) is not int or update.transmitted_bytes < 0:
        raise ValueError("Invalid submitted timing/size metadata")
    try:
        delta = checkpoints.load(workspace_path(root, update.delta.path), update.delta.sha256)["delta"]
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, RuntimeError) as exc:
        raise ValueError("Missing or unreadable submitted update") from exc
    checkpoints.validate_state(delta, parent)
    checkpoints.validate_state({k: parent[k] + delta[k] for k in parent}, parent)
    return delta


class TrustSession(BaselineSession):
    def __init__(self, config, data, directory, comp, data_ref, method, seed, *, settings, metadata, plan, calibration, bindings):
        super().__init__(config, data, directory, comp, data_ref, method, seed)
        self.settings, self.plan, self.calibration, self.bindings = settings, plan, calibration, bindings
        self.profiles = metadata.spec["profiles"]
        if sorted(self.counts) != plan["clients"] or any(self.counts[c] != self.profiles[c]["count"] for c in self.counts):
            raise ValueError("Attack/metadata client population mismatch")
        self.trainer = AttackTrainer(self.trainer, plan)
        self.reputation = ReputationStore(self.counts, settings["reputation_initial"], settings["reputation_retention"])
        self.frozen = None
        self.effective_count = 0
        self.round_artifacts = {}
        self.trusted_ref = PartitionRef(data_ref, Partition.TRUSTED)
        originals = [r["original"] for r in metadata.rows("trusted")]
        vocabulary = sorted(set(originals) | {name for p in self.profiles.values() for name in p["support"]})
        self.specialist = metadata.spec["declared_specialist_scenario"]
        self.scorer = TrustedClientScorer(config.workspace, data.view("trusted"), originals, self.trusted_ref, comp,
            data.spec["labels"], data.spec["identity"]["task"], self.values, vocabulary,
            settings["expertise_support_scale"], settings["minimum_valid_clients"])

    def select(self, current):
        return sorted(self.counts) if current == 1 else super().select(current)

    def state_dict(self):
        return {"schema_version": "phase-e.session.v1", "bindings": copy.deepcopy(self.bindings),
                "reputation": self.reputation.state_dict(), "frozen_weights": copy.deepcopy(self.frozen),
                "effective_attacking_client_rounds": self.effective_count, "round_artifacts": copy.deepcopy(self.round_artifacts)}

    def load_state_dict(self, state):
        if set(state) != {"schema_version", "bindings", "reputation", "frozen_weights", "effective_attacking_client_rounds", "round_artifacts"} or state["schema_version"] != "phase-e.session.v1" or state["bindings"] != self.bindings:
            raise ValueError("Phase E session identity mismatch")
        frozen = state["frozen_weights"]
        if frozen is not None:
            validate_weights(frozen)
            if set(frozen) != set(self.counts) or self.method != "frozen":
                raise ValueError("Invalid frozen snapshot")
        self.reputation.load_state_dict(state["reputation"])
        self.frozen = copy.deepcopy(frozen)
        if type(state["effective_attacking_client_rounds"]) is not int or state["effective_attacking_client_rounds"] < 0:
            raise ValueError("Invalid simulator coverage counter")
        self.effective_count = state["effective_attacking_client_rounds"]
        for current, refs in state["round_artifacts"].items():
            if not current.isdigit() or int(current) < 1 or set(refs) != {"assessment", "simulator", "reputation"}:
                raise ValueError("Invalid round artifact index")
            for ref in refs.values():
                checked(self.config.workspace, ref)
        self.round_artifacts = copy.deepcopy(state["round_artifacts"])

    def create_aggregator(self, assessments, root):
        if self.method in ("adaptive", "frozen"):
            return FixedWeightAggregator(self.config.workspace, self.directory, self.method, self.counts,
                                         self.settings["weight_floor"], self.frozen), assessments
        return BaselineAggregator(self.config.workspace, self.directory, self.method, self.counts,
                                  self.values["trim_fraction"], root), ()

    def execute(self, current, parent_ref, parent):
        selected, updates, reports, invalid = self.select(current), [], {}, {}
        prior = self.reputation.prior()
        training_seconds = 0.
        for client in selected:
            request = self.request(client, current, parent_ref)
            update = self.trainer.train(request)
            training_seconds += self.trainer.diagnostics[client]["seconds"]
            try:
                delta = validate_submission(self.config.workspace, update, request, parent, self.counts[client])
            except (ValueError, KeyError) as exc:
                invalid[client] = str(exc)
                continue
            updates.append(update)
            reports[client] = {**self.trainer.diagnostics[client], "update": magnitude(delta, parent), "transmitted_bytes": update.transmitted_bytes}
        prefix = self.directory / f"assessment-{current:04d}"
        write_json(prefix.with_suffix(".submissions.json"), {"selected": selected, "invalid": invalid, "updates": [asdict(u) for u in updates]})
        if len(updates) < self.settings["minimum_valid_clients"]:
            raise ValueError("Insufficient valid clients for peer risk")
        started = time.perf_counter()
        observation = self.scorer.observe(ScoringRequest(self.trusted_ref, parent_ref, tuple(updates), self.comp))
        invalid.update(observation["invalid_predictions"])
        updates = [u for u in updates if u.client_id not in invalid]
        assessments = self.scorer.assessments(observation, self.calibration, prior) if self.calibration else ()
        if self.values["device"] == "cuda":
            import torch
            torch.cuda.synchronize()
        scoring_seconds = time.perf_counter() - started
        risks = {a.client_id: a.current_risk for a in assessments}
        write_json(prefix.with_suffix(".json"), {"observation": observation, "assessments": [asdict(a) for a in assessments],
                   "prior_reputation": prior, "invalid": invalid, "updates": [asdict(u) for u in updates],
                   "submission_dispositions": [asdict(SubmissionDisposition(c, "invalid" if c in invalid else "valid" if c in selected else "absent", invalid.get(c))) for c in sorted(self.counts)],
                   "trusted": asdict(self.trusted_ref), "parent": asdict(parent_ref), "compatibility": asdict(self.comp)})
        root, root_report, root_seconds = self.root_update(parent, current)
        aggregate, requested_assessments = self.create_aggregator(assessments, root)
        request = AggregationRequest(current, parent_ref, self.comp, tuple(updates), requested_assessments,
                                     self.trusted_ref if self.root_service or requested_assessments else None,
                                     derive_seed(self.seed, "evolution", round_id=current))
        result = aggregate.aggregate(request)
        if result.failure_reason:
            return FederatedRound(result, reports, training_seconds, root_report, root_seconds)
        if self.method == "frozen":
            self.frozen = dict(aggregate.frozen)
        transitions = ()
        if self.calibration:
            pending, transitions = self.reputation.propose(current, risks, invalid, selected)
            self.reputation.load_state_dict(pending)
        events = {c: self.trainer.events[c] for c in selected}
        self.effective_count += sum(e["effective"] for e in events.values())
        write_json(self.directory / f"simulator-{current:04d}.json", {"events": events, "condition": self.plan["condition"],
                   "risk_diagnostics": truth_summary(events, risks, self.calibration["flag_threshold"], self.profiles, self.specialist) if self.calibration else None})
        write_json(self.directory / f"reputation-{current:04d}.json", {"transitions": [asdict(t) for t in transitions],
                   "state": self.reputation.state_dict(), "staleness": {c: current - e["last_valid"] for c, e in self.reputation.entries.items()}})
        self.round_artifacts[str(current)] = {"assessment": reference(self.config.workspace, prefix.with_suffix(".json")),
            "simulator": reference(self.config.workspace, self.directory / f"simulator-{current:04d}.json"),
            "reputation": reference(self.config.workspace, self.directory / f"reputation-{current:04d}.json")}
        result = replace(result, reputation_transitions=transitions)
        return FederatedRound(result, reports, training_seconds, root_report, root_seconds,
                              {"assessment": {"seconds": scoring_seconds, "valid_clients": len(updates), "invalid": invalid,
                                  "trusted_model_evaluations": 1 + len(observation["clients"]) + len(observation["invalid_predictions"]),
                                  "usage": "aggregation_required" if requested_assessments else "passive_diagnostics",
                                  "risk": risks, "prior_reputation": prior}})
