"""Shared clean training orchestration with immutable epoch/round boundaries."""

import time
from dataclasses import asdict
import psutil
import torch
from torch import nn
from edgefl.config import workspace_path
from edgefl.data.storage import read_json, write_json
from edgefl.learning import checkpoints
from edgefl.learning.models import build, compatibility, configure
from edgefl.learning.training import adam, train_epochs
from edgefl.learning.federated import BaselineSession
from edgefl.learning.metrics import evaluate, magnitude, weight_change
from edgefl.learning.storage import artifact, reference, implementation_hash


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


class MajorityModel(nn.Module):
    def __init__(self, classes, winner):
        super().__init__()
        self.classes, self.winner = classes, winner

    def forward(self, x):
        logits = torch.zeros((len(x), self.classes), device=x.device)
        logits[:, self.winner] = 1
        return logits


def evaluate_roles(model, views, spec, values):
    result, timings = {}, {}
    for role, view in views.items():
        start = time.perf_counter()
        result[role] = evaluate(model, view, len(spec["labels"]), spec["labels"]["Normal"],
                                values["batch_size"], values["device"], role)
        sync(values["device"])
        timings[role] = time.perf_counter() - start
    return result, timings


def selection_key(metrics, step):
    return (metrics["macro_f1"], -metrics["benign_fpr"], -step)


def run(config, data, prepared, initialization, stage, method, seed, resume=None,
        session_factory=None, artifact_policy=None):
    values, spec = config.values, data.spec
    environment = configure(values)
    write_json(stage.directory / "runtime.json", environment)
    if values["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if seed not in values["seeds"]:
        raise ValueError("Seed not registered in configuration")
    family = "logistic" if method == "logistic" else "mlp"
    comp = compatibility(spec, family)
    if initialization["identity"] != stage.identity or initialization["options"]["seed"] != seed:
        raise ValueError("Initialization identity/seed mismatch")
    initial_path = artifact(config, initialization, family + ".pt")
    initial = checkpoints.load(initial_path)["model"]
    model = build(spec["dimension"], len(spec["labels"]), family).to(values["device"])
    checkpoints.validate_state(initial, model.state_dict())
    model.load_state_dict(initial)
    central = method in ("majority", "logistic", "mlp")
    views = {role: data.view(role) for role in ("local_train", "local_validation", "selection_validation")}
    if not len(views["selection_validation"]):
        raise ValueError("Selection validation is empty")
    optimizer = adam(model, values)
    step, previous_weights, best, history = 0, None, None, []
    initial_ref = reference(config.workspace, initial_path)
    if resume:
        payload, resume_meta = resume
        expected = {"method": method, "seed": seed, "compatibility": asdict(comp),
                    "initialization": initial_ref, "environment": environment,
                    "prepared": prepared["artifacts"]["data.json"]}
        if any(payload.get(k) != v for k, v in expected.items()):
            raise ValueError("Resume identity/environment mismatch")
        checkpoints.validate_state(payload["model"], model.state_dict())
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"]) if central else None
        step, previous_weights, best, history = payload["step"], payload["weights"], payload["best"], payload["history"]
        torch.set_rng_state(payload["cpu_rng"])
        if values["device"] == "cuda":
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    if method == "majority":
        if resume:
            raise ValueError("Majority baseline does not resume")
        counts = torch.zeros(len(spec["labels"]), dtype=torch.int64)
        for x, y in torch.utils.data.DataLoader(views["local_train"], batch_size=values["batch_size"]):
            counts += torch.bincount(y, minlength=len(counts))
        if not int(counts.sum()):
            raise ValueError("Empty majority training set")
        model = MajorityModel(len(counts), int(counts.argmax())).to(values["device"])
        metrics, timings = evaluate_roles(model, views, spec, values)
        # The hard-label majority baseline has no fitted probability model.
        for measurement in metrics.values():
            measurement["loss"] = None
            measurement["loss_reason"] = "hard_label_majority_baseline"
        write_json(stage.directory / "majority.json", {"winner": int(counts.argmax()), "training_counts": counts.tolist(), "metrics": metrics, "timings": timings})
        return {"method": method, "seed": seed, "correctness": "PASS", "completed_steps": 0, "metrics": metrics}

    if not history:
        metrics, timings = evaluate_roles(model, views, spec, values)
        history.append({"step": 0, "metrics": metrics, "evaluation_seconds": timings})
    target = values["centralized_epochs"] if central else values["rounds"]
    data_ref = checkpoints.artifact(config.workspace, artifact(config, prepared, "data.json"))
    session = None if central else (session_factory or BaselineSession)(config, data, stage.directory, comp, data_ref, method, seed)
    if session and resume:
        session.load_state_dict(resume[0].get("phase_state", {}))
    phase = artifact_policy.phase if artifact_policy else "phase-d"
    boundary_path = stage.directory / f"boundary-{step:04d}.pt"
    boundary = {**resume[0], "model": checkpoints.cpu_state(model)} if resume else {"model": checkpoints.cpu_state(model)}
    if artifact_policy:
        boundary["phase_state"] = session.state_dict() if session else {}
    checkpoints.save(boundary_path, boundary)
    failed = None
    for current in range(step + 1, target + 1):
        started = time.perf_counter()
        parent = checkpoints.cpu_state(model)
        training_seconds, root_seconds, aggregation_seconds = 0.0, 0.0, 0.0
        client_reports, weights, root_report = {}, None, None
        extra = {}
        prior_session_state = session.state_dict() if session else {}
        try:
            if central:
                tick = time.perf_counter()
                training_report = train_epochs(model, views["local_train"], optimizer, values, seed,
                                                "centralized", current, 1)
                sync(values["device"])
                training_seconds = time.perf_counter() - tick
            else:
                parent_ref = checkpoints.artifact(config.workspace, boundary_path)
                executed = session.execute(current, parent_ref, parent)
                result, client_reports, training_seconds = executed.result, executed.client_reports, executed.training_seconds
                root_report, root_seconds, extra = executed.root_report, executed.root_seconds, executed.extra
                state = checkpoints.load(workspace_path(config.workspace, result.checkpoint.path), result.checkpoint.sha256)["model"]
                model.load_state_dict(state)
                agg_report = read_json(workspace_path(config.workspace, result.diagnostics.path))
                aggregation_seconds = agg_report["seconds"]
                weights = agg_report["weights"]
                training_report = None
            metrics, evaluation_times = evaluate_roles(model, views, spec, values)
            state = checkpoints.cpu_state(model)
            selection = metrics["selection_validation"]
            candidate_key = selection_key(selection, current)
            checkpoint_path = stage.directory / f"checkpoint-{current:04d}.pt"
            if best is None or candidate_key > tuple(best["key"]):
                best = {"step": current, "key": list(candidate_key), "path": checkpoint_path.relative_to(config.workspace).as_posix()}
            report = {"schema_version": phase + ".diagnostics.v1", "step": current, "identity": stage.identity,
                      "method": method, "seed": seed, "metrics": metrics, "client_reports": client_reports,
                      "training": training_report, "root": root_report, "weights": weights,
                      "weight_change": weight_change(previous_weights, weights),
                      "global_update": magnitude({k: state[k] - parent[k] for k in state}, parent),
                      "timing": {"client_training_cumulative": training_seconds, "root_training": root_seconds,
                                 "aggregation_including_aggregate_checkpoint": aggregation_seconds, "evaluation": evaluation_times},
                      "memory": {"rss_bytes": psutil.Process().memory_info().rss,
                                 "cuda_peak_bytes": torch.cuda.max_memory_allocated() if values["device"] == "cuda" else None}}
            report.update(extra)
            history.append(report)
            tick = time.perf_counter()
            payload = {"model": state, "optimizer": optimizer.state_dict() if central else {}, "step": current,
                       "weights": weights, "best": best, "history": history, "method": method, "seed": seed,
                       "compatibility": asdict(comp), "environment": environment, "initialization": initial_ref,
                       "prepared": prepared["artifacts"]["data.json"], "cpu_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if values["device"] == "cuda" else []}
            if artifact_policy:
                payload["phase_state"] = session.state_dict() if session else {}
            checkpoints.save(checkpoint_path, payload)
            report["timing"]["checkpoint"] = time.perf_counter() - tick
            report["timing"]["round_elapsed"] = time.perf_counter() - started
            if artifact_policy:
                scoring = extra.get("assessment", {}).get("seconds", 0.)
                passive = extra.get("assessment", {}).get("usage") == "passive_diagnostics"
                report["timing"]["scoring"] = scoring
                report["timing"]["round_excluding_passive_scoring"] = report["timing"]["round_elapsed"] - (scoring if passive else 0.)
            write_json(stage.directory / f"round-{current:04d}.json", report)
            write_json(stage.directory / f"resume-{current:04d}.json",
                       {"schema_version": phase + ".resume.v1", "configuration_sha256": artifact_policy.configuration_sha256 if artifact_policy else config.sha256,
                        "implementation_sha256": artifact_policy.implementation_sha256 if artifact_policy else implementation_hash(),
                        "checkpoint": reference(config.workspace, checkpoint_path), "identity": stage.identity,
                        "gate": stage.gate, "inputs": stage.inputs})
            boundary_path = checkpoint_path
            previous_weights, step = weights, current
        except (ValueError, RuntimeError) as exc:
            if session:
                session.load_state_dict(prior_session_state)
            failed = {"step": current, "reason": str(exc), "retained_parent": reference(config.workspace, boundary_path)}
            write_json(stage.directory / "failed_round.json", failed)
            break
    write_json(stage.directory / "history.json", history)
    if best:
        best["reference"] = reference(config.workspace, workspace_path(config.workspace, best["path"]))
    summary = {"method": method, "seed": seed, "correctness": "FAIL" if failed else "PASS",
               "completed_steps": step, "expected_steps": target, "best": best, "failure": failed,
               "final_checkpoint": reference(config.workspace, boundary_path), "optimizer": "adam",
               "paper_optimizer_adaptation": method == "fltrust"}
    write_json(stage.directory / "result.json", summary)
    return summary
