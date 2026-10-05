"""Shared clean training orchestration with immutable epoch/round boundaries."""

import math
import time
from dataclasses import asdict
import psutil
import torch
from torch import nn
from edgefl.config import workspace_path
from edgefl.contracts.records import ClientManifest, PartitionRef, Partition
from edgefl.contracts.interfaces import TrainingRequest, AggregationRequest
from edgefl.data.storage import read_json, write_json
from edgefl.learning import checkpoints
from edgefl.learning.models import build, compatibility, configure
from edgefl.learning.training import LocalTrainer, ServerReference, adam, train_epochs
from edgefl.learning.aggregation import BaselineAggregator
from edgefl.learning.metrics import evaluate, magnitude, weight_change
from edgefl.learning.storage import artifact, reference, implementation_hash
from edgefl.reproducibility import derive_seed


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


def run(config, data, prepared, initialization, stage, method, seed, resume=None):
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
    clients = {c: data.view("local_train", c) for c in data.clients} if not central else {}
    trainer = LocalTrainer(config.workspace, stage.directory, clients, comp, values, seed,
                           values["fedprox_mu"] if method == "fedprox" else 0)
    trusted = data.view("trusted") if method == "fltrust" else None
    root_service = ServerReference(trusted, values, seed, comp) if trusted is not None else None
    data_ref = checkpoints.artifact(config.workspace, artifact(config, prepared, "data.json"))
    boundary_path = stage.directory / f"boundary-{step:04d}.pt"
    checkpoints.save(boundary_path, {"model": checkpoints.cpu_state(model)})
    failed = None
    for current in range(step + 1, target + 1):
        started = time.perf_counter()
        parent = checkpoints.cpu_state(model)
        training_seconds, root_seconds, aggregation_seconds = 0.0, 0.0, 0.0
        client_reports, weights, root_report = {}, None, None
        try:
            if central:
                tick = time.perf_counter()
                training_report = train_epochs(model, views["local_train"], optimizer, values, seed,
                                                "centralized", current, 1)
                sync(values["device"])
                training_seconds = time.perf_counter() - tick
            else:
                generator = torch.Generator().manual_seed(derive_seed(seed, "participation", round_id=current))
                n = max(1, math.ceil(len(data.clients) * values["participation"]))
                selected = sorted(data.clients[i] for i in torch.randperm(len(data.clients), generator=generator).tolist()[:n])
                parent_ref = checkpoints.artifact(config.workspace, boundary_path)
                updates = []
                for client in selected:
                    manifest = ClientManifest(client, data_ref, PartitionRef(data_ref, Partition.LOCAL_TRAIN),
                                              PartitionRef(data_ref, Partition.LOCAL_VALIDATION), (), seed)
                    request = TrainingRequest(manifest, current, parent_ref, comp,
                                               derive_seed(seed, "minibatch", client=client, round_id=current))
                    update = trainer.train(request)
                    updates.append(update)
                    delta = checkpoints.load(workspace_path(config.workspace, update.delta.path), update.delta.sha256)["delta"]
                    client_reports[client] = {**trainer.diagnostics[client], "update": magnitude(delta, parent), "transmitted_bytes": update.transmitted_bytes}
                training_seconds = sum(u.training_seconds for u in updates)
                root_delta = None
                if root_service:
                    tick = time.perf_counter()
                    root_delta, root_report = root_service.update(parent, current)
                    sync(values["device"])
                    root_seconds = time.perf_counter() - tick
                aggregate = BaselineAggregator(config.workspace, stage.directory, method,
                                               {c: len(v) for c, v in clients.items()}, values["trim_fraction"], root_delta)
                request = AggregationRequest(current, parent_ref, comp, tuple(updates), (),
                                             PartitionRef(data_ref, Partition.TRUSTED) if root_service else None,
                                             derive_seed(seed, "evolution", round_id=current))
                result = aggregate.aggregate(request)
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
            report = {"schema_version": "phase-d.diagnostics.v1", "step": current, "identity": stage.identity,
                      "method": method, "seed": seed, "metrics": metrics, "client_reports": client_reports,
                      "training": training_report, "root": root_report, "weights": weights,
                      "weight_change": weight_change(previous_weights, weights),
                      "global_update": magnitude({k: state[k] - parent[k] for k in state}, parent),
                      "timing": {"client_training_cumulative": training_seconds, "root_training": root_seconds,
                                 "aggregation_including_aggregate_checkpoint": aggregation_seconds, "evaluation": evaluation_times},
                      "memory": {"rss_bytes": psutil.Process().memory_info().rss,
                                 "cuda_peak_bytes": torch.cuda.max_memory_allocated() if values["device"] == "cuda" else None}}
            history.append(report)
            tick = time.perf_counter()
            payload = {"model": state, "optimizer": optimizer.state_dict() if central else {}, "step": current,
                       "weights": weights, "best": best, "history": history, "method": method, "seed": seed,
                       "compatibility": asdict(comp), "environment": environment, "initialization": initial_ref,
                       "prepared": prepared["artifacts"]["data.json"], "cpu_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if values["device"] == "cuda" else []}
            checkpoints.save(checkpoint_path, payload)
            report["timing"]["checkpoint"] = time.perf_counter() - tick
            report["timing"]["round_elapsed"] = time.perf_counter() - started
            write_json(stage.directory / f"round-{current:04d}.json", report)
            write_json(stage.directory / f"resume-{current:04d}.json",
                       {"schema_version": "phase-d.resume.v1", "configuration_sha256": config.sha256,
                        "implementation_sha256": implementation_hash(),
                        "checkpoint": reference(config.workspace, checkpoint_path), "identity": stage.identity,
                        "gate": stage.gate, "inputs": stage.inputs})
            boundary_path = checkpoint_path
            previous_weights, step = weights, current
        except (ValueError, RuntimeError) as exc:
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
