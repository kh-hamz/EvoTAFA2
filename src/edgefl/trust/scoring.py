"""Trusted-only predictive observations and the ClientScorer adapter."""

import math
from collections import Counter
import torch
from torch.utils.data import DataLoader
from edgefl.config import workspace_path
from edgefl.contracts.records import ClassExpertise, ClassSupport, ClientAssessment
from edgefl.learning import checkpoints
from edgefl.learning.models import build
from edgefl.learning.metrics import summarize
from edgefl.trust.risk import raw_signals, calibrated_risk


class InvalidPrediction(ValueError):
    """A submitted finite state produces nonfinite trusted predictions."""


def evaluate_trusted(model, data, originals, labels, task, values):
    if len(data) == 0 or len(originals) != len(data):
        raise ValueError("Empty or misaligned trusted panel")
    classes, benign = len(labels), labels["Normal"]
    confusion = torch.zeros((classes, classes), dtype=torch.int64)
    losses = torch.zeros(classes, dtype=torch.float64)
    detected, original_support = Counter(), Counter(originals)
    cursor = 0
    model.eval()
    with torch.inference_mode():
        for x, y in DataLoader(data, batch_size=values["batch_size"], shuffle=False, num_workers=0):
            logits = model(x.to(values["device"]))
            ce = torch.nn.functional.cross_entropy(logits, y.to(values["device"]), reduction="none")
            if not torch.isfinite(logits).all() or not torch.isfinite(ce).all():
                raise InvalidPrediction("Nonfinite submitted-model predictions/loss")
            pred = logits.argmax(1).cpu()
            confusion += torch.bincount(y * classes + pred, minlength=classes * classes).reshape(classes, classes)
            losses.scatter_add_(0, y, ce.cpu().double())
            for original, prediction in zip(originals[cursor:cursor + len(y)], pred.tolist()):
                if prediction != benign:
                    detected[original] += 1
            cursor += len(y)
    result = summarize(confusion, float(losses.sum()))
    support = confusion.sum(1)
    present = support > 0
    recalls = torch.where(present, confusion.diag().double() / support.clamp_min(1), 0.)
    precision = confusion.diag().double() / confusion.sum(0).clamp_min(1)
    n_benign = int(support[benign])
    reverse = {index: name for name, index in labels.items()}
    result.update(role="trusted", balanced_accuracy=float(recalls[present].mean()),
                  balanced_accuracy_classes=[reverse[i] for i in range(classes) if present[i]],
                  class_precision=precision.tolist(), class_recall=recalls.tolist(),
                  class_balanced_loss=float((losses[present] / support[present]).mean()),
                  benign_fpr=float((support[benign] - confusion[benign, benign]) / n_benign) if n_benign else None,
                  benign_fpr_reason=None if n_benign else "no_benign_support", original_support=dict(original_support))
    if task == "binary":
        result["expertise_kind"] = "attack_type_detection_recall"
        result["raw_expertise"] = {name: detected[name] / n for name, n in original_support.items() if name != "Normal"}
        result["expertise_support"] = {name: n for name, n in original_support.items() if name != "Normal"}
    else:
        result["expertise_kind"] = "multiclass_f1"
        result["raw_expertise"] = {name: result["class_f1"][i] if support[i] else None for name, i in labels.items()}
        result["expertise_support"] = {name: int(support[i]) for name, i in labels.items()}
    return result


def expertise(client, parent, vocabulary, scale=50.0):
    entries = []
    for label in sorted(vocabulary):
        n = client["expertise_support"].get(label, 0)
        score = (n * client["raw_expertise"][label] + scale * parent["raw_expertise"][label]) / (n + scale) if n else None
        entries.append(ClassExpertise(label, score, n))
    return tuple(entries)


class TrustedClientScorer:
    def __init__(self, root, trusted, originals, trusted_ref, compatibility, labels, task, values,
                 original_vocabulary, support_scale=50.0, minimum_valid=3):
        self.root, self.trusted, self.originals = root, trusted, tuple(originals)
        self.trusted_ref, self.compatibility = trusted_ref, compatibility
        self.labels, self.task, self.values = labels, task, values
        self.vocabulary = tuple(label for label in original_vocabulary if task != "binary" or label != "Normal")
        self.scale, self.minimum_valid = support_scale, minimum_valid
        self.calibration, self.prior = None, None

    def observe(self, request):
        if request.trusted != self.trusted_ref or request.compatibility != self.compatibility:
            raise ValueError("Trusted scoring resource/compatibility mismatch")
        ids = [u.client_id for u in request.updates]
        if not ids or len(set(ids)) != len(ids) or len({u.round_id for u in request.updates}) != 1:
            raise ValueError("Duplicate, empty or mixed-round scoring request")
        parent = checkpoints.load(workspace_path(self.root, request.parent_model.path), request.parent_model.sha256)["model"]
        model = build(self.compatibility.input_features, self.compatibility.output_classes).to(self.values["device"])
        checkpoints.validate_state(parent, model.state_dict())
        model.load_state_dict(parent)
        parent_report = evaluate_trusted(model, self.trusted, self.originals, self.labels, self.task, self.values)
        reports, states, invalid = {}, {}, {}
        for update in request.updates:
            delta = checkpoints.load(workspace_path(self.root, update.delta.path), update.delta.sha256)["delta"]
            checkpoints.validate_state(delta, parent)
            state = {k: parent[k] + delta[k] for k in parent}
            checkpoints.validate_state(state, parent)
            model.load_state_dict(state)
            try:
                report = evaluate_trusted(model, self.trusted, self.originals, self.labels, self.task, self.values)
            except InvalidPrediction as exc:
                invalid[update.client_id] = str(exc)
                continue
            report["loss_difference"] = report["loss"] - parent_report["loss"]
            report["expertise"] = [{"label": e.label, "score": e.score, "support": e.support} for e in expertise(report, parent_report, self.vocabulary, self.scale)]
            reports[update.client_id], states[update.client_id] = report, delta
        raw = raw_signals(states, {c: r["class_balanced_loss"] for c, r in reports.items()}, parent_report["class_balanced_loss"], self.minimum_valid)
        for client in reports:
            reports[client]["raw_risk"] = raw[client]
        return {"schema_version": "phase-e.observations.v1", "round_id": request.updates[0].round_id,
                "parent": parent_report, "clients": reports, "invalid_predictions": invalid}

    def assessments(self, observation, calibration, prior):
        result = []
        for client, report in observation["clients"].items():
            components, risk = calibrated_risk(report["raw_risk"], calibration)
            if client not in prior or not math.isfinite(prior[client]) or not 0 <= prior[client] <= 1:
                raise ValueError("Invalid prior reputation")
            result.append(ClientAssessment(client, observation["round_id"], report["macro_f1"],
                tuple(ClassExpertise(**entry) for entry in report["expertise"]), tuple(components.items()), risk, prior[client],
                tuple(ClassSupport(name, report["support"][index]) for name, index in sorted(self.labels.items()))))
        return tuple(result)

    def assess(self, request):
        if self.calibration is None or self.prior is None:
            raise ValueError("Scoring assessments require frozen calibration and prior reputation")
        return self.assessments(self.observe(request), self.calibration, self.prior)
