"""Role-labelled predictive metrics and deterministic numerical diagnostics."""

import math
import torch
from torch.utils.data import DataLoader


def summarize(confusion, loss_sum=0.0):
    matrix = confusion.to(torch.float64)
    tp = matrix.diag()
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = torch.where(denominator > 0, 2 * tp / denominator.clamp_min(1), 0)
    count = int(matrix.sum())
    return {"observations": count, "loss": loss_sum / count if count else None,
            "macro_f1": float(f1.mean()) if count else None, "class_f1": f1.tolist(),
            "support": matrix.sum(1).to(torch.int64).tolist(), "confusion": confusion.tolist(),
            "zero_division": 0}


def evaluate(model, data, classes, benign_index, batch_size, device, role):
    confusion = torch.zeros((classes, classes), dtype=torch.int64)
    loss_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for x, y in DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0):
            logits = model(x.to(device))
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite evaluation logits")
            loss_sum += float(torch.nn.functional.cross_entropy(logits, y.to(device), reduction="sum"))
            predictions = logits.argmax(1).cpu()
            confusion += torch.bincount(y * classes + predictions, minlength=classes * classes).reshape(classes, classes)
    report = summarize(confusion, loss_sum)
    benign = int(confusion[benign_index].sum())
    report.update(role=role, benign_fpr=float((benign - confusion[benign_index, benign_index]) / benign) if benign else None,
                  benign_fpr_reason=None if benign else "no_benign_support")
    return report


def norm(state):
    return math.sqrt(sum(float(v.detach().double().square().sum()) for v in state.values()))


def magnitude(delta, parent):
    absolute, base = norm(delta), norm(parent)
    return {"absolute": absolute, "relative": absolute / base if base else None,
            "relative_reason": None if base else "zero_parent_norm"}


def weight_change(previous, current):
    if previous is None or current is None:
        return {"distance": None, "reason": "first_round_or_inapplicable", "participation_changed": None}
    return {"distance": sum(abs(previous.get(k, 0) - current.get(k, 0)) for k in previous.keys() | current.keys()) / 2,
            "reason": None, "participation_changed": set(previous) != set(current)}
