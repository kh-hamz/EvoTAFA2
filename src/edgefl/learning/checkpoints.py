"""Hash-verified state dictionaries, atomically written and never overwritten."""

import os
import torch
from edgefl.contracts.records import ArtifactRef
from edgefl.data.storage import sha256


def save(path, payload):
    if path.exists():
        raise ValueError("Refusing to overwrite completed checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return sha256(path)


def load(path, digest=None):
    if digest is not None and sha256(path) != digest:
        raise ValueError("Checkpoint hash mismatch")
    return torch.load(path, map_location="cpu", weights_only=True)


def validate_state(state, expected):
    if set(state) != set(expected):
        raise ValueError("Model parameter keys mismatch")
    for key, value in state.items():
        template = expected[key]
        if not isinstance(value, torch.Tensor) or value.shape != template.shape or value.dtype != template.dtype:
            raise ValueError("Model parameter shape/dtype mismatch")
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite model/update tensor")


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def artifact(root, path):
    return ArtifactRef(path.relative_to(root).as_posix(), sha256(path))
