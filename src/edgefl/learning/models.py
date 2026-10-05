"""Deterministic model factories and explicit compute configuration."""

import os
import platform
import torch
from torch import nn
from edgefl.reproducibility import derive_seed
from edgefl.contracts.records import Compatibility


def configure(values):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(values["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = values["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; select CPU explicitly")
    return {"python": platform.python_version(), "torch": str(torch.__version__), "device": device,
            "cuda": torch.version.cuda, "threads": values["threads"],
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "deterministic": True, "precision": "float32"}


def build(dimension, classes, family="mlp"):
    if family == "mlp":
        return nn.Sequential(nn.Linear(dimension, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, classes))
    if family == "logistic":
        return nn.Sequential(nn.Linear(dimension, classes))
    raise ValueError("Unknown model family")


def initialized(dimension, classes, seed, family="mlp"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(seed, "initialization"))
        return build(dimension, classes, family)


def compatibility(spec, family="mlp"):
    return Compatibility(spec["transformer_sha256"], spec["features"]["feature_order_sha256"],
                         spec["label_map_sha256"], f"{family}-float32-v1", spec["dimension"], len(spec["labels"]))
