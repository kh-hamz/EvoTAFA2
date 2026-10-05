"""Independently callable federated pipeline over the shared execution service."""
from edgefl.pipelines.phase_d_centralized import execute as _execute


def execute(config, prepared, initialization, method, seed, reviews=(), resume=None):
    if method not in ("fedavg", "fedprox", "median", "trimmed_mean", "fltrust"):
        raise ValueError("Federated command requires a federated method")
    return _execute(config, prepared, initialization, method, seed, reviews, resume)
