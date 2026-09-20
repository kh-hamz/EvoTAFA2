"""Deterministic seed namespaces; no global RNG is seeded at import time."""

import hashlib

from edgefl.config import canonical_json

SEED_VERSION = "edgefl.seed.v1"
STREAMS = ("partition", "initialization", "minibatch", "participation", "poisoning", "evolution")

def derive_seed(master: int, stream: str, *, client: str | None = None, round_id: int | None = None) -> int:
    """Derive a portable unsigned 32-bit seed from explicit identity coordinates."""
    if type(master) is not int or not 0 <= master <= 2**32 - 1:
        raise ValueError("master must be an unsigned 32-bit integer")
    if stream not in (*STREAMS, "global_split"):
        raise ValueError(f"Unknown random stream: {stream}")
    if client is not None and (not isinstance(client, str) or not client):
        raise ValueError("client must be a nonempty string when supplied")
    if round_id is not None and (type(round_id) is not int or round_id < 0):
        raise ValueError("round_id must be a nonnegative integer")
    identity = [SEED_VERSION, master, stream, client, round_id]
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def seed_manifest(master: int, global_split_seed: int) -> dict[str, object]:
    return {
        "version": SEED_VERSION,
        "master_seed": master,
        "global_split_base_seed": global_split_seed,
        "global_split": derive_seed(global_split_seed, "global_split"),
        "streams": {name: derive_seed(master, name) for name in STREAMS},
        "coordinate_rule": "Derive per-client/per-round seeds explicitly; never use Python hash().",
    }
