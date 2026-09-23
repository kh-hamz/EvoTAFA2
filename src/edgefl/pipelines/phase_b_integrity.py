"""Public command boundary: cached outputs still require the registered source bytes."""

from edgefl.data.inventory import verify_sources
from edgefl.data.storage import checked, read_json


def assert_registered_sources(config, completion):
    """Walk explicit lineage only; do not discover or execute predecessor stages."""
    pending = [completion]
    seen = set()
    registries = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        metadata = read_json(path)
        if metadata["stage"] == "audit":
            reference = metadata["artifacts"]["registry"]
            if reference["sha256"] not in registries:
                registry = read_json(checked(config.workspace, reference, "registry"))
                verify_sources(config.workspace, registry)
                registries.add(reference["sha256"])
        for reference in metadata["inputs"].values():
            if reference["role"] == "completion":
                pending.append(checked(config.workspace, reference, "completion"))
    if not registries:
        raise ValueError("Phase B command has no registered-source lineage")
