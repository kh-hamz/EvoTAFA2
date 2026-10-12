"""Operation-scoped source verification shared by data consumers."""

from dataclasses import dataclass, field

from edgefl.data.inventory import verify_sources
from edgefl.data.storage import checked, read_json


@dataclass
class VerificationContext:
    """Never persist this context across commands or training-ready loads."""

    registries: set[str] = field(default_factory=set)
    completions: dict = field(default_factory=dict)


def assert_registered_sources(config, completion, context=None):
    context = context if context is not None else VerificationContext()
    pending, seen, found = [completion], set(), False
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        metadata = read_json(path)
        if metadata["stage"] == "audit":
            reference = metadata["artifacts"]["registry"]
            found = True
            if reference["sha256"] not in context.registries:
                registry = read_json(checked(config.workspace, reference, "registry"))
                verify_sources(config.workspace, registry)
                context.registries.add(reference["sha256"])
        for reference in metadata["inputs"].values():
            if reference["role"] == "completion":
                pending.append(checked(config.workspace, reference, "completion"))
    if not found:
        raise ValueError("Phase B command has no registered-source lineage")
