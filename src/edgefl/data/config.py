"""Phase B validation independent of Phase A's immutable foundation schema."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from edgefl.config import Configuration, ConfigurationError, canonical_json, workspace_path
from edgefl.data.storage import read_json


def load(path: Path, root: Path) -> Configuration:
    root = root.resolve()
    source = path.resolve()
    if not source.is_relative_to(root):
        raise ConfigurationError("Phase B configuration must be inside the workspace")
    value = read_json(source)
    schema = json.loads((Path(__file__).parents[1] / "schemas/phase_b.schema.json").read_text())
    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        raise ConfigurationError("; ".join(error.message for error in errors))
    archive = workspace_path(root, value["archive"])
    output = workspace_path(root, value["output"])
    source_root = workspace_path(root, value["source_root"])
    if not source_root.is_relative_to(archive):
        raise ConfigurationError("source_root must be inside archive")
    protected = [archive, *(root / name for name in
                  ("src", "tests", "configs", "scripts", "docs", ".git", ".venv", ".agents", ".codex"))]
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output)
           for p in protected):
        raise ConfigurationError("Phase B output overlaps protected inputs")
    for selected in value["selected"].values():
        if not workspace_path(root, selected).is_relative_to(archive):
            raise ConfigurationError("Selected dataset must be in archive")
    for name, counterpart in value["pairings"].items():
        if any(Path(s).name != s or "/" in s or "\\" in s or ":" in s for s in (name, counterpart)):
            raise ConfigurationError("Pairings require unique basenames")
    aliases = value["label_aliases"]
    if any(k == v or v in aliases for k, v in aliases.items()):
        raise ConfigurationError("Label aliases must be nonrecursive")
    return Configuration(root, source, canonical_json(value))
