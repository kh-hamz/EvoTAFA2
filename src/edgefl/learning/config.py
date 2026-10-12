"""Separate strict configuration for Phase D; no implicit Phase A overrides."""

from pathlib import Path
from jsonschema import Draft202012Validator
from edgefl.config import Configuration, ConfigurationError, canonical_json, workspace_path
from edgefl.data.storage import read_json
from edgefl.client_data.config import load as load_c, phase_b


def load(path, root):
    root = root.resolve()
    if not path.resolve().is_relative_to(root):
        raise ConfigurationError("Phase D configuration must be inside workspace")
    value = read_json(path)
    schema = read_json(Path(__file__).parents[1] / "schemas/phase_d.schema.json")
    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        raise ConfigurationError("; ".join(e.message for e in errors))
    c = load_c(workspace_path(root, value["phase_c_config"]), root)
    b = phase_b(c)
    output = workspace_path(root, value["output"])
    protected = [workspace_path(root, b.values["archive"]),
                 workspace_path(root, b.values["output"]), workspace_path(root, c.values["output"])]
    protected += [root / name for name in ("src", "tests", "configs", "scripts", "docs", ".git", ".venv", ".agents", ".codex")]
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected):
        raise ConfigurationError("Phase D output overlaps protected inputs")
    return Configuration(root, path.resolve(), canonical_json(value))


def phase_c(config):
    return load_c(workspace_path(config.workspace, config.values["phase_c_config"]), config.workspace)
