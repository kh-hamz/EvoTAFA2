"""Strict Phase C configuration, kept independent from Phase A and Phase B."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from edgefl.config import Configuration, ConfigurationError, canonical_json, workspace_path
from edgefl.data.config import load as load_phase_b_config
from edgefl.data.storage import read_json
from edgefl.client_data.policies import support_policy, validate_feature_policy


def load(path: Path, root: Path) -> Configuration:
    root = root.resolve()
    source = path.resolve()
    if not source.is_relative_to(root):
        raise ConfigurationError("Phase C configuration must be inside the workspace")
    value = read_json(source)
    schema = json.loads((Path(__file__).parents[1] / "schemas/phase_c.schema.json").read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(e.path))
    if errors:
        raise ConfigurationError("; ".join(error.message for error in errors))
    phase_b_path = workspace_path(root, value["phase_b_config"])
    phase_b = load_phase_b_config(phase_b_path, root)
    output = workspace_path(root, value["output"])
    archive = workspace_path(root, phase_b.values["archive"])
    protected = [archive, *(root / name for name in
                  ("src", "tests", "configs", "scripts", "docs", ".git", ".venv",
                   ".agents", ".codex"))]
    if any(output.is_relative_to(item.resolve()) or item.resolve().is_relative_to(output)
           for item in protected):
        raise ConfigurationError("Phase C output overlaps protected inputs")
    if value["minimum_training_observations"] > value["minimum_client_observations"]:
        raise ConfigurationError("minimum_training_observations exceeds client minimum")
    preprocessing = value["preprocessing"]
    try:
        validate_feature_policy(preprocessing)
        support_policy(value)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    for scenario in value["scenarios"].values():
        if scenario["kind"] == "specialist" and scenario["specialists_per_attack"] > value["clients"]:
            raise ConfigurationError("specialists_per_attack exceeds client count")
    return Configuration(root, source, canonical_json(value))


def phase_b(config: Configuration) -> Configuration:
    return load_phase_b_config(workspace_path(config.workspace,
                                               config.values["phase_b_config"]),
                               config.workspace)
