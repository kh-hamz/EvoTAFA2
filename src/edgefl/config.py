"""Strict, explicit configuration loading. No data access or implicit overrides."""

import hashlib
import json
import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PureWindowsPath
from typing import Any

from jsonschema import Draft202012Validator


class ConfigurationError(ValueError):
    """The supplied configuration cannot describe a Phase A foundation run."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ConfigurationError(f"Non-finite JSON value: {value}")


def workspace_path(root: Path, value: str) -> Path:
    """Resolve a portable workspace-relative path and reject traversal/symlink escape."""
    windows = PureWindowsPath(value)
    if not value or "\\" in value or windows.drive or windows.root:
        raise ConfigurationError(f"Use a relative path with forward slashes: {value!r}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or ":" in value:
        raise ConfigurationError(f"Unsafe workspace-relative path: {value!r}")
    resolved_root = root.resolve()
    target = (resolved_root / relative).resolve()
    if not target.is_relative_to(resolved_root) or target == resolved_root:
        raise ConfigurationError(f"Path escapes or equals the workspace: {value!r}")
    return target


@dataclass(frozen=True)
class Configuration:
    """Canonical JSON is immutable; callers receive a fresh decoded copy."""

    workspace: Path
    source: Path
    canonical: str

    @property
    def values(self) -> dict[str, Any]:
        return json.loads(self.canonical)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()


def load_config(path: Path, workspace: Path) -> Configuration:
    root = workspace.resolve()
    source = path.resolve()
    if not source.is_relative_to(root):
        raise ConfigurationError("Configuration must be inside the workspace")
    try:
        values = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read configuration: {exc}") from exc
    schema = json.loads(files("edgefl").joinpath("schemas/config.schema.json").read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(values), key=lambda e: str(e.path))
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ConfigurationError(details)
    try:
        canonical = canonical_json(values)
    except ValueError as exc:
        raise ConfigurationError("Configuration contains a non-finite number") from exc
    split = values["research_defaults"]["global_split"]
    if not math.isclose(sum(split.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ConfigurationError("research_defaults.global_split must sum to 1")
    resolved_paths = {name: workspace_path(root, value) for name, value in values["paths"].items()}
    protected = [resolved_paths["archive"], *(root / name for name in (
        "src", "docs", "configs", "scripts", "tests", ".git", ".venv",
    ))]
    output_paths = [path for name, path in resolved_paths.items() if name != "archive"]
    for index, target in enumerate(output_paths):
        for other in [*protected, *output_paths[:index]]:
            other = other.resolve()
            if target.is_relative_to(other) or other.is_relative_to(target):
                raise ConfigurationError("Output paths overlap protected inputs or another output")
    defaults = values["research_defaults"]
    if defaults["attack_start_round"] > defaults["rounds"]:
        raise ConfigurationError("attack_start_round cannot exceed rounds")
    for role in ("primary", "smoke"):
        target = workspace_path(root, values["dataset"][role])
        if not target.is_relative_to(workspace_path(root, values["paths"]["archive"])):
            raise ConfigurationError(f"dataset.{role} must be under the immutable archive")
    protocol = workspace_path(root, values["protocol_document"])
    if not protocol.is_file():
        raise ConfigurationError(f"Missing research protocol: {protocol}")
    return Configuration(root, source, canonical)
