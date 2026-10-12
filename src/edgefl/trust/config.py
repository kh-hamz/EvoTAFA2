"""Strict Phase E configuration with explicit, unchanged Phase D training settings."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
from jsonschema import Draft202012Validator
from edgefl.config import Configuration, ConfigurationError, canonical_json, workspace_path
from edgefl.data.storage import read_json
from edgefl.learning.config import load as load_d, phase_c
from edgefl.client_data.config import phase_b


@dataclass(frozen=True)
class TrustConfiguration:
    configuration: Configuration
    learning: Configuration

    @property
    def workspace(self):
        return self.configuration.workspace

    @property
    def values(self):
        return self.configuration.values

    @property
    def sha256(self):
        return hashlib.sha256(canonical_json({"phase_e": self.values, "phase_d": self.learning.sha256}).encode()).hexdigest()


def load(path, root):
    root = root.resolve()
    if not path.resolve().is_relative_to(root):
        raise ConfigurationError("Phase E configuration outside workspace")
    value = read_json(path)
    schema = read_json(Path(__file__).parents[1] / "schemas/phase_e.schema.json")
    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        raise ConfigurationError("; ".join(e.message for e in errors))
    canonical = canonical_json(value)
    learning = load_d(workspace_path(root, value["phase_d_config"]), root)
    c = phase_c(learning)
    b = phase_b(c)
    protected = [workspace_path(root, cfg.values[key]) for cfg, key in
                 ((learning, "output"), (c, "output"), (b, "output"), (b, "archive"))]
    protected += [root / name for name in ("src", "tests", "configs", "scripts", "docs", ".git", ".venv", ".agents", ".codex")]
    output = workspace_path(root, value["output"])
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected):
        raise ConfigurationError("Phase E output overlaps protected inputs")
    if set(learning.values["seeds"]) & {101, 102, 103}:
        raise ConfigurationError("Comparison seeds must exclude calibration seeds")
    return TrustConfiguration(Configuration(root, path.resolve(), canonical), learning)


def training_configuration(config, seed):
    """Register a pilot seed explicitly in the execution snapshot, without changing D artifacts."""
    values = config.learning.values
    if seed not in values["seeds"] and seed not in (*config.values["fit_seeds"], config.values["audit_seed"]):
        raise ConfigurationError("Unregistered Phase E seed")
    values["seeds"] = sorted(set(values["seeds"]) | {seed})
    return Configuration(config.workspace, config.learning.source, canonical_json(values))
