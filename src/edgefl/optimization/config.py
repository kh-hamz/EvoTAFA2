"""Strict F settings layered on unchanged E/D configuration."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
from jsonschema import Draft202012Validator
from edgefl.config import Configuration, ConfigurationError, canonical_json, workspace_path
from edgefl.data.storage import read_json
from edgefl.trust.config import load as load_e
from edgefl.learning.config import phase_c
from edgefl.client_data.config import phase_b

@dataclass(frozen=True)
class SearchConfiguration:
    configuration: Configuration
    trust: object

    @property
    def workspace(self):
        return self.configuration.workspace

    @property
    def values(self):
        return self.configuration.values

    @property
    def learning(self):
        return self.trust.learning

    @property
    def sha256(self):
        return hashlib.sha256(canonical_json({"f": self.values, "e": self.trust.sha256}).encode()).hexdigest()

def load(path, root):
    root = root.resolve()
    if not path.resolve().is_relative_to(root):
        raise ConfigurationError("Phase F configuration outside workspace")
    value = read_json(path)
    schema = read_json(Path(__file__).parents[1] / "schemas/phase_f.schema.json")
    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        raise ConfigurationError("; ".join(e.message for e in errors))
    # JSON Schema integers include 8.0; libraries require Python integer indices.
    for name in ("population_size", "offspring_generations", "refill_multiplier"):
        value[name] = int(value[name])
    canonical = canonical_json(value)
    trust = load_e(workspace_path(root, value["phase_e_config"]), root)
    c = phase_c(trust.learning)
    b = phase_b(c)
    protected = [workspace_path(root, cfg.values[key]) for cfg, key in
                 ((trust, "output"), (trust.learning, "output"), (c, "output"), (b, "output"), (b, "archive"))]
    protected += [root / p for p in ("src", "tests", "configs", "scripts", "docs", ".git", ".venv", ".agents", ".codex")]
    output = workspace_path(root, value["output"])
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected):
        raise ConfigurationError("Phase F output overlaps protected inputs")
    return SearchConfiguration(Configuration(root, path.resolve(), canonical), trust)
