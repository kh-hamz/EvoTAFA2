"""Strict experiment specification; outputs cannot overwrite research inputs."""
from dataclasses import dataclass
from pathlib import Path
from jsonschema import Draft202012Validator
from edgefl.config import Configuration, ConfigurationError, canonical_json, workspace_path
from edgefl.data.storage import read_json
from edgefl.optimization.config import load as load_f
from edgefl.contracts.phase_g import digest


@dataclass(frozen=True)
class ExperimentConfiguration:
    configuration: Configuration
    search: object

    @property
    def values(self): return self.configuration.values
    @property
    def workspace(self): return self.configuration.workspace
    @property
    def sha256(self): return digest({"g": self.values, "f": self.search.sha256})


def load(path, root):
    root, path = root.resolve(), path.resolve()
    if not path.is_relative_to(root):
        raise ConfigurationError("Experiment configuration outside workspace")
    value = read_json(path)
    schema = read_json(Path(__file__).parents[1] / "schemas/phase_g.schema.json")
    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        raise ConfigurationError("; ".join(e.message for e in errors))
    value.setdefault("evidence_kind", "research")
    value.setdefault("fixture_cases", [])
    if value["fixture_cases"] and value["evidence_kind"] != "fixture":
        raise ConfigurationError("Explicit small cases cannot enter a research matrix")
    search = load_f(workspace_path(root, value["phase_f_config"]), root)
    output = workspace_path(root, value["output"])
    protected = [root / p for p in ("src", "tests", "configs", "scripts", "docs", ".git", ".venv", ".agents", ".codex", "archive", "data")]
    protected += [workspace_path(root, c.values["output"]) for c in (search, search.trust, search.learning)]
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected):
        raise ConfigurationError("Experiment output overlaps protected inputs")
    return ExperimentConfiguration(Configuration(root, path, canonical_json(value)), search)
