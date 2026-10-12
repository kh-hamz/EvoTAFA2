"""Publish shared CPU initial states with complete compatibility identities."""
from dataclasses import asdict
from edgefl.learning.storage import stage, load_stage, checked, artifact, read_json, write_json


def execute(config, prepared_path, seed):
    prepared = load_stage(config, prepared_path, "prepare-learning-data")
    if seed not in config.values["seeds"]:
        raise ValueError("Unregistered seed")
    from edgefl.learning.models import initialized, compatibility
    from edgefl.learning.checkpoints import save
    spec = read_json(artifact(config, prepared, "data.json"))
    with stage(config, "initialize-model", prepared["identity"], checked(config.workspace, prepared["gate"]),
               {"prepared": prepared_path}, {"seed": seed}) as run:
        for family in ("mlp", "logistic"):
            model = initialized(spec["dimension"], len(spec["labels"]), seed, family)
            save(run.directory / (family + ".pt"), {"model": model.state_dict()})
            write_json(run.directory / (family + "-compatibility.json"), asdict(compatibility(spec, family)))
        return run.finish({"seed": seed, "families": ["mlp", "logistic"]})
