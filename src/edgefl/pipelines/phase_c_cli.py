"""Thin Phase C command registration and routing."""

from edgefl.client_data.config import load
from edgefl.client_data.storage import artifact, load_completion, read_json
from edgefl.config import workspace_path
from edgefl.data.integrity import VerificationContext


COMMANDS = ("assign-clients", "local-split", "fit-preprocessing", "validate-phase-c")


def add_commands(sub) -> None:
    for name in COMMANDS:
        parser = sub.add_parser(name, help="Phase C: " + name)
        parser.add_argument("--config", required=True)
        parser.add_argument("--dataset", choices=("primary", "smoke"), required=True)
        if name == "assign-clients":
            parser.add_argument("--task", choices=("binary", "multiclass"), required=True)
            parser.add_argument("--phase-b-validation", required=True)
            parser.add_argument("--split", required=True)
            parser.add_argument("--fold", required=True)
            parser.add_argument("--scenario", required=True)
        elif name == "local-split":
            parser.add_argument("--assignments", required=True)
        elif name == "fit-preprocessing":
            parser.add_argument("--local-split", required=True)
            parser.add_argument("--provenance", required=True)
        else:
            for dependency in ("phase-b-validation", "split", "assignments", "local-split",
                               "preprocessing", "provenance"):
                parser.add_argument("--" + dependency, required=True)
            parser.add_argument("--fold", required=True)
            parser.add_argument("--scenario", required=True)


def execute(args):
    root = args.workspace.resolve()
    config = load(workspace_path(root, args.config), root)
    context = VerificationContext()

    def path(name):
        return workspace_path(root, getattr(args, name))

    if args.command == "assign-clients":
        from edgefl.pipelines.phase_c_assign import execute as run
        completion = run(config, path("phase_b_validation"), path("split"), args.dataset,
                         args.fold, args.scenario, args.task, context=context)
    elif args.command == "local-split":
        from edgefl.pipelines.phase_c_local_split import execute as run
        completion = run(config, path("assignments"), args.dataset, context=context)
    elif args.command == "fit-preprocessing":
        from edgefl.pipelines.phase_c_preprocessing import execute as run
        completion = run(config, path("local_split"), path("provenance"), args.dataset, context=context)
    else:
        from edgefl.pipelines.phase_c_validate import execute as run
        inputs = {name: path(name) for name in
                  ("phase_b_validation", "split", "assignments", "local_split",
                   "preprocessing", "provenance")}
        inputs.update(fold=args.fold, scenario=args.scenario)
        completion = run(config, inputs, args.dataset, context=context)
    metadata = load_completion(config, completion, args.command, args.dataset, context=context)
    result = {"completion": completion.relative_to(root).as_posix(),
              "eligible_for_training": metadata["eligible_for_training"]}
    if args.command == "validate-phase-c":
        report = read_json(artifact(config, metadata, "pretraining_report"))
        result["status"] = report["status"]
    return result
