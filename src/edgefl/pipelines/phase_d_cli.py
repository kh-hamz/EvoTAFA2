"""Thin Phase D command routing; importing this module needs no training libraries."""
from edgefl.config import workspace_path
from edgefl.learning.config import load
from edgefl.learning.storage import read_json

COMMANDS = ("prepare-learning-data", "initialize-model", "train-centralized", "run-federated", "review-learning", "validate-phase-d")


def add_commands(sub):
    for name in COMMANDS:
        parser = sub.add_parser(name, help="Phase D: " + name)
        parser.add_argument("--config", required=True)
        if name == "prepare-learning-data":
            for key in ("phase-c-validation", "dataset", "task", "fold", "scenario"):
                parser.add_argument("--" + key, required=True)
        elif name in ("initialize-model", "train-centralized", "run-federated"):
            parser.add_argument("--prepared", required=True)
            parser.add_argument("--seed", required=True, type=int)
            if name != "initialize-model":
                parser.add_argument("--initialization", required=True)
                choices = ("majority", "logistic", "mlp") if name == "train-centralized" else ("fedavg", "fedprox", "median", "trimmed_mean", "fltrust")
                parser.add_argument("--method", choices=choices, required=True)
                parser.add_argument("--resume")
                if name == "run-federated":
                    parser.add_argument("--review", action="append", required=True)
        elif name == "review-learning":
            parser.add_argument("--run", required=True)
            parser.add_argument("--disposition", choices=("acceptable", "unresolved", "explained_negative"), required=True)
            parser.add_argument("--rationale-file", required=True)
            parser.add_argument("--limitations-file")
        else:
            parser.add_argument("--review", action="append", required=True)
            parser.add_argument("--test-report", required=True)


def execute(args):
    root = args.workspace.resolve()
    config = load(workspace_path(root, args.config), root)
    def path(value):
        return workspace_path(root, value)
    if args.command == "prepare-learning-data":
        from edgefl.pipelines.phase_d_prepare import execute as run
        result = run(config, path(args.phase_c_validation), args.dataset, args.task, args.fold, args.scenario)
    elif args.command == "initialize-model":
        from edgefl.pipelines.phase_d_initialize import execute as run
        result = run(config, path(args.prepared), args.seed)
    elif args.command in ("train-centralized", "run-federated"):
        if args.command == "train-centralized":
            from edgefl.pipelines.phase_d_centralized import execute as run
        else:
            from edgefl.pipelines.phase_d_federated import execute as run
        result = run(config, path(args.prepared), path(args.initialization), args.method, args.seed,
                     [path(p) for p in getattr(args, "review", ())], path(args.resume) if args.resume else None)
    elif args.command == "review-learning":
        from edgefl.pipelines.phase_d_review import execute as run
        result = run(config, path(args.run), args.disposition, path(args.rationale_file).read_text(encoding="utf-8"),
                     path(args.limitations_file).read_text(encoding="utf-8") if args.limitations_file else "")
    else:
        from edgefl.pipelines.phase_d_validate import execute as run
        result = run(config, [path(p) for p in args.review], path(args.test_report))
    metadata = read_json(result)
    summary = metadata["summary"]
    return {"completion": result.relative_to(root).as_posix(),
            "status": summary.get("status", summary.get("correctness", "complete")), "summary": summary}
