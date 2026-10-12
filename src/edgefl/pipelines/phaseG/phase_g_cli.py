"""Lazy Phase G dispatch preserves dependency-light CLI discovery."""
from edgefl.config import workspace_path
from edgefl.data.storage import read_json

FIELDS = {
    "compile-experiments": (), "prepare-experiment": ("matrix", "case", "binding"),
    "preflight-experiments": ("matrix", "bindings"), "run-experiment": ("prepared",),
    "run-experiments": ("matrix", "bindings"), "fit-experiment-clipping": ("prepared",),
    "calibrate-experiment": ("prepared", "audit"), "profile-experiments": ("matrix", "bindings", "measurements"),
    "review-experiments": ("manifest",), "summarize-experiments": ("matrix", "runs"),
    "validate-phase-g": ("matrix", "summary", "reviews", "test-report"), "export-phase-h-handoff": ("acceptance",)}
COMMANDS = tuple(FIELDS)


def add_commands(sub):
    for name, fields in FIELDS.items():
        parser = sub.add_parser(name, help="Phase G: " + name)
        parser.add_argument("--config", required=True)
        for field in fields:
            parser.add_argument("--" + field, required=True)
        if name == "run-experiment":
            parser.add_argument("--resume")
            parser.add_argument("--pilot-seed", type=int, choices=(101, 102, 103))
        if name == "run-experiments":
            parser.add_argument("--cost-review")
            parser.add_argument("--suite", choices=("main", "ablations", "sensitivity", "capture", "fixture"))
            parser.add_argument("--recover", action="store_true")
            parser.add_argument("--resumes")
        if name == "calibrate-experiment":
            parser.add_argument("--fit", action="append", required=True)


def execute(args):
    from edgefl.experiments.config import load
    config = load(workspace_path(args.workspace.resolve(), args.config), args.workspace.resolve())
    def path(name):
        value = getattr(args, name, None)
        return workspace_path(config.workspace, value) if value else None
    command = args.command
    if command == "compile-experiments":
        from edgefl.pipelines.phaseG.phase_g_compile import execute as run
        result = run(config)
    elif command == "prepare-experiment":
        from edgefl.pipelines.phaseG.phase_g_prepare import execute as run
        result = run(config, path("matrix"), args.case, path("binding"))
    elif command == "preflight-experiments":
        from edgefl.pipelines.phaseG.phase_g_preflight import execute as run
        result = run(config, path("matrix"), path("bindings"))
    elif command == "run-experiment":
        from edgefl.pipelines.phaseG.phase_g_run import execute as run
        result = run(config, path("prepared"), path("resume"), args.pilot_seed)
    elif command == "run-experiments":
        from edgefl.pipelines.phaseG.phase_g_run import queue
        return queue(config, path("matrix"), path("bindings"), path("cost_review"), args.suite, args.recover, path("resumes"))
    elif command == "fit-experiment-clipping":
        from edgefl.pipelines.phaseG.phase_g_prepare import fit_clipping
        result = fit_clipping(config, path("prepared"))
    elif command == "calibrate-experiment":
        from edgefl.pipelines.phaseG.phase_g_prepare import calibrate
        result = calibrate(config, path("prepared"), [workspace_path(config.workspace, p) for p in args.fit], path("audit"))
    elif command == "profile-experiments":
        from edgefl.pipelines.phaseG.phase_g_profile import execute as run
        result = run(config, path("matrix"), path("bindings"), path("measurements"))
    elif command == "review-experiments":
        from edgefl.pipelines.phaseG.phase_g_review import execute as run
        result = run(config, path("manifest"))
    elif command == "summarize-experiments":
        from edgefl.pipelines.phaseG.phase_g_summarize import execute as run
        result = run(config, path("matrix"), path("runs"))
    elif command == "validate-phase-g":
        from edgefl.pipelines.phaseG.phase_g_validate import execute as run
        result = run(config, path("matrix"), path("summary"), path("reviews"), path("test_report"))
    else:
        from edgefl.pipelines.phaseG.phase_g_handoff import execute as run
        result = run(config, path("acceptance"))
    summary = read_json(result)["summary"]
    status = summary.get("status", summary.get("correctness", "complete"))
    if summary.get("attack_coverage", "PASS") != "PASS":
        status = "UNSUCCESSFUL"
    return {"completion": result.relative_to(config.workspace).as_posix(), "status": status, "summary": summary}
