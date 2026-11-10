"""Lazy F command routing, without optimization imports during CLI discovery."""
from edgefl.config import workspace_path
from edgefl.contracts.phase_f import STAGES
from edgefl.optimization.config import load
from edgefl.data.storage import read_json

COMMANDS = tuple(STAGES)

def add_commands(sub):
    required = {
        "prepare-phase-f": ("phase-e-acceptance", "initialization"),
        "optimize-round": ("prepared-f", "run"),
        "run-evolution-federated": ("prepared-f", "attack-plan"),
        "profile-search": ("run",),
        "review-phase-f": ("subject", "rationale-file"),
        "validate-phase-f": ("prepared-f", "run-review", "profile-review", "test-report"),
    }
    for name, fields in required.items():
        parser = sub.add_parser(name, help="Phase F: " + name)
        parser.add_argument("--config", required=True)
        for field in fields:
            parser.add_argument("--" + field, required=True)
        if name in ("optimize-round", "profile-search"):
            parser.add_argument("--round", type=int, required=True)
        if name == "profile-search":
            parser.add_argument("--projection-runs", type=int, required=True)
            parser.add_argument("--projection-rounds", type=int, required=True)
        if name == "run-evolution-federated":
            parser.add_argument("--resume")
        if name == "review-phase-f":
            parser.add_argument("--disposition", choices=("acceptable", "explained_negative", "unresolved"), required=True)
            parser.add_argument("--limitations-file")

def execute(args):
    config = load(workspace_path(args.workspace.resolve(), args.config), args.workspace.resolve())
    def path(value):
        return workspace_path(config.workspace, value)
    if args.command == "prepare-phase-f":
        from edgefl.pipelines.phaseF.phase_f_prepare import execute as run
        result = run(config, path(args.phase_e_acceptance), path(args.initialization))
    elif args.command == "optimize-round":
        from edgefl.pipelines.phaseF.phase_f_optimize import execute as run
        result = run(config, path(args.prepared_f), path(args.run), args.round)
    elif args.command == "run-evolution-federated":
        from edgefl.pipelines.phaseF.phase_f_federated import execute as run
        result = run(config, path(args.prepared_f), path(args.attack_plan), path(args.resume) if args.resume else None)
    elif args.command == "profile-search":
        from edgefl.pipelines.phaseF.phase_f_profile import execute as run
        result = run(config, path(args.run), args.round, args.projection_runs, args.projection_rounds)
    elif args.command == "review-phase-f":
        from edgefl.pipelines.phaseF.phase_f_review import execute as run
        result = run(config, path(args.subject), args.disposition, path(args.rationale_file).read_text(encoding="utf-8"),
                     path(args.limitations_file).read_text(encoding="utf-8") if args.limitations_file else "")
    else:
        from edgefl.pipelines.phaseF.phase_f_validate import execute as run
        result = run(config, path(args.prepared_f), path(args.run_review), path(args.profile_review), path(args.test_report))
    summary = read_json(result)["summary"]
    status = summary.get("status", summary.get("correctness", "complete"))
    if summary.get("attack_coverage", "PASS") != "PASS":
        status = "UNSUCCESSFUL"
    return {"completion": result.relative_to(config.workspace).as_posix(), "status": status, "summary": summary}
