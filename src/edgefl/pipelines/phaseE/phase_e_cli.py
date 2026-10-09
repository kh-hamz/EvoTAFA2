"""Lazy Phase E command routing; foundation commands do not import training libraries."""

from edgefl.config import workspace_path
from edgefl.contracts.phase_e import CONDITIONS, METHODS, STAGES
from edgefl.trust.config import load
from edgefl.data.storage import read_json

COMMANDS = tuple(STAGES)


def add_commands(sub):
    required = {
        "prepare-phase-e": ("prepared", "phase-d-acceptance"),
        "plan-attacks": ("metadata",),
        "run-clean-pilot": ("metadata", "phase-d-acceptance"),
        "calibrate-risk": ("metadata", "audit-pilot"),
        "run-trust-federated": ("metadata", "initialization", "phase-d-acceptance", "attack-plan", "calibration", "calibration-review"),
        "assess-round": ("run",),
        "review-phase-e": ("subject", "rationale-file"),
        "validate-phase-e": ("metadata", "calibration", "calibration-review", "test-report"),
    }
    for name in COMMANDS:
        parser = sub.add_parser(name, help="Phase E: " + name)
        parser.add_argument("--config", required=True)
        for field in required[name]:
            parser.add_argument("--" + field, required=True)
        if name in ("plan-attacks", "run-clean-pilot", "run-trust-federated", "validate-phase-e"):
            parser.add_argument("--seed", type=int, required=True)
        if name == "plan-attacks":
            parser.add_argument("--condition", choices=CONDITIONS, required=True)
        elif name == "run-clean-pilot":
            parser.add_argument("--resume")
        elif name == "calibrate-risk":
            parser.add_argument("--fit-pilot", action="append", required=True)
        elif name == "run-trust-federated":
            parser.add_argument("--method", choices=METHODS, required=True)
            parser.add_argument("--resume")
        elif name == "assess-round":
            parser.add_argument("--round", type=int, required=True)
        elif name == "review-phase-e":
            parser.add_argument("--disposition", choices=("acceptable", "explained_negative", "unresolved"), required=True)
            parser.add_argument("--limitations-file")
        elif name == "validate-phase-e":
            parser.add_argument("--review", action="append", required=True)


def execute(args):
    config = load(workspace_path(args.workspace.resolve(), args.config), args.workspace.resolve())
    def path(value):
        return workspace_path(config.workspace, value)
    if args.command == "prepare-phase-e":
        from edgefl.pipelines.phaseE.phase_e_prepare import execute as run
        result = run(config, path(args.prepared), path(args.phase_d_acceptance))
    elif args.command == "plan-attacks":
        from edgefl.pipelines.phaseE.phase_e_attack_plan import execute as run
        result = run(config, path(args.metadata), args.seed, args.condition)
    elif args.command == "run-clean-pilot":
        from edgefl.pipelines.phaseE.phase_e_pilot import execute as run
        result = run(config, path(args.metadata), path(args.phase_d_acceptance), args.seed, path(args.resume) if args.resume else None)
    elif args.command == "calibrate-risk":
        from edgefl.pipelines.phaseE.phase_e_calibrate import execute as run
        result = run(config, path(args.metadata), [path(p) for p in args.fit_pilot], path(args.audit_pilot))
    elif args.command == "run-trust-federated":
        from edgefl.pipelines.phaseE.phase_e_federated import execute as run
        result = run(config, path(args.metadata), path(args.initialization), path(args.phase_d_acceptance), path(args.attack_plan),
                     path(args.calibration), path(args.calibration_review), args.method, args.seed, path(args.resume) if args.resume else None)
    elif args.command == "assess-round":
        from edgefl.pipelines.phaseE.phase_e_assess import execute as run
        result = run(config, path(args.run), args.round)
    elif args.command == "review-phase-e":
        from edgefl.pipelines.phaseE.phase_e_review import execute as run
        result = run(config, path(args.subject), args.disposition, path(args.rationale_file).read_text(encoding="utf-8"),
                     path(args.limitations_file).read_text(encoding="utf-8") if args.limitations_file else "")
    else:
        from edgefl.pipelines.phaseE.phase_e_validate import execute as run
        result = run(config, path(args.metadata), path(args.calibration), path(args.calibration_review),
                     [path(p) for p in args.review], path(args.test_report), args.seed)
    summary = read_json(result)["summary"]
    status = summary.get("status", summary.get("correctness", "complete"))
    if summary.get("attack_coverage", "PASS") != "PASS":
        status = "UNSUCCESSFUL"
    return {"completion": result.relative_to(config.workspace).as_posix(), "status": status, "summary": summary}
