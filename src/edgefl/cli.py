"""Research foundation plus independently executable Phase B and Phase C commands."""

import argparse
import json
import sys
from pathlib import Path

from edgefl.config import ConfigurationError, load_config, workspace_path
from edgefl.pipelines.catalog import describe_pipelines
from edgefl.runs import initialize_foundation_run
from edgefl.pipelines.phase_b_cli import COMMANDS, add_commands, execute as execute_phase_b
from edgefl.pipelines.phase_c_cli import (COMMANDS as PHASE_C_COMMANDS,
                                          add_commands as add_phase_c_commands,
                                          execute as execute_phase_c)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-config", help="Validate only; no output files or dataset access")
    validate.add_argument("--config", default="configs/phase_a.json")
    sub.add_parser("pipelines", help="Show available/planned pipeline boundaries")
    initialize = sub.add_parser("init-run", help="Create a foundation record, not a training run")
    initialize.add_argument("--config", default="configs/phase_a.json")
    initialize.add_argument("--seed", type=int, default=11)
    add_commands(sub)
    add_phase_c_commands(sub)
    args = parser.parse_args(argv)
    try:
        if args.command in PHASE_C_COMMANDS:
            result = execute_phase_c(args)
            print(json.dumps(result, indent=2))
            return 1 if result.get("status") in ("FAIL", "PASS_WITH_LIMITATIONS") else 0
        if args.command in COMMANDS:
            result = execute_phase_b(args)
            print(json.dumps(result, indent=2))
            return 1 if result.get("status") == "FAIL" else 0
        if args.command == "pipelines":
            print(json.dumps(describe_pipelines(), indent=2))
            return 0
        root = args.workspace.resolve()
        config = load_config(workspace_path(root, args.config), root)
        if args.command == "validate-config":
            result = {"valid": True, "scope": "phase_a", "configuration_sha256": config.sha256}
        else:
            destination = initialize_foundation_run(config, args.seed)
            result = {"run_directory": str(destination), "eligible_for_training": False}
        print(json.dumps(result, indent=2))
        return 0
    except (ConfigurationError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
