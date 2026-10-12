"""Thin Phase B command registration and routing; no processing algorithms."""

from edgefl.config import workspace_path
from edgefl.data.config import load
from edgefl.data.storage import read_json

COMMANDS = ("audit","provenance","verify-captures","group","global-split","trusted-panel","validate-phase-b")


def add_commands(sub):
    for name in COMMANDS:
        parser = sub.add_parser(name,help="Phase B: " + name)
        parser.add_argument("--config",required=True)
        if name != "audit":
            parser.add_argument("--dataset",choices=("primary","smoke"),required=True)
        dependencies = {
            "audit":("foundation",), "provenance":("audit",),
            "verify-captures":("audit","provenance"), "group":("provenance","evidence"),
            "global-split":("groups",), "trusted-panel":("splits",),
            "validate-phase-b":("audit","provenance","evidence","groups","splits-a","splits-b","panel-a","panel-b"),
        }[name]
        for dependency in dependencies:
            parser.add_argument("--"+dependency,required=True)
        if name == "global-split":
            parser.add_argument("--protocol",choices=("A","B"),required=True)


def execute(args):
    root = args.workspace.resolve()
    config = load(workspace_path(root,args.config),root)
    def path(name):
        return workspace_path(root,getattr(args,name))
    if args.command == "audit":
        from edgefl.pipelines.phase_b_audit import execute
        completion = execute(config,path("foundation"))
    elif args.command == "provenance":
        from edgefl.pipelines.phase_b_provenance import execute
        completion = execute(config,path("audit"),args.dataset)
    elif args.command == "verify-captures":
        from edgefl.pipelines.phase_b_captures import execute
        completion = execute(config,path("audit"),path("provenance"),args.dataset)
    elif args.command == "group":
        from edgefl.pipelines.phase_b_group import execute
        completion = execute(config,path("provenance"),path("evidence"),args.dataset)
    elif args.command == "global-split":
        from edgefl.pipelines.phase_b_split import execute
        completion = execute(config,path("groups"),args.dataset,args.protocol)
    elif args.command == "trusted-panel":
        from edgefl.pipelines.phase_b_panel import execute
        completion = execute(config,path("splits"),args.dataset)
    else:
        from edgefl.pipelines.phase_b_validate import EXPECTED,execute
        completion = execute(config,{key:path(key) for key in EXPECTED},args.dataset)
    from edgefl.pipelines.phase_b_integrity import assert_registered_sources
    assert_registered_sources(config,completion)
    metadata = read_json(completion)
    result = {"completion":completion.relative_to(root).as_posix(),"eligible_for_training":False}
    if args.command == "validate-phase-b":
        report = read_json(workspace_path(root,metadata["artifacts"]["summary"]["path"]))
        result["status"] = report["status"]
    return result
