"""Explicit stage boundaries used for planning, discoverability, and future debugging."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    phase: str
    requires: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    status: str = "planned"


PIPELINES = (
    PipelineSpec("foundation", "A", (), ("configuration", "protocol"), ("foundation_run",), "available"),
    PipelineSpec("audit", "B", ("foundation",), ("immutable_sources",), ("registration", "audit_report"), "available"),
    PipelineSpec("provenance", "B", ("audit",), ("registration",), ("provenance", "candidates"), "available"),
    PipelineSpec("verify_captures", "B", ("audit", "provenance"), ("pcap", "provenance"), ("evidence",), "available"),
    PipelineSpec("group", "B", ("provenance", "verify_captures"), ("evidence",), ("groups", "duplicates"), "available"),
    PipelineSpec("global_split", "B", ("group",), ("groups",), ("split_collection",), "available"),
    PipelineSpec("trusted_panel", "B", ("global_split",), ("split_collection",), ("panel",), "available"),
    PipelineSpec("validate_phase_b", "B", ("trusted_panel",), ("phase_b_artifacts",), ("handoff",), "available"),
    PipelineSpec("clients", "C", ("global_split",), ("client_pool",), ("client_manifest", "local_splits")),
    PipelineSpec("preprocessing", "C", ("clients",), ("local_training_manifest",), ("transformer", "feature_contract")),
    PipelineSpec("pretraining_gate", "C", ("preprocessing",), ("all_data_contracts",), ("pretraining_report",)),
    PipelineSpec("baselines", "D", ("pretraining_gate",), ("validated_training_inputs",), ("baseline_results",)),
    PipelineSpec("attack_assessment", "E", ("baselines",), ("updates", "trusted_panel"), ("client_assessments",)),
    PipelineSpec("optimization", "F", ("attack_assessment",), ("assessments", "updates"), ("round_results",)),
    PipelineSpec("experiments", "G", ("optimization",), ("frozen_matrix",), ("development_results",)),
    PipelineSpec("final_evaluation", "H", ("experiments",), ("frozen_checkpoints", "locked_test"), ("final_results",)),
    PipelineSpec("deployment", "H", ("final_evaluation",), ("frozen_system",), ("vm_demonstration",)),
)


def describe_pipelines() -> list[dict[str, object]]:
    return [asdict(stage) for stage in PIPELINES]
