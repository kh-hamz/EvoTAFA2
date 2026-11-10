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
    PipelineSpec("diagnose_captures", "B", ("audit", "provenance"), ("pcap", "provenance"), ("diagnostic_review",), "available"),
    PipelineSpec("group", "B", ("provenance", "verify_captures"), ("evidence",), ("groups", "duplicates"), "available"),
    PipelineSpec("global_split", "B", ("group",), ("groups",), ("split_collection",), "available"),
    PipelineSpec("trusted_panel", "B", ("global_split",), ("split_collection",), ("panel",), "available"),
    PipelineSpec("validate_phase_b", "B", ("trusted_panel",), ("phase_b_artifacts",), ("handoff",), "available"),
    PipelineSpec("assign_clients", "C", ("validate_phase_b", "global_split"),
                 ("verified_client_pool",), ("client_assignments", "distribution_diagnostics"), "available"),
    PipelineSpec("local_split", "C", ("assign_clients",), ("client_assignments",),
                 ("client_manifests", "local_splits"), "available"),
    PipelineSpec("preprocessing", "C", ("local_split", "provenance"),
                 ("local_training_manifest",), ("transformer", "feature_contract", "fit_rows"), "available"),
    PipelineSpec("pretraining_gate", "C", ("preprocessing", "validate_phase_b"),
                 ("all_phase_c_contracts",), ("pretraining_report", "training_inputs"), "available"),
    PipelineSpec("prepare_learning_data", "D", ("pretraining_gate",), ("validated_training_inputs",), ("learning_arrays",), "available"),
    PipelineSpec("initialize_model", "D", ("prepare_learning_data",), ("learning_arrays",), ("common_initializations",), "available"),
    PipelineSpec("train_centralized", "D", ("initialize_model",), ("local_training_union",), ("centralized_results",), "available"),
    PipelineSpec("run_federated", "D", ("initialize_model", "train_centralized"), ("client_training", "applicable_reviews"), ("federated_results",), "available"),
    PipelineSpec("review_learning", "D", ("train_centralized",), ("completed_training_run",), ("learning_review",), "available"),
    PipelineSpec("baselines", "D", ("run_federated", "review_learning"), ("reviewed_phase_d_methods",), ("baseline_results",), "available"),
    PipelineSpec("prepare_phase_e", "E", ("baselines",), ("verified_learning_data", "phase_d_acceptance"), ("original_label_metadata",), "available"),
    PipelineSpec("plan_attacks", "E", ("prepare_phase_e",), ("training_metadata",), ("simulator_plan",), "available"),
    PipelineSpec("clean_pilot", "E", ("prepare_phase_e",), ("verified_learning_data",), ("raw_clean_observations",), "available"),
    PipelineSpec("calibrate_risk", "E", ("clean_pilot",), ("fit_pilots", "audit_pilot"), ("frozen_calibration", "independent_audit"), "available"),
    PipelineSpec("attack_assessment", "E", ("calibrate_risk", "plan_attacks"), ("updates", "trusted_panel", "reviewed_calibration"), ("client_assessments", "weighted_rounds", "reputation"), "available"),
    PipelineSpec("assess_round", "E", ("attack_assessment",), ("recorded_round",), ("reassessment",), "available"),
    PipelineSpec("review_phase_e", "E", ("calibrate_risk",), ("completed_evidence",), ("phase_e_review",), "available"),
    PipelineSpec("validate_phase_e", "E", ("attack_assessment", "review_phase_e"), ("matched_reviews", "current_tests"), ("scoped_acceptance",), "available"),
    PipelineSpec("prepare_phase_f", "F", ("validate_phase_e",), ("phase_e_acceptance", "initialization"), ("search_inputs",), "available"),
    PipelineSpec("optimize_round", "F", ("prepare_phase_f",), ("recorded_round",), ("search_replay",), "available"),
    PipelineSpec("optimization", "F", ("prepare_phase_f",), ("assessments", "updates", "trusted_panel"), ("round_results",), "available"),
    PipelineSpec("profile_search", "F", ("optimization",), ("completed_round",), ("actual_search_profile",), "available"),
    PipelineSpec("review_phase_f", "F", ("optimization",), ("completed_evidence",), ("phase_f_review",), "available"),
    PipelineSpec("validate_phase_f", "F", ("profile_search", "review_phase_f"), ("current_tests", "reviewed_evidence"), ("scoped_acceptance",), "available"),
    PipelineSpec("compile_experiments", "G", (), ("experiment_specification",), ("matrix", "dependencies"), "available"),
    PipelineSpec("prepare_experiment", "G", ("compile_experiments", "validate_phase_f"), ("accepted_scope",), ("prepared_case",), "available"),
    PipelineSpec("experiment_run", "G", ("prepare_experiment",), ("prepared_case",), ("case_results",), "available"),
    PipelineSpec("profile_experiments", "G", ("experiment_run",), ("completed_cases",), ("cost_projection",), "available"),
    PipelineSpec("experiments", "G", ("profile_experiments",), ("frozen_matrix", "cost_review"), ("development_results",), "available"),
    PipelineSpec("review_experiments", "G", ("experiment_run",), ("explicit_decisions",), ("batch_reviews",), "available"),
    PipelineSpec("validate_phase_g", "G", ("experiments", "review_experiments"), ("coverage", "current_tests"), ("scoped_acceptance",), "available"),
    PipelineSpec("final_evaluation", "H", ("experiments",), ("frozen_checkpoints", "locked_test"), ("final_results",)),
    PipelineSpec("deployment", "H", ("final_evaluation",), ("frozen_system",), ("vm_demonstration",)),
)


def describe_pipelines() -> list[dict[str, object]]:
    return [asdict(stage) for stage in PIPELINES]
