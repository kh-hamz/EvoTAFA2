"""Accept only a complete, explicitly scoped set of reviewed Phase D methods."""
from edgefl.learning.storage import stage, load_stage, checked, write_json
from edgefl.learning.review import validate_scope


def execute(config, reviews, test_report):
    summary = validate_scope(config, reviews, test_report)
    first = load_stage(config, reviews[0], "review-learning")
    with stage(config, "validate-phase-d", first["identity"], checked(config.workspace, first["gate"]),
               {f"review_{i}": path for i, path in enumerate(reviews)},
               {"test_report": summary["software_evidence"]}) as attempt:
        write_json(attempt.directory / "acceptance.json", summary)
        return attempt.finish(summary)
