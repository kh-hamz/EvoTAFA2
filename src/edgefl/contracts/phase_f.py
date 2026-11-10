"""Phase F stage names and library-independent search results."""
from dataclasses import dataclass
from edgefl.contracts.records import CandidateEvaluation

VERSION = "phase-f.v1"

class SearchFailure(ValueError):
    """An exhausted search budget or population construction failed."""

STAGES = {
    "prepare-phase-f": {"phase_e_acceptance", "initialization"},
    "optimize-round": {"prepared_f", "run"},
    "run-evolution-federated": {"prepared_f", "attack_plan"},
    "profile-search": {"run"},
    "review-phase-f": {"subject"},
    "validate-phase-f": {"prepared_f", "run_review", "profile_review"},
}

@dataclass(frozen=True)
class SearchResult:
    selected: CandidateEvaluation | None
    candidates: tuple[CandidateEvaluation, ...]
    front: tuple[str, ...]
    generations: tuple[dict, ...]
    initialization: dict
    failure_reason: str | None = None
