"""Phase D identities; foundation and data contracts retain their versions."""

from dataclasses import dataclass

VERSION = "phase-d.v1"
METHODS = ("majority", "logistic", "mlp", "fedavg", "fedprox", "median", "trimmed_mean", "fltrust")
DISPOSITIONS = ("acceptable", "unresolved", "explained_negative")


@dataclass(frozen=True)
class ExperimentIdentity:
    dataset: str
    task: str
    fold: str
    scenario: str
    variant: str


@dataclass(frozen=True)
class LearningReview:
    disposition: str
    rationale: str
    limitations: str

    def __post_init__(self):
        if self.disposition not in DISPOSITIONS or not self.rationale.strip():
            raise ValueError("Review requires a declared disposition and rationale")
        if self.disposition == "explained_negative" and not self.limitations.strip():
            raise ValueError("Explained negative findings require limitations")
