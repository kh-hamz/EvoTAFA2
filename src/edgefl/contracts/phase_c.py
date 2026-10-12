"""Versioned Phase C contracts for client data and frozen preprocessing."""

from dataclasses import dataclass
from enum import Enum


VERSION = "phase-c.v2"


@dataclass(frozen=True)
class SupportPolicy:
    """Binary minima are opt-in metric requirements, never inferred from task."""

    profile: str = "server_scored"
    assigned: tuple[int, int] = (0, 0)
    training: tuple[int, int] = (0, 0)
    validation: tuple[int, int] = (0, 0)

    def __post_init__(self):
        if self.profile not in ("server_scored", "local_binary_metrics"):
            raise ValueError("Unknown support profile")
        if any(type(n) is not int or n < 0 for pair in
               (self.assigned, self.training, self.validation) for n in pair):
            raise ValueError("Support minima must be nonnegative integers")
        if self.profile == "local_binary_metrics" and min(*self.training, *self.validation) < 1:
            raise ValueError("Local binary metrics require both classes in train and validation")

    def violations(self, counts, role):
        actual = (counts.get("Normal", 0), sum(n for label, n in counts.items() if label != "Normal"))
        minima = getattr(self, role)
        if role == "assigned":
            minima = tuple(max(n, train + validation) for n, train, validation in
                           zip(self.assigned, self.training, self.validation))
        return [{"constraint": category, "actual": count, "minimum": minimum}
                for category, count, minimum in zip(("benign", "attack"), actual, minima)
                if count < minimum]


@dataclass(frozen=True)
class ExperimentContract:
    dataset: str
    task: str
    protocol: str
    fold: str
    interpretation: str
    scope: str
    labels: tuple[str, ...]
    held_out_captures: tuple[str, ...] = ()

    def __post_init__(self):
        if self.task not in ("binary", "multiclass") or self.protocol not in ("A", "B"):
            raise ValueError("Invalid experiment task/protocol")
        if self.protocol == "B" and self.task != "binary":
            raise ValueError("Protocol B currently supports binary evaluation only")
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise ValueError("Experiment requires a unique common label vocabulary")


@dataclass(frozen=True)
class AllocationResult:
    ownership: tuple[tuple[str, str], ...]
    targets: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]


@dataclass(frozen=True)
class TrainingReadyBundle:
    dataset: str
    task: str
    fold: str
    scenario: str
    references: tuple[tuple[str, str, str], ...]


class ScenarioKind(str, Enum):
    NEAR_IID = "near_iid"
    DIRICHLET = "dirichlet"
    SPECIALIST = "specialist"


class LocalRole(str, Enum):
    TRAIN = "local_train"
    VALIDATION = "local_validation"


@dataclass(frozen=True)
class ClientConstraint:
    minimum_observations: int
    minimum_training_observations: int
    minimum_groups: int
    require_binary_support: bool

    def __post_init__(self) -> None:
        if min(self.minimum_observations, self.minimum_training_observations,
               self.minimum_groups) < 1:
            raise ValueError("Client constraints must be positive")
        if self.minimum_training_observations > self.minimum_observations:
            raise ValueError("Training minimum cannot exceed the client minimum")


MANIFEST_FIELDS = {
    "assignments": ("fold", "scenario", "client_id", "observation_id", "label",
                    "capture_id", "session_id", "group_id"),
    "local_splits": ("fold", "scenario", "client_id", "observation_id", "label",
                     "capture_id", "session_id", "group_id", "partition"),
    "fit_rows": ("client_id", "observation_id", "record", "label"),
    "feature_dictionary": ("variant", "index", "output_feature", "source_feature",
                           "kind", "detail"),
}
