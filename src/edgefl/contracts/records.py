"""Immutable record contracts shared by future pipelines.

References describe artifacts, not loaded tensors. Access-role checks are application
boundaries, not an operating-system security sandbox.
"""

import math
from dataclasses import dataclass
from enum import Enum

CONTRACT_VERSION = "1.0"


class Partition(str, Enum):
    CLIENT_POOL = "client_pool"
    LOCAL_TRAIN = "local_train"
    LOCAL_VALIDATION = "local_validation"
    TRUSTED = "trusted"
    SELECTION_VALIDATION = "selection_validation"
    FINAL_TEST = "final_test"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class ArtifactRef:
    """A verified artifact identity. Pending Phase B sources are not ArtifactRefs."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("Artifact path cannot be empty")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("Artifact SHA-256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    sources: tuple[ArtifactRef, ...]
    schema: ArtifactRef
    provenance: ArtifactRef
    provenance_status: str
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class SplitEntry:
    observation_id: str
    capture_id: str | None
    session_id: str | None
    group_id: str
    partition: Partition
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class PartitionRef:
    manifest: ArtifactRef
    role: Partition


@dataclass(frozen=True)
class ClassSupport:
    label: str
    count: int

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0:
            raise ValueError("Class support must be a nonnegative integer")


@dataclass(frozen=True)
class ClientManifest:
    client_id: str
    groups: ArtifactRef
    training: PartitionRef
    local_validation: PartitionRef
    support: tuple[ClassSupport, ...]
    partition_seed: int

    def __post_init__(self) -> None:
        if self.training.role is not Partition.LOCAL_TRAIN:
            raise ValueError("Client training requires local_train")
        if self.local_validation.role is not Partition.LOCAL_VALIDATION:
            raise ValueError("Client validation requires local_validation")


@dataclass(frozen=True)
class Compatibility:
    """Compare the full record at boundaries, not only the number of features."""

    preprocessing_sha256: str
    feature_order_sha256: str
    label_map_sha256: str
    model_signature: str
    input_features: int
    output_classes: int
    contract_version: str = CONTRACT_VERSION


def require_compatible(expected: Compatibility, actual: Compatibility) -> None:
    if expected != actual:
        raise ValueError("Preprocessing, feature order, labels, model, or contract mismatch")


@dataclass(frozen=True)
class ClientUpdate:
    client_id: str
    round_id: int
    parent_model: ArtifactRef
    delta: ArtifactRef
    compatibility: Compatibility
    registered_training_count: int
    training_seconds: float
    transmitted_bytes: int


@dataclass(frozen=True)
class ClassExpertise:
    label: str
    score: float | None
    support: int
    # None means unsupported; it must not be silently converted to zero expertise.


@dataclass(frozen=True)
class ClientAssessment:
    client_id: str
    round_id: int
    quality: float
    expertise: tuple[ClassExpertise, ...]
    risk_components: tuple[tuple[str, float], ...]
    current_risk: float
    prior_reputation: float
    support: tuple[ClassSupport, ...]


@dataclass(frozen=True)
class FitnessVector:
    classification_error: float
    benign_fpr: float
    current_risk: float
    historical_unreliability: float

    def __post_init__(self) -> None:
        values = (
            self.classification_error, self.benign_fpr,
            self.current_risk, self.historical_unreliability,
        )
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Fitness objectives must be finite values in [0, 1]")


@dataclass(frozen=True)
class ClientWeight:
    client_id: str
    weight: float


@dataclass(frozen=True)
class CandidateEvaluation:
    evaluation_id: str
    weights: tuple[ClientWeight, ...]
    objectives: FitnessVector | None
    feasible: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.feasible:
            if self.objectives is None or self.failure_reason is not None or not self.weights:
                raise ValueError("Feasible candidates require objectives, weights, and no failure")
            ids = [entry.client_id for entry in self.weights]
            if len(ids) != len(set(ids)):
                raise ValueError("Candidate contains duplicate clients")
            if any(not math.isfinite(e.weight) or e.weight < 0 for e in self.weights):
                raise ValueError("Candidate weights must be finite and nonnegative")
            if not math.isclose(sum(e.weight for e in self.weights), 1, abs_tol=1e-9, rel_tol=0):
                raise ValueError("Candidate weights must sum to one")
        elif not self.failure_reason:
            raise ValueError("Infeasible candidates require an explicit failure reason")


@dataclass(frozen=True)
class ReputationTransition:
    client_id: str
    previous: float
    current: float
    reason: str


@dataclass(frozen=True)
class RoundResult:
    round_id: int
    checkpoint: ArtifactRef
    weights: tuple[ClientWeight, ...] | None
    diagnostics: ArtifactRef
    reputation_transitions: tuple[ReputationTransition, ...]
    failure_reason: str | None = None
    # None weights supports coordinate-wise robust methods with no scalar weight vector.


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    protocol_id: str
    seed: int
    metrics: ArtifactRef
    costs: ArtifactRef
    status: str
    contract_version: str = CONTRACT_VERSION
