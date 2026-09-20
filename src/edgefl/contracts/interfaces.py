"""Structural interfaces for later implementations. Nothing here trains or scores."""

from dataclasses import dataclass
from typing import Protocol

from edgefl.contracts.records import (
    ArtifactRef, ClientAssessment, ClientManifest, ClientUpdate,
    Compatibility, Partition, PartitionRef, RoundResult, require_compatible,
)


@dataclass(frozen=True)
class TrainingRequest:
    client: ClientManifest
    round_id: int
    parent_model: ArtifactRef
    compatibility: Compatibility
    minibatch_seed: int


@dataclass(frozen=True)
class ScoringRequest:
    trusted: PartitionRef
    parent_model: ArtifactRef
    updates: tuple[ClientUpdate, ...]
    compatibility: Compatibility

    def __post_init__(self) -> None:
        if self.trusted.role is not Partition.TRUSTED:
            raise ValueError("Scoring accepts only trusted data")
        for update in self.updates:
            require_compatible(self.compatibility, update.compatibility)
            if update.parent_model != self.parent_model:
                raise ValueError("Scoring cannot mix parent models")


@dataclass(frozen=True)
class AggregationRequest:
    round_id: int
    parent_model: ArtifactRef
    compatibility: Compatibility
    updates: tuple[ClientUpdate, ...]
    assessments: tuple[ClientAssessment, ...]
    trusted: PartitionRef | None
    evolution_seed: int

    def __post_init__(self) -> None:
        if self.trusted is not None and self.trusted.role is not Partition.TRUSTED:
            raise ValueError("Aggregation accepts only trusted data")
        update_ids = [update.client_id for update in self.updates]
        if len(update_ids) != len(set(update_ids)):
            raise ValueError("Duplicate client updates")
        for update in self.updates:
            require_compatible(self.compatibility, update.compatibility)
            if update.round_id != self.round_id or update.parent_model != self.parent_model:
                raise ValueError("Stale round or mixed parent model")
        assessment_ids = [assessment.client_id for assessment in self.assessments]
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError("Duplicate client assessments")
        if set(assessment_ids) - set(update_ids):
            raise ValueError("Assessment without a corresponding update")
        if any(a.round_id != self.round_id for a in self.assessments):
            raise ValueError("Stale assessment")
        # Baseline methods may not request assessments. Methods needing them must
        # require one for every accepted update at their own future boundary.


class ClientTrainer(Protocol):
    def train(self, request: TrainingRequest) -> ClientUpdate: ...


class ClientScorer(Protocol):
    def assess(self, request: ScoringRequest) -> tuple[ClientAssessment, ...]: ...


class Aggregator(Protocol):
    """All aggregators return a checkpoint, including methods without scalar weights."""

    def aggregate(self, request: AggregationRequest) -> RoundResult: ...
