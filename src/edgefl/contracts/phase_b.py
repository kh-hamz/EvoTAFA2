"""Versioned Phase B contracts. Foundation records and ArtifactRef remain unchanged."""

from dataclasses import dataclass
from enum import Enum

from edgefl.contracts.records import ArtifactRef

VERSION = "phase-b.v1"


class EvidenceState(str, Enum):
    VERIFIED = "verified"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"
    QUARANTINED = "quarantined"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class SourceRegistration:
    source: ArtifactRef
    size: int
    kind: str
    schema_version: str = VERSION


@dataclass(frozen=True)
class StageCompletion:
    stage: str
    configuration_sha256: str
    implementation_sha256: str
    inputs: tuple[ArtifactRef, ...]
    artifacts: tuple[ArtifactRef, ...]
    eligible_for_training: bool = False
    schema_version: str = VERSION


# Portable manifest row contracts. Unresolved capture IDs remain empty strings,
# rather than masquerading as verified ArtifactRefs.
MANIFEST_FIELDS = {
    "provenance": ("observation_id", "record", "record_key", "exact_key", "feature_key",
                   "evidence_key", "label", "status", "candidate_count", "reason"),
    "candidates": ("record_key", "capture_id", "source_sha256", "source_record", "exact_key"),
    "evidence": ("observation_id", "label", "status", "reason", "capture_id", "packet",
                 "epoch", "session_id", "session_start", "session_end", "evidence_key"),
    "groups": ("observation_id", "label", "capture_id", "session_id", "group_id",
               "start", "end", "status", "reason", "representative_id"),
    "duplicates": ("observation_id", "representative_id", "reason"),
    "splits": ("fold", "observation_id", "label", "capture_id", "session_id",
               "group_id", "partition", "reason"),
    "panel": ("fold", "observation_id", "label", "capture_id", "session_id", "group_id",
              "partition"),
}
