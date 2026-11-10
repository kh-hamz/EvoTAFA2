"""Explicit research policies; the default is the accepted Phase F algorithm."""
from dataclasses import dataclass

OBJECTIVES = ("classification_error", "benign_fpr", "current_risk", "historical_unreliability")


@dataclass(frozen=True)
class SearchPolicy:
    backend: str = "nsga2"
    cap: float = .5
    initialization: str = "mixed"
    direct_risk: bool = True
    active_objectives: tuple[str, ...] = OBJECTIVES

    def __post_init__(self):
        if type(self.direct_risk) is not bool:
            raise ValueError("Direct risk switch must be boolean")
        if self.backend not in ("nsga2", "random") or self.cap not in (.5, 1.):
            raise ValueError("Unsupported search backend or cap")
        if self.initialization not in ("mixed", "no_expertise", "uniform_random"):
            raise ValueError("Unsupported initialization policy")
        if (not self.active_objectives or len(set(self.active_objectives)) != len(self.active_objectives)
                or any(n not in OBJECTIVES for n in self.active_objectives)
                or tuple(n for n in OBJECTIVES if n in self.active_objectives) != self.active_objectives):
            raise ValueError("Invalid ordered active-objective mask")
