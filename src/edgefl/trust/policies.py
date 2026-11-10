"""Decision policies are separate from simulator truth and diagnostic measurements."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrustPolicy:
    signals: tuple[str, ...] = ("magnitude", "direction", "loss")
    reputation: bool = True
    direct_risk: bool = True
    clipping_threshold: float | None = None
    fixed_weight_cap: float | None = None

    def __post_init__(self):
        names = ("magnitude", "direction", "loss")
        if not self.signals or tuple(n for n in names if n in self.signals) != self.signals:
            raise ValueError("Invalid risk component mask")
        if type(self.reputation) is not bool or type(self.direct_risk) is not bool:
            raise ValueError("Decision switches must be booleans")
        if self.clipping_threshold is not None and (not math.isfinite(self.clipping_threshold) or self.clipping_threshold <= 0):
            raise ValueError("Clipping requires a positive finite frozen threshold")
        if self.fixed_weight_cap not in (None, .5, 1.):
            raise ValueError("Invalid fixed-weight cap")
