"""Phase G records distinguish experiment intent from execution evidence."""
from dataclasses import dataclass, field, asdict
import hashlib
from edgefl.config import canonical_json

VERSION = "phase-g.v1"
METHODS = ("fedavg", "fedprox", "median", "trimmed_mean", "fltrust", "adaptive", "frozen", "nsga2")
SEEDS = (11, 22, 33, 44, 55)
SCENARIOS = ("near_iid", "dirichlet_moderate", "dirichlet_strong", "specialist")
ATTACKS = ("untargeted_label", "targeted_label", "sign_flip", "update_scaling", "additive_noise", "intermittent_sign")


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class ExperimentCase:
    task: str
    seed: int
    method: str
    condition: str = "clean"
    scenario: str = "dirichlet_moderate"
    fold: str = "A"
    variant: str = "strict"
    clients: int = 10
    participation: float = 1.
    local_epochs: int = 1
    compromised_fraction: float = .2
    label_fraction: float = .3
    update_scale: float = 5.
    activation_round: int = 11
    retention: float = .9
    panel_fraction: float = 1.
    missing_class: bool = False
    population: int = 24
    generations: int = 8
    intervention: str = "full"

    def __post_init__(self):
        if self.task not in ("binary", "multiclass") or self.seed not in SEEDS or self.method not in (*METHODS, "random", "adaptive_capped"):
            raise ValueError("Invalid task, comparison seed or method")
        if self.condition not in ("clean", *ATTACKS) or self.scenario not in SCENARIOS or self.variant not in ("strict", "base"):
            raise ValueError("Invalid experiment condition")
        if self.fold != "A" and self.task != "binary":
            raise ValueError("Protocol B supports binary only")
        if type(self.clients) is not int or self.clients < 3:
            raise ValueError("Scored experiments require at least three clients")
        choices = {"participation": (.5, 1.), "local_epochs": (1, 3, 5),
                   "compromised_fraction": (0., .1, .2, .3, .4), "label_fraction": (.1, .3, .5),
                   "update_scale": (2., 5., 10.), "activation_round": (11, 51), "retention": (.5, .9, .99),
                   "panel_fraction": (.25, .5, 1.), "population": (12, 24, 48), "generations": (4, 8, 16)}
        if any(getattr(self, name) not in values for name, values in choices.items()) or type(self.missing_class) is not bool:
            raise ValueError("Value outside frozen study grid")

    @property
    def case_id(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    status: str
    completion: dict | None = None
    reason: str | None = None
    details: dict = field(default_factory=dict)
