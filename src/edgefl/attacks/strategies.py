"""Pure perturbations and immutable label overlays."""

import torch
from edgefl.learning.metrics import norm


class PoisonedTrainingView:
    def __init__(self, original, changes):
        self.original = original
        self.changes = {entry["row"]: entry["replacement"] for entry in changes}
        if len(self.changes) != len(changes) or any(not 0 <= i < len(original) for i in self.changes):
            raise ValueError("Invalid poisoned row selection")
        for entry in changes:
            if original[entry["row"]][1] != entry["original_target"] or entry["replacement"] == entry["original_target"]:
                raise ValueError("Poison overlay disagrees with original label")

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        x, y = self.original[index]
        return x, self.changes.get(index, y)


def transform(delta, condition, seed, scale=5.0):
    if condition in ("sign_flip", "intermittent_sign"):
        return {k: -v for k, v in delta.items()}
    if condition == "update_scaling":
        return {k: v * scale for k, v in delta.items()}
    if condition == "additive_noise":
        generator = torch.Generator().manual_seed(seed)
        noise = {k: torch.randn(v.shape, generator=generator, dtype=torch.float64) for k, v in sorted(delta.items())}
        base, random_norm = norm(delta), norm(noise)
        if random_norm == 0:
            raise ValueError("Degenerate Gaussian noise draw")
        return {k: (v.double() + noise[k] * (base / random_norm)).to(v.dtype) for k, v in delta.items()}
    if condition not in ("clean", "untargeted_label", "targeted_label"):
        raise ValueError("Unknown delta transformation")
    return {k: v.clone() for k, v in delta.items()}
