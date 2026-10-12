"""Simulator wrapper: immutable label views and separate submitted update artifacts."""

from dataclasses import replace
import hashlib
import torch
from edgefl.config import workspace_path
from edgefl.learning import checkpoints
from edgefl.attacks.plan import active, attack_seed
from edgefl.attacks.strategies import PoisonedTrainingView, transform


class AttackTrainer:
    def __init__(self, trainer, plan):
        self.trainer, self.plan = trainer, plan
        self.originals = dict(trainer.views)
        self.overlays = {c: PoisonedTrainingView(self.originals[c], entries) for c, entries in plan["changes"].items() if entries}
        self.diagnostics = trainer.diagnostics
        self.events = {}

    def train(self, request):
        client, current = request.client.client_id, request.round_id
        attacking = active(self.plan, client, current)
        view = self.overlays.get(client, self.originals[client]) if attacking else self.originals[client]
        self.trainer.views = {**self.originals, client: view}
        try:
            update = self.trainer.train(request)
        finally:
            self.trainer.views = dict(self.originals)
        effective = attacking and client in self.overlays
        if attacking and self.plan["condition"] not in ("targeted_label", "untargeted_label"):
            delta = checkpoints.load(workspace_path(self.trainer.root, update.delta.path), update.delta.sha256)["delta"]
            submitted = transform(delta, self.plan["condition"], attack_seed(self.plan["seed"], "delta_noise", client, current), self.plan["scale"])
            effective = any(not torch.equal(delta[k], submitted[k]) for k in delta)
            token = hashlib.sha256(client.encode()).hexdigest()[:16]
            path = self.trainer.directory / "submissions" / f"round-{current:04d}-{token}.pt"
            checkpoints.save(path, {"delta": submitted})
            update = replace(update, delta=checkpoints.artifact(self.trainer.root, path))
        self.events[client] = {"compromised": client in self.plan["compromised"], "scheduled_active": attacking,
                               "effective": bool(effective), "changed_training_rows": len(self.plan["changes"].get(client, [])) if attacking else 0}
        return update
