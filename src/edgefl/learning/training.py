"""Training services with explicit data access and fresh local optimizers."""

import time
import hashlib
import torch
from torch.utils.data import DataLoader
from edgefl.config import workspace_path
from edgefl.contracts.records import ClientUpdate, require_compatible
from edgefl.learning import checkpoints
from edgefl.learning.models import build
from edgefl.reproducibility import derive_seed


def adam(model, values):
    return torch.optim.Adam(model.parameters(), lr=values["learning_rate"], weight_decay=0, foreach=False)


def train_epochs(model, data, optimizer, values, seed, client, round_id, epochs, parent=None, mu=0.0):
    if len(data) == 0:
        raise ValueError("Empty training view")
    device = values["device"]
    generator = torch.Generator().manual_seed(derive_seed(seed, "minibatch", client=client, round_id=round_id))
    loader = DataLoader(data, batch_size=values["batch_size"], shuffle=True, generator=generator, num_workers=0)
    references = {k: v.detach().to(device) for k, v in (parent or {}).items()}
    loss_total, ce_total, observations = 0.0, 0.0, 0
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            ce = torch.nn.functional.cross_entropy(model(x), y)
            loss = ce
            if mu:
                loss = loss + mu / 2 * sum((p - references[name]).square().sum() for name, p in model.named_parameters())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite gradient")
            optimizer.step()
            count = len(y)
            loss_total += float(loss.detach()) * count
            ce_total += float(ce.detach()) * count
            observations += count
    return {"training_objective": loss_total / observations, "training_cross_entropy": ce_total / observations,
            "observations_processed": observations, "optimizer": "adam", "proximal_mu": mu}


class LocalTrainer:
    """Only assigned training views are injected; no trusted or evaluation resource."""

    def __init__(self, root, directory, views, compatibility, values, seed, mu=0.0):
        self.root, self.directory, self.views = root, directory, views
        self.compatibility, self.values, self.seed, self.mu = compatibility, values, seed, mu
        self.diagnostics = {}

    def train(self, request):
        require_compatible(self.compatibility, request.compatibility)
        client = request.client.client_id
        if client not in self.views:
            raise ValueError("Unknown client training view")
        expected_seed = derive_seed(self.seed, "minibatch", client=client, round_id=request.round_id)
        if request.minibatch_seed != expected_seed:
            raise ValueError("Minibatch seed mismatch")
        start = time.perf_counter()
        parent = checkpoints.load(workspace_path(self.root, request.parent_model.path), request.parent_model.sha256)["model"]
        model = build(self.compatibility.input_features, self.compatibility.output_classes).to(self.values["device"])
        checkpoints.validate_state(parent, model.state_dict())
        model.load_state_dict(parent)
        report = train_epochs(model, self.views[client], adam(model, self.values), self.values,
                              self.seed, client, request.round_id, self.values["local_epochs"], parent, self.mu)
        state = checkpoints.cpu_state(model)
        delta = {k: state[k] - parent[k] for k in state}
        checkpoints.validate_state(delta, parent)
        token = hashlib.sha256(client.encode()).hexdigest()[:16]
        path = self.directory / "updates" / f"round-{request.round_id:04d}-{token}.pt"
        checkpoints.save(path, {"delta": delta})
        elapsed = time.perf_counter() - start
        self.diagnostics[client] = {**report, "seconds": elapsed}
        return ClientUpdate(client, request.round_id, request.parent_model, checkpoints.artifact(self.root, path),
                            self.compatibility, len(self.views[client]), elapsed,
                            sum(v.numel() * v.element_size() for v in delta.values()))


class ServerReference:
    """FLTrust-only root training, from the same parent and frozen trusted panel."""

    def __init__(self, trusted, values, seed, compatibility):
        self.trusted, self.values, self.seed, self.compatibility = trusted, values, seed, compatibility

    def update(self, parent, round_id):
        model = build(self.compatibility.input_features, self.compatibility.output_classes).to(self.values["device"])
        model.load_state_dict(parent)
        report = train_epochs(model, self.trusted, adam(model, self.values), self.values,
                              self.seed, "server_reference", round_id, self.values["root_epochs"])
        state = checkpoints.cpu_state(model)
        return {k: state[k] - parent[k] for k in state}, {**report, "role": "trusted", "paper_optimizer_adaptation": "adam"}
