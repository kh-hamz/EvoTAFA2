"""Phase-neutral round interface and the existing clean baseline behavior."""

import math
import time
from dataclasses import dataclass, field
from typing import Protocol
import torch
from edgefl.config import workspace_path
from edgefl.contracts.records import ClientManifest, PartitionRef, Partition, RoundResult
from edgefl.contracts.interfaces import TrainingRequest, AggregationRequest
from edgefl.learning import checkpoints
from edgefl.learning.training import LocalTrainer, ServerReference
from edgefl.learning.aggregation import BaselineAggregator
from edgefl.learning.metrics import magnitude
from edgefl.reproducibility import derive_seed


@dataclass
class FederatedRound:
    result: RoundResult
    client_reports: dict
    training_seconds: float
    root_report: dict | None = None
    root_seconds: float = 0.0
    extra: dict = field(default_factory=dict)


class FederatedSession(Protocol):
    def execute(self, current, parent_ref, parent) -> FederatedRound: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...


class BaselineSession:
    def __init__(self, config, data, directory, comp, data_ref, method, seed):
        self.config, self.data, self.directory = config, data, directory
        self.comp, self.data_ref, self.method, self.seed = comp, data_ref, method, seed
        self.values = config.values
        self.clients = {c: data.view("local_train", c) for c in data.clients}
        self.counts = {c: len(v) for c, v in self.clients.items()}
        self.trainer = LocalTrainer(config.workspace, directory, self.clients, comp, self.values, seed,
                                    self.values["fedprox_mu"] if method == "fedprox" else 0)
        self.root_service = ServerReference(data.view("trusted"), self.values, seed, comp) if method == "fltrust" else None

    def select(self, current):
        generator = torch.Generator().manual_seed(derive_seed(self.seed, "participation", round_id=current))
        n = max(1, math.ceil(len(self.data.clients) * self.values["participation"]))
        return sorted(self.data.clients[i] for i in torch.randperm(len(self.data.clients), generator=generator).tolist()[:n])

    def request(self, client, current, parent_ref):
        manifest = ClientManifest(client, self.data_ref, PartitionRef(self.data_ref, Partition.LOCAL_TRAIN),
                                  PartitionRef(self.data_ref, Partition.LOCAL_VALIDATION), (), self.seed)
        return TrainingRequest(manifest, current, parent_ref, self.comp,
                               derive_seed(self.seed, "minibatch", client=client, round_id=current))

    def root_update(self, parent, current):
        if self.root_service is None:
            return None, None, 0.0
        tick = time.perf_counter()
        delta, report = self.root_service.update(parent, current)
        if self.values["device"] == "cuda":
            torch.cuda.synchronize()
        return delta, report, time.perf_counter() - tick

    def execute(self, current, parent_ref, parent):
        updates, reports = [], {}
        for client in self.select(current):
            update = self.trainer.train(self.request(client, current, parent_ref))
            updates.append(update)
            delta = checkpoints.load(workspace_path(self.config.workspace, update.delta.path), update.delta.sha256)["delta"]
            reports[client] = {**self.trainer.diagnostics[client], "update": magnitude(delta, parent), "transmitted_bytes": update.transmitted_bytes}
        root, root_report, root_seconds = self.root_update(parent, current)
        aggregate = BaselineAggregator(self.config.workspace, self.directory, self.method, self.counts, self.values["trim_fraction"], root)
        request = AggregationRequest(current, parent_ref, self.comp, tuple(updates), (),
                                     PartitionRef(self.data_ref, Partition.TRUSTED) if self.root_service else None,
                                     derive_seed(self.seed, "evolution", round_id=current))
        return FederatedRound(aggregate.aggregate(request), reports, sum(u.training_seconds for u in updates), root_report, root_seconds)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("Unexpected baseline session state")
