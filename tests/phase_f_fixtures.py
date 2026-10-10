"""Small independent numerical fixtures, distinct from source-verified pipeline fixtures."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from edgefl.contracts.records import (ArtifactRef, Compatibility, ClientUpdate, ClientAssessment,
    ClassExpertise, ClassSupport, Partition, PartitionRef)
from edgefl.contracts.interfaces import AggregationRequest
from edgefl.learning import checkpoints
from edgefl.learning.models import build
from edgefl.data.storage import read_json
from edgefl.optimization.fitness import CandidateEvaluator

ROOT = Path(__file__).resolve().parents[1]

def numerical(owner, device="cpu", classes=2):
    temporary = tempfile.TemporaryDirectory()
    owner(temporary.cleanup)
    root = Path(temporary.name)
    torch.set_num_threads(1)
    comp = Compatibility("a"*64, "b"*64, "c"*64, "mlp-float32-v1", 2, classes)
    parent = {k: torch.zeros_like(v) for k, v in build(2, classes).state_dict().items()}
    checkpoints.save(root / "parent.pt", {"model": parent})
    parent_ref = checkpoints.artifact(root, root / "parent.pt")
    trusted_ref = PartitionRef(ArtifactRef("trusted.json", "d"*64), Partition.TRUSTED)
    clients, updates, assessments = ("a", "b", "c"), [], []
    for i, client in enumerate(clients):
        delta = {k: torch.zeros_like(v) for k, v in parent.items()}
        delta["4.bias"][0:2] = torch.tensor([-2., 2.]) if i == 0 else torch.tensor([1., -1.])
        checkpoints.save(root / (client + ".pt"), {"delta": delta})
        updates.append(ClientUpdate(client, 1, parent_ref, checkpoints.artifact(root, root / (client + ".pt")), comp, 4+i, 0., 100))
        assessments.append(ClientAssessment(client, 1, .5, (ClassExpertise("Attack", .2+.3*i, 50),),
            (("magnitude", .1), ("direction", .2), ("loss", .3)), .1+.2*i, .5, (ClassSupport("Normal", 3), ClassSupport("Attack", 1))))
    data = torch.utils.data.TensorDataset(torch.zeros(4, 2), torch.tensor([0, 0, 0, 1]))
    request = AggregationRequest(1, parent_ref, comp, tuple(updates), tuple(assessments), trusted_ref, 11)
    counts = dict(zip(clients, (4, 5, 6)))
    settings = read_json(ROOT / "configs/phase_f.json")
    settings.update(population_size=6, offspring_generations=1)
    labels = {"Normal": 0, "Attack": 1} if classes == 2 else {"Normal": 0, "Attack": 1, "Other": 2}
    values = {"device": device, "batch_size": 4}
    evaluator = CandidateEvaluator(root, request, data, trusted_ref, labels, values, counts)
    return SimpleNamespace(root=root, request=request, data=data, trusted_ref=trusted_ref, labels=labels,
        values=values, settings=settings, counts=counts, evaluator=evaluator)
