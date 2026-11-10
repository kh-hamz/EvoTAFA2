"""Round-local, trusted-only evaluation of actual weighted models."""
import math
import time
import torch
from dataclasses import asdict
from torch.utils.data import DataLoader
from edgefl.config import workspace_path
from edgefl.contracts.records import CandidateEvaluation, ClientWeight, FitnessVector, Partition
from edgefl.learning import checkpoints
from edgefl.learning.models import build
from edgefl.learning.metrics import summarize
from edgefl.optimization.chromosomes import validate, identity

class NumericalCandidateError(ValueError):
    """A finite chromosome produces invalid model values."""

class CandidateEvaluator:
    def __init__(self, root, request, trusted, trusted_ref, labels, values, counts, policy=None):
        from edgefl.optimization.policies import SearchPolicy
        self.policy = policy or SearchPolicy()
        if request.trusted != trusted_ref or trusted_ref.role is not Partition.TRUSTED:
            raise ValueError("Fitness requires the bound trusted panel")
        self.request, self.trusted, self.labels, self.values = request, trusted, labels, values
        self.clients = tuple(sorted(u.client_id for u in request.updates))
        if len(self.clients) < 3 or set(self.clients) != {a.client_id for a in request.assessments}:
            raise ValueError("Fitness requires three valid, fully assessed clients")
        self.assessments = {a.client_id: a for a in request.assessments}
        for a in request.assessments:
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (a.quality, a.current_risk, a.prior_reputation)):
                raise ValueError("Invalid bounded assessment")
        self.parent = checkpoints.load(workspace_path(root, request.parent_model.path), request.parent_model.sha256)["model"]
        self.deltas = {}
        for u in request.updates:
            if type(u.registered_training_count) is not int or u.registered_training_count != counts.get(u.client_id):
                raise ValueError("Unregistered client count")
            delta = checkpoints.load(workspace_path(root, u.delta.path), u.delta.sha256)["delta"]
            checkpoints.validate_state(delta, self.parent)
            self.deltas[u.client_id] = delta
        with torch.random.fork_rng(devices=[]):
            self.model = build(request.compatibility.input_features, request.compatibility.output_classes).to(values["device"])
        checkpoints.validate_state(self.parent, self.model.state_dict())
        self.benign = labels["Normal"]
        self.support = torch.zeros(len(labels), dtype=torch.int64)
        for _, y in DataLoader(trusted, batch_size=values["batch_size"], shuffle=False, num_workers=0):
            if (y < 0).any() or (y >= len(labels)).any():
                raise ValueError("Invalid trusted labels")
            self.support += torch.bincount(y, minlength=len(labels))
        if not int(self.support.sum()) or not int(self.support[self.benign]):
            raise ValueError("Fitness requires trusted observations and benign support")
        self.reset()

    def reset(self):
        self.cache, self.requests, self.reports = {}, [], {}
        self.counters = dict(candidate_requests=0, cache_hits=0, actual_model_evaluations=0,
                             pre_inference_failures=0, invalid_candidate_requests=0)
        self.timings = dict(construction_seconds=0., inference_seconds=0.)

    def sync(self):
        if self.values["device"] == "cuda":
            torch.cuda.synchronize()

    def construct(self, weights):
        x = validate(weights, self.policy.cap)
        if len(x) != len(self.clients):
            raise ValueError("Candidate dimension mismatch")
        accum = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in self.parent.items()}
        for client, weight in zip(self.clients, x):
            for key in accum:
                accum[key].add_(self.deltas[client][key].double(), alpha=float(weight))
        state = {k: self.parent[k] + v.to(self.parent[k].dtype) for k, v in accum.items()}
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise NumericalCandidateError("Nonfinite candidate parameters")
        return state

    def predict(self, state):
        self.model.load_state_dict(state)
        self.model.eval()
        confusion = torch.zeros((len(self.labels), len(self.labels)), dtype=torch.int64)
        loss_sum = 0.
        with torch.inference_mode():
            for x, y in DataLoader(self.trusted, batch_size=self.values["batch_size"], shuffle=False, num_workers=0):
                logits = self.model(x.to(self.values["device"]))
                loss = torch.nn.functional.cross_entropy(logits, y.to(self.values["device"]), reduction="sum")
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise NumericalCandidateError("Nonfinite candidate predictions/loss")
                predictions = logits.argmax(1).cpu()
                confusion += torch.bincount(y * len(self.labels) + predictions,
                                            minlength=len(self.labels)**2).reshape(confusion.shape)
                loss_sum += float(loss)
        report = summarize(confusion, loss_sum)
        report.update(role="trusted", benign_fpr=float((self.support[self.benign] - confusion[self.benign, self.benign])
                                                     / self.support[self.benign]))
        return report

    def evaluate(self, weights):
        x = validate(weights, self.policy.cap)
        key = identity(self.clients, x, self.policy.cap)
        self.counters["candidate_requests"] += 1
        cached = key in self.cache
        if cached:
            self.counters["cache_hits"] += 1
            result = self.cache[key]
        else:
            records = tuple(ClientWeight(c, float(w)) for c, w in zip(self.clients, x))
            tick = time.perf_counter()
            try:
                state = self.construct(x)
            except NumericalCandidateError as exc:
                self.counters["pre_inference_failures"] += 1
                self.timings["construction_seconds"] += time.perf_counter() - tick
                result = CandidateEvaluation(key, records, None, False, str(exc))
            else:
                self.timings["construction_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                self.counters["actual_model_evaluations"] += 1
                try:
                    report = self.predict(state)
                except NumericalCandidateError as exc:
                    result = CandidateEvaluation(key, records, None, False, str(exc))
                else:
                    self.reports[key] = report
                    result = CandidateEvaluation(key, records, FitnessVector(
                        1 - report["macro_f1"], report["benign_fpr"],
                        min(1., math.fsum(float(w) * self.assessments[c].current_risk for c, w in zip(self.clients, x))),
                        min(1., math.fsum(float(w) * (1 - self.assessments[c].prior_reputation) for c, w in zip(self.clients, x)))),
                        True)
                finally:
                    self.sync()
                    self.timings["inference_seconds"] += time.perf_counter() - tick
            self.cache[key] = result
        self.counters["invalid_candidate_requests"] += int(not result.feasible)
        self.requests.append({"candidate": key, "cache_hit": cached, "feasible": result.feasible})
        return result

    def report(self):
        return {"counts": dict(self.counters), "timing": dict(self.timings), "requests": self.requests,
                "evaluations": [asdict(v) for v in self.cache.values()], "trusted_metrics": self.reports}
