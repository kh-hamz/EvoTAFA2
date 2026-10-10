"""Existing Aggregator interface backed by trusted-model pymoo search."""
import time
from dataclasses import asdict
import torch
from edgefl.contracts.records import RoundResult
from edgefl.contracts.phase_f import SearchResult, SearchFailure
from edgefl.learning import checkpoints
from edgefl.data.storage import write_json
from edgefl.optimization.fitness import CandidateEvaluator
from edgefl.optimization.pymoo_adapter import search
from edgefl.optimization.diagnostics import correlations, environment, MemoryMonitor

class NSGA2Aggregator:
    def __init__(self, root, directory, counts, trusted, trusted_ref, labels, values, settings,
                 previous=None, floor=1e-6, policy=None):
        from edgefl.optimization.policies import SearchPolicy
        self.policy = policy or SearchPolicy()
        self.root, self.directory, self.counts = root, directory, counts
        self.trusted, self.trusted_ref, self.labels = trusted, trusted_ref, labels
        self.values, self.settings, self.previous, self.floor = values, settings, previous, floor
        self.last_report = None

    def evaluator(self, request):
        return CandidateEvaluator(self.root, request, self.trusted, self.trusted_ref, self.labels,
                                  self.values, self.counts, self.policy)

    def aggregate(self, request):
        started = time.perf_counter()
        runtime = environment()
        evaluator = self.evaluator(request)
        setup_seconds = time.perf_counter() - started
        if self.values["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        with MemoryMonitor() as memory:
            try:
                backend = search
                if self.policy.backend == "random":
                    from edgefl.optimization.random_search import search as backend
                result = backend(evaluator, self.settings, self.counts, self.previous, self.floor)
            except SearchFailure as exc:
                result = SearchResult(None, tuple(evaluator.cache.values()), (), (), {}, str(exc))
            evaluator.sync()
        search_seconds = time.perf_counter() - tick
        weights, checkpoint, selected_metrics = None, request.parent_model, None
        publication_start = time.perf_counter()
        if result.selected:
            weights = {w.client_id: w.weight for w in result.selected.weights}
            state = evaluator.construct([weights[c] for c in evaluator.clients])
            checkpoints.validate_state(state, evaluator.parent)
            path = self.directory / f"aggregate-{request.round_id:04d}.pt"
            checkpoints.save(path, {"model": state})
            checkpoint = checkpoints.artifact(self.root, path)
            selected_metrics = evaluator.reports[result.selected.evaluation_id]
        publication_seconds = time.perf_counter() - publication_start
        report = {"schema_version": "phase-f.aggregation.v1", "method": self.policy.backend, "weights": weights,
                  "normalization": {"sum": 1, "cap": self.policy.cap}, "policy": asdict(self.policy), "seconds": time.perf_counter() - started,
                  "request": asdict(request), "previous_weights": self.previous, "settings": self.settings,
                  "environment": runtime, "clients": list(evaluator.clients),
                  "workload": {"trusted_observations": len(self.trusted), "features": request.compatibility.input_features,
                               "classes": request.compatibility.output_classes, "clients": len(evaluator.clients)},
                  "search": {**evaluator.report(), "generations": result.generations, "initialization": result.initialization,
                             "front": result.front, "selected": result.selected.evaluation_id if result.selected else None,
                             "selected_objectives": asdict(result.selected.objectives) if result.selected else None,
                             "selected_metrics": selected_metrics, "ideal": [0] * len(self.policy.active_objectives),
                             "selection_rule": "equal_weight_euclidean_then_error_fpr_candidate_id",
                             "objective_correlations": correlations(result.candidates),
                             "failure_reason": result.failure_reason,
                             "timing_totals": {"setup": setup_seconds, "search": search_seconds,
                                 "evolution_and_accounting": max(0., search_seconds - sum(evaluator.timings.values())),
                                 "selected_checkpoint": publication_seconds},
                             "memory": {"sampled_peak_rss_bytes": memory.peak, "sample_interval_seconds": memory.interval,
                                 "cuda_peak_allocated": torch.cuda.max_memory_allocated() if self.values["device"] == "cuda" else None,
                                 "cuda_peak_reserved": torch.cuda.max_memory_reserved() if self.values["device"] == "cuda" else None}}}
        diagnostics = self.directory / f"aggregation-{request.round_id:04d}.json"
        write_json(diagnostics, report)
        self.last_report = report
        return RoundResult(request.round_id, checkpoint, result.selected.weights if result.selected else None,
                           checkpoints.artifact(self.root, diagnostics), (), result.failure_reason)
