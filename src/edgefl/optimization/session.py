"""Phase E orchestration extended only at the aggregation and state boundaries."""
import copy
from edgefl.trust.session import TrustSession
from edgefl.learning.storage import checked
from edgefl.optimization.aggregation import NSGA2Aggregator
from edgefl.optimization.chromosomes import validate
from edgefl.optimization.diagnostics import environment

class EvolutionSession(TrustSession):
    def __init__(self, *args, search_settings, search_policy=None, **kwargs):
        from edgefl.optimization.policies import SearchPolicy
        self.search_policy = search_policy or SearchPolicy()
        super().__init__(*args, **kwargs)
        self.search_settings = search_settings
        self.previous_selected = None
        self.search_artifacts = {}
        self.search_environment = environment()

    def create_aggregator(self, assessments, root):
        self.aggregator = NSGA2Aggregator(self.config.workspace, self.directory, self.counts,
            self.scorer.trusted, self.trusted_ref, self.data.spec["labels"], self.values,
            self.search_settings, self.previous_selected, self.settings["weight_floor"], self.search_policy)
        return self.aggregator, assessments

    def state_dict(self):
        return {"schema_version": "phase-f.session.v1", "trust": super().state_dict(),
                "previous_selected": copy.deepcopy(self.previous_selected),
                "search_artifacts": copy.deepcopy(self.search_artifacts),
                "search_environment": self.search_environment}

    def load_state_dict(self, state):
        if set(state) != {"schema_version", "trust", "previous_selected", "search_artifacts", "search_environment"} or state["schema_version"] != "phase-f.session.v1":
            raise ValueError("Invalid Phase F session state")
        if state["search_environment"] != self.search_environment:
            raise ValueError("Search environment mismatch")
        previous = state["previous_selected"]
        if previous is not None:
            if set(previous) - set(self.counts):
                raise ValueError("Unknown previous client")
            validate(list(previous.values()), self.search_policy.cap)
        super().load_state_dict(state["trust"])
        if set(state["search_artifacts"]) != set(self.round_artifacts):
            raise ValueError("Search/assessment round coverage mismatch")
        for ref in state["search_artifacts"].values():
            checked(self.config.workspace, ref)
        self.previous_selected = copy.deepcopy(previous)
        self.search_artifacts = copy.deepcopy(state["search_artifacts"])

    def execute(self, current, parent_ref, parent):
        executed = super().execute(current, parent_ref, parent)
        if not executed.result.failure_reason:
            self.previous_selected = {w.client_id: w.weight for w in executed.result.weights}
            self.search_artifacts[str(current)] = vars(executed.result.diagnostics).copy()
            report = self.aggregator.last_report["search"]
            # Full candidate traces live in the indexed artifact, not every growing checkpoint history.
            executed.extra["search"] = {k: report[k] for k in
                ("counts", "timing", "timing_totals", "selected", "selected_objectives", "memory", "failure_reason")}
        return executed
