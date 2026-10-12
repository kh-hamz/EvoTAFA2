"""Explicit prior snapshots and transactional temporal reputation."""

import copy
import math
from edgefl.contracts.records import ReputationTransition


class ReputationStore:
    def __init__(self, clients, initial=.5, retention=.9):
        if not 0 <= initial <= 1 or not 0 <= retention < 1:
            raise ValueError("Invalid reputation coefficients")
        self.retention = retention
        self.entries = {c: {"value": initial, "last_submitted": 0, "last_valid": 0} for c in sorted(clients)}

    def prior(self):
        return {c: entry["value"] for c, entry in self.entries.items()}

    def propose(self, current, risks, invalid, selected):
        if type(current) is not int or current < 1 or any(e["last_submitted"] >= current for e in self.entries.values()):
            raise ValueError("Reputation cannot use current or future outcomes")
        selected, invalid = set(selected), set(invalid)
        if set(risks) & invalid or set(risks) | invalid != selected or not selected <= set(self.entries):
            raise ValueError("Incomplete/conflicting reputation dispositions")
        state, transitions = self.state_dict(), []
        for client, entry in state.items():
            before = entry["value"]
            if client in risks:
                risk = risks[client]
                if not math.isfinite(risk) or not 0 <= risk <= 1:
                    raise ValueError("Invalid reputation risk")
                entry.update(value=self.retention * before + (1 - self.retention) * (1 - risk), last_submitted=current, last_valid=current)
                reason = "valid_observation"
            elif client in invalid:
                entry.update(value=self.retention * before, last_submitted=current)
                reason = "invalid_submission_penalty"
            else:
                reason = "ordinary_nonparticipation"
            transitions.append(ReputationTransition(client, before, entry["value"], reason))
        return state, tuple(transitions)

    def state_dict(self):
        return copy.deepcopy(self.entries)

    def load_state_dict(self, state):
        if set(state) != set(self.entries):
            raise ValueError("Reputation client coverage mismatch")
        for entry in state.values():
            if set(entry) != {"value", "last_submitted", "last_valid"} or not math.isfinite(entry["value"]) or not 0 <= entry["value"] <= 1:
                raise ValueError("Invalid reputation snapshot")
            if any(type(entry[k]) is not int or entry[k] < 0 for k in ("last_submitted", "last_valid")) or entry["last_valid"] > entry["last_submitted"]:
                raise ValueError("Invalid reputation round history")
        self.entries = copy.deepcopy(state)
