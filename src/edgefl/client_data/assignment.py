"""Deterministic, group-preserving client-allocation strategies."""

import hashlib
import math
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from edgefl.config import canonical_json
from edgefl.contracts.phase_c import MANIFEST_FIELDS, SupportPolicy, AllocationResult
from edgefl.client_data.policies import class_counts, metric_support
from edgefl.reproducibility import derive_seed
from edgefl.client_data.storage import rows, write_json, writer


@dataclass(frozen=True)
class GroupProfile:
    group_id: str
    size: int
    labels: tuple[tuple[str, int], ...]

    @property
    def counts(self) -> Counter:
        return Counter(dict(self.labels))


@dataclass
class ClientState:
    groups: list[GroupProfile] = field(default_factory=list)
    labels: Counter = field(default_factory=Counter)
    size: int = 0

    def add(self, group: GroupProfile) -> None:
        self.groups.append(group)
        self.labels.update(group.counts)
        self.size += group.size

    def remove(self, group: GroupProfile) -> None:
        self.groups.remove(group)
        self.labels.subtract(group.counts)
        self.size -= group.size


def _priority(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


class AllocationStrategy(ABC):
    """Strategy interface keeps scenario behavior separate from stage orchestration."""

    @abstractmethod
    def allocate(self, groups: list[GroupProfile], clients: tuple[str, ...],
                 seed: int) -> dict[str, str]:
        raise NotImplementedError

    @staticmethod
    def _ordered(groups: list[GroupProfile], seed: int) -> list[GroupProfile]:
        return sorted(groups, key=lambda group: (-group.size, _priority(seed, group.group_id)))

    def allocate_result(self, groups, clients, seed):
        ownership = self.allocate(groups, clients, seed)
        return AllocationResult(tuple(sorted(ownership.items())), tuple(
            (label, tuple(sorted(values.items()))) for label, values in sorted(self.targets.items())))


def _cost(state, target_size, targets):
    return ((state.size - target_size) / max(target_size, 1)) ** 2 + sum(
        ((state.labels[label] - target) / max(target, 1)) ** 2
        for label, target in targets.items()) / max(len(targets), 1)


class NearIIDStrategy(AllocationStrategy):
    def allocate(self, groups, clients, seed):
        ordered = self._ordered(groups, seed)
        states = {client: ClientState() for client in clients}
        totals = sum((group.counts for group in groups), Counter())
        target_size = sum(group.size for group in groups) / len(clients)
        target_labels = {label: count / len(clients) for label, count in totals.items()}
        self.targets = {label: {client: target for client in clients} for label, target in target_labels.items()}
        result = {}
        for group in ordered:
            def score(client):
                state = states[client]
                before = _cost(state, target_size, target_labels)
                state.add(group)
                after = _cost(state, target_size, target_labels)
                state.remove(group)
                return after - before, state.size, _priority(seed, client + ":" + group.group_id)
            chosen = min(clients, key=score)
            states[chosen].add(group)
            result[group.group_id] = chosen
        return result


class DirichletStrategy(AllocationStrategy):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def allocate(self, groups, clients, seed):
        rng = random.Random(seed)
        labels = sorted({label for group in groups for label in group.counts})
        totals = sum((group.counts for group in groups), Counter())
        targets = {}
        for label in labels:
            draws = [rng.gammavariate(self.alpha, 1.0) for _ in clients]
            normalizer = sum(draws)
            targets[label] = {client: totals[label] * draw / normalizer
                              for client, draw in zip(clients, draws)}
        states = {client: ClientState() for client in clients}
        self.targets = targets
        result = {}
        for index, group in enumerate(self._ordered(groups, seed)):
            if index < len(clients):
                chosen = clients[index]
            else:
                def score(client):
                    state = states[client]
                    deficit = sum(group.counts[label] *
                                  (targets[label][client] - state.labels[label])
                                  / max(targets[label][client], 1)
                                  for label in group.counts)
                    return -deficit, state.size, client
                chosen = min(clients, key=score)
            states[chosen].add(group)
            result[group.group_id] = chosen
        return result


class SpecialistStrategy(AllocationStrategy):
    def __init__(self, specialists_per_attack: int):
        self.specialists_per_attack = specialists_per_attack

    def allocate(self, groups, clients, seed):
        attacks = sorted({label for group in groups for label in group.counts if label != "Normal"})
        specialists = {}
        for label in attacks:
            specialists[label] = set(sorted(clients, key=lambda c: _priority(seed, label + ":" + c))
                                     [:self.specialists_per_attack])
        states = {client: ClientState() for client in clients}
        target_size = sum(group.size for group in groups) / len(clients)
        self.targets = {label: {client: float(client in selected) for client in clients}
                        for label, selected in specialists.items()}
        result = {}
        for index, group in enumerate(self._ordered(groups, seed)):
            if index < len(clients):
                chosen = clients[index]
            else:
                def score(client):
                    matched = sum(count for label, count in group.counts.items()
                                  if label == "Normal" or client in specialists.get(label, set()))
                    preference = -matched / group.size
                    balance = (states[client].size + group.size) / max(target_size, 1)
                    return preference, balance, client
                chosen = min(clients, key=score)
            states[chosen].add(group)
            result[group.group_id] = chosen
        return result


def strategy(specification: dict) -> AllocationStrategy:
    kind = specification["kind"]
    if kind == "near_iid":
        return NearIIDStrategy()
    if kind == "dirichlet":
        return DirichletStrategy(specification["alpha"])
    if kind == "specialist":
        return SpecialistStrategy(specification["specialists_per_attack"])
    raise ValueError(f"Unknown client scenario kind: {kind}")


def _states(groups: list[GroupProfile], ownership: dict[str, str],
            clients: tuple[str, ...]) -> dict[str, ClientState]:
    result = {client: ClientState() for client in clients}
    for group in groups:
        result[ownership[group.group_id]].add(group)
    return result


def _problems(states: dict[str, ClientState], minimum: int, minimum_groups: int,
              require_binary: bool, policy=None) -> list[dict]:
    policy = policy or SupportPolicy(assigned=(int(require_binary), int(require_binary)))
    problems = []
    for client, state in sorted(states.items()):
        if state.size < minimum:
            problems.append({"client": client, "constraint": "observations", "actual": state.size, "minimum": minimum})
        if len(state.groups) < minimum_groups:
            problems.append({"client": client, "constraint": "groups", "actual": len(state.groups), "minimum": minimum_groups})
        problems.extend({"client": client, **item} for item in policy.violations(state.labels, "assigned"))
    return problems


def _repair(groups: list[GroupProfile], ownership: dict[str, str], clients: tuple[str, ...],
            minimum: int, minimum_groups: int, require_binary: bool,
            seed: int, policy=None, targets=None, kind="near_iid") -> list[dict]:
    repairs = []
    for _ in range(len(groups) * 2):
        states = _states(groups, ownership, clients)
        problems = _problems(states, minimum, minimum_groups, require_binary, policy)
        if not problems:
            return repairs
        recipient = problems[0]["client"]
        message = problems[0]
        need = message["constraint"]
        before_deficit = sum(max(0, p["minimum"] - p["actual"]) / max(p["minimum"], 1) for p in problems)
        candidates = []
        for donor, state in states.items():
            if donor == recipient:
                continue
            for group in list(state.groups):
                matches = (need in ("observations", "groups") or (need == "benign" and group.counts.get("Normal"))
                           or (need == "attack" and
                               sum(n for label, n in group.counts.items() if label != "Normal")))
                if not matches or state.size - group.size < minimum or len(state.groups) - 1 < minimum_groups:
                    continue
                before_cost = _scenario_cost(states, targets, kind)
                state.remove(group)
                states[recipient].add(group)
                donor_problems = _problems({donor: state}, minimum, minimum_groups, require_binary, policy)
                after_problems = _problems(states, minimum, minimum_groups, require_binary, policy)
                deficit = sum(max(0, p["minimum"] - p["actual"]) / max(p["minimum"], 1) for p in after_problems)
                change = _scenario_cost(states, targets, kind) - before_cost
                states[recipient].remove(group)
                state.add(group)
                if not donor_problems and deficit < before_deficit:
                    candidates.append((deficit, change, _priority(seed, group.group_id), donor, group))
        if not candidates:
            return repairs
        _, change, _, donor, group = min(candidates)
        ownership[group.group_id] = recipient
        repairs.append({"group_id": group.group_id, "from": donor, "to": recipient,
                        "reason": message, "scenario_cost_change": change})
    return repairs


def _scenario_cost(states, targets, kind):
    if not targets:
        return 0.0
    mean = sum(state.size for state in states.values()) / len(states)
    if kind == "specialist":
        return sum(count for client, state in states.items() for label, count in state.labels.items()
                   if label != "Normal" and not targets.get(label, {}).get(client))
    return sum(_cost(state, mean, {label: values[client] for label, values in targets.items()})
               for client, state in states.items())


def _improve(groups, ownership, clients, minimum, minimum_groups, require_binary, policy, targets, seed):
    states = _states(groups, ownership, clients)
    selected = sorted(groups, key=lambda g: _priority(seed, g.group_id))[:128]
    changes = []
    for _ in range(min(100, len(groups) * 2)):
        baseline = _scenario_cost(states, targets, "near_iid")
        best = None
        for index, first in enumerate(selected):
            donor = ownership[first.group_id]
            for second in [None, *selected[index + 1:]]:
                recipients = clients if second is None else (ownership[second.group_id],)
                for recipient in recipients:
                    if donor == recipient:
                        continue
                    states[donor].remove(first); states[recipient].add(first)
                    if second:
                        states[recipient].remove(second); states[donor].add(second)
                    cost = _scenario_cost(states, targets, "near_iid")
                    valid = not _problems({c: states[c] for c in (donor, recipient)}, minimum,
                                          minimum_groups, require_binary, policy)
                    if second:
                        states[donor].remove(second); states[recipient].add(second)
                    states[recipient].remove(first); states[donor].add(first)
                    if valid and cost < baseline - 1e-12:
                        rank = (cost, first.group_id, second.group_id if second else "", recipient)
                        if best is None or rank < best[0]:
                            best = (rank, first, second, donor, recipient)
        if best is None:
            break
        _, first, second, donor, recipient = best
        states[donor].remove(first); states[recipient].add(first)
        ownership[first.group_id] = recipient
        if second:
            states[recipient].remove(second); states[donor].add(second)
            ownership[second.group_id] = donor
        changes.append({"group_id": first.group_id, "swap_group_id": second.group_id if second else None,
                        "from": donor, "to": recipient, "reason": "fidelity_improvement"})
    return changes


def _heterogeneity(states: dict[str, ClientState]) -> dict:
    totals = sum((state.labels for state in states.values()), Counter())
    labels = sorted(totals)
    global_p = {label: totals[label] / max(sum(totals.values()), 1) for label in labels}
    distributions = {}
    distances = []
    client_distances = {}
    for client, state in sorted(states.items()):
        p = {label: state.labels[label] / max(state.size, 1) for label in labels}
        distributions[client] = p
        distance = sum(abs(p[label] - global_p[label]) for label in labels) / 2
        distances.append(distance)
        client_distances[client] = distance
    sizes = [state.size for state in states.values()]
    mean = sum(sizes) / len(sizes)
    deviations = {client: abs(state.size - mean) / mean if mean else 0 for client, state in states.items()}
    variance = sum((size - mean) ** 2 for size in sizes) / len(sizes)
    entropy = 0.0
    for value in global_p.values():
        if value:
            entropy -= value * math.log(value, 2)
    return {"global_class_proportions": global_p, "client_class_proportions": distributions,
            "binary_class_proportions": {client: {"benign": values.get("Normal", 0),
                "attack": sum(n for label, n in values.items() if label != "Normal")}
                for client, values in distributions.items()},
            "mean_client_total_variation": sum(distances) / len(distances),
            "client_total_variation": client_distances, "maximum_total_variation": max(distances),
            "client_size_deviation": deviations, "maximum_size_deviation": max(deviations.values()),
            "reference": "selected_experiment_client_pool_original_labels", "target_size": mean,
            "size_coefficient_of_variation": math.sqrt(variance) / mean if mean else 0,
            "global_label_entropy_bits": entropy}


def collect_groups(splits: Path, fold: str) -> list[GroupProfile]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    sizes = Counter()
    for row in rows(splits):
        if row["fold"] == fold and row["partition"] == "client_pool":
            grouped[row["group_id"]][row["label"]] += 1
            sizes[row["group_id"]] += 1
    return [GroupProfile(gid, sizes[gid], tuple(sorted(labels.items())))
            for gid, labels in sorted(grouped.items())]


def assign(splits: Path, fold: str, scenario_name: str, specification: dict,
           directory: Path, seed: int, client_count: int, attempts: int,
           minimum: int, minimum_groups: int, require_binary: bool, *, policy=None, limits=None) -> dict:
    policy = policy or SupportPolicy(assigned=(int(require_binary), int(require_binary)))
    limits = limits or {"maximum_size_deviation": 0.1, "maximum_total_variation": 0.05}
    groups = collect_groups(splits, fold)
    if not groups:
        raise ValueError("Selected fold has no client-pool groups")
    clients = tuple(f"client-{index:02d}" for index in range(client_count))
    if sum(group.size for group in groups) < client_count * minimum:
        write_json(directory / "allocation_failure.json", {"outcome": "necessary_condition_failed",
                   "reason": "observation_total", "eligible_for_training": False})
        raise ValueError("Client pool cannot meet the configured observation minimum")
    if len(groups) < client_count * minimum_groups:
        write_json(directory / "allocation_failure.json", {"outcome": "necessary_condition_failed",
                   "reason": "group_total", "eligible_for_training": False})
        raise ValueError("Client pool cannot meet the configured independent-group minimum")
    allocation_strategy = strategy(specification)
    history, accepted, accepted_seed, repairs, best = [], None, None, [], None
    for attempt in range(attempts):
        attempt_seed = derive_seed(seed, "partition", client=f"{fold}:{scenario_name}",
                                   round_id=attempt)
        result = allocation_strategy.allocate_result(groups, clients, attempt_seed)
        ownership = dict(result.ownership)
        targets = {label: dict(values) for label, values in result.targets}
        before = _heterogeneity(_states(groups, ownership, clients))
        changes = _repair(groups, ownership, clients, minimum, minimum_groups,
                          require_binary, attempt_seed, policy, targets, specification["kind"])
        preliminary = _heterogeneity(_states(groups, ownership, clients))
        if specification["kind"] == "near_iid" and (
                _problems(_states(groups, ownership, clients), minimum, minimum_groups, require_binary, policy)
                or any(preliminary[key] > value + 1e-12 for key, value in limits.items())):
            changes += _improve(groups, ownership, clients, minimum, minimum_groups,
                                require_binary, policy, targets, attempt_seed)
        states = _states(groups, ownership, clients)
        problems = _problems(states, minimum, minimum_groups, require_binary, policy)
        achieved = _heterogeneity(states)
        if specification["kind"] == "near_iid":
            problems += [{"constraint": key, "actual": achieved[key], "maximum": value}
                         for key, value in limits.items() if achieved[key] > value + 1e-12]
        rank = (len(problems), achieved["maximum_total_variation"], achieved["maximum_size_deviation"])
        if best is None or rank < best[0]:
            best = (rank, dict(ownership), achieved, problems)
        history.append({"attempt": attempt, "seed": attempt_seed, "repairs": changes,
                        "rejected": bool(problems), "reasons": problems, "targets": targets,
                        "before_repairs": before, "after_repairs": achieved})
        if not problems:
            accepted, accepted_seed, repairs = ownership, attempt_seed, changes
            break
    write_json(directory / "partition_attempts.json", history)
    if accepted is None:
        write_json(directory / "allocation_failure.json", {"outcome": "search_exhausted",
                   "attempts": attempts, "observed_deviations": best[2], "violations": best[3],
                   "does_not_prove_infeasibility": True, "eligible_for_training": False})
        write_json(directory / "best_candidate.json", {"label": "approximate_balance_diagnostic",
                   "ownership": best[1], "heterogeneity": best[2], "eligible_for_training": False})
        raise ValueError("Client allocation search exhausted; no valid allocation found, infeasibility not proven")
    states = _states(groups, accepted, clients)
    stream, output = writer(directory / "assignments.csv", MANIFEST_FIELDS["assignments"])
    observation_counts = Counter()
    with stream:
        for row in rows(splits):
            if row["fold"] != fold or row["partition"] != "client_pool":
                continue
            client = accepted[row["group_id"]]
            output.writerow({"fold": fold, "scenario": scenario_name, "client_id": client,
                             "observation_id": row["observation_id"], "label": row["label"],
                             "capture_id": row["capture_id"], "session_id": row["session_id"],
                             "group_id": row["group_id"]})
            observation_counts[client] += 1
    clients_report = {}
    for client, state in sorted(states.items()):
        clients_report[client] = {"observations": state.size, "groups": len(state.groups),
                                  "class_counts": class_counts(state.labels),
                                  "class_group_counts": {label: sum(bool(g.counts[label]) for g in state.groups)
                                                         for label in class_counts(state.labels)},
                                  "metric_support": metric_support(state.labels),
                                  "binary_support": {"benign": state.labels["Normal"],
                                      "attack": sum(n for label, n in state.labels.items()
                                                    if label != "Normal")}}
    write_json(directory / "clients.json", clients_report)
    return {"fold": fold, "scenario": scenario_name, "scenario_specification": specification,
            "accepted_attempt": len(history) - 1, "partition_seed": accepted_seed,
            "attempts": history, "repairs": repairs, "clients": clients_report,
            "heterogeneity": _heterogeneity(states), "groups": len(groups),
            "outcome": "accepted", "limits": limits if specification["kind"] == "near_iid" else None,
            "largest_group": max(group.size for group in groups),
            "observations": sum(observation_counts.values()),
            "assignment_fingerprint": hashlib.sha256(canonical_json(sorted(accepted.items())).encode()).hexdigest(),
            "eligible_for_training": False}
