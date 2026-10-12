"""Sequential isolated workers with explicit recovery and no automatic numerical retries."""
from pathlib import Path
import json
import os
import subprocess
import sys
from edgefl.config import workspace_path
from edgefl.data.storage import read_json, write_json
from edgefl.experiments.storage import load_stage, artifact, reference, checked
from edgefl.experiments.ledger import Ledger, scheduler_lock


def preflight(config, matrix_path, bindings_path):
    compiled = load_stage(config, matrix_path, "compile-experiments")
    matrix = read_json(artifact(config, compiled, "cases.json"))
    bindings = read_json(bindings_path)
    if not isinstance(bindings, dict) or set(bindings) - set(matrix["cases"]):
        raise ValueError("Prepared-case registry contains unknown matrix cases")
    results, memo, matching = {}, {}, {}
    for key in matrix["cases"]:
        if key not in bindings:
            results[key] = {"status": "blocked", "reason": "No source-verified prepared case"}
            continue
        try:
            prepared = load_stage(config, workspace_path(config.workspace, bindings[key]), "prepare-experiment", memo=memo)
            if prepared["inputs"]["matrix"] != reference(config.workspace, matrix_path) or prepared["summary"]["case_id"] != key:
                raise ValueError("Prepared case belongs to a different matrix/case")
            results[key] = {"status": "ready" if prepared["summary"]["status"] == "READY" else "blocked",
                            "reason": ", ".join(prepared["summary"]["missing"]) or None,
                            "prepared": reference(config.workspace, workspace_path(config.workspace, bindings[key]))}
            matching[key] = prepared["summary"]["matching"]
        except (OSError, ValueError, KeyError) as exc:
            results[key] = {"status": "blocked", "reason": str(exc)}
    from edgefl.contracts.phase_g import digest
    groups = {}
    for key, fields in matrix["cases"].items():
        if key in matching:
            group = digest({k: v for k, v in fields.items() if k != "method"})
            groups.setdefault(group, []).append(key)
    for ids in groups.values():
        if any(matching[key] != matching[ids[0]] for key in ids):
            for key in ids:
                results[key].update(status="blocked", reason="Methods have unmatched data, initialization, participation, panel or attacks")
    return matrix, results


def verify_profile_runtime(config, profile):
    """A reviewed projection cannot authorize execution on a changed runtime."""
    from edgefl.optimization.diagnostics import environment
    from edgefl.experiments.sessions import runtime_context
    from edgefl.learning.models import configure
    for name, ref in profile["inputs"].items():
        if not name.startswith("measurement_") or name == "measurement_manifest":
            continue
        measurement = load_stage(config, checked(config.workspace, ref), "measure-experiment")
        run = load_stage(config, checked(config.workspace, measurement["inputs"]["run"]), "run-experiment")
        prepared = load_stage(config, checked(config.workspace, run["inputs"]["prepared"]), "prepare-experiment")
        base, _, _ = runtime_context(config, prepared)
        if (measurement["summary"]["training_runtime"] != configure(base["f"].learning.values)
                or measurement["summary"]["runtime"] != environment()):
            raise ValueError("Runtime changed since reviewed cost profiling; measure and review again")


def execute_queue(config, matrix_path, bindings_path, cost_review_path=None, suite=None, recover=False, resumes_path=None):
    matrix, readiness = preflight(config, matrix_path, bindings_path)
    selected = {r["case_id"] for r in matrix["memberships"] if suite is None or r["suite"] == suite}
    if not selected:
        raise ValueError("Empty or unknown experiment suite")
    if config.values["evidence_kind"] == "research":
        if cost_review_path is None:
            raise ValueError("Full-scale launch requires an explicit reviewed cost profile")
        from edgefl.experiments.review import accepted_subject
        profile = accepted_subject(config, cost_review_path, "profile-experiments")
        if profile["evidence_kind"] != "research" or profile["summary"]["matrix_id"] != matrix["matrix_id"]:
            raise ValueError("Cost review scope mismatch")
        if not selected <= set(profile["summary"]["covered_cases"]):
            raise ValueError("Cost review does not cover selected workload")
        if any(readiness[k].get("prepared") != profile["summary"]["prepared_cases"][k] for k in selected):
            raise ValueError("Prepared inputs changed since the reviewed cost projection")
        verify_profile_runtime(config, profile)
    resumes = read_json(resumes_path) if resumes_path else {}
    if not isinstance(resumes, dict) or set(resumes) - selected:
        raise ValueError("Resume registry contains unselected cases")
    base = workspace_path(config.workspace, config.values["output"]) / matrix["matrix_id"]
    worker_logs = base / "worker_logs"
    worker_logs.mkdir(parents=True, exist_ok=True)
    with scheduler_lock(base / "scheduler.lock", recover) as bind, Ledger(base / "ledger.sqlite", matrix["matrix_id"]) as ledger:
        ledger.register(matrix["cases"])
        previous = ledger.rows()
        for key in sorted(selected):
            prior = previous[key]
            if prior["status"] in ("complete", "reviewed"):
                done = load_stage(config, workspace_path(config.workspace, prior["completion"]), "run-experiment")
                if done["summary"]["case_id"] != key or done["summary"]["correctness"] != "PASS":
                    raise ValueError("Previously completed ledger entry has invalid evidence")
                continue
            if prior["status"] in ("failed", "unsuccessful", "interrupted", "running") and key not in resumes:
                ledger.transition(key, "interrupted" if prior["status"] == "running" else prior["status"], reason="Explicit resume required; no automatic retry")
                continue
            if readiness[key]["status"] != "ready":
                ledger.transition(key, "blocked", reason=readiness[key]["reason"])
                continue
            prepared = readiness[key]["prepared"]["path"]
            command = [sys.executable, str(Path(__file__).resolve().parents[3] / "scripts/edgefl.py"),
                       "--workspace", str(config.workspace), "run-experiment", "--config",
                       config.configuration.source.relative_to(config.workspace).as_posix(), "--prepared", prepared]
            if key in resumes:
                command += ["--resume", workspace_path(config.workspace, resumes[key]).relative_to(config.workspace).as_posix()]
            import uuid
            log = worker_logs / (key + "-" + uuid.uuid4().hex + ".log")
            ledger.transition(key, "running", log=log.relative_to(config.workspace).as_posix())
            with log.open("w", encoding="utf-8") as stream:
                worker = subprocess.Popen(command, cwd=config.workspace, stdout=stream, stderr=subprocess.STDOUT,
                                          shell=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                bind(worker)
                try:
                    code = worker.wait()
                except BaseException:
                    worker.terminate()
                    worker.wait()
                    ledger.transition(key, "interrupted", reason="Scheduler interrupted; use last committed round")
                    raise
                finally:
                    bind(None)
            try:
                response = json.loads(log.read_text(encoding="utf-8"))
                if "completion" not in response:
                    raise ValueError(response.get("error", "Worker published no completion"))
                completion = workspace_path(config.workspace, response["completion"])
                record = load_stage(config, completion, "run-experiment")
                if record["summary"]["case_id"] != key:
                    raise ValueError("Worker returned another case")
                state = "complete" if code == 0 and record["summary"]["correctness"] == "PASS" else "failed"
                if record["summary"].get("attack_coverage") != "PASS":
                    state = "unsuccessful"
                ledger.transition(key, state, completion=completion.relative_to(config.workspace).as_posix(), reason=record["summary"].get("failure"))
            except (OSError, ValueError, KeyError) as exc:
                ledger.transition(key, "failed", reason=str(exc))
        result = ledger.rows()
    report = {"matrix_id": matrix["matrix_id"], "selected": sorted(selected), "cases": result,
              "status": "COMPLETE" if all(result[k]["status"] in ("complete", "reviewed") for k in selected) else "INCOMPLETE"}
    write_json(base / "queue_status.json", report)
    return report
