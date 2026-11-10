"""Explicit individual decisions and immutable batch publication."""
from dataclasses import asdict
from edgefl.contracts.phase_d import LearningReview
from edgefl.data.storage import read_json, write_json
from edgefl.config import workspace_path
from edgefl.experiments.storage import stage, load_stage, checked, reference

REVIEWABLE = {"run-experiment", "calibrate-experiment", "fit-experiment-clipping", "profile-experiments"}


def review_subject(config, path, disposition, rationale, limitations=""):
    decision = LearningReview(disposition, rationale, limitations)
    subject = load_stage(config, path)
    if subject["stage"] not in REVIEWABLE:
        raise ValueError("Unsupported scientific review subject")
    summary = subject["summary"]
    if summary.get("correctness") != "PASS" or summary.get("attack_coverage", "PASS") != "PASS":
        raise ValueError("Review cannot override failed numerical, coverage or profiling evidence")
    with stage(config, "review-experiment", {"subject": path}, identity=subject["identity"], gate=subject["gate"],
               matrix_id=summary.get("matrix_id", "shared"), case_id=summary.get("case_id", "shared")) as attempt:
        return attempt.finish({"decision": asdict(decision), "subject_stage": subject["stage"]})


def accepted_subject(config, review_path, expected=None, memo=None):
    memo = {} if memo is None else memo
    review = load_stage(config, review_path, "review-experiment", memo=memo)
    decision = LearningReview(**review["summary"]["decision"])
    subject = load_stage(config, checked(config.workspace, review["inputs"]["subject"]), expected, memo=memo)
    if (decision.disposition == "unresolved" or subject["summary"].get("correctness") != "PASS"
            or subject["summary"].get("attack_coverage", "PASS") != "PASS"):
        raise ValueError("Unresolved or unsuccessful evidence blocks advancement")
    return subject


def review_batch(config, manifest_path):
    entries = read_json(manifest_path)
    if not isinstance(entries, list) or not entries:
        raise ValueError("Batch review requires explicit per-subject decisions")
    inputs, decisions, seen = {"manifest": manifest_path}, [], set()
    for index, item in enumerate(entries):
        if not {"subject", "disposition", "rationale"} <= set(item) or set(item) - {"subject", "disposition", "rationale", "limitations", "config"}:
            raise ValueError("Invalid batch review entry")
        path = workspace_path(config.workspace, item["subject"])
        if path in seen:
            raise ValueError("Duplicate review subject")
        seen.add(path)
        raw = read_json(path)
        args = (path, item["disposition"], item["rationale"], item.get("limitations", ""))
        version = raw.get("schema_version")
        if version == "phase-g.v1":
            reviewed = review_subject(config, *args)
        else:
            if "config" not in item:
                raise ValueError("Upstream batch review requires its exact phase configuration")
            source = workspace_path(config.workspace, item["config"])
            inputs[f"configuration_{index}"] = source
            if version == "phase-d.v1":
                from edgefl.learning.config import load
                from edgefl.pipelines.phaseD.phase_d_review import execute
            elif version == "phase-e.v1":
                from edgefl.trust.config import load
                from edgefl.pipelines.phaseE.phase_e_review import execute
            elif version == "phase-f.v1":
                from edgefl.optimization.config import load
                from edgefl.pipelines.phaseF.phase_f_review import execute
            else:
                raise ValueError("Unknown review subject version")
            reviewed = execute(load(source, config.workspace), *args)
        inputs[f"subject_{index}"] = path
        inputs[f"review_{index}"] = reviewed
        decisions.append({"subject": reference(config.workspace, path), "review": reference(config.workspace, reviewed),
                          "disposition": item["disposition"]})
    with stage(config, "review-experiments", inputs) as attempt:
        write_json(attempt.directory / "decisions.json", decisions)
        return attempt.finish({"subjects": decisions, "status": "UNRESOLVED" if any(d["disposition"] == "unresolved" for d in decisions) else "REVIEWED"})
