"""Source-verified synthetic G evidence; never authorizes research execution."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import test_phase_f_pipeline as f_fixture
from edgefl.data.storage import read_json, write_json
from edgefl.contracts.phase_g import ExperimentCase
from edgefl.optimization.storage import implementation_hash as f_hash, regression_hash as f_regression
from edgefl.optimization.diagnostics import environment
from edgefl.pipelines.phaseF import phase_f_profile, phase_f_review, phase_f_validate
from edgefl.pipelines.phaseG.phase_g_compile import execute as compile_experiments
from edgefl.experiments.config import load
from edgefl.experiments.preparation import prepare
from edgefl.experiments.sessions import execute, fit_clipping, calibrate
from edgefl.experiments.review import review_subject
from edgefl.experiments.storage import artifact, load_stage

ROOT = Path(__file__).resolve().parents[1]


def setup(owner):
    f = f_fixture.setup(owner, trusted_repeats=3)
    profile = phase_f_profile.execute(f.f, f.clean, 2, 1, 2)
    clean_review = phase_f_review.execute(f.f, f.clean, "acceptable", "Source-verified synthetic clean fixture.")
    profile_review = phase_f_review.execute(f.f, profile, "acceptable", "Actual-model synthetic profiling fixture.")
    report = f.root / "fixture-f-software.json"
    write_json(report, {"schema_version": "phase-f.tests.v1", "tests": 1, "successful": True,
        "failures": 0, "errors": 0, "skipped": 0, "implementation_sha256": f_hash(),
        "regression_sha256": f_regression(), "search_environment": environment(), "evidence_kind": "synthetic_validator_fixture"})
    acceptance = phase_f_validate.execute(f.f, f.prepared_f, clean_review, profile_review, report)
    base = ExperimentCase("binary", 11, "nsga2", scenario="near_iid", clients=5, population=12, generations=4)
    cases = {"clean": base, "random": replace(base, method="random"), "capped": replace(base, method="adaptive_capped"),
             "no_reputation": replace(base, intervention="no_reputation"), "poison": replace(base, condition="sign_flip"),
             "no_loss": replace(base, intervention="no_loss"), "clipping": replace(base, intervention="clipping"),
             "half_panel": replace(base, panel_fraction=.5)}
    from edgefl.experiments.matrix import normalized
    cases = {k: normalized(v) for k, v in cases.items()}
    values = read_json(ROOT / "configs/phase_g.json")
    values.update(output="out_g", evidence_kind="fixture", fixture_cases=[asdict(c) for c in cases.values()])
    path = f.root / "configs/phase_g.json"
    write_json(path, values)
    g = load(path, f.root)
    matrix = compile_experiments(g)
    binding = {"phase_f_config": "configs/phase_f.json", "phase_f_acceptance": acceptance.relative_to(f.root).as_posix()}
    f.g, f.g_matrix, f.g_cases, f.g_binding = g, matrix, cases, binding
    f.g_prepared, f.g_runs, f.g_reviews = {}, {}, {}
    return f


def bind(f, name, **extra):
    import uuid
    path = f.root / ("binding-" + uuid.uuid4().hex + ".json")
    write_json(path, {**f.g_binding, **{k: v.relative_to(f.root).as_posix() for k, v in extra.items()}})
    return prepare(f.g, f.g_matrix, f.g_cases[name].case_id, path)


def execute_all(f):
    for name in ("clean", "random", "capped", "no_reputation"):
        prepared = bind(f, name)
        run = execute(f.g, prepared)
        if read_json(run)["summary"]["correctness"] != "PASS":
            raise AssertionError(read_json(run)["summary"])
        review = review_subject(f.g, run, "acceptable", "Verified synthetic experiment fixture.")
        f.g_prepared[name], f.g_runs[name], f.g_reviews[name] = prepared, run, review
    prepared = bind(f, "poison", clean_review=f.g_reviews["clean"])
    run = execute(f.g, prepared)
    review = review_subject(f.g, run, "acceptable", "Synthetic controlled poisoning.")
    f.g_prepared["poison"], f.g_runs["poison"], f.g_reviews["poison"] = prepared, run, review
    for name in ("no_loss", "clipping", "half_panel"):
        prepared = bind(f, name)
        extras = {}
        if name == "clipping":
            threshold = fit_clipping(f.g, prepared)
            threshold_review = review_subject(f.g, threshold, "acceptable", "Clean-only threshold and independent audit.")
            extras["clipping_review"] = threshold_review
            prepared = bind(f, name, **extras)
        pilots = {seed: execute(f.g, prepared, pilot_seed=seed) for seed in (101, 102, 103)}
        calibrated = calibrate(f.g, prepared, [pilots[101], pilots[102]], pilots[103])
        extras["calibration_review"] = review_subject(f.g, calibrated, "acceptable", "Independent synthetic calibration audit.")
        prepared = bind(f, name, **extras)
        run = execute(f.g, prepared)
        review = review_subject(f.g, run, "acceptable", "Synthetic study policy integration.")
        f.g_prepared[name], f.g_runs[name], f.g_reviews[name] = prepared, run, review
