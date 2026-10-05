# Development workflow

Phases B and C are implemented in separate domain and orchestration modules. See
[Phase B architecture, contracts, and commands](phase_b.md) and
[Phase C architecture and commands](phase_c.md). The foundation specification below
records the Phase A baseline; runnable stages now extend through the pretraining gate,
subject to upstream acceptance.

## Supported foundation environment

Python 3.11+ is declared; Phase A was executed with Python 3.12.6 on Windows.
The only direct runtime dependency is jsonschema 4.23.0. requirements.lock records
its installed dependency versions. Scientific/training packages are deliberately
not installed until their phases need them. No global Python packages are modified.

From the project root in PowerShell:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.lock
    .\.venv\Scripts\python.exe scripts/check_phase_a.py

Use an existing .venv when it is already prepared. The script launcher avoids
requiring build tools. Optional packaging with pip install -e . uses the setuptools
build requirement in pyproject.toml; that installation route has not been tested here.

## Phase A commands

    .\.venv\Scripts\python.exe scripts/edgefl.py validate-config
    .\.venv\Scripts\python.exe scripts/edgefl.py pipelines
    .\.venv\Scripts\python.exe scripts/edgefl.py init-run --seed 11

validate-config reads the configuration and protocol, validates the local JSON schema,
checks ratios and path boundaries, and writes nothing. It does not audit CSVs.
pipelines lists stage metadata and availability; planned entries cannot execute.
init-run creates only a foundation record. It does not produce datasets or models.

To use a different explicit configuration:

    .\.venv\Scripts\python.exe scripts/edgefl.py validate-config --config configs/phase_a.json

The workspace may be specified before the subcommand. Configuration paths remain
workspace-relative. There are no environment-variable overrides, implicit merges,
or last-used notebook settings. Research defaults in the JSON are recorded parameters,
not implementations. Unknown properties, unsupported scope/schema, duplicate keys,
non-finite values, invalid seeds, and unsafe paths fail clearly with exit code 2.

## Foundation outputs

Each runs/foundation-<UTC>-<unique-id>/ directory contains:

- run.json: identity, pending dataset registration, source/config/protocol fingerprints,
  Git availability, and links to metadata artifacts.
- config.json: exact resolved configuration snapshot.
- environment.json: Python/platform and installed package versions.
- seeds.json: separate random streams and global split base seed.
- source_manifest.json: hashes of source/configuration/protocol inputs.
- events.jsonl: structured foundation event.

run.json is the completion marker and is written last. A directory without it is
incomplete. Do not resume by overwriting it. Inspect the failure and create a new
foundation record. Later training-resume semantics are reserved for the round engine.

Dataset hashes are null, dataset registration is pending_phase_b, pretraining_gate
is not_run, and eligible_for_training is false. These values must not be manually
changed to skip future gates. Phase B will introduce verified source manifests; later
experiment entrypoints must require them rather than accepting this foundation record.

## Randomness contract

The global split uses an independent base seed (42) and remains fixed across matched
training seeds. Each matched seed derives partition, initialization, minibatch,
participation, poisoning, and evolution streams using a versioned SHA-256 identity.
Pass client and round coordinates when deriving local streams. Never use Python's
process-randomized hash(), a shared mutable global RNG, or method name in seed identity
for comparisons intended to be matched. Adapter libraries receive the derived unsigned
32-bit values; future checkpointing must also save generator states where needed.

Seed separation improves reproducibility; it is not a guarantee of bitwise equality
across different hardware, kernels, library versions, or training schedules. Record
those differences and define numerical tolerances during model implementation.

## Debugging one future pipeline

1. Select the next authorized roadmap step and document its input/output contracts.
2. Add its own module under pipelines and a small domain module where needed.
3. Accept upstream manifest references, not raw implicit directory scans.
4. Test with tiny fixtures, including malformed/missing/unsupported data.
5. Add a stage-specific command only after its implementation exists.
6. Log input/config/schema hashes, stage identity and explicit failure reasons.
7. Run the focused tests and the existing Phase A boundary tests.
8. Update pipeline availability and acceptance evidence only after the stage passes.

For future model pipelines, keep software tests, real-data acceptance, and clean-learning
review as separate evidence. Tiny synthetic tests may exercise models and contracts, but
cannot authorize real-data experiments. Use the [readiness specification](phase_d_readiness_refinement_plan.md)
for phase ownership and the [diagnostic definitions](interfaces.md#future-round-diagnostics)
when implementing the training engine. Add behavioral tests in their owning phase rather
than tests that merely match documentation wording.

Keep notebooks exploratory: move accepted transformations into tested source modules.
Avoid one large preprocess/train/evaluate function. Do not add fake success stubs for
planned pipelines. Fail loudly on unsupported stages instead of producing empty outputs.

## Next implementation boundary

Phase D is the next software boundary, but it cannot execute on the current real-data
artifacts: Phase B validation is FAIL, so no real Phase C pretraining PASS exists.
The evidence limitation must be resolved and Phases B-C rerun before real-data model
experiments. Capture-recovery approval remains a separate checkpoint. Synthetic component
tests may support development without changing that requirement.
Phase D remains planned and no training command is available.

Every future real-data experiment entrypoint, including resume, must call
load_training_ready from edgefl.client_data.storage with explicit dataset, task, fold,
and scenario expectations. Missing, stale, incompatible, or ineligible inputs block
execution; do not load a raw CSV as a fallback or manually override eligibility.

Phase D completion covers sanity models, FedAvg, FedProx, median, trimmed mean, and
FLTrust, with correctness and clean-learning evidence. Review centralized behavior,
then near-IID FedAvg, then non-IID FedAvg. Unresolved learning behavior blocks the affected
condition. An explained negative finding may proceed with documented limitations, but
never overrides failed software or data checks. E/F methods are not D prerequisites.

Measure training and evaluation costs in D. Profile one representative completed actual
candidate-evaluation/search round in F before scaling G, recording workload/environment,
candidate counts, component and total time, memory, warm-up, and projected matrix cost.
Keep current development defaults until explicit measured revisions are recorded before
main comparisons. Neither final-test results nor hidden runtime adjustments choose budgets.

The 2026-10-04 refinement changes documentation only. Existing contract versions,
configuration, tests, and historical acceptance results remain intact. Future source and
protocol snapshots capture the revised documents; do not rewrite historical snapshots.
