# Development workflow

Phase B is implemented in separate domain and orchestration modules. See
[Phase B architecture, contracts, and commands](phase_b.md) for the current dataset
pipelines. The foundation specification below records the Phase A baseline;
its planned data boundaries are now implemented for roadmap steps 4-9.

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

Keep notebooks exploratory: move accepted transformations into tested source modules.
Avoid one large preprocess/train/evaluate function. Do not add fake success stubs for
planned pipelines. Fail loudly on unsupported stages instead of producing empty outputs.

## Next implementation boundary

The next step is Phase B source registration and streaming audit. It must separately
address semantic schema issues, provenance recovery, observation duplication, session
construction, and the two split protocols before any client or model work.
No Phase B work was performed by the Phase A commands or checks.
