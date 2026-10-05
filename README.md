# Edge-IIoTset FL Research

See [Phase B/C v2 stabilization](docs/phase_bc_stabilization.md) for current configuration and interface
changes. Client assignment now requires an explicit task; historical completions cannot authorize training.

Phase A provides the research contracts and reproducibility foundation.
Phase B implements source registration, streaming audits, provenance and capture
verification, isolated groups, both global split protocols, trusted panels, and
the final dataset gate. Phase C implements group-preserving client assignment,
client-local validation, frozen training-only preprocessing, and the pretraining gate.
Learning remains outside scope.

Read [Phase B architecture and commands](docs/phase_b.md) for execution and debugging.
Read [Phase C architecture and commands](docs/phase_c.md) for client-data execution.

## Start here

1. Read [Research protocol](docs/research_protocol.md) for hypotheses and claim boundaries.
2. Read [Architecture](docs/architecture.md) for folder ownership and stage boundaries.
3. Read [Interfaces](docs/interfaces.md) for data access and compatibility contracts.
4. Follow [Development workflow](docs/development.md) for setup, checks, and the next stage.
5. Review [Phase A acceptance](docs/phase_a_acceptance.md) for what is complete and deferred.
6. Review [Phase C acceptance](docs/phase_c_acceptance.md) for tests and the real-data block.
7. Read [Phase D readiness refinements](docs/phase_d_readiness_refinement_plan.md) for comparison ownership, clean-learning evidence, diagnostics, and profiling prerequisites.

The [final roadmap](Edge-IIoTset_FL_Final_Implementation_Roadmap.md) governs the full project.
It supersedes the older start/workflow documents where they conflict.

## Local commands (PowerShell, from this folder)

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.lock
    .\.venv\Scripts\python.exe scripts/edgefl.py validate-config
    .\.venv\Scripts\python.exe scripts/edgefl.py pipelines
    .\.venv\Scripts\python.exe scripts/check_phase_a.py
    .\.venv\Scripts\python.exe scripts/check_phase_c.py
    .\.venv\Scripts\python.exe scripts/edgefl.py init-run --seed 11

The project-local environment was created during Phase A setup. Recreating it is
necessary only on a fresh checkout or when rebuilding the environment.
An editable installation is optional; the script launcher works directly from src.
For other working directories, pass --workspace with the project folder before the subcommand.

## What a foundation run means

A new directory under runs contains the exact configuration, protocol fingerprint,
source snapshot, environment, seed namespaces, metadata, and event log.
Dataset hashes are explicitly pending Phase B; eligible_for_training is false.
These records are setup evidence, not experimental results or pretraining approval.

Archive files and the three existing planning documents are preserved.
Generated data and model files are excluded from version control; source, schemas,
protocols, tests, and the dependency lock should be versioned.
