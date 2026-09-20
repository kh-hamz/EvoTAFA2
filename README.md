# Edge-IIoTset FL Research

Phase A (roadmap steps 1–3) establishes the research and software foundation.
The only executable operations are configuration validation, pipeline inspection,
and creation of foundation metadata. No dataset processing or learning is implemented.

## Start here

1. Read [Research protocol](docs/research_protocol.md) for hypotheses and claim boundaries.
2. Read [Architecture](docs/architecture.md) for folder ownership and stage boundaries.
3. Read [Interfaces](docs/interfaces.md) for data access and compatibility contracts.
4. Follow [Development workflow](docs/development.md) for setup, checks, and the next stage.
5. Review [Phase A acceptance](docs/phase_a_acceptance.md) for what is complete and deferred.

The [final roadmap](Edge-IIoTset_FL_Final_Implementation_Roadmap.md) governs the full project.
It supersedes the older start/workflow documents where they conflict.

## Local commands (PowerShell, from this folder)

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.lock
    .\.venv\Scripts\python.exe scripts/edgefl.py validate-config
    .\.venv\Scripts\python.exe scripts/edgefl.py pipelines
    .\.venv\Scripts\python.exe scripts/check_phase_a.py
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
