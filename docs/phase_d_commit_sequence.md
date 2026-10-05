# Phase D commit sequence

Create these commits in order. Each group contains whole files, so selective hunk staging
is unnecessary. Create all commits locally and push together or separately as desired.
No commits or pushes were made while preparing this guide.

## 1. Configuration and dependencies

Commit: `build: add Phase D dependencies and configuration contracts`

```text
pyproject.toml
requirements-phase-d.lock
configs/phase_d.json
src/edgefl/schemas/phase_d.schema.json
src/edgefl/contracts/phase_d.py
src/edgefl/learning/__init__.py
src/edgefl/learning/config.py
```

## 2. Verified data and artifacts

Commit: `feat: add verified Phase D data preparation and artifact lineage`

```text
src/edgefl/learning/storage.py
src/edgefl/learning/data.py
```

## 3. Baselines and learning engine

Commit: `feat: implement Phase D baselines, diagnostics, resume and review`

```text
src/edgefl/learning/models.py
src/edgefl/learning/checkpoints.py
src/edgefl/learning/metrics.py
src/edgefl/learning/training.py
src/edgefl/learning/aggregation.py
src/edgefl/learning/engine.py
src/edgefl/learning/review.py
tests/test_phase_d.py
```

## 4. Pipeline integration and regression coverage

Commit: `feat: expose Phase D pipelines with acceptance and integration tests`

```text
src/edgefl/pipelines/phase_d_prepare.py
src/edgefl/pipelines/phase_d_initialize.py
src/edgefl/pipelines/phase_d_centralized.py
src/edgefl/pipelines/phase_d_federated.py
src/edgefl/pipelines/phase_d_review.py
src/edgefl/pipelines/phase_d_validate.py
src/edgefl/pipelines/phase_d_cli.py
src/edgefl/pipelines/catalog.py
src/edgefl/cli.py
tests/test_phase_a.py
tests/test_phase_c.py
tests/test_phase_d_pipeline.py
scripts/check_phase_d.py
```

Keep CLI registration, catalogue changes and updated availability assertions together.
This prevents an intermediate commit from exposing missing commands or retaining obsolete
phase-availability expectations. The complete Phase D test runner becomes usable here.

## 5. Documentation and acceptance evidence

Commit: `docs: document Phase D usage, architecture and acceptance limits`

```text
Edge-IIoTset_FL_Final_Implementation_Roadmap.md
README.md
docs/architecture.md
docs/development.md
docs/interfaces.md
docs/phase_d.md
docs/phase_d_acceptance.md
docs/phase_d_implementation_plan.md
docs/phase_d_commit_sequence.md
```

The 107-test result covers the combined implementation. Intermediate commits were not
independently checked out and tested. Generated logs under reports/generated remain
ignored; the acceptance Markdown records their result and real-data limitation.

Do not force-add archive/, .venv/, data/, runs/, artifacts/ or reports/generated/.
The prior phase_d_readiness_refinement_plan.md has no current changes and needs no new
staging for this sequence.
