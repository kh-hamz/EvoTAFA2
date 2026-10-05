# Project architecture

Active Phase B/C v2 contracts are described in [the stabilization guide](phase_bc_stabilization.md).
Source verification belongs to the data domain. Future training consumes a verified TrainingReadyBundle.

Phases B and C use separate domain and orchestration modules. See
[Phase B architecture, contracts, and commands](phase_b.md) and
[Phase C architecture and commands](phase_c.md). The foundation specification below
records the Phase A baseline; its data boundaries are implemented through roadmap
step 13. Phases D-H remain planned.

## Principles

Use a small src-layout Python package. Keep CLI parsing, configuration, contracts,
metadata, and future algorithms separate. Domain modules must not import the CLI.
Importing a module must not read datasets, create files, seed global randomness,
start workers, or train a model. Prefer explicit inputs and dependency injection.

## Initial filesystem layout

    archive/                          Existing immutable downloads (not versioned)
    configs/phase_a.json               Explicit foundation and research defaults
    docs/
      research_protocol.md            Questions, endpoints, claims, freeze policy
      architecture.md                 Ownership and dependency rules
      interfaces.md                   Contracts and access matrix
      development.md                  Commands and future stage workflow
      phase_a_acceptance.md           Scope and completion evidence
    src/edgefl/
      cli.py                          Only implemented command routing
      config.py                       Schema validation and path checks
      reproducibility.py              Deterministic seed namespaces
      runs.py                         Foundation records and source snapshots
      schemas/config.schema.json      Authoritative configuration shape
      contracts/records.py            Versioned immutable records
      contracts/interfaces.py         Trainer/scorer/aggregator Protocols
      pipelines/catalog.py            Stage metadata; no future executors
    scripts/edgefl.py                  Source-tree CLI launcher
    scripts/check_phase_a.py           unittest entrypoint
    tests/                            Foundation tests with temporary fixtures
    data/manifests/                    Future identities/splits, never raw copies
    data/clients/                      Future client manifests
    data/processed/                    Future materialized arrays
    artifacts/preprocessing/           Future fitted transformers
    artifacts/labels/                  Future label maps
    artifacts/models/                 Future model state
    runs/                             Unique per-run metadata and logs
    reports/generated/                Generated verification reports
    requirements.lock                 Installed runtime dependency versions
    pyproject.toml                    Package metadata and conventions

Existing roadmap/workflow documents remain at the root. No archive contents are
moved or rewritten. .gitignore excludes datasets, environments and generated artifacts.
There was no workspace Git repository at setup; records identify this honestly and
include a source-content fingerprint. Establish version control before experiment work.

## Future pipeline boundaries

Each future stage gets its own module and targeted tests only when that phase is
implemented. The catalogue declares dependencies and outputs; it is not an execution
engine and cannot run a planned stage.

| Stage | Owns | Does not own |
|---|---|---|
| audit | Streaming source/schema/label audit | Repair, splitting, fitting |
| provenance | Observation identity, provenance, sessions, duplicate reports | Models and client sampling |
| global_split | Protocol-specific global manifests and trusted panel definition | Learned preprocessing |
| clients | Client ownership and local validation | Global holdout reassignment |
| preprocessing | Training-only fitted transformation and feature contract | Global/client split generation |
| pretraining_gate | Cross-artifact validation and failure report | Silent repair |
| baselines | Common initialization, Phase D clean methods including FLTrust, training diagnostics and learning review | E/F assessment/search implementations, poisoning-resistance claims, final-test selection |
| attack_assessment | Controlled attack harness plus separate defense scorer | Leaking simulator truth to scorer |
| optimization | Fitness, evolutionary search, selected aggregation | Training client models internally |
| experiments | Matched scenario orchestration, ablations, external replication | Hidden configuration changes |
| final_evaluation | Frozen checkpoint/test evaluation and statistics | Tuning |
| deployment | Frozen protocol transport demonstration | Redefining research settings |

Although attack_assessment is one roadmap milestone, its future attack harness and
scorer must be separate modules with separate inputs. Future rounds similarly
compose trainer, scorer, optimizer, and aggregator instead of embedding them in a
single FL-loop file. Final evaluation remains a separate entrypoint.

The [readiness specification](phase_d_readiness_refinement_plan.md) assigns fixed
assessment-based comparisons to E and NSGA-II to F. Preserve the existing catalogue
order: pretraining_gate -> baselines -> attack_assessment -> optimization -> experiments.
Baseline execution must not require later scorers or fake assessments. FedProx changes
local optimization; coordinate-wise aggregation need not expose scalar client weights.

FLTrust is a D comparator with a separate server-reference operation using the permitted
trusted resource. This operation is composed with the round engine; it does not expand
ordinary client-trainer access. Record its procedure, access budget, normalization and
server cost. Method-specific requirements belong at their implementation boundaries.

## Dependencies and artifacts

Allowed direction: entrypoint -> orchestration -> domain modules -> contracts.
Configuration and reproducibility helpers are leaf services. No domain module may
import another stage's command-line handler or access its private files by convention.
Pass verified artifact references rather than directory globbing or notebook variables.

Large arrays/models live in artifacts, not JSON records. JSON metadata references
content hashes and schema versions. Future loaders validate hashes, roles, compatibility,
and stage gates before returning data. JSON itself is metadata, not a tensor format.

Real-data experiment entrypoints and resumed runs call the existing load_training_ready
with explicit dataset/task/fold/scenario expectations. Consume its verified bundle and
fail on absent, stale, incompatible or ineligible artifacts. Do not add a generic workflow
engine, new foundation record fields, or another data-loading bypass for this refinement.

## Debugging and failure isolation

Every stage must be independently invokable with explicit upstream manifests.
A completed upstream artifact is reusable only when its input/config/schema hashes
match. Future stages must not silently resplit or refit after a downstream error.
A failure produces a stage-specific report and a nonzero exit; partial output does
not become a successful artifact. Avoid a run-all command until individual stages pass.

Future structured logs include run ID, stage, event, round/client identity when
applicable, elapsed time and artifact references. Never log raw addresses or payloads.
Phase A writes only foundation events. Future stages add their own event files.

Phase D establishes the [common diagnostic meanings](interfaces.md#future-round-diagnostics)
through the existing RoundResult.diagnostics reference. Trainers own local measurements,
scorers and aggregators own their stage measurements, and orchestration combines their
references with selection-evaluation results. Keep nested timing explicit, failures and
unsupported metrics visible, and final-test data outside development paths.

Keep software correctness, real-data acceptance, and clean-learning review as separate
evidence. An unexplained learning failure blocks advancement of that condition; a
documented negative finding can proceed with limitations after investigation, but cannot
override a failed software or data gate. Synthetic tests are development evidence only.

Training/evaluation profiling belongs to D. Actual candidate-evaluation and completed
search profiling belongs to F and is required before the full G matrix. No Phase F
profiler is required to complete D, and profiling must not modify production checkpoints
or silently select a new search budget.

## Reproducibility

Each init-run creates a unique directory. run.json is written last and links the
configuration, environment, seeds and source manifest with hashes. A directory without
run.json is incomplete; investigate it rather than reusing it. Repeating init-run
preserves prior runs. No existing run directory is overwritten.

Dataset registration is explicitly pending Phase B. Source snapshots hash code,
configuration and protocol inputs only, not multi-gigabyte datasets. Future experiment
runs must require verified dataset/split/preprocessing identities and a passing gate.
A foundation run is never a training authorization.
