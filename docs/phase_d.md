# Phase D learning pipelines

Scope: roadmap Steps 14-17. See the [implementation specification](phase_d_implementation_plan.md)
and [readiness refinement](phase_d_readiness_refinement_plan.md). Phase D implements clean
baselines; E-H remain planned. Runtime availability does not establish real-data acceptance.

## Environment

The training extra adds PyTorch, NumPy and psutil. The installed Windows Python 3.12
environment is pinned in requirements-phase-d.lock; requirements.lock retains the
foundation environment. Install the CUDA build from the official PyTorch index:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-phase-d.lock --extra-index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe scripts/check_phase_d.py
```

CUDA is explicit in configs/phase_d.json. Unavailable CUDA fails rather than silently
switching devices. A separate CPU config uses device=cpu and threads=1 for correctness
tests. Model/data dtype is Float32; diagnostic reductions use Float64. Determinism is
scoped to the recorded environment. No training dependency is imported by A-C CLI routing.

## Commands and dependencies

All commands require --config. Paths are workspace-relative. Query individual --help
for exact syntax. There is no generic train or run-all bypass.

| Command | Additional required inputs | Output |
|---|---|---|
| prepare-learning-data | --phase-c-validation, --dataset, --task, --fold, --scenario | Verified arrays, membership CSV, data.json |
| initialize-model | --prepared, --seed | Common MLP/logistic states and compatibility |
| train-centralized | --prepared, --initialization, --method, --seed | Majority/logistic/MLP evidence and checkpoints |
| run-federated | --prepared, --initialization, --method, --seed, repeated --review | FedAvg/FedProx/median/trimmed_mean/FLTrust results |
| review-learning | --run, --disposition, --rationale-file; --limitations-file for explained negatives | Immutable review linked to exact run evidence |
| validate-phase-d | Repeated --review covering all eight methods, --test-report from check_phase_d.py | Acceptance report for one identity and seed, linked to current regression evidence |

Real-data entry and resume require a source-verified Phase C PASS for the exact dataset,
task, fold and scenario. The configured feature variant is recorded throughout. Arrays
reuse frozen preprocessing; no test array is materialized. A data source may be scanned
for hashes and membership lookup without making final-test examples available to learning.

## Method behavior

All neural methods use the saved MLP initialization except the separate linear-softmax
logistic model. Majority ties follow label-map order and its probability loss is undefined.
FedProx uses the configured proximal term; median and trimmed mean have no scalar weights.
Trimmed mean requires enough valid clients for the declared nonzero trimming count.
FLTrust uses a separate server-reference update on the fixed trusted panel, records its
Adam adaptation, and does not depend on Phase E assessments.

The first real-data FL run requires a completed centralized MLP review. Additional
comparators require an applicable FedAvg review. Non-IID FedAvg additionally requires a
near-IID review from the same data protocol/task/fold/variant/seed. Review means an explicit
operator judgment supported by diagnostics, not merely that a process exited successfully.
Unresolved learning blocks advancement. Negative findings need rationale and limitations.

## Debugging and resume

Each attempt has configuration/environment metadata and a completion.json published last.
An incomplete attempt has failure.json. Training-level failures have failed_round.json
and retain the parent model. Inspect those before changing any experiment parameter.

Round/epoch diagnostics identify data roles, support, objective versus predictive loss,
weights and participation, update norms, memory and costs. Nested aggregate checkpoint
cost is labelled inside aggregation; avoid summing it twice. No final-test metric is used.

For centralized or federated resume, pass --resume with the selected resume-NNNN.json
marker and the same inputs/configuration/method/seed. Completed checkpoints are immutable;
resume starts a new attempt and reruns only work after that boundary. Centralized Adam
state is restored; FL local optimizers are reset by protocol. Do not resume across changed
source data, preprocessing, implementation or device/software settings.

## Acceptance

Separate software evidence, actual data eligibility and learning review. Synthetic fixture
acceptance demonstrates behavior only. Phase D cannot report real-data readiness until the
upstream gate and applicable real clean-learning reviews pass. See [acceptance evidence](phase_d_acceptance.md).
