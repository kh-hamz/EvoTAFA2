# Phase D implementation specification

Approved scope: roadmap Steps 14-17, incorporating the 2026-10-04 readiness refinement.
Implementation date: 2026-10-05. Acceptance results are recorded separately.

## Decisions

- CUDA experiments on the RTX 3070; CPU correctness tests and explicit CPU configuration.
- Strict features by default; explicit separate base-variant configurations remain supported.
- Shared Adam across neural methods. FLTrust records its optimizer adaptation to the paper's SGD procedure.
- Implement majority-class, logistic, centralized MLP, FedAvg, FedProx, coordinate median, trimmed mean and FLTrust.
- Preserve the source-verified B-to-C gate. Synthetic tests do not authorize real-data experiments.
- Keep poisoning, risk, reputation, assessment-based weighting, NSGA-II and final-test evaluation outside D.

## Architecture

The learning domain separates config, data, models, training, aggregation, engine, metrics,
checkpoints, storage and review. Thin Phase D pipeline modules and CLI routing compose
these services. Existing ClientTrainer/Aggregator protocols and foundation records remain;
new Phase D records and artifact versions describe new behavior without migrating A-C.

Composition gives LocalTrainer only client training views and ServerReference only the
trusted view. BaselineAggregator composes pure numerical reductions with artifact IO.
The engine coordinates rounds without embedding dataset recovery or assessment modules.

Six independently callable commands prepare arrays, initialize models, train centrally,
run federation, record learning review, and validate an explicit acceptance scope.

## Configuration and data

Use configs/phase_d.json and a dedicated strict JSON schema. Preserve A-C settings.
Defaults: strict features, CUDA, six CPU threads, 128/64 ReLU MLP, Float32, Adam 0.001,
batch 256, one local epoch, 100 rounds/centralized epochs, full participation, FedProx
mu 0.01, trimmed fraction 0.20 per tail, and one root-training epoch. Matched seeds remain
11, 22, 33, 44, 55. No batch normalization, dropout, mixed precision, automatic tuning or
early stopping is introduced.

Prepare bounded memory-mapped arrays using only accepted local-training/local-validation,
selection-validation and trusted-panel membership. Reuse FrozenPreprocessor, preserve row
identities, check disk space and finite Float32 conversion, and exclude final-test arrays.
Verify the original sources and lineage on entry and resume. No implicit raw-data fallback.

## Models and numerical behavior

Save common CPU MLP and linear-softmax initial states per task, feature contract and seed.
Use training counts for majority choice, with label-order tie resolution. Centralized
models train only on the union of client-training rows. Reset local optimizer state each
round; retain centralized optimizer state across epochs.

FedAvg uses registered sample counts. FedProx adds mu/2 times squared distance to the
parent model and uses FedAvg aggregation. Median averages the middle two coordinates for
even counts. Trimmed mean requires at least one discarded observation per tail and a
nonempty remainder. FLTrust uses clipped cosine scores and root-norm normalization,
without sample weighting or adding the root as a client. Invalid or degenerate rounds
produce explicit failure and preserve the parent; never substitute a different method.

## Evidence and persistence

Record role-specific losses/F1/FPR/support, update magnitudes, applicable weight changes,
timing and memory. Preserve undefined results and distinguish participation changes.
Selection validation chooses checkpoints by macro-F1, then lower FPR, then earlier step.
Final-test data is unavailable to training and selection.

Publish immutable stage attempts, per-step model states and resume markers, hashes,
environment/configuration identities, and failure reports. Resume completed boundaries
with matching data/method/configuration/environment. Use restricted state-dictionary loading.

Require explicit operator reviews: acceptable, unresolved, or explained negative with
limitations. Review cannot override failed software or data acceptance. Centralized MLP
review precedes near-IID FedAvg; non-IID and other methods require applicable predecessor
reviews. Phase D scope acceptance requires all eight reviewed methods.

## Verification obligations

Test known-answer numerical reductions, learning fixtures, feature/membership isolation,
changed-source/cache/checkpoint rejection, review prerequisites, CPU/CUDA execution and
resume, and synthetic B/C/D integration. Run A-C regressions after integration. Record
actual evidence in [Phase D acceptance](phase_d_acceptance.md).

Current real-data feasibility is independent of software completion. Inspect the gate
before experiments and report blocked acceptance while recovery remains unresolved.
