# Phase C architecture and execution

## Active v2 interface

The [stabilization guide](phase_bc_stabilization.md) is authoritative for current v2 contracts.
assign-clients requires --task binary or --task multiclass. support_policy replaces require_binary_support;
closed_set_scope replaces unconditional required_labels. ARP addresses are excluded. Earlier interface
descriptions/examples below describe the v1 implementation; use the v2 guide and CLI --help for execution.

Phase C implements roadmap steps 10-13 only: stable client ownership, client-local
validation, fitted preprocessing, and the pretraining gate. It does not implement a
model, optimizer, federated round, poisoning, scoring, reputation, or evaluation.

The current real-data Phase B report is `FAIL`. Every Phase C entrypoint that needs
data requires a supplied Phase B validation with status `PASS` or
`PASS_WITH_LIMITATIONS` and a feasible fold accepted by that exact validation.
Consequently, the present real dataset cannot produce Phase C manifests. This is an
intentional gate, not an empty or synthetic substitute for real outputs.

## File ownership

    configs/phase_c.json
    src/edgefl/contracts/phase_c.py
    src/edgefl/schemas/phase_c.schema.json
    src/edgefl/client_data/
      config.py                 Phase C configuration and protected paths
      storage.py                Immutable attempts, hashes, lineage, completion loading
      assignment.py             Group profiles and allocation strategy classes
      local_split.py            Group-level local train/validation reservation
      preprocessing.py          Streaming fitter and frozen transformer class
      validation.py             Independent pretraining gate
    src/edgefl/pipelines/
      phase_c_common.py         Phase B acceptance boundary
      phase_c_assign.py         Client-assignment orchestration
      phase_c_local_split.py    Local-split orchestration
      phase_c_preprocessing.py  Preprocessing orchestration
      phase_c_validate.py       Gate orchestration
      phase_c_cli.py            Argument registration and routing only
    tests/test_phase_c.py

Domain modules do not import command handlers. The command router contains no
allocation, fitting, or validation algorithm. Each stage receives explicit completion
references and can be debugged without silently running a predecessor.

## Stage contracts

| Command | Required completed inputs | Published artifacts |
|---|---|---|
| `assign-clients` | Phase B validation and its exact global split | assignments, client distribution report, attempt/repair log |
| `local-split` | client-assignment completion | local split and client manifests |
| `fit-preprocessing` | local split and its Phase B provenance | transformer, feature dictionary, label maps, fitting rows, feature contract |
| `validate-phase-c` | all above plus Phase B validation/split | pretraining report and validated training-input references |

All outputs use immutable attempt directories under the Phase C configuration and
dataset identities. `completion.json` is written last. A failed or interrupted attempt
has `failure.json` and cannot be loaded as complete. Reuse requires identical input,
configuration, schema, options, and stage implementation hashes.

## Client assignment

The assignment manifest contains only the selected fold's `client_pool` rows. Complete
Phase B groups are assigned to a single client. Verified sessions therefore remain
intact because Phase B session groups are indivisible.

The configuration defines four initial scenarios:

- `near_iid`: greedily balances group size and per-class support.
- `dirichlet_moderate`: class targets drawn with alpha 0.5.
- `dirichlet_strong`: class targets drawn with alpha 0.1.
- `specialist`: attack classes are deterministically associated with two preferred clients.

The strategy classes share one interface. Attempts use the versioned `partition` seed
with fold, scenario, and attempt coordinates. Constraint repair may move complete groups
only. The stage records every attempted seed, repair, rejection reason, class support,
client size, independent groups, total-variation heterogeneity, size variation, and
global label entropy. If the configured minimum or binary support cannot be met, the
stage fails rather than weakening the constraint.

The repository configuration uses ten clients, at least 10,000 observations and two
groups per client, plus benign and attack support.

## Local validation

For every client, groups receive a deterministic hash order. A prefix closest to the
20% target becomes `local_validation`, subject to leaving the configured observation
minimum and binary support in `local_train`. A group or session cannot cross the two
roles. A client with fewer than two groups or no valid group-level holdout is rejected.

Local-validation rows are marked nonattackable. They do not appear in the fitting-row
manifest and are not part of the future registered training count.

## Frozen preprocessing

The fitter streams the selected CSV and looks up permitted local-training record
numbers in SQLite. It never fits from local validation, trusted, selection-validation,
or final-test observations.

The base variant removes time, source/destination hosts, targets, and declared payload
fields. It learns:

- mean numeric imputation;
- population standard scaling;
- bounded deterministic one-hot vocabularies;
- numeric and categorical constant-feature removal;
- exact output ordering;
- binary and observed multiclass label mappings.

The strict variant additionally removes configured counters, checksums, stream IDs,
names, messages, topics, and request strings. Shortcut fields are summarized in the
fit report before removal. Both variants are fitted from the same training rows and
serialized as JSON rather than executable object deserialization.

`FrozenPreprocessor` checks the full input-field order and rejects nonfinite output.
The feature contract hashes each output order and links the transformer, label map,
and exact fitting-row manifest.

## Pretraining gate

The final validator independently checks:

- source and upstream artifact identities;
- exact client-pool coverage and absence of protected global roles;
- group/session isolation across clients and local roles;
- configured per-client size, group, and binary-support constraints;
- valid labels and explicit unsupported required classes;
- exact equality between preprocessing fitting rows and local-training rows;
- feature-dictionary order against both frozen variants;
- finite transformations for client and global evaluation observations;
- byte-identical regeneration of assignments, client manifests, local splits, fit rows,
  transformers, feature dictionaries, feature contracts, and label mappings.

Only a `PASS` completion sets `eligible_for_training` to true. A limitation or error
does not authorize Phase D.

## Commands

Use workspace-relative completion paths. These examples are templates because the
current real-data Phase B result cannot satisfy the first command's gate.

    .\.venv\Scripts\python.exe scripts/edgefl.py assign-clients `
      --config configs/phase_c.json --dataset primary `
      --phase-b-validation <phase-b-validation-completion> `
      --split <global-split-completion> --fold <fold> --scenario near_iid

    .\.venv\Scripts\python.exe scripts/edgefl.py local-split `
      --config configs/phase_c.json --dataset primary `
      --assignments <assign-clients-completion>

    .\.venv\Scripts\python.exe scripts/edgefl.py fit-preprocessing `
      --config configs/phase_c.json --dataset primary `
      --local-split <local-split-completion> --provenance <phase-b-provenance-completion>

    .\.venv\Scripts\python.exe scripts/edgefl.py validate-phase-c `
      --config configs/phase_c.json --dataset primary `
      --phase-b-validation <phase-b-validation-completion> `
      --split <global-split-completion> --assignments <assign-clients-completion> `
      --local-split <local-split-completion> --preprocessing <fit-preprocessing-completion> `
      --provenance <phase-b-provenance-completion> --fold <fold> --scenario near_iid

Run all Phase A-C behavior checks with:

    .\.venv\Scripts\python.exe scripts/check_phase_c.py

## Phase boundary

Phase C publishes data ownership and preprocessing contracts. It does not materialize
model tensors or import a training framework. Phase D remains `planned` in the pipeline
catalogue, and the generic `train` command remains unavailable.
