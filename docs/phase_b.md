# Phase B implementation and execution

Phase B implements roadmap steps 4–9. Phase A configuration, seed namespaces,
Partition values, ArtifactRef, and immutable foundation records remain intact.
Every output retains `eligible_for_training: false`.

The interrupted work contained unconnected drafts of registration, audit, provenance,
CSV reading, configuration, and capture summaries. The resumed implementation fixes
those drafts and adds the missing commands, grouping, both protocols, panels,
validation, tests, and documentation.

## Architecture

The CLI only parses arguments and dispatches. Every executable stage has a separate
orchestrator under `src/edgefl/pipelines/phase_b_*.py`. Algorithms live under
`src/edgefl/data/`; domain code does not import CLI handlers.

| Module | Responsibility |
|---|---|
| config / inventory | Safe configuration, hashes, sizes, explicit CSV/PCAP pairs |
| csv_reader / schema | Logical CSV records and declared field equivalence rules |
| audit | Streaming counts, structural/semantic concerns, duplicates, cardinalities |
| provenance | Original selected membership and every candidate origin |
| pcap / verification | Isolated Wireshark adapter and packet correspondence |
| grouping | Observation identity, duplicate lineage, sessions, feature collisions |
| splitting | Chronological/purged groups and whole-capture folds |
| trusted_panel | Deterministic trusted-only panels |
| origin_validation / validation | Source identity, coverage, lineage, leakage, support |
| storage | Verified artifact IO and SQLite working indexes |

Versioned row contracts live in `contracts/phase_b.py`; configuration is validated
against `schemas/phase_b.schema.json`. Source registration precedes provenance and
does not claim to be a DatasetManifest. Grouping publishes the existing Phase A
DatasetManifest only when a verified subset exists, with an explicit retained scope.

Client allocation, preprocessing fitting, learning, scoring, poisoning, NSGA-II,
and experiment evaluation remain unimplemented.

## Commands

Each command requires explicit configuration and upstream completion paths. No
command runs an earlier stage automatically. Run tests from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase_b.py
```

Use an existing Phase A `run.json`, or create one with `init-run --seed 11`. In the
following commands, assign each variable the completion path returned by its stage.
The audit is shared by ML (`smoke`) and DNN (`primary`).

```powershell
$foundation = "runs/<foundation-run>/run.json"
.\.venv\Scripts\python.exe scripts/edgefl.py audit --config configs/phase_b.json --foundation $foundation
$audit = "<audit-completion>"

.\.venv\Scripts\python.exe scripts/edgefl.py provenance --config configs/phase_b.json --dataset smoke --audit $audit
$provenance = "<provenance-completion>"
.\.venv\Scripts\python.exe scripts/edgefl.py verify-captures --config configs/phase_b.json --dataset smoke --audit $audit --provenance $provenance
$evidence = "<capture-completion>"
.\.venv\Scripts\python.exe scripts/edgefl.py group --config configs/phase_b.json --dataset smoke --provenance $provenance --evidence $evidence
$groups = "<group-completion>"

.\.venv\Scripts\python.exe scripts/edgefl.py global-split --config configs/phase_b.json --dataset smoke --groups $groups --protocol A
$splitA = "<A-completion>"
.\.venv\Scripts\python.exe scripts/edgefl.py global-split --config configs/phase_b.json --dataset smoke --groups $groups --protocol B
$splitB = "<B-completion>"
.\.venv\Scripts\python.exe scripts/edgefl.py trusted-panel --config configs/phase_b.json --dataset smoke --splits $splitA
$panelA = "<A-panel-completion>"
.\.venv\Scripts\python.exe scripts/edgefl.py trusted-panel --config configs/phase_b.json --dataset smoke --splits $splitB
$panelB = "<B-panel-completion>"

.\.venv\Scripts\python.exe scripts/edgefl.py validate-phase-b --config configs/phase_b.json --dataset smoke --audit $audit --provenance $provenance --evidence $evidence --groups $groups --splits-a $splitA --splits-b $splitB --panel-a $panelA --panel-b $panelB
```

Repeat selected-data stages with `--dataset primary` for DNN. ML outputs are smoke
evidence, never primary benchmark results. Inspect `failure.json` and `events.jsonl`
in the affected attempt when a command fails.

## Evidence policy

Canonicalization is field-specific: declared numeric representations, hexadecimal
fields, Wireshark boolean formatting, explicit archive zero-fill sentinels, and
the OS_Fingerprinting → Fingerprinting label alias. Payloads and identifiers are
not generically coerced to numbers. Original files remain immutable.

Provenance `verified` means one canonical source origin, not verified predictor
semantics. Ambiguous candidates are all exported. Every parseable selected logical
record has an ID made from the full selected-file SHA-256 and record number.

Capture verification compares all declared predictors except labels and truncated
CSV timestamps. It requires multiple active packet attributes, unique matching
packets, at least two ordered source anchors, no capture timestamp regression,
and a stable observed clock offset. Actual dates come from packet epochs. Neither
matching row counts nor equal row numbers proves correspondence. Invalid semantics,
tool/field incompatibility, ambiguity, and missing evidence cause quarantine.

The policy is conservative; version-dependent dissector differences can reduce
support. Tool executable hashes and versions are recorded. Any predictor repair
would require a separately versioned corrected dataset and is outside this work.

TCP stream identities include the capture and retain the entire observed span.
UDP streams use 60-second inactivity boundaries. Verified observations without a
session use capture-scoped 300-second temporal blocks. Full capture spans prevent
selected-only gaps from dividing sessions. Feature collisions and conflicting
labels are reported separately from duplicated observations.

## Protocols and panels

Protocol A targets 70:7.5:7.5:15 chronologically, purges 30 seconds on each side of
boundaries, and excludes crossing groups. No class-driven redraw occurs. Reports
contain achieved proportions and independent-group support. `closed_set.csv` is
the reduced-support variant: unsupported classes and affected whole groups are
excluded without changing the benchmark's original membership records.

Protocol B holds complete eligible captures outside all development roles. Benign
folds measure benign-source generalization. Attack folds measure binary unseen-attack
detection, paired deterministically with a distinct eligible benign test capture
for FPR. Development ratios are normalized from 70:7.5:7.5. These attack folds are
not ordinary closed-set multiclass experiments.

Panels draw only from trusted rows, up to 512 per attack class and 512 benign rows.
Capture/session/group IDs are preserved; panel membership adds no global partition.
Reports expose missing classes and independent-group counts. Without benign support,
a panel is unusable for FPR objectives.

## Storage, reuse, and acceptance

Outputs are under configuration/dataset/stage identities in
`data/manifests/phase_b/`. Each immutable attempt contains configuration/environment,
events, summary, and artifact inventory. `completion.json` is written last.
Incomplete attempts are never reused. Completed artifacts require matching source,
configuration, schema, implementation, upstream references, and output hashes.

SQLite indexes and packet spools under `_work/` are working state, not dependencies
of portable CSV/JSON manifests. Large stages run sequentially. The archive audit
checks free disk against a conservative four-times-archive estimate.

The final gate reports PASS, PASS_WITH_LIMITATIONS, or FAIL. It checks selected
coverage, candidate identity/ranges, provenance status, exclusions, duplicate
representatives, packet/group identity, protected roles, held-out captures, purge
gaps, panel membership, and support counts. No verified records or no feasible fold
with a benign-supported trusted panel yields FAIL. Unsupported scope remains explicit.

No Phase B status authorizes training. Phase C must create client/local partitions,
fit preprocessing only on allowed training records, and pass its pretraining gate.
The real-data execution index and handoff are saved under `reports/generated/`.
