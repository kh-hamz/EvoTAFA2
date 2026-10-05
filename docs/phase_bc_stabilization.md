# Phase B/C stabilization guide

Active contracts are phase-b.v2 and phase-c.v2. Historical completions cannot authorize new runs.
Downloads and historical outputs are preserved. Recovery requires the agreed evidence-review checkpoint.

## Responsibilities

| Domain | Responsibility |
|---|---|
| data.integrity | Registered bytes and operation-scoped verification |
| client_data.policies | Local support, metric capabilities, mandatory feature exclusions |
| client_data.experiments | Frozen task scope and consistent global/trusted views |
| client_data.assignment | Whole-group targets, repairs, fidelity, search diagnostics |
| client_data.local_split | Non-prefix local subsets and support tables |
| client_data.storage | Lineage and load_training_ready boundary |
| data.capture_diagnostics | Decimal timing evidence without authorization |

CLI/pipeline modules compose services. Contexts live for one command/load. Future training uses TrainingReadyBundle.

## Configuration and commands

support_policy selects server_scored or local_binary_metrics and explicit assigned/training/validation benign/attack
minima. server_scored defaults to zero and permits sparse clients. Structural minima remain 10,000 assigned,
8,000 training and two independent groups, with nonempty validation.
closed_set_scope selects supported_classes or full_15. near_iid_limits specifies maximum_size_deviation (0.10)
and maximum_total_variation (0.05), using original labels and the selected client pool as reference.

assign-clients requires --task binary or multiclass. Downstream commands inherit it. Protocol B rejects multiclass.
Both feature variants exclude IP/ARP addresses, time, targets and mandatory payloads.

```powershell
.venv/Scripts/python.exe scripts/check_phase_c.py
.venv/Scripts/python.exe scripts/edgefl.py diagnose-captures --help
.venv/Scripts/python.exe scripts/edgefl.py assign-clients --help
```

diagnose-captures takes --audit, --provenance, --dataset and --config. Repeated --capture limits inspection.
--historical-evidence permits source/hash-verified forensic inspection of old ineligible completions. Its dedicated
diagnostic-review schema cannot enter active training lineage, does not validate historical implementation
fingerprints against current code and does not upgrade historical acceptance.

## Debug artifacts

- partition_attempts.json: targets, repairs, measured fidelity and structured violations.
- allocation_failure.json: necessary-condition failure or search exhaustion.
- best_candidate.json: ineligible approximate-balance diagnostic, never near-IID PASS.
- local_split_failure.json: unsuccessful bounded local search.
- timing_anomalies.csv: exact backward steps and frame ranges.
- anchor_reversals.csv: source/frame ordering disagreements.
- Diagnostic summary.json: offsets, modal residuals, correspondence and tool failures.

Failures do not silently alter thresholds, scope, source data, split membership or fitting rows.
