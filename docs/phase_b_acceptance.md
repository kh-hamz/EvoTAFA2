# Phase B acceptance evidence

Scope: roadmap steps 4–9 only. Phase A's configuration and record contracts are
preserved; later phases remain unavailable in the command catalogue.

## Current stabilization evidence

This document preserves the original Phase B acceptance results. For the v2 changes, the 89-test result, and the pending real-data recovery checkpoint, see the [2026-10-03 stabilization acceptance report](../reports/generated/phase_bc_stabilization_acceptance_2026-10-03.md).

## Software checks

`scripts/check_phase_b.py` passes 55 tests, including all 22 Phase A regression tests.
The report is saved as `reports/generated/phase_b_tests.txt`.

The tests cover logical quoted/multiline CSV records; duplicate headers and invalid
encodings; bounded examples with complete malformed counts; numeric, hexadecimal,
boolean, label and sentinel rules; duplicate and feature-collision distinctions;
unique/ambiguous/missing provenance; equal capture counts without correspondence;
tool failure, timeout and malformed output; missing dates, timestamp resets and
sequence reversals; deterministic boundaries, purging and long sessions; whole-capture
isolation; trusted-panel roles, budgets, determinism and missing benign support;
artifact/source tampering; stale code/configuration; incomplete attempts; safe paths;
and independent stage execution.

A complete synthetic run exercises registration through the final validator for
both protocols. Its deliberate duplicate produces PASS_WITH_LIMITATIONS, retains
320 verified observations, and leaves training eligibility false.

## Full archive audit

The archive audit completed over 26 CSVs and 23,316,623 logical records. No source
had an unreadable structural failure. Registered sources include CSVs, captures,
README, and dataset documentation with SHA-256 and byte sizes.

| DNN expectation | Recomputed result |
|---|---:|
| Logical records | 2,219,201 |
| Columns | 63 |
| Canonical classes | 15 |
| Exact duplicate occurrences | 815 |

All four expectations match. The MITM source contains 1,229 rows with 63 fields,
but every timestamp is invalid or a sentinel. Its numeric-field violations remain
explicit semantic concerns; row width does not authorize capture interpretation.

The exact audit completion path, per-dataset stage paths, and execution state are
recorded in `reports/generated/phase_b_execution.json`. The consolidated real-data
handoff is saved in `reports/generated/phase_b_handoff.json` and its Markdown companion.

## Claim and completion boundary

Implementation acceptance and dataset support are separate. The real-data validator
may return PASS_WITH_LIMITATIONS or FAIL when the archive cannot establish required
provenance, capture correspondence, isolated pools, or benign panel support. Its
saved report is authoritative; software tests do not establish real class support.

Unsupported records/classes/folds remain traceable and excluded from corresponding
claims. Original DNN membership is preserved. No repaired predictor benchmark is
substituted. No Phase B result permits training: Phase C client/local allocation,
training-only fitting, and the pretraining gate are still required.

See [Phase B execution guide](phase_b.md) for architecture, contracts, commands,
reuse rules, and debugging.

## Completed real-data execution

All independent stages completed for the ML smoke dataset and selected DNN dataset.
Both final validators returned **FAIL**: no feasible split had a benign-supported
trusted panel. No integrity or leakage errors were reported. These results do not
authorize Phase C or training.

| Dataset | Selected records | Capture-verified representatives | Quarantined | Gate |
|---|---:|---:|---:|---|
| ML smoke | 157,800 | 19,815 | 137,985 | FAIL |
| DNN primary | 2,219,201 | 33,980 | 2,185,221 | FAIL |

Neither run supports full closed-set evaluation or an eligible Protocol B fold.
The verified attack-only subsets remain traceable but cannot support FPR objectives.

### Evidence limitations and follow-up

- The verifier conservatively rejects a whole capture with any negative timestamp
  step. Temperature_and_Humidity has 119 such steps. This is an exclusion policy,
  not proof that every affected observation is corrupt. Packet ordering and actual
  clock resets still need to be distinguished using a versioned evidence rule.
- DNN matching against that benign capture also reports 19 packet-sequence
  reversals and 20 distinct rounded clock offsets among 1,266,773 anchors. A future
  investigation must resolve those discrepancies as well; relaxing only the
  timestamp rule would not establish DNN support.
- Wireshark rejects the vulnerability-scanner PCAP as damaged: a packet length of
  136,146,411 bytes exceeds the reported maximum of 262,144 bytes.
- MITM has equal CSV and packet counts but no sufficient correspondence anchors,
  invalid/sentinel CSV timestamps, and unresolved predictor semantics.
- Recovering additional benign capture support must preserve selected benchmark
  membership. Raw source rows cannot silently augment the selected DNN dataset.

These are evidence-recovery follow-ups, not implemented predictor repairs.
Source files and Phase A foundation records remain unchanged. Any revised
verification policy requires new fingerprinted artifacts and full revalidation.
