# Phase B/C stabilization plan

Agreed on 2026-10-03. Scope: six readiness findings before Phase D.

## Decisions and architecture

Preserve selected benchmark membership, predictors and historical artifacts. Review diagnostics before
authorizing a versioned recovery rule. Permit a named supported-class Protocol A scope and a separate
full-15 mode. Permit sparse clients in server-scored experiments; universal 100/80/20 minima are superseded.
Keep ten clients, 10,000 assigned rows, 8,000 training rows, two groups and nonempty local validation.
The 0.2 validation target is subordinate to group integrity.

Maintain CLI -> orchestration -> domain -> immutable contracts. Use operation-scoped VerificationContext,
immutable SupportPolicy, ExperimentContract, AllocationResult and TrainingReadyBundle. Preserve AllocationStrategy
implementations and internal mutable ClientState. Do not add a generic workflow engine. Domain modules do not
import pipeline handlers. Version changed B/C contracts as v2; preserve Phase A records and seed namespaces.

## Fix 1: data evidence

1. Add diagnose-captures with explicit audit/provenance dependencies and no training eligibility.
2. Measure backward-step magnitudes/ranges, source/frame reversals, exact clock offsets, modal residuals,
   correspondence, session context and tool failures. Prioritize Temperature_and_Humidity and an attack capture;
   keep MITM and corrupt vulnerability-scanner failures separately visible.
3. Publish hash-linked evidence and proposed recovery decisions. Review and approve numerical tolerances before
   implementing acceptance changes. Never choose tolerances just to obtain PASS.
4. After approval, implement the frozen rule without modifying predictors, appending raw rows, equating CSV rows
   to frames or guessing ambiguous dates. Preserve session isolation and quarantine affected sessions as needed.
5. Regenerate grouping, scoped splits, trusted panels and gates. Check sufficient observations, independent groups
   and task-specific support. If no usable fold results, report the blocker; dataset substitution needs a separate decision.

## Fix 2: cached authorization

1. Recursively check versions, stage, config/code fingerprints, artifact hashes, expected dependency roles,
   dataset and consistent task/fold/scenario. Detect active-path cycles while permitting shared upstream nodes.
2. Verify registered bytes once per registry per operation. Never persist successful verification across commands.
3. Only validate-phase-c can enable training. Require agreement between completion, PASS report, exported eligibility,
   exact training references, experiment scope, features, labels and scoped trusted panel.
4. Expose load_training_ready: begin fresh verification and return TrainingReadyBundle only for a matching PASS.
5. Reject changed CSV/PCAP bytes, forged eligibility, missing reports and stale schemas.

## Fix 3: near-IID fidelity

For N client-pool observations and K clients, target t=N/K. Size deviation is abs(n_i-t)/t.
Total variation is half the sum over original labels of abs(p_client(label)-p_client_pool(label)).
The reference excludes global holdouts and excluded observations. Every client must have size deviation <=0.10
and total variation <=0.05. Group granularity must not automatically increase either limit.

1. Allocate whole groups using changes in normalized squared target error, weighting size error and mean label
   error equally. Use descending group size and seeded deterministic ties.
2. Repair structured violations without invalidating donors; prioritize deficit reduction then scenario cost.
   Record targets, changes and before/after distributions. Improve fidelity with bounded whole-group moves/swaps.
3. Independently recompute limits in the final gate. Separate accepted, necessary_condition_failed and
   search_exhausted outcomes. Search exhaustion does not prove mathematical infeasibility.
4. Retain the best unsuccessful candidate as an ineligible approximate-balance diagnostic.
5. Record Dirichlet targets and specialist preferences without applying near-IID limits to those scenarios.

## Fix 4: support policies

1. Default server_scored has no universal local binary minima; preserve structural minima/nonempty partitions.
2. Allow local_binary_metrics profiles with explicit assigned/training/validation benign and attack minima.
   Assigned support must cover required training plus validation support.
3. Apply one policy to assignment, donor repair, local splitting and final validation.
4. Replace prefix-only splitting with bounded deterministic subset construction and moves/swaps.
5. Record zero-filled original-class counts, independent-group counts and binary aggregates for assigned,
   training and validation. Counts describe exposure, not demonstrated expertise.
6. Mark FPR unavailable without benign support, and binary ROC-AUC/PR-AUC unavailable without both classes.
   Nonempty loss/accuracy remains usable. Never invent absent-class scores as zero.

## Fix 5: identifiers

1. Derive mandatory exclusions from shared address semantics plus time, targets and payload fields.
2. Exclude both ARP protocol-address columns in base and strict variants. Reject weakened policies/unknown fields;
   known mandatory exclusions may be absent in valid reduced fixtures.
3. Persist feature-policy version/hash and independently inspect output source descriptors in the gate.
4. Retain identifiers for provenance. Regenerate affected preprocessing/gates. Retention ablations are out of scope.

## Fix 6: experiment-aware labels

1. Require --task at client assignment and inherit/validate it downstream.
2. Freeze dataset, task, protocol, fold, interpretation, scope, ordered vocabulary, selected manifest,
   held-out captures and support policy in ExperimentContract.
3. Protocol A multiclass uses the supported closed-set manifest: exclude unsupported whole groups, recalculate
   support, filter the trusted panel consistently and freeze vocabulary. full_15 fails unless all labels are supported.
4. Protocol B supports binary evaluation, Normal=0 and any attack=1; the held-out attack type is not required in training.
5. Keep held-out captures outside development/fitting/scoring. Require both binary classes in development;
   benign-source final tests may be benign-only with unavailable metrics disclosed.
6. All clients share the applicable vocabulary even when locally single-class.

## Tests and rollout

Add independent regressions for changed sources after PASS, exports, shared lineage, pure-label balance,
explicit formulas, failed searches, sparse clients, metric profiles, varying ARP addresses, supported/full-15
scopes, binary unseen attacks and diagnostic non-authorization. Regeneration is not the only oracle.

Run Phase A/B/C tests, dependency checks and CLI smoke checks, and publish evidence. After approved recovery,
regenerate affected real artifacts. Phase D requires source-verified B acceptance, accepted assignment, valid
local splits, training-only fitting, matching feature/label/experiment contracts and a fresh load_training_ready
PASS. Unsupported experiments remain blocked independently. No Phase D-H implementation or GitHub push is included.
