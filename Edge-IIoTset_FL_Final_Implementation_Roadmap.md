# Finalized Edge-IIoTset Federated Learning Research Roadmap

## Phase B/C stabilization revision, 2026-10-03

The [agreed stabilization plan](docs/phase_bc_stabilization_plan.md) supersedes universal local binary
minima and unconditional fifteen-class requirements. Sparse server-scored clients are permitted;
supported-class Protocol A and binary Protocol B scopes are explicit. Near-IID requires maximum
relative size deviation 0.10 and maximum total variation 0.05 against the experiment client pool.
Search exhaustion is unsuccessful, not proof of infeasibility. Phase D requires a fresh source-verified
v2 training-ready PASS. Capture recovery requires the agreed diagnostic evidence review.

## Phase D readiness refinement, 2026-10-04

The [targeted readiness specification](docs/phase_d_readiness_refinement_plan.md) clarifies
phase ownership, clean-learning review, common diagnostics, and profiling timing.
Phase D includes FLTrust but does not depend on Phase E assessments or Phase F search.
This is a specification revision; implemented availability and Phase A-C contracts are unchanged.
Real-data experiments remain blocked until their source-verified B-to-C acceptance succeeds.

## 1. Review conclusion and required improvements

The initial implementation plan provides a strong foundation for preprocessing and reproducibility. It should be extended and corrected before implementation, particularly around provenance, evaluation partitions, poisoning experiments, and the complete NSGA-II lifecycle.

This review covered both local planning documents, the archive inventory and README, selected CSV records, and the complete small MITM and OS Fingerprinting source CSVs. **The previously documented full-DNN counts remain expected audit results; they were not independently recomputed during this review.**

Your selected research direction is:

- Four objectives: aggregate classification performance, benign false-positive rate, current poisoning risk, and historical unreliability.
- Two evaluation protocols: session/time-separated classification and independent whole-capture generalization.
- External replication on CICIoT2023 using independently trained models.

### Changes needed in the existing plan

| Issue | Required improvement |
|---|---|
| DNN selection versus reconstruction from all source CSVs | Keep DNN membership as the primary benchmark. Recover its provenance from source files; concatenating all source files creates a different dataset and must be reported separately. |
| Timestamp reliability | Validate timestamp semantics before constructing temporal groups. Sampled DNN timestamps lack month/day, and all 1,229 MITM source rows contain 0.0 or 6.0 in frame.time. |
| Structural versus semantic validation | Checking 63 columns is insufficient. Validate that values actually belong to their declared fields, especially in MITM records. |
| Whole-capture holdouts | The archive has one source CSV/PCAP per attack type. Holding out complete attack captures can remove entire classes from training. Separate closed-set classification from unseen-capture or unseen-attack evaluation. |
| Ambiguous grouping | A combined capture–flow–window identifier does not automatically prevent one flow from crossing windows. Explicitly enforce session isolation and boundary handling. |
| Duplicate feature vectors | Identical reduced feature vectors are not necessarily duplicate observations. Prevent duplicated observations from crossing partitions; treat feature collisions separately and measure their impact. |
| Shared validation | Separate trusted aggregation data from validation used for tuning and checkpoint selection. |
| Preprocessing before local validation | Create local holdouts before fitting learned preprocessing. Fit on the union of actual client-training rows, excluding local validation rows. |
| Delayed poisoning development | Build and verify the attack harness after clean baselines, before calibrating risk and reputation. Clean NSGA-II integration should still precede full adversarial experiments. |
| Undefined fixed weighted baseline | Specify both its weighting rule and whether coefficients or client weights remain fixed. |
| Communication objective | Remove communication from the four objectives. Post-upload weighting cannot undo communication already incurred. Measure communication separately. |
| Incomplete NSGA-II sequence | Include parent–offspring elitist survival, failure handling, evaluation budgets, and deterministic compromise selection. |
| Statistical interpretation | Distinguish variability across training seeds from uncertainty across independent captures. Millions of packets do not constitute millions of independent observations. |

The dataset publication supports its centralized/FL purpose and fourteen attack categories, but the local archive must determine the actual schema and usable observations. [Edge-IIoTset publication](https://doi.org/10.1109/ACCESS.2022.3165809)

## 2. Research contract and initial defaults

These defaults establish a reproducible starting protocol. Changes motivated by development results must be versioned and frozen before final experiments.

| Component | Decision |
|---|---|
| Primary data | Archived DNN selected dataset with verified provenance |
| Smoke-test data | ML selected dataset; its results are not primary research results |
| Tasks | Binary first for debugging; separate multiclass experiments for attack expertise claims |
| Global allocation | Approximately 70% client pool, 7.5% trusted server pool, 7.5% selection validation, 15% locked test |
| Local allocation | Approximately 80% local training and 20% local validation, respecting groups |
| Initial federation | 10 simulated clients, full participation |
| Heterogeneity | Near-IID, Dirichlet α = 0.5, Dirichlet α = 0.1, and declared specialists |
| Model | MLP with hidden layers 128 and 64, ReLU, no batch normalization or dropout initially |
| Initial training | Adam, learning rate 0.001, batch size 256, one local epoch per round |
| Initial duration | 100 rounds; retain both final and best-selection-validation checkpoints |
| Primary loss | Unweighted cross-entropy; class-weighted loss as a controlled sensitivity study |
| NSGA-II development budget | Population 24, eight offspring generations per round |
| Main experimental replication | Five matched seeds: 11, 22, 33, 44, 55 |
| Initial attack setting | 20% malicious clients, attack activation at round 11 |
| External dataset | CICIoT2023, independently preprocessed and trained |
| Execution environment | Single-machine simulation first; VM demonstration after research results |

The proposed architecture and numerical settings are **starting research defaults, not claims of optimality**. Development experiments will establish adequacy using selection validation, with equal tuning opportunities for competing methods.

The trusted server pool is an adaptive training resource because aggregation repeatedly consults it. It must never be described as independent validation of final performance.

## 3. Detailed implementation roadmap

### Phase A — Establish the experimental foundation

**Step 1 — Freeze the research questions and claim boundaries**

- Define the primary question: whether NSGA-II weighting improves the robustness–detection tradeoff under synthetic non-IID data and controlled poisoning.
- Define secondary questions covering rare-class performance, reputation, expertise, trusted-data dependence, and computational cost.
- State that simulation with pooled preprocessing does not establish a privacy-preserving deployment.

**Deliverable:** Research protocol containing hypotheses, primary endpoints, comparison methods, and permitted claims.

**Step 2 — Establish the project and configuration contract**

- Separate immutable sources, manifests, processed arrays, model artifacts, experiment configurations, and reports.
- Give every run a unique identity linked to dataset hashes, code revision, configuration, and environment.
- Separate random streams for partitions, initialization, minibatches, participation, poisoning, and evolutionary search.

**Deliverable:** Reproducible project specification and configuration schema.

**Step 3 — Define module interfaces before implementation**

The implementation should expose these conceptual boundaries:

| Interface | Responsibility |
|---|---|
| Dataset manifest | Source identity, schema, checksums, provenance confidence |
| Split manifest | Observation, capture/session/group, partition, exclusion reason |
| Client manifest | Client ownership, local split, class support, partition seed |
| Client update | Client/round identity, parent-model identity, parameter delta, registered training count, timing |
| Client assessment | Quality, expertise, risk components, prior reputation, support |
| Candidate evaluation | Ordered client weights, four objectives, feasibility, evaluation identity |
| Round result | Selected weights, resulting checkpoint, diagnostics, reputation transition |
| Experiment result | Protocol identity, seed, metrics, costs, failure status |

- Prevent training and aggregation modules from receiving final-test data or simulator-only malicious identities.
- Keep preprocessing, model dimensions, and label mappings versioned and checked at every boundary.
- Make methods interchangeable through a common aggregation interface.

**Deliverable:** Module responsibility and data-access specification.

### Phase B — Prove that the dataset supports the intended experiments

**Step 4 — Register and audit immutable inputs**

- Record source hashes, sizes, CSV/PCAP pairings, schemas, label counts, malformed records, non-finite values, and exact duplicates.
- Reproduce or explain deviations from the draft’s DNN expectations: 2,219,201 rows, 63 columns, 15 classes, and 815 duplicate occurrences.
- Validate label consistency and explicit mappings such as source OS_Fingerprinting versus selected-data Fingerprinting.

**Gate:** Every discrepancy has an explanation; semantic problems cannot pass merely because row widths match.

**Step 5 — Recover provenance without changing the benchmark**

- Match selected DNN observations to source CSV observations using declared canonicalization rules.
- Record unambiguous matches, multiple possible matches, and unmatched observations separately.
- Use PCAP evidence where necessary to verify timestamps, protocol fields, and source relationships; never assume CSV row numbers equal packet numbers.

**Deliverable:** Provenance mapping with confidence and unresolved-record reports.

**Step 6 — Resolve timestamp and field-semantics problems**

- Audit each source’s time format, ordering, resets, missing date components, and capture duration.
- Investigate the MITM field values against its PCAP before accepting temporal or flow interpretation.
- Quarantine unresolved observations from protocols requiring verified provenance. Preserve their counts and explain any effect on class coverage.

**Gate:** A class with unresolved provenance cannot silently appear in a claimed leakage-resistant temporal benchmark.

If records must be repaired or re-extracted, create a separately versioned corrected dataset. Do not silently replace values while retaining the original dataset identity.

**Step 7 — Define observation identity, duplicates, and grouping**

- Remove repeated source observations deterministically and preserve a resolution log.
- Build protocol-aware bidirectional sessions where possible; use verified temporal blocks where sessions are unavailable.
- Keep an entire session together or exclude sessions crossing a split boundary. Do not divide a session simply because its packets occupy different windows.
- Audit identical predictor vectors and conflicting labels separately from source duplication.

**Defaults:** Five-minute blocks and a 30-second purge are initial candidates, subject to source-duration and session-feasibility checks using development data.

**Deliverable:** Group manifest and leakage audit, including the largest groups and rare-class group counts.

**Step 8 — Construct the two evaluation protocols**

**Protocol A: closed-set, session/time-separated classification**

- Allocate verified groups into client, trusted, selection-validation, and test pools.
- Prefer chronological allocation within verified sources, with boundary purging.
- Require training support for every class described as closed-set; disclose unsupported evaluation classes and their group counts.

**Protocol B: independent whole-capture generalization**

- Hold complete captures outside training and preprocessing.
- Evaluate leave-one-benign-capture-out generalization while retaining other benign sources for development.
- Evaluate leave-one-attack-capture-out binary detection as unseen-attack generalization, with development holdouts drawn from the remaining captures.
- Do not present an unseen attack class as ordinary closed-set multiclass evaluation.

**Gate:** Each protocol has its own locked manifests and an explicit statement of what generalization it measures.

**Step 9 — Separate trusted optimization data from selection validation**

- Use trusted data only for server quality, expertise, risk, fitness, and reference computations.
- Use selection validation for hyperparameters, checkpoint selection, and development comparisons.
- Restrict final-test access to a dedicated evaluation stage after protocol freeze.

For NSGA-II, create one deterministic trusted fitness panel per run, targeting at most 512 observations per attack type while respecting groups. Disclose its distribution; it is an optimization panel, not a prevalence estimate. Use the full trusted pool for periodic diagnostics.

**Gate:** No candidate or client scoring operation can read selection validation or final test.

### Phase C — Build stable client datasets and preprocessing

**Step 10 — Generate immutable client assignments**

- Allocate complete eligible training groups to the ten clients.
- Produce near-IID, moderate/strong Dirichlet skew, and specialist scenarios.
- Fix client assignments across aggregation methods for each matched seed.
- Record retries, repairs, rejected partitions, class proportions, size distributions, and measured heterogeneity.

Preserve the initial 10,000-observation minimum and group integrity. Apply explicit metric-dependent support policies: server-scored clients may be single-class, while local binary metric profiles declare their own support minima. Report unsuccessful searches and achieved heterogeneity without silently weakening constraints or claiming mathematical infeasibility.

**Deliverable:** Client manifests and distribution diagnostics.

**Step 11 — Reserve client-local validation**

- Split each client’s groups into local training and validation before learning preprocessing.
- Keep local validation clean and outside the attackable training pool.
- Use local validation for diagnostics initially; do not let client-specific early stopping change the primary fixed local-training budget.

**Gate:** No local-validation observation contributes to preprocessing fitting, gradients, or registered training counts.

**Step 12 — Fit and freeze preprocessing**

- Fit preprocessing on the union of actual client-training rows only.
- Apply the declared identifier/payload exclusions; inspect remaining counters, checksums, names, and stream identifiers for capture-specific shortcuts.
- Learn numeric imputation, categorical handling, constant-feature removal, scaling, and output order.
- Save a strict feature-removal variant for a later generalization ablation.

All competing methods within a matched scenario use identical preprocessing. A new partition seed may require a new transformer because the actual local-training membership changes.

**Deliverable:** Serialized transformer, feature dictionary, label mappings, and fitting-row manifest.

**Step 13 — Execute the complete pretraining gate**

- Verify source identity; observation/session separation; valid labels; finite transformed arrays; and identical feature ordering.
- Check all client constraints and explicitly report unsupported classes.
- Verify that preprocessing used only permitted fitting observations.
- Demonstrate that manifests regenerate deterministically.

**Gate:** Every real-data model experiment requires a fresh source-verified Phase C PASS
for its dataset, task, fold, and scenario. Synthetic component tests cannot authorize
real-data experiments. Entry and resume checks use the existing training-ready loader.

### Phase D — Establish trustworthy learning baselines

**Step 14 — Create common model initializations**

- Before real-data model initialization, load the verified training-ready bundle with explicit dataset, task, fold, and scenario expectations.
- Instantiate independent binary and multiclass MLPs.
- Save one initial state per task, matched seed, and feature schema.
- Start every compared FL method from that same state.
- Reset local optimizer state each round in the initial protocol.

**Deliverable:** Initialization artifacts and model compatibility tests.

**Step 15 — Verify centralized learning**

- Run majority-class, logistic-regression, and centralized MLP baselines.
- Train the centralized MLP on exactly the union of client-training rows, excluding local holdouts.
- Verify learning, finite losses, rare-class behavior, and the relationship between training and selection-validation performance.

- Record initial and subsequent training and selection-validation measurements, class support, configuration, seed, initialization, and input artifact identities.
- Compare against the sanity baselines and retain an investigation record for weak learning, instability, or suspiciously perfect performance.

**Gate:** Complete the [clean-learning evidence review](docs/research_protocol.md#clean-learning-evidence-and-review)
before advancing the condition. Unexplained behavior blocks advancement. An explained
negative finding may proceed with documented limitations, but cannot override a software
failure or failed data gate. No universal performance threshold is imposed.

**Step 16 — Verify FedAvg and the round engine**

- Broadcast one global model, train selected clients, collect deltas, and aggregate using registered training-set sizes.
- Validate client and round identities, model compatibility, finite updates, and successful-client accounting.
- Test single-client equivalence and a manually checkable aggregation example.
- Run near-IID first, then moderate and strong non-IID scenarios.
- Use the common [diagnostic definitions](docs/interfaces.md#future-round-diagnostics) from the first training-engine implementation.
- Complete centralized learning review before near-IID FedAvg, then review near-IID before non-IID FedAvg. Review each condition before its downstream experiments.

**Gate:** Record software correctness and clean-learning review separately. Label/size
balance alone does not establish identical feature distributions or guarantee learning.

Use standard sample-weighted FedAvg as the reference definition. [FedAvg paper](https://proceedings.mlr.press/v54/mcmahan17a.html)

**Step 17 — Establish comparison baselines**

Comparison implementation and acceptance belong to the following phases:

| Phase | Methods and required evidence |
|---|---|
| D | Majority-class, logistic regression, centralized MLP, FedAvg, FedProx, coordinate-wise median, trimmed mean, and FLTrust: correctness, reproducible clean execution, assumptions, and clean-learning review |
| E | Fixed-coefficient adaptive weighting and frozen client weighting using actual quality, risk, and reputation outputs at Step 23 |
| F | NSGA-II weighting using actual candidate fitness evaluation and evolutionary search |
| G | Complete matched comparisons, including adversarial evaluation and the existing experimental matrix |

FedProx changes local optimization. Median and trimmed mean need not produce scalar
client weights; trimmed mean requires a declared trimming fraction and eligibility checks.
FLTrust is required in D and computes a server reference update from the permitted
trusted resource. Give it the same trusted-data access budget and disclose its different
use and server computation. Ordinary client trainers do not receive trusted data.
[FedProx](https://arxiv.org/abs/1812.06127), [FLTrust](https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/)

**Gate:** All Phase D methods have reproducible clean execution, correctness evidence,
recorded assumptions, and completed clean-learning review. D completion does not require
E/F algorithms or poisoning-resistance evidence. Evaluate poisoning after the Phase E
attack harness exists. Do not fabricate quality, risk, or reputation inputs for baselines.

### Phase E — Build and validate poisoning, quality, and trust

**Step 18 — Implement the controlled attack harness**

- Keep malicious identities in the simulator/evaluator, outside the defense interface.
- Freeze malicious identities, attack randomness, start rounds, and intensity across matched methods.
- Distinguish compromised clients from clients actively attacking in a particular round.

Initial separately evaluated attacks:

| Attack | Initial specification |
|---|---|
| Untargeted label flipping | Corrupt 30% of local-training labels; map to a different valid label |
| Targeted label flipping | Relabel 30% of a supported target attack class as Normal |
| Sign flipping | Negate the client’s parameter delta |
| Update scaling | Multiply the submitted delta by five |
| Additive noise | Add random noise with norm matched to the honest delta |
| Intermittent attack | Alternate five attacking and five honest rounds after activation |

Select the targeted class deterministically from eligible classes using training support only. Log when selected malicious clients lack target-class examples.

**Gate:** Attacks change only authorized training labels or submitted updates; trusted and evaluation data remain intact.

**Step 19 — Calculate server-observed client quality**

- Evaluate each submitted client model on the common trusted panel.
- Record task macro-F1, balanced accuracy, benign FPR, and loss relative to the current global model.
- Retain class support alongside every score.
- Ignore client-reported performance when deciding weights.

**Deliverable:** Per-round quality records independent of malicious identity labels.

**Step 20 — Calculate per-class expertise**

- For multiclass models, measure per-class precision, recall, and F1.
- For binary models, measure detection recall separately for each underlying attack type; describe this as attack-type detection expertise.
- Shrink low-support expertise estimates toward the current global model’s corresponding score, using a fixed support scale of 50 observations initially.
- Mark absent trusted classes as unsupported rather than assigning invented expertise.

Use expertise to initialize candidate weights and interpret specialist behavior. Its removal must be included in ablations.

**Step 21 — Implement poisoning-risk estimators**

Start with three independently logged signals:

- Excess update magnitude relative to a robust peer reference.
- Directional disagreement with the coordinate-median update.
- Positive deterioration in trusted class-balanced loss relative to the current global model.

Normalize each signal to a bounded scale using rules established in clean development runs; use their equal-weight mean as the initial risk score.

- Freeze calibration before final adversarial comparisons.
- Report individual signals as well as the composite.
- Do not interpret anomaly scores as calibrated probabilities of maliciousness.
- Test false accusations against honest specialists explicitly.

**Gate:** The risk module never consumes malicious identities or test outcomes.

**Step 22 — Implement temporal reputation**

- Initialize reputation at 0.5.
- Use prior-round reputation during current-round optimization.
- After scoring the round, update observed clients with an exponential moving average of 1 − current risk, initially retaining 90% of the previous value.
- Leave absent clients unchanged initially; record staleness.
- Apply an explicit penalty to invalid submissions, while recording ordinary nonparticipation separately.

**Gate:** Reputation cannot depend on a future result or on the aggregation weight it is simultaneously helping select.

**Step 23 — Define the fixed weighted comparisons**

Use two clearly named comparators:

- **Fixed-coefficient adaptive weighting:** normalize the product of registered training count, quality, one minus current risk, and prior reputation, with a declared small floor preventing numerical collapse.
- **Frozen client weighting:** compute those weights after the first scored round and freeze them thereafter, renormalizing over participating clients.

The first tests whether evolutionary search improves over a simple rule using the same information. The second tests whether adapting client weights matters.

Both comparators are implemented and validated in Phase E, after their assessment
dependencies exist. Their completion is not a Phase D prerequisite.

### Phase F — Implement the complete NSGA-II aggregator

**Step 24 — Specify chromosomes and feasibility**

- Represent a chromosome as one weight per valid participating client.
- Require nonnegative weights summing to one.
- Limit an individual weight to 0.5 when at least two clients are available; permit weight one for the single-client case.
- Use deterministic projection onto the feasible bounded simplex.
- Exclude rejected updates before optimization and preserve a stable client-to-gene mapping.

**Gate:** Every evaluated candidate satisfies the constraints.

**Step 25 — Initialize the population**

Include:

- Feasible sample-weighted FedAvg.
- Uniform weighting.
- Fixed-coefficient adaptive weighting.
- Expertise-informed weighting.
- The previous selected solution remapped to current clients.
- Seeded random feasible candidates.

Ensure newly participating clients can receive nonzero weights. Repair, deduplicate, and refill the population deterministically.

**Step 26 — Define the four-objective fitness vector**

All objectives are minimized:

| Objective | Definition |
|---|---|
| Classification error | One minus task macro-F1 of the actual aggregated model on the trusted panel |
| Benign false-positive rate | Fraction of trusted benign observations classified as attacks |
| Current poisoning risk | Aggregation-weighted mean of current client risk |
| Historical unreliability | Aggregation-weighted mean of one minus prior reputation |

- Evaluate the actual aggregated parameters for performance objectives; a weighted average of client scores is not equivalent.
- Use the same panel and evaluation mode for every candidate in a round.
- Cache identical candidates.
- Record candidate requests, cache hits, actual model evaluations, and invalid candidates separately using the common diagnostics.
- Treat failed or non-finite model evaluations as infeasible.
- Log objective correlations and the fraction of candidates that are non-dominated.

Risk and historical unreliability may correlate strongly. Their separate value must be established through ablation.

**Step 27 — Implement non-dominated sorting and crowding distance**

- Rank feasible candidates using all four minimization objectives.
- Calculate crowding distance within each front.
- Handle constant objective ranges, duplicate vectors, boundary points, and ties.
- Make tie handling reproducible.

**Gate:** Hand-constructed objective sets produce known fronts and boundary behavior.

**Step 28 — Implement tournament selection, crossover, and mutation**

- Use binary tournaments based on rank, then crowding distance.
- Use simulated binary crossover and polynomial mutation.
- Start with crossover probability 0.9, per-gene mutation probability 1 / participating clients, and distribution indices of 20.
- Repair offspring after variation and maintain the configured population size.

**Gate:** Operators preserve client mapping, reproducibility, and feasible output after repair.

**Step 29 — Implement elitist survival**

- Combine parent and offspring populations each generation.
- Retain candidates front by front.
- Use crowding distance to fill the remaining places in the final accepted front.
- Stop at the declared generation/evaluation budget.
- Preserve the declared budget during a run; any revision based on development profiling must be explicit and recorded before main comparisons.

This is a required part of NSGA-II, not an optional optimization detail.

**Step 30 — Select one Pareto compromise**

- Select from the final non-dominated feasible front.
- Use the smallest equal-weight Euclidean distance to the ideal vector, with the objectives’ fixed [0,1] scales.
- Break ties by lower classification error, then lower FPR, then stable candidate identity.
- Save the full front and selection rationale each round.

If no finite feasible candidate exists, retain the previous global model and record an explicit failed aggregation round.

**Step 31 — Aggregate and complete the round**

- Apply selected weights to client deltas relative to the shared current global model.
- Validate the resulting model and save its checkpoint.
- Update reputations only after the current selection has used prior reputation.
- Record selection-validation metrics outside the fitness module.
- Persist enough state to resume with identical client, attack, and optimizer randomness.
- Extend the Phase D diagnostic artifact with actual search timings and evaluation counts; keep selection-validation measurements outside fitness and avoid double-counting nested timers.

**Gate:** An interrupted/resumed run matches an uninterrupted run within declared numerical tolerances.

**Profiling requirement:** Before scaling Phase G, measure one representative completed
NSGA-II round through actual candidate aggregation and model inference. Record hardware,
software, model, clients, panel size, configuration, evaluation/cache/failure counts,
candidate construction, inference, evolutionary overhead, total search time, and peak
memory. Separate initialization/warm-up costs. Publish projected matrix cost and its
assumptions. Sorting-only or synthetic-objective timings do not satisfy this requirement.

### Phase G — Run the research experiments

**Step 32 — Execute staged integration experiments**

Proceed in this order:

1. Small deterministic component tests.
2. Small-data clean centralized and FedAvg runs.
3. Clean real-data baselines.
4. Attack-harness and risk-module checks.
5. Clean NSGA-II integration.
6. One controlled poisoning scenario.
7. Full experimental matrix.

Do not require the proposed method to outperform baselines as a software acceptance criterion. Correctly measured negative results remain valid research outcomes.

This sequence reuses the clean evidence established in D and the measured search path
from F. Each real-data condition requires its own applicable training-ready PASS and
clean-learning review; fixture results do not replace either requirement.

**Step 33 — Freeze the main comparison matrix**

**Prerequisite:** The actual Phase F search profile and projected compute costs are
recorded, and development-based budget revisions are explicit. Full evolutionary
profiling is not a prerequisite for Phase D. Preserve matched budgets and disclose
compute accounting; do not silently alter budgets between methods or seeds.

Use:

- Both binary and multiclass tasks.
- Near-IID, α = 0.5, α = 0.1, and specialist partitions.
- Clean runs and separately reported attacks.
- Five matched seeds.
- Identical preprocessing, initializations, participation schedules, and attack manifests across methods.

Run the complete attack matrix initially at α = 0.5 and 20% malicious clients. Expand heterogeneity and malicious fractions through the sensitivity plan rather than immediately creating an impractical full Cartesian product.

Record actual submitted deltas independently for each method: trajectories diverge after aggregation, so later client updates cannot generally be reused across methods.

**Step 34 — Run ablations**

Required ablations:

- Remove reputation.
- Remove current-risk weighting.
- Remove expertise-informed initialization.
- Use uniform/random-only chromosome initialization.
- Remove each risk-estimator component separately.
- Remove each fitness objective separately.
- Replace NSGA-II with fixed-coefficient weighting.
- Replace NSGA-II with random search using the same candidate-evaluation budget.
- Use strict feature removal.
- Compare uncapped and capped weights.
- Compare shared clipping across methods as a separately labelled defense condition.

These distinguish benefits from scoring, constraints, trusted-data access, and search.

**Step 35 — Run sensitivity and scalability studies**

Vary one factor at a time around the frozen primary configuration:

- Malicious fraction: 0%, 10%, 20%, 30%, 40%.
- Poisoned-data fraction: 10%, 30%, 50%.
- Update scaling: 2, 5, 10.
- Clients: 10, 20, 50.
- Participation: 50% and 100%.
- Local epochs: 1, 3, 5.
- Reputation memory: 0.5, 0.9, 0.99.
- Trusted-panel size: quarter, half, and full reference panel.
- Population: 12, 24, 48.
- Offspring generations: 4, 8, 16.

Also examine missing trusted classes, late-onset poisoning, intermittent attacks, and honest specialist rejection.

Report infeasible configurations rather than weakening data or aggregation constraints silently.

**Step 36 — Evaluate independent-capture generalization**

- Rebuild preprocessing and client assignments within each Protocol B training fold.
- Keep held-out captures unavailable to quality scoring, tuning, and feature fitting.
- Report results by held-out source and attack type.
- Separate binary unseen-attack detection from supported multiclass performance.

Avoid pooling these results into the closed-set table without identifying the protocol change.

**Step 37 — Replicate on CICIoT2023**

- Register its official data, provenance, feature definitions, and label taxonomy.
- Build dataset-specific preprocessing and leakage-resistant partitions.
- Preserve the aggregation algorithm, risk/reputation definitions, optimization budget, and evaluation structure.
- Train fresh models with dataset-appropriate input/output dimensions.
- Repeat clean, targeted label-flipping, and update-scaling comparisons across five matched seeds, followed by the principal ablations.

CICIoT2023 supplies different attack categories and extracted features. This stage tests replication of the method, not direct transfer of an Edge-IIoT model. [Official CICIoT2023 description](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

### Phase H — Final evaluation and reproducibility

**Step 38 — Freeze the protocol before accessing final tests**

- Freeze dataset versions, manifests, preprocessing, methods, attack settings, checkpoint rules, thresholds, and reporting metrics.
- Select checkpoints using selection validation only.
- Record the exact set of checkpoints eligible for final evaluation.
- Any later methodological revision becomes a new explicitly labelled experiment.

**Step 39 — Generate final metrics and confidence intervals**

Report:

- Binary: attack F1, macro-F1, PR-AUC, ROC-AUC, recall, FPR, FNR, and confusion matrices.
- Multiclass: macro-F1, weighted-F1, balanced accuracy, and per-class precision/recall/F1/support.
- Robustness: clean-to-poisoned degradation, target-class miss rate, malicious aggregate weight, and honest-client suppression.
- Risk estimation: ranking metrics and, where thresholds are frozen, identification precision/recall and false-positive rate.
- Systems: client training time, server scoring time, NSGA-II time, candidate evaluations, memory, rounds, and transmitted bytes.

Use paired seed-level differences and 95% intervals for method comparisons. Use session/capture bootstrap intervals for observational uncertainty where enough independent groups exist. Report that seed intervals are conditional on the fixed global split.

**Step 40 — Produce an auditable research package**

Include:

- Original-source inventory and hashes.
- Provenance and repair/quarantine reports.
- Global/client/local manifests.
- Preprocessing artifacts and feature variants.
- Common initializations.
- Attack manifests.
- Per-round updates or reproducible references, scores, reputations, and weights.
- Pareto fronts and optimization budgets.
- Seed-level results and statistical reports.
- Resumption instructions and environment specification.

**Completion criterion:** Another researcher can reconstruct partitions, reproduce a matched comparison, and trace every reported result to its configuration and checkpoint.

**Step 41 — Perform the VM demonstration**

- Deploy the frozen coordinator and three to five clients.
- Reuse persisted partitions and model/preprocessing versions.
- Add authentication, transport security, round identity checks, timeouts, and retry handling.
- Measure actual end-to-end latency and bytes.
- Describe this as a systems demonstration; retain repeated simulation and external replication as the research evidence.

## 4. Mandatory verification gates

| Gate | Required evidence |
|---|---|
| Dataset integrity | Semantic schema checks, source relationships, label mappings, and timestamp feasibility |
| Leakage control | No duplicated observation or protected session crosses forbidden boundaries |
| Preprocessing | Fitting rows belong only to actual client training |
| Client construction | Deterministic ownership, valid support, documented heterogeneity |
| Real-data model entry/resume | Fresh source-verified Phase C PASS matching dataset, task, fold, and scenario |
| Baseline correctness | Checkable centralized/FedAvg behavior and reproducible Phase D methods, including FLTrust |
| Clean-learning review | Applicable condition evidence; unexplained behavior blocks advancement; explained negative findings carry limitations |
| Attack correctness | Controlled scope, reproducibility, no evaluation-data contamination |
| Trust correctness | No malicious-identity access; valid support handling; correct temporal ordering |
| Evolutionary correctness | Feasible repair, known Pareto fronts, crowding edge cases, elitist survival |
| Round integration | Correct delta aggregation, failure behavior, deterministic resume |
| Diagnostics and scaling | Consistent role-aware metrics; actual Phase F evaluation profile and recorded budget before the full matrix |
| Experimental fairness | Matched data/seeds, stated information access, separate compute accounting |
| Final evaluation | Frozen checkpoints and protocol, paired results, defensible uncertainty |

## 5. Execution assumptions and stopping conditions

- Begin with single-machine simulation and streaming/chunked data processing. The inspected machine reports 32 GiB RAM and a six-core Ryzen CPU; GPU availability and achievable runtime remain unverified.
- Profile representative training/evaluation costs in D and one completed actual NSGA-II round in F before scheduling the full matrix. Publish projected and actual compute costs, with warm-up costs separate.
- Population 24 and eight offspring generations remain recorded development defaults. Change budgets only through explicit development-based revisions before main comparisons; final-test scores must not determine them.
- Treat trusted-data cleanliness as an explicit experimental assumption; test reduced coverage and distribution mismatch.
- Do not force all fifteen classes into every partition if verified source structure makes that impossible.
- If provenance remains unresolved for a class, report that limitation and withhold the corresponding generalization claim.
- Keep server-visible individual updates as an explicit assumption. Secure aggregation and differential privacy require separate protocols.
- Accept the research outcome whether NSGA-II wins, ties, or loses. Completion means a correct, reproducible comparison with documented limitations.
