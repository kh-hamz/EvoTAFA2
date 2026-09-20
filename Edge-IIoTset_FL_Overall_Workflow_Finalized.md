# Finalized Edge-IIoTset Federated-Learning Analysis and Workflow

## 1. Review conclusion

The original workflow is **directionally correct**, but it is not yet safe to execute as an experimental protocol. Its main sequence - clean the data, reserve global validation/test data, build non-IID clients, establish FedAvg, add adversaries, optimize aggregation, and then demonstrate the system on VMs - is appropriate.

The finalized workflow below corrects the points that could otherwise invalidate the results:

1. split by capture, flow, or contiguous time block rather than randomly by packet;
2. remove duplicates **before** any split;
3. fit every learned preprocessing operation on training data only;
4. run binary and 15-class classification as separate experiments;
5. keep the global test set out of model selection, trust scoring, and NSGA-II fitness;
6. constrain NSGA-II weights and predefine how one solution is chosen from its Pareto front;
7. do not trust client-reported quality metrics from potentially malicious clients;
8. compare the proposed method with both FedAvg and established robust aggregators;
9. repeat experiments across multiple seeds and report uncertainty;
10. describe label-based client partitions as **synthetic non-IID clients**, not as naturally occurring organizations or devices.

---

## 2. What is in the local archive

The archive contains three useful data representations:

| Archive component | Intended use | Recommendation |
|---|---|---|
| `Attack traffic/*.csv` and `Normal traffic/*/*.csv` | Source-specific packet-feature CSV files | Best option when capture provenance and leakage-resistant splitting are required |
| `Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv` | Large selected dataset for deep-learning IDS experiments | Primary source for the MLP/FL experiment, provided split provenance is reconstructed or time-blocked |
| `Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv` | Smaller selected/downsampled dataset for traditional ML | Use for smoke tests only; do not use it to claim performance under the original traffic prevalence |

Both selected CSVs have 63 columns: 61 candidate predictors plus `Attack_label` and `Attack_type`.

### 2.1 Empirical audit of the archived selected CSVs

The following counts were computed directly from the two archived CSV files:

| File | Rows | Benign | Attack | Classes |
|---|---:|---:|---:|---:|
| DNN dataset | 2,219,201 | 1,615,643 | 603,558 | 15 total: Normal plus 14 attack types |
| ML dataset | 157,800 | 24,301 | 133,499 | 15 total: Normal plus 14 attack types |

The DNN dataset has the following natural selected-data distribution:

| Class | Rows | Share |
|---|---:|---:|
| Normal | 1,615,643 | 72.803% |
| DDoS_UDP | 121,568 | 5.478% |
| DDoS_ICMP | 116,436 | 5.247% |
| SQL_injection | 51,203 | 2.307% |
| Password | 50,153 | 2.260% |
| Vulnerability_scanner | 50,110 | 2.258% |
| DDoS_TCP | 50,062 | 2.256% |
| DDoS_HTTP | 49,911 | 2.249% |
| Uploading | 37,634 | 1.696% |
| Backdoor | 24,862 | 1.120% |
| Port_Scanning | 22,564 | 1.017% |
| XSS | 15,915 | 0.717% |
| Ransomware | 10,925 | 0.492% |
| MITM | 1,214 | 0.055% |
| Fingerprinting | 1,001 | 0.045% |

This imbalance is material. Accuracy alone will be misleading, and random client partitioning can easily leave rare classes absent from most clients and evaluation folds. The small ML dataset also changes class prevalence substantially, so results from it are not interchangeable with results from the DNN file.

The archived README describes 13 attack documents but lists 14 attack CSVs; the actual selected datasets contain 14 attack labels. Experimental code should derive and validate the label set from the data rather than hard-code the README's stated count.

---

## 3. Final experimental decisions

### 3.1 Primary dataset

Use `DNN-EdgeIIoT-dataset.csv` for the reported MLP and FL results. Use the smaller ML CSV only to validate code paths quickly.

For the strongest leakage control, reconstruct a master table from the source traffic CSVs and add these columns before concatenation:

- `source_file`;
- `capture_id`;
- `time_block_id`;
- a stable `row_id`.

These fields are partitioning metadata only and must not be model inputs. If reconstruction is not practical, retain `frame.time` long enough to form contiguous time blocks in the DNN CSV, split using those blocks, and only then remove `frame.time` from predictors.

### 3.2 Learning tasks

Run two independent experiments with the same partition manifest:

1. **Binary IDS:** target is `Attack_label` (`0 = Normal`, `1 = Attack`).
2. **Multiclass IDS:** target is `Attack_type` with a fixed 15-class mapping.

Do not supply either label column as a predictor. For the binary experiment, also exclude `Attack_type`; for the multiclass experiment, also exclude `Attack_label`. A hierarchical binary-then-multiclass IDS may be studied later, but it is a different model and must be reported separately.

### 3.3 Fair comparison rule

Centralized, FedAvg, robust-aggregation, and NSGA-II runs must use:

- the same global train/validation/test manifest;
- the same feature schema and training-fitted preprocessing;
- the same MLP architecture, loss family, and initialization distribution;
- the same client availability schedule where applicable;
- comparable tuning budgets;
- multiple matched random seeds.

The centralized model must not supply pretrained weights to one FL method unless warm-starting is explicitly evaluated as a separate ablation. FedAvg and the proposed method should begin from identical initial weights for each matched seed.

---

## 4. Leakage-resistant data pipeline

### Step 1 - Register immutable inputs

Record file paths, sizes, cryptographic hashes, row counts, column names, label counts, and software versions. Never overwrite the source CSVs.

### Step 2 - Validate schema and labels

Verify that:

- every row has 63 fields;
- `Attack_label` is either 0 or 1;
- `Attack_type == Normal` if and only if `Attack_label == 0`;
- all expected attack labels use one canonical spelling;
- numeric columns contain no non-finite values;
- protocol-inapplicable zero sentinels are distinguished from genuinely missing values where necessary.

The row-count audit found no malformed-width rows and found label totals consistent at the file level, but the full cleaning run must still persist its validation report.

### Step 3 - Remove exact duplicates before splitting

Count and remove exact duplicate rows before train/validation/test assignment. Also report duplicate counts after excluding identifiers, because two rows can become identical after leakage-prone fields are removed. Keep both pre-cleaning and post-cleaning class counts; deduplication can change class prevalence.

Do not independently deduplicate each final partition: that can leave the same observation in both training and test data.

### Step 4 - Build group-aware partitions

Random packet-level splitting is not acceptable for the primary result. Neighboring packets from the same capture or flow are highly correlated and can make test performance optimistic.

Preferred grouping order:

1. capture/source file plus contiguous time window;
2. flow/session identifier within capture;
3. contiguous `frame.time` blocks when stronger provenance is unavailable.

Assign entire groups to partitions, with a purge gap between adjacent temporal blocks when possible. A practical target is 70% global training pool, 15% global validation, and 15% global test, but group integrity takes priority over exact percentages. Confirm that every evaluable class occurs in validation and test; if a rare class cannot be represented without breaking group isolation, disclose that limitation rather than silently reverting to random packet splitting.

### Step 5 - Remove identifiers from predictors

At minimum, evaluate removal of the archive README's leakage/high-cardinality fields:

```text
frame.time
ip.src_host
ip.dst_host
arp.src.proto_ipv4
arp.dst.proto_ipv4
http.file_data
http.request.full_uri
icmp.transmit_timestamp
http.request.uri.query
tcp.options
tcp.payload
tcp.srcport
tcp.dstport
udp.port
mqtt.msg
```

This list is a reproducibility baseline, not proof that every remaining field is safe. Run a strict-generalization ablation for capture-specific counters and checksums such as raw acknowledgements, sequence values, checksums, and stream identifiers. Report the final retained feature list.

### Step 6 - Fit preprocessing without leakage

Fit learned transformations on the **global training pool only**, never on global validation or test data. Recommended processing is:

- one-hot encode true categorical fields with an explicit unknown category policy;
- standardize continuous numerical fields using training-only statistics;
- leave binary indicators unscaled unless the implementation requires otherwise;
- preserve a fixed output column order;
- serialize the fitted transformer and label mapping.

In a simulation, one transformer fitted on the pooled training partition is acceptable and gives every client an identical representation. It must be disclosed as server-known preprocessing. For a strict privacy-preserving deployment, obtain preprocessing parameters from public reference data or federated statistics; fitting centrally on pooled private client data would contradict the deployment's privacy premise.

### Step 7 - Treat imbalance deliberately

Choose the imbalance strategy using training data only: class-weighted cross-entropy, focal loss, or balanced local mini-batches are reasonable options. Do not rebalance global validation or test data for the primary prevalence-sensitive result. If a balanced test set is additionally reported, label it as a secondary diagnostic set.

---

## 5. Federated client construction

### 5.1 Partition unit

Partition only the global training pool. Assign complete capture/flow/time groups to a single client so that correlated records are not distributed across clients. No row may belong to more than one client.

### 5.2 Recommended client scenarios

Use at least three scenarios:

| Scenario | Purpose |
|---|---|
| Near-IID | Sanity check and upper-bound behavior for FedAvg |
| Dirichlet label skew | Controlled non-IID benchmark, for example several predefined values of alpha |
| Pathological specialists | Clients dominated by selected attacks, used to study expertise and aggregation failure |

For each scenario, predefine the number of clients, client sample-size distribution, minimum samples, minimum class support, and client participation rate. Save the client assignment manifest so every aggregator receives exactly the same data.

Label-driven partitioning creates synthetic statistical heterogeneity. It does not reproduce feature drift, device drift, organization boundaries, or real network ownership. If those claims are important, construct clients by sensor, source network, capture period, or traffic source and analyze the resulting label distributions.

### 5.3 Client-local validation

Within each client's assigned groups, create group-aware local training and local validation partitions. Local validation is suitable for early stopping and descriptive local performance, but it is not a trustworthy aggregation signal when clients may be malicious: a malicious client can fabricate its reported score.

For server-side client quality and NSGA-II objectives, use server-observable update statistics and a small trusted **global validation/calibration set**. This set comes only from the global validation partition and must remain distinct from the final test set.

### 5.4 Required client metadata

Persist, per client:

- client ID and partition seed;
- group IDs and row IDs;
- training and validation sample counts;
- binary and per-class counts;
- benign/attack ratio;
- preprocessing/model version;
- participation history;
- update norm, cosine similarity, and clipping status;
- bytes sent/received and training time.

Do not publish raw row identifiers or sensitive addresses outside the controlled experiment.

---

## 6. Baseline models and FL cycle

### 6.1 Centralized baseline

Train an MLP on the complete global training pool, tune/early-stop on global validation, and evaluate once on global test after the configuration is frozen. Also report a trivial majority baseline and, if feasible, a standard non-neural baseline.

### 6.2 FedAvg baseline

At round `t`, the server samples eligible clients, broadcasts the same global weights, and each selected client performs the same configured local optimizer schedule. For selected client set `S_t`, standard sample-weighted FedAvg is:

```text
w_(t+1) = sum over k in S_t of [n_k / sum_j(n_j)] * w_(t+1,k)
```

Log the sampled clients, successful clients, local epochs/steps, effective sample counts, update norms, aggregation weights, validation metrics, and communication volume for every round.

### 6.3 Additional baselines

For non-IID claims, include FedProx or another recognized heterogeneity baseline. For poisoning claims, compare against appropriate robust methods such as coordinate-wise median, trimmed mean, or another method compatible with the threat model. Comparing only NSGA-II with FedAvg is insufficient to establish poisoning robustness.

### 6.4 Stopping and tuning

Use global validation for round selection and hyperparameter tuning. Predetermine the maximum rounds and patience. The global test set must not be inspected per round or used to select the best checkpoint.

---

## 7. Poisoning and unreliable-client protocol

Begin with a completely clean run. After it is reproducible, define each adversarial experiment with:

- malicious-client fraction;
- whether adversaries are always selected or sampled normally;
- attack start/end rounds;
- adversary knowledge and collusion assumptions;
- poisoned-data fraction or update scaling factor;
- whether attacks target availability or a particular class;
- whether the server knows the malicious identities for evaluation only.

Recommended distinct scenarios are label flipping, targeted label flipping, sign flipping, update scaling, and additive-noise updates. Do not combine them into a single undefined "poisoned client" condition.

Update clipping should be consistently applied to all compared aggregators when it is part of the defense. Detection quality should include malicious-client identification precision/recall when the proposed method claims to detect adversaries, not merely global model accuracy.

---

## 8. Correct use of NSGA-II for aggregation

### 8.1 Decision variables and constraints

Let the NSGA-II chromosome be the selected-client aggregation vector:

```text
a = (a_1, ..., a_m), where a_k >= 0 and sum(a_k) = 1
```

Optionally impose an upper bound per client and/or a top-`K` communication constraint. Repair or reject infeasible chromosomes deterministically. Aggregate updates, rather than unrelated absolute model states, unless all implementations use the same defined convention.

### 8.2 Fitness data

Never calculate NSGA-II fitness on the global test set. Fitness may use:

- performance of the candidate aggregate on trusted global validation data;
- false-positive rate on trusted benign validation records;
- server-observed update anomaly or poisoning-risk score;
- deviation from stable historical behavior;
- measured or estimated communication cost.

Client-reported local accuracy, expertise, or trust is not reliable under a Byzantine threat model. If such values are used in a benign-only experiment, clearly state that they are assumed honest.

### 8.3 Objective definition

Use a small set of non-redundant, normalized objectives. One defensible example is:

```text
minimize 1 - macro_F1(validation)
minimize false_positive_rate(validation)
minimize update_risk(a)
minimize communication_cost(a)
```

Do not optimize accuracy and F1 as if they were independent without checking redundancy. Define exactly how per-attack expertise is calculated and ensure rare classes have sufficient validation support; otherwise the optimizer will chase noisy estimates.

### 8.4 Pareto-front selection

NSGA-II returns a set of non-dominated solutions, not one automatic answer. Before running the test experiment, specify the final selection rule, for example:

- normalized knee-point distance to the ideal point;
- a fixed priority rule subject to a maximum FPR;
- hypervolume contribution with a fixed reference point.

The rule and all normalization bounds must be fitted or chosen using training/validation information only. Log the entire Pareto front and the selected solution each round.

### 8.5 Computational fairness

Evaluating many chromosomes on global validation gives NSGA-II substantially more server-side computation than FedAvg. Report population size, generations, validation evaluations, runtime, energy or compute proxy, and communication overhead. Keep communication accounting separate from server optimization compute.

---

## 9. Evaluation plan

### 9.1 Primary comparisons

For each matched seed and scenario, compare:

1. centralized MLP;
2. local-only client models, where useful;
3. FedAvg;
4. non-IID baseline such as FedProx;
5. poisoning-robust baseline(s);
6. proposed NSGA-II aggregation.

### 9.2 Required metrics

For binary IDS, report:

- precision, recall, F1, PR-AUC, and ROC-AUC;
- false-positive rate `FP / (FP + TN)`;
- false-negative rate;
- confusion matrix;
- calibration if output probabilities will drive alarms.

For multiclass IDS, report:

- macro-F1, weighted-F1, and balanced accuracy;
- per-class precision, recall, F1, and support;
- full confusion matrix;
- one-vs-rest PR-AUC when support permits;
- benign false-positive rate and per-attack miss rate.

For FL behavior, report:

- best-validation and final-round performance;
- rounds/time to a predefined threshold;
- total bytes and client participation;
- performance variance across clients and classes;
- robustness as malicious fraction increases;
- NSGA-II server compute and Pareto stability.

Because MITM and Fingerprinting are extremely rare in the DNN CSV, include confidence intervals or bootstrap intervals and avoid strong conclusions from a handful of test examples.

### 9.3 Statistical reporting

Use at least five matched seeds if resources allow. Report mean, standard deviation, and 95% confidence intervals. Use paired comparisons across identical partitions, initializations, and participation schedules. Freeze the complete protocol before final test evaluation.

---

## 10. VM demonstration

The VM stage is a systems demonstration, not a substitute for repeated simulation experiments.

Recommended topology:

```text
Host or server VM
  - FL coordinator
  - model registry and experiment log
  - FedAvg / robust / NSGA-II aggregator
  - trusted validation/calibration service

Three to five client VMs
  - one persisted client partition per VM
  - local preprocessing transform
  - local training and validation
  - authenticated update transport
```

Use TLS, client authentication, timeouts, retry rules, and version checks. Measure end-to-end round duration and transmitted bytes. FL alone does not guarantee privacy: gradients or model updates can leak information. If secure aggregation or differential privacy is added, evaluate it as a separate configuration. Note that secure aggregation can conflict with defenses that require inspection or individual weighting of client updates.

---

## 11. Final executable sequence

```text
1. Hash and inventory archived source files
2. Select DNN data for reported MLP/FL experiments
3. Attach source/capture/time-block metadata
4. Validate schema and binary/multiclass label consistency
5. Quantify and remove duplicates before splitting
6. Create immutable group-aware train/validation/test manifests
7. Remove labels and leakage-prone predictors
8. Fit preprocessing on the global training pool only
9. Save transformer, feature order, and label mapping
10. Create matched near-IID and synthetic non-IID client manifests
11. Create group-aware local train/validation partitions
12. Train centralized and trivial baselines
13. Run clean FedAvg and heterogeneity baselines
14. Run clean NSGA-II with validation-only objectives
15. Freeze hyperparameters and Pareto-solution rule
16. Introduce separately defined poisoning scenarios
17. Compare FedAvg, robust baselines, and NSGA-II
18. Repeat matched experiments across seeds
19. Evaluate frozen checkpoints once on untouched global test data
20. Report accuracy-independent, per-class, robustness, compute, and communication metrics
21. Deploy the frozen protocol on three to five VMs
22. Archive manifests, configs, logs, code revision, and model artifacts
```

---

## 12. Minimum reproducibility artifacts

The completed experiment should produce:

- `dataset_manifest.json` with hashes, schema, and counts;
- `split_manifest.parquet` or CSV with row/group/partition IDs;
- `client_manifest.json` with class counts and group assignments;
- serialized preprocessing transformer and feature-order file;
- separate binary and multiclass label maps;
- one version-controlled configuration per scenario;
- per-round client and server logs;
- poisoning-scenario manifests;
- Pareto fronts and selected NSGA-II weights;
- aggregate result tables with seed-level observations;
- final test report generated only after model selection is frozen.

---

## 13. Sources in the archive

- [Edge-IIoTset dataset README](archive/Readme.txt)
- [Edge-IIoTset dataset paper archived locally](archive/Edge_IIoTset__DatasetFL.pdf)
- Ferrag, M. A., Friha, O., Hamouda, D., Maglaras, L., and Janicke, H. *Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning*. DOI: `10.36227/techrxiv.18857336.v1`.

The row counts and class distributions in this document were independently computed from the archived `DNN-EdgeIIoT-dataset.csv` and `ML-EdgeIIoT-dataset.csv`; they should be regenerated by the final pipeline and stored with the run artifacts.
