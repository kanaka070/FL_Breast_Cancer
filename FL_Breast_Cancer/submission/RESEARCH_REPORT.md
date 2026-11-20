# Privacy-Preserving Federated Learning Framework for Collaborative Breast Cancer Diagnosis

## A Comparative Study of Centralized Learning, Federated Averaging, and Differential Privacy

**Research report — reproducible project outputs, 2026**

## Abstract

This project compares centralized Logistic Regression, three-client federated averaging (FedAvg), and an experimental differentially private federated Logistic Regression approach on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The dashboard exposes saved evaluation results, explicit model approval, prediction with the approved model's preprocessing, and a confirmed-label-only candidate retraining workflow. Three simulated hospitals are used; they are partitions in one process rather than independent clinical institutions.

The saved primary comparison reports held-out accuracy of 0.9649 for centralized Logistic Regression, 0.9737 for FedAvg, and 0.9474 for the experimental DP-FedAvg model. Additional reproducible evaluations were run separately from saved artifacts: three cross-validation protocols for each approach on the training partition, a training/validation IID versus Non-IID comparison, and a trust-weighted aggregation experiment. Across the three protocols, mean accuracy ranged from 0.9736 to 0.9758 for centralized Logistic Regression, 0.9713 to 0.9758 for FedAvg, and 0.9317 to 0.9341 for experimental DP-FedAvg. IID and Non-IID validation results were identical in this run; the trust-weighted test metrics were also very close to the saved FedAvg result. These are results from this dataset and run, not evidence of generalization to another population.

The DP implementation clips patient gradients and adds Gaussian noise, but the coordinator can see pre-noise clipped client contributions, and pooled data-dependent scaling is not included in the accountant. Its accountant output is conditional on assumptions not established end-to-end. Consequently, the DP model is reported as an experimental privacy-motivated approach, not as providing a verified end-to-end differential privacy guarantee. The project is a research prototype and must not be used for clinical diagnosis.

## 1. Problem statement

Hospitals may wish to train a shared classifier without pooling their records. This project demonstrates centralized training as a reference, coefficient aggregation across simulated clients as a federated alternative, and clipping plus Gaussian noise as an experimental privacy mechanism. The comparison asks how these methods perform on a common WDBC split and what additional workflow controls are needed to inspect, approve, and use a model.

This prototype does not establish that real hospitals can securely collaborate: the clients and coordinator execute within one application, there is no authentication or transport/security boundary, and the DP threat model is not fully satisfied.

## 2. Objectives

1. Preserve and compare the saved centralized, FedAvg, and DP-FedAvg Logistic Regression baselines.
2. Evaluate fixed Logistic Regression using stratified 5-fold, stratified 10-fold, and unstratified 10-fold cross-validation without using the held-out test set for model selection.
3. Compare IID and documented Non-IID simulated-hospital partitions.
4. Evaluate trust-based client weighting as a distinct aggregation heuristic.
5. Provide explicit candidate approval, approved-model inference, and guarded retraining on explicitly confirmed labels.
6. State what the privacy implementation and accountant do—and do not—establish.

## 3. Background and literature

Federated averaging combines locally computed model updates instead of directly collecting clients' training examples [2]. Differential privacy formalizes a bound on how much a mechanism's output can change when an individual's record is changed [3]. The DP-SGD literature combines per-example gradient clipping with Gaussian noise and analyzes composition [4]. Rényi Differential Privacy (RDP) is one accounting framework for composing mechanisms [5]. These concepts motivate this project but do not, by themselves, prove that this implementation or its complete data pipeline satisfies DP.

The WDBC data contain measurements computed from digitized fine-needle-aspirate images and describe cell-nucleus characteristics [1]. It is a benchmark dataset, not a representative prospective clinical cohort. The 30 numeric input features and diagnosis label are used here; the record identifier is not a model feature. SHAP is a model-agnostic framework for interpreting individual predictions [6], but SHAP was not executed in this project because the current environment lacks the SHAP package. No explanation values are claimed.

## 4. System architecture and workflow

The existing Streamlit application reads saved model-comparison rows and model/scaler artifacts. A user chooses a candidate in the comparison view and explicitly approves it. Approval updates a persisted active-model pointer. Prediction follows that pointer and loads the corresponding model and preprocessing; there is no implicit fallback to FedAvg. Input validation requires the 30 canonical WDBC feature names in order and rejects missing, nonnumeric, or non-finite data. Probabilities are shown when supported.

Retraining registration requires a CSV with the 30 ordered features and a diagnosis label plus explicit confirmation that the labels are verified ground truth. Predictions are not accepted as labels. Duplicate feature/diagnosis rows are deduplicated. Training writes a candidate separately, evaluates it, records the active-model state used during evaluation, and requires a second explicit approval to promote it. If the active model changed meanwhile, stale approval is blocked. The dashboard's admin controls are workflow checks, not an authentication or authorization boundary.

The federated experiments use three **simulated hospitals**. They are not independent organizations or isolated services. In the new trust experiment, each simulated client splits its local training partition into a stratified 75% local fit and 25% trust-validation subset. Client Logistic Regression coefficients and intercepts are aggregated in a common pooled-training scaler coordinate system.

## 5. Dataset and experimental methodology

### Dataset and split

The local `data/wdbc.data` contains the UCI WDBC benchmark (569 observations, 30 numeric measurements, binary benign/malignant diagnosis). The primary comparison uses the existing saved evaluation CSV; its underlying held-out partition contains 114 records. The separate experiment runner uses a reproducible stratified 80/20 split with seed 42: 455 records for training and 114 for the held-out test partition. The SHA-256 of the input file for the latest run is recorded in `research_outputs/verified/execution_metadata.json`.

### Cross-validation

The new CV experiment runs on the 455-record training partition only; the 114-record test partition is not evaluated in any CV fold. The model is a `StandardScaler` + Logistic Regression (`C=1`, `lbfgs`, `max_iter=1000`, seed 42) pipeline. Each fold fits its own scaler on that fold's training rows. Stratified 5-fold, stratified 10-fold, and shuffled unstratified 10-fold protocols are reported. Mean and sample standard deviation (`ddof=1`) are computed across folds. Fold-level confusion counts, fit times, and class metrics are in `cross_validation_folds.csv`.

The CV configuration is fixed rather than tuned on the held-out set. Accuracy, precision, malignant recall/sensitivity, specificity, F1, ROC-AUC, and fit-time means and standard deviations are reported. In an unstratified fold, class balance is not guaranteed; this is why malignant recall variability is also shown.

Federated CV uses the same 455-record development partition and fold assignments. For each fold, a scaler is fit only on that fold's training rows; three simulated clients are then created from those rows. FedAvg uses the project's sigmoid LogisticRegressionModel with 30 communication rounds and 5 local epochs. The experimental DP-FedAvg uses patient-gradient clipping at norm 1, 30 noisy global updates, and noise multiplier 2.0, with deterministic per-fold seeds. Fold-level confusion counts and all listed metrics are retained. The test partition is never used in any CV model selection or evaluation.

The DP-fold experiment evaluates the numerical behavior of the existing clipping/noise approach; it does not validate its accountant or establish a privacy guarantee. The same pre-noise client visibility and pooled-fold scaler concerns apply.

### IID and Non-IID

The separate IID/Non-IID runner partitions only the 455-record training data into three simulated clients, then uses a further training/validation split for evaluation. No held-out test score is reported for this experiment. It uses the existing LogisticRegressionModel, seed 42, 30 communication rounds, and 5 local epochs. IID partitioning is balanced by class across clients; Non-IID partitioning uses the project-defined label-skewed client allocation. The training history and validation summary are in the two `iid_noniid_*.csv` files. Hospital-level distributions can be inspected in those outputs and the project's existing hospital diagnostics; these partitions represent simulated hospitals.

### Trust-weighted aggregation

For simulated client `i`, trust `t_i` is local holdout balanced accuracy and `n_i` is its number of fit-partition examples. The normalized weight is:

`alpha_i = (n_i * t_i) / sum_j(n_j * t_j)`

The aggregated parameter vector is `w = sum_i(alpha_i * w_i)` and the intercept is `b = sum_i(alpha_i * b_i)`. All local coefficient vectors use the same StandardScaler fitted on pooled training-only rows. This is a separate heuristic experiment, not DP: trust validation uses local labels and the coordinator receives model parameters. It is evaluated once on the fixed 114-record held-out test partition.

### Saved primary baseline and DP evaluation

The primary three-model table is read from `metrics/final_three_model_comparison.csv`; it is a historical saved experiment and was not regenerated in this work. The loaded centralized artifact is a scaler/model pipeline with Logistic Regression `C=1`. The baseline CSV supplies the saved held-out comparison and confusion-derived counts. Comparable training times were not present for all historical model runs, so no such times are inferred.

The report does not rerun a training entry point that could overwrite primary artifacts. The DP performance row is the saved historical evaluation, not a newly established privacy guarantee.

## 6. Results

### 6.1 Saved three-model held-out comparison

Metrics below are the actual saved values for the common 114-record test partition. Malignant is the positive class. Confusion entries are TN, FP, FN, TP.

| Saved approach | Accuracy | Precision | Malignant recall | Specificity | F1 | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Centralized Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9861 | 0.9512 | 0.9960 | 71 | 1 | 3 | 39 |
| FedAvg Logistic Regression | 0.9737 | 1.0000 | 0.9286 | 1.0000 | 0.9630 | 0.9980 | 72 | 0 | 3 | 39 |
| Experimental DP-FedAvg Logistic Regression | 0.9474 | 0.9500 | 0.9048 | 0.9722 | 0.9268 | 0.9934 | 70 | 2 | 4 | 38 |

The score comparison is descriptive for one saved split. It does not establish statistical significance, clinical utility, or a privacy-performance tradeoff under a proven privacy budget. The DP label in this table describes the historical experimental implementation, not a verified formal guarantee.

### 6.2 Cross-validation on training rows

All values are mean ± sample standard deviation across folds. Each protocol uses the fixed Logistic Regression pipeline; `held_out_test_evaluated=False` in the generated summary.

| Protocol | Accuracy | Precision | Malignant recall | Specificity | F1 | ROC-AUC | Fit time (seconds) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stratified 5-fold | 0.9736 ± 0.0167 | 0.9771 ± 0.0313 | 0.9529 ± 0.0446 | 0.9860 ± 0.0192 | 0.9640 ± 0.0231 | 0.9958 ± 0.0052 | 0.00835 ± 0.00348 |
| Stratified 10-fold | 0.9758 ± 0.0241 | 0.9775 ± 0.0392 | 0.9588 ± 0.0484 | 0.9860 ± 0.0243 | 0.9672 ± 0.0326 | 0.9952 ± 0.0088 | 0.00651 ± 0.00048 |
| Unstratified 10-fold | 0.9736 ± 0.0308 | 0.9799 ± 0.0351 | 0.9449 ± 0.0754 | 0.9891 ± 0.0179 | 0.9614 ± 0.0557 | 0.9950 ± 0.0081 | 0.00621 ± 0.00060 |

Fit times are local `fit()` wall-clock measurements in this environment, not end-to-end federated communication or deployment latency.

### 6.3 Federated cross-validation

Federated rows are mean ± sample standard deviation across folds. These are additional training-partition results, not the saved primary test comparison. Fit times include local fitting and aggregation for FedAvg, or gradient, clipping, noise, and update computation for DP-FedAvg.

| Approach | Protocol | Accuracy | Precision | Malignant recall | Specificity | F1 | ROC-AUC | Fit time (seconds) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | Stratified 5-fold | 0.9714 ± 0.0060 | 0.9768 ± 0.0236 | 0.9471 ± 0.0322 | 0.9860 ± 0.0147 | 0.9611 ± 0.0087 | 0.9936 ± 0.0068 | 0.01004 ± 0.00124 |
| DP-FedAvg (experimental) | Stratified 5-fold | 0.9341 ± 0.0233 | 0.9153 ± 0.0561 | 0.9118 ± 0.0550 | 0.9474 ± 0.0372 | 0.9118 ± 0.0314 | 0.9818 ± 0.0143 | 0.00391 ± 0.00099 |
| FedAvg | Stratified 10-fold | 0.9758 ± 0.0243 | 0.9832 ± 0.0368 | 0.9529 ± 0.0541 | 0.9895 ± 0.0234 | 0.9668 ± 0.0333 | 0.9921 ± 0.0135 | 0.00961 ± 0.00109 |
| DP-FedAvg (experimental) | Stratified 10-fold | 0.9317 ± 0.0382 | 0.9175 ± 0.0752 | 0.9059 ± 0.0690 | 0.9473 ± 0.0530 | 0.9089 ± 0.0502 | 0.9807 ± 0.0251 | 0.00367 ± 0.00028 |
| FedAvg | Unstratified 10-fold | 0.9713 ± 0.0315 | 0.9747 ± 0.0356 | 0.9430 ± 0.0870 | 0.9856 ± 0.0188 | 0.9572 ± 0.0581 | 0.9932 ± 0.0109 | 0.00903 ± 0.00045 |
| DP-FedAvg (experimental) | Unstratified 10-fold | 0.9339 ± 0.0499 | 0.9220 ± 0.0681 | 0.9106 ± 0.0858 | 0.9480 ± 0.0459 | 0.9146 ± 0.0683 | 0.9795 ± 0.0203 | 0.00362 ± 0.00020 |

The test split was held out from all 50 federated CV fits (25 folds for each federated approach). The DP scores characterize this chosen implementation/configuration only and do not establish formal privacy.

### 6.4 IID and Non-IID validation

| Partition | Validation n | Accuracy | Precision | Malignant recall | Specificity | F1 | ROC-AUC | TN | FP | FN | TP | Fit time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IID | 91 | 0.9780 | 1.0000 | 0.9412 | 1.0000 | 0.9697 | 0.9990 | 57 | 0 | 2 | 32 | 0.01008 |
| Non-IID | 91 | 0.9780 | 1.0000 | 0.9412 | 1.0000 | 0.9697 | 0.9990 | 57 | 0 | 2 | 32 | 0.00926 |

This one seeded validation run produced the same displayed classification outcomes. That is an observation, not evidence that partitioning has no effect. Both values are validation results, not held-out-test results.

### 6.5 Trust-weighted aggregation

| Simulated client | Fit records | Local validation balanced accuracy (trust) | Normalized trust weight |
|---|---:|---:|---:|
| Simulated Hospital 1 | 115 | 0.9583 | 0.2469 |
| Simulated Hospital 2 | 145 | 1.0000 | 0.3249 |
| Simulated Hospital 3 | 195 | 0.9800 | 0.4282 |

Trust-weighted held-out test results: accuracy 0.9737, precision 1.0000, malignant recall 0.9286, specificity 1.0000, F1 0.9630, ROC-AUC 0.9987, and confusion counts TN=72, FP=0, FN=3, TP=39. The sample-size FedAvg and trust-weighted scores happened to be identical on this split, including ROC-AUC; this does not make the methods identical. The shared training elapsed time was 0.02352 seconds for the common client-training/aggregation run; it is repeated in the result rows and is not a separately isolated per-aggregation benchmark.

For context, sample-count-only weights for client sizes 115, 145, and 195 are approximately 0.2527, 0.3187, and 0.4286. This is a heuristic comparison on one split, not a robustness or fairness guarantee.

## 7. Differential privacy mechanism and unproven guarantees

The implemented DP training path clips patient-level gradient vectors to a configured norm and adds Gaussian noise to a client aggregate before the federated model update. At a high level, a clipped vector is:

`g_bar = g * min(1, C / ||g||_2)`

and a Gaussian perturbation has scale related to the clipping sensitivity and noise multiplier. The RDP accountant composes assumed Gaussian releases and converts an order-dependent RDP value to an `(epsilon, delta)` estimate. This description is not enough to certify privacy.

For replace-one adjacency with fixed dataset size `N`, the implementation assumes average-gradient sensitivity `Δ = 2C/N` and Gaussian standard deviation `σ = zΔ`, where `C` is the clip norm and `z` the noise multiplier. Its accountant computes `RDP_α = Kα/(2z²)` across `K` releases and converts using `ε = min_α [RDP_α + ln(1/δ)/(α-1)]` over its configured finite set of orders. These are the implemented accountant's assumptions and arithmetic, not proof that all observable operations satisfy those assumptions. Noise uses a seeded NumPy generator; secure or cryptographic randomness is not established.

The FedAvg update is a sample-count-weighted mean of client parameters, `w_(t+1) = Σ_i (n_i / N) w_i^(t+1)`, with the intercept aggregated using the same weights. The trust-weighted variant replaces `n_i/N` with the normalized `n_i t_i` heuristic above.

The following end-to-end conditions remain unproven or are violated by the current design:

1. **Observer visibility:** the simulated coordinator receives clipped client sums before noise; the individual client's unnoised contribution is therefore not hidden from that observer.
2. **Preprocessing:** the primary DP path uses a scaler fit on pooled training features. This data-dependent preprocessing is not included in the accounted mechanism.
3. **Accounting assumptions:** a formal bound requires a precise neighboring-dataset definition, sensitivity for the exact released quantity, valid noise calibration, all releases/composition, and a specified observer. The current estimate is conditional on these assumptions and is not independently validated as a deployment-wide bound.
4. **Randomness and system boundary:** the experiment does not establish a production-grade cryptographic randomness, secure aggregation, transport, access-control, or multi-party deployment threat model.
5. **Repeated use and releases:** no complete deployment accounting is shown for repeated retraining, model approval, inference, metrics, or other observable outputs.

The last row of the existing historical `metrics/dp_federated_history.csv` records an accountant estimate of `epsilon = 17.0065` at `delta = 1e-5`, noise multiplier 2.0, and 30 accounted rounds. This is the output of the project's configured RDP computation under its assumed Gaussian mechanism; it is **not a verified privacy bound for the complete implementation or observer**.

Accordingly, **no end-to-end differential privacy guarantee or certified epsilon is claimed**. The saved accountant history is retained as historical experimental output only. Resolving these gaps requires a defined threat model and mechanism redesign/verification, not a wording change.

## 8. Application safety and persistence

The dashboard's workflow is designed to keep candidate and active models separate:

- Saved baseline comparison uses actual metric rows and maps each model to its corresponding artifact/preprocessing.
- Approval is a distinct user action; the approved-model pointer is persisted locally and controls inference.
- Prediction input is checked for exact canonical feature order, numeric values, and finite values. The approved model's scaler or embedded pipeline is used, and prediction is never treated as ground truth.
- Retraining registration requires explicit verified-label confirmation, canonical features, and a diagnosis column. Duplicate records are rejected/deduplicated.
- Candidate artifacts are separate from the active artifact; candidate evaluation is required before promotion. Stale evaluation is rejected if the active model changed.
- Tests use synthetic fixtures and temporary directories; they do not use predictions as labels or overwrite saved baselines.

These are application-level workflow controls, not protection from a malicious local user. There is no authentication/authorization boundary or concurrent multi-user transaction service.

## 9. Limitations

- WDBC is a small, public benchmark and is not representative of a prospective clinical setting.
- Hospital partitions are simulated and have no independent institutional or network boundary.
- The primary historical baseline results are a single saved holdout evaluation; no confidence intervals or repeated-seed baseline distribution are reported.
- CV and IID/Non-IID results use fixed settings and limited runs; they do not support broad claims about all distributions.
- Federated CV results are for the code's current simulated-client procedures and fixed hyperparameters; they do not establish performance across other hospitals, cohort shifts, or implementations.
- Trust is local validation balanced accuracy and is not validated for fairness, robustness, or privacy.
- Historical training-time comparisons for all three baseline approaches are unavailable.
- SHAP was not run because SHAP is absent from the project environment. It is pending rather than fabricated.
- The current DP implementation/accountant does not prove end-to-end privacy for the stated observer and complete workflow.
- The app is a local research prototype with no clinical validation, security review, user authentication, or production deployment architecture.

## 10. Conclusion and future work

The current project supports an end-to-end demonstration of saved-model comparison, explicit approval, approved-model prediction, and guarded candidate retraining. The three primary approaches remain Logistic Regression-based. The saved comparison and newly generated CV, IID/Non-IID, and trust-weighted outputs provide reproducible evidence for the experiments that were actually executed. Scores are strong on this benchmark split but must not be interpreted as clinical performance.

The most important future work is a formal, reviewer-verified DP threat model and end-to-end mechanism: hide per-client pre-noise contributions (for example through an appropriately analyzed secure aggregation protocol), account for data-dependent preprocessing and every release, justify adjacency and sensitivity, use approved randomness, and validate the accountant independently. Further work can add repeated-seed/fold uncertainty for federated baselines, assess clinically relevant external cohorts under governance, and run SHAP in a pinned environment with verified compatibility.

## References

1. UCI Machine Learning Repository, **Breast Cancer Wisconsin (Diagnostic)** dataset, dataset 17. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
2. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” *Proceedings of AISTATS*, PMLR 54, 2017, pp. 1273–1282. https://proceedings.mlr.press/v54/mcmahan17a.html
3. C. Dwork and A. Roth, *The Algorithmic Foundations of Differential Privacy*, Foundations and Trends in Theoretical Computer Science, 9(3–4), 2014, pp. 211–407. https://doi.org/10.1561/0400000042
4. M. Abadi et al., “Deep Learning with Differential Privacy,” *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security*, 2016, pp. 308–318. https://doi.org/10.1145/2976749.2978318
5. I. Mironov, “Rényi Differential Privacy,” *2017 IEEE 30th Computer Security Foundations Symposium*, 2017, pp. 263–275. https://doi.org/10.1109/CSF.2017.11 (preprint: https://arxiv.org/abs/1702.07476)
6. S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” *Advances in Neural Information Processing Systems 30*, 2017. https://arxiv.org/abs/1705.07874

## Reproduction and artifact map

From the project root, run `python academic_experiments.py --output-dir research_outputs\verified`. Results are written to `research_outputs/verified/`; the saved baseline table remains `metrics/final_three_model_comparison.csv`. Run unit tests with `python -m unittest discover -s tests -v`. See `README.md` for dashboard startup instructions. The report's numeric tables are rounded for readability; the CSV files retain full precision.
