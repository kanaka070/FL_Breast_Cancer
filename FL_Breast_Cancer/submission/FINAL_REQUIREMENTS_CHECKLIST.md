# Final requirements and evidence checklist

Status meanings: **Complete** means implemented and supported by inspection/tests or recorded experiment outputs; **Partial** means code or historical evidence exists but the requested claim is not fully verified; **Pending** is not represented as completed.

| Requirement | Status | Implementation / evidence | Qualification |
|---|---|---|---|
| Preserve existing project, WDBC data, saved baselines, and historical metrics | Complete | Original root project; SHA-256 comparison before/after packaging and final verification | Packaging copies artifacts; it does not retrain |
| Centralized Logistic Regression | Complete (saved result) | `centralized.py`; `saved_models/centralized_logistic_regression.pkl`; `metrics/final_three_model_comparison.csv` | Saved historical held-out evaluation |
| Three-hospital FedAvg Logistic Regression | Complete (saved result) | `server.py`, `model.py`; saved FedAvg model; historical three-model CSV | Hospitals are simulated partitions |
| Differentially private federated Logistic Regression | Partial | `dp_federated.py`, `dp_federated_fixed.py`, `dp_accountant.py`; saved DP row/history | Experimental mechanism; end-to-end DP is unproven and not claimed |
| Actual three-model comparison and model approval | Complete | `dashboard.py`; saved comparison CSV; explicit approval and persisted active pointer in `retraining.py`/`inference.py` | The dashboard's workflow controls are not authentication |
| Prediction with correct approved model and scaler | Complete | `inference.py`; model/scaler mapping; synthetic-feature workflow tests | Prediction is never a ground-truth label |
| Validate canonical 30-feature CSV and reject malformed input | Complete | `feature_schema.py`, `inference.py`; workflow tests | Exact feature order and finite numeric checks |
| Candidate retraining requires confirmed labels; candidate separate from active | Complete | `retraining.py`, `dashboard.py`; tests for confirmation, duplicate handling, explicit promotion | Requires user attestation; no clinical label verification service |
| Block stale candidate approval | Complete | Candidate evaluation captures active state; approval rechecks it; `tests/test_workflow.py` | Local single-user workflow, not distributed concurrency control |
| Stratified 5-fold CV | Complete (all three Logistic Regression approaches) | `research_outputs/verified/cross_validation_summary.csv`, `federated_cross_validation_summary.csv` | 455 training records only; fold-local scaler; final test excluded |
| Stratified 10-fold CV | Complete (all three approaches) | Same CV summary outputs | Fixed settings; final test excluded |
| Unstratified 10-fold CV | Complete (all three approaches) | Same CV summary outputs | Shuffled folds; class variability reflected in fold SD |
| Metrics incl. malignant recall, specificity, F1, ROC-AUC, confusion/errors, fit time and fold variability | Complete for all CV approaches | `cross_validation_folds.csv`, `federated_cross_validation_folds.csv` plus summaries | Historical primary baseline training-time comparison remains unavailable |
| IID and Non-IID experiment | Complete (executed) | `iid_noniid_results.csv`, `iid_noniid_training_history.csv` | Further 91-record validation split; not final test results |
| Trust-based aggregation | Complete (executed) | `trust_weights.csv`, `trust_aggregation_test_metrics.csv`, `trust_aggregation_method.json` | Local validation balanced accuracy heuristic; one held-out evaluation; not DP |
| SHAP explainability | Pending | No SHAP output is included | SHAP package absent in environment; no explanation values fabricated |
| Formal end-to-end DP guarantee/accountant correctness | Blocked / unproven | Mechanism and assumptions documented in `RESEARCH_REPORT.md` | Coordinator sees pre-noise client contributions; pooled scaler outside accounting; assumptions need redesign and independent proof |
| Full source/test/data/model/reproducible-output package | Complete | `submission/` created by `prepare_submission.py` | Excludes virtualenv, caches, and unrelated historical clutter |
| Research report and presentation | Complete | `submission/RESEARCH_REPORT.md`; `submission/PRESENTATION.html` | HTML slides; no dashboard screenshot file was available to package |
| Tests, compilation, dashboard smoke test | Complete | Root and packaged test suite, compileall, Streamlit smoke check across all six tabs | See `VERIFICATION_RECORD.md`; no baseline training or clinical-patient test data were used |

## Verified experiment results (rounded)

- Saved held-out baseline accuracy: centralized **0.9649**, FedAvg **0.9737**, experimental DP-FedAvg **0.9474**.
- Stratified 5-fold accuracy **0.9736 ± 0.0167**; malignant recall **0.9529 ± 0.0446**.
- Stratified 10-fold accuracy **0.9758 ± 0.0241**; malignant recall **0.9588 ± 0.0484**.
- Unstratified 10-fold accuracy **0.9736 ± 0.0308**; malignant recall **0.9449 ± 0.0754**.
- FedAvg CV accuracy: stratified 5-fold **0.9714 ± 0.0060**, stratified 10-fold **0.9758 ± 0.0243**, unstratified 10-fold **0.9713 ± 0.0315**.
- Experimental DP-FedAvg CV accuracy: stratified 5-fold **0.9341 ± 0.0233**, stratified 10-fold **0.9317 ± 0.0382**, unstratified 10-fold **0.9339 ± 0.0499**; this does not prove DP.
- IID and Non-IID validation accuracy were both **0.9780**, recall **0.9412** (91 validation records each); identical results in this run do not establish equivalence.
- Trust-weighted test accuracy **0.9737**, recall **0.9286**, confusion TN=72, FP=0, FN=3, TP=39.

None of the CV or IID/Non-IID results evaluated the held-out test partition. The trust evaluation and saved primary comparison use that fixed partition; these evaluations are reported as separate experiments, not pooled.

## Protected source artifacts

The public source dataset, saved model files, and pre-existing historical metric CSVs are copied unchanged into the curated package. Final verification records their hashes before and after packaging. New experiments write only to `research_outputs/verified/`; the packaging script is copy-only and refuses to overwrite an existing output folder.
