# Federated Breast Cancer Diagnosis

This Streamlit research prototype implements Centralized Logistic Regression, sample-count-weighted FedAvg, and experimental DP-FedAvg for the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The dashboard can run live evaluations and simulate three hospital clients. It is for demonstration and academic research only, not clinical diagnosis.

## Existing data and artifacts

- Dataset: `data/wdbc.data` (569 records, 30 numeric features)
- Baseline models and preprocessing: `saved_models/`
- Historical evaluation and experiment outputs: `metrics/`
- Three hospitals are simulated partitions in one process, not separate hospitals or isolated services.

Existing model files and historical metric CSVs are inputs to the dashboard. Opening the app does not start training. On-demand dashboard evaluations use temporary/in-memory results and do not overwrite the approved model, candidate artifacts, or historical metric files.

## Project structure

- `dashboard.py`, `dashboard_charts.py`, `app_services.py`, `live_evaluation.py`: authenticated Streamlit workflows, tenant-scoped local services, simulated client training, and on-demand evaluation.
- `inference.py`, `feature_schema.py`, `retraining.py`: feature validation, approved-model inference, and confirmed-label candidate workflow.
- `centralized.py`, `server.py`, `client.py`, `dp_federated.py`, `dp_training.py`, `dp_accountant.py`: existing Logistic Regression approaches and experimental accountant.
- `academic_experiments.py`: isolated CV, IID/Non-IID validation, and trust-weighting runs.
- `data/`, `saved_models/`, `metrics/`: existing public benchmark, preserved baselines/scalers, and historical results.
- `tests/`: automated service, workflow, and research tests; `research_outputs/verified/`: separate experimental CSV/JSON outputs.
- `submission/`: report, presentation, evidence checklist, verification record, and curated `project/` copy.

## Setup and run

Python 3.13 and the packages in `requirements.txt` are used by this project. In PowerShell from the project root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Before first launch, set a separate, unique password for each demo account. The existing local database upgrades its old shared-password accounts on first initialization; after that, changing environment variables does not change stored passwords. Do not set the deprecated shared `FL_DEMO_PASSWORD` variable.

```powershell
$env:FL_ADMIN_PASSWORD = "replace-with-a-unique-admin-password"
$env:FL_HOSPITAL1_PASSWORD = "replace-with-a-unique-hospital1-password"
$env:FL_HOSPITAL2_PASSWORD = "replace-with-a-unique-hospital2-password"
$env:FL_HOSPITAL3_PASSWORD = "replace-with-a-unique-hospital3-password"
streamlit run dashboard.py
```

The app is available at the local URL printed by Streamlit (normally `http://localhost:8501`). Stop it with `Ctrl+C`. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` for that PowerShell window, then activate the environment.

### Optional no-login local college demonstration

Authentication remains enabled by default. For an in-person demonstration on the same computer only, explicitly opt in before launching:

```powershell
$env:FL_LOCAL_DEMO_MODE = "1"
$env:STREAMLIT_SERVER_ADDRESS = "127.0.0.1"
streamlit run dashboard.py
```

Keep the four distinct account-password variables above configured. Local demo mode automatically signs in through the existing `authenticate("admin", FL_ADMIN_PASSWORD)` path, so `FL_ADMIN_PASSWORD` must match the password already stored for the Admin account. It does not create an unauthenticated service principal. The app refuses to enter this mode unless Streamlit is bound to a loopback address and the request has a direct loopback `Host` header; forwarded requests are rejected. `.streamlit/config.toml` sets the default server address to `127.0.0.1`. The mode is disabled unless `FL_LOCAL_DEMO_MODE` is exactly `1`; unset it or set it to `0` to restore the regular login screen.

**Never enable `FL_LOCAL_DEMO_MODE` on a public deployment.** If a deployment overrides the bind address or is accessed through a public/forwarded host, the app stops with an error instead of opening the demo. Keep the setting unset in all shared, hosted, and public environments.

### Demo accounts

There is no shared or built-in password. **Set a unique private value for all four environment variables before first launch.** Each password must be at least 12 characters and distinct. Existing state migrates old shared-password hashes once to these account-specific values; prediction history, rounds, audit logs, and model artifacts are not part of this credential migration.

| Role | Username | Password environment variable |
|---|---|---|---|
| Admin | `admin` | `FL_ADMIN_PASSWORD` |
| Hospital | `hospital1` | `FL_HOSPITAL1_PASSWORD` |
| Hospital | `hospital2` | `FL_HOSPITAL2_PASSWORD` |
| Hospital | `hospital3` | `FL_HOSPITAL3_PASSWORD` |

## Application workflows

### Admin

1. Sign in as `admin`. Model Comparison shows stored historical metrics and offers a live evaluation button. New live metrics are computed from actual predictions on one fixed, stratified 114-record holdout; the scaler is fit only on the 455-record training split. Do not use those exploratory holdout results to tune or select a deployment model.
2. In Federated Operations, select FedAvg or experimental DP-FedAvg and open a round. One round trains three simulated clients on disjoint partitions and aggregates their submitted parameter updates using `nᵢ / Σnⱼ`. The dashboard displays the submitted client count, local sample counts, aggregation weights/status, and candidate outcome. An incomplete round can be explicitly cancelled; cancelled rounds cannot create or deploy a candidate.
3. Sign out and submit one update from each of `hospital1`, `hospital2`, and `hospital3`. Each client receives the common initial parameters, trains locally, and returns model parameters plus an exact sample count; raw feature rows are not sent in the update. Sign in as admin to review the aggregated candidate and holdout metrics.
4. Explicitly approve or reject a pending candidate. Only approval updates `saved_models/active_model.json` and assigns a deployed model version; candidate creation and rejection leave the active model unchanged.
5. Optionally use Admin Retraining to upload 30 canonical-order features plus a `diagnosis` (`B`/`M` or `0`/`1`) column, explicitly attest that labels are verified ground truth, and create a candidate after the configured threshold is met (100 by default, minimum 6). Candidates are evaluated against the active model on a stratified holdout from those newly registered records. The active model is checked again at approval; if it changed after candidate evaluation, approval is blocked. Duplicate feature/diagnosis rows are deduplicated. Prediction outputs without `diagnosis` cannot be registered as ground truth.

### Hospital

1. Sign in as one of the hospital accounts. The account sees its own simulated training-partition summary, participation, and prediction history; it cannot view the admin hospital-status table or perform admin-only operations.
2. For a collecting round, select it and submit the simulated local update. Each hospital can submit once per round. Raw feature rows are not included in the update; the client and coordinator nevertheless run in the same process and share local files.
3. Upload a CSV containing exactly the 30 WDBC feature columns in canonical order, or use the manual form. Predictions use only the approved active model. The result includes benign/malignant probabilities and a class prediction; it is not a diagnosis. Prediction history stores the model version, probabilities, and result, not the uploaded feature rows.

### Live evaluation and DP demonstration

- **Federated round:** the admin opens a FedAvg round; each hospital account submits once; the admin sees each participant's sample count and sample-count aggregation weight. A candidate appears only after all three updates are received. Approving it is a separate action; before approval the active model pointer/version is unchanged.
- **DP-FedAvg:** open a round in DP-FedAvg mode and set the clipping norm, noise multiplier, and delta. Every client clips each per-record concatenated weight/bias gradient, adds noise to each full-batch mean, performs its configured local epochs, and submits only its final noisy parameters. It does not return or persist a per-client training loss for DP rounds. The coordinator aggregates these parameters with ordinary sample-count FedAvg. The Dashboard DP section displays conditional cumulative RDP accounting for all submitted DP updates, including partial/cancelled rounds, and details what is visible to the coordinator.
- **Live model comparison:** Model Comparison trains Centralized Logistic Regression, FedAvg, and DP-FedAvg with 30 rounds × 5 local epochs by default. DP-FedAvg utility is shown at noise multipliers 0.5, 1.0, and 2.0. All approaches use the same held-out 114 records; accuracy, precision, malignant recall, F1, ROC-AUC, and fit time are computed from model predictions.
- **IID / Non-IID and noise utility:** Experimental Evaluation compares FedAvg and DP-FedAvg on a further training-only fit/validation split (364 fit, 91 validation by default). It reports actual predictions for IID and Non-IID client partitions and the same three noise multipliers. It does not use the final test partition.
- **Cross-validation:** Experimental Evaluation can run 5-fold, 10-fold, and unstratified 10-fold centralized CV plus corresponding FedAvg/research-DP CV. Fold scaling is fit on fold training records; the final test split is excluded. The research DP CV uses a separate coordinator-side gradient experiment and is not the dashboard's DP-FedAvg client-update mechanism. Its output is temporary and not saved into `metrics/`.
- **Trust weighting:** a separate saved research experiment calculates local validation balanced-accuracy trust scores and weights client parameters by normalized `nᵢ × trustᵢ`. It is not ordinary FedAvg, is not the live round aggregation mode, and is not differentially private.

### State and model files

- `data/application_state.sqlite3`: seeded accounts and salted PBKDF2 password hashes, prediction history, federated-round state and client updates, and the admin audit log.
- `data/confirmed_records.csv`: admin-registered confirmed-label records, identifiers, deduplication fingerprints, and candidate-consumption status. This file is created only when labels are registered.
- `saved_models/`: preserved baseline model/scaler artifacts. New federated and confirmed-label candidate artifacts are stored in `saved_models/candidates/`; federated candidate metadata is in SQLite, while confirmed-label candidates have JSON manifests alongside their model files.
- `saved_models/active_model.json`: active-model pointer. It changes only when an admin explicitly approves a baseline or candidate. Candidate creation and rejection do not replace it.
- `metrics/`: stored historical evaluation CSVs used for comparison; the dashboard reads but does not retrain or overwrite these.
- `research_outputs/verified/`: existing research CSV/JSON results. The dashboard labels them as historical; on-demand evaluations are separate and do not overwrite them.

### Tests and verification

Run the automated tests and Python compile check from the project root:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q -x '[\\/]\.venv[\\/]|[\\/]submission[\\/]' .
```

To run the app without starting any training, use `streamlit run dashboard.py --server.headless true`, open the printed local URL, and stop with `Ctrl+C`. For a workflow demonstration, use the Admin and Hospital steps above. Tests use temporary state/model paths; some unit fixtures are synthetic, while the federated end-to-end test uses the existing public WDBC dataset.

## Implemented, simulated, and incomplete

- **Implemented and tested:** local demo authentication; Admin/Hospital role checks; tenant-scoped hospital status, prediction history, and round participation; SQLite persistence; disjoint simulated client partitions; client-side local Logistic Regression updates; parameter-only client submissions; sample-count-weighted FedAvg; experimental clipped/noisy client updates; RDP composition for submitted noisy updates; explicit candidate approval/rejection; approved-model predictions; and actual-prediction evaluation functions for held-out, IID/Non-IID, and CV experiments.
- **Simulated:** all three hospital clients, their partitions, and the coordinator execute in one Streamlit/Python process and share files. This is not three independently secured hospital servers. The live round workflow models local updates and parameter aggregation but is not production federation.
- **Historical versus newly measured:** saved CSV comparisons remain historical. Dashboard live comparisons and evaluation buttons compute new predictions on click and label results as new; merely opening the app does not run a training experiment.
- **Incomplete / not production-ready:** there is no production identity provider, deployment or secrets management, production-grade tenant/process isolation, real hospital-data federation, clinical validation, or verified end-to-end differential-privacy guarantee. The experimental accountant does not cover all preprocessing and system-level leakage.

## Differential privacy limitations

The dashboard's experimental DP-FedAvg clips each record's concatenated feature/bias gradient to L2 norm `C`. For each full-batch local epoch it adds Gaussian noise to the local mean with standard deviation `σ × 2C/n`, where `σ` is the noise multiplier and `n` is that client's fixed partition size. The `2C/n` sensitivity assumes fixed-size replace-one adjacency within an assigned hospital partition. There is no client or record subsampling and no amplification claim. Noise draws use Python's OS-backed `SystemRandom`; the floating-point Gaussian sampler is not independently audited as an exact DP sampler.

The integer-order RDP accountant composes every submitted client update and local epoch sequentially, with no sampling amplification, and converts at the displayed delta using `min_α(RDP_α + ln(1/δ)/(α−1))`. Epsilon/delta shown for a dashboard DP run is a **conditional mechanism-only bound**, assuming ideal Gaussian noise, the adjacency/sensitivity above, and fixed client partitions/counts. The single-process demo does not enforce a client boundary against coordinator access. For persisted federated rounds, accounting includes all DP client updates recorded in application state, including partial rounds with one or more submissions.

No verified end-to-end differential-privacy guarantee is claimed. Exact sample counts and participation identity/timing are disclosed; candidate holdout metrics are not privatized; pooled training-feature scaling is outside the accountant; and the clients, coordinator, and dataset share one process and local files, so the coordinator can inspect client memory and data. The accountant does not cover every observable or output. The local simulation provides no production-grade isolation boundary. The older `dp_federated.py` experiment and research CV implementation are separate coordinator-side experiments, not the dashboard's client-side DP-FedAvg path.

## Evaluation and limitations

Saved comparison values are historical outputs from stored experiments; they are not regenerated at dashboard startup. The on-demand Model Comparison run uses the fixed 80/20 stratified WDBC split and computes all metrics from actual predictions on the same 114-record holdout. The IID/Non-IID validation experiment uses a separate split within the 455-record training partition and does not use the final holdout. Federated-round candidates are evaluated on the fixed held-out WDBC test partition; confirmed-label retraining candidates are evaluated on a held-out part of newly registered records. These evaluations are exploratory, and repeated inspection of test results can cause test-set overfitting.

## Reproducible research outputs

`academic_experiments.py` runs the additional, isolated research evaluations without loading the side-effecting training/data module or replacing saved baseline models and historical metrics. From the project root, run:

```powershell
python academic_experiments.py --output-dir research_outputs\verified
```

This writes fold-level and summary CSVs for centralized Logistic Regression, simulated-client FedAvg Logistic Regression, and experimental DP-FedAvg under stratified 5-fold, stratified 10-fold, and unstratified 10-fold CV; the IID/Non-IID validation comparison; and trust weights plus trust-aggregation held-out metrics. Every new result is separated under `research_outputs/verified/`. CV uses only the 455-record training partition; scaling is fit within each fold. IID/Non-IID model selection uses a further training/validation split. Trust aggregation is evaluated once on the fixed 114-record held-out test partition. The saved three-model results in `metrics/final_three_model_comparison.csv` are historical baseline results and are not silently regenerated by this command.

The experiment runner records the dataset SHA-256, split seed, Python version, DP-noise randomness, and output inventory in `research_outputs/verified/execution_metadata.json`. Data folds/partitions are seeded, but DP Gaussian noise uses OS-backed `SystemRandom`, so DP results are intentionally not bit-for-bit reproducible. Hospital partitions are simulated. The trust experiment uses local validation balanced accuracy as a heuristic trust score; it is a separate, non-private experiment, not a privacy mechanism.

The federated CV protocols reproduce the project's current simulated-client procedures with fixed settings; they do not establish generalization across different hospitals or distributions. DP-fold outputs characterize the experimental clipping/noise implementation only and do not establish end-to-end DP. The three methods' fold results are not inferred from the historical holdout.

## Final academic materials

- Research report: `submission/RESEARCH_REPORT.md`
- Presentation: `submission/PRESENTATION.html` (open in a browser; use arrow keys to navigate)
- Requirement/evidence checklist: `submission/FINAL_REQUIREMENTS_CHECKLIST.md`
- Final command/results record: `submission/VERIFICATION_RECORD.md`
- Curated source/data/model/test package: `submission/`

To create the curated package once, after the materials above are in place:

```powershell
python prepare_submission.py
```

The preparation script refuses to overwrite an existing `submission/project/` package. It omits virtual environments, caches, and unselected historical clutter, while retaining the public WDBC input, saved baseline models/scalers, relevant dashboard metrics, source, tests, and verified experiment outputs. It does not modify the working project or run training.

## Research limitations and interpretation

All three primary methods are Logistic Regression-based, but the dashboard's DP-FedAvg result is experimental and **does not establish end-to-end differential privacy**. In the modeled execution, each simulated client clips per-record gradients and adds Gaussian noise to each local mean before submitting its final parameters. The coordinator process still shares memory and files with those clients and can inspect local data; pooled data-dependent preprocessing, exact counts/participation, and candidate metrics are outside the accountant. The reported accountant value is conditional on its mechanism, adjacency, sensitivity, noise, and release assumptions. It must not be described as a verified deployment-wide privacy guarantee.

The existing historical comparison reports accuracy, precision, malignant recall, specificity, F1, ROC-AUC, and confusion-derived errors; it does not contain comparable training-time measurements for all three stored runs. The dashboard's newly run live comparison and CV report measured elapsed training time for that execution. SHAP was not run: the current environment does not include the SHAP package, and no synthetic or unverified explanation results are presented. These methods are research demonstrations, not clinical decision support; retain the dashboard's clinical disclaimer.
