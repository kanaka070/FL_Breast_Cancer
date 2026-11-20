# Final verification record

Environment: Windows, Python 3.13.2 in the existing project virtual environment. All paths below are relative to the project root unless explicitly noted.

## Commands and results

| Purpose | Command / action | Result |
|---|---|---|
| Full automated suite in working project | `python -m unittest discover -s tests -v` | **23 tests passed** |
| Full automated suite in curated copy | From `submission/project`: `python -m unittest discover -s tests -v` | **23 tests passed** |
| Python compilation | `python -m compileall -q -x '[\\/]\.venv[\\/]|[\\/]submission[\\/]' .` | **Passed**; virtualenv and generated submission tree excluded |
| Additional reproducible experiments | `python academic_experiments.py --output-dir research_outputs\verified` | **Passed**; completed 75 approach/fold fits (25 each for centralized, FedAvg, and DP-FedAvg), IID/Non-IID validation runs, and trust aggregation; wrote only under `research_outputs/verified/` |
| Dashboard smoke | `streamlit run dashboard.py --server.headless true --server.port 8502 --server.address 127.0.0.1` | HTTP page loaded; Overview, Hospitals, Model Comparison, Differential Privacy, Prediction, and Admin Retraining tabs each rendered non-empty content with no exception |
| Presentation smoke | Open `submission/PRESENTATION.html` in browser | Loaded first of 11 slides; next/previous controls available |
| Protected source artifact audit | SHA-256 for all files under existing `data/`, `saved_models/`, and `metrics/` before and after work | **32 files unchanged** |

The dashboard smoke test did not upload a file, approve a model, train a candidate, or register labels. The existing active-model pointer remained byte-for-byte unchanged; `data/confirmed_records.csv` and `saved_models/candidates/` were not created by the smoke test. Automated retraining tests use synthetic fixtures in temporary directories.

The trust-weighted and sample-size FedAvg evaluation rows share the same 0.023524-second measured client-training/aggregation interval; they are not independent timing benchmarks. IID/Non-IID fit times and CV fit times are run-specific wall-clock measurements, not deployment latency.

## Experiment results actually executed

- Saved historical primary held-out accuracy: centralized Logistic Regression **0.964912**, FedAvg Logistic Regression **0.973684**, experimental DP-FedAvg **0.947368**.
- Centralized stratified 5-fold, stratified 10-fold, and unstratified 10-fold CV ran on the 455-record training partition. Mean accuracy was **0.973626**, **0.975845**, and **0.973623**, respectively. FedAvg mean accuracy was **0.971429**, **0.975797**, and **0.971304**; experimental DP-FedAvg was **0.934066**, **0.931739**, and **0.933913**. Full fold metrics, confusion counts, standard deviations, and timings are in the CV CSVs.
- IID and Non-IID runs used a further 91-record validation set: both accuracy **0.978022**, malignant recall **0.941176**. They are not final test evaluations.
- Trust-weighted aggregation was evaluated on the fixed 114-record test partition: accuracy **0.973684**, malignant recall **0.928571**, specificity **1.000000**, ROC-AUC **0.998677**, TN=72, FP=0, FN=3, TP=39.
- No existing baseline artifact was retrained or replaced, and no standalone full-baseline DP training script was run. DP-FedAvg was trained only in the isolated cross-validation folds. No repeated-seed experiment or SHAP run was performed; historical saved DP metrics were read only.
- The existing historical accountant CSV ends at estimated `epsilon=17.0065`, `delta=1e-5`, noise multiplier 2, and 30 rounds. This is a conditional accountant estimate, not an end-to-end privacy guarantee.
- No real patient records were used for software tests; test fixtures are synthetic. The benchmark experiment uses the existing public WDBC dataset.

## Reproduction commands

In PowerShell from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard.py
python -m unittest discover -s tests -v
python -m compileall -q -x '[\\/]\.venv[\\/]|[\\/]submission[\\/]' .
python academic_experiments.py --output-dir research_outputs\verified
```

To prepare the curated package once, run `python prepare_submission.py`. It creates `submission/project/` and refuses to overwrite that existing package. To view the research report and presentation, open `submission/RESEARCH_REPORT.md` and `submission/PRESENTATION.html`; the evidence checklist is `submission/FINAL_REQUIREMENTS_CHECKLIST.md`.

## Remaining research limitations

The DP implementation does not demonstrate end-to-end differential privacy: the coordinator observes pre-noise clipped client sums, pooled preprocessing is not included in accounting, and the full observer/release assumptions have not been independently established. SHAP and CV variability for the two federated methods are pending. The hospitals are simulated, and none of the results establishes clinical utility or production readiness.
