
# ============================================================
# EVALUATE FIXED-SCALER DP FEDERATED MODEL
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from data import X_test, y_test


BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_models"
METRICS_DIR = BASE_DIR / "metrics"


# STEP 1: LOAD SAVED MODEL

model = np.load(
    MODEL_DIR / "dp_fixed_model.npz"
)

weights = model["weights"]
bias = float(model["bias"])

scaler = joblib.load(
    MODEL_DIR / "dp_fixed_scaler.pkl"
)


# STEP 2: TRANSFORM TEST FEATURES

X_scaled = scaler.transform(X_test)


# STEP 3: PREDICT MALIGNANT PROBABILITIES

z = X_scaled @ weights + bias

z = np.clip(z, -500, 500)

probabilities = 1.0 / (1.0 + np.exp(-z))

predictions = (probabilities >= 0.5).astype(int)


# STEP 4: CALCULATE EVALUATION METRICS

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

results = {
    "model": "DP Federated Fixed Scaler",
    "accuracy": accuracy_score(
        y_test, predictions
    ),
    "precision": precision_score(
        y_test, predictions, zero_division=0
    ),
    "malignant_recall": recall_score(
        y_test, predictions, zero_division=0
    ),
    "specificity": tn / (tn + fp),
    "f1_score": f1_score(
        y_test, predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(
        y_test, probabilities
    ),
    "false_negatives": int(fn),
    "false_positives": int(fp)
}


# STEP 5: DISPLAY RESULTS

print("\nFIXED-SCALER DP MODEL EVALUATION")
print("Test patients:", len(y_test))

for metric, value in results.items():
    print(f"{metric}: {value}")


# STEP 6: SAVE RESULTS

new_result = pd.DataFrame([results])

new_result.to_csv(
    METRICS_DIR / "dp_fixed_evaluation.csv",
    index=False
)


# STEP 7: COMPARE WITH PREVIOUS MODELS

previous_results = pd.read_csv(
    METRICS_DIR / "final_three_model_comparison.csv"
)

comparison = pd.concat(
    [previous_results, new_result],
    ignore_index=True
)

comparison.to_csv(
    METRICS_DIR / "four_model_comparison.csv",
    index=False
)

print("\nFOUR-MODEL COMPARISON")
print(
    comparison.round(6).to_string(index=False)
)

print("\nEVALUATION COMPLETED!")
