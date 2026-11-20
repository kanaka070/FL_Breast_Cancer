
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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

METRICS_DIR.mkdir(exist_ok=True)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def evaluate_predictions(name, y_true, probabilities):

    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "model": name,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "malignant_recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "specificity": tn / (tn + fp),
        "f1_score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "false_negatives": int(fn),
        "false_positives": int(fp)
    }


def load_custom_model(model_name, scaler_name):

    model_data = np.load(MODEL_DIR / model_name)

    weights = model_data["weights"]
    bias = float(model_data["bias"])

    scaler = joblib.load(MODEL_DIR / scaler_name)

    X_scaled = scaler.transform(X_test)

    return sigmoid(X_scaled @ weights + bias)


def main():

    results = []

    # Centralized model
    centralized = joblib.load(
        MODEL_DIR / "centralized_logistic_regression.pkl"
    )

    centralized_probs = centralized.predict_proba(
        X_test
    )[:, 1]

    results.append(
        evaluate_predictions(
            "Centralized Logistic Regression",
            y_test,
            centralized_probs
        )
    )

    # Original FedAvg model
    fedavg_probs = load_custom_model(
        "fedavg_model.npz",
        "fedavg_scaler.pkl"
    )

    results.append(
        evaluate_predictions(
            "FedAvg Logistic Regression",
            y_test,
            fedavg_probs
        )
    )

    # New DP federated model
    dp_probs = load_custom_model(
        "dp_federated_model.npz",
        "dp_federated_scaler.pkl"
    )

    results.append(
        evaluate_predictions(
            "DP Federated Gradient Model",
            y_test,
            dp_probs
        )
    )

    comparison = pd.DataFrame(results)

    comparison.to_csv(
        METRICS_DIR / "final_three_model_comparison.csv",
        index=False
    )

    print("\nFINAL THREE-MODEL COMPARISON")
    print("Test patients:", len(y_test))

    print(
        comparison.round(6).to_string(index=False)
    )

    print("\nSaved:")
    print("metrics/final_three_model_comparison.csv")


if __name__ == "__main__":
    main()
