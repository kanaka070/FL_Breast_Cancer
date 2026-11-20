
# ============================================================
# FEDERATED MODEL EVALUATION
# LOGISTIC REGRESSION - FEDAVG VS CENTRALIZED
# ============================================================

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
    confusion_matrix,
    classification_report
)

from data import X_test, y_test
from model import LogisticRegressionModel


# ============================================================
# STEP 1: SET FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_models"
METRICS_DIR = BASE_DIR / "metrics"

METRICS_DIR.mkdir(exist_ok=True)


# ============================================================
# STEP 2: DEFINE EVALUATION FUNCTION
# ============================================================

def evaluate_predictions(model_name, y_true, probabilities):

    probabilities = np.asarray(
        probabilities,
        dtype=float
    ).reshape(-1)

    y_true = np.asarray(
        y_true,
        dtype=int
    ).reshape(-1)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    results = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }

    print(f"\n{model_name.upper()} EVALUATION")

    print("Accuracy:", round(accuracy, 6))
    print("Precision:", round(precision, 6))
    print("Recall:", round(recall, 6))
    print("Specificity:", round(specificity, 6))
    print("F1-score:", round(f1, 6))
    print("ROC-AUC:", round(auc, 6))

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        )
    )

    print("\nFalse negatives:", fn)
    print("False positives:", fp)

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            zero_division=0
        )
    )

    return results, predictions


# ============================================================
# STEP 3: LOAD SAVED FEDAVG MODEL
# ============================================================

def evaluate_fedavg():

    print("\nLOADING FEDAVG MODEL")

    model_path = MODEL_DIR / "fedavg_model.npz"
    scaler_path = MODEL_DIR / "fedavg_scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "FedAvg model not found. Run server.py first."
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            "FedAvg scaler not found. Run server.py first."
        )

    saved_parameters = np.load(model_path)

    weights = saved_parameters["weights"]
    bias = float(saved_parameters["bias"])

    scaler = joblib.load(scaler_path)

    global_model = LogisticRegressionModel(
        n_features=len(weights),
        learning_rate=0.05
    )

    global_model.set_parameters({
        "weights": weights,
        "bias": bias
    })

    # Transform the held-out test features
    X_test_scaled = scaler.transform(X_test)

    # Predict malignant probabilities
    probabilities = global_model.predict_proba(
        X_test_scaled
    )

    results, predictions = evaluate_predictions(
        "FedAvg",
        y_test,
        probabilities
    )

    return results, predictions, probabilities


# ============================================================
# STEP 4: EVALUATE CENTRALIZED MODEL
# ============================================================

def evaluate_centralized():

    model_path = (
        MODEL_DIR / "centralized_logistic_regression.pkl"
    )

    if not model_path.exists():

        print(
            "\nCentralized model file not found. "
            "Skipping centralized comparison."
        )

        return None, None, None

    print("\nLOADING CENTRALIZED MODEL")

    centralized_model = joblib.load(model_path)

    # The saved centralized pipeline includes its scaler
    probabilities = centralized_model.predict_proba(
        X_test
    )[:, 1]

    results, predictions = evaluate_predictions(
        "Centralized",
        y_test,
        probabilities
    )

    return results, predictions, probabilities


# ============================================================
# STEP 5: RUN EVALUATION AND SAVE RESULTS
# ============================================================

if __name__ == "__main__":

    print("\nBREAST CANCER MODEL EVALUATION")
    print("Held-out test patients:", len(y_test))

    assert len(y_test) == 114

    fedavg_results, fedavg_predictions, fedavg_probabilities = (
        evaluate_fedavg()
    )

    centralized_results, centralized_predictions, centralized_probabilities = (
        evaluate_centralized()
    )

    all_results = [fedavg_results]

    if centralized_results is not None:
        all_results.append(centralized_results)

    comparison_df = pd.DataFrame(all_results)

    comparison_df.to_csv(
        METRICS_DIR / "model_comparison.csv",
        index=False
    )

    fedavg_df = pd.DataFrame([fedavg_results])

    fedavg_df.to_csv(
        METRICS_DIR / "fedavg_results.csv",
        index=False
    )

    # Save patient-level predictions without patient IDs
    predictions_df = pd.DataFrame({
        "actual_diagnosis": np.asarray(y_test),
        "fedavg_probability": fedavg_probabilities,
        "fedavg_prediction": fedavg_predictions
    })

    if centralized_predictions is not None:

        predictions_df["centralized_probability"] = (
            centralized_probabilities
        )

        predictions_df["centralized_prediction"] = (
            centralized_predictions
        )

    predictions_df.to_csv(
        METRICS_DIR / "test_predictions.csv",
        index=False
    )

    print("\nMODEL COMPARISON SUMMARY")

    print(
        comparison_df[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "false_negative"
            ]
        ].to_string(index=False)
    )

    print("\nResults saved successfully!")
    print("metrics/fedavg_results.csv")
    print("metrics/model_comparison.csv")
    print("metrics/test_predictions.csv")

    print("\nFEDERATED MODEL EVALUATION COMPLETED!")
