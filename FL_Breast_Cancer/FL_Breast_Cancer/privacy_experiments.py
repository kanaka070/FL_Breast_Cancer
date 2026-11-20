
# ============================================================
# PRIVACY-UTILITY TRADE-OFF EXPERIMENT
# MATCHED NON-PRIVATE VS PRIVATE TRAINING
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from data import X_train, y_train
from dp_accountant import DPAccountant
from dp_training import (
    calculate_patient_gradients,
    clip_gradients
)


BASE_DIR = Path(__file__).resolve().parent
METRICS_DIR = BASE_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True)


# ============================================================
# STEP 1: PREPARE VALIDATION DATA
# ============================================================

def prepare_validation_data():

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=123,
        stratify=y_train
    )

    scaler = StandardScaler()

    X_fit_scaled = scaler.fit_transform(X_fit)
    X_val_scaled = scaler.transform(X_val)

    return (
        X_fit_scaled,
        np.asarray(y_fit, dtype=float),
        X_val_scaled,
        np.asarray(y_val, dtype=int)
    )


# ============================================================
# STEP 2: TRAIN MATCHED MODEL
# ============================================================

def train_model(
    X,
    y,
    noise_multiplier,
    rounds=30,
    learning_rate=0.05,
    clipping_norm=1.0,
    delta=1e-5,
    seed=42
):

    n_patients, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0.0

    rng = np.random.default_rng(seed)

    sensitivity = 2 * clipping_norm / n_patients

    accountant = None

    if noise_multiplier > 0:
        accountant = DPAccountant(
            noise_multiplier=noise_multiplier,
            delta=delta,
            sensitivity=sensitivity
        )

    for _ in range(rounds):

        gradients = calculate_patient_gradients(
            X, y, weights, bias
        )

        clipped = clip_gradients(
            gradients,
            clipping_norm
        )

        average_gradient = clipped.mean(axis=0)

        if noise_multiplier > 0:

            noise_std = (
                noise_multiplier * sensitivity
            )

            noise = rng.normal(
                0.0,
                noise_std,
                size=n_features + 1
            )

            average_gradient += noise

            accountant.record_step()

        weights -= (
            learning_rate * average_gradient[:-1]
        )

        bias -= (
            learning_rate * average_gradient[-1]
        )

    epsilon = (
        accountant.get_epsilon()
        if accountant is not None
        else None
    )

    return weights, bias, epsilon


# ============================================================
# STEP 3: EVALUATE MODEL
# ============================================================

def evaluate_model(
    X,
    y,
    weights,
    bias
):

    z = np.clip(
        X @ weights + bias,
        -35,
        35
    )

    probabilities = 1 / (1 + np.exp(-z))

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "malignant_recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y, probabilities
        ),
        "false_negatives": int(fn),
        "false_positives": int(fp)
    }


# ============================================================
# STEP 4: RUN PRIVACY EXPERIMENTS
# ============================================================

def run_privacy_experiments():

    X_fit, y_fit, X_val, y_val = (
        prepare_validation_data()
    )

    noise_levels = [0.0, 1.0, 2.0, 4.0, 8.0]

    results = []

    print("\nPRIVACY-UTILITY TRADE-OFF ANALYSIS")
    print("Training patients:", len(y_fit))
    print("Validation patients:", len(y_val))

    for noise_multiplier in noise_levels:

        print(
            f"\nTraining with noise multiplier: "
            f"{noise_multiplier}"
        )

        weights, bias, epsilon = train_model(
            X_fit,
            y_fit,
            noise_multiplier=noise_multiplier
        )

        metrics = evaluate_model(
            X_val,
            y_val,
            weights,
            bias
        )

        metrics["noise_multiplier"] = (
            noise_multiplier
        )

        metrics["epsilon_upper_bound"] = (
            epsilon
        )

        metrics["delta"] = (
            1e-5 if epsilon is not None else None
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        METRICS_DIR / "privacy_utility_comparison.csv",
        index=False
    )

    print("\nPRIVACY-UTILITY RESULTS")

    print(
        results_df[
            [
                "noise_multiplier",
                "epsilon_upper_bound",
                "accuracy",
                "malignant_recall",
                "f1_score",
                "roc_auc",
                "false_negatives"
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print("metrics/privacy_utility_comparison.csv")

    print(
        "\nPRIVACY-UTILITY EXPERIMENT COMPLETED!"
    )


if __name__ == "__main__":

    run_privacy_experiments()
