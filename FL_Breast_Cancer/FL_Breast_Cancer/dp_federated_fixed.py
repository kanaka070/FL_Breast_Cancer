
# ============================================================
# DP FEDERATED TRAINING WITH FIXED PREPROCESSING
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from data import hospitals
from fixed_private_scaler import FixedPrivateScaler
from dp_accountant import DPAccountant

from dp_training import (
    calculate_patient_gradients,
    clip_gradients
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_models"
METRICS_DIR = BASE_DIR / "metrics"

MODEL_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)


def train_fixed_dp_federated(
    rounds=30,
    learning_rate=0.05,
    clipping_norm=1.0,
    noise_multiplier=2.0,
    delta=1e-5,
    seed=42
):

    # STEP 1: DATA-INDEPENDENT PREPROCESSING

    scaler = FixedPrivateScaler()

    hospital_data = {}

    for name, hospital in hospitals.items():

        X = scaler.transform(hospital["X"])

        y = hospital["y"].to_numpy(
            dtype=float
        )

        hospital_data[name] = {
            "X": X,
            "y": y
        }

    # These dataset dimensions are assumed public
    # for this academic privacy analysis.

    total_patients = 455
    n_features = 30

    actual_patients = sum(
        len(h["y"])
        for h in hospital_data.values()
    )

    if actual_patients != total_patients:
        raise ValueError(
            "Unexpected training patient count."
        )

    for hospital in hospital_data.values():
        if hospital["X"].shape[1] != n_features:
            raise ValueError(
                "Unexpected feature count."
            )

    # STEP 2: INITIALIZE GLOBAL MODEL

    weights = np.zeros(n_features)
    bias = 0.0

    rng = np.random.default_rng(seed)
    sensitivity = 2.0 * clipping_norm / total_patients

    accountant = DPAccountant(
        noise_multiplier=noise_multiplier,
        delta=delta,
        sensitivity=sensitivity
    )

    history = []

    print("\nFIXED-SCALER DP FEDERATED TRAINING")
    print("Hospitals:", len(hospital_data))
    print("Training patients:", total_patients)
    print("Features:", n_features)

    # STEP 3: TRAINING ROUNDS

    for round_number in range(1, rounds + 1):

        gradient_sum = np.zeros(
            n_features + 1
        )

        for name, hospital in hospital_data.items():

            X = hospital["X"]
            y = hospital["y"]

            patient_gradients = (
                calculate_patient_gradients(
                    X, y, weights, bias
                )
            )

            clipped = clip_gradients(
                patient_gradients,
                clipping_norm=clipping_norm
            )

            hospital_sum = np.sum(
                clipped,
                axis=0
            )

            gradient_sum += hospital_sum

        average_gradient = (
            gradient_sum / total_patients
        )

        # Replace-one patient adjacency:
        # sensitivity <= 2C/N

        noise_std = (
            noise_multiplier * sensitivity
        )

        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=n_features + 1
        )

        private_gradient = (
            average_gradient + noise
        )

        # STEP 4: UPDATE LOGISTIC REGRESSION

        weights -= (
            learning_rate * private_gradient[:-1]
        )

        bias -= (
            learning_rate * private_gradient[-1]
        )

        # STEP 5: ACCOUNT FOR PRIVACY

        accountant.record_step()

        epsilon = accountant.get_epsilon()

        history.append({
            "round": round_number,
            "epsilon": epsilon,
            "delta": delta,
            "noise_multiplier": noise_multiplier
        })

        if (
            round_number == 1
            or round_number % 5 == 0
        ):
            print(
                f"Round {round_number:02d}/{rounds}"
                f" | Epsilon: {epsilon:.4f}"
            )

    # STEP 6: SAVE MODEL

    np.savez(
        MODEL_DIR / "dp_fixed_model.npz",
        weights=weights,
        bias=bias
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "dp_fixed_scaler.pkl"
    )

    pd.DataFrame(history).to_csv(
        METRICS_DIR / "dp_fixed_history.csv",
        index=False
    )

    print("\nTRAINING COMPLETED")
    print("Final epsilon:", round(epsilon, 4))
    print("Delta:", delta)

    print("\nSaved files:")
    print("saved_models/dp_fixed_model.npz")
    print("saved_models/dp_fixed_scaler.pkl")
    print("metrics/dp_fixed_history.csv")


if __name__ == "__main__":

    train_fixed_dp_federated()
