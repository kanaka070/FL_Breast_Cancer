
# ============================================================
# DIFFERENTIALLY PRIVATE FEDERATED TRAINING
# THREE HOSPITALS + LOGISTIC REGRESSION
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

from data import hospitals
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


def train_dp_federated(
    rounds=30,
    learning_rate=0.05,
    clipping_norm=1.0,
    noise_multiplier=2.0,
    delta=1e-5,
    seed=42
):

    # --------------------------------------------------------
    # PREPARE HOSPITAL DATA
    # --------------------------------------------------------

    X_all = pd.concat(
        [hospital["X"] for hospital in hospitals.values()],
        axis=0
    )

    scaler = StandardScaler()
    scaler.fit(X_all)

    hospital_data = {}

    for name, hospital in hospitals.items():

        X_scaled = scaler.transform(hospital["X"])
        y = hospital["y"].to_numpy(dtype=float)

        hospital_data[name] = {
            "X": X_scaled,
            "y": y
        }

    total_patients = sum(
        len(hospital["y"])
        for hospital in hospital_data.values()
    )

    n_features = X_all.shape[1]

    # --------------------------------------------------------
    # INITIALIZE GLOBAL MODEL
    # --------------------------------------------------------

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

    print("\nDP FEDERATED TRAINING")
    print("Hospitals:", len(hospital_data))
    print("Patients:", total_patients)
    print("Features:", n_features)

    # --------------------------------------------------------
    # FEDERATED TRAINING ROUNDS
    # --------------------------------------------------------

    for round_number in range(1, rounds + 1):

        total_gradient = np.zeros(n_features + 1)

        for name, hospital in hospital_data.items():

            X = hospital["X"]
            y = hospital["y"]

            # Each hospital computes patient-level gradients
            patient_gradients = calculate_patient_gradients(
                X, y, weights, bias
            )

            # Clip each patient's gradient
            clipped_gradients = clip_gradients(
                patient_gradients,
                clipping_norm=clipping_norm
            )

            # Send summed gradients to trusted aggregator
            hospital_gradient_sum = np.sum(
                clipped_gradients,
                axis=0
            )

            total_gradient += hospital_gradient_sum

        # Average over all patients
        average_gradient = (
            total_gradient / total_patients
        )

        # Gaussian noise calibrated for replace-one adjacency
        noise_std = noise_multiplier * sensitivity

        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=n_features + 1
        )

        private_gradient = average_gradient + noise

        # Global Logistic Regression update
        weights -= learning_rate * private_gradient[:-1]
        bias -= learning_rate * private_gradient[-1]

        accountant.record_step()

        epsilon = accountant.get_epsilon()

        history.append({
            "round": round_number,
            "epsilon": epsilon,
            "noise_multiplier": noise_multiplier,
            "noise_std": noise_std
        })

        if round_number == 1 or round_number % 5 == 0:

            print(
                f"Round {round_number:02d}/{rounds}"
                f" | Epsilon: {epsilon:.4f}"
            )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    np.savez(
        MODEL_DIR / "dp_federated_model.npz",
        weights=weights,
        bias=bias
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "dp_federated_scaler.pkl"
    )

    pd.DataFrame(history).to_csv(
        METRICS_DIR / "dp_federated_history.csv",
        index=False
    )

    print("\nDP FEDERATED TRAINING COMPLETED")
    print("Final epsilon:", round(epsilon, 4))
    print("Delta:", delta)

    print("\nSaved:")
    print("saved_models/dp_federated_model.npz")
    print("saved_models/dp_federated_scaler.pkl")
    print("metrics/dp_federated_history.csv")


if __name__ == "__main__":

    train_dp_federated()
