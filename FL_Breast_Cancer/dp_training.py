
# ============================================================
# DIFFERENTIALLY PRIVATE LOGISTIC REGRESSION
# THREE SIMULATED HOSPITALS
# TRUSTED CENTRAL AGGREGATOR
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from data import hospitals
from dp_accountant import DPAccountant


BASE_DIR = Path(__file__).resolve().parent
METRICS_DIR = BASE_DIR / "metrics"
MODEL_DIR = BASE_DIR / "saved_models"

METRICS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# STEP 1: PREPARE HOSPITAL DATA
# ============================================================

def prepare_dp_data():

    all_features = pd.concat(
        [h["X"] for h in hospitals.values()],
        axis=0
    )

    scaler = StandardScaler()
    scaler.fit(all_features)

    prepared = {}

    for name, hospital in hospitals.items():

        prepared[name] = {
            "X": scaler.transform(hospital["X"]),
            "y": np.asarray(
                hospital["y"],
                dtype=float
            )
        }

    return prepared, scaler


# ============================================================
# STEP 2: CALCULATE PER-PATIENT GRADIENTS
# ============================================================

def calculate_patient_gradients(X, y, weights, bias):

    z = X @ weights + bias

    z = np.clip(z, -35, 35)

    probabilities = 1 / (1 + np.exp(-z))

    errors = probabilities - y

    weight_gradients = errors[:, None] * X

    bias_gradients = errors[:, None]

    # Treat weights and bias as one parameter vector.
    gradients = np.concatenate(
        [weight_gradients, bias_gradients],
        axis=1
    )

    return gradients


# ============================================================
# STEP 3: CLIP INDIVIDUAL GRADIENTS
# ============================================================

def clip_gradients(gradients, clipping_norm=1.0):

    norms = np.linalg.norm(
        gradients,
        axis=1,
        keepdims=True
    )

    factors = np.minimum(
        1.0,
        clipping_norm / np.maximum(norms, 1e-12)
    )

    return gradients * factors


# ============================================================
# STEP 4: TRAIN WITH GAUSSIAN NOISE
# ============================================================

def train_dp_model(
    rounds=30,
    learning_rate=0.05,
    clipping_norm=1.0,
    noise_multiplier=2.0,
    delta=1e-5,
    seed=42
):

    hospital_data, scaler = prepare_dp_data()

    total_patients = sum(
        len(h["y"]) for h in hospital_data.values()
    )

    feature_count = next(
        iter(hospital_data.values())
    )["X"].shape[1]

    weights = np.zeros(feature_count)
    bias = 0.0

    rng = np.random.default_rng(seed)

    # Replace-one patient adjacency:
    # Two clipped gradients can differ by at most 2*C.
    #
    # Sensitivity of the average gradient:
    # 2*C / N

    sensitivity = (
        2 * clipping_norm / total_patients
    )

    noise_std = (
        noise_multiplier * sensitivity
    )

    accountant = DPAccountant(
        noise_multiplier=noise_multiplier,
        delta=delta,
        sensitivity=sensitivity
    )

    history = []

    print("\nDIFFERENTIALLY PRIVATE TRAINING")
    print("Simulated hospitals:", len(hospital_data))
    print("Total training patients:", total_patients)
    print("Clipping norm:", clipping_norm)
    print("Noise multiplier:", noise_multiplier)
    print("Noise standard deviation:", noise_std)

    for round_number in range(1, rounds + 1):

        summed_gradients = np.zeros(
            feature_count + 1
        )

        # Trusted aggregator receives clipped sums.
        for name, hospital in hospital_data.items():

            gradients = calculate_patient_gradients(
                hospital["X"],
                hospital["y"],
                weights,
                bias
            )

            clipped = clip_gradients(
                gradients,
                clipping_norm
            )

            summed_gradients += np.sum(
                clipped,
                axis=0
            )

        average_gradient = (
            summed_gradients / total_patients
        )

        # One Gaussian release per communication round.
        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=feature_count + 1
        )

        private_gradient = (
            average_gradient + noise
        )

        weights -= (
            learning_rate * private_gradient[:-1]
        )

        bias -= (
            learning_rate * private_gradient[-1]
        )

        accountant.record_step()

        epsilon = accountant.get_epsilon()

        history.append({
            "round": round_number,
            "epsilon": epsilon,
            "delta": delta,
            "noise_multiplier": noise_multiplier
        })

        if round_number == 1 or round_number % 5 == 0:

            print(
                f"Round {round_number}/{rounds} "
                f"| Epsilon upper bound: {epsilon:.4f}"
            )

    # Save private model parameters.
    np.savez(
        MODEL_DIR / "dp_model.npz",
        weights=weights,
        bias=bias
    )

    # Save scaler for evaluation.
    import joblib

    joblib.dump(
        scaler,
        MODEL_DIR / "dp_scaler.pkl"
    )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        METRICS_DIR / "dp_privacy_history.csv",
        index=False
    )

    print("\nFINAL PRIVACY REPORT")
    print(accountant.get_privacy_report())

    print("\nSaved files:")
    print("saved_models/dp_model.npz")
    print("saved_models/dp_scaler.pkl")
    print("metrics/dp_privacy_history.csv")

    print("\nDP TRAINING COMPLETED SUCCESSFULLY!")

    return weights, bias, accountant


# ============================================================
# STEP 5: RUN TRAINING
# ============================================================

if __name__ == "__main__":

    train_dp_model(
        rounds=30,
        learning_rate=0.05,
        clipping_norm=1.0,
        noise_multiplier=2.0,
        delta=1e-5
    )
