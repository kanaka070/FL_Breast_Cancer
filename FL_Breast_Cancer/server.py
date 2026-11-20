'''
# ============================================================
# FEDERATED AVERAGING SERVER
# ============================================================

import numpy as np
from sklearn.preprocessing import StandardScaler

from data import hospitals, X_train
from model import LogisticRegressionModel
from client import HospitalClient
from workflow_utils import federated_averaging


# Combine hospital parameters using weighted averaging
def federated_averaging(client_results):

    total_patients = sum(
        result["num_patients"] for result in client_results
    )

    if total_patients == 0:
        raise ValueError("No patients available for aggregation.")

    first_weights = client_results[0]["parameters"]["weights"]

    global_weights = np.zeros_like(first_weights, dtype=float)
    global_bias = 0.0

    for result in client_results:

        proportion = result["num_patients"] / total_patients

        local_weights = result["parameters"]["weights"]
        local_bias = result["parameters"]["bias"]

        global_weights += proportion * local_weights
        global_bias += proportion * local_bias

    return {
        "weights": global_weights,
        "bias": float(global_bias)
    }


# ============================================================
# TEST FEDERATED AVERAGING
# ============================================================

if __name__ == "__main__":

    print("\nFEDERATED AVERAGING SERVER TEST")

    # Use the same shared scaling approach as client.py
    scaler = StandardScaler()
    scaler.fit(X_train)

    clients = []

    for name, hospital in hospitals.items():

        X_scaled = scaler.transform(hospital["X"])

        client = HospitalClient(
            name=name,
            X=X_scaled,
            y=hospital["y"],
            learning_rate=0.05
        )

        clients.append(client)

    # Initial global model
    global_model = LogisticRegressionModel(
        n_features=30,
        learning_rate=0.05
    )

    global_parameters = global_model.get_parameters()

    # Each hospital trains from the same initial model
    client_results = []

    for client in clients:

        result = client.train_local(
            global_parameters,
            epochs=5
        )

        client_results.append(result)

        print(
            f"{result['hospital']} submitted parameters "
            f"from {result['num_patients']} patients."
        )

    # Aggregate local model parameters
    updated_parameters = federated_averaging(client_results)

    # Update global model
    global_model.set_parameters(updated_parameters)

    print("\nFEDAVG AGGREGATION COMPLETED")

    print("Global weights shape:", global_model.weights.shape)
    print("Global bias:", round(global_model.bias, 6))

    print("\nHospital aggregation weights:")

    total = sum(
        result["num_patients"] for result in client_results
    )

    for result in client_results:

        percentage = result["num_patients"] / total * 100

        print(
            f"{result['hospital']}: {percentage:.2f}%"
        )

    assert len(client_results) == 3
    assert total == 455
    assert global_model.weights.shape == (30,)
    assert np.isfinite(global_model.weights).all()
    assert np.isfinite(global_model.bias)

    print("\nAll FedAvg verification checks passed!")
    print("FEDERATED AVERAGING SERVER COMPLETED SUCCESSFULLY!")
'''



# ============================================================
# FEDERATED LEARNING SERVER
# FEDAVG WITH MULTIPLE COMMUNICATION ROUNDS
# ============================================================

import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler

from data import hospitals, X_train
from model import LogisticRegressionModel
from client import HospitalClient


# ============================================================
# STEP 2: CREATE HOSPITAL CLIENTS
# ============================================================

def create_clients():

    # Shared scaling for our simulated hospital environment
    scaler = StandardScaler()
    scaler.fit(X_train)

    clients = []

    for name, hospital in hospitals.items():

        X_scaled = scaler.transform(hospital["X"])

        client = HospitalClient(
            name=name,
            X=X_scaled,
            y=hospital["y"],
            learning_rate=0.05
        )

        clients.append(client)

    return clients, scaler


# ============================================================
# STEP 3: MULTIPLE COMMUNICATION ROUNDS
# ============================================================

def train_federated_model(rounds=30, local_epochs=5):

    clients, scaler = create_clients()

    global_model = LogisticRegressionModel(
        n_features=30,
        learning_rate=0.05
    )

    global_parameters = global_model.get_parameters()

    training_history = []

    print("\nFEDERATED LEARNING TRAINING STARTED")
    print("Hospitals:", len(clients))
    print("Communication rounds:", rounds)
    print("Local epochs per round:", local_epochs)

    for round_number in range(1, rounds + 1):

        client_results = []

        # Each hospital receives the current global model
        # and trains using its own patient records

        for client in clients:

            result = client.train_local(
                global_parameters,
                epochs=local_epochs
            )

            client_results.append(result)

        # Combine local parameters using FedAvg
        global_parameters = federated_averaging(
            client_results
        )

        # Update the global model
        global_model.set_parameters(global_parameters)

        # Calculate training loss of the updated global model
        weighted_loss = 0.0

        total_patients = sum(
            client.num_patients
            for client in clients
        )

        for client in clients:

            hospital_loss = global_model.calculate_loss(
                client.X,
                client.y
            )

            weighted_loss += (
                hospital_loss * client.num_patients
            )

        average_loss = weighted_loss / total_patients

        training_history.append({
            "round": round_number,
            "training_loss": average_loss
        })

        print(
            f"Round {round_number:02d}/{rounds}"
            f" - Training loss: {average_loss:.6f}"
        )

    # ========================================================
    # STEP 4: SAVE TRAINED MODEL AND RESULTS
    # ========================================================

    base_dir = Path(__file__).resolve().parent

    models_folder = base_dir / "saved_models"
    metrics_folder = base_dir / "metrics"

    models_folder.mkdir(exist_ok=True)
    metrics_folder.mkdir(exist_ok=True)

    np.savez(
        models_folder / "fedavg_model.npz",
        weights=global_model.weights,
        bias=global_model.bias
    )

    joblib.dump(
        scaler,
        models_folder / "fedavg_scaler.pkl"
    )

    history_df = pd.DataFrame(training_history)

    history_df.to_csv(
        metrics_folder / "fedavg_training_history.csv",
        index=False
    )

    # ========================================================
    # STEP 5: VERIFY RESULTS
    # ========================================================

    assert len(clients) == 3

    assert sum(
        client.num_patients
        for client in clients
    ) == 455

    assert global_model.weights.shape == (30,)

    assert np.isfinite(global_model.weights).all()
    assert np.isfinite(global_model.bias)

    assert np.isfinite(
        history_df["training_loss"]
    ).all()

    print("\nFEDERATED TRAINING COMPLETED")

    print(
        "Initial training loss:",
        round(training_history[0]["training_loss"], 6)
    )

    print(
        "Final training loss:",
        round(training_history[-1]["training_loss"], 6)
    )

    print("Global weights:", len(global_model.weights))
    print("Global bias:", round(global_model.bias, 6))

    print("\nSaved files:")
    print("saved_models/fedavg_model.npz")
    print("saved_models/fedavg_scaler.pkl")
    print("metrics/fedavg_training_history.csv")

    print("\nMULTIPLE COMMUNICATION ROUNDS COMPLETED SUCCESSFULLY!")

    return global_model, scaler, history_df


# ============================================================
# RUN FEDERATED TRAINING
# ============================================================

if __name__ == "__main__":

    train_federated_model(
        rounds=30,
        local_epochs=5
    )
