
# ============================================================
# ADVANCED FEDERATED LEARNING EXPERIMENTS
# HOSPITAL-LEVEL DIAGNOSTIC RELIABILITY
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
    confusion_matrix
)

from data import hospitals
from model import LogisticRegressionModel
from client import HospitalClient
from server import federated_averaging


# ============================================================
# STEP 1: CREATE HOSPITAL TRAINING AND VALIDATION SPLITS
# ============================================================

def prepare_hospital_splits():

    hospital_splits = {}

    for name, hospital in hospitals.items():

        X = hospital["X"]
        y = hospital["y"]

        X_train_local, X_val_local, y_train_local, y_val_local = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

        hospital_splits[name] = {
            "X_train": X_train_local,
            "X_val": X_val_local,
            "y_train": y_train_local,
            "y_val": y_val_local
        }

    return hospital_splits


# ============================================================
# STEP 2: TRAIN A FRESH FEDERATED MODEL
# ============================================================

def train_validation_model(
    hospital_splits,
    rounds=30,
    local_epochs=5
):

    # Fit the scaler only on training patients
    combined_training_features = pd.concat(
        [
            hospital["X_train"]
            for hospital in hospital_splits.values()
        ],
        axis=0
    )

    scaler = StandardScaler()
    scaler.fit(combined_training_features)

    clients = []

    for name, hospital in hospital_splits.items():

        X_scaled = scaler.transform(
            hospital["X_train"]
        )

        client = HospitalClient(
            name=name,
            X=X_scaled,
            y=hospital["y_train"],
            learning_rate=0.05
        )

        clients.append(client)

    global_model = LogisticRegressionModel(
        n_features=30,
        learning_rate=0.05
    )

    global_parameters = global_model.get_parameters()

    print("\nTRAINING HOSPITAL RELIABILITY MODEL")

    for round_number in range(1, rounds + 1):

        client_results = []

        for client in clients:

            result = client.train_local(
                global_parameters,
                epochs=local_epochs
            )

            client_results.append(result)

        global_parameters = federated_averaging(
            client_results
        )

        global_model.set_parameters(
            global_parameters
        )

        if round_number == 1 or round_number % 5 == 0:

            print(
                f"Communication round: "
                f"{round_number}/{rounds}"
            )

    return global_model, scaler


# ============================================================
# STEP 3: EVALUATE EACH HOSPITAL
# ============================================================

def evaluate_hospitals(
    hospital_splits,
    global_model,
    scaler
):

    results = []

    for name, hospital in hospital_splits.items():

        X_val_scaled = scaler.transform(
            hospital["X_val"]
        )

        y_true = np.asarray(
            hospital["y_val"],
            dtype=int
        )

        probabilities = global_model.predict_proba(
            X_val_scaled
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()

        accuracy = accuracy_score(
            y_true,
            predictions
        )

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

        results.append({
            "hospital": name,
            "validation_patients": len(y_true),
            "malignant_patients": int(sum(y_true)),
            "accuracy": accuracy,
            "precision": precision,
            "malignant_recall": recall,
            "f1_score": f1,
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_positives": int(tp),
            "true_negatives": int(tn)
        })

    return pd.DataFrame(results)


# ============================================================
# STEP 4: MEASURE PERFORMANCE DIFFERENCES
# ============================================================

def calculate_reliability_summary(results_df):

    recall_gap = (
        results_df["malignant_recall"].max()
        - results_df["malignant_recall"].min()
    )

    accuracy_gap = (
        results_df["accuracy"].max()
        - results_df["accuracy"].min()
    )

    summary = pd.DataFrame([{
        "recall_gap": recall_gap,
        "accuracy_gap": accuracy_gap,
        "total_false_negatives": int(
            results_df["false_negatives"].sum()
        ),
        "total_validation_patients": int(
            results_df["validation_patients"].sum()
        )
    }])

    return summary


# ============================================================
# STEP 5: RUN EXPERIMENT AND SAVE RESULTS
# ============================================================

if __name__ == "__main__":

    print("\nHOSPITAL-LEVEL DIAGNOSTIC RELIABILITY")

    hospital_splits = prepare_hospital_splits()

    # Verify training and validation groups do not overlap
    for name, hospital in hospital_splits.items():

        training_indices = set(
            hospital["X_train"].index
        )

        validation_indices = set(
            hospital["X_val"].index
        )

        assert training_indices.isdisjoint(
            validation_indices
        )

        print(f"\n{name}")
        print(
            "Training patients:",
            len(hospital["y_train"])
        )
        print(
            "Validation patients:",
            len(hospital["y_val"])
        )

    global_model, scaler = train_validation_model(
        hospital_splits,
        rounds=30,
        local_epochs=5
    )

    results_df = evaluate_hospitals(
        hospital_splits,
        global_model,
        scaler
    )

    summary_df = calculate_reliability_summary(
        results_df
    )

    base_dir = Path(__file__).resolve().parent

    metrics_dir = base_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        metrics_dir / "hospital_reliability.csv",
        index=False
    )

    summary_df.to_csv(
        metrics_dir / "hospital_reliability_summary.csv",
        index=False
    )

    print("\nHOSPITAL-WISE RESULTS")

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nRELIABILITY SUMMARY")

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nSaved files:")
    print("metrics/hospital_reliability.csv")
    print("metrics/hospital_reliability_summary.csv")

    print(
        "\nHOSPITAL RELIABILITY EXPERIMENT "
        "COMPLETED SUCCESSFULLY!"
    )



# ============================================================
# IID VS NON-IID FEDERATED LEARNING EXPERIMENT
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from data import X_train, y_train


def create_iid_non_iid_partitions():

    # Reserve the same validation patients for both experiments.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=123
    )

    X_fit = X_fit.reset_index(drop=True)
    y_fit = y_fit.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    # Fit preprocessing only on training records.
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_val_scaled = scaler.transform(X_val)

    benign_indices = np.where(y_fit.to_numpy() == 0)[0]
    malignant_indices = np.where(y_fit.to_numpy() == 1)[0]

    rng = np.random.default_rng(42)
    rng.shuffle(benign_indices)
    rng.shuffle(malignant_indices)

    # IID-style: each hospital gets similar class proportions.
    iid_benign = np.array_split(benign_indices, 3)
    iid_malignant = np.array_split(malignant_indices, 3)

    iid_partitions = []

    for i in range(3):
        indices = np.concatenate([
            iid_benign[i],
            iid_malignant[i]
        ])
        rng.shuffle(indices)
        iid_partitions.append(indices)

    # Non-IID: distribute malignant patients unevenly.
    # Every patient still appears exactly once.
    malignant_counts = [
        len(malignant_indices) // 6,
        len(malignant_indices) // 3
    ]
    malignant_counts.append(
        len(malignant_indices) - sum(malignant_counts)
    )

    non_iid_partitions = []
    start = 0

    for i, count in enumerate(malignant_counts):
        malignant_group = malignant_indices[
            start:start + count
        ]
        start += count

        benign_group = iid_benign[i]

        indices = np.concatenate([
            benign_group,
            malignant_group
        ])
        rng.shuffle(indices)
        non_iid_partitions.append(indices)

    # Verify that each partition scheme uses all patients once.
    expected = np.arange(len(y_fit))

    for partitions in [iid_partitions, non_iid_partitions]:
        combined = np.concatenate(partitions)
        assert len(combined) == len(expected)
        assert np.array_equal(
            np.sort(combined),
            expected
        )

    return (
        X_fit_scaled,
        y_fit.to_numpy(),
        X_val_scaled,
        y_val.to_numpy(),
        iid_partitions,
        non_iid_partitions
    )


def train_partition_experiment(
    experiment_name,
    X_fit,
    y_fit,
    partitions,
    rounds=30,
    local_epochs=5
):

    clients = []

    print(f"\n{experiment_name} HOSPITAL DISTRIBUTION")

    for i, indices in enumerate(partitions):

        X_local = X_fit[indices]
        y_local = y_fit[indices]

        print(
            f"Hospital {i + 1}: "
            f"{len(indices)} patients, "
            f"{int(np.sum(y_local))} malignant, "
            f"{np.mean(y_local) * 100:.2f}% malignant"
        )

        clients.append(
            HospitalClient(
                name=f"{experiment_name}_Hospital_{i + 1}",
                X=X_local,
                y=y_local,
                learning_rate=0.05
            )
        )

    global_model = LogisticRegressionModel(
        n_features=X_fit.shape[1],
        learning_rate=0.05
    )

    parameters = global_model.get_parameters()

    history = []

    for round_number in range(1, rounds + 1):

        client_results = []

        for client in clients:
            result = client.train_local(
                parameters,
                epochs=local_epochs
            )
            client_results.append(result)

        parameters = federated_averaging(
            client_results
        )

        global_model.set_parameters(parameters)

        # This loss is measured on the training records.
        probabilities = np.clip(
            global_model.predict_proba(X_fit),
            1e-12,
            1 - 1e-12
        )

        training_loss = -np.mean(
            y_fit * np.log(probabilities)
            + (1 - y_fit) * np.log(1 - probabilities)
        )

        history.append({
            "experiment": experiment_name,
            "round": round_number,
            "training_loss": training_loss
        })

    return global_model, history


def evaluate_partition_model(
    experiment_name,
    model,
    X_val,
    y_val
):

    probabilities = model.predict_proba(X_val)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "experiment": experiment_name,
        "accuracy": accuracy_score(
            y_val, predictions
        ),
        "precision": precision_score(
            y_val, predictions,
            zero_division=0
        ),
        "malignant_recall": recall_score(
            y_val, predictions,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_val, predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_val, probabilities
        ),
        "false_negatives": int(
            np.sum((y_val == 1) & (predictions == 0))
        )
    }


def run_iid_non_iid_experiment(output_dir=None):

    print("\nIID VS NON-IID FEDERATED LEARNING")

    (
        X_fit,
        y_fit,
        X_val,
        y_val,
        iid_partitions,
        non_iid_partitions
    ) = create_iid_non_iid_partitions()

    all_results = []
    all_history = []

    for name, partitions in [
        ("IID", iid_partitions),
        ("Non-IID", non_iid_partitions)
    ]:

        model, history = train_partition_experiment(
            name,
            X_fit,
            y_fit,
            partitions
        )

        results = evaluate_partition_model(
            name,
            model,
            X_val,
            y_val
        )

        all_results.append(results)
        all_history.extend(history)

    results_df = pd.DataFrame(all_results)
    history_df = pd.DataFrame(all_history)

    metrics_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        metrics_dir / "iid_non_iid_comparison.csv",
        index=False
    )

    history_df.to_csv(
        metrics_dir / "iid_non_iid_convergence.csv",
        index=False
    )

    print("\nIID VS NON-IID COMPARISON")
    print(results_df.to_string(index=False))

    print("\nFINAL TRAINING LOSSES")
    print(
        history_df.groupby("experiment")[
            "training_loss"
        ].last()
    )

    print("\nIID VS NON-IID EXPERIMENT COMPLETED!")


if __name__ == "__main__":
    run_iid_non_iid_experiment()
