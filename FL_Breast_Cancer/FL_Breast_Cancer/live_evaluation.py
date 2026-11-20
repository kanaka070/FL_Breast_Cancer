"""On-demand evaluations using the dashboard's simulated client training path."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import app_services
from dp_accountant import DPAccountant
from feature_schema import FEATURES
from model import LogisticRegressionModel
from workflow_utils import federated_averaging, partition_hospital_indices


def _validate_settings(rounds, local_epochs, clipping_norm, noise_multipliers, delta):
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("Evaluation rounds must be a positive integer.")
    if not isinstance(local_epochs, int) or local_epochs < 1:
        raise ValueError("Local epochs must be a positive integer.")
    if not np.isfinite(clipping_norm) or clipping_norm <= 0:
        raise ValueError("Clipping norm must be finite and positive.")
    if not noise_multipliers or any(
        not np.isfinite(value) or value <= 0 for value in noise_multipliers
    ):
        raise ValueError("Noise multipliers must be finite and positive.")
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("Delta must be finite and between 0 and 1.")


def _iid_partitions(labels, seed=42):
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    partitions = [[] for _ in range(len(app_services.HOSPITALS))]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for partition, group in zip(partitions, np.array_split(indices, len(partitions))):
            partition.extend(group.tolist())
    result = [np.asarray(partition, dtype=int) for partition in partitions]
    for partition in result:
        rng.shuffle(partition)
    combined = np.concatenate(result)
    if len(combined) != len(labels) or len(np.unique(combined)) != len(labels):
        raise RuntimeError("IID partition construction lost or duplicated training records.")
    return result


def _train_federated(
    X,
    y,
    partitions,
    *,
    rounds,
    local_epochs,
    learning_rate,
    clipping_norm,
    noise_multiplier=None,
):
    model = LogisticRegressionModel(
        n_features=X.shape[1],
        learning_rate=learning_rate,
    )
    parameters = model.get_parameters()
    started = time.perf_counter()

    for _ in range(rounds):
        client_updates = []
        for indices in partitions:
            client_X = X[indices]
            client_y = y[indices]
            if noise_multiplier is None:
                local_model = LogisticRegressionModel(
                    n_features=X.shape[1],
                    learning_rate=learning_rate,
                )
                local_model.set_parameters(parameters)
                local_model.train(client_X, client_y, epochs=local_epochs)
                local_parameters = local_model.get_parameters()
            else:
                local_parameters, _ = app_services._train_dp_local(
                    client_X,
                    client_y,
                    parameters,
                    epochs=local_epochs,
                    learning_rate=learning_rate,
                    clip=clipping_norm,
                    sigma=noise_multiplier,
                )
            client_updates.append({
                "num_patients": len(indices),
                "parameters": local_parameters,
            })
        parameters = federated_averaging(client_updates)

    model.set_parameters(parameters)
    return model, time.perf_counter() - started


def _metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "malignant_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def _conditional_accounting(noise_multiplier, delta, rounds, local_epochs):
    accountant = DPAccountant(
        noise_multiplier=noise_multiplier,
        delta=delta,
        sensitivity=1.0,
    )
    accountant.record_steps(
        len(app_services.HOSPITALS) * rounds * local_epochs
    )
    return {
        "epsilon_upper_bound": accountant.get_epsilon(),
        "delta": delta,
        "accounted_client_epoch_mechanisms": accountant.steps,
    }


def _evaluate_row(
    *,
    approach,
    partition,
    model,
    X_validation,
    y_validation,
    elapsed,
    training_records,
    validation_records,
    rounds,
    local_epochs,
    noise_multiplier=None,
    clipping_norm=None,
    delta=None,
):
    row = {
        "approach": approach,
        "partition": partition,
        "training_records": training_records,
        "validation_records": validation_records,
        "rounds": rounds if approach != "Centralized Logistic Regression" else None,
        "local_epochs": local_epochs if approach != "Centralized Logistic Regression" else None,
        "noise_multiplier": noise_multiplier,
        "clipping_norm": clipping_norm,
        "training_time_seconds": float(elapsed),
    }
    row.update(_metrics(y_validation, model.predict_proba(X_validation)))
    if noise_multiplier is not None:
        row.update(_conditional_accounting(
            noise_multiplier,
            delta,
            rounds,
            local_epochs,
        ))
    else:
        row.update({
            "epsilon_upper_bound": None,
            "delta": None,
            "accounted_client_epoch_mechanisms": None,
        })
    return row


def run_live_model_comparison(
    *,
    rounds=30,
    local_epochs=5,
    learning_rate=0.05,
    clipping_norm=1.0,
    noise_multipliers=(0.5, 1.0, 2.0),
    delta=1e-5,
):
    """Compare Logistic Regression methods on the same untouched WDBC holdout."""
    _validate_settings(rounds, local_epochs, clipping_norm, noise_multipliers, delta)
    X_train, X_test, y_test, y_train = app_services._training_data()
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    labels = y_train.to_numpy(dtype=int)
    partitions = partition_hospital_indices(pd.Series(labels), seed=42)
    rows = []

    central_model = LogisticRegressionModel(
        n_features=len(FEATURES),
        learning_rate=learning_rate,
    )
    started = time.perf_counter()
    central_model.train(
        X_train_scaled,
        labels,
        epochs=rounds * local_epochs,
    )
    central_elapsed = time.perf_counter() - started
    rows.append(_evaluate_row(
        approach="Centralized Logistic Regression",
        partition="Not applicable",
        model=central_model,
        X_validation=X_test_scaled,
        y_validation=y_test.to_numpy(dtype=int),
        elapsed=central_elapsed,
        training_records=len(X_train),
        validation_records=len(X_test),
        rounds=rounds,
        local_epochs=local_epochs,
    ))

    fedavg_model, fedavg_elapsed = _train_federated(
        X_train_scaled,
        labels,
        partitions,
        rounds=rounds,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        clipping_norm=clipping_norm,
    )
    rows.append(_evaluate_row(
        approach="FedAvg Logistic Regression",
        partition="Non-IID",
        model=fedavg_model,
        X_validation=X_test_scaled,
        y_validation=y_test.to_numpy(dtype=int),
        elapsed=fedavg_elapsed,
        training_records=len(X_train),
        validation_records=len(X_test),
        rounds=rounds,
        local_epochs=local_epochs,
    ))

    for noise_multiplier in noise_multipliers:
        dp_model, dp_elapsed = _train_federated(
            X_train_scaled,
            labels,
            partitions,
            rounds=rounds,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            clipping_norm=clipping_norm,
            noise_multiplier=float(noise_multiplier),
        )
        rows.append(_evaluate_row(
            approach="DP-FedAvg Logistic Regression (experimental)",
            partition="Non-IID",
            model=dp_model,
            X_validation=X_test_scaled,
            y_validation=y_test.to_numpy(dtype=int),
            elapsed=dp_elapsed,
            training_records=len(X_train),
            validation_records=len(X_test),
            rounds=rounds,
            local_epochs=local_epochs,
            noise_multiplier=float(noise_multiplier),
            clipping_norm=clipping_norm,
            delta=delta,
        ))

    results = pd.DataFrame(rows)
    metadata = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (local WDBC)",
        "split": "stratified 80/20 split; random_state=42",
        "training_records": int(len(X_train)),
        "held_out_records": int(len(X_test)),
        "scaler_fit_records": int(len(X_train)),
        "hospital_count": len(app_services.HOSPITALS),
        "hospital_partitions": [
            {
                "hospital": app_services.HOSPITALS[hospital_id],
                "records": int(len(indices)),
            }
            for hospital_id, indices in zip(app_services.HOSPITALS, partitions)
        ],
        "rounds": rounds,
        "local_epochs": local_epochs,
        "learning_rate": learning_rate,
        "clipping_norm": clipping_norm,
        "noise_multipliers": [float(value) for value in noise_multipliers],
        "delta": delta,
        "evaluation": (
            "One fixed held-out test split for all rows. Do not use these exploratory "
            "test results to tune or select a deployment model."
        ),
    }
    return results, metadata


def run_iid_noniid_validation(
    *,
    rounds=30,
    local_epochs=5,
    learning_rate=0.05,
    clipping_norm=1.0,
    noise_multipliers=(0.5, 1.0, 2.0),
    delta=1e-5,
):
    """Evaluate partition heterogeneity and DP utility on training-only validation."""
    _validate_settings(rounds, local_epochs, clipping_norm, noise_multipliers, delta)
    X_development, _, _, y_development = app_services._training_data()
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.2,
        random_state=123,
        stratify=y_development,
    )
    scaler = StandardScaler().fit(X_fit)
    X_fit_scaled = scaler.transform(X_fit)
    X_validation_scaled = scaler.transform(X_validation)
    labels = y_fit.to_numpy(dtype=int)
    partition_sets = {
        "IID": _iid_partitions(labels, seed=42),
        "Non-IID": partition_hospital_indices(pd.Series(labels), seed=42),
    }
    rows = []
    for partition_name, partitions in partition_sets.items():
        fedavg_model, fedavg_elapsed = _train_federated(
            X_fit_scaled,
            labels,
            partitions,
            rounds=rounds,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            clipping_norm=clipping_norm,
        )
        rows.append(_evaluate_row(
            approach="FedAvg Logistic Regression",
            partition=partition_name,
            model=fedavg_model,
            X_validation=X_validation_scaled,
            y_validation=y_validation.to_numpy(dtype=int),
            elapsed=fedavg_elapsed,
            training_records=len(X_fit),
            validation_records=len(X_validation),
            rounds=rounds,
            local_epochs=local_epochs,
        ))
        for noise_multiplier in noise_multipliers:
            dp_model, dp_elapsed = _train_federated(
                X_fit_scaled,
                labels,
                partitions,
                rounds=rounds,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                clipping_norm=clipping_norm,
                noise_multiplier=float(noise_multiplier),
            )
            rows.append(_evaluate_row(
                approach="DP-FedAvg Logistic Regression (experimental)",
                partition=partition_name,
                model=dp_model,
                X_validation=X_validation_scaled,
                y_validation=y_validation.to_numpy(dtype=int),
                elapsed=dp_elapsed,
                training_records=len(X_fit),
                validation_records=len(X_validation),
                rounds=rounds,
                local_epochs=local_epochs,
                noise_multiplier=float(noise_multiplier),
                clipping_norm=clipping_norm,
                delta=delta,
            ))

    metadata = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (local WDBC)",
        "split": (
            "The fixed 114-record final test split is excluded. A stratified 80/20 "
            "split of the 455-record development partition creates fit and validation sets."
        ),
        "training_records": int(len(X_fit)),
        "validation_records": int(len(X_validation)),
        "scaler_fit_records": int(len(X_fit)),
        "hospital_count": len(app_services.HOSPITALS),
        "hospital_partitions": {
            name: [int(len(indices)) for indices in partitions]
            for name, partitions in partition_sets.items()
        },
        "rounds": rounds,
        "local_epochs": local_epochs,
        "learning_rate": learning_rate,
        "clipping_norm": clipping_norm,
        "noise_multipliers": [float(value) for value in noise_multipliers],
        "delta": delta,
        "evaluation": (
            "Training-only validation comparison. Results are exploratory and should "
            "not be used to tune on the final test partition."
        ),
    }
    return pd.DataFrame(rows), metadata
