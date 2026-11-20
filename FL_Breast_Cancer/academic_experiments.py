"""Reproducible research evaluations that preserve saved baseline artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_schema import FEATURES
from model import LogisticRegressionModel
from workflow_utils import (
    partition_hospital_indices,
    federated_averaging,
    trust_weighted_averaging,
)


PROJECT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
MODEL_NAME = "LogisticRegression(C=1.0, solver=lbfgs)"


def load_wdbc():
    frame = pd.read_csv(PROJECT_DIR / "data" / "wdbc.data", header=None)
    if frame.shape[1] != len(FEATURES) + 2:
        raise ValueError("Expected WDBC ID, diagnosis, and 30 feature columns.")
    labels = frame.iloc[:, 1].map({"B": 0, "M": 1})
    if labels.isna().any():
        raise ValueError("WDBC diagnosis labels contain unknown values.")
    features = frame.iloc[:, 2:].copy()
    features.columns = FEATURES
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("WDBC features must all be finite.")
    return features, labels.astype(int)


def classification_metrics(y_true, probabilities, elapsed_seconds):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Evaluation probabilities must be finite values in [0, 1].")
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "malignant_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "training_time_seconds": float(elapsed_seconds),
    }


def logistic_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])


def run_cross_validation(output_dir: Path):
    """Run CV only on the original 80% training partition; holdout is not evaluated."""
    features, labels = load_wdbc()
    X_train, _, y_train, _ = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    protocols = [
        ("stratified_5_fold", StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)),
        ("stratified_10_fold", StratifiedKFold(10, shuffle=True, random_state=RANDOM_STATE)),
        ("unstratified_10_fold", KFold(10, shuffle=True, random_state=RANDOM_STATE)),
    ]
    fold_rows = []
    for protocol, splitter in protocols:
        for fold_number, (train_indices, validation_indices) in enumerate(
            splitter.split(X_train, y_train if protocol.startswith("stratified") else None),
            start=1,
        ):
            estimator = logistic_pipeline()
            started = time.perf_counter()
            estimator.fit(X_train.iloc[train_indices], y_train.iloc[train_indices])
            elapsed = time.perf_counter() - started
            probabilities = estimator.predict_proba(X_train.iloc[validation_indices])[:, 1]
            row = {
                "protocol": protocol,
                "fold": fold_number,
                "train_records": len(train_indices),
                "validation_records": len(validation_indices),
                "model": MODEL_NAME,
            }
            row.update(classification_metrics(
                y_train.iloc[validation_indices],
                probabilities,
                elapsed,
            ))
            fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    metric_columns = [
        "accuracy", "precision", "malignant_recall", "specificity",
        "f1_score", "roc_auc", "training_time_seconds",
    ]
    summary_rows = []
    for protocol, group in folds.groupby("protocol", sort=False):
        row = {
            "protocol": protocol,
            "folds": len(group),
            "model": MODEL_NAME,
            "training_records": len(X_train),
            "held_out_test_evaluated": False,
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_dir / "cross_validation_folds.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "cross_validation_summary.csv", index=False)
    return folds, pd.DataFrame(summary_rows)


def _federated_cv_probabilities(X_train, y_train, X_validation, partitions):
    model = LogisticRegressionModel(
        n_features=X_train.shape[1],
        learning_rate=0.05,
    )
    parameters = model.get_parameters()
    started = time.perf_counter()
    for _ in range(30):
        client_results = []
        for indices in partitions:
            local_model = LogisticRegressionModel(
                n_features=X_train.shape[1],
                learning_rate=0.05,
            )
            local_model.set_parameters(parameters)
            local_model.train(X_train[indices], y_train[indices], epochs=5)
            client_results.append({
                "num_patients": len(indices),
                "parameters": local_model.get_parameters(),
            })
        parameters = federated_averaging(client_results)
        model.set_parameters(parameters)
    elapsed = time.perf_counter() - started
    return model.predict_proba(X_validation), elapsed


def _dp_federated_cv_probabilities(
    X_train,
    y_train,
    X_validation,
    partitions,
    *,
    clipping_norm=1.0,
    noise_multiplier=2.0,
):
    weights = np.zeros(X_train.shape[1], dtype=float)
    bias = 0.0
    total_records = len(y_train)
    sensitivity = 2.0 * clipping_norm / total_records
    noise_std = noise_multiplier * sensitivity
    rng = secrets.SystemRandom()
    started = time.perf_counter()

    for _ in range(30):
        summed_gradient = np.zeros(X_train.shape[1] + 1, dtype=float)
        for indices in partitions:
            local_X = X_train[indices]
            local_y = y_train[indices]
            logits = np.clip(local_X @ weights + bias, -35, 35)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            errors = probabilities - local_y
            gradients = np.concatenate(
                [errors[:, None] * local_X, errors[:, None]],
                axis=1,
            )
            norms = np.linalg.norm(gradients, axis=1, keepdims=True)
            clipping_factors = np.minimum(
                1.0,
                clipping_norm / np.maximum(norms, 1e-12),
            )
            summed_gradient += np.sum(gradients * clipping_factors, axis=0)

        average_gradient = summed_gradient / total_records
        noise = np.fromiter(
            (
                rng.gauss(0.0, noise_std)
                for _ in range(X_train.shape[1] + 1)
            ),
            dtype=float,
            count=X_train.shape[1] + 1,
        )
        noisy_gradient = average_gradient + noise
        weights -= 0.05 * noisy_gradient[:-1]
        bias -= 0.05 * noisy_gradient[-1]

    elapsed = time.perf_counter() - started
    validation_logits = np.clip(X_validation @ weights + bias, -500, 500)
    return 1.0 / (1.0 + np.exp(-validation_logits)), elapsed


def run_federated_cross_validation(output_dir: Path):
    """Evaluate the existing FedAvg and experimental DP updates on train folds."""
    features, labels = load_wdbc()
    X_train, _, y_train, _ = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    protocols = [
        ("stratified_5_fold", StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)),
        ("stratified_10_fold", StratifiedKFold(10, shuffle=True, random_state=RANDOM_STATE)),
        ("unstratified_10_fold", KFold(10, shuffle=True, random_state=RANDOM_STATE)),
    ]
    folds = []
    for protocol, splitter in protocols:
        split_y = y_train if protocol.startswith("stratified") else None
        for fold_number, (train_indices, validation_indices) in enumerate(
            splitter.split(X_train, split_y),
            start=1,
        ):
            X_fit_frame = X_train.iloc[train_indices]
            y_fit = y_train.iloc[train_indices].to_numpy(dtype=int)
            X_validation_frame = X_train.iloc[validation_indices]
            y_validation = y_train.iloc[validation_indices].to_numpy(dtype=int)
            scaler = StandardScaler().fit(X_fit_frame)
            X_fit = scaler.transform(X_fit_frame)
            X_validation = scaler.transform(X_validation_frame)
            partitions = partition_hospital_indices(
                pd.Series(y_fit),
                seed=RANDOM_STATE + fold_number,
            )

            for approach, trainer in (
                ("FedAvg", _federated_cv_probabilities),
                ("DP-FedAvg (experimental)", _dp_federated_cv_probabilities),
            ):
                if approach == "FedAvg":
                    probabilities, elapsed = trainer(
                        X_fit,
                        y_fit,
                        X_validation,
                        partitions,
                    )
                else:
                    probabilities, elapsed = trainer(
                        X_fit,
                        y_fit,
                        X_validation,
                        partitions,
                    )
                row = {
                    "protocol": protocol,
                    "fold": fold_number,
                    "approach": approach,
                    "train_records": len(train_indices),
                    "validation_records": len(validation_indices),
                    "client_count": len(partitions),
                    "rounds": 30,
                    "local_epochs": 5 if approach == "FedAvg" else None,
                    "clipping_norm": 1.0 if approach.startswith("DP-") else None,
                    "noise_multiplier": 2.0 if approach.startswith("DP-") else None,
                }
                row.update(classification_metrics(y_validation, probabilities, elapsed))
                folds.append(row)

    fold_frame = pd.DataFrame(folds)
    metric_columns = [
        "accuracy", "precision", "malignant_recall", "specificity",
        "f1_score", "roc_auc", "training_time_seconds",
    ]
    summaries = []
    for (protocol, approach), group in fold_frame.groupby(
        ["protocol", "approach"],
        sort=False,
    ):
        row = {
            "protocol": protocol,
            "approach": approach,
            "folds": len(group),
            "training_records": len(X_train),
            "held_out_test_evaluated": False,
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summaries.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(output_dir / "federated_cross_validation_folds.csv", index=False)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(
        output_dir / "federated_cross_validation_summary.csv",
        index=False,
    )
    return fold_frame, summary_frame


def _local_trust_and_model(X, y, *, seed):
    if len(np.unique(y)) != 2 or min(np.bincount(y.astype(int))) < 2:
        raise ValueError("Each simulated hospital needs at least two records from each class.")
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )
    validation_model = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=seed,
    ).fit(X_fit, y_fit)
    trust = float(balanced_accuracy_score(
        y_validation,
        validation_model.predict(X_validation),
    ))
    final_model = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=seed,
    ).fit(X, y)
    return trust, final_model


def run_trust_weighting(output_dir: Path):
    """Evaluate trust-weighted coefficient aggregation on the untouched test set."""
    features, labels = load_wdbc()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    hospital_indices = partition_hospital_indices(
        y_train.reset_index(drop=True),
        seed=RANDOM_STATE,
    )

    local_results = []
    trust_rows = []
    started = time.perf_counter()
    for index, patient_indices in enumerate(hospital_indices):
        local_X = X_train_scaled[patient_indices]
        local_y = y_train.to_numpy()[patient_indices]
        trust, model = _local_trust_and_model(local_X, local_y, seed=RANDOM_STATE + index)
        local_results.append({
            "num_patients": len(patient_indices),
            "trust_score": trust,
            "parameters": {
                "weights": model.coef_.reshape(-1),
                "bias": float(model.intercept_[0]),
            },
        })
        trust_rows.append({
            "hospital": f"Simulated Hospital {index + 1}",
            "training_records": len(patient_indices),
            "local_validation_balanced_accuracy_trust": trust,
        })

    trust_mass = sum(item["num_patients"] * item["trust_score"] for item in local_results)
    for row, item in zip(trust_rows, local_results):
        row["normalized_trust_weight"] = item["num_patients"] * item["trust_score"] / trust_mass

    ordinary = federated_averaging(local_results)
    trusted = trust_weighted_averaging(local_results)
    elapsed = time.perf_counter() - started
    result_rows = []
    for aggregation_name, parameters in (
        ("sample_size_fedavg", ordinary),
        ("trust_weighted_fedavg", trusted),
    ):
        logits = X_test_scaled @ parameters["weights"] + parameters["bias"]
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        row = {
            "experiment": aggregation_name,
            "model": MODEL_NAME,
            "seed": RANDOM_STATE,
            "training_time_seconds": elapsed,
            "test_records": len(y_test),
            "held_out_test_used_for_selection": False,
        }
        row.update(classification_metrics(y_test, probabilities, elapsed))
        result_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trust_rows).to_csv(output_dir / "trust_weights.csv", index=False)
    pd.DataFrame(result_rows).to_csv(output_dir / "trust_aggregation_test_metrics.csv", index=False)
    (output_dir / "trust_aggregation_method.json").write_text(
        json.dumps({
            "trust_score": "client-local holdout balanced accuracy from a stratified 75/25 split",
            "aggregation_formula": "alpha_i = n_i * trust_i / sum_j(n_j * trust_j)",
            "parameter_aggregation": "global_coefficients = sum_i(alpha_i * local_coefficients_i); global_intercept = sum_i(alpha_i * local_intercept_i)",
            "preprocessing": "one StandardScaler fit only on the training partition; identical coordinate system for all clients",
            "validation_use": "per-client trust holdouts are separate subsets of training records; the fixed WDBC test partition is not used to choose weights",
            "privacy_note": "local trust scores and model coefficients are not differentially private",
        }, indent=2),
        encoding="utf-8",
    )
    return pd.DataFrame(trust_rows), pd.DataFrame(result_rows)


def _build_iid_noniid_partitions(labels, seed=RANDOM_STATE):
    labels = np.asarray(labels)
    benign = np.flatnonzero(labels == 0)
    malignant = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    rng.shuffle(benign)
    rng.shuffle(malignant)
    benign_groups = np.array_split(benign, 3)
    iid_malignant_groups = np.array_split(malignant, 3)
    iid = []
    for hospital in range(3):
        indices = np.concatenate([benign_groups[hospital], iid_malignant_groups[hospital]])
        rng.shuffle(indices)
        iid.append(indices)
    malignant_counts = [
        len(malignant) // 6,
        len(malignant) // 3,
        len(malignant) - len(malignant) // 6 - len(malignant) // 3,
    ]
    non_iid = []
    start = 0
    for hospital, count in enumerate(malignant_counts):
        local_malignant = malignant[start:start + count]
        start += count
        indices = np.concatenate([benign_groups[hospital], local_malignant])
        rng.shuffle(indices)
        non_iid.append(indices)
    expected = np.arange(len(labels))
    for partitions in (iid, non_iid):
        combined = np.concatenate(partitions)
        if len(combined) != len(expected) or not np.array_equal(np.sort(combined), expected):
            raise RuntimeError("Partition construction lost or duplicated training records.")
    return iid, non_iid


def _train_federated_gradient(X_train, y_train, partitions, rounds=30, local_epochs=5):
    model = LogisticRegressionModel(n_features=X_train.shape[1], learning_rate=0.05)
    parameters = model.get_parameters()
    history = []
    started = time.perf_counter()
    for round_number in range(1, rounds + 1):
        client_results = []
        for indices in partitions:
            local_model = LogisticRegressionModel(
                n_features=X_train.shape[1],
                learning_rate=0.05,
            )
            local_model.set_parameters(parameters)
            local_model.train(X_train[indices], y_train[indices], epochs=local_epochs)
            client_results.append({
                "num_patients": len(indices),
                "parameters": local_model.get_parameters(),
            })
        parameters = federated_averaging(client_results)
        model.set_parameters(parameters)
        history.append({
            "round": round_number,
            "training_loss": model.calculate_loss(X_train, y_train),
        })
    return model, history, time.perf_counter() - started


def run_iid_noniid(output_dir: Path):
    """Reproduce the existing 30-round IID/non-IID analysis without touching metrics/."""
    features, labels = load_wdbc()
    X_development, _, y_development, _ = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.20,
        random_state=123,
        stratify=y_development,
    )
    scaler = StandardScaler().fit(X_fit)
    X_fit_scaled = scaler.transform(X_fit)
    X_validation_scaled = scaler.transform(X_validation)
    y_fit_array = y_fit.to_numpy(dtype=int)
    iid, non_iid = _build_iid_noniid_partitions(y_fit_array)
    rows = []
    histories = []
    for name, partitions in (("IID", iid), ("Non-IID", non_iid)):
        model, history, elapsed = _train_federated_gradient(
            X_fit_scaled,
            y_fit_array,
            partitions,
        )
        probabilities = model.predict_proba(X_validation_scaled)
        metrics = classification_metrics(y_validation, probabilities, elapsed)
        metrics.update({
            "experiment": name,
            "validation_records": len(y_validation),
            "seed": RANDOM_STATE,
            "rounds": 30,
            "local_epochs": 5,
            "held_out_test_evaluated": False,
        })
        rows.append(metrics)
        for item in history:
            histories.append({"experiment": name, **item})
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "iid_noniid_results.csv", index=False)
    pd.DataFrame(histories).to_csv(output_dir / "iid_noniid_training_history.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(histories)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "research_outputs" / "verified",
    )
    args = parser.parse_args()
    _, summary = run_cross_validation(args.output_dir)
    federated_folds, federated_summary = run_federated_cross_validation(args.output_dir)
    trust, trust_results = run_trust_weighting(args.output_dir)
    iid_noniid, _ = run_iid_noniid(args.output_dir)
    print("Cross-validation summary:")
    print(summary.to_string(index=False))
    print("\nFedAvg / DP-FedAvg cross-validation summary:")
    print(federated_summary.to_string(index=False))
    print("\nTrust weighting:")
    print(trust.to_string(index=False))
    print(trust_results.to_string(index=False))
    print("\nIID / non-IID validation results:")
    print(iid_noniid.to_string(index=False))
    dataset_digest = hashlib.sha256(
        (PROJECT_DIR / "data" / "wdbc.data").read_bytes()
    ).hexdigest()
    (args.output_dir / "execution_metadata.json").write_text(
        json.dumps({
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "seed": RANDOM_STATE,
            "dp_noise_randomness": (
                "OS-backed secrets.SystemRandom; DP metrics may vary between runs"
            ),
            "dataset": "UCI WDBC local data/wdbc.data",
            "dataset_sha256": dataset_digest,
            "outputs": [
                "cross_validation_folds.csv",
                "cross_validation_summary.csv",
                "federated_cross_validation_folds.csv",
                "federated_cross_validation_summary.csv",
                "trust_weights.csv",
                "trust_aggregation_test_metrics.csv",
                "trust_aggregation_method.json",
                "iid_noniid_results.csv",
                "iid_noniid_training_history.csv",
            ],
            "baseline_models_and_historical_metrics_overwritten": False,
            "federated_cv_folds": len(federated_folds),
            "held_out_test_evaluated_in_cv": False,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved results under {args.output_dir}")


if __name__ == "__main__":
    main()
