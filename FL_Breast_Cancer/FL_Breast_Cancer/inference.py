from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from feature_schema import FEATURES


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

MODEL_ARTIFACTS = {
    "Centralized Logistic Regression": {
        "kind": "pipeline",
        "model": "centralized_logistic_regression.pkl",
    },
    "FedAvg Logistic Regression": {
        "kind": "linear",
        "model": "fedavg_model.npz",
        "scaler": "fedavg_scaler.pkl",
    },
    "DP Federated Gradient Model": {
        "kind": "linear",
        "model": "dp_federated_model.npz",
        "scaler": "dp_federated_scaler.pkl",
    },
}


def validate_feature_frame(frame):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if frame.columns.has_duplicates:
        raise ValueError("Input contains duplicate column names.")
    missing = [feature for feature in FEATURES if feature not in frame.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    supplied_order = [column for column in frame.columns if column in FEATURES]
    if supplied_order != FEATURES:
        raise ValueError("Feature columns must be in the canonical WDBC order.")
    try:
        features = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("All required features must be numeric.") from error
    values = features.to_numpy(dtype=float)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Features must contain at least one row of finite numeric values.")
    return features


def predict_probabilities(features, model_name=None, *, active_path=None):
    feature_frame = validate_feature_frame(features)
    if active_path is not None:
        pointer_path = Path(active_path)
        if not pointer_path.exists():
            raise FileNotFoundError(f"Active model pointer not found: {pointer_path}")
        import json

        active = json.loads(pointer_path.read_text(encoding="utf-8"))
        artifact_path = MODEL_DIR / active["model"]
        kind = active["kind"]
        scaler_path = MODEL_DIR / active["scaler"] if active.get("scaler") else None
        pipeline = joblib.load(artifact_path) if kind == "pipeline" else None
        if kind == "linear":
            with np.load(artifact_path) as model:
                weights = model["weights"]
                bias = float(model["bias"])
            scaler = joblib.load(scaler_path)
            probabilities = 1.0 / (
                1.0 + np.exp(-np.clip(scaler.transform(feature_frame) @ weights + bias, -500, 500))
            )
        elif kind == "pipeline":
            probabilities = pipeline.predict_proba(feature_frame)[:, 1]
        else:
            raise ValueError(f"Unsupported active model kind: {kind}")
    else:
        if model_name not in MODEL_ARTIFACTS:
            raise ValueError(f"Unknown model: {model_name}")
        artifact = MODEL_ARTIFACTS[model_name]
        model_path = MODEL_DIR / artifact["model"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        if artifact["kind"] == "pipeline":
            probabilities = joblib.load(model_path).predict_proba(feature_frame)[:, 1]
        else:
            scaler_path = MODEL_DIR / artifact["scaler"]
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")
            with np.load(model_path) as model:
                weights = model["weights"]
                bias = float(model["bias"])
            scaler = joblib.load(scaler_path)
            probabilities = 1.0 / (
                1.0 + np.exp(-np.clip(scaler.transform(feature_frame) @ weights + bias, -500, 500))
            )

    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if (
        len(probabilities) != len(feature_frame)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("Model returned invalid probabilities.")
    return probabilities
