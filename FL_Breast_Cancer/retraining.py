import json
import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_schema import FEATURES
from inference import MODEL_ARTIFACTS, MODEL_DIR, predict_probabilities, validate_feature_frame


BASE_DIR = Path(__file__).resolve().parent
RECORDS_PATH = BASE_DIR / "data" / "confirmed_records.csv"
CANDIDATE_DIR = MODEL_DIR / "candidates"
ACTIVE_MODEL_PATH = MODEL_DIR / "active_model.json"
DEFAULT_ACTIVE_MODEL = "FedAvg Logistic Regression"


def _require_admin(principal):
    from app_services import _assert_principal

    _assert_principal(principal, "admin")


def _next_model_version():
    from app_services import _connect

    with _connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM audit_log WHERE action = 'model_approved'"
        ).fetchone()[0]
    return f"v{int(count) + 1:04d}"


def _record_admin_action(principal, action, details):
    from app_services import _audit

    _audit(principal, action, details)


def _atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _labels_from_frame(frame):
    if "diagnosis" not in frame.columns:
        raise ValueError("A 'diagnosis' label column is required; predictions are not labels.")
    labels = frame["diagnosis"].astype(str).str.strip().str.upper()
    mapped = labels.map({"B": 0, "M": 1, "0": 0, "1": 1})
    if mapped.isna().any():
        raise ValueError("Diagnosis values must be B, M, 0, or 1.")
    return mapped.astype(int)


def _record_fingerprints(features, labels):
    fingerprints = []
    values = features.to_numpy(dtype="<f8")
    for row, label in zip(values, labels.to_numpy(dtype=np.int8)):
        digest = hashlib.sha256()
        digest.update(row.tobytes())
        digest.update(bytes([int(label)]))
        fingerprints.append(digest.hexdigest())
    return fingerprints


def register_confirmed_records(frame, *, admin_confirmed, principal=None):
    if not admin_confirmed:
        raise PermissionError("An explicit administrator confirmation is required.")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Confirmed records must be provided as a DataFrame.")
    if "diagnosis" not in frame.columns:
        raise ValueError("A 'diagnosis' label column is required; prediction outputs cannot be registered.")
    features = validate_feature_frame(frame)
    labels = _labels_from_frame(frame)
    _require_admin(principal)
    fingerprints = _record_fingerprints(features, labels)
    existing = pd.DataFrame()
    if RECORDS_PATH.exists():
        existing = pd.read_csv(RECORDS_PATH)
        required = {*FEATURES, "diagnosis", "record_id", "consumed_candidate"}
        if not required.issubset(existing.columns):
            raise ValueError("Confirmed-record ledger is missing required columns.")
        if "record_fingerprint" not in existing.columns:
            existing_features = validate_feature_frame(existing)
            existing_labels = _labels_from_frame(existing)
            existing["record_fingerprint"] = _record_fingerprints(
                existing_features, existing_labels
            )
    known_fingerprints = (
        set(existing["record_fingerprint"].dropna().astype(str))
        if not existing.empty
        else set()
    )
    accepted = []
    seen = set(known_fingerprints)
    for index, fingerprint in enumerate(fingerprints):
        if fingerprint not in seen:
            accepted.append(index)
            seen.add(fingerprint)
    if not accepted:
        return 0

    features = features.iloc[accepted].reset_index(drop=True)
    labels = labels.iloc[accepted].reset_index(drop=True)
    fingerprints = [fingerprints[index] for index in accepted]
    records = features.copy()
    records["diagnosis"] = labels.to_numpy()
    records["record_id"] = [uuid.uuid4().hex for _ in range(len(records))]
    records["record_fingerprint"] = fingerprints
    records["confirmed_at"] = datetime.now(timezone.utc).isoformat()
    records["consumed_candidate"] = ""

    if not existing.empty:
        records = pd.concat([existing, records], ignore_index=True)
    _atomic_write_csv(records, RECORDS_PATH)
    _record_admin_action(principal, "confirmed_records_registered", {"count": len(features)})
    return len(features)


def pending_confirmed_records(principal):
    _require_admin(principal)
    if not RECORDS_PATH.exists():
        return pd.DataFrame(columns=[*FEATURES, "diagnosis", "record_id", "consumed_candidate"])
    records = pd.read_csv(RECORDS_PATH)
    required = {*FEATURES, "diagnosis", "record_id", "consumed_candidate"}
    if not required.issubset(records.columns):
        raise ValueError("Confirmed-record ledger is missing required columns.")
    pending = records[records["consumed_candidate"].isna() | (records["consumed_candidate"] == "")]
    return pending.reset_index(drop=True)


def _active_descriptor():
    if ACTIVE_MODEL_PATH.exists():
        return json.loads(ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))
    artifact = MODEL_ARTIFACTS[DEFAULT_ACTIVE_MODEL]
    return {
        "name": DEFAULT_ACTIVE_MODEL,
        "kind": artifact["kind"],
        "model": artifact["model"],
        "scaler": artifact.get("scaler"),
    }


def _active_model_snapshot():
    descriptor = _active_descriptor()
    return {
        key: descriptor.get(key)
        for key in ("name", "kind", "model", "scaler", "candidate_id")
    }


def approve_baseline_model(model_name, *, admin_approved, principal=None):
    if not admin_approved:
        raise PermissionError("Explicit administrator approval is required.")
    _require_admin(principal)
    if model_name not in MODEL_ARTIFACTS:
        raise ValueError(f"Unknown model: {model_name}")
    artifact = MODEL_ARTIFACTS[model_name]
    model_path = MODEL_DIR / artifact["model"]
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    scaler = artifact.get("scaler")
    if scaler and not (MODEL_DIR / scaler).is_file():
        raise FileNotFoundError(f"Scaler artifact not found: {MODEL_DIR / scaler}")
    active = {
        "name": model_name,
        "kind": artifact["kind"],
        "model": artifact["model"],
        "scaler": scaler,
        "version": _next_model_version(),
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ACTIVE_MODEL_PATH, active)
    _record_admin_action(
        principal,
        "model_approved",
        {"name": model_name, "version": active["version"]},
    )
    return active


def _metrics(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "malignant_recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
    }
    result["roc_auc"] = (
        float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else None
    )
    return result


def create_candidate(minimum_records=100, *, random_state=42, principal=None):
    if not isinstance(minimum_records, int) or minimum_records < 6:
        raise ValueError("The retraining threshold must be an integer of at least 6.")
    _require_admin(principal)
    records = pending_confirmed_records(principal)
    if len(records) < minimum_records:
        raise ValueError(
            f"Only {len(records)} newly confirmed records are available; "
            f"the configured threshold is {minimum_records}."
        )
    labels = _labels_from_frame(records)
    if labels.value_counts().min() < 2:
        raise ValueError("At least two confirmed records from each diagnosis class are required.")
    features = validate_feature_frame(records)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    candidate.fit(X_train, y_train)

    candidate_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    artifact_name = f"{candidate_id}.joblib"
    artifact_path = CANDIDATE_DIR / artifact_name
    joblib.dump(candidate, artifact_path)

    candidate_probabilities = candidate.predict_proba(X_test)[:, 1]
    active_snapshot = _active_model_snapshot()
    active_probabilities = predict_probabilities(
        X_test,
        active_path=ACTIVE_MODEL_PATH if ACTIVE_MODEL_PATH.exists() else None,
        model_name=None if ACTIVE_MODEL_PATH.exists() else DEFAULT_ACTIVE_MODEL,
    )
    manifest = {
        "candidate_id": candidate_id,
        "status": "pending_approval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(Path("candidates") / artifact_name),
        "kind": "pipeline",
        "training_records": int(len(X_train)),
        "evaluation_records": int(len(X_test)),
        "metrics": {
            "candidate": _metrics(y_test, candidate_probabilities),
            "active": _metrics(y_test, active_probabilities),
        },
        "active_model": active_snapshot["name"],
        "active_model_snapshot": active_snapshot,
    }
    manifest_path = CANDIDATE_DIR / f"{candidate_id}.json"
    _atomic_write_json(manifest_path, manifest)
    all_records = pd.read_csv(RECORDS_PATH)
    all_records["consumed_candidate"] = (
        all_records["consumed_candidate"].fillna("").astype(object)
    )
    all_records.loc[all_records["record_id"].isin(records["record_id"]), "consumed_candidate"] = candidate_id
    _atomic_write_csv(all_records, RECORDS_PATH)
    _record_admin_action(
        principal,
        "retraining_candidate_created",
        {"candidate_id": candidate_id, "metrics": manifest["metrics"]},
    )
    return manifest


def approve_candidate(candidate_id, *, admin_approved, principal=None):
    if not admin_approved:
        raise PermissionError("Explicit administrator approval is required.")
    _require_admin(principal)
    manifest_path = CANDIDATE_DIR / f"{candidate_id}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Candidate not found: {candidate_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pending_approval":
        raise ValueError("Only a pending candidate can be approved.")
    active_snapshot = _active_model_snapshot()
    expected_snapshot = manifest.get("active_model_snapshot")
    if expected_snapshot is not None:
        if expected_snapshot != active_snapshot:
            raise ValueError("The active model changed after candidate evaluation; retrain and review again.")
    elif manifest.get("active_model") != active_snapshot["name"]:
        raise ValueError("The active model changed after candidate evaluation; retrain and review again.")
    model_path = CANDIDATE_DIR / Path(manifest["model"]).name
    if not model_path.is_file():
        raise FileNotFoundError(f"Candidate artifact not found: {model_path}")
    active = {
        "name": f"Approved candidate {candidate_id}",
        "kind": "pipeline",
        "model": str(Path("candidates") / model_path.name),
        "candidate_id": candidate_id,
        "version": _next_model_version(),
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ACTIVE_MODEL_PATH, active)
    manifest["status"] = "approved"
    manifest["approved_at"] = active["approved_at"]
    _atomic_write_json(manifest_path, manifest)
    _record_admin_action(
        principal,
        "model_approved",
        {"candidate_id": candidate_id, "version": active["version"]},
    )
    return active
