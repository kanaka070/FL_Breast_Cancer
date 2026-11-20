"""Authentication, tenant-scoped application state, and federated workflows."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dp_accountant import DPAccountant
from feature_schema import FEATURES
from inference import predict_probabilities, validate_feature_frame
from model import LogisticRegressionModel
from retraining import ACTIVE_MODEL_PATH
from workflow_utils import federated_averaging, partition_hospital_indices


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATE_PATH = DATA_DIR / "application_state.sqlite3"
MODEL_DIR = BASE_DIR / "saved_models"
CANDIDATE_DIR = MODEL_DIR / "candidates"
HOSPITALS = {
    "hospital_1": "Hospital 1",
    "hospital_2": "Hospital 2",
    "hospital_3": "Hospital 3",
}
DEMO_USER_PASSWORDS = {
    "admin": ("admin", None, "FL_ADMIN_PASSWORD"),
    "hospital1": ("hospital", "hospital_1", "FL_HOSPITAL1_PASSWORD"),
    "hospital2": ("hospital", "hospital_2", "FL_HOSPITAL2_PASSWORD"),
    "hospital3": ("hospital", "hospital_3", "FL_HOSPITAL3_PASSWORD"),
}
PASSWORD_ITERATIONS = 310_000
_PRINCIPAL_SECRET = secrets.token_bytes(32)


@dataclass(frozen=True)
class Principal:
    user_id: int
    username: str
    role: str
    hospital_id: str | None
    _proof: bytes = field(default=b"", repr=False, compare=False)


class SimulatedHospitalClient:
    """In-process demo client; returns updates, not patient feature rows."""

    def __init__(self, hospital_id: str):
        if hospital_id not in HOSPITALS:
            raise ValueError("Unknown simulated hospital.")
        self.hospital_id = hospital_id

    def train_update(self, scaler, parameters, config, mode):
        X_local, y_local = _hospital_training_partition(self.hospital_id)
        X_scaled = scaler.transform(pd.DataFrame(X_local, columns=FEATURES))
        if mode.startswith("DP"):
            updated_parameters, _ = _train_dp_local(
                X_scaled,
                y_local,
                parameters,
                epochs=config["local_epochs"],
                learning_rate=config["learning_rate"],
                clip=config["clipping_norm"],
                sigma=config["noise_multiplier"],
            )
            training_loss = None
            shared_loss = 0.0
        else:
            model = LogisticRegressionModel(
                n_features=len(FEATURES), learning_rate=config["learning_rate"]
            )
            model.set_parameters(parameters)
            loss = model.train(X_scaled, y_local, epochs=config["local_epochs"])
            updated_parameters = model.get_parameters()
            training_loss = float(loss)
            shared_loss = float(loss)
        return {
            "parameters": updated_parameters,
            "sample_count": len(y_local),
            "training_loss": training_loss,
            "shared_training_loss": shared_loss,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STATE_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )


def _demo_passwords() -> dict[str, str]:
    if "FL_DEMO_PASSWORD" in os.environ:
        raise RuntimeError(
            "FL_DEMO_PASSWORD is no longer supported because it is shared by all accounts."
        )
    per_user_variables = {
        details[2] for details in DEMO_USER_PASSWORDS.values()
    }
    missing = per_user_variables.difference(os.environ)
    if missing:
        raise RuntimeError(
            "Set a separate password for every demo account before initialization: "
            + ", ".join(sorted(missing))
        )
    passwords = {}
    for username, (_, _, environment_name) in DEMO_USER_PASSWORDS.items():
        password = os.environ[environment_name]
        if not isinstance(password, str) or len(password) < 12:
            raise ValueError(f"{environment_name} must contain at least 12 characters.")
        passwords[username] = password
    if len(set(passwords.values())) != len(passwords):
        raise ValueError("Each demo account must have a distinct password.")
    return passwords


def _principal_proof(
    user_id: int, username: str, role: str, hospital_id: str | None
) -> bytes:
    payload = json.dumps(
        [user_id, username, role, hospital_id],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.digest(_PRINCIPAL_SECRET, payload, "sha256")


def initialize() -> None:
    """Create persistent state and clearly marked local-demo accounts."""
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL CHECK(role IN ('admin', 'hospital')),
                hospital_id TEXT UNIQUE,
                salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prediction_history (
                prediction_id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(user_id),
                hospital_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                model_version TEXT NOT NULL,
                malignant_probability REAL NOT NULL,
                prediction TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rounds (
                round_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_by INTEGER NOT NULL REFERENCES users(user_id),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                config_json TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                candidate_json TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS client_updates (
                round_id TEXT NOT NULL REFERENCES rounds(round_id),
                hospital_id TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                training_loss REAL NOT NULL,
                parameters_json TEXT NOT NULL,
                PRIMARY KEY(round_id, hospital_id)
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL,
                user_id INTEGER REFERENCES users(user_id),
                action TEXT NOT NULL,
                details_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS app_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        existing = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if existing == 0:
            demo_passwords = _demo_passwords()
            for username, (role, hospital_id, _) in DEMO_USER_PASSWORDS.items():
                salt = secrets.token_bytes(16)
                connection.execute(
                    """
                    INSERT INTO users
                        (username, role, hospital_id, salt, password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        username,
                        role,
                        hospital_id,
                        salt,
                        _password_hash(demo_passwords[username], salt),
                        _now(),
                    ),
                )
            connection.execute(
                "INSERT OR REPLACE INTO app_metadata(key, value) VALUES ('account_password_version', '2')"
            )
        else:
            version = connection.execute(
                "SELECT value FROM app_metadata WHERE key = 'account_password_version'"
            ).fetchone()
            if version is None:
                demo_passwords = _demo_passwords()
                accounts = {
                    row["username"]: row
                    for row in connection.execute(
                        "SELECT user_id, username, role, hospital_id FROM users"
                    )
                }
                if set(accounts) != set(DEMO_USER_PASSWORDS):
                    raise RuntimeError(
                        "Cannot upgrade demo account passwords: expected the four "
                        "seeded demo accounts. Review the users table before continuing."
                    )
                for username, (role, hospital_id, _) in DEMO_USER_PASSWORDS.items():
                    row = accounts[username]
                    if row["role"] != role or row["hospital_id"] != hospital_id:
                        raise RuntimeError(
                            f"Cannot upgrade demo account {username}: its role or tenant assignment changed."
                        )
                    salt = secrets.token_bytes(16)
                    connection.execute(
                        "UPDATE users SET salt = ?, password_hash = ? WHERE user_id = ?",
                        (
                            salt,
                            _password_hash(demo_passwords[username], salt),
                            row["user_id"],
                        ),
                    )
                connection.execute(
                    "INSERT INTO app_metadata(key, value) VALUES ('account_password_version', '2')"
                )


def authenticate(username: str, password: str) -> Principal | None:
    initialize()
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
    if row is None or not isinstance(password, str):
        return None
    if not hmac.compare_digest(
        _password_hash(password, row["salt"]), row["password_hash"]
    ):
        return None
    return Principal(
        user_id=row["user_id"],
        username=row["username"],
        role=row["role"],
        hospital_id=row["hospital_id"],
        _proof=_principal_proof(
            row["user_id"], row["username"], row["role"], row["hospital_id"]
        ),
    )


def _assert_principal(principal: Principal, role: str) -> None:
    if not isinstance(principal, Principal):
        raise PermissionError("An authenticated account is required.")
    if (
        not isinstance(principal.user_id, int)
        or not isinstance(principal.username, str)
        or not isinstance(principal.role, str)
        or (
            principal.hospital_id is not None
            and not isinstance(principal.hospital_id, str)
        )
        or not isinstance(principal._proof, bytes)
    ):
        raise PermissionError("The authenticated account context is invalid.")
    expected_proof = _principal_proof(
        principal.user_id,
        principal.username,
        principal.role,
        principal.hospital_id,
    )
    if not hmac.compare_digest(principal._proof, expected_proof):
        raise PermissionError("The authenticated account context is invalid.")
    initialize()
    with _connect() as connection:
        row = connection.execute(
            "SELECT username, role, hospital_id FROM users WHERE user_id = ?",
            (principal.user_id,),
        ).fetchone()
    if (
        row is None
        or row["username"] != principal.username
        or row["role"] != principal.role
        or row["hospital_id"] != principal.hospital_id
        or principal.role != role
    ):
        raise PermissionError(f"This action requires an authenticated {role} account.")


def _audit(principal: Principal | None, action: str, details: dict) -> None:
    with _connect() as connection:
        connection.execute(
            "INSERT INTO audit_log(created_at, user_id, action, details_json) VALUES (?, ?, ?, ?)",
            (
                _now(),
                principal.user_id if principal else None,
                action,
                json.dumps(details, sort_keys=True),
            ),
        )


def hospital_status(principal: Principal) -> dict:
    _assert_principal(principal, "hospital")
    return _status_for(principal.hospital_id)


def _status_for(hospital_id: str) -> dict:
    _, y_local = _hospital_training_partition(hospital_id)
    with _connect() as connection:
        update = connection.execute(
            """
            SELECT COUNT(*) AS rounds, MAX(submitted_at) AS last_participation
            FROM client_updates WHERE hospital_id = ?
            """,
            (hospital_id,),
        ).fetchone()
        pending = connection.execute(
            """
            SELECT COUNT(*) FROM rounds r
            WHERE r.status = 'collecting'
              AND NOT EXISTS (
                SELECT 1 FROM client_updates u
                WHERE u.round_id = r.round_id AND u.hospital_id = ?
              )
            """,
            (hospital_id,),
        ).fetchone()[0]
    return {
        "hospital_id": hospital_id,
        "name": HOSPITALS[hospital_id],
        "training_records": len(y_local),
        "benign": int(np.sum(y_local == 0)),
        "malignant": int(np.sum(y_local == 1)),
        "completed_participations": int(update["rounds"]),
        "last_participation": update["last_participation"],
        "open_rounds": int(pending),
    }


def all_hospital_statuses(principal: Principal) -> list[dict]:
    _assert_principal(principal, "admin")
    return [_status_for(hospital_id) for hospital_id in HOSPITALS]


def _training_data():
    dataset_path = DATA_DIR / "wdbc.data"
    if not dataset_path.is_file():
        dataset_path = BASE_DIR / "data" / "wdbc.data"
    raw = pd.read_csv(dataset_path, header=None)
    features = raw.iloc[:, 2:].copy()
    features.columns = FEATURES
    labels = raw.iloc[:, 1].map({"B": 0, "M": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    return X_train, X_test, y_test, y_train


def _hospital_training_partition(hospital_id: str):
    X_train, _, _, y_train = _training_data()
    indices = partition_hospital_indices(y_train, seed=42)
    partition_index = list(HOSPITALS).index(hospital_id)
    rows = indices[partition_index]
    return (
        X_train.loc[rows].to_numpy(dtype=float),
        y_train.loc[rows].to_numpy(dtype=int),
    )


def start_federated_round(
    principal: Principal,
    *,
    mode: str = "FedAvg",
    local_epochs: int = 5,
    learning_rate: float = 0.05,
    clipping_norm: float = 1.0,
    noise_multiplier: float = 2.0,
    delta: float = 1e-5,
) -> str:
    _assert_principal(principal, "admin")
    if mode not in {"FedAvg", "DP-FedAvg (experimental)"}:
        raise ValueError("Unsupported federated training mode.")
    if not isinstance(local_epochs, int) or not 1 <= local_epochs <= 50:
        raise ValueError("Local epochs must be between 1 and 50.")
    if mode.startswith("DP"):
        if not np.isfinite(clipping_norm) or clipping_norm <= 0:
            raise ValueError("Clipping norm must be finite and positive.")
        if not np.isfinite(noise_multiplier) or noise_multiplier <= 0:
            raise ValueError("Noise multiplier must be finite and positive.")
        if not np.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("Delta must be between 0 and 1.")

    X_train, _, _, _ = _training_data()
    scaler = StandardScaler().fit(X_train)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    round_id = uuid.uuid4().hex[:12]
    scaler_path = CANDIDATE_DIR / f"round_{round_id}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    config = {
        "local_epochs": local_epochs,
        "learning_rate": learning_rate,
        "scaler": str(scaler_path.relative_to(MODEL_DIR)),
        "clipping_norm": clipping_norm,
        "noise_multiplier": noise_multiplier,
        "delta": delta,
    }
    initial = {
        "weights": np.zeros(len(FEATURES), dtype=float).tolist(),
        "bias": 0.0,
    }
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO rounds
                (round_id, status, mode, created_by, created_at, updated_at,
                 config_json, parameters_json)
            VALUES (?, 'collecting', ?, ?, ?, ?, ?, ?)
            """,
            (
                round_id,
                mode,
                principal.user_id,
                _now(),
                _now(),
                json.dumps(config),
                json.dumps(initial),
            ),
        )
    _audit(principal, "federated_round_started", {"round_id": round_id, "mode": mode})
    return round_id


def cancel_federated_round(principal: Principal, round_id: str) -> dict:
    _assert_principal(principal, "admin")
    with _connect() as connection:
        row = connection.execute(
            "SELECT status FROM rounds WHERE round_id = ?", (round_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Unknown federated round.")
        if row["status"] != "collecting":
            raise ValueError("Only a collecting federated round can be cancelled.")
        updated_at = _now()
        result = connection.execute(
            """
            UPDATE rounds SET status = 'cancelled', updated_at = ?
            WHERE round_id = ? AND status = 'collecting'
            """,
            (updated_at, round_id),
        )
        if result.rowcount != 1:
            raise ValueError("This federated round is no longer collecting.")
    _audit(principal, "federated_round_cancelled", {"round_id": round_id})
    return {"round_id": round_id, "status": "cancelled"}


def _dp_privacy_accounting_report(delta: float) -> dict:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT r.round_id, r.status, r.config_json,
                   COUNT(u.hospital_id) AS participants
            FROM rounds r
            LEFT JOIN client_updates u ON u.round_id = r.round_id
            WHERE r.mode = 'DP-FedAvg (experimental)'
            GROUP BY r.round_id
            ORDER BY r.created_at
            """,
        ).fetchall()

    accountant = DPAccountant(
        noise_multiplier=1.0,
        delta=delta,
        sensitivity=1.0,
    )
    accounted_client_updates = 0
    rounds_with_updates = []
    for row in rows:
        config = json.loads(row["config_json"])
        participants = int(row["participants"])
        if participants:
            mechanisms = participants * config["local_epochs"]
            accountant.record_steps(
                mechanisms,
                noise_multiplier=config["noise_multiplier"],
            )
            accounted_client_updates += participants
            rounds_with_updates.append({
                "round_id": row["round_id"],
                "status": row["status"],
                "participants": participants,
                "local_epochs": config["local_epochs"],
                "noise_multiplier": config["noise_multiplier"],
                "clipping_norm": config["clipping_norm"],
                "client_epoch_mechanisms": mechanisms,
            })
    return {
        "epsilon_upper_bound": accountant.get_epsilon(),
        "delta": delta,
        "accounted_client_epoch_mechanisms": accountant.steps,
        "accounted_client_updates": accounted_client_updates,
        "accounted_rounds_with_updates": len(rounds_with_updates),
        "accounted_rounds": rounds_with_updates,
        "composed_noise_multiplier_steps": [
            {
                "noise_multiplier": float(noise_multiplier),
                "client_epoch_mechanisms": int(mechanisms),
            }
            for noise_multiplier, mechanisms in sorted(
                accountant.get_privacy_report()["noise_multiplier_counts"].items()
            )
        ],
        "accountant_method": "Gaussian mechanism with integer-order RDP composition",
        "adjacency": "Fixed-size replace-one record within a fixed hospital partition",
        "sampling": "No client or record subsampling; no amplification claimed",
        "accounting_scope": (
            "Conditional bound for the noisy client-update mechanism only. It "
            "sequentially composes every submitted client update and local epoch."
        ),
        "limitations": (
            "Does not account for pooled preprocessing, disclosed counts and "
            "participation, unnoised candidate metrics, shared-process coordinator "
            "access, or other application outputs."
        ),
    }


def dp_privacy_accounting(principal: Principal) -> dict | None:
    """Return cumulative conditional accounting for all submitted DP client updates."""
    _assert_principal(principal, "admin")
    with _connect() as connection:
        latest_round = connection.execute(
            """
            SELECT config_json FROM rounds
            WHERE mode = 'DP-FedAvg (experimental)'
            ORDER BY created_at DESC LIMIT 1
            """,
        ).fetchone()
    if latest_round is None:
        return None
    delta = json.loads(latest_round["config_json"])["delta"]
    return _dp_privacy_accounting_report(delta)


def _clip_per_record_gradients(per_record_gradients, clipping_norm):
    if not np.isfinite(clipping_norm) or clipping_norm <= 0:
        raise ValueError("Gradient clipping norm must be finite and positive.")
    norms = np.linalg.norm(per_record_gradients, axis=1, keepdims=True)
    scales = np.minimum(
        1.0,
        clipping_norm / np.maximum(norms, 1e-12),
    )
    return per_record_gradients * scales


def _train_dp_local(X, y, parameters, *, epochs, learning_rate, clip, sigma):
    model = LogisticRegressionModel(n_features=X.shape[1], learning_rate=learning_rate)
    model.set_parameters(parameters)
    count = len(y)
    if count == 0:
        raise ValueError("DP local training requires at least one record.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("DP noise multiplier must be finite and positive.")
    rng = secrets.SystemRandom()
    for _ in range(epochs):
        probabilities = model.predict_proba(X)
        residuals = probabilities - y
        per_record = np.column_stack((X * residuals[:, None], residuals))
        clipped_mean = _clip_per_record_gradients(per_record, clip).mean(axis=0)
        local_noise_std = sigma * (2.0 * clip / count)
        gaussian_noise = np.fromiter(
            (
                rng.gauss(0.0, local_noise_std)
                for _ in range(len(clipped_mean))
            ),
            dtype=float,
            count=len(clipped_mean),
        )
        noisy_mean = clipped_mean + gaussian_noise
        model.weights -= learning_rate * noisy_mean[:-1]
        model.bias -= learning_rate * noisy_mean[-1]
    return model.get_parameters(), None


def submit_hospital_update(principal: Principal, round_id: str) -> dict:
    _assert_principal(principal, "hospital")
    if principal.hospital_id not in HOSPITALS:
        raise PermissionError("This account has no assigned training partition.")
    with _connect() as connection:
        round_row = connection.execute(
            "SELECT * FROM rounds WHERE round_id = ?", (round_id,)
        ).fetchone()
        if round_row is None or round_row["status"] != "collecting":
            raise ValueError("This federated round is not accepting updates.")
        submitted = connection.execute(
            "SELECT 1 FROM client_updates WHERE round_id = ? AND hospital_id = ?",
            (round_id, principal.hospital_id),
        ).fetchone()
        if submitted:
            raise ValueError("This hospital has already participated in this round.")
        config = json.loads(round_row["config_json"])
        global_parameters = json.loads(round_row["parameters_json"])

    scaler = joblib.load(MODEL_DIR / config["scaler"])
    local_result = SimulatedHospitalClient(principal.hospital_id).train_update(
        scaler, global_parameters, config, round_row["mode"]
    )

    # Only model parameters, a sample count, and aggregate loss leave the simulated client.
    with _connect() as connection:
        current_status = connection.execute(
            "SELECT status FROM rounds WHERE round_id = ?", (round_id,)
        ).fetchone()
        if current_status is None or current_status["status"] != "collecting":
            raise ValueError("This federated round is no longer accepting updates.")
        connection.execute(
            """
            INSERT INTO client_updates
                (round_id, hospital_id, submitted_at, sample_count,
                 training_loss, parameters_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                round_id,
                principal.hospital_id,
                _now(),
                local_result["sample_count"],
                local_result["shared_training_loss"],
                json.dumps(
                    {
                        "weights": local_result["parameters"]["weights"].tolist(),
                        "bias": local_result["parameters"]["bias"],
                    }
                ),
            ),
        )
        connection.execute(
            "UPDATE rounds SET updated_at = ? WHERE round_id = ?",
            (_now(), round_id),
        )
    _audit(
        principal,
        "hospital_update_submitted",
        {"round_id": round_id, "hospital_id": principal.hospital_id},
    )
    _complete_round_if_ready(round_id)
    return {
        "round_id": round_id,
        "sample_count": local_result["sample_count"],
        "training_loss": local_result["training_loss"],
    }


def _complete_round_if_ready(round_id: str) -> None:
    with _connect() as connection:
        round_row = connection.execute(
            "SELECT * FROM rounds WHERE round_id = ?", (round_id,)
        ).fetchone()
        updates = connection.execute(
            "SELECT * FROM client_updates WHERE round_id = ?", (round_id,)
        ).fetchall()
    if (
        round_row is None
        or round_row["status"] != "collecting"
        or len(updates) != len(HOSPITALS)
    ):
        return
    update_inputs = [
        {
            "num_patients": row["sample_count"],
            "parameters": json.loads(row["parameters_json"]),
        }
        for row in updates
    ]
    global_parameters = federated_averaging(update_inputs)
    X_train, X_test, y_test, _ = _training_data()
    config = json.loads(round_row["config_json"])
    scaler = joblib.load(MODEL_DIR / config["scaler"])
    model = LogisticRegressionModel(n_features=len(FEATURES))
    model.set_parameters(global_parameters)
    probabilities = model.predict_proba(scaler.transform(X_test))
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "malignant_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "holdout_records": int(len(y_test)),
        "training_records": int(len(X_train)),
    }
    candidate_id = f"fl-{round_id}"
    model_filename = f"{candidate_id}.npz"
    scaler_filename = f"{candidate_id}_scaler.pkl"
    model_path = CANDIDATE_DIR / model_filename
    scaler_path = CANDIDATE_DIR / scaler_filename
    np.savez(model_path, weights=global_parameters["weights"], bias=global_parameters["bias"])
    joblib.dump(scaler, scaler_path)
    candidate = {
        "candidate_id": candidate_id,
        "round_id": round_id,
        "name": f"Federated Logistic Regression {candidate_id}",
        "kind": "linear",
        "model": str(model_path.relative_to(MODEL_DIR)),
        "scaler": str(scaler_path.relative_to(MODEL_DIR)),
        "metrics": metrics,
        "mode": round_row["mode"],
        "status": "pending_approval",
        "created_at": _now(),
    }
    if round_row["mode"].startswith("DP"):
        candidate["privacy_accounting"] = _dp_privacy_accounting_report(
            config["delta"]
        )
        candidate["privacy_accounting"]["current_round_parameters"] = {
            "clipping_norm": config["clipping_norm"],
            "noise_multiplier": config["noise_multiplier"],
            "local_epochs": config["local_epochs"],
        }
    with _connect() as connection:
        result = connection.execute(
            """
            UPDATE rounds
            SET status = 'candidate_pending', parameters_json = ?,
                candidate_json = ?, updated_at = ?
            WHERE round_id = ? AND status = 'collecting'
            """,
            (
                json.dumps(
                    {
                        "weights": global_parameters["weights"].tolist(),
                        "bias": global_parameters["bias"],
                    }
                ),
                json.dumps(candidate),
                _now(),
                round_id,
            ),
        )
        if result.rowcount != 1:
            model_path.unlink(missing_ok=True)
            scaler_path.unlink(missing_ok=True)
            return
    _audit(None, "federated_candidate_created", {"candidate_id": candidate_id, "metrics": metrics})


def list_rounds(principal: Principal) -> list[dict]:
    if principal.role == "admin":
        _assert_principal(principal, "admin")
        scope_id = None
    else:
        _assert_principal(principal, "hospital")
        scope_id = principal.hospital_id
    with _connect() as connection:
        rows = connection.execute("SELECT * FROM rounds ORDER BY created_at DESC").fetchall()
        result = []
        for row in rows:
            updates = connection.execute(
                """
                SELECT hospital_id, submitted_at, sample_count, training_loss
                FROM client_updates WHERE round_id = ?
                """,
                (row["round_id"],),
            ).fetchall()
            candidate = json.loads(row["candidate_json"]) if row["candidate_json"] else None
            status_to_aggregation = {
                "collecting": "waiting_for_client_updates",
                "candidate_pending": "aggregated_pending_admin_approval",
                "approved": "approved_and_deployed",
                "rejected": "candidate_rejected",
                "cancelled": "cancelled_without_candidate",
            }
            total_samples = sum(item["sample_count"] for item in updates)
            if scope_id and not any(item["hospital_id"] == scope_id for item in updates):
                result.append({
                    "round_id": row["round_id"],
                    "status": row["status"],
                    "mode": row["mode"],
                    "created_at": row["created_at"],
                    "participated": False,
                    "participants": len(updates),
                    "expected_participants": len(HOSPITALS),
                    "aggregation_method": "sample-count-weighted FedAvg",
                    "aggregation_status": status_to_aggregation.get(
                        row["status"], "unknown"
                    ),
                    "global_model_version": (
                        candidate.get("version") if candidate else None
                    ),
                    "candidate": candidate,
                })
                continue
            visible_updates = (
                updates
                if scope_id is None
                else [item for item in updates if item["hospital_id"] == scope_id]
            )
            result.append({
                "round_id": row["round_id"],
                "status": row["status"],
                "mode": row["mode"],
                "created_at": row["created_at"],
                "participants": len(updates),
                "expected_participants": len(HOSPITALS),
                "participated": any(item["hospital_id"] == scope_id for item in updates) if scope_id else None,
                "aggregation_method": "sample-count-weighted FedAvg",
                "aggregation_status": status_to_aggregation.get(
                    row["status"], "unknown"
                ),
                "global_model_version": (
                    candidate.get("version") if candidate else None
                ),
                "participant_results": [
                    {
                        "hospital": HOSPITALS[item["hospital_id"]],
                        "submitted_at": item["submitted_at"],
                        "sample_count": item["sample_count"],
                        "training_loss": (
                            None if row["mode"].startswith("DP")
                            else item["training_loss"]
                        ),
                        "aggregation_weight": (
                            item["sample_count"] / total_samples
                            if scope_id is None and total_samples
                            else None
                        ),
                    }
                    for item in visible_updates
                ],
                "candidate": candidate,
            })
    return result


def review_candidate(principal: Principal, candidate_id: str, *, approve: bool) -> dict:
    _assert_principal(principal, "admin")
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM rounds WHERE candidate_json IS NOT NULL"
        ).fetchall()
        candidate_row = next(
            (item for item in row if json.loads(item["candidate_json"])["candidate_id"] == candidate_id),
            None,
        )
        if candidate_row is None or candidate_row["status"] != "candidate_pending":
            raise ValueError("No pending candidate matches that ID.")
        candidate = json.loads(candidate_row["candidate_json"])
        if approve:
            model_path = MODEL_DIR / candidate["model"]
            scaler_path = MODEL_DIR / candidate["scaler"]
            if not model_path.is_file() or not scaler_path.is_file():
                raise FileNotFoundError("Candidate model artifacts are missing.")
            with np.load(model_path) as artifact:
                if artifact["weights"].shape != (len(FEATURES),):
                    raise ValueError("Candidate model has an invalid feature shape.")
            version_row = connection.execute(
                "SELECT COUNT(*) FROM audit_log WHERE action = 'model_approved'"
            ).fetchone()
            version = int(version_row[0]) + 1
            active = {
                "name": candidate["name"],
                "kind": "linear",
                "model": candidate["model"],
                "scaler": candidate["scaler"],
                "candidate_id": candidate_id,
                "version": f"v{version:04d}",
                "mode": candidate["mode"],
                "approved_at": _now(),
            }
            temporary = ACTIVE_MODEL_PATH.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(active, indent=2), encoding="utf-8")
            os.replace(temporary, ACTIVE_MODEL_PATH)
            candidate["status"] = "approved"
            candidate["version"] = active["version"]
            connection.execute(
                "UPDATE rounds SET status = 'approved', candidate_json = ?, updated_at = ? WHERE round_id = ?",
                (json.dumps(candidate), _now(), candidate_row["round_id"]),
            )
        else:
            candidate["status"] = "rejected"
            connection.execute(
                "UPDATE rounds SET status = 'rejected', candidate_json = ?, updated_at = ? WHERE round_id = ?",
                (json.dumps(candidate), _now(), candidate_row["round_id"]),
            )
            active = candidate
    _audit(
        principal,
        "model_approved" if approve else "candidate_rejected",
        {"candidate_id": candidate_id, "version": active.get("version")},
    )
    return active


def make_hospital_predictions(principal: Principal, features: pd.DataFrame) -> pd.DataFrame:
    _assert_principal(principal, "hospital")
    if not ACTIVE_MODEL_PATH.is_file():
        raise FileNotFoundError("No model has been approved for inference.")
    clean_features = validate_feature_frame(features)
    active = json.loads(ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))
    probabilities = predict_probabilities(clean_features, active_path=ACTIVE_MODEL_PATH)
    results = pd.DataFrame(
        {
            "Record": np.arange(1, len(probabilities) + 1),
            "Benign Probability": 1.0 - probabilities,
            "Malignant Probability": probabilities,
            "Prediction": np.where(probabilities >= 0.5, "Malignant", "Benign"),
            "Model version": active.get("version", active.get("name", "approved model")),
        }
    )
    with _connect() as connection:
        connection.executemany(
            """
            INSERT INTO prediction_history
                (user_id, hospital_id, created_at, model_version,
                 malignant_probability, prediction)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    principal.user_id,
                    principal.hospital_id,
                    _now(),
                    active.get("version", active.get("name", "approved model")),
                    float(probability),
                    str(prediction),
                )
                for probability, prediction in zip(
                    probabilities, results["Prediction"]
                )
            ],
        )
    _audit(principal, "predictions_generated", {"count": len(results)})
    return results


def prediction_history(principal: Principal) -> pd.DataFrame:
    _assert_principal(principal, "hospital")
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT created_at, model_version, malignant_probability, prediction
            FROM prediction_history
            WHERE user_id = ? AND hospital_id = ?
            ORDER BY prediction_id DESC LIMIT 500
            """,
            (principal.user_id, principal.hospital_id),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def audit_log(principal: Principal) -> pd.DataFrame:
    _assert_principal(principal, "admin")
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT a.created_at, u.username, a.action, a.details_json
            FROM audit_log a LEFT JOIN users u ON a.user_id = u.user_id
            ORDER BY a.audit_id DESC LIMIT 500
            """
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def current_model(principal: Principal) -> dict | None:
    _assert_principal(principal, principal.role)
    if not ACTIVE_MODEL_PATH.is_file():
        return None
    return json.loads(ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))


def privacy_description(principal: Principal) -> dict:
    _assert_principal(principal, principal.role)
    return {
        "dp_mode": "DP-FedAvg (experimental)",
        "noise_location": (
            "Each client clips concatenated per-record feature/bias gradients to L2 "
            "norm C, adds Gaussian noise to every full-batch mean with standard "
            "deviation noise_multiplier * 2C / n, and submits only its final model "
            "parameters after the configured local epochs."
        ),
        "noise_randomness": (
            "Python secrets.SystemRandom, which draws from the operating system "
            "random source. The floating-point Gaussian sampler has not been "
            "independently audited as an exact DP sampler."
        ),
        "server_observes": (
            "Each noisy final client model update, exact local sample counts, "
            "participation identity/timing, and unnoised candidate metrics on the "
            "public WDBC holdout. DP rounds do not return or store per-client "
            "training loss. The coordinator process can still inspect shared client "
            "memory and local dataset files."
        ),
        "limitations": (
            "All clients, data partitions, and coordinator run in one Python process "
            "and share local files; this is not an isolation boundary. Any reported "
            "epsilon/delta is a conditional Gaussian/RDP bound for the noisy client "
            "updates under fixed-size replace-one adjacency, fixed partitions, exact "
            "counts, no subsampling, and an ideal Gaussian mechanism. The accountant "
            "sequentially composes submitted client updates and local epochs, but does "
            "not cover pooled training-feature scaling, exact counts or participation, "
            "unnoised candidate metrics, coordinator access to data/memory, or other "
            "application outputs. The Gaussian sampler and complete release surface "
            "have not been independently validated; no end-to-end DP guarantee is "
            "established."
        ),
    }
