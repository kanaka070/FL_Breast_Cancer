import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dashboard_charts import (
    fedavg_history_chart_data,
    fedavg_training_history_chart,
    format_comparison_for_display,
    horizontal_grouped_count_chart,
    horizontal_metric_chart,
)
from academic_experiments import run_cross_validation, run_federated_cross_validation
from dp_accountant import DPAccountant
from live_evaluation import (
    run_iid_noniid_validation,
    run_live_model_comparison,
)
import app_services
from feature_schema import FEATURES
from inference import MODEL_ARTIFACTS, predict_probabilities, validate_feature_frame
from retraining import (
    approve_baseline_model,
    approve_candidate,
    create_candidate,
    register_confirmed_records,
)
from workflow_utils import (
    federated_averaging,
    partition_hospital_indices,
    trust_weighted_averaging,
)


BASE_DIR = Path(__file__).resolve().parents[1]


class AuthenticatedTestCase(unittest.TestCase):
    def setUp(self):
        self.auth_dir = tempfile.TemporaryDirectory()
        self.auth_patches = [
            patch.dict(os.environ, {
                "FL_ADMIN_PASSWORD": "test-admin-password",
                "FL_HOSPITAL1_PASSWORD": "test-hospital1-password",
                "FL_HOSPITAL2_PASSWORD": "test-hospital2-password",
                "FL_HOSPITAL3_PASSWORD": "test-hospital3-password",
            }),
            patch.object(
                app_services,
                "STATE_PATH",
                Path(self.auth_dir.name) / "application.sqlite3",
            ),
        ]
        for patcher in self.auth_patches:
            patcher.start()
        app_services.initialize()
        self.admin = app_services.authenticate("admin", "test-admin-password")

    def tearDown(self):
        for patcher in reversed(self.auth_patches):
            patcher.stop()
        self.auth_dir.cleanup()


class PartitioningTests(unittest.TestCase):
    def test_partitions_cover_training_records_without_overlap_or_test_leakage(self):
        labels = pd.Series(
            np.concatenate([np.zeros(357, dtype=int), np.ones(212, dtype=int)]),
            index=np.arange(1000, 1569),
        )
        train_labels, test_labels = train_test_split(
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
        partitions = partition_hospital_indices(train_labels, seed=42)
        combined = np.concatenate(partitions)
        self.assertEqual(len(combined), len(train_labels))
        self.assertEqual(len(np.unique(combined)), len(train_labels))
        self.assertEqual(set(combined), set(train_labels.index))
        self.assertTrue(set(combined).isdisjoint(set(test_labels.index)))


class FederatedAveragingTests(unittest.TestCase):
    def test_weights_and_bias_are_patient_weighted(self):
        result = federated_averaging([
            {"num_patients": 1, "parameters": {"weights": [1.0, 3.0], "bias": 2.0}},
            {"num_patients": 3, "parameters": {"weights": [5.0, 7.0], "bias": 6.0}},
        ])
        np.testing.assert_allclose(result["weights"], [4.0, 6.0])
        self.assertAlmostEqual(result["bias"], 5.0)

    def test_rejects_empty_updates(self):
        with self.assertRaises(ValueError):
            federated_averaging([])

    def test_trust_weighted_aggregation_uses_sample_count_times_trust(self):
        result = trust_weighted_averaging([
            {
                "num_patients": 1,
                "trust_score": 1.0,
                "parameters": {"weights": [0.0], "bias": 0.0},
            },
            {
                "num_patients": 3,
                "trust_score": 0.5,
                "parameters": {"weights": [4.0], "bias": 4.0},
            },
        ])
        np.testing.assert_allclose(result["weights"], [2.4])
        self.assertAlmostEqual(result["bias"], 2.4)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            trust_weighted_averaging([{
                "num_patients": 1,
                "trust_score": np.nan,
                "parameters": {"weights": [0.0], "bias": 0.0},
            }])


class ResearchExperimentTests(unittest.TestCase):
    def test_cross_validation_runs_expected_folds_and_does_not_evaluate_final_holdout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folds, summary = run_cross_validation(Path(temp_dir))
            self.assertEqual(len(folds), 25)
            self.assertEqual(
                summary.set_index("protocol")["folds"].to_dict(),
                {
                    "stratified_5_fold": 5,
                    "stratified_10_fold": 10,
                    "unstratified_10_fold": 10,
                },
            )
            self.assertFalse(summary["held_out_test_evaluated"].any())
            self.assertTrue((folds["training_time_seconds"] >= 0).all())
            self.assertTrue(np.isfinite(folds["accuracy"]).all())
            self.assertTrue((folds["true_negatives"] + folds["false_positives"] > 0).all())

    def test_federated_cross_validation_covers_both_logistic_approaches_without_test_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folds, summary = run_federated_cross_validation(Path(temp_dir))
            self.assertEqual(len(folds), 50)
            self.assertEqual(set(folds["approach"]), {
                "FedAvg",
                "DP-FedAvg (experimental)",
            })
            self.assertEqual(
                summary.groupby("approach")["folds"].sum().to_dict(),
                {
                    "FedAvg": 25,
                    "DP-FedAvg (experimental)": 25,
                },
            )
            self.assertFalse(folds.empty)
            self.assertFalse(summary["held_out_test_evaluated"].any())
            self.assertTrue(np.isfinite(folds["accuracy"]).all())
            self.assertTrue(np.isfinite(folds["roc_auc"]).all())


class InferenceTests(AuthenticatedTestCase):
    def test_saved_models_predict_and_reject_incorrect_feature_order(self):
        rng = np.random.default_rng(913)
        frame = pd.DataFrame(
            rng.uniform(0.1, 1.0, size=(3, len(FEATURES))),
            columns=FEATURES,
        )
        shuffled = frame[list(reversed(FEATURES))]
        for model_name, artifact in MODEL_ARTIFACTS.items():
            model_path = BASE_DIR / "saved_models" / artifact["model"]
            if not model_path.is_file():
                self.skipTest(f"Missing existing artifact: {model_path.name}")
            first = predict_probabilities(frame, model_name)
            self.assertTrue(np.isfinite(first).all())
            self.assertTrue(np.all((first >= 0) & (first <= 1)))
            with self.assertRaisesRegex(ValueError, "canonical WDBC order"):
                predict_probabilities(shuffled, model_name)

    def test_explicit_baseline_approval_persists_model_and_corresponding_scaler(self):
        rng = np.random.default_rng(916)
        features = pd.DataFrame(
            rng.uniform(0.1, 1.0, size=(3, len(FEATURES))),
            columns=FEATURES,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            active_path = Path(temp_dir) / "active_model.json"
            with patch("retraining.ACTIVE_MODEL_PATH", active_path):
                with self.assertRaises(PermissionError):
                    approve_baseline_model(
                        "FedAvg Logistic Regression",
                        admin_approved=False,
                    )
                self.assertFalse(active_path.exists())

                for model_name, artifact in MODEL_ARTIFACTS.items():
                    active = approve_baseline_model(
                        model_name,
                        admin_approved=True,
                        principal=self.admin,
                    )
                    persisted = json.loads(active_path.read_text(encoding="utf-8"))
                    self.assertEqual(persisted, active)
                    self.assertEqual(active["model"], artifact["model"])
                    self.assertEqual(active.get("scaler"), artifact.get("scaler"))
                    self.assertRegex(active["version"], r"^v\d{4}$")
                    with patch("inference.MODEL_DIR", BASE_DIR / "saved_models"):
                        approved_probabilities = predict_probabilities(
                            features,
                            active_path=active_path,
                        )
                        direct_probabilities = predict_probabilities(features, model_name)
                    np.testing.assert_allclose(approved_probabilities, direct_probabilities)

    def test_missing_and_nonfinite_features_are_rejected(self):
        valid = pd.DataFrame([[1.0] * len(FEATURES)], columns=FEATURES)
        with self.assertRaisesRegex(ValueError, "Missing required"):
            validate_feature_frame(valid.drop(columns=[FEATURES[0]]))
        valid.loc[0, FEATURES[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_feature_frame(valid)

    def test_nonnumeric_and_infinite_features_are_rejected(self):
        valid = pd.DataFrame([[1.0] * len(FEATURES)], columns=FEATURES)
        nonnumeric = valid.astype(object)
        nonnumeric.loc[0, FEATURES[0]] = "not-numeric"
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            validate_feature_frame(nonnumeric)

        for invalid_value in (np.inf, -np.inf):
            nonfinite = valid.copy()
            nonfinite.loc[0, FEATURES[0]] = invalid_value
            with self.assertRaisesRegex(ValueError, "finite"):
                validate_feature_frame(nonfinite)


class DashboardChartTests(unittest.TestCase):
    def test_all_dashboard_charts_share_the_research_canvas_theme(self):
        comparison = pd.read_csv(
            BASE_DIR / "metrics" / "final_three_model_comparison.csv"
        )
        distribution = pd.read_csv(
            BASE_DIR / "metrics" / "hospital_distribution.csv"
        ).rename(columns={
            "hospital": "Hospital",
            "benign_patients": "Benign",
            "malignant_patients": "Malignant",
        })
        charts = [
            horizontal_metric_chart(
                comparison,
                category="model",
                metric="accuracy",
                title="Accuracy",
                percentage=True,
            ),
            horizontal_grouped_count_chart(
                distribution,
                category="Hospital",
                metrics=["Benign", "Malignant"],
                title="Hospital partitions",
            ),
            fedavg_training_history_chart(pd.DataFrame({
                "round": [1, 2],
                "training_loss": [0.5, 0.3],
            })),
        ]
        for chart in charts:
            with self.subTest(chart=chart):
                config = chart.to_dict()["config"]
                self.assertEqual(config["background"], "#f4f7fb")
                self.assertEqual(config["view"]["stroke"], "transparent")

    def test_fedavg_history_chart_skips_invalid_values_without_filling_them(self):
        history = pd.DataFrame({
            "round": ["1", "2", "bad", "4", "5"],
            "training_loss": ["0.5", None, "0.3", "inf", "0.1"],
            "validation_loss": [np.nan, "0.4", "NaN", "-inf", "not-a-number"],
        })

        chart_data = fedavg_history_chart_data(history)
        self.assertEqual(
            sorted(chart_data["round"].tolist()),
            [1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 5.0, 5.0],
        )
        self.assertEqual(
            sorted(chart_data["value"].dropna().tolist()),
            [0.1, 0.4, 0.5],
        )
        self.assertTrue(
            np.isfinite(chart_data["value"].dropna().to_numpy(dtype=float)).all()
        )
        chart = fedavg_training_history_chart(history).to_dict()
        self.assertEqual(chart["mark"]["invalid"], "break-paths-filter-domains")
        self.assertIsNotNone(chart)

    def test_fedavg_history_chart_is_omitted_when_no_finite_loss_exists(self):
        history = pd.DataFrame({
            "round": [1, 2],
            "training_loss": [np.nan, "inf"],
        })
        self.assertTrue(fedavg_history_chart_data(history).empty)
        self.assertIsNone(fedavg_training_history_chart(history))

    def test_model_comparison_keeps_expected_models_and_artifacts(self):
        comparison = pd.read_csv(
            BASE_DIR / "metrics" / "final_three_model_comparison.csv"
        )
        expected_models = {
            "Centralized Logistic Regression",
            "FedAvg Logistic Regression",
            "DP Federated Gradient Model",
        }
        self.assertEqual(set(comparison["model"]), expected_models)
        np.testing.assert_allclose(
            comparison.set_index("model").loc[
                "Centralized Logistic Regression", "accuracy"
            ],
            0.9649122807017544,
        )
        np.testing.assert_allclose(
            comparison.set_index("model").loc[
                "FedAvg Logistic Regression", "accuracy"
            ],
            0.9736842105263158,
        )
        np.testing.assert_allclose(
            comparison.set_index("model").loc[
                "DP Federated Gradient Model", "accuracy"
            ],
            0.9473684210526315,
        )
        for artifact in MODEL_ARTIFACTS.values():
            model_path = BASE_DIR / "saved_models" / artifact["model"]
            self.assertTrue(model_path.is_file(), model_path.name)
            if artifact.get("scaler"):
                self.assertTrue(
                    (BASE_DIR / "saved_models" / artifact["scaler"]).is_file(),
                    artifact["scaler"],
                )

    def test_percentage_chart_is_horizontal_and_bounded_to_100(self):
        comparison = pd.read_csv(
            BASE_DIR / "metrics" / "final_three_model_comparison.csv"
        )
        chart = horizontal_metric_chart(
            comparison,
            category="model",
            metric="accuracy",
            title="Accuracy",
            percentage=True,
        ).to_dict()
        x_encoding = chart["encoding"]["x"]
        self.assertEqual(x_encoding["scale"]["domain"], [0, 100])
        self.assertEqual(chart["encoding"]["y"]["field"], "category")
        self.assertEqual(chart["encoding"]["y"]["type"], "nominal")
        self.assertIn("Accuracy (%)", x_encoding["axis"]["title"])

    def test_comparison_table_shows_explicit_percentage_and_count_units(self):
        saved_path = BASE_DIR / "metrics" / "final_three_model_comparison.csv"
        comparison = pd.read_csv(saved_path)
        original_accuracy = comparison["accuracy"].copy()
        display = format_comparison_for_display(comparison)
        self.assertEqual(display["Accuracy (%)"].max(), 97.37)
        self.assertIn("Malignant Recall (%)", display.columns)
        self.assertIn("False Negatives (patients)", display.columns)
        pd.testing.assert_series_equal(comparison["accuracy"], original_accuracy)
        self.assertEqual(pd.read_csv(saved_path)["accuracy"].tolist(), original_accuracy.tolist())

    def test_count_chart_starts_at_zero_and_rejects_negative_counts(self):
        comparison = pd.read_csv(
            BASE_DIR / "metrics" / "final_three_model_comparison.csv"
        )
        chart = horizontal_metric_chart(
            comparison,
            category="model",
            metric="false_negatives",
            title="False Negatives (patients)",
        ).to_dict()
        self.assertEqual(chart["encoding"]["x"]["scale"]["domain"][0], 0)
        invalid = comparison.assign(false_negatives=[3, -1, 4])
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            horizontal_metric_chart(
                invalid,
                category="model",
                metric="false_negatives",
                title="False Negatives (patients)",
            )

    def test_hospital_count_chart_uses_horizontal_zero_based_axis(self):
        distribution = pd.read_csv(
            BASE_DIR / "metrics" / "hospital_distribution.csv"
        ).rename(columns={
            "hospital": "Hospital",
            "benign_patients": "Benign",
            "malignant_patients": "Malignant",
        })
        chart = horizontal_grouped_count_chart(
            distribution,
            category="Hospital",
            metrics=["Benign", "Malignant"],
            title="Patients by Simulated Hospital",
        ).to_dict()
        self.assertEqual(chart["encoding"]["x"]["scale"]["domain"][0], 0)
        self.assertEqual(chart["encoding"]["y"]["field"], "category")


class RetrainingRulesTests(AuthenticatedTestCase):
    def _labeled_rows(self, count=8):
        rng = np.random.default_rng(914)
        features = rng.normal(size=(count, len(FEATURES)))
        frame = pd.DataFrame(
            features,
            columns=FEATURES,
        )
        frame["diagnosis"] = (features[:, 0] > 0).astype(int)
        return frame

    def test_retraining_service_requires_authenticated_admin_not_confirmation_flags(self):
        valid_records = self._labeled_rows()
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("retraining.RECORDS_PATH", Path(temp_dir) / "confirmed.csv"):
                with self.assertRaises(PermissionError):
                    register_confirmed_records(
                        valid_records, admin_confirmed=True
                    )
                with self.assertRaises(PermissionError):
                    register_confirmed_records(
                        valid_records,
                        admin_confirmed=True,
                        principal=app_services.authenticate(
                            "hospital1", "test-hospital1-password"
                        ),
                    )
        with self.assertRaises(PermissionError):
            approve_baseline_model(
                "FedAvg Logistic Regression", admin_approved=True
            )

    def test_predictions_are_not_accepted_as_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "confirmed.csv"
            with patch("retraining.RECORDS_PATH", ledger):
                with self.assertRaisesRegex(ValueError, "diagnosis"):
                    register_confirmed_records(
                        self._labeled_rows().rename(columns={"diagnosis": "Prediction"}),
                        admin_confirmed=True,
                        principal=self.admin,
                    )
                with self.assertRaises(PermissionError):
                    register_confirmed_records(
                        self._labeled_rows(), admin_confirmed=False, principal=self.admin
                    )
                self.assertFalse(ledger.exists())

    def test_confirmed_registration_is_idempotent_for_repeated_upload_rows(self):
        frame = self._labeled_rows(8)
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "confirmed.csv"
            with patch("retraining.RECORDS_PATH", ledger):
                self.assertEqual(
                    register_confirmed_records(
                        frame, admin_confirmed=True, principal=self.admin
                    ),
                    len(frame) - 1,
                )
                self.assertEqual(
                    register_confirmed_records(
                        frame, admin_confirmed=True, principal=self.admin
                    ),
                    0,
                )
                self.assertEqual(len(pd.read_csv(ledger)), len(frame) - 1)

    def test_candidate_requires_threshold_and_explicit_approval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            ledger = base / "confirmed.csv"
            candidates = base / "candidates"
            active = base / "active_model.json"
            with (
                patch("retraining.RECORDS_PATH", ledger),
                patch("retraining.CANDIDATE_DIR", candidates),
                patch("retraining.ACTIVE_MODEL_PATH", active),
            ):
                register_confirmed_records(
                    self._labeled_rows(), admin_confirmed=True, principal=self.admin
                )
                with self.assertRaisesRegex(ValueError, "at least 6"):
                    create_candidate(minimum_records=4, principal=self.admin)
                with self.assertRaisesRegex(ValueError, "threshold"):
                    create_candidate(minimum_records=100, principal=self.admin)

                candidates.mkdir()
                candidate_id = "test-candidate"
                (candidates / f"{candidate_id}.joblib").write_bytes(b"test artifact")
                (candidates / f"{candidate_id}.json").write_text(
                    json.dumps({
                        "candidate_id": candidate_id,
                        "status": "pending_approval",
                        "model": f"candidates/{candidate_id}.joblib",
                    }),
                    encoding="utf-8",
                )
                with self.assertRaises(PermissionError):
                    approve_candidate(
                        candidate_id, admin_approved=False, principal=self.admin
                    )
                self.assertFalse(active.exists())

    def test_confirmed_records_create_evaluate_then_activate_candidate(self):
        rng = np.random.default_rng(915)
        synthetic_features = rng.normal(size=(100, len(FEATURES)))
        frame = pd.DataFrame(synthetic_features, columns=FEATURES)
        frame["diagnosis"] = (synthetic_features[:, 0] > 0).astype(int)

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            model_dir = base / "models"
            candidates = model_dir / "candidates"
            ledger = base / "confirmed.csv"
            active = model_dir / "active_model.json"
            with (
                patch("retraining.RECORDS_PATH", ledger),
                patch("retraining.CANDIDATE_DIR", candidates),
                patch("retraining.ACTIVE_MODEL_PATH", active),
            ):
                initial_active = {
                    "name": "Existing active model",
                    "kind": "pipeline",
                    "model": "centralized_logistic_regression.pkl",
                }
                active.parent.mkdir(parents=True, exist_ok=True)
                active.write_text(json.dumps(initial_active), encoding="utf-8")
                register_confirmed_records(
                    frame, admin_confirmed=True, principal=self.admin
                )
                manifest = create_candidate(minimum_records=100, principal=self.admin)
                self.assertEqual(manifest["status"], "pending_approval")
                self.assertEqual(manifest["evaluation_records"], 20)
                self.assertIn("candidate", manifest["metrics"])
                self.assertIn("active", manifest["metrics"])
                self.assertEqual(
                    manifest["active_model_snapshot"]["name"],
                    "Existing active model",
                )
                self.assertEqual(json.loads(active.read_text(encoding="utf-8")), initial_active)
                self.assertEqual(len(pd.read_csv(ledger)["consumed_candidate"].dropna().unique()), 1)

                approved = approve_candidate(
                    manifest["candidate_id"],
                    admin_approved=True,
                    principal=self.admin,
                )
                self.assertTrue(active.is_file())
                self.assertEqual(approved["candidate_id"], manifest["candidate_id"])
                with patch("inference.MODEL_DIR", model_dir):
                    probabilities = predict_probabilities(
                        frame[FEATURES].iloc[:3],
                        active_path=active,
                    )
                self.assertEqual(len(probabilities), 3)
                self.assertTrue(np.isfinite(probabilities).all())


    def test_candidate_cannot_activate_if_active_model_changed_after_evaluation(self):
        features = self._labeled_rows(100)
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            model_dir = base / "models"
            candidates = model_dir / "candidates"
            active = model_dir / "active_model.json"
            ledger = base / "confirmed.csv"
            with (
                patch("retraining.RECORDS_PATH", ledger),
                patch("retraining.CANDIDATE_DIR", candidates),
                patch("retraining.ACTIVE_MODEL_PATH", active),
            ):
                register_confirmed_records(
                    features, admin_confirmed=True, principal=self.admin
                )
                manifest = create_candidate(minimum_records=100, principal=self.admin)
                changed_active = {
                    "name": "Another approved model",
                    "kind": "pipeline",
                    "model": "another_model.pkl",
                }
                active.parent.mkdir(parents=True, exist_ok=True)
                active.write_text(json.dumps(changed_active), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "changed after candidate evaluation"):
                    approve_candidate(
                        manifest["candidate_id"],
                        admin_approved=True,
                        principal=self.admin,
                    )
                self.assertEqual(json.loads(active.read_text(encoding="utf-8")), changed_active)


class DPAccountantTests(unittest.TestCase):
    def test_rejects_non_finite_parameters_and_handles_tiny_valid_delta(self):
        for parameters in (
            {"noise_multiplier": np.inf},
            {"noise_multiplier": 1.0, "sensitivity": np.nan},
            {"noise_multiplier": 1.0, "delta": np.nan},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                DPAccountant(**parameters)

        accountant = DPAccountant(noise_multiplier=2.0, delta=5e-324)
        accountant.record_step()
        self.assertTrue(np.isfinite(accountant.get_epsilon()))
        self.assertGreater(accountant.get_epsilon(), 0)

    def test_rdp_composes_heterogeneous_gaussian_steps_and_rejects_bad_counts(self):
        accountant = DPAccountant(noise_multiplier=1.0, delta=1e-5)
        accountant.record_steps(2, noise_multiplier=2.0)
        accountant.record_step(noise_multiplier=1.0)
        expected_rdp = (
            2 * accountant.orders / (2 * 2.0 ** 2)
            + accountant.orders / (2 * 1.0 ** 2)
        )
        expected_epsilon = np.min(
            expected_rdp + (-np.log(accountant.delta)) / (accountant.orders - 1)
        )
        self.assertEqual(accountant.steps, 3)
        self.assertAlmostEqual(accountant.get_epsilon(), expected_epsilon)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            accountant.record_steps(-1)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            accountant.record_steps(True)
        report = accountant.get_privacy_report()
        self.assertEqual(
            report["method"],
            "Gaussian mechanism with integer-order RDP composition",
        )


class LiveEvaluationTests(unittest.TestCase):
    def test_comparison_uses_common_heldout_records_and_reports_actual_predictions(self):
        results, metadata = run_live_model_comparison(
            rounds=1,
            local_epochs=1,
            noise_multipliers=(1.0,),
        )
        self.assertEqual(
            set(results["approach"]),
            {
                "Centralized Logistic Regression",
                "FedAvg Logistic Regression",
                "DP-FedAvg Logistic Regression (experimental)",
            },
        )
        self.assertEqual(set(results["validation_records"]), {114})
        self.assertEqual(metadata["training_records"], 455)
        self.assertEqual(metadata["held_out_records"], 114)
        self.assertEqual(metadata["scaler_fit_records"], 455)
        self.assertEqual(sum(
            item["records"] for item in metadata["hospital_partitions"]
        ), 455)
        self.assertTrue(np.isfinite(
            results[["accuracy", "precision", "malignant_recall", "f1_score", "roc_auc"]]
        ).all().all())
        dp_row = results.loc[results["noise_multiplier"].notna()].iloc[0]
        self.assertEqual(dp_row["accounted_client_epoch_mechanisms"], 3)
        self.assertGreater(dp_row["epsilon_upper_bound"], 0)

    def test_iid_noniid_and_noise_utility_use_training_only_validation(self):
        results, metadata = run_iid_noniid_validation(
            rounds=1,
            local_epochs=1,
            noise_multipliers=(1.0,),
        )
        self.assertEqual(set(results["partition"]), {"IID", "Non-IID"})
        self.assertEqual(set(results["validation_records"]), {91})
        self.assertEqual(metadata["training_records"], 364)
        self.assertEqual(metadata["validation_records"], 91)
        self.assertEqual(metadata["scaler_fit_records"], 364)
        self.assertEqual(
            set(results["approach"]),
            {
                "FedAvg Logistic Regression",
                "DP-FedAvg Logistic Regression (experimental)",
            },
        )
        self.assertTrue(np.isfinite(
            results[["accuracy", "precision", "malignant_recall", "f1_score", "roc_auc"]]
        ).all().all())


if __name__ == "__main__":
    unittest.main()
