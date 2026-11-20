import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import app_services
import inference
from app_services import cancel_federated_round
from feature_schema import FEATURES
from retraining import approve_candidate


BASE_DIR = Path(__file__).resolve().parents[1]
TEST_PASSWORDS = {
    "FL_ADMIN_PASSWORD": "test-admin-password",
    "FL_HOSPITAL1_PASSWORD": "test-hospital1-password",
    "FL_HOSPITAL2_PASSWORD": "test-hospital2-password",
    "FL_HOSPITAL3_PASSWORD": "test-hospital3-password",
}
TEST_CREDENTIALS = {
    "admin": TEST_PASSWORDS["FL_ADMIN_PASSWORD"],
    "hospital1": TEST_PASSWORDS["FL_HOSPITAL1_PASSWORD"],
    "hospital2": TEST_PASSWORDS["FL_HOSPITAL2_PASSWORD"],
    "hospital3": TEST_PASSWORDS["FL_HOSPITAL3_PASSWORD"],
}


class AuthenticatedFederatedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.models = root / "saved_models"
        self.candidates = self.models / "candidates"
        self.models.mkdir()
        self.candidates.mkdir()
        self.active_path = self.models / "active_model.json"
        self.patches = [
            patch.dict(os.environ, TEST_PASSWORDS),
            patch.object(app_services, "STATE_PATH", root / "state.sqlite3"),
            patch.object(app_services, "MODEL_DIR", self.models),
            patch.object(app_services, "CANDIDATE_DIR", self.candidates),
            patch.object(app_services, "ACTIVE_MODEL_PATH", self.active_path),
            patch.object(inference, "MODEL_DIR", self.models),
        ]
        for item in self.patches:
            item.start()
        app_services.initialize()
        self.admin = app_services.authenticate("admin", TEST_CREDENTIALS["admin"])
        self.hospitals = [
            app_services.authenticate(
                f"hospital{number}", TEST_CREDENTIALS[f"hospital{number}"]
            )
            for number in range(1, 4)
        ]

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp_dir.cleanup()

    def test_accounts_are_authenticated_and_hospital_services_are_role_scoped(self):
        self.assertEqual(self.admin.role, "admin")
        self.assertEqual(
            [hospital.role for hospital in self.hospitals],
            ["hospital", "hospital", "hospital"],
        )
        self.assertIsNone(app_services.authenticate("admin", "incorrect"))
        self.assertEqual([item.hospital_id for item in self.hospitals], list(app_services.HOSPITALS))
        self.assertEqual(len(app_services.all_hospital_statuses(self.admin)), 3)
        with self.assertRaises(PermissionError):
            app_services.all_hospital_statuses(self.hospitals[0])
        with self.assertRaises(PermissionError):
            app_services.hospital_status(self.admin)
        with self.assertRaises(PermissionError):
            app_services.audit_log(self.hospitals[0])
        with self.assertRaises(PermissionError):
            app_services.start_federated_round(self.hospitals[0])

    def test_demo_accounts_require_distinct_passwords(self):
        for username, password in TEST_CREDENTIALS.items():
            self.assertIsNotNone(app_services.authenticate(username, password))
            for other_username, other_password in TEST_CREDENTIALS.items():
                if other_username != username:
                    self.assertIsNone(
                        app_services.authenticate(other_username, password),
                        f"{username}'s password authenticated as {other_username}",
                    )

    def test_initialization_rejects_missing_or_shared_password_configuration(self):
        cases = (
            ({}, RuntimeError, "Set a separate password"),
            ({"FL_DEMO_PASSWORD": "legacy-shared"}, RuntimeError, "no longer supported"),
            (
                {
                    environment_name: "same-test-password"
                    for environment_name in TEST_PASSWORDS
                },
                ValueError,
                "distinct password",
            ),
        )
        for environment, error_type, message in cases:
            with self.subTest(environment=environment):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with (
                        patch.object(
                            app_services, "STATE_PATH", Path(temp_dir) / "state.sqlite3"
                        ),
                        patch.dict(os.environ, environment, clear=True),
                    ):
                        with self.assertRaisesRegex(error_type, message):
                            app_services.initialize()

    def test_legacy_shared_password_accounts_are_migrated_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "legacy.sqlite3"
            with patch.object(app_services, "STATE_PATH", state_path):
                app_services.initialize()
                legacy_salt = b"legacy-shared-salt"
                with app_services._connect() as connection:
                    connection.execute("DELETE FROM app_metadata")
                    connection.execute(
                        "UPDATE users SET salt = ?, password_hash = ?",
                        (
                            legacy_salt,
                            app_services._password_hash("legacy-shared", legacy_salt),
                        ),
                    )
                for username, password in TEST_CREDENTIALS.items():
                    principal = app_services.authenticate(username, password)
                    self.assertIsNotNone(principal, username)
                    self.assertIsNone(
                        app_services.authenticate(username, "legacy-shared"),
                        username,
                    )
                with app_services._connect() as connection:
                    version = connection.execute(
                        "SELECT value FROM app_metadata WHERE key = 'account_password_version'"
                    ).fetchone()
                self.assertEqual(version["value"], "2")
                with patch.dict(os.environ, {}, clear=True):
                    app_services.initialize()
                    self.assertIsNotNone(
                        app_services.authenticate(
                            "hospital1", TEST_CREDENTIALS["hospital1"]
                        )
                    )

    def test_admin_can_cancel_an_incomplete_round_without_creating_a_candidate(self):
        round_id = app_services.start_federated_round(
            self.admin, mode="FedAvg", local_epochs=1
        )
        app_services.submit_hospital_update(self.hospitals[0], round_id)

        with self.assertRaises(PermissionError):
            cancel_federated_round(self.hospitals[0], round_id)

        result = cancel_federated_round(self.admin, round_id)
        self.assertEqual(result, {"round_id": round_id, "status": "cancelled"})
        cancelled = next(
            item for item in app_services.list_rounds(self.admin)
            if item["round_id"] == round_id
        )
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIsNone(cancelled["candidate"])
        self.assertEqual(app_services.hospital_status(self.hospitals[1])["open_rounds"], 0)
        self.assertFalse(self.active_path.exists())

        with self.assertRaisesRegex(ValueError, "not accepting updates"):
            app_services.submit_hospital_update(self.hospitals[1], round_id)
        with self.assertRaisesRegex(ValueError, "Only a collecting"):
            cancel_federated_round(self.admin, round_id)

    def test_privacy_accounting_includes_partial_cancelled_dp_updates(self):
        round_id = app_services.start_federated_round(
            self.admin,
            mode="DP-FedAvg (experimental)",
            local_epochs=2,
            clipping_norm=1.0,
            noise_multiplier=2.0,
            delta=1e-5,
        )
        submitted = app_services.submit_hospital_update(self.hospitals[0], round_id)
        self.assertIsNone(submitted["training_loss"])
        app_services.cancel_federated_round(self.admin, round_id)

        report = app_services.dp_privacy_accounting(self.admin)
        self.assertEqual(report["accounted_client_updates"], 1)
        self.assertEqual(report["accounted_client_epoch_mechanisms"], 2)
        self.assertEqual(report["accounted_rounds_with_updates"], 1)
        self.assertEqual(report["accounted_rounds"][0]["status"], "cancelled")
        self.assertGreater(report["epsilon_upper_bound"], 0)
        with self.assertRaises(PermissionError):
            app_services.dp_privacy_accounting(self.hospitals[0])

    def test_forged_or_reassigned_principals_cannot_cross_tenants_or_admin_boundaries(self):
        round_id = app_services.start_federated_round(
            self.admin, mode="FedAvg", local_epochs=1
        )
        hospital2 = self.hospitals[1]
        fabricated_hospital2 = app_services.Principal(
            user_id=hospital2.user_id,
            username=hospital2.username,
            role=hospital2.role,
            hospital_id=hospital2.hospital_id,
        )
        reassigned_hospital1 = replace(
            self.hospitals[0], hospital_id=hospital2.hospital_id
        )
        forged_admin = app_services.Principal(
            user_id=self.admin.user_id,
            username=self.admin.username,
            role=self.admin.role,
            hospital_id=self.admin.hospital_id,
        )

        tampered_hospital1 = (
            reassigned_hospital1,
            replace(self.hospitals[0], user_id=hospital2.user_id),
            replace(self.hospitals[0], username=hospital2.username),
            replace(self.hospitals[0], role="admin"),
        )
        for principal in (fabricated_hospital2, *tampered_hospital1):
            for operation in (
                app_services.hospital_status,
                app_services.prediction_history,
                app_services.list_rounds,
            ):
                with self.subTest(principal=principal, operation=operation.__name__):
                    with self.assertRaises(PermissionError):
                        operation(principal)
            with self.assertRaises(PermissionError):
                app_services.submit_hospital_update(principal, round_id)
            with self.assertRaises(PermissionError):
                app_services.make_hospital_predictions(
                    principal,
                    pd.DataFrame(
                        np.ones((1, len(FEATURES))), columns=FEATURES
                    ),
                )
            with self.assertRaises(PermissionError):
                approve_candidate(
                    "not-a-candidate", admin_approved=True, principal=principal
                )
            with self.assertRaises(PermissionError):
                app_services.make_hospital_predictions(
                    principal,
                    pd.DataFrame(
                        np.ones((1, len(FEATURES))), columns=FEATURES
                    ),
                )

        with self.assertRaises(PermissionError):
            app_services.all_hospital_statuses(forged_admin)
        with self.assertRaises(PermissionError):
            app_services.start_federated_round(forged_admin)
        with self.assertRaises(PermissionError):
            cancel_federated_round(forged_admin, round_id)
        for approve in (True, False):
            with self.subTest(approve=approve):
                with self.assertRaises(PermissionError):
                    app_services.review_candidate(
                        forged_admin, "not-a-candidate", approve=approve
                    )
                with self.assertRaises(PermissionError):
                    approve_candidate(
                        "not-a-candidate",
                        admin_approved=True,
                        principal=forged_admin,
                    )

        round_state = next(
            item for item in app_services.list_rounds(self.admin)
            if item["round_id"] == round_id
        )
        self.assertEqual(round_state["status"], "collecting")
        self.assertEqual(round_state["participants"], 0)
        self.assertFalse(self.active_path.exists())

    def test_failed_round_cannot_be_deployed_and_remains_cancellable(self):
        active_before = {
            "name": "Previously approved fixture",
            "version": "v0007",
            "model": "existing-fixture.npz",
        }
        self.active_path.write_text(json.dumps(active_before), encoding="utf-8")
        round_id = app_services.start_federated_round(
            self.admin, mode="FedAvg", local_epochs=1
        )

        original_completion = app_services._complete_round_if_ready

        def fail_after_all_updates(completed_round_id):
            with app_services._connect() as connection:
                update_count = connection.execute(
                    "SELECT COUNT(*) FROM client_updates WHERE round_id = ?",
                    (completed_round_id,),
                ).fetchone()[0]
            if update_count == len(self.hospitals):
                raise RuntimeError("simulated candidate evaluation failure")
            original_completion(completed_round_id)

        with patch.object(
            app_services,
            "_complete_round_if_ready",
            side_effect=fail_after_all_updates,
        ):
            for hospital in self.hospitals[:-1]:
                app_services.submit_hospital_update(hospital, round_id)
            with self.assertRaisesRegex(RuntimeError, "simulated candidate evaluation failure"):
                app_services.submit_hospital_update(self.hospitals[-1], round_id)

        round_state = next(
            item for item in app_services.list_rounds(self.admin)
            if item["round_id"] == round_id
        )
        self.assertEqual(round_state["status"], "collecting")
        self.assertEqual(round_state["participants"], 3)
        self.assertIsNone(round_state["candidate"])
        with self.assertRaisesRegex(ValueError, "No pending candidate"):
            app_services.review_candidate(
                self.admin, f"fl-{round_id}", approve=True
            )
        app_services.cancel_federated_round(self.admin, round_id)
        self.assertEqual(
            json.loads(self.active_path.read_text(encoding="utf-8")),
            active_before,
        )
        self.assertEqual(
            next(
                item for item in app_services.list_rounds(self.admin)
                if item["round_id"] == round_id
            )["status"],
            "cancelled",
        )

    def test_three_clients_train_candidate_admin_approves_and_prediction_history_is_isolated(self):
        round_id = app_services.start_federated_round(
            self.admin, mode="FedAvg", local_epochs=2
        )
        training_partition = app_services._training_data()[0]
        partition_labels = app_services._training_data()[3].reset_index(drop=True)
        client_partition_indices = app_services.partition_hospital_indices(
            partition_labels, seed=42
        )
        self.assertEqual(len(client_partition_indices), len(app_services.HOSPITALS))
        self.assertEqual(
            sum(len(indices) for indices in client_partition_indices),
            len(training_partition),
        )
        self.assertEqual(
            len(np.unique(np.concatenate(client_partition_indices))),
            len(training_partition),
        )
        client_partition_calls = []
        original_partition = app_services._hospital_training_partition

        def observed_partition(hospital_id):
            client_partition_calls.append(hospital_id)
            return original_partition(hospital_id)

        with patch.object(
            app_services,
            "_hospital_training_partition",
            side_effect=observed_partition,
        ):
            for hospital in self.hospitals:
                submitted = app_services.submit_hospital_update(hospital, round_id)
                self.assertGreater(submitted["sample_count"], 0)
                expected_count = len(
                    original_partition(hospital.hospital_id)[1]
                )
                self.assertEqual(submitted["sample_count"], expected_count)
        self.assertEqual(client_partition_calls, list(app_services.HOSPITALS))
        rounds = app_services.list_rounds(self.admin)
        completed = next(item for item in rounds if item["round_id"] == round_id)
        self.assertEqual(completed["status"], "candidate_pending")
        self.assertEqual(completed["participants"], 3)
        self.assertEqual(completed["expected_participants"], 3)
        self.assertEqual(completed["aggregation_status"], "aggregated_pending_admin_approval")
        self.assertEqual(completed["aggregation_method"], "sample-count-weighted FedAvg")
        self.assertIsNone(completed["global_model_version"])
        self.assertAlmostEqual(
            sum(
                item["aggregation_weight"]
                for item in completed["participant_results"]
            ),
            1.0,
        )
        self.assertEqual(
            {
                item["sample_count"]
                for item in completed["participant_results"]
            },
            {
                len(
                    app_services._hospital_training_partition(
                        hospital.hospital_id
                    )[1]
                )
                for hospital in self.hospitals
            },
        )
        with app_services._connect() as connection:
            stored_updates = connection.execute(
                """
                SELECT sample_count, parameters_json FROM client_updates
                WHERE round_id = ? ORDER BY hospital_id
                """,
                (round_id,),
            ).fetchall()
            aggregated_round = connection.execute(
                "SELECT parameters_json FROM rounds WHERE round_id = ?",
                (round_id,),
            ).fetchone()
        self.assertEqual(len(stored_updates), 3)
        client_parameters = []
        for stored in stored_updates:
            client_parameters.append({
                "num_patients": stored["sample_count"],
                "parameters": json.loads(stored["parameters_json"]),
            })
            self.assertEqual(
                set(json.loads(stored["parameters_json"])),
                {"weights", "bias"},
            )
        expected_aggregate = app_services.federated_averaging(client_parameters)
        saved_aggregate = json.loads(aggregated_round["parameters_json"])
        np.testing.assert_allclose(
            saved_aggregate["weights"],
            expected_aggregate["weights"],
        )
        self.assertAlmostEqual(saved_aggregate["bias"], expected_aggregate["bias"])
        self.assertEqual(completed["candidate"]["mode"], "FedAvg")
        self.assertEqual(completed["candidate"]["metrics"]["holdout_records"], 114)
        self.assertFalse(self.active_path.exists())

        for hospital in self.hospitals:
            hospital_view = next(
                item for item in app_services.list_rounds(hospital)
                if item["round_id"] == round_id
            )
            self.assertEqual(len(hospital_view["participant_results"]), 1)
            self.assertEqual(
                hospital_view["participant_results"][0]["hospital"],
                app_services.HOSPITALS[hospital.hospital_id],
            )
        with self.assertRaises(ValueError):
            app_services.submit_hospital_update(self.hospitals[0], round_id)
        for hospital in self.hospitals:
            for approve in (True, False):
                with self.subTest(hospital=hospital.username, approve=approve):
                    with self.assertRaises(PermissionError):
                        app_services.review_candidate(
                            hospital,
                            completed["candidate"]["candidate_id"],
                            approve=approve,
                        )

        deployed = app_services.review_candidate(
            self.admin, completed["candidate"]["candidate_id"], approve=True
        )
        self.assertEqual(deployed["version"], "v0001")
        self.assertEqual(json.loads(self.active_path.read_text())["candidate_id"],
                         completed["candidate"]["candidate_id"])
        approved_round = next(
            item
            for item in app_services.list_rounds(self.admin)
            if item["round_id"] == round_id
        )
        self.assertEqual(approved_round["aggregation_status"], "approved_and_deployed")
        self.assertEqual(approved_round["global_model_version"], "v0001")
        self.assertEqual(app_services.current_model(self.admin), deployed)
        frame = pd.DataFrame(
            np.random.default_rng(10).uniform(0.1, 1.0, (2, len(FEATURES))),
            columns=FEATURES,
        )
        result = app_services.make_hospital_predictions(self.hospitals[0], frame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["Model version"].tolist(), [deployed["version"]] * 2)
        self.assertEqual(len(app_services.prediction_history(self.hospitals[0])), 2)
        for index, hospital in enumerate(self.hospitals[1:], start=1):
            other_frame = frame * (index + 1)
            other_results = app_services.make_hospital_predictions(
                hospital, other_frame
            )
            self.assertEqual(len(other_results), 2)
            self.assertEqual(
                len(app_services.prediction_history(hospital)),
                len(other_results),
            )
        for hospital in self.hospitals:
            self.assertEqual(len(app_services.prediction_history(hospital)), 2)
        with self.assertRaises(PermissionError):
            app_services.make_hospital_predictions(self.admin, frame)

        dp_round = app_services.start_federated_round(
            self.admin,
            mode="DP-FedAvg (experimental)",
            local_epochs=1,
            clipping_norm=0.5,
            noise_multiplier=2.0,
            delta=1e-5,
        )
        for hospital in self.hospitals:
            submitted = app_services.submit_hospital_update(hospital, dp_round)
            self.assertIsNone(submitted["training_loss"])
        dp_round_result = next(
            item for item in app_services.list_rounds(self.admin)
            if item["round_id"] == dp_round
        )
        dp_candidate = dp_round_result["candidate"]
        self.assertEqual(dp_candidate["mode"], "DP-FedAvg (experimental)")
        self.assertEqual(dp_candidate["status"], "pending_approval")
        with app_services._connect() as connection:
            dp_updates = connection.execute(
                """
                SELECT sample_count, parameters_json FROM client_updates
                WHERE round_id = ? ORDER BY hospital_id
                """,
                (dp_round,),
            ).fetchall()
            dp_aggregate = connection.execute(
                "SELECT parameters_json FROM rounds WHERE round_id = ?",
                (dp_round,),
            ).fetchone()
        expected_dp_aggregate = app_services.federated_averaging([
            {
                "num_patients": row["sample_count"],
                "parameters": json.loads(row["parameters_json"]),
            }
            for row in dp_updates
        ])
        stored_dp_aggregate = json.loads(dp_aggregate["parameters_json"])
        np.testing.assert_allclose(
            stored_dp_aggregate["weights"],
            expected_dp_aggregate["weights"],
        )
        self.assertAlmostEqual(
            stored_dp_aggregate["bias"],
            expected_dp_aggregate["bias"],
        )
        self.assertTrue(all(
            item["training_loss"] is None
            for item in dp_round_result["participant_results"]
        ))
        self.assertGreater(dp_candidate["privacy_accounting"]["epsilon_upper_bound"], 0)
        self.assertEqual(
            dp_candidate["privacy_accounting"]["accounted_client_epoch_mechanisms"], 3
        )
        self.assertEqual(
            dp_candidate["privacy_accounting"]["accountant_method"],
            "Gaussian mechanism with integer-order RDP composition",
        )
        self.assertEqual(
            json.loads(self.active_path.read_text())["candidate_id"],
            completed["candidate"]["candidate_id"],
        )
        before_reject = json.loads(self.active_path.read_text(encoding="utf-8"))
        with self.assertRaises(PermissionError):
            app_services.review_candidate(
                self.hospitals[0],
                dp_candidate["candidate_id"],
                approve=False,
            )
        rejected = app_services.review_candidate(
            self.admin, dp_candidate["candidate_id"], approve=False
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(
            json.loads(self.active_path.read_text(encoding="utf-8")),
            before_reject,
        )

    def test_dp_fedavg_is_a_distinct_clipped_noisy_client_path(self):
        X = np.ones((4, 2), dtype=float)
        y = np.array([0, 0, 1, 1], dtype=float)
        parameters = {"weights": np.zeros(2), "bias": 0.0}
        class FixedNoise:
            def __init__(self):
                self.values = iter([0.2, -0.3, 0.4])
                self.calls = []

            def gauss(self, mean, standard_deviation):
                self.calls.append((mean, standard_deviation))
                return next(self.values)

        noise_source = FixedNoise()
        with (
            patch.object(
                app_services.secrets,
                "SystemRandom",
                return_value=noise_source,
            ) as system_random_factory,
            patch.object(
                app_services.np.random,
                "default_rng",
                side_effect=AssertionError("privacy noise must not use a seeded PRNG"),
            ),
        ):
            noisy_parameters, _ = app_services._train_dp_local(
                np.array([[1000.0, 0.0]]),
                np.array([0.0]),
                parameters,
                epochs=1,
                learning_rate=0.2,
                clip=0.1,
                sigma=2.0,
            )
        system_random_factory.assert_called_once_with()
        self.assertEqual(len(noise_source.calls), 3)
        self.assertEqual(
            {standard_deviation for _, standard_deviation in noise_source.calls},
            {0.4},
        )
        gradient = np.array([500.0, 0.0, 0.5])
        clipped_gradient = gradient * (0.1 / np.linalg.norm(gradient))
        expected = -0.2 * (clipped_gradient + np.array([0.2, -0.3, 0.4]))
        np.testing.assert_allclose(
            noisy_parameters["weights"],
            expected[:2],
            atol=1e-12,
        )
        self.assertAlmostEqual(noisy_parameters["bias"], expected[2])
        privacy = app_services.privacy_description(self.admin)
        self.assertIn("not an isolation boundary", privacy["limitations"])
        self.assertIn("noisy final client model update", privacy["server_observes"])

    def test_dp_clipping_bounds_each_concatenated_record_gradient(self):
        gradients = np.array([
            [3.0, 4.0, 0.0],
            [0.3, 0.4, 0.0],
            [0.0, 0.0, 0.0],
        ])
        clipped = app_services._clip_per_record_gradients(gradients, 1.0)
        norms = np.linalg.norm(clipped, axis=1)
        np.testing.assert_allclose(norms, [1.0, 0.5, 0.0])
        self.assertTrue(np.all(norms <= 1.0))
        np.testing.assert_allclose(clipped[1:], gradients[1:])
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            app_services._clip_per_record_gradients(gradients, 0.0)

    def test_dp_noise_uses_full_batch_replace_one_sensitivity_scale(self):
        class RecordingNoise:
            def __init__(self):
                self.calls = []

            def gauss(self, mean, standard_deviation):
                self.calls.append((mean, standard_deviation))
                return 0.0

        noise_source = RecordingNoise()
        X = np.array([[1000.0, 0.0], [0.0, 1000.0]])
        y = np.array([0.0, 0.0])
        parameters = {"weights": np.zeros(2), "bias": 0.0}
        with patch.object(
            app_services.secrets,
            "SystemRandom",
            return_value=noise_source,
        ) as system_random_factory:
            app_services._train_dp_local(
                X,
                y,
                parameters,
                epochs=2,
                learning_rate=0.2,
                clip=0.1,
                sigma=2.0,
            )
        system_random_factory.assert_called_once_with()
        self.assertEqual(len(noise_source.calls), 6)
        self.assertEqual(
            {standard_deviation for _, standard_deviation in noise_source.calls},
            {2.0 * (2.0 * 0.1 / len(y))},
        )
        with self.assertRaisesRegex(ValueError, "at least one record"):
            app_services._train_dp_local(
                np.empty((0, 2)),
                np.empty(0),
                parameters,
                epochs=1,
                learning_rate=0.2,
                clip=0.1,
                sigma=2.0,
            )


if __name__ == "__main__":
    unittest.main()
