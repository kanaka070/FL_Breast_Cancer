import unittest

from local_demo import is_local_demo_request, local_demo_mode_enabled


class LocalDemoModeTests(unittest.TestCase):
    def test_demo_mode_is_disabled_unless_exactly_enabled(self):
        self.assertFalse(local_demo_mode_enabled({}))
        self.assertFalse(local_demo_mode_enabled({"FL_LOCAL_DEMO_MODE": "true"}))
        self.assertFalse(local_demo_mode_enabled({"FL_LOCAL_DEMO_MODE": "0"}))
        self.assertTrue(local_demo_mode_enabled({"FL_LOCAL_DEMO_MODE": "1"}))

    def test_demo_access_requires_loopback_bind_and_host(self):
        self.assertTrue(is_local_demo_request(
            "127.0.0.1",
            {"Host": "localhost:8501"},
        ))
        self.assertTrue(is_local_demo_request(
            "::1",
            {"host": "[::1]:8501"},
        ))
        self.assertFalse(is_local_demo_request(
            None,
            {"host": "localhost:8501"},
        ))
        self.assertFalse(is_local_demo_request(
            "0.0.0.0",
            {"host": "localhost:8501"},
        ))
        self.assertFalse(is_local_demo_request(
            "127.0.0.1",
            {"host": "demo.example.edu"},
        ))

    def test_demo_access_rejects_forwarded_requests_and_invalid_hosts(self):
        self.assertFalse(is_local_demo_request(
            "127.0.0.1",
            {"host": "localhost:8501", "x-forwarded-for": "127.0.0.1"},
        ))
        self.assertFalse(is_local_demo_request(
            "127.0.0.1",
            {"host": "localhost:8501", "X-Forwarded-Host": "localhost:8501"},
        ))
        self.assertFalse(is_local_demo_request(
            "127.0.0.1",
            {"host": "attacker@localhost:8501"},
        ))
        self.assertFalse(is_local_demo_request(
            "127.0.0.1",
            {"host": "localhost:invalid"},
        ))
        self.assertFalse(is_local_demo_request("127.0.0.1", {}))


if __name__ == "__main__":
    unittest.main()
