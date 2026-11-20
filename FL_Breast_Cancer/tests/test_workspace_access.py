import unittest

from workspace_access import (
    ADMIN_PAGES,
    HOSPITAL_PAGES,
    page_is_allowed,
    pages_for_role,
)


class WorkspaceAccessTests(unittest.TestCase):
    def test_admin_keeps_all_existing_pages(self):
        self.assertEqual(pages_for_role("admin"), ADMIN_PAGES)
        self.assertEqual(
            ADMIN_PAGES,
            (
                "Home",
                "Overview",
                "Hospitals",
                "Models",
                "Privacy",
                "Prediction",
                "Retraining",
                "Federated Operations",
                "Experimental Evaluation",
            ),
        )

    def test_hospital_navigation_contains_only_hospital_pages(self):
        self.assertEqual(pages_for_role("hospital"), HOSPITAL_PAGES)
        self.assertEqual(
            HOSPITAL_PAGES,
            (
                "Home",
                "Local Dataset",
                "Local Training",
                "Federated Participation",
                "Prediction",
                "Prediction History",
            ),
        )
        for page in HOSPITAL_PAGES:
            with self.subTest(page=page):
                self.assertTrue(page_is_allowed("hospital", page))
        for page in set(ADMIN_PAGES) - set(HOSPITAL_PAGES):
            with self.subTest(page=page):
                self.assertFalse(page_is_allowed("hospital", page))

    def test_unknown_roles_and_pages_fail_closed(self):
        self.assertFalse(page_is_allowed("unknown", "Home"))
        self.assertFalse(page_is_allowed("hospital", "Federated Operations"))
        with self.assertRaises(ValueError):
            pages_for_role("unknown")


if __name__ == "__main__":
    unittest.main()
