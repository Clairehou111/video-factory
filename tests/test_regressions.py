import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from video_factory.regressions import validate_regression_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RegressionRegistryTest(unittest.TestCase):
    def test_every_fixed_bug_references_an_existing_test_case(self) -> None:
        self.assertEqual(validate_regression_registry(PROJECT_ROOT), [])

    def test_fixed_bug_without_test_is_rejected(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "regressions.json").write_text(
                '[{"id":"VF-X","title":"bug","stage":"render",'
                '"reproduction":"steps","impact":"bad","status":"fixed","tests":[]}]',
                encoding="utf-8",
            )
            self.assertIn(
                "VF-X: fixed bug must reference at least one regression test",
                validate_regression_registry(root),
            )


if __name__ == "__main__":
    unittest.main()
