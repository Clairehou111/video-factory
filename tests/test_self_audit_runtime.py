from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.self_audit_runtime import (
    OpenRouterGeminiAuditModel, RepositoryCandidateVerifier, ReviewBranchExecutor,
)


class FakeWriter:
    def __init__(self) -> None:
        self.messages = []

    def _request_json(self, messages, max_tokens):
        self.messages = messages
        return (
            {"kind": "diagnosis", "rationale": "wrong story hierarchy"},
            {"provider": "openrouter", "model": "google/gemini-3.7-flash", "usage": {"cost": 0.04}},
        )


class SelfAuditRuntimeTest(unittest.TestCase):
    def test_gemini_adapter_receives_product_position_and_archived_boundary(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "src" / "video_factory"
            source.mkdir(parents=True)
            (source / "factory.py").write_text("# factory", encoding="utf-8")
            (source / "automation.py").write_text("# automation", encoding="utf-8")
            writer = FakeWriter()
            model = OpenRouterGeminiAuditModel(writer, root)

            result = model.propose(
                {"stage": "pipeline", "observed": "failed"},
                {"mode": "archived_only", "external_actions_allowed": False},
                0.5,
            )

            prompt = writer.messages[-1]["content"]
            self.assertIn("BGM-only visual technology-intelligence radar", prompt)
            self.assertIn("external actions are forbidden", prompt)
            self.assertIn("# factory", prompt)
            self.assertEqual(result["cost_usd"], 0.04)

    def test_repository_verifier_runs_only_fixed_unittest_command(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "ok", "")
        proposal = {"verification": {
            "reproduction_fixed": True, "grounding": True, "safety": True,
            "golden_regressions": True, "confidence": 0.8,
        }}
        with TemporaryDirectory() as temp, patch(
            "video_factory.self_audit_runtime.subprocess.run", return_value=completed,
        ) as run:
            verifier = RepositoryCandidateVerifier(Path(temp))
            result = verifier.verify({}, proposal, {})
            verifier.verify({}, proposal, {})

        self.assertTrue(result["deterministic_gates"])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            run.call_args.args[0][:5],
            [run.call_args.args[0][0], "-m", "unittest", "discover", "-s"],
        )
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_review_executor_creates_local_branch_and_commit_without_merging(self) -> None:
        with TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            repo.mkdir()
            (repo / "src").mkdir()
            (repo / "src" / "calc.py").write_text("def value():\n    return 1\n", encoding="utf-8")
            (repo / "tests").mkdir()
            (repo / "tests" / "test_calc.py").write_text(
                "import unittest\nfrom calc import value\n\n"
                "class CalcTest(unittest.TestCase):\n"
                "    def test_value(self):\n        self.assertEqual(value(), 1)\n",
                encoding="utf-8",
            )
            (repo / ".gitignore").write_text("workspace/\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run([
                "git", "-c", "user.name=Test", "-c", "user.email=test@localhost",
                "commit", "-qm", "initial",
            ], cwd=repo, check=True)
            initial_branch = subprocess.run(
                ["git", "branch", "--show-current"], cwd=repo,
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            diff = (
                "diff --git a/src/calc.py b/src/calc.py\n"
                "--- a/src/calc.py\n+++ b/src/calc.py\n"
                "@@ -1,2 +1,2 @@\n def value():\n-    return 1\n+    return 2\n"
                "diff --git a/tests/test_calc.py b/tests/test_calc.py\n"
                "--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n"
                "@@ -3,4 +3,4 @@ from calc import value\n \n class CalcTest(unittest.TestCase):\n"
                "     def test_value(self):\n-        self.assertEqual(value(), 1)\n"
                "+        self.assertEqual(value(), 2)\n"
            )
            plan = {
                "problem_id": "problem-demo", "branch": "agent-fix/problem-demo",
                "diff": diff,
                "regression_ref": "tests/test_calc.py::CalcTest::test_value",
            }

            result = ReviewBranchExecutor(repo, repo / "workspace").stage(plan)

            self.assertTrue(result["tests_passed"])
            self.assertFalse(result["merge"])
            self.assertFalse(result["push"])
            self.assertEqual(
                subprocess.run(["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip(),
                initial_branch,
            )
            self.assertEqual(len(result["commit"]), 40)

    def test_invalid_regression_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "regression"):
            ReviewBranchExecutor._unittest_name("tests/test_x.py")


if __name__ == "__main__":
    unittest.main()
