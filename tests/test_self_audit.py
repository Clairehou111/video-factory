from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from video_factory.self_audit import (
    PolicyStore, ProblemLedger, ProblemObservation, SelfAuditService,
    load_active_policy,
)


NOW = datetime(2026, 8, 31, 18, 15, tzinfo=UTC)
PASSING_VERIFICATION = {
    "reproduction_fixed": True,
    "deterministic_gates": True,
    "grounding": True,
    "safety": True,
    "golden_regressions": True,
    "confidence": 0.8,
}
SAFE_POLICY = {
    "runtime": {
        "narrative_guidance": "Lead with the audience-recognizable event, then prove it with archived evidence.",
        "evaluator_thresholds": {"story_axis": 0.8, "grounding": 1.0},
    },
    "director": {"narrative_rules": ["preserve the main event"]},
}


class Clock:
    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class FakeModel:
    def __init__(self, proposal=None, estimate: float = 0.1) -> None:
        self.proposal = proposal or {"kind": "diagnosis", "rationale": "inspect story axis", "cost_usd": estimate}
        self.estimate = estimate
        self.calls = []

    def estimate_cost(self, problem):
        return self.estimate

    def propose(self, problem, archive, budget_remaining_usd):
        self.calls.append((problem, archive, budget_remaining_usd))
        return dict(self.proposal)


class FakeVerifier:
    def __init__(self, result=None) -> None:
        self.result = result or PASSING_VERIFICATION

    def verify(self, problem, proposal, archive):
        return dict(self.result)


def observation(category: str = "story", **overrides) -> ProblemObservation:
    values = {
        "stage": "generation", "category": category,
        "expected": "The main company event remains the story axis",
        "observed": "A secondary technical detail replaced the main event",
        "severity": "high",
    }
    values.update(overrides)
    return ProblemObservation(**values)


class ProblemLedgerTest(unittest.TestCase):
    def test_record_deduplicates_and_preserves_occurrence_count(self) -> None:
        with TemporaryDirectory() as temp:
            ledger = ProblemLedger(Path(temp), Clock())
            first = ledger.record(observation(artifact_refs=["jobs/a/manifest.json"]))
            second = ledger.record(observation(artifact_refs=["jobs/b/final.mp4"]))

            self.assertEqual(first["id"], second["id"])
            self.assertEqual(second["occurrence_count"], 2)
            self.assertEqual(
                second["artifact_refs"],
                ["jobs/a/manifest.json", "jobs/b/final.mp4"],
            )
            self.assertEqual(ledger.status()["counts"]["queued"], 1)
            self.assertEqual(len((ledger.root / "observations.jsonl").read_text().splitlines()), 2)

    def test_import_legacy_is_incremental_and_idempotent(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "automation" / "problems.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps({
                "scope": "render", "kind": "overflow", "detail": "translation clipped",
                "run_id": "run-1",
            }) + "\n", encoding="utf-8")
            ledger = ProblemLedger(root, Clock())

            self.assertEqual(ledger.import_legacy(), 1)
            self.assertEqual(ledger.import_legacy(), 0)
            self.assertEqual(ledger.status()["total"], 1)

    def test_selection_prioritizes_severity_frequency_then_age(self) -> None:
        with TemporaryDirectory() as temp:
            clock = Clock()
            ledger = ProblemLedger(Path(temp), clock)
            low = ledger.record(observation("low", severity="low"))
            critical = ledger.record(observation("critical", severity="critical"))
            ledger.record(observation("critical", severity="critical"))
            high = ledger.record(observation("high", severity="high"))

            selected = ledger.eligible(2)
            self.assertEqual([row["id"] for row in selected], [critical["id"], high["id"]])
            self.assertNotEqual(low["id"], selected[0]["id"])

    def test_failed_attempts_back_off_one_three_seven_then_block(self) -> None:
        with TemporaryDirectory() as temp:
            clock = Clock()
            ledger = ProblemLedger(Path(temp), clock)
            problem = ledger.record(observation())

            first = ledger.fail_attempt(problem["id"], "failed 1")
            self.assertEqual(first["state"], "backed_off")
            self.assertEqual(first["attempts"][-1]["backoff_days"], 1)
            self.assertEqual(ledger.eligible(5), [])
            clock.value += timedelta(days=1)
            second = ledger.fail_attempt(problem["id"], "failed 2")
            self.assertEqual(second["attempts"][-1]["backoff_days"], 3)
            clock.value += timedelta(days=3)
            third = ledger.fail_attempt(problem["id"], "failed 3")
            self.assertEqual(third["attempts"][-1]["backoff_days"], 7)
            self.assertEqual(third["state"], "human_blocked")
            clock.value += timedelta(days=8)
            self.assertEqual(ledger.eligible(5), [])

    def test_fixed_state_requires_regression_reference(self) -> None:
        with TemporaryDirectory() as temp:
            ledger = ProblemLedger(Path(temp), Clock())
            problem = ledger.record(observation())
            with self.assertRaisesRegex(ValueError, "regression"):
                ledger.transition(problem["id"], "fixed")

    def test_artifact_hash_drift_becomes_a_deduplicated_observation(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            job = root / "jobs" / "job-1"
            job.mkdir(parents=True)
            manifest = job / "manifest.json"
            manifest.write_text('{"title":"changed"}', encoding="utf-8")
            (job / "result.json").write_text(json.dumps({
                "job_id": "job-1", "manifest": str(manifest),
                "artifact_identity": {"manifest_sha256": "0" * 64},
            }), encoding="utf-8")
            ledger = ProblemLedger(root, Clock())

            self.assertEqual(ledger.scan_artifact_drift(), 1)
            self.assertEqual(ledger.scan_artifact_drift(), 1)

            status = ledger.status()
            self.assertEqual(status["total"], 1)
            self.assertEqual(status["problems"][0]["occurrence_count"], 2)


class PolicyStoreTest(unittest.TestCase):
    def test_rejects_unsafe_or_unbounded_runtime_policy(self) -> None:
        with TemporaryDirectory() as temp:
            store = PolicyStore(Path(temp), Clock())
            with self.assertRaisesRegex(ValueError, "unsafe policy sections"):
                store.validate({"publishing": {"automatic": True}})
            with self.assertRaisesRegex(ValueError, "4000"):
                store.validate({
                    "runtime": {
                        "narrative_guidance": "x" * 4_001,
                        "evaluator_thresholds": {"story_axis": 0.8},
                    },
                })
            with self.assertRaisesRegex(ValueError, "unsafe evaluator"):
                store.validate({
                    "runtime": {
                        "narrative_guidance": "clear story",
                        "evaluator_thresholds": {"automatic_publish": 1.0},
                    },
                })

    def test_promote_load_and_atomic_rollback(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            store = PolicyStore(root / "automation" / "self-audit", Clock())
            first = store.promote(
                SAFE_POLICY, problem_id="problem-a", verification=PASSING_VERIFICATION,
                version="policy-a",
            )
            second_policy = {
                **SAFE_POLICY,
                "runtime": {
                    **SAFE_POLICY["runtime"],
                    "narrative_guidance": "State the company-level conflict before supporting details.",
                },
            }
            store.promote(
                second_policy, problem_id="problem-b", verification=PASSING_VERIFICATION,
                version="policy-b",
            )
            self.assertEqual(load_active_policy(root)["version"], "policy-b")

            result = store.rollback("policy-a")

            self.assertEqual(result["previous_version"], "policy-b")
            loaded = load_active_policy(root)
            self.assertEqual(loaded["version"], "policy-a")
            self.assertEqual(loaded["digest"], first["digest"])
            self.assertIn("audience-recognizable", loaded["narrative_guidance"])


class SelfAuditServiceTest(unittest.TestCase):
    def test_run_caps_issue_count_and_cost(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = ProblemLedger(root, Clock())
            for index in range(7):
                ledger.record(observation(f"story-{index}"))
            model = FakeModel(estimate=0.26)
            service = SelfAuditService(root, model, clock=Clock(), max_issues=20, max_cost_usd=2)

            report = service.run()

            self.assertEqual(report["max_issues"], 5)
            self.assertEqual(report["max_cost_usd"], 1.0)
            self.assertEqual(len(model.calls), 3)
            self.assertAlmostEqual(report["spent_usd"], 0.78)
            self.assertEqual(len(report["skipped"]), 2)
            proposal = root / report["processed"][0]["proposal_artifact"]
            self.assertTrue(proposal.is_file())
            self.assertEqual(len(report["processed"][0]["proposal_sha256"]), 64)

    def test_external_or_escaping_artifacts_are_rejected_before_model_call(self) -> None:
        for unsafe_ref in ("https://example.com/live", "../outside.json"):
            with self.subTest(unsafe_ref=unsafe_ref), TemporaryDirectory() as temp:
                root = Path(temp)
                ledger = ProblemLedger(root, Clock())
                ledger.record(observation(artifact_refs=[unsafe_ref]))
                model = FakeModel()

                report = SelfAuditService(root, model, clock=Clock()).run()

                self.assertEqual(model.calls, [])
                self.assertEqual(report["processed"][0]["status"], "backed_off")

    def test_archived_reproduction_contains_hash_not_external_capabilities(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "jobs" / "one" / "manifest.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text('{"fixed_title": "Google loses another team"}', encoding="utf-8")
            ProblemLedger(root, Clock()).record(observation(artifact_refs=["jobs/one/manifest.json"]))
            model = FakeModel()

            SelfAuditService(root, model, clock=Clock()).run()

            archive = model.calls[0][1]
            self.assertEqual(archive["mode"], "archived_only")
            self.assertFalse(archive["external_actions_allowed"])
            self.assertEqual(len(archive["assets"][0]["sha256"]), 64)
            self.assertIn("fixed_title", archive["assets"][0]["text"])

    def test_policy_can_promote_only_after_verification_and_outside_shadow_mode(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ProblemLedger(root, Clock()).record(observation())
            model = FakeModel({
                "kind": "policy", "policy": SAFE_POLICY, "version": "candidate-a",
                "regression_ref": "tests/test_story.py::test_story_axis", "cost_usd": 0.1,
            })
            service = SelfAuditService(
                root, model, FakeVerifier(), clock=Clock(), shadow_only=False,
            )

            report = service.run()

            self.assertEqual(report["processed"][0]["status"], "fixed")
            self.assertEqual(load_active_policy(root)["version"], "candidate-a")
            self.assertEqual(ProblemLedger(root, Clock()).status()["counts"]["fixed"], 1)

    def test_code_proposal_creates_review_metadata_never_merge_or_push(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            problem = ProblemLedger(root, Clock()).record(observation())
            diff = (
                "diff --git a/src/video_factory/agent.py b/src/video_factory/agent.py\n"
                "--- a/src/video_factory/agent.py\n"
                "+++ b/src/video_factory/agent.py\n"
                "@@ -1 +1 @@\n-old\n+new\n"
                "diff --git a/tests/test_agent.py b/tests/test_agent.py\n"
                "--- a/tests/test_agent.py\n"
                "+++ b/tests/test_agent.py\n"
                "@@ -1 +1 @@\n-old test\n+new regression test\n"
            )
            model = FakeModel({
                "kind": "code", "diff": diff,
                "regression_ref": "tests/test_agent.py::test_story_axis", "cost_usd": 0.1,
            })

            report = SelfAuditService(root, model, clock=Clock(), shadow_only=False).run()

            result = report["processed"][0]
            self.assertEqual(result["status"], "review_required")
            self.assertEqual(result["branch"], f"agent-fix/{problem['id']}")
            plan = json.loads((
                root / "automation" / "self-audit" / "code-repairs" / f"{problem['id']}.json"
            ).read_text())
            self.assertFalse(plan["merge"])
            self.assertFalse(plan["push"])
            self.assertFalse((root / ".git").exists())

    def test_shadow_mode_validates_code_patch_without_creating_review_plan(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ProblemLedger(root, Clock()).record(observation())
            diff = (
                "diff --git a/src/video_factory/agent.py b/src/video_factory/agent.py\n"
                "--- a/src/video_factory/agent.py\n+++ b/src/video_factory/agent.py\n@@ -1 +1 @@\n-old\n+new\n"
                "diff --git a/tests/test_agent.py b/tests/test_agent.py\n"
                "--- a/tests/test_agent.py\n+++ b/tests/test_agent.py\n@@ -1 +1 @@\n-old\n+new test\n"
            )
            model = FakeModel({
                "kind": "code", "diff": diff,
                "regression_ref": "tests/test_agent.py::AgentTest::test_story_axis",
                "cost_usd": 0.1,
            })

            report = SelfAuditService(root, model, clock=Clock(), shadow_only=True).run()

            self.assertEqual(report["processed"][0]["reason"], "shadow_only")
            self.assertFalse((root / "automation" / "self-audit" / "code-repairs").exists())
            self.assertEqual(ProblemLedger(root, Clock()).eligible(5), [])

    def test_code_proposal_rejects_path_escape_and_push_commands(self) -> None:
        cases = [
            "diff --git a/x b/../secret\n--- a/x\n+++ b/../secret\n",
            "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n+git push origin main\n",
        ]
        for diff in cases:
            with self.subTest(diff=diff), TemporaryDirectory() as temp:
                root = Path(temp)
                ProblemLedger(root, Clock()).record(observation())
                model = FakeModel({"kind": "code", "diff": diff, "cost_usd": 0.1})
                report = SelfAuditService(root, model, clock=Clock()).run()
                self.assertEqual(report["processed"][0]["status"], "backed_off")


if __name__ == "__main__":
    unittest.main()
