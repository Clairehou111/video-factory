from __future__ import annotations

import unittest
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from video_factory.automation import (
    AutomationAuditService, AutomationPolicy, DiscoveryPublishBridge,
    PipelinePublishConfig, PreparedPublishBatch, pipeline_lock,
    is_pipeline_lock_collision,
)
from video_factory.discovery import (
    ChannelConfig, ChannelRun, DiscoveryCandidate, DiscoveryChannel,
    ResourceDiscoveryRun, evaluate_candidate,
)
from video_factory.models import TopicType
from video_factory.publish import PublishBatchState, PublishPlatform


NOW = datetime(2026, 8, 30, 1, 0, tzinfo=UTC)


class FakeWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.publish_dir = root / "publish"
        self.publish_dir.mkdir(parents=True)
        self.saved = []

    def save_publish_batch(self, batch) -> None:
        self.saved.append(batch)

    def load_discovery_state(self):
        return {}


class AutomationTest(unittest.TestCase):
    def test_pipeline_lock_rejects_concurrent_run(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with pipeline_lock(root):
                with self.assertRaisesRegex(RuntimeError, "another pipeline run"):
                    with pipeline_lock(root):
                        pass

    def test_pipeline_lock_collision_is_a_scheduler_skip_not_a_crash(self) -> None:
        self.assertTrue(is_pipeline_lock_collision(
            RuntimeError("another pipeline run already holds /tmp/pipeline.lock"),
        ))
        self.assertFalse(is_pipeline_lock_collision(RuntimeError("generation failed")))

    def test_run_audit_records_trial_repairs_cost_and_human_gate(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = FakeWorkspace(root)
            job = root / "jobs" / "job-1"
            job.mkdir(parents=True)
            (job / "result.json").write_text(json.dumps({
                "job_id": "job-1", "started_at": NOW.isoformat(), "status": "completed",
                "source_type": "youtube",
                "model_selection": {"provider": "openrouter", "quote": {
                    "model_id": "vendor/cheap", "prompt_price": 0.000001,
                    "completion_price": 0.000002,
                }},
                "translation_trace": [{
                    "step": "editorial_plan_repair", "status": "ok",
                    "provenance": {
                        "provider": "openrouter", "model": "vendor/cheap",
                        "generated_at": NOW.isoformat(),
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    },
                }],
            }), encoding="utf-8")
            candidate = DiscoveryCandidate(
                id="youtube-1", channel=DiscoveryChannel.YOUTUBE,
                url="https://youtube.com/watch?v=1", title="Technical talk",
            )
            run = ResourceDiscoveryRun("run-1", "completed", NOW.isoformat(), completed_at=NOW.isoformat(), channels={
                "youtube": ChannelRun(
                    DiscoveryChannel.YOUTUBE, "generated", selected=candidate,
                    adoption={"status": "generated", "attempts": [
                        {"attempt": 1, "status": "failed", "error": "render failed"},
                        {"attempt": 2, "status": "generated", "result": {}},
                    ]},
                ),
            })
            prepared = [PreparedPublishBatch(
                "youtube", candidate.id, "collection-1", "batch-1",
                PublishBatchState.READY_FOR_REVIEW.value, ["tencent", "bilibili"], True,
            )]
            audit = AutomationAuditService(
                workspace, AutomationPolicy(7), clock=lambda: NOW,
            ).record(run, prepared, "auto", None)

            self.assertTrue(audit["trial"]["active"])
            self.assertFalse(audit["publication_gate"]["automatic_publish"])
            self.assertEqual(audit["publication_gate"]["ready_batch_ids"], ["batch-1"])
            self.assertEqual(audit["llm_usage"]["total_tokens"], 150)
            self.assertAlmostEqual(audit["llm_usage"]["estimated_cost_usd"], 0.0002)
            self.assertAlmostEqual(audit["llm_usage"]["accounted_cost_usd"], 0.0002)
            self.assertEqual(audit["problems"][0]["status"], "resolved_automatically")
            self.assertTrue((root / "automation" / "latest.md").is_file())

    def test_notification_never_changes_publish_state(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = FakeWorkspace(Path(temp))
            report = {
                "id": "run-1", "publication_gate": {"ready_batch_ids": ["batch-1"]},
                "problems": [], "pipeline_status": "completed",
                "trial": {"day": 1, "active": True},
                "llm_usage": {"total_tokens": 0, "accounted_cost_usd": 0.0, "unpriced_calls": 0},
                "human_message": "review", "automatic_fixes": [], "human_action_required": True,
            }
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")
            with patch("video_factory.automation.AutomationAuditService._write_report") as write:
                status = AutomationAuditService(workspace).notify(report, runner=lambda *args, **kwargs: completed)
            self.assertTrue(status["delivered"])
            self.assertNotIn("approved", report["publication_gate"])
            write.assert_called_once()

    def test_duplicate_notification_is_suppressed_during_reminder_cooldown(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = FakeWorkspace(Path(temp))
            report = {
                "id": "run-1", "publication_gate": {"ready_batch_ids": ["batch-1"]},
                "problems": [], "needs_human_candidates": [], "human_action_required": True,
                "human_message": "review",
            }
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")
            service = AutomationAuditService(workspace, clock=lambda: NOW)
            with patch("video_factory.automation.AutomationAuditService._write_report"):
                first = service.notify(report, runner=lambda *args, **kwargs: completed)
                second = service.notify(report, runner=lambda *args, **kwargs: completed)
            self.assertTrue(first["delivered"])
            self.assertFalse(second["attempted"])
            self.assertEqual(second["skipped_reason"], "duplicate_reminder_cooldown")

    def test_non_youtube_generation_routes_only_to_tencent(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = FakeWorkspace(Path(temp))
            candidate = DiscoveryCandidate(
                id="official-1", channel=DiscoveryChannel.OFFICIAL,
                url="https://example.com/model", title="Model launch",
                publisher="Example", topic_type=TopicType.MODEL_OR_PRODUCT,
            )
            run = ResourceDiscoveryRun("run-1", "completed", NOW.isoformat(), channels={
                "official": ChannelRun(
                    DiscoveryChannel.OFFICIAL, "generated", selected=candidate,
                    adoption={"status": "generated", "result": {"manifest": "/tmp/manifest.json"}},
                ),
            })
            manifest = SimpleNamespace(
                id="manifest-1", fixed_title="新模型真正改变了什么",
                fixed_hook="", topic_type=TopicType.MODEL_OR_PRODUCT,
            )
            created = SimpleNamespace(
                id="publish-1", state=PublishBatchState.READY_FOR_REVIEW,
            )
            with patch("video_factory.automation.load_manifest", return_value=manifest), patch(
                "video_factory.automation.create_publish_batch", return_value=created,
            ) as create:
                prepared = DiscoveryPublishBridge(workspace, PipelinePublishConfig()).prepare(run)

            targets = create.call_args.args[1]
            self.assertEqual([target.platform for target in targets], [PublishPlatform.TENCENT])
            self.assertEqual(prepared[0].platforms, ["tencent"])
            self.assertTrue(prepared[0].created)

    def test_technical_youtube_generation_routes_only_to_wechat_while_bilibili_paused(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = FakeWorkspace(Path(temp))
            candidate = DiscoveryCandidate(
                id="youtube-1", channel=DiscoveryChannel.YOUTUBE,
                url="https://youtube.com/watch?v=1", title="Software architecture talk",
                publisher="AI Engineer", metadata={"technical_share": True},
            )
            run = ResourceDiscoveryRun("run-1", "completed", NOW.isoformat(), channels={
                "youtube": ChannelRun(
                    DiscoveryChannel.YOUTUBE, "generated", selected=candidate,
                    adoption={"status": "generated", "result": {"collection_manifest": "/tmp/collection.json"}},
                ),
            })
            manifest = SimpleNamespace(
                id="collection-1", collection_title="AI 工程实践",
                source_channel="AI Engineer", source_url=candidate.url,
            )
            created = SimpleNamespace(
                id="publish-collection-1", state=PublishBatchState.READY_FOR_REVIEW,
            )
            with patch("video_factory.automation.load_collection_manifest", return_value=manifest), patch(
                "video_factory.automation.create_collection_publish_batch", return_value=created,
            ) as create:
                prepared = DiscoveryPublishBridge(workspace, PipelinePublishConfig()).prepare(run)

            spec = create.call_args.args[1]
            self.assertEqual(set(spec["targets"]), {"tencent"})
            self.assertEqual(prepared[0].platforms, ["tencent"])

    def test_nontechnical_youtube_never_creates_a_publish_batch(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = FakeWorkspace(Path(temp))
            candidate = DiscoveryCandidate(
                id="youtube-2", channel=DiscoveryChannel.YOUTUBE,
                url="https://youtube.com/watch?v=2", title="Founder lifestyle interview",
                publisher="Interview Channel", metadata={"technical_share": False},
            )
            run = ResourceDiscoveryRun("run-2", "completed", NOW.isoformat(), channels={
                "youtube": ChannelRun(
                    DiscoveryChannel.YOUTUBE, "generated", selected=candidate,
                    adoption={"status": "generated", "result": {"collection_manifest": "/tmp/collection.json"}},
                ),
            })
            manifest = SimpleNamespace(
                id="collection-2", collection_title="Founder interview",
                source_channel="Interview Channel", source_url=candidate.url,
            )
            with patch("video_factory.automation.load_collection_manifest", return_value=manifest), patch(
                "video_factory.automation.create_collection_publish_batch",
            ) as create:
                prepared = DiscoveryPublishBridge(workspace, PipelinePublishConfig()).prepare(run)

            create.assert_not_called()
            self.assertFalse(prepared[0].created)
            self.assertIn("neither technical coverage nor a known-tech interview", prepared[0].reason)

    def test_unified_youtube_gate_rejects_nontechnical_interview(self) -> None:
        body = (
            "A founder discusses fundraising, hiring, lifestyle, and the company journey in a long interview. "
            "The conversation covers personal routines, leadership stories, and career decisions. "
            "It contains enough narrative material for a profile but no implementation details."
        )
        candidate = DiscoveryCandidate(
            id="youtube-profile", channel=DiscoveryChannel.YOUTUBE,
            url="https://youtube.com/watch?v=profile", title="Founder journey interview",
            publisher="Interview Channel", published_at=NOW.isoformat(), summary=body, body_text=body,
            metadata={"duration_seconds": 1800, "transcript_available": True, "technical_share": False},
        )

        evaluate_candidate(candidate, ChannelConfig.from_dict(DiscoveryChannel.YOUTUBE, {}), NOW)

        self.assertFalse(candidate.eligible)
        self.assertIn("not_technical_share_or_known_tech_interview", candidate.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
