import hashlib
import threading
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.collection_publish import (
    CollectionPublishBatch, CollectionPublishItem, CollectionPublishItemState,
)
from video_factory.dashboard import PublishDashboard
from video_factory.publish import (
    BackendResult, PublishBatch, PublishBatchState, PublishPlatform, PublishTarget,
    SOCIAL_AUTO_UPLOAD_COMMIT,
)
from video_factory.storage import Workspace


class FakeDashboardBackend:
    commit = SOCIAL_AUTO_UPLOAD_COMMIT

    def __init__(self) -> None:
        self.uploaded = []
        self.schedules = []
        self.auth_valid = True
        self.login_calls = 0

    def check_account(self, target):
        if not self.auth_valid:
            return BackendResult(
                ["check"], 1, "cookie 已失效（页面跳转到登录页）",
                started=True,
            )
        return BackendResult(["check"], 0, "valid", started=True)

    def login_account(self, platform, account_name, headless=False):
        self.login_calls += 1
        self.auth_valid = True
        return BackendResult(["login"], 0, "login completed", started=True)

    def submit_collection_video(self, target, video_path):
        self.uploaded.append((target.platform, video_path.name))
        self.schedules.append(target.schedule_at)
        return BackendResult(["upload"], 0, '{"video_id":"wechat-1"}', started=True)

    def ensure_bilibili_collection(self, account_name, title):
        raise AssertionError("Bilibili must never be called by the dashboard")

    def add_bilibili_collection(self, account_name, collection_id, bvid, position):
        raise AssertionError("Bilibili must never be called by the dashboard")


class DashboardTest(unittest.TestCase):
    def make_batch(self, root: Path):
        workspace = Workspace(root / "workspace")
        workspace.initialize()
        wechat = root / "wechat.mp4"
        bilibili = root / "bilibili.mp4"
        wechat.write_bytes(b"wechat-video")
        bilibili.write_bytes(b"bilibili-video")

        def item(identifier, platform, path):
            return CollectionPublishItem(
                id=identifier, collection_item_id=identifier,
                platform=platform, account_name="main", collection_title="AI 高光",
                order=1, video_path=str(path),
                video_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                title="AI 写得越快，人越要会验", description="来源：访谈",
                tags=["AI"], options={"collection": "AI 高光"}
                if platform == PublishPlatform.TENCENT else {"tid": 231},
            )

        batch = CollectionPublishBatch(
            id="dashboard-batch", manifest_id="missing-manifest", collection_title="AI 高光",
            items=[
                item("wechat-item", PublishPlatform.TENCENT, wechat),
                item("bilibili-item", PublishPlatform.BILIBILI, bilibili),
            ],
            state=PublishBatchState.READY_FOR_REVIEW, checks=[],
        )
        workspace.save_publish_batch(batch)
        return workspace, batch

    def test_queue_hides_bilibili_and_exposes_wechat_preview(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, _ = self.make_batch(Path(temp))
            rows, media = PublishDashboard(workspace).queue()

            self.assertEqual([row["item_id"] for row in rows], ["wechat-item"])
            self.assertTrue(rows[0]["can_publish"])
            self.assertEqual(rows[0]["sequence"], 1)
            self.assertEqual(rows[0]["sequence_total"], 1)
            self.assertEqual(rows[0]["sequence_label"], "01/01")
            self.assertEqual(rows[0]["display_title"], "01/01 · AI 写得越快，人越要会验")
            self.assertEqual(list(media.values())[0].name, "wechat.mp4")

    def test_queue_exposes_collection_sequence_without_changing_publish_title(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, batch = self.make_batch(root)
            second_video = root / "wechat-second.mp4"
            second_video.write_bytes(b"wechat-video-second")
            second = CollectionPublishItem(
                id="wechat-item-2", collection_item_id="wechat-item-2",
                platform=PublishPlatform.TENCENT, account_name="main",
                collection_title="AI 高光", order=2, video_path=str(second_video),
                video_sha256=hashlib.sha256(second_video.read_bytes()).hexdigest(),
                title="第二条原始发布标题", description="来源：访谈", tags=["AI"],
                options={"collection": "AI 高光"},
            )
            batch.items.append(second)
            workspace.save_publish_batch(batch)

            rows, _ = PublishDashboard(workspace).queue()

            self.assertEqual(
                [
                    (row["title"], row["display_title"], row["sequence_label"])
                    for row in rows
                ],
                [
                    ("AI 写得越快，人越要会验", "01/02 · AI 写得越快，人越要会验", "01/02"),
                    ("第二条原始发布标题", "02/02 · 第二条原始发布标题", "02/02"),
                ],
            )

            refreshed_rows, _ = PublishDashboard(workspace).queue()
            self.assertEqual(
                [row["display_title"] for row in refreshed_rows],
                ["01/02 · AI 写得越快，人越要会验", "02/02 · 第二条原始发布标题"],
            )

    def test_queue_shows_only_latest_rerender_for_same_youtube_source(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(Path(temp))
            batch.id = "dashboard-batch-new"
            batch.manifest_id = "rerendered-manifest"
            batch.created_at = "9999-01-01T00:00:00Z"
            workspace.save_publish_batch(batch)

            with patch.object(
                workspace, "load_collection_manifest",
                return_value=SimpleNamespace(
                    source_video_id="same-video", source_url="https://youtube.test/watch",
                    source_title="Interview", editorial_mode="known_tech_interview_clip",
                ),
            ):
                rows, _ = PublishDashboard(workspace).queue()

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["batch_id"], "dashboard-batch-new")

    def test_queue_keeps_distinct_clips_from_same_youtube_source(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(Path(temp))
            batch.id = "dashboard-batch-second-clip"
            batch.manifest_id = "second-clip-manifest"
            batch.created_at = "9999-01-01T00:00:00Z"
            workspace.save_publish_batch(batch)

            def manifest_for(identifier):
                original_start = 10.0 if identifier == "missing-manifest" else 90.0
                return SimpleNamespace(
                    source_video_id="same-video",
                    source_url="https://youtube.test/watch",
                    source_title="Interview",
                    editorial_mode="known_tech_interview_clip",
                    items=[SimpleNamespace(source_ranges=[SimpleNamespace(
                        original_start=original_start,
                        original_end=original_start + 45.0,
                    )])],
                )

            with patch.object(
                workspace, "load_collection_manifest", side_effect=manifest_for,
            ):
                rows, _ = PublishDashboard(workspace).queue()

            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {row["batch_id"] for row in rows},
                {"dashboard-batch", "dashboard-batch-second-clip"},
            )

    def test_publish_button_approves_and_submits_only_selected_wechat_item(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(Path(temp))
            backend = FakeDashboardBackend()
            dashboard = PublishDashboard(workspace, actor="claire", backend_factory=lambda: backend)

            result = dashboard.publish(batch.id, "wechat-item")

            self.assertTrue(result["published"])
            self.assertEqual(backend.uploaded, [(PublishPlatform.TENCENT, "wechat.mp4")])
            restored = workspace.load_publish_batch(batch.id)
            self.assertEqual(restored.items[0].state, CollectionPublishItemState.SUBMITTED)
            self.assertEqual(restored.items[1].state, CollectionPublishItemState.PENDING)

    def test_publish_can_set_audited_beijing_schedule_before_approval(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(Path(temp))
            backend = FakeDashboardBackend()
            dashboard = PublishDashboard(workspace, actor="claire", backend_factory=lambda: backend)

            result = dashboard.publish(
                batch.id, "wechat-item", "2099-09-16 20:30", update_schedule=True,
            )

            self.assertTrue(result["published"])
            self.assertEqual(backend.schedules, ["2099-09-16 20:30"])
            restored = workspace.load_publish_batch(batch.id)
            self.assertEqual(restored.items[0].schedule_at, "2099-09-16 20:30")
            self.assertEqual(
                restored.approval_payload()["items"][0]["schedule_at"],
                "2099-09-16 20:30",
            )

    def test_collection_retry_allows_unchanged_empty_schedule(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(Path(temp))
            for item in batch.items:
                item.options = item.as_publish_target().options
            batch.approve("claire")
            batch.state = PublishBatchState.FAILED
            batch.items[0].state = CollectionPublishItemState.FAILED_PRE_SUBMIT
            workspace.save_publish_batch(batch)
            backend = FakeDashboardBackend()
            dashboard = PublishDashboard(
                workspace, actor="claire", backend_factory=lambda: backend,
            )

            result = dashboard.publish(
                batch.id, "wechat-item", None, update_schedule=True,
            )

            self.assertTrue(result["published"])
            self.assertEqual(backend.uploaded, [(PublishPlatform.TENCENT, "wechat.mp4")])

    def test_collection_retry_still_rejects_schedule_change(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(Path(temp))
            for item in batch.items:
                item.options = item.as_publish_target().options
            batch.approve("claire")
            batch.state = PublishBatchState.FAILED
            batch.items[0].state = CollectionPublishItemState.FAILED_PRE_SUBMIT
            workspace.save_publish_batch(batch)
            dashboard = PublishDashboard(workspace)

            with self.assertRaisesRegex(ValueError, "最终审核前"):
                dashboard.publish(
                    batch.id, "wechat-item", "2099-09-16 20:30",
                    update_schedule=True,
                )

    def test_publish_rejects_schedule_without_two_hour_lead_time(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(Path(temp))
            dashboard = PublishDashboard(workspace)

            with self.assertRaisesRegex(ValueError, "至少需要提前 2 小时"):
                dashboard.publish(
                    batch.id, "wechat-item", "2020-01-01 20:30",
                    update_schedule=True,
                )

    def test_withdrawn_batch_is_kept_for_audit_but_hidden_from_queue(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = Workspace(root / "workspace")
            workspace.initialize()
            video = root / "withdrawn.mp4"
            video.write_bytes(b"withdrawn-video")
            batch = PublishBatch(
                id="withdrawn-batch", manifest_id="missing-withdrawn-manifest",
                video_path=str(video),
                video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                targets=[PublishTarget(PublishPlatform.TENCENT, "main", "广告感太强")],
                state=PublishBatchState.READY_FOR_REVIEW,
            )
            batch.withdraw_from_queue("editorial review: feels like an advertisement")
            workspace.save_publish_batch(batch)

            rows, _ = PublishDashboard(workspace).queue()

            self.assertEqual(rows, [])
            restored = workspace.load_publish_batch(batch.id)
            self.assertTrue(restored.queue_hidden)
            self.assertIn("advertisement", restored.queue_hidden_reason)

    def test_sensitive_video_requires_separate_review_before_publish(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = Workspace(root / "workspace")
            workspace.initialize()
            video = root / "safety.mp4"
            video.write_bytes(b"safety-video")
            batch = PublishBatch(
                id="safety-batch", manifest_id="missing-safety-manifest",
                video_path=str(video),
                video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                targets=[PublishTarget(
                    PublishPlatform.TENCENT, "main", "AI 安全研究",
                    options={"collection": "AI 前沿动态"},
                )],
                state=PublishBatchState.BLOCKED,
                checks=[{
                    "name": "editorial_safety_review", "passed": False,
                    "detail": "Sensitive security terms found: jailbreak",
                }],
            )
            workspace.save_publish_batch(batch)
            dashboard = PublishDashboard(workspace, actor="claire")

            rows, _ = dashboard.queue()
            self.assertTrue(rows[0]["can_review"])
            self.assertFalse(rows[0]["can_publish"])

            result = dashboard.review(batch.id, "tencent")

            self.assertTrue(result["reviewed"])
            restored = workspace.load_publish_batch(batch.id)
            self.assertEqual(restored.state, PublishBatchState.READY_FOR_REVIEW)
            self.assertIn("editorial_safety_review", restored.review_overrides)
            rows, _ = dashboard.queue()
            self.assertFalse(rows[0]["can_review"])
            self.assertTrue(rows[0]["can_publish"])

    def test_collection_rights_review_is_explicit_and_bound_into_approval(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(Path(temp))
            batch.state = PublishBatchState.BLOCKED
            batch.checks = [{
                "name": "rights_review", "passed": False,
                "detail": "human reuse-basis review required before publication",
            }]
            workspace.save_publish_batch(batch)
            dashboard = PublishDashboard(workspace, actor="claire")

            rows, _ = dashboard.queue()
            self.assertTrue(rows[0]["can_review"])
            self.assertEqual(rows[0]["review_check"], "rights_review")
            self.assertEqual(rows[0]["action_label"], "确认复用依据已审核")

            result = dashboard.review(batch.id, "wechat-item")

            self.assertTrue(result["reviewed"])
            restored = workspace.load_publish_batch(batch.id)
            self.assertEqual(restored.state, PublishBatchState.READY_FOR_REVIEW)
            self.assertIn("rights_review", restored.review_overrides)
            self.assertIn("review_overrides", restored.approval_payload())

    def test_expired_login_becomes_bounded_login_and_publish_recovery(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(Path(temp))
            backend = FakeDashboardBackend()
            backend.auth_valid = False
            dashboard = PublishDashboard(
                workspace, actor="claire", backend_factory=lambda: backend,
            )

            first = dashboard.publish(batch.id, "wechat-item")
            self.assertFalse(first["published"])
            self.assertTrue(first["requires_login"])
            rows, _ = dashboard.queue()
            self.assertTrue(rows[0]["requires_login"])
            self.assertEqual(rows[0]["action_label"], "登录视频号并继续发布")

            recovered = dashboard.login_and_publish(batch.id, "wechat-item")

            self.assertTrue(recovered["published"])
            self.assertTrue(recovered["login_recovered"])
            self.assertEqual(backend.login_calls, 1)
            self.assertEqual(backend.uploaded, [(PublishPlatform.TENCENT, "wechat.mp4")])
            attempts = list((workspace.publish_dir / batch.id / "attempts").glob("*.json"))
            self.assertTrue(any("login_recovery" in path.name for path in attempts))

    def test_discovery_failures_show_only_active_candidates(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            active = {
                "id": "youtube-active", "channel": "youtube",
                "url": "https://youtube.com/watch?v=active", "title": "Active failure",
                "eligible": True, "status": "needs_human", "metadata": {},
            }
            resolved = {
                "id": "youtube-resolved", "channel": "youtube",
                "url": "https://youtube.com/watch?v=resolved", "title": "Resolved failure",
                "eligible": True, "status": "generated", "metadata": {},
            }
            workspace.save_discovery_candidate(active)
            workspace.save_discovery_candidate(resolved)
            workspace.save_discovery_state({
                "channels": {}, "generated_events": [], "history": [], "skipped_ids": [],
                "needs_human_candidates": [
                    {"candidate_id": "youtube-active", "title": "Active failure", "status": "needs_human"},
                    {"candidate_id": "youtube-resolved", "title": "Resolved failure", "status": "needs_human"},
                ],
            })

            rows = PublishDashboard(workspace).discovery_failures()

            self.assertEqual([row["candidate_id"] for row in rows], ["youtube-active"])

    def test_discovery_retries_are_serial_background_tasks_with_visible_logs(self) -> None:
        class SerialDiscoveryService:
            def __init__(self) -> None:
                self.calls: list[str] = []
                self.first_started = threading.Event()
                self.release_first = threading.Event()
                self.finished = threading.Event()

            def adopt_candidate(self, candidate_id, config):
                self.calls.append(candidate_id)
                if candidate_id == "youtube-one":
                    self.first_started.set()
                    self.release_first.wait(timeout=5)
                if candidate_id == "youtube-two":
                    self.finished.set()
                return {
                    "status": "blocked", "last_error": f"validation failed for {candidate_id}",
                    "attempts": [{"status": "failed"}],
                }

        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            rows = []
            for candidate_id in ("youtube-one", "youtube-two"):
                workspace.save_discovery_candidate({
                    "id": candidate_id, "channel": "youtube",
                    "url": f"https://youtube.com/watch?v={candidate_id}",
                    "title": candidate_id, "eligible": True,
                    "status": "needs_human", "metadata": {},
                })
                rows.append({
                    "candidate_id": candidate_id, "channel": "youtube",
                    "title": candidate_id, "status": "needs_human",
                })
            workspace.save_discovery_state({
                "channels": {}, "generated_events": [], "history": [],
                "skipped_ids": [], "needs_human_candidates": rows,
            })
            service = SerialDiscoveryService()
            dashboard = PublishDashboard(workspace, discovery_service=service)

            first = dashboard.retry_discovery("youtube-one")
            second = dashboard.retry_discovery("youtube-two")
            self.assertEqual(first["status"], "queued")
            self.assertEqual(second["status"], "queued")
            self.assertTrue(service.first_started.wait(timeout=2))
            self.assertEqual(service.calls, ["youtube-one"])

            visible = {row["candidate_id"]: row for row in dashboard.discovery_failures()}
            self.assertEqual(visible["youtube-one"]["status"], "running")
            self.assertEqual(visible["youtube-two"]["status"], "queued")
            self.assertIn("Generation started", str(visible["youtube-one"]["retry_task"]["logs"]))

            service.release_first.set()
            self.assertTrue(service.finished.wait(timeout=2))
            deadline = time.time() + 2
            while time.time() < deadline:
                visible = {row["candidate_id"]: row for row in dashboard.discovery_failures()}
                if visible["youtube-two"]["retry_task"]["status"] == "failed":
                    break
                time.sleep(0.01)

            self.assertEqual(service.calls, ["youtube-one", "youtube-two"])
            self.assertEqual(visible["youtube-one"]["retry_task"]["status"], "failed")
            self.assertEqual(visible["youtube-two"]["retry_task"]["status"], "failed")
            self.assertIn("validation failed", visible["youtube-two"]["last_error"])
            self.assertTrue(list((workspace.root / "discovery" / "retry-tasks").glob("*.json")))

    def test_successful_discovery_retry_remains_visible_with_terminal_status(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            candidate = {
                "id": "youtube-success", "channel": "youtube",
                "url": "https://youtube.com/watch?v=success",
                "title": "Successful retry", "eligible": True,
                "status": "needs_human", "metadata": {},
            }
            workspace.save_discovery_candidate(candidate)
            workspace.save_discovery_state({
                "channels": {}, "generated_events": [], "history": [],
                "skipped_ids": [], "needs_human_candidates": [{
                    "candidate_id": candidate["id"], "channel": "youtube",
                    "title": candidate["title"], "status": "needs_human",
                }],
            })

            class SuccessfulDiscoveryService:
                def adopt_candidate(self, candidate_id, config):
                    generated = workspace.load_discovery_candidate(candidate_id)
                    generated["status"] = "generated"
                    workspace.save_discovery_candidate(generated)
                    return {"status": "generated", "attempts": [{"status": "generated"}]}

            dashboard = PublishDashboard(
                workspace, discovery_service=SuccessfulDiscoveryService(),
            )
            dashboard.retry_discovery(candidate["id"])
            deadline = time.time() + 2
            while time.time() < deadline:
                rows = dashboard.discovery_failures()
                if rows and rows[0].get("retry_task", {}).get("status") == "succeeded":
                    break
                time.sleep(0.01)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "succeeded")
            self.assertFalse(rows[0]["retry_eligible"])
            self.assertIn("Generation completed", str(rows[0]["retry_task"]["logs"]))

    def test_dashboard_restart_recovers_queued_and_running_retries(self) -> None:
        class RecoveredDiscoveryService:
            def __init__(self) -> None:
                self.calls: list[str] = []
                self.finished = threading.Event()

            def adopt_candidate(self, candidate_id, config):
                self.calls.append(candidate_id)
                if len(self.calls) == 2:
                    self.finished.set()
                return {
                    "status": "blocked", "last_error": "still needs repair",
                    "attempts": [{"status": "failed"}],
                }

        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            old = PublishDashboard(workspace)
            state_rows = []
            for index, status in enumerate(("running", "queued"), start=1):
                candidate_id = f"youtube-recover-{index}"
                workspace.save_discovery_candidate({
                    "id": candidate_id, "channel": "youtube",
                    "url": f"https://youtube.com/watch?v=recover-{index}",
                    "title": candidate_id, "eligible": True,
                    "status": "needs_human", "metadata": {},
                })
                state_rows.append({
                    "candidate_id": candidate_id, "channel": "youtube",
                    "title": candidate_id, "status": "needs_human",
                })
                old._save_retry_task({
                    "candidate_id": candidate_id, "status": status,
                    "stage": "generation" if status == "running" else "queued",
                    "created_at": old._now_iso(), "updated_at": old._now_iso(),
                    "instance": "dead-dashboard", "logs": [],
                })
                time.sleep(0.001)
            workspace.save_discovery_state({
                "channels": {}, "generated_events": [], "history": [],
                "skipped_ids": [], "needs_human_candidates": state_rows,
            })

            service = RecoveredDiscoveryService()
            dashboard = PublishDashboard(workspace, discovery_service=service)
            self.assertTrue(service.finished.wait(timeout=2))
            deadline = time.time() + 2
            while time.time() < deadline:
                rows = dashboard.discovery_failures()
                if len(rows) == 2 and all(
                    row.get("retry_task", {}).get("status") == "failed"
                    for row in rows
                ):
                    break
                time.sleep(0.01)

            self.assertEqual(service.calls, ["youtube-recover-1", "youtube-recover-2"])
            self.assertTrue(all(
                row["retry_task"]["recovery_count"] == 1 for row in rows
            ))
            self.assertTrue(all(
                "recovered after Dashboard restart" in str(row["retry_task"]["logs"])
                for row in rows
            ))


if __name__ == "__main__":
    unittest.main()
