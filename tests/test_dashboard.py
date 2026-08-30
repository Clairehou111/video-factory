import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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

    def check_account(self, target):
        return BackendResult(["check"], 0, "valid", started=True)

    def submit_collection_video(self, target, video_path):
        self.uploaded.append((target.platform, video_path.name))
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
            self.assertEqual(list(media.values())[0].name, "wechat.mp4")

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


if __name__ == "__main__":
    unittest.main()
