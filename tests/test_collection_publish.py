import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from video_factory.collection_publish import (
    CollectionPublishBatch, CollectionPublishBatchService, CollectionPublishItem,
    CollectionPublishItemState,
)
from video_factory.publish import (
    BILIBILI_ORDINARY_UPLOAD_COLLECTION_ID, BackendResult, PublishBatchState,
    PublishPlatform, SOCIAL_AUTO_UPLOAD_COMMIT,
)
from video_factory.storage import Workspace


class FakeCollectionBackend:
    commit = SOCIAL_AUTO_UPLOAD_COMMIT

    def __init__(self) -> None:
        self.uploaded: list[str] = []
        self.added: list[str] = []
        self.fail_add_once = False

    def check_account(self, target):
        return BackendResult(["check"], 0, "valid", started=True)

    def ensure_bilibili_collection(self, account_name, title):
        return BackendResult(["ensure"], 0, json.dumps({"collection_id": "season-1"}), started=True)

    def submit_collection_video(self, target, video_path):
        self.uploaded.append(video_path.name)
        bvid = "BV1234567890" if len(self.uploaded) == 1 else "BV0987654321"
        return BackendResult(["upload"], 0, json.dumps({"bvid": bvid}), started=True)

    def add_bilibili_collection(self, account_name, collection_id, bvid, position):
        self.added.append(bvid)
        if self.fail_add_once:
            self.fail_add_once = False
            return BackendResult(["add"], 1, stderr="temporary", started=True)
        return BackendResult(["add"], 0, json.dumps({"ok": True}), started=True)


class FakeOrdinaryUploadBackend(FakeCollectionBackend):
    def ensure_bilibili_collection(self, account_name, title):
        return BackendResult(
            ["ensure"], 0,
            json.dumps({
                "collection_id": BILIBILI_ORDINARY_UPLOAD_COLLECTION_ID,
                "supported": False,
            }),
            started=True,
        )

    def submit_collection_video(self, target, video_path):
        self.uploaded.append(video_path.name)
        return BackendResult(["upload"], 0, "ordinary upload completed", started=True)


class FakeRejectedBilibiliBackend(FakeOrdinaryUploadBackend):
    def submit_collection_video(self, target, video_path):
        self.uploaded.append(video_path.name)
        return BackendResult(
            ["upload"], 1,
            stderr='ResponseData { code: -101, message: "账号未登录" }',
            started=True,
        )


class CollectionPublishTest(unittest.TestCase):
    def make_batch(self, temp: str, count: int = 2):
        root = Path(temp)
        workspace = Workspace(root / "workspace")
        workspace.initialize()
        items = []
        for index in range(count):
            video = root / f"video-{index}.mp4"
            video.write_bytes(f"video-{index}".encode())
            import hashlib
            digest = hashlib.sha256(video.read_bytes()).hexdigest()
            items.append(CollectionPublishItem(
                id=f"item-{index}", collection_item_id=f"collection-item-{index}",
                platform=PublishPlatform.BILIBILI, account_name="main",
                collection_title="AI 工程精选", order=index + 1,
                video_path=str(video), video_sha256=digest, title=f"标题 {index}",
                description="来源说明", tags=["AI"], options={"tid": 231},
            ))
        batch = CollectionPublishBatch(
            id="collection-batch", manifest_id="manifest", collection_title="AI 工程精选",
            items=items, state=PublishBatchState.READY_FOR_REVIEW, checks=[],
        )
        workspace.save_publish_batch(batch)
        return workspace, batch

    def make_tencent_batch(self, temp: str, count: int = 2):
        workspace, batch = self.make_batch(temp, count)
        for item in batch.items:
            item.platform = PublishPlatform.TENCENT
            item.options = {"collection": "AI 工程精选"}
        workspace.save_publish_batch(batch)
        return workspace, batch

    def test_dashboard_style_run_item_submits_only_selected_wechat_video(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_tencent_batch(temp)
            backend = FakeCollectionBackend()
            service = CollectionPublishBatchService(workspace, backend)
            service.approve(batch, "editor")

            service.run_item(batch, "item-1")

            self.assertEqual(backend.uploaded, ["video-1.mp4"])
            self.assertEqual(batch.items[0].state, CollectionPublishItemState.PENDING)
            self.assertEqual(batch.items[1].state, CollectionPublishItemState.SUBMITTED)
            self.assertEqual(batch.state, PublishBatchState.PARTIAL_SUCCESS)

    def test_dashboard_style_run_item_refuses_bilibili(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp, count=1)
            service = CollectionPublishBatchService(workspace, FakeCollectionBackend())
            service.approve(batch, "editor")
            with self.assertRaisesRegex(ValueError, "Bilibili publishing is paused"):
                service.run_item(batch, "item-0")

    def test_uploads_each_item_once_and_adds_it_to_collection(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp)
            backend = FakeCollectionBackend()
            service = CollectionPublishBatchService(workspace, backend)
            service.approve(batch, "editor")
            service.run(batch)
            self.assertEqual(batch.state, PublishBatchState.SUCCEEDED)
            self.assertEqual(len(backend.uploaded), 2)
            self.assertEqual(len(backend.added), 2)
            self.assertTrue(all(item.state == CollectionPublishItemState.COLLECTED for item in batch.items))
            restored = workspace.load_publish_batch(batch.id)
            self.assertIsInstance(restored, CollectionPublishBatch)

    def test_collection_link_retry_never_uploads_again(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp, count=1)
            backend = FakeCollectionBackend()
            backend.fail_add_once = True
            service = CollectionPublishBatchService(workspace, backend)
            service.approve(batch, "editor")
            service.run(batch)
            self.assertEqual(batch.items[0].state, CollectionPublishItemState.UPLOADED_UNCOLLECTED)
            self.assertEqual(len(backend.uploaded), 1)
            service.retry_collection_link(batch, batch.items[0].id)
            self.assertEqual(batch.state, PublishBatchState.SUCCEEDED)
            self.assertEqual(len(backend.uploaded), 1)
            self.assertEqual(len(backend.added), 2)

    def test_unsupported_collection_cli_falls_back_to_ordinary_upload(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp, count=1)
            backend = FakeOrdinaryUploadBackend()
            service = CollectionPublishBatchService(workspace, backend)
            service.approve(batch, "editor")
            service.run(batch)
            self.assertEqual(batch.state, PublishBatchState.SUCCEEDED)
            self.assertEqual(batch.remote_collection_id, BILIBILI_ORDINARY_UPLOAD_COLLECTION_ID)
            self.assertEqual(batch.items[0].state, CollectionPublishItemState.SUBMITTED)
            self.assertEqual(len(backend.uploaded), 1)
            self.assertEqual(backend.added, [])

    def test_explicit_bilibili_auth_rejection_is_retryable_pre_submit_failure(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp, count=1)
            backend = FakeRejectedBilibiliBackend()
            service = CollectionPublishBatchService(workspace, backend)
            service.approve(batch, "editor")
            service.run(batch)
            self.assertEqual(batch.state, PublishBatchState.FAILED)
            self.assertEqual(batch.items[0].state, CollectionPublishItemState.FAILED_PRE_SUBMIT)

    def test_reconciles_historical_uncertain_bilibili_auth_rejection(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp, count=1)
            service = CollectionPublishBatchService(workspace, FakeCollectionBackend())
            service.approve(batch, "editor")
            batch.state = PublishBatchState.PARTIAL_SUCCESS
            batch.items[0].state = CollectionPublishItemState.UNCERTAIN
            batch.items[0].last_error = 'ResponseData { code: -101, message: "账号未登录" }'
            workspace.save_publish_batch(batch)

            service.confirm_pre_submit_auth_rejection(batch, batch.items[0].id, "editor")

            self.assertEqual(batch.items[0].state, CollectionPublishItemState.FAILED_PRE_SUBMIT)
            restored = workspace.load_publish_batch(batch.id)
            self.assertEqual(restored.items[0].state, CollectionPublishItemState.FAILED_PRE_SUBMIT)

    def test_approval_digest_binds_every_video(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp, count=1)
            service = CollectionPublishBatchService(workspace, FakeCollectionBackend())
            service.approve(batch, "editor")
            Path(batch.items[0].video_path).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed or is missing"):
                service.run(batch)
            self.assertEqual(batch.state, PublishBatchState.READY_FOR_REVIEW)

    def test_approval_digest_binds_thumbnail_bytes(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp, count=1)
            thumbnail = Path(temp) / "cover.png"
            thumbnail.write_bytes(b"approved-cover")
            batch.items[0].options["thumbnail"] = str(thumbnail)
            service = CollectionPublishBatchService(workspace, FakeCollectionBackend())
            service.approve(batch, "editor")
            thumbnail.write_bytes(b"replacement-cover")
            with self.assertRaisesRegex(ValueError, "new human approval"):
                service.run(batch)
            self.assertEqual(batch.state, PublishBatchState.READY_FOR_REVIEW)


if __name__ == "__main__":
    unittest.main()
