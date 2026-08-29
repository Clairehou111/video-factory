import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from video_factory.collection_publish import (
    CollectionPublishBatch, CollectionPublishBatchService, CollectionPublishItem,
    CollectionPublishItemState,
)
from video_factory.publish import BackendResult, PublishBatchState, PublishPlatform, SOCIAL_AUTO_UPLOAD_COMMIT
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
