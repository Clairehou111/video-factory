from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .models import RenderProfile, VideoCollectionManifest, now_iso
from .publish import (
    BackendResult, FILE_OPTION_NAMES, PublishBatchState, PublishPlatform, PublishTarget,
    SOCIAL_AUTO_UPLOAD_COMMIT, SocialAutoUploadBackend,
)
from .quality import CheckResult
from .youtube import validate_collection


class CollectionPublishItemState(StrEnum):
    PENDING = "pending"
    PREFLIGHT_PASSED = "preflight_passed"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    COLLECTED = "collected"
    UPLOADED_UNCOLLECTED = "uploaded_uncollected"
    FAILED_PRE_SUBMIT = "failed_pre_submit"
    UNCERTAIN = "uncertain"


@dataclass(slots=True)
class CollectionPublishItem:
    id: str
    collection_item_id: str
    platform: PublishPlatform
    account_name: str
    collection_title: str
    order: int
    video_path: str
    video_sha256: str
    title: str
    description: str
    tags: list[str]
    options: dict[str, Any]
    state: CollectionPublishItemState = CollectionPublishItemState.PENDING
    attempts: int = 0
    remote_id: str = ""
    last_error: str = ""
    submitted_at: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.platform, PublishPlatform):
            self.platform = PublishPlatform(self.platform)
        if not isinstance(self.state, CollectionPublishItemState):
            self.state = CollectionPublishItemState(self.state)
        self.as_publish_target().validate()

    def as_publish_target(self) -> PublishTarget:
        options = dict(self.options)
        if self.platform == PublishPlatform.TENCENT:
            options.setdefault("collection", self.collection_title)
        return PublishTarget(
            self.platform, self.account_name, self.title, self.description,
            list(self.tags), options=options,
        )

    def approval_payload(self) -> dict[str, Any]:
        option_file_sha256 = {
            name: _sha256_file(Path(str(self.options[name])))
            for name in sorted(FILE_OPTION_NAMES & set(self.options))
            if self.options[name]
        }
        return {
            "id": self.id, "collection_item_id": self.collection_item_id,
            "platform": self.platform.value, "account_name": self.account_name,
            "collection_title": self.collection_title, "order": self.order,
            "video_path": str(Path(self.video_path).resolve()), "video_sha256": self.video_sha256,
            "title": self.title, "description": self.description,
            "tags": self.tags, "options": self.options,
            "option_file_sha256": option_file_sha256,
        }


@dataclass(slots=True)
class CollectionPublishBatch:
    id: str
    manifest_id: str
    collection_title: str
    items: list[CollectionPublishItem]
    state: PublishBatchState
    checks: list[dict[str, Any]]
    backend_commit: str = SOCIAL_AUTO_UPLOAD_COMMIT
    batch_type: str = "collection"
    remote_collection_id: str = ""
    approval_digest: str | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        if not isinstance(self.state, PublishBatchState):
            self.state = PublishBatchState(self.state)
        self.items = [item if isinstance(item, CollectionPublishItem) else CollectionPublishItem(**item) for item in self.items]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CollectionPublishBatch":
        return cls(**{**data, "items": [CollectionPublishItem(**item) for item in data.get("items", [])]})

    def approval_payload(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id, "collection_title": self.collection_title,
            "backend_commit": self.backend_commit,
            "items": [item.approval_payload() for item in self.items],
        }

    def compute_approval_digest(self) -> str:
        encoded = json.dumps(self.approval_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def approve(self, actor: str) -> None:
        if self.state != PublishBatchState.READY_FOR_REVIEW:
            raise ValueError(f"batch must be ready_for_review before approval; current state is {self.state}")
        if not actor.strip():
            raise ValueError("approval actor must not be empty")
        self.approval_digest = self.compute_approval_digest()
        self.approved_by = actor.strip()
        self.approved_at = now_iso()
        self.state = PublishBatchState.APPROVED
        self.updated_at = now_iso()

    def verify_approval(self) -> None:
        try:
            for item in self.items:
                path = Path(item.video_path)
                if not path.is_file() or _sha256_file(path) != item.video_sha256:
                    raise ValueError(f"approved collection video changed or is missing: {path}")
            current_approval_digest = self.compute_approval_digest()
        except Exception:
            self._invalidate_approval()
            raise
        if self.approval_digest != current_approval_digest:
            self._invalidate_approval()
            raise ValueError("approved collection payload changed; a new human approval is required")

    def _invalidate_approval(self) -> None:
        self.state = PublishBatchState.READY_FOR_REVIEW
        self.approval_digest = None
        self.approved_by = None
        self.approved_at = None
        self.updated_at = now_iso()


class CollectionPublisherBackend(Protocol):
    commit: str

    def check_account(self, target: PublishTarget) -> BackendResult: ...
    def submit_collection_video(self, target: PublishTarget, video_path: Path) -> BackendResult: ...
    def ensure_bilibili_collection(self, account_name: str, title: str) -> BackendResult: ...
    def add_bilibili_collection(
        self, account_name: str, collection_id: str, bvid: str, position: int,
    ) -> BackendResult: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_json_result(result: BackendResult) -> dict[str, Any]:
    if not result.succeeded:
        return {}
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _remote_video_id(result: BackendResult) -> str:
    payload = _parse_json_result(result)
    for key in ("bvid", "video_id", "id"):
        if str(payload.get(key) or "").strip():
            return str(payload[key]).strip()
    match = re.search(r"\bBV[0-9A-Za-z]{10}\b", result.stdout)
    return match.group(0) if match else ""


def create_collection_publish_batch(
    manifest: VideoCollectionManifest, spec: dict[str, Any], workspace: Path,
) -> CollectionPublishBatch:
    allowed = {"collection_title", "targets"}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError("unsupported collection publish spec fields: " + ", ".join(sorted(unknown)))
    targets = spec.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("collection publish spec targets must be an object")
    collection_title = str(spec.get("collection_title") or manifest.collection_title).strip()
    items: list[CollectionPublishItem] = []
    for platform_name, raw in targets.items():
        if platform_name not in {PublishPlatform.BILIBILI.value, PublishPlatform.TENCENT.value}:
            raise ValueError(f"collection publishing does not support platform: {platform_name}")
        if not isinstance(raw, dict):
            raise ValueError(f"collection target {platform_name} must be an object")
        allowed_target = {"account", "tid", "tags", "description", "collection"}
        unknown_target = set(raw) - allowed_target
        if unknown_target:
            raise ValueError(f"unsupported {platform_name} collection fields: {', '.join(sorted(unknown_target))}")
        platform = PublishPlatform(platform_name)
        profile = RenderProfile.BILIBILI_LANDSCAPE if platform == PublishPlatform.BILIBILI else RenderProfile.WECHAT_VERTICAL
        platform_rows = [
            (collection_item, render)
            for collection_item in manifest.items
            for render in collection_item.renders
            if render.profile == profile
        ]
        for platform_order, (collection_item, render) in enumerate(platform_rows, start=1):
            path = workspace / render.video_path
            options: dict[str, Any] = {}
            if platform == PublishPlatform.BILIBILI:
                options["tid"] = int(raw.get("tid") or 0)
            else:
                options["collection"] = str(raw.get("collection") or collection_title)
            items.append(CollectionPublishItem(
                id=f"{collection_item.id}-{platform.value}", collection_item_id=collection_item.id,
                platform=platform, account_name=str(raw.get("account") or ""),
                collection_title=collection_title, order=platform_order,
                video_path=str(path.resolve()), video_sha256=_sha256_file(path) if path.is_file() else "",
                title=render.title or collection_item.title,
                description=str(raw.get("description") or render.description),
                tags=[str(tag) for tag in raw.get("tags", render.tags)], options=options,
            ))
    checks = validate_collection(manifest, workspace)
    all_files = all(Path(item.video_path).is_file() and item.video_sha256 for item in items)
    checks.append(CheckResult("collection_publish_files", all_files, "all rendered files exist" if all_files else "one or more rendered files are missing"))
    state = PublishBatchState.READY_FOR_REVIEW if checks and all(item.passed for item in checks) else PublishBatchState.BLOCKED
    return CollectionPublishBatch(
        id=f"publish-collection-{manifest.id}-{uuid4().hex[:8]}", manifest_id=manifest.id,
        collection_title=collection_title, items=items, state=state,
        checks=[item.to_dict() for item in checks],
    )


class CollectionPublishBatchService:
    def __init__(self, workspace: Any, backend: CollectionPublisherBackend | SocialAutoUploadBackend) -> None:
        self.workspace = workspace
        self.backend = backend

    def save(self, batch: CollectionPublishBatch) -> None:
        self.workspace.save_publish_batch(batch)

    def approve(self, batch: CollectionPublishBatch, actor: str) -> CollectionPublishBatch:
        batch.approve(actor)
        self.save(batch)
        return batch

    def run(self, batch: CollectionPublishBatch) -> CollectionPublishBatch:
        if batch.backend_commit != self.backend.commit:
            raise ValueError(f"batch requires publisher {batch.backend_commit}; backend is {self.backend.commit}")
        if batch.state not in {PublishBatchState.APPROVED, PublishBatchState.FAILED, PublishBatchState.PARTIAL_SUCCESS}:
            raise ValueError(f"collection batch is not runnable from state {batch.state}")
        batch.verify_approval()
        batch.state = PublishBatchState.RUNNING
        self.save(batch)

        pending = [item for item in batch.items if item.state in {
            CollectionPublishItemState.PENDING, CollectionPublishItemState.FAILED_PRE_SUBMIT,
        }]
        accounts: dict[tuple[PublishPlatform, str], PublishTarget] = {}
        for item in pending:
            accounts.setdefault((item.platform, item.account_name), item.as_publish_target())
        failed_accounts: set[tuple[PublishPlatform, str]] = set()
        for key, target in accounts.items():
            result = self.backend.check_account(target)
            self._record(batch, target.platform.value, "preflight", result)
            if not result.succeeded:
                failed_accounts.add(key)
        if failed_accounts:
            for item in pending:
                item.state = CollectionPublishItemState.FAILED_PRE_SUBMIT
                item.last_error = (
                    "account preflight failed" if (item.platform, item.account_name) in failed_accounts
                    else "batch stopped because another target failed account preflight"
                )
            self._finish(batch)
            return batch

        bilibili_items = [item for item in pending if item.platform == PublishPlatform.BILIBILI]
        if bilibili_items and not batch.remote_collection_id:
            ensure = self.backend.ensure_bilibili_collection(bilibili_items[0].account_name, batch.collection_title)
            self._record(batch, PublishPlatform.BILIBILI.value, "collection_ensure", ensure)
            payload = _parse_json_result(ensure)
            collection_id = str(payload.get("collection_id") or payload.get("season_id") or "")
            if not ensure.succeeded or not collection_id:
                for item in bilibili_items:
                    item.state = CollectionPublishItemState.FAILED_PRE_SUBMIT
                    item.last_error = "Bilibili collection ensure failed or returned no collection id"
                self._finish(batch)
                return batch
            batch.remote_collection_id = collection_id
            self.save(batch)

        for item in sorted(pending, key=lambda value: (value.platform.value, value.order)):
            item.state = CollectionPublishItemState.SUBMITTING
            item.attempts += 1
            self.save(batch)
            target = item.as_publish_target()
            result = self.backend.submit_collection_video(target, Path(item.video_path))
            self._record(batch, item.platform.value, f"submit:{item.id}", result)
            if not result.succeeded:
                item.state = CollectionPublishItemState.UNCERTAIN if result.started else CollectionPublishItemState.FAILED_PRE_SUBMIT
                item.last_error = (result.stderr or result.stdout or "submission failed")[-1000:]
                self.save(batch)
                continue
            item.submitted_at = now_iso()
            item.remote_id = _remote_video_id(result)
            if item.platform != PublishPlatform.BILIBILI:
                item.state = CollectionPublishItemState.SUBMITTED
                item.last_error = ""
                self.save(batch)
                continue
            if not item.remote_id:
                item.state = CollectionPublishItemState.UPLOADED_UNCOLLECTED
                item.last_error = "upload succeeded but no BVID was returned; do not upload again"
                self.save(batch)
                continue
            link = self.backend.add_bilibili_collection(
                item.account_name, batch.remote_collection_id, item.remote_id, item.order,
            )
            self._record(batch, item.platform.value, f"collection_add:{item.id}", link)
            if link.succeeded:
                item.state = CollectionPublishItemState.COLLECTED
                item.last_error = ""
            else:
                item.state = CollectionPublishItemState.UPLOADED_UNCOLLECTED
                item.last_error = (link.stderr or link.stdout or "collection association failed")[-1000:]
            self.save(batch)
        self._finish(batch)
        return batch

    def retry_collection_link(self, batch: CollectionPublishBatch, item_id: str) -> CollectionPublishBatch:
        item = next((value for value in batch.items if value.id == item_id), None)
        if item is None:
            raise KeyError(item_id)
        if item.state != CollectionPublishItemState.UPLOADED_UNCOLLECTED or not item.remote_id:
            raise ValueError("only an uploaded_uncollected item with remote_id can retry collection association")
        batch.verify_approval()
        result = self.backend.add_bilibili_collection(
            item.account_name, batch.remote_collection_id, item.remote_id, item.order,
        )
        self._record(batch, item.platform.value, f"collection_retry:{item.id}", result)
        if result.succeeded:
            item.state = CollectionPublishItemState.COLLECTED
            item.last_error = ""
        else:
            item.last_error = (result.stderr or result.stdout or "collection association failed")[-1000:]
        self._finish(batch)
        return batch

    def _record(self, batch: CollectionPublishBatch, platform: str, action: str, result: BackendResult) -> None:
        self.workspace.append_publish_attempt(
            batch.id, platform, action, {**result.to_audit_dict(), "recorded_at": now_iso()},
        )

    def _finish(self, batch: CollectionPublishBatch) -> None:
        success_states = {CollectionPublishItemState.SUBMITTED, CollectionPublishItemState.COLLECTED}
        states = {item.state for item in batch.items}
        if states and states <= success_states:
            batch.state = PublishBatchState.SUCCEEDED
        elif states & success_states or CollectionPublishItemState.UPLOADED_UNCOLLECTED in states:
            batch.state = PublishBatchState.PARTIAL_SUCCESS
        else:
            batch.state = PublishBatchState.FAILED
        batch.updated_at = now_iso()
        self.save(batch)
