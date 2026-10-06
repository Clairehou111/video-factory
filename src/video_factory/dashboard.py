from __future__ import annotations

import hashlib
import json
import mimetypes
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse
from zoneinfo import ZoneInfo

from .collection_publish import (
    CollectionPublishBatch, CollectionPublishBatchService, CollectionPublishItemState,
)
from .discovery import ResourceDiscoveryConfig, ResourceDiscoveryService
from .publish import (
    PublishBatch, PublishBatchService, PublishBatchState, PublishPlatform,
    PublishTargetState, SocialAutoUploadBackend, account_auth_required,
    publisher_error_text,
)
from .serde import load_collection_manifest, load_manifest


PUBLISHABLE_BATCH_STATES = {
    PublishBatchState.READY_FOR_REVIEW,
    PublishBatchState.APPROVED,
    PublishBatchState.FAILED,
    PublishBatchState.PARTIAL_SUCCESS,
}


@dataclass(slots=True)
class DashboardCard:
    batch_id: str
    item_id: str
    manifest_id: str
    title: str
    description: str
    video_path: str
    source_url: str
    source_title: str
    editorial_mode: str
    batch_state: str
    item_state: str
    created_at: str
    failed_checks: list[str]
    can_publish: bool
    can_review: bool
    requires_login: bool
    last_error: str
    review_check: str
    action_label: str
    sequence: int = 0
    sequence_total: int = 0
    schedule_at: str = ""

    def to_dict(self, media_id: str) -> dict[str, Any]:
        sequence_label = (
            f"{self.sequence:02d}/{self.sequence_total:02d}"
            if self.sequence > 0 and self.sequence_total > 0 else ""
        )
        return {
            "batch_id": self.batch_id,
            "item_id": self.item_id,
            "manifest_id": self.manifest_id,
            "title": self.title,
            "display_title": (
                f"{sequence_label} · {self.title}" if sequence_label else self.title
            ),
            "description": self.description,
            "video_path": self.video_path,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "editorial_mode": self.editorial_mode,
            "batch_state": self.batch_state,
            "item_state": self.item_state,
            "created_at": self.created_at,
            "failed_checks": self.failed_checks,
            "can_publish": self.can_publish,
            "can_review": self.can_review,
            "requires_login": self.requires_login,
            "last_error": self.last_error,
            "review_check": self.review_check,
            "action_label": self.action_label,
            "sequence": self.sequence,
            "sequence_total": self.sequence_total,
            "sequence_label": sequence_label,
            "video_available": Path(self.video_path).is_file(),
            "media_url": f"/media/{media_id}",
            "platform": "视频号",
            "schedule_at": self.schedule_at,
        }


class PublishDashboard:
    """Read the human review queue and execute one explicitly confirmed item."""

    def __init__(
        self, workspace: Any, actor: str = "dashboard-reviewer",
        backend_factory: Callable[[], Any] = SocialAutoUploadBackend,
        discovery_service: ResourceDiscoveryService | None = None,
        discovery_config: ResourceDiscoveryConfig | None = None,
    ) -> None:
        self.workspace = workspace
        self.actor = actor.strip() or "dashboard-reviewer"
        self.backend_factory = backend_factory
        self.discovery_service = discovery_service or ResourceDiscoveryService(workspace)
        self.discovery_config = discovery_config or ResourceDiscoveryConfig()
        self.csrf_token = secrets.token_urlsafe(32)
        self._publish_lock = threading.RLock()
        self._retry_lock = threading.RLock()
        self._retry_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dashboard-discovery-retry",
        )
        self._active_retry_ids: set[str] = set()
        self._retry_instance = secrets.token_hex(8)
        self._recover_retry_tasks()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def _retry_task_path(self, candidate_id: str) -> Path:
        digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:24]
        return self.workspace.root / "discovery" / "retry-tasks" / f"{digest}.json"

    def _save_retry_task(self, task: dict[str, Any]) -> None:
        path = self._retry_task_path(str(task["candidate_id"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{threading.get_ident()}.tmp")
        temporary.write_text(
            json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(path)

    def _load_retry_task(self, candidate_id: str) -> dict[str, Any] | None:
        path = self._retry_task_path(candidate_id)
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        if task.get("candidate_id") != candidate_id:
            return None
        if (
            task.get("status") in {"queued", "running"}
            and task.get("instance") != self._retry_instance
        ):
            now = self._now_iso()
            task["status"] = "interrupted"
            task["updated_at"] = now
            task["finished_at"] = now
            task["last_error"] = "Dashboard restarted before the retry returned a result"
            task.setdefault("logs", []).append({
                "at": now, "message": "Retry interrupted by Dashboard restart",
            })
            self._save_retry_task(task)
        return task

    def _recover_retry_tasks(self) -> None:
        """Resume queued work left behind when the Dashboard process restarted."""
        directory = self.workspace.root / "discovery" / "retry-tasks"
        if not directory.is_dir():
            return
        pending: list[dict[str, Any]] = []
        for path in directory.glob("*.json"):
            try:
                task = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, TypeError, json.JSONDecodeError):
                continue
            if (
                task.get("candidate_id")
                and task.get("status") in {"queued", "running"}
                and task.get("instance") != self._retry_instance
            ):
                pending.append(task)
        pending.sort(key=lambda task: (
            str(task.get("created_at") or ""), str(task.get("candidate_id") or ""),
        ))
        for task in pending:
            candidate_id = str(task["candidate_id"])
            try:
                candidate = self.workspace.load_discovery_candidate(candidate_id)
            except (KeyError, OSError, ValueError, TypeError):
                continue
            now = self._now_iso()
            if str(candidate.get("status") or "") == "generated":
                task.update({
                    "status": "succeeded", "stage": "completed",
                    "updated_at": now, "finished_at": now,
                    "instance": self._retry_instance, "last_error": "",
                })
                task.setdefault("logs", []).append({
                    "at": now,
                    "message": "Generation had completed before Dashboard restart",
                })
                self._save_retry_task(task)
                continue
            task.update({
                "status": "queued", "stage": "queued", "updated_at": now,
                "instance": self._retry_instance,
                "recovery_count": int(task.get("recovery_count") or 0) + 1,
            })
            task.pop("finished_at", None)
            task.pop("last_error", None)
            task.setdefault("logs", []).append({
                "at": now, "message": "Retry recovered after Dashboard restart",
            })
            task["logs"] = task["logs"][-20:]
            self._active_retry_ids.add(candidate_id)
            self._save_retry_task(task)
            self._retry_executor.submit(self._run_discovery_retry, candidate_id)

    def _recent_retry_tasks(self) -> list[dict[str, Any]]:
        directory = self.workspace.root / "discovery" / "retry-tasks"
        if not directory.is_dir():
            return []
        cutoff = datetime.now(UTC) - timedelta(minutes=15)
        recent: list[dict[str, Any]] = []
        for path in directory.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                candidate_id = str(payload.get("candidate_id") or "")
                if not candidate_id:
                    continue
                task = self._load_retry_task(candidate_id)
                if not task:
                    continue
                if task.get("status") in {"queued", "running"}:
                    recent.append(task)
                    continue
                updated = datetime.fromisoformat(
                    str(task.get("updated_at") or "").replace("Z", "+00:00")
                )
                if task.get("status") == "succeeded" and updated >= cutoff:
                    recent.append(task)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return recent

    def _update_retry_task(
        self, candidate_id: str, status: str, message: str, **values: Any,
    ) -> dict[str, Any]:
        with self._retry_lock:
            task = self._load_retry_task(candidate_id) or {
                "candidate_id": candidate_id, "logs": [],
                "created_at": self._now_iso(), "instance": self._retry_instance,
            }
            now = self._now_iso()
            task.update(values)
            task["status"] = status
            task["updated_at"] = now
            task.setdefault("logs", []).append({"at": now, "message": message})
            task["logs"] = task["logs"][-20:]
            self._save_retry_task(task)
            return dict(task)

    def _run_discovery_retry(self, candidate_id: str) -> None:
        self._update_retry_task(
            candidate_id, "running", "Generation started", started_at=self._now_iso(),
            stage="generation",
        )
        try:
            result = self.discovery_service.adopt_candidate(
                candidate_id, self.discovery_config,
            )
            generated = result.get("status") == "generated"
            final_status = "succeeded" if generated else "failed"
            last_error = str(result.get("last_error") or "")[-2000:]
            self._update_retry_task(
                candidate_id, final_status,
                "Generation completed" if generated else "Generation stopped at a validation gate",
                stage="completed" if generated else "validation",
                finished_at=self._now_iso(), last_error=last_error,
                result_status=str(result.get("status") or ""),
                attempts=len(result.get("attempts") or []),
            )
        except Exception as error:
            self._update_retry_task(
                candidate_id, "failed", "Retry process raised an exception",
                stage="internal_error", finished_at=self._now_iso(),
                last_error=f"{type(error).__name__}: {str(error)[-1900:]}",
            )
        finally:
            with self._retry_lock:
                self._active_retry_ids.discard(candidate_id)

    def discovery_failures(self) -> list[dict[str, Any]]:
        """Return only actionable discovery failures; history stays in state artifacts."""
        state = self.workspace.load_discovery_state()
        rows: dict[str, dict[str, Any]] = {}
        for row in state.get("needs_human_candidates") or []:
            if isinstance(row, dict) and row.get("candidate_id"):
                rows[str(row["candidate_id"])] = dict(row)
        for channel, channel_state in (state.get("channels") or {}).items():
            if not isinstance(channel_state, dict):
                continue
            for candidate_id, pending in (channel_state.get("pending_candidates") or {}).items():
                if not isinstance(pending, dict):
                    continue
                failure = dict(pending.get("failure") or {})
                candidate = dict(pending.get("candidate") or {})
                failure.setdefault("candidate_id", str(candidate_id))
                failure.setdefault("channel", channel)
                failure.setdefault("title", candidate.get("title", ""))
                failure.setdefault("url", candidate.get("url", ""))
                failure.setdefault("status", candidate.get("status", "retry_wait"))
                failure.setdefault("attempts", int(pending.get("retry_runs") or 0))
                failure.setdefault("next_retry_at", str(pending.get("retry_at") or ""))
                failure.setdefault("last_error", str(pending.get("last_error") or ""))
                failure.setdefault("retry_eligible", True)
                rows[str(candidate_id)] = failure
        for task in self._recent_retry_tasks():
            candidate_id = str(task["candidate_id"])
            if candidate_id in rows:
                continue
            try:
                candidate = self.workspace.load_discovery_candidate(candidate_id)
            except (KeyError, OSError, ValueError, TypeError):
                candidate = {}
            rows[candidate_id] = {
                "candidate_id": candidate_id,
                "channel": str(candidate.get("channel") or "discovery"),
                "title": str(candidate.get("title") or candidate_id),
                "url": str(candidate.get("url") or ""),
                "status": str(task.get("status") or ""),
            }
        active: list[dict[str, Any]] = []
        for candidate_id, row in rows.items():
            try:
                candidate = self.workspace.load_discovery_candidate(candidate_id)
            except (KeyError, OSError, ValueError, TypeError):
                candidate = {}
            task = self._load_retry_task(candidate_id)
            status = str(candidate.get("status") or row.get("status") or "")
            if (
                status in {"generated", "skipped", "resolved", "not_selected", "not_adopted"}
                and not (task and task.get("status") == "succeeded")
            ):
                continue
            row["status"] = status or "needs_human"
            row["candidate_id"] = candidate_id
            if task:
                row["retry_task"] = task
                if task.get("status") in {"queued", "running", "succeeded"}:
                    row["status"] = str(task["status"])
                    row["retry_eligible"] = False
                elif task.get("status") in {"failed", "interrupted"}:
                    row["last_error"] = str(
                        task.get("last_error") or row.get("last_error") or ""
                    )
            active.append(row)
        active.sort(
            key=lambda row: str(row.get("last_failed_at") or row.get("recorded_at") or ""),
            reverse=True,
        )
        return active

    def retry_discovery(self, candidate_id: str) -> dict[str, Any]:
        candidate_id = candidate_id.strip()
        if not candidate_id:
            raise ValueError("candidate_id is required")
        # Validate before accepting work so a typo cannot create a phantom task.
        self.workspace.load_discovery_candidate(candidate_id)
        with self._retry_lock:
            if candidate_id in self._active_retry_ids:
                return self._load_retry_task(candidate_id) or {
                    "candidate_id": candidate_id, "status": "running",
                }
            now = self._now_iso()
            task = {
                "candidate_id": candidate_id, "status": "queued",
                "stage": "queued", "created_at": now, "updated_at": now,
                "instance": self._retry_instance,
                "logs": [{"at": now, "message": "Retry queued"}],
            }
            self._active_retry_ids.add(candidate_id)
            self._save_retry_task(task)
            self._retry_executor.submit(self._run_discovery_retry, candidate_id)
            return dict(task)

    def skip_discovery(self, candidate_id: str, reason: str) -> dict[str, Any]:
        return self.discovery_service.skip(candidate_id.strip(), reason)

    def queue(self) -> tuple[list[dict[str, Any]], dict[str, Path]]:
        latest: dict[str, dict[str, Any]] = {}
        for path in self.workspace.publish_dir.glob("*/batch.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("queue_hidden"):
                continue
            manifest_id = str(payload.get("manifest_id") or "")
            if not manifest_id:
                continue
            logical_id = manifest_id
            if payload.get("batch_type") == "collection":
                try:
                    manifest = self.workspace.load_collection_manifest(manifest_id)
                except (KeyError, OSError, TypeError, ValueError):
                    manifest = None
                if manifest is not None and manifest.source_video_id:
                    # Rerenders of the same clip have new collection ids but
                    # retain their original source ranges. Distinct highlights
                    # from one interview must remain separate queue cards.
                    logical_id = self._collection_logical_id(manifest)
            current = latest.get(logical_id)
            if current is None or str(payload.get("created_at") or "") > str(current.get("created_at") or ""):
                latest[logical_id] = payload

        cards: list[DashboardCard] = []
        for payload in latest.values():
            if payload.get("batch_type") == "collection":
                cards.extend(self._collection_cards(payload))
            else:
                cards.extend(self._ordinary_cards(payload))
        # Keep each source collection in its authored sequence while still
        # showing newer batches before older batches.
        cards.sort(key=lambda row: (row.sequence, row.item_id))
        cards.sort(key=lambda row: (row.created_at, row.batch_id), reverse=True)

        media: dict[str, Path] = {}
        rows: list[dict[str, Any]] = []
        for card in cards:
            path = Path(card.video_path).resolve()
            media_id = hashlib.sha256(
                f"{card.batch_id}\0{card.item_id}\0{path}".encode("utf-8")
            ).hexdigest()[:24]
            media[media_id] = path
            rows.append(card.to_dict(media_id))
        return rows, media

    @staticmethod
    def _collection_logical_id(manifest: Any) -> str:
        source_video_id = str(getattr(manifest, "source_video_id", "") or "")
        clip_ranges: list[tuple[float, float]] = []
        for item in getattr(manifest, "items", []) or []:
            for source_range in getattr(item, "source_ranges", []) or []:
                if isinstance(source_range, dict):
                    start = source_range.get("original_start")
                    end = source_range.get("original_end")
                    start = source_range.get("start") if start is None else start
                    end = source_range.get("end") if end is None else end
                else:
                    start = getattr(source_range, "original_start", None)
                    end = getattr(source_range, "original_end", None)
                    start = getattr(source_range, "start", None) if start is None else start
                    end = getattr(source_range, "end", None) if end is None else end
                if start is not None and end is not None:
                    clip_ranges.append((round(float(start), 3), round(float(end), 3)))
        if not clip_ranges:
            return f"youtube:{source_video_id}"
        encoded = json.dumps(sorted(clip_ranges), separators=(",", ":"))
        clip_digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        return f"youtube:{source_video_id}:clip:{clip_digest}"

    def _collection_cards(self, payload: dict[str, Any]) -> list[DashboardCard]:
        manifest_id = str(payload.get("manifest_id") or "")
        source_url = source_title = editorial_mode = ""
        try:
            manifest = self.workspace.load_collection_manifest(manifest_id)
        except (KeyError, OSError, TypeError, ValueError):
            path = self.workspace.collections_dir / f"{manifest_id}.json"
            try:
                manifest = load_collection_manifest(path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                manifest = None
        if manifest is not None:
            source_url = manifest.source_url
            source_title = manifest.source_title
            editorial_mode = manifest.editorial_mode
        failed_checks = self._failed_checks(payload)
        batch_state = str(payload.get("state") or "")
        cards: list[DashboardCard] = []
        tencent_items = [
            item for item in payload.get("items") or []
            if isinstance(item, dict)
            and item.get("platform") == PublishPlatform.TENCENT.value
        ]
        sequence_total = len(tencent_items)
        for fallback_sequence, item in enumerate(tencent_items, start=1):
            item_state = str(item.get("state") or "")
            if item_state == CollectionPublishItemState.SUBMITTED.value:
                continue
            try:
                sequence = int(item.get("order") or fallback_sequence)
            except (TypeError, ValueError):
                sequence = fallback_sequence
            if sequence < 1:
                sequence = fallback_sequence
            can_publish = (
                batch_state in {state.value for state in PUBLISHABLE_BATCH_STATES}
                and item_state in {
                    CollectionPublishItemState.PENDING.value,
                    CollectionPublishItemState.FAILED_PRE_SUBMIT.value,
                }
                and Path(str(item.get("video_path") or "")).is_file()
                and not failed_checks
            )
            can_review = (
                batch_state == PublishBatchState.BLOCKED.value
                and failed_checks == ["rights_review"]
                and Path(str(item.get("video_path") or "")).is_file()
            )
            last_error = publisher_error_text(str(item.get("last_error") or ""))
            requires_login = account_auth_required(last_error)
            cards.append(DashboardCard(
                batch_id=str(payload.get("id") or ""), item_id=str(item.get("id") or ""),
                manifest_id=manifest_id, title=str(item.get("title") or "未命名视频"),
                description=str(item.get("description") or ""),
                video_path=str(item.get("video_path") or ""), source_url=source_url,
                source_title=source_title, editorial_mode=editorial_mode,
                batch_state=batch_state, item_state=item_state,
                created_at=str(payload.get("created_at") or ""), failed_checks=failed_checks,
                can_publish=can_publish, can_review=can_review,
                requires_login=requires_login, last_error=last_error,
                review_check="rights_review" if can_review else "",
                action_label=("确认复用依据已审核" if can_review else self._action_label(
                    batch_state, item_state, can_publish, can_review, requires_login,
                )),
                sequence=sequence, sequence_total=sequence_total,
                schedule_at=str(item.get("schedule_at") or ""),
            ))
        return cards

    def _ordinary_cards(self, payload: dict[str, Any]) -> list[DashboardCard]:
        manifest_id = str(payload.get("manifest_id") or "")
        source_url = source_title = ""
        try:
            manifest = load_manifest(self.workspace.manifests_dir / f"{manifest_id}.json")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            manifest = None
        if manifest is not None:
            source_url = manifest.source_urls[0] if manifest.source_urls else ""
            source_title = manifest.fixed_title or manifest.fixed_hook
        failed_checks = self._failed_checks(payload)
        batch_state = str(payload.get("state") or "")
        cards: list[DashboardCard] = []
        for target in payload.get("targets") or []:
            if not isinstance(target, dict) or target.get("platform") != PublishPlatform.TENCENT.value:
                continue
            item_state = str(target.get("state") or "")
            if item_state == PublishTargetState.SUBMITTED.value:
                continue
            video_path = str(payload.get("video_path") or "")
            can_publish = (
                batch_state in {state.value for state in PUBLISHABLE_BATCH_STATES}
                and item_state in {
                    PublishTargetState.PENDING.value,
                    PublishTargetState.FAILED_PRE_SUBMIT.value,
                }
                and Path(video_path).is_file()
                and not failed_checks
            )
            can_review = (
                batch_state == PublishBatchState.BLOCKED.value
                and failed_checks == ["editorial_safety_review"]
                and Path(video_path).is_file()
            )
            last_error = publisher_error_text(str(target.get("last_error") or ""))
            requires_login = account_auth_required(last_error)
            cards.append(DashboardCard(
                batch_id=str(payload.get("id") or ""), item_id="tencent",
                manifest_id=manifest_id, title=str(target.get("title") or "未命名视频"),
                description=str(target.get("description") or ""), video_path=video_path,
                source_url=source_url, source_title=source_title, editorial_mode="news_brief",
                batch_state=batch_state, item_state=item_state,
                created_at=str(payload.get("created_at") or ""), failed_checks=failed_checks,
                can_publish=can_publish, can_review=can_review,
                requires_login=requires_login, last_error=last_error,
                review_check="editorial_safety_review" if can_review else "",
                action_label=self._action_label(
                    batch_state, item_state, can_publish, can_review, requires_login,
                ),
                schedule_at=str(target.get("schedule_at") or ""),
            ))
        return cards

    @staticmethod
    def _failed_checks(payload: dict[str, Any]) -> list[str]:
        return [
            str(row.get("name") or "unknown_check")
            for row in payload.get("checks") or []
            if isinstance(row, dict) and not row.get("passed", False)
        ]

    @staticmethod
    def _action_label(
        batch_state: str, item_state: str, can_publish: bool, can_review: bool,
        requires_login: bool = False,
    ) -> str:
        if can_publish and requires_login:
            return "登录视频号并继续发布"
        if can_publish:
            return "重新检查并发布" if item_state == "failed_pre_submit" else "确认并发布"
        if can_review:
            return "确认安全角度已审核"
        if item_state == "uncertain":
            return "结果不确定，需人工核对"
        if batch_state == PublishBatchState.BLOCKED.value:
            return "质量门未通过"
        return "暂不可发布"

    def review(self, batch_id: str, item_id: str) -> dict[str, Any]:
        if not batch_id or not item_id:
            raise ValueError("dashboard review requires batch_id and item_id")
        with self._publish_lock:
            batch = self.workspace.load_publish_batch(batch_id)
            if isinstance(batch, CollectionPublishBatch):
                item = next((row for row in batch.items if row.id == item_id), None)
                if item is None or item.platform != PublishPlatform.TENCENT:
                    raise ValueError("collection rights review requires one Tencent item")
                batch.record_review_override(
                    "rights_review", self.actor,
                    "已完整观看并确认访谈片段、来源署名与复用依据可用于本次发布",
                )
                self.workspace.save_publish_batch(batch)
                return {
                    "batch_id": batch.id, "item_id": item_id,
                    "batch_state": batch.state.value, "reviewed": True,
                    "reviewed_by": self.actor, "review_check": "rights_review",
                }
            if not isinstance(batch, PublishBatch) or item_id != "tencent":
                raise ValueError("ordinary safety review requires one Tencent video")
            batch.record_review_override(
                "editorial_safety_review", self.actor,
                "已完整观看；内容仅用于 AI 安全、可解释性与防御研究，不提供滥用操作指导",
            )
            self.workspace.save_publish_batch(batch)
            return {
                "batch_id": batch.id, "item_id": item_id,
                "batch_state": batch.state.value, "reviewed": True,
                "reviewed_by": self.actor,
            }

    def login_and_publish(self, batch_id: str, item_id: str) -> dict[str, Any]:
        """Run one bounded headed-login recovery, then resume the selected item."""
        if not batch_id or not item_id:
            raise ValueError("batch_id and item_id are required")
        with self._publish_lock:
            batch = self.workspace.load_publish_batch(batch_id)
            backend = self.backend_factory()
            if not hasattr(backend, "login_account"):
                raise RuntimeError("publisher backend does not support interactive login recovery")
            if isinstance(batch, CollectionPublishBatch):
                item = next((row for row in batch.items if row.id == item_id), None)
                if item is None or item.platform != PublishPlatform.TENCENT:
                    raise ValueError("login recovery requires one Tencent collection item")
                account_name = item.account_name
            elif isinstance(batch, PublishBatch):
                targets = [row for row in batch.targets if row.platform == PublishPlatform.TENCENT]
                if len(targets) != 1 or item_id != "tencent":
                    raise ValueError("login recovery requires one Tencent target")
                account_name = targets[0].account_name
            else:
                raise TypeError("unsupported publish batch")
            login = backend.login_account(PublishPlatform.TENCENT, account_name, headless=False)
            self.workspace.append_publish_attempt(
                batch.id, PublishPlatform.TENCENT.value, "login_recovery",
                {**login.to_audit_dict(), "recorded_at": str(getattr(batch, "updated_at", ""))},
            )
            if not login.succeeded:
                detail = (login.stderr or login.stdout or "视频号登录未完成")[-1000:]
                if isinstance(batch, CollectionPublishBatch):
                    item.last_error = detail
                else:
                    targets[0].last_error = detail
                self.workspace.save_publish_batch(batch)
                return {
                    "batch_id": batch.id, "item_id": item_id, "published": False,
                    "requires_login": True, "error": detail,
                }
            result = self.publish(batch_id, item_id)
            result["login_recovered"] = True
            return result

    @staticmethod
    def _schedule_at(value: str | None) -> str | None:
        cleaned = str(value or "").strip().replace("T", " ")
        if not cleaned:
            return None
        try:
            scheduled = datetime.strptime(cleaned, "%Y-%m-%d %H:%M")
        except ValueError as error:
            raise ValueError("定时发布时间必须使用 YYYY-MM-DD HH:MM") from error
        beijing_now = datetime.now(ZoneInfo("Asia/Shanghai")).replace(tzinfo=None)
        if scheduled <= beijing_now + timedelta(hours=2):
            raise ValueError("视频号定时发布至少需要提前 2 小时（北京时间）")
        return scheduled.strftime("%Y-%m-%d %H:%M")

    def publish(
        self, batch_id: str, item_id: str, schedule_at: str | None = None,
        update_schedule: bool = False,
    ) -> dict[str, Any]:
        if not batch_id or not item_id:
            raise ValueError("batch_id and item_id are required")
        requested_schedule = self._schedule_at(schedule_at) if update_schedule else None
        with self._publish_lock:
            batch = self.workspace.load_publish_batch(batch_id)
            backend = self.backend_factory()
            if isinstance(batch, CollectionPublishBatch):
                item = next((row for row in batch.items if row.id == item_id), None)
                if item is None:
                    raise KeyError(item_id)
                if item.platform != PublishPlatform.TENCENT:
                    raise ValueError("Bilibili publishing is paused")
                if update_schedule and item.schedule_at != requested_schedule:
                    if batch.state != PublishBatchState.READY_FOR_REVIEW:
                        raise ValueError("定时发布时间只能在最终审核前设置")
                    item.schedule_at = requested_schedule
                    item.as_publish_target().validate()
                    self.workspace.save_publish_batch(batch)
                service = CollectionPublishBatchService(self.workspace, backend)
                if batch.state == PublishBatchState.READY_FOR_REVIEW:
                    service.approve(batch, self.actor)
                result = service.run_item(batch, item_id)
                final_item = next(row for row in result.items if row.id == item_id)
                return {
                    "batch_id": result.id, "item_id": item_id,
                    "batch_state": result.state.value, "item_state": final_item.state.value,
                    "published": final_item.state == CollectionPublishItemState.SUBMITTED,
                    "error": final_item.last_error,
                    "requires_login": account_auth_required(final_item.last_error),
                }
            if not isinstance(batch, PublishBatch):
                raise TypeError("unsupported publish batch")
            targets = [row for row in batch.targets if row.platform == PublishPlatform.TENCENT]
            if len(targets) != 1 or item_id != "tencent":
                raise ValueError("dashboard ordinary batches require exactly one Tencent target")
            target = targets[0]
            if update_schedule and target.schedule_at != requested_schedule:
                if batch.state != PublishBatchState.READY_FOR_REVIEW:
                    raise ValueError("定时发布时间只能在最终审核前设置")
                target.schedule_at = requested_schedule
                target.validate()
                self.workspace.save_publish_batch(batch)
            service = PublishBatchService(self.workspace, backend)
            if batch.state == PublishBatchState.READY_FOR_REVIEW:
                service.approve(batch, self.actor)
            if target.state == PublishTargetState.FAILED_PRE_SUBMIT:
                result = service.retry(batch, PublishPlatform.TENCENT)
            else:
                result = service.run(batch)
            return {
                "batch_id": result.id, "item_id": item_id,
                "batch_state": result.state.value, "item_state": target.state.value,
                "published": target.state == PublishTargetState.SUBMITTED,
                "error": target.last_error,
                "requires_login": account_auth_required(target.last_error),
            }


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], dashboard: PublishDashboard) -> None:
        self.dashboard = dashboard
        super().__init__(address, DashboardRequestHandler)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send(HTTPStatus.OK, DASHBOARD_HTML, "text/html; charset=utf-8")
            return
        if path == "/healthz":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        if path == "/api/queue":
            rows, _ = self.server.dashboard.queue()
            self._json(HTTPStatus.OK, {
                "csrf_token": self.server.dashboard.csrf_token,
                "items": rows,
                "summary": {
                    "total": len(rows),
                    "ready": sum(1 for row in rows if row["can_publish"]),
                    "blocked": sum(1 for row in rows if row["batch_state"] == "blocked"),
                    "attention": sum(
                        1 for row in rows
                        if row["can_review"] or row["item_state"] == "uncertain"
                    ),
                },
                "bilibili_paused": True,
            })
            return
        if path == "/api/discovery-failures":
            rows = self.server.dashboard.discovery_failures()
            self._json(HTTPStatus.OK, {
                "csrf_token": self.server.dashboard.csrf_token,
                "items": rows,
                "summary": {"active": len(rows)},
            })
            return
        if path.startswith("/media/"):
            media_id = unquote(path.removeprefix("/media/"))
            _, media = self.server.dashboard.queue()
            target = media.get(media_id)
            if target is None or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_file(target)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {
            "/api/publish", "/api/review", "/api/login-and-publish",
            "/api/discovery-retry", "/api/discovery-skip",
        }:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if self.headers.get("X-Video-Factory-CSRF") != self.server.dashboard.csrf_token:
            self._json(HTTPStatus.FORBIDDEN, {"error": "invalid CSRF token"})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            if not 0 < length <= 65536:
                raise ValueError("invalid request length")
            payload = json.loads(self.rfile.read(length))
            if path == "/api/discovery-retry":
                result = self.server.dashboard.retry_discovery(
                    str(payload.get("candidate_id") or ""),
                )
                self._json(HTTPStatus.ACCEPTED, result)
                return
            if path == "/api/discovery-skip":
                result = self.server.dashboard.skip_discovery(
                    str(payload.get("candidate_id") or ""),
                    str(payload.get("reason") or ""),
                )
                self._json(HTTPStatus.OK, result)
                return
            batch_id = str(payload.get("batch_id") or "")
            item_id = str(payload.get("item_id") or "")
            schedule_at = payload.get("schedule_at")
            result = (
                self.server.dashboard.review(batch_id, item_id)
                if path == "/api/review"
                else self.server.dashboard.login_and_publish(batch_id, item_id)
                if path == "/api/login-and-publish"
                else self.server.dashboard.publish(
                    batch_id, item_id, schedule_at, update_schedule=True,
                )
            )
        except Exception as error:  # Request boundary records a safe, concise error.
            self._json(HTTPStatus.CONFLICT, {
                "error": f"{type(error).__name__}: {str(error)[-1000:]}",
            })
            return
        self._json(HTTPStatus.OK, result)

    def _send_file(self, path: Path) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            raw = range_header.removeprefix("bytes=").split(",", 1)[0]
            first, _, last = raw.partition("-")
            try:
                if first:
                    start = int(first)
                    end = int(last) if last else end
                elif last:
                    start = max(0, size - int(last))
            except ValueError:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            end = min(end, size - 1)
            if start < 0 or start > end:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # Browsers routinely close a prior range request when the
                    # user seeks or leaves the preview. It is not a server error.
                    return
                remaining -= len(chunk)

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._send(
            status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _send(self, status: HTTPStatus, body: str | bytes, content_type: str) -> None:
        encoded = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; media-src 'self'")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve_dashboard(
    workspace: Any, host: str = "127.0.0.1", port: int = "8765",
    actor: str = "dashboard-reviewer",
    discovery_config: ResourceDiscoveryConfig | None = None,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("dashboard is intentionally local-only; bind to a loopback address")
    server = DashboardServer(
        (host, port),
        PublishDashboard(workspace, actor, discovery_config=discovery_config),
    )
    print(f"Video Factory dashboard: http://{host}:{port}", flush=True)
    server.serve_forever()


DASHBOARD_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video Factory · 审核台</title>
<style>
:root{--ink:#171914;--muted:#71776b;--paper:#f3f1e9;--card:#fffdf8;--green:#124d38;--lime:#d9ff63;--red:#a33b2b;--line:#d9d6ca}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 ui-sans-serif,-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}.shell{max-width:1360px;margin:auto;padding:32px}.top{display:flex;justify-content:space-between;gap:24px;align-items:end;border-bottom:2px solid var(--ink);padding-bottom:22px}.eyebrow{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--green);font-weight:800}h1{font:700 clamp(34px,5vw,72px)/.95 ui-serif,Georgia,"Songti SC",serif;margin:8px 0 0;letter-spacing:-.04em}.note{max-width:470px;color:var(--muted)}.tabs{display:flex;gap:8px;margin:20px 0}.tabs button{margin:0;box-shadow:none}.tabs button.active{background:var(--ink)}.panel[hidden]{display:none}.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:22px 0}.stat{background:var(--ink);color:white;padding:14px 16px;border-radius:4px}.stat b{display:block;font-size:28px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:18px}.card{background:var(--card);border:1px solid var(--line);box-shadow:5px 5px 0 var(--ink);padding:14px;display:flex;flex-direction:column;gap:12px}.card video{width:100%;aspect-ratio:9/16;max-height:560px;background:#111;object-fit:contain}.missing{aspect-ratio:9/16;max-height:560px;display:grid;place-items:center;text-align:center;padding:30px;background:#dedbd0;color:var(--muted);border:1px dashed var(--muted)}.meta{display:flex;justify-content:space-between;gap:8px;align-items:center}.pill{font-size:12px;background:var(--lime);padding:3px 8px;border:1px solid var(--ink);border-radius:99px}.state{font:12px ui-monospace,SFMono-Regular,monospace;color:var(--muted)}h2{font-size:21px;line-height:1.18;margin:0}.source{color:var(--muted);font-size:13px}.source a{color:var(--green)}.checks{font-size:12px;color:var(--red);white-space:pre-wrap}.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px}.actions .skip{background:var(--red)}.schedule{display:grid;gap:5px;color:var(--muted);font-size:12px}.schedule input{width:100%;border:1px solid var(--line);background:white;color:var(--ink);padding:9px 10px;font:14px inherit}button{margin-top:auto;border:1px solid var(--ink);background:var(--green);color:white;padding:13px 16px;font:700 15px inherit;cursor:pointer;box-shadow:3px 3px 0 var(--ink)}button:hover{transform:translate(-1px,-1px);box-shadow:4px 4px 0 var(--ink)}button:disabled{background:#d2d1c9;color:#777;cursor:not-allowed;box-shadow:none}.empty{padding:64px 0;color:var(--muted);font-size:20px}.toast{position:fixed;right:24px;bottom:24px;max-width:430px;padding:14px 18px;background:var(--ink);color:white;display:none;z-index:9}.pause{color:var(--red);font-weight:700}@media(max-width:720px){.shell{padding:20px}.top{display:block}.note{margin-top:18px}.stats{grid-template-columns:1fr}.card video,.missing{max-height:70vh}}
</style></head><body><main class="shell"><header class="top"><div><div class="eyebrow">Human review queue</div><h1>成片审核台</h1></div><div class="note">只显示尚未发布到视频号的成片。先完整预览，再逐条确认发布。<span class="pause">Bilibili 已暂停</span>，不会出现在发布动作中。</div></header><nav class="tabs"><button class="active" data-panel="publish">成片队列</button><button data-panel="failures">失败来源 <span id="failure-count"></span></button></nav><section id="publish" class="panel"><section class="stats"><div class="stat"><span>待处理</span><b id="total">—</b></div><div class="stat"><span>可发布</span><b id="ready">—</b></div><div class="stat"><span>需人工排查</span><b id="attention">—</b></div></section><section id="grid" class="grid"></section></section><section id="failures" class="panel" hidden><section id="failure-grid" class="grid"></section></section></main><div id="toast" class="toast"></div>
<script>
let csrf='';const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function toast(message){const el=document.querySelector('#toast');el.textContent=message;el.style.display='block';setTimeout(()=>el.style.display='none',5000)}
async function load(){const r=await fetch('/api/queue',{cache:'no-store'});const data=await r.json();csrf=data.csrf_token;for(const k of ['total','ready','attention'])document.querySelector('#'+k).textContent=data.summary[k];const grid=document.querySelector('#grid');if(!data.items.length){grid.innerHTML='<div class="empty">队列已清空。下一轮发现与生成完成后，新成片会自动出现在这里。</div>';return}grid.innerHTML=data.items.map(x=>`<article class="card">${x.video_available?`<video controls preload="metadata" src="${esc(x.media_url)}"></video>`:'<div class="missing">成片文件已被清理，需重新生成后才能发布</div>'}<div class="meta"><span>${x.sequence_label?`<span class="pill">短片 ${esc(x.sequence_label)}</span> `:''}<span class="pill">${esc(x.editorial_mode||'news')}</span></span><span class="state">${esc(x.item_state)}</span></div><h2>${esc(x.display_title||x.title)}</h2><div class="source">${x.source_url?`来源：<a href="${esc(x.source_url)}" target="_blank" rel="noreferrer">${esc(x.source_title||x.source_url)}</a>`:'来源已归档'}</div>${x.failed_checks.length?`<div class="checks">未通过：${esc(x.failed_checks.join('、'))}</div>`:''}${x.last_error?`<div class="checks">上次错误：${esc(x.last_error)}</div>`:''}${x.can_publish?`<label class="schedule">定时发布（北京时间，至少提前 2 小时；留空则立即发布）<input type="datetime-local" value="${esc((x.schedule_at||'').replace(' ','T'))}"></label>`:''}<button ${(x.can_publish||x.can_review)?'':'disabled'} data-action="${x.can_review?'review':x.requires_login?'login':'publish'}" data-review-check="${esc(x.review_check)}" data-batch="${esc(x.batch_id)}" data-item="${esc(x.item_id)}">${esc(x.action_label)}</button></article>`).join('');grid.querySelectorAll('button:not(:disabled)').forEach(b=>b.addEventListener('click',handleAction))}
async function loadFailures(){const r=await fetch('/api/discovery-failures',{cache:'no-store'});const data=await r.json();csrf=data.csrf_token||csrf;document.querySelector('#failure-count').textContent=data.items.length?`(${data.items.length})`:'';const grid=document.querySelector('#failure-grid');if(!data.items.length){grid.innerHTML='<div class="empty">当前没有需要处理的失败来源。</div>';return}grid.innerHTML=data.items.map(x=>{const task=x.retry_task||{};const busy=['queued','running'].includes(task.status);const done=task.status==='succeeded';const logs=(task.logs||[]).map(row=>`${row.at||''}  ${row.message||''}`).join('\\n');const actions=done?'<div class="source">已生成，稍后会从失败来源列表移除。</div>':`<div class="actions"><button ${busy?'disabled':''} data-retry="${esc(x.candidate_id)}">${busy?'Retry '+esc(task.status):'Retry Now'}</button><button class="skip" ${busy?'disabled':''} data-skip="${esc(x.candidate_id)}">Skip</button></div>`;return `<article class="card"><div class="meta"><span class="pill">${esc(x.channel||'discovery')}</span><span class="state">${esc(task.status||x.status)}</span></div><h2>${esc(x.title||x.candidate_id)}</h2><div class="source"><a href="${esc(x.url)}" target="_blank" rel="noreferrer">打开来源</a> · score ${esc(x.score||'—')} · attempts ${esc(task.attempts??x.attempts??0)}</div><div>${esc(task.stage||x.stage||'generation')} / ${esc(x.category||'unknown')}</div>${task.started_at?`<div class="source">开始：${esc(task.started_at)}</div>`:''}${x.cue_id?`<div>字幕卡：${esc(x.cue_id)}</div>`:''}${x.source_range?`<div>范围：${esc(x.source_range.start)}–${esc(x.source_range.end)}s (${esc(x.source_range.duration)}s)</div>`:''}<div class="checks">${esc(task.last_error||x.last_error||x.reason||'未记录详细错误')}</div>${logs?`<div class="checks">${esc(logs)}</div>`:''}${x.next_retry_at?`<div class="source">下次重试：${esc(x.next_retry_at)}</div>`:''}${actions}</article>`}).join('');grid.querySelectorAll('[data-retry]:not(:disabled)').forEach(b=>b.addEventListener('click',retryFailure));grid.querySelectorAll('[data-skip]:not(:disabled)').forEach(b=>b.addEventListener('click',skipFailure))}
async function retryFailure(e){const id=e.currentTarget.dataset.retry;e.currentTarget.disabled=true;e.currentTarget.textContent='Retry queued';try{const r=await fetch('/api/discovery-retry',{method:'POST',headers:{'Content-Type':'application/json','X-Video-Factory-CSRF':csrf},body:JSON.stringify({candidate_id:id})});const data=await r.json();if(!r.ok)throw new Error(data.error||'重试失败');toast('已加入重试队列');await loadFailures()}catch(err){toast(err.message);e.currentTarget.disabled=false;e.currentTarget.textContent='Retry Now'}}
async function skipFailure(e){const id=e.currentTarget.dataset.skip;const reason=prompt('请输入跳过原因（必填）');if(!reason||!reason.trim())return;try{const r=await fetch('/api/discovery-skip',{method:'POST',headers:{'Content-Type':'application/json','X-Video-Factory-CSRF':csrf},body:JSON.stringify({candidate_id:id,reason:reason.trim()})});const data=await r.json();if(!r.ok)throw new Error(data.error||'跳过失败');toast('已跳过该来源');await loadFailures()}catch(err){toast(err.message)}}
async function handleAction(e){const b=e.currentTarget;if(b.dataset.action==='review')return reviewOne(b);if(b.dataset.action==='login')return loginOne(b);return publishOne(e)}
async function reviewOne(b){const rights=b.dataset.reviewCheck==='rights_review';const message=rights?'确认你已完整观看这段访谈，并核对来源署名、片段范围与本次复用依据？此操作只解除 rights_review 门禁，不会发布。':'确认你已完整观看：内容仅从 AI 安全、可解释性与防御研究角度呈现，不提供滥用操作指导？此操作只解除安全人工审核门禁，不会发布。';if(!confirm(message))return;b.disabled=true;b.textContent='正在记录审核…';try{const r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Video-Factory-CSRF':csrf},body:JSON.stringify({batch_id:b.dataset.batch,item_id:b.dataset.item})});const data=await r.json();if(!r.ok)throw new Error(data.error||'审核记录失败');toast('人工审核已记录；请再次确认后发布');await load()}catch(err){toast(err.message);b.disabled=false;b.textContent='重试审核'}}
async function loginOne(b){if(!confirm('视频号登录已失效。现在打开受管登录流程，登录成功后继续发布这一条视频？'))return;b.disabled=true;b.textContent='等待视频号登录…';try{const r=await fetch('/api/login-and-publish',{method:'POST',headers:{'Content-Type':'application/json','X-Video-Factory-CSRF':csrf},body:JSON.stringify({batch_id:b.dataset.batch,item_id:b.dataset.item})});const data=await r.json();if(!r.ok)throw new Error(data.error||'登录恢复失败');toast(data.published?'登录已恢复，发布完成':'登录或发布未完成：'+(data.error||data.item_state));await load()}catch(err){toast(err.message);b.disabled=false;b.textContent='重试登录恢复'}}
async function publishOne(e){const b=e.currentTarget;const input=b.closest('.card').querySelector('.schedule input');const raw=input?input.value:'';const schedule=raw?raw.replace('T',' '):null;const action=schedule?`定时到 ${schedule}（北京时间）`:'立即发布';if(!confirm(`确认已完整审核这条视频、标题和来源，并${action}到视频号？`))return;b.disabled=true;b.textContent='正在检查账号并提交…';try{const r=await fetch('/api/publish',{method:'POST',headers:{'Content-Type':'application/json','X-Video-Factory-CSRF':csrf},body:JSON.stringify({batch_id:b.dataset.batch,item_id:b.dataset.item,schedule_at:schedule})});const data=await r.json();if(!r.ok)throw new Error(data.error||'发布失败');toast(data.published?(schedule?'定时发布已提交':'发布完成'):data.requires_login?'登录已失效；请点击登录并继续发布':'提交未成功：'+(data.error||data.item_state));await load()}catch(err){toast(err.message);b.disabled=false;b.textContent='重试发布'}}
document.querySelectorAll('.tabs button').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.panel').forEach(x=>x.hidden=x.id!==b.dataset.panel)}));Promise.all([load(),loadFailures()]).catch(err=>toast('队列加载失败：'+err.message));setInterval(loadFailures,5000);setInterval(load,60000);
</script></body></html>"""
