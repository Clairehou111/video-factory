from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

from .discovery import same_source
from .storage import Workspace


RETRYABLE_CANDIDATE_STATUSES = {"blocked", "retry_wait", "retry_pending", "needs_human"}


@dataclass(frozen=True, slots=True)
class CleanupCandidate:
    path: str
    status: str
    started_at: str
    age_days: int
    file_count: int
    bytes: int
    reason: str


class WorkspaceCleanup:
    """Plan and apply conservative cleanup of immutable failed job directories."""

    def __init__(
        self,
        workspace: Workspace,
        retention_days: int = 14,
        clock: Callable[[], datetime] | None = None,
    ):
        if retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        self.workspace = workspace
        self.retention_days = retention_days
        self.clock = clock or (lambda: datetime.now(UTC))

    def plan(self) -> dict[str, Any]:
        now = self.clock().astimezone(UTC)
        cutoff = now - timedelta(days=self.retention_days)
        protected = self._protected_jobs()
        candidates: list[CleanupCandidate] = []
        skipped: list[dict[str, Any]] = []
        jobs_root = self.workspace.root / "jobs"

        if jobs_root.is_dir():
            for job in sorted(jobs_root.iterdir(), key=lambda path: path.name):
                if not job.is_dir() or job.is_symlink():
                    continue
                result_path = job / "result.json"
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if result.get("status") != "failed":
                    continue
                started_raw = str(result.get("started_at") or "")
                started = _parse_timestamp(started_raw)
                if started is None:
                    skipped.append({
                        "path": str(job.relative_to(self.workspace.root)),
                        "reason": "failed job has no valid started_at timestamp",
                    })
                    continue
                if started >= cutoff:
                    continue
                relative = str(job.relative_to(self.workspace.root))
                reasons = sorted(protected.get(job.resolve(), set()))
                if reasons:
                    skipped.append({"path": relative, "reason": "; ".join(reasons)})
                    continue
                file_count, size = _tree_usage(job)
                candidates.append(CleanupCandidate(
                    path=relative,
                    status="failed",
                    started_at=started_raw,
                    age_days=max(0, (now - started).days),
                    file_count=file_count,
                    bytes=size,
                    reason=f"unreferenced failed job older than {self.retention_days} days",
                ))

        candidate_rows = [asdict(candidate) for candidate in candidates]
        return {
            "mode": "dry_run",
            "workspace": str(self.workspace.root),
            "retention_days": self.retention_days,
            "cutoff": cutoff.isoformat().replace("+00:00", "Z"),
            "candidate_count": len(candidate_rows),
            "candidate_files": sum(item["file_count"] for item in candidate_rows),
            "candidate_bytes": sum(item["bytes"] for item in candidate_rows),
            "candidates": candidate_rows,
            "protected": skipped,
            "untouched": [
                "completed and running jobs",
                "failed jobs inside the retention window",
                "canonical assets, manifests, renders, and caches",
                "discovery history, SQLite records, logs, and automation history",
            ],
        }

    def apply(self) -> dict[str, Any]:
        # Recompute immediately before deletion so a newly recorded publish or
        # audit reference cannot be removed based on an older dry-run snapshot.
        report = self.plan()
        deleted: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        jobs_root = (self.workspace.root / "jobs").resolve()

        for candidate in report["candidates"]:
            relative = Path(str(candidate["path"]))
            path = self.workspace.root / relative
            try:
                if relative.parts[:1] != ("jobs",) or len(relative.parts) != 2:
                    raise ValueError("cleanup target is not a direct child of workspace/jobs")
                if path.is_symlink():
                    raise ValueError("cleanup target is a symlink")
                if path.parent.resolve() != jobs_root:
                    raise ValueError("cleanup target escapes workspace/jobs")
                if not path.is_dir():
                    raise FileNotFoundError(path)
                shutil.rmtree(path)
                deleted.append(candidate)
            except (OSError, ValueError) as error:
                errors.append({"path": str(relative), "error": f"{type(error).__name__}: {error}"})

        deleted_bytes = sum(int(item["bytes"]) for item in deleted)
        return {
            **report,
            "mode": "apply",
            "deleted_count": len(deleted),
            "deleted_bytes": deleted_bytes,
            "deleted": deleted,
            "errors": errors,
        }

    def _protected_jobs(self) -> dict[Path, set[str]]:
        protected: dict[Path, set[str]] = defaultdict(set)
        publish_root = self.workspace.root / "publish"
        if publish_root.is_dir():
            for batch in publish_root.glob("*/batch.json"):
                try:
                    payload = json.loads(batch.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for value in _iter_strings(payload):
                    self._protect_reference(protected, value, "referenced by a publish batch")

        audit_root = self.workspace.root / "automation" / "self-audit"
        audit_index = audit_root / "problems.json"
        if audit_index.is_file():
            try:
                payload = json.loads(audit_index.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            for reference in _artifact_references(payload):
                self._protect_reference(protected, reference, "referenced by self-audit evidence")
        for observations in (
            audit_root / "observations.jsonl",
            self.workspace.root / "automation" / "problems.jsonl",
        ):
            if not observations.is_file():
                continue
            try:
                lines = observations.read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = []
            for line in lines:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for reference in _artifact_references(payload):
                    self._protect_reference(protected, reference, "referenced by self-audit evidence")

        self._protect_retry_artifacts(protected)
        return protected

    def _protect_reference(
        self, protected: dict[Path, set[str]], reference: str, reason: str,
    ) -> None:
        value = Path(reference)
        path = value if value.is_absolute() else self.workspace.root / value
        try:
            relative = path.resolve().relative_to((self.workspace.root / "jobs").resolve())
        except (OSError, ValueError):
            return
        if not relative.parts:
            return
        job = (self.workspace.root / "jobs" / relative.parts[0]).resolve()
        protected[job].add(reason)

    def _protect_retry_artifacts(self, protected: dict[Path, set[str]]) -> None:
        retry_urls = self._retryable_source_urls()
        if not retry_urls:
            return
        jobs_root = self.workspace.root / "jobs"
        if not jobs_root.is_dir():
            return
        results = sorted(
            jobs_root.glob("*/result.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        remaining = set(retry_urls)
        for result_path in results:
            if not remaining:
                break
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            source_url = str(payload.get("url") or "")
            matches = {url for url in remaining if same_source(source_url, url)}
            if not matches or not _has_recovery_artifact(result_path.parent, payload):
                continue
            protected[result_path.parent.resolve()].add(
                "latest recoverable job for a retryable discovery candidate"
            )
            remaining.difference_update(matches)

    def _retryable_source_urls(self) -> set[str]:
        urls: set[str] = set()
        if self.workspace.db_path.is_file():
            try:
                with closing(sqlite3.connect(self.workspace.db_path)) as db:
                    rows = db.execute(
                        "SELECT payload FROM records WHERE kind = 'discovery_candidate'"
                    ).fetchall()
                for (raw,) in rows:
                    payload = json.loads(raw)
                    if str(payload.get("status") or "") in RETRYABLE_CANDIDATE_STATUSES:
                        if url := str(payload.get("url") or "").strip():
                            urls.add(url)
            except (sqlite3.Error, json.JSONDecodeError):
                pass

        state_path = self.workspace.root / "discovery" / "state.json"
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
            for item in _iter_dicts(state):
                status = str(item.get("status") or "")
                if status in RETRYABLE_CANDIDATE_STATUSES or item.get("retry_eligible") is True:
                    if url := str(item.get("url") or "").strip():
                        urls.add(url)
        return urls


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _tree_usage(root: Path) -> tuple[int, int]:
    file_count = 0
    size = 0
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            try:
                stat = path.lstat()
            except OSError:
                continue
            file_count += 1
            size += stat.st_size
    return file_count, size


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_dicts(item)


def _artifact_references(value: Any) -> Iterable[str]:
    for item in _iter_dicts(value):
        references = item.get("artifact_refs")
        if isinstance(references, list):
            for reference in references:
                if isinstance(reference, str):
                    yield reference


def _has_recovery_artifact(job: Path, result: dict[str, Any]) -> bool:
    for name in ("translation-plan.json", "manifest.json", "collection-manifest.json"):
        if (job / name).is_file():
            return True
    if any((job / "caption-scope-checkpoints").glob("*.json")):
        return True
    manifest = result.get("manifest") or result.get("collection_manifest")
    return bool(manifest and Path(str(manifest)).is_file())
