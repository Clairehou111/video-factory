import json
import os
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory import cli
from video_factory.storage import Workspace
from video_factory.workspace_cleanup import WorkspaceCleanup


NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
OLD = "2026-09-01T00:00:00Z"
RECENT = "2026-10-01T00:00:00Z"


class WorkspaceCleanupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(Path(self.temp.name) / "workspace")
        self.workspace.initialize()

    def _job(
        self, name: str, *, status: str = "failed", started_at: str = OLD,
        url: str = "https://example.com/source", files: dict[str, bytes] | None = None,
    ) -> Path:
        job = self.workspace.root / "jobs" / name
        job.mkdir(parents=True)
        (job / "result.json").write_text(json.dumps({
            "job_id": name, "status": status, "started_at": started_at, "url": url,
        }), encoding="utf-8")
        for relative, content in (files or {"intermediate.mp4": b"video"}).items():
            path = job / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return job

    def _cleaner(self, retention_days: int = 14) -> WorkspaceCleanup:
        return WorkspaceCleanup(
            self.workspace, retention_days=retention_days, clock=lambda: NOW,
        )

    def test_dry_run_only_selects_old_unreferenced_failed_jobs(self) -> None:
        old = self._job("old-failed")
        recent = self._job("recent-failed", started_at=RECENT)
        completed = self._job("completed", status="completed")
        running = self._job("running", status="running")

        report = self._cleaner().plan()

        self.assertEqual(report["mode"], "dry_run")
        self.assertEqual(
            [item["path"] for item in report["candidates"]], ["jobs/old-failed"],
        )
        self.assertGreater(report["candidate_bytes"], 0)
        for path in (old, recent, completed, running):
            self.assertTrue(path.is_dir())

    def test_apply_removes_candidate_and_is_idempotent(self) -> None:
        old = self._job("old-failed")

        first = self._cleaner().apply()
        second = self._cleaner().apply()

        self.assertFalse(old.exists())
        self.assertEqual(first["deleted_count"], 1)
        self.assertGreater(first["deleted_bytes"], 0)
        self.assertEqual(first["errors"], [])
        self.assertEqual(second["deleted_count"], 0)

    def test_publish_and_audit_references_protect_entire_jobs(self) -> None:
        published = self._job("published")
        audited = self._job("audited")
        publish_dir = self.workspace.root / "publish" / "batch-1"
        publish_dir.mkdir(parents=True)
        (publish_dir / "batch.json").write_text(json.dumps({
            "video_path": str(published / "intermediate.mp4"),
        }), encoding="utf-8")
        audit_dir = self.workspace.root / "automation" / "self-audit"
        audit_dir.mkdir(parents=True)
        (audit_dir / "problems.json").write_text(json.dumps({
            "problem": {"artifact_refs": ["jobs/audited/result.json"]},
        }), encoding="utf-8")

        report = self._cleaner().plan()

        self.assertEqual(report["candidate_count"], 0)
        protected = {item["path"]: item["reason"] for item in report["protected"]}
        self.assertIn("publish batch", protected["jobs/published"])
        self.assertIn("self-audit", protected["jobs/audited"])

    def test_latest_recovery_job_for_retryable_candidate_is_protected(self) -> None:
        url = "https://example.com/retry-me"
        older = self._job("retry-older", url=url, files={"manifest.json": b"{}"})
        latest = self._job("retry-latest", url=url, files={"translation-plan.json": b"{}"})
        os.utime(older / "result.json", (1, 1))
        os.utime(latest / "result.json", (2, 2))
        self.workspace.save_discovery_candidate({
            "id": "candidate-1", "url": url, "status": "needs_human",
        })

        report = self._cleaner().plan()

        self.assertEqual(
            [item["path"] for item in report["candidates"]], ["jobs/retry-older"],
        )
        protected = {item["path"]: item["reason"] for item in report["protected"]}
        self.assertIn("latest recoverable job", protected["jobs/retry-latest"])

    def test_invalid_timestamp_and_symlink_are_never_candidates(self) -> None:
        invalid = self._job("invalid-time", started_at="not-a-time")
        target = self._job("symlink-target")
        link = self.workspace.root / "jobs" / "linked-job"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            self.skipTest("directory symlinks are unavailable")

        report = self._cleaner().plan()

        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["candidates"][0]["path"], "jobs/symlink-target")
        self.assertTrue(invalid.is_dir())
        self.assertTrue(link.is_symlink())

    def test_custom_retention_changes_boundary(self) -> None:
        old = self._job("old-failed")

        report = self._cleaner(retention_days=60).plan()

        self.assertEqual(report["candidate_count"], 0)
        self.assertTrue(old.is_dir())

    def test_apply_rejects_a_target_replaced_by_a_symlink(self) -> None:
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        link = self.workspace.root / "jobs" / "old-failed"
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("directory symlinks are unavailable")
        fake_report = {
            "mode": "dry_run", "workspace": str(self.workspace.root),
            "retention_days": 14, "cutoff": "", "candidate_count": 1,
            "candidate_files": 1, "candidate_bytes": 10,
            "candidates": [{
                "path": "jobs/old-failed", "status": "failed", "started_at": OLD,
                "age_days": 30, "file_count": 1, "bytes": 10, "reason": "old",
            }],
            "protected": [], "untouched": [],
        }

        with patch.object(WorkspaceCleanup, "plan", return_value=fake_report):
            report = self._cleaner().apply()

        self.assertEqual(report["deleted_count"], 0)
        self.assertIn("symlink", report["errors"][0]["error"])
        self.assertTrue(outside.is_dir())

    def test_cli_is_dry_run_by_default(self) -> None:
        old = self._job("old-failed")
        output = StringIO()
        with patch("sys.argv", [
            "video-factory", "--workspace", str(self.workspace.root),
            "cleanup", "--retention-days", "14",
        ]):
            with redirect_stdout(output):
                cli.main()

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["mode"], "dry_run")
        self.assertTrue(old.is_dir())


if __name__ == "__main__":
    unittest.main()
