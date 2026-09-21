from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.cli import main


class SelfAuditCLITest(unittest.TestCase):
    def run_cli(self, *arguments: str) -> dict:
        output = io.StringIO()
        with patch("sys.argv", ["video-factory", *arguments]), redirect_stdout(output):
            main()
        return json.loads(output.getvalue())

    def test_problem_note_status_and_archived_replay_need_no_model(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            artifact = workspace / "jobs" / "job-1" / "manifest.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text('{"fixed_title":"Google loses another AI team"}', encoding="utf-8")
            base = ("--workspace", str(workspace))

            note = self.run_cli(
                *base, "problem-note", "--stage", "generation",
                "--category", "story_axis",
                "--expected", "Google remains the recognizable lead",
                "--observed", "a secondary mechanism replaced the event",
                "--severity", "high", "--artifact", str(artifact),
            )
            status = self.run_cli(*base, "self-audit", "status")
            replay = self.run_cli(*base, "self-audit", "replay", note["id"])

            self.assertEqual(status["total"], 1)
            self.assertEqual(replay["mode"], "archived_only")
            self.assertFalse(replay["external_actions_allowed"])
            self.assertIn("fixed_title", replay["assets"][0]["text"])


if __name__ == "__main__":
    unittest.main()
