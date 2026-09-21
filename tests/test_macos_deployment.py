from __future__ import annotations

import plistlib
import stat
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MACOS = ROOT / "deploy" / "macos"


class MacOSDeploymentTests(unittest.TestCase):
    def test_nightly_self_audit_is_bounded_and_does_not_publish(self) -> None:
        runner = MACOS / "run-self-audit.zsh"
        payload = runner.read_text(encoding="utf-8")
        self.assertTrue(runner.stat().st_mode & stat.S_IXUSR)
        subprocess.run(["zsh", "-n", str(runner)], check=True)
        self.assertIn("self-audit run", payload)
        self.assertIn("--max-issues 5", payload)
        self.assertIn("--max-cost-usd 1.0", payload)
        self.assertNotIn("publish-run", payload)

    def test_nightly_self_audit_runs_at_0315_local_time(self) -> None:
        with (MACOS / "com.clairehou.video-factory.self-audit.plist").open("rb") as handle:
            config = plistlib.load(handle)
        self.assertEqual(config["StartCalendarInterval"], {"Hour": 3, "Minute": 15})
        self.assertNotIn("RunAtLoad", config)
        self.assertNotIn("KeepAlive", config)


if __name__ == "__main__":
    unittest.main()
