from __future__ import annotations

import subprocess
import stat
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OCI = ROOT / "deploy" / "oci"


class OracleDeploymentTests(unittest.TestCase):
    def test_shell_scripts_are_valid_bash(self) -> None:
        scripts = sorted(OCI.glob("*.sh"))
        self.assertTrue(scripts)
        for path in scripts:
            self.assertTrue(path.stat().st_mode & stat.S_IXUSR, f"not executable: {path}")
        subprocess.run(["bash", "-n", *(str(path) for path in scripts)], check=True)

    def test_environment_uses_persistent_linux_paths_without_secrets(self) -> None:
        payload = (OCI / "video-factory.env.example").read_text(encoding="utf-8")
        required = (
            "VIDEO_FACTORY_WORKSPACE=/srv/video-factory/workspace",
            "VIDEO_FACTORY_SAU_HOME=/srv/video-factory/runtime/social-auto-upload",
            "MPT_ROOT=/opt/video-factory/MoneyPrinterTurbo",
            "WEB_SCROLL_VIDEO_ROOT=/opt/video-factory/web-scroll-video",
            "VIDEO_FACTORY_FONT=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        )
        for line in required:
            self.assertIn(line, payload)
        self.assertNotIn("sk-", payload)
        self.assertNotIn("/Users/", payload)
        self.assertIn("VIDEO_FACTORY_DISCOVERY_CHANNELS=github,official,paper,news,openrouter", payload)

    def test_systemd_units_do_not_publish(self) -> None:
        units = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((OCI / "systemd").iterdir())
        )
        self.assertIn("run-discovery.sh", units)
        self.assertIn("backup.sh", units)
        self.assertNotIn("publish-run", units)
        self.assertNotIn("login-desktop.sh", units)

    def test_dependency_pins_match_runtime_contract(self) -> None:
        bootstrap = (OCI / "bootstrap.sh").read_text(encoding="utf-8")
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        pins = (
            "d4c0e45da4ac0889af77f7307f52f9d5d4f74942",
            "7c004aefb8ec4610a18ad21577105a9ddce60b15",
            "1c66b7db4b30585bbb40c58eb0aa572ffa3cce97",
        )
        for pin in pins[:2]:
            self.assertIn(pin, bootstrap)
        for pin in pins:
            self.assertIn(pin, notices)

    def test_bootstrap_clears_future_chrome_path_before_managed_install(self) -> None:
        bootstrap = (OCI / "bootstrap.sh").read_text(encoding="utf-8")
        setup = bootstrap.split("publisher setup", 1)[0].rsplit("runuser", 1)[-1]
        self.assertIn("VIDEO_FACTORY_CHROME_PATH=", setup)
        self.assertIn("CHROME_PATH=", setup)


if __name__ == "__main__":
    unittest.main()
