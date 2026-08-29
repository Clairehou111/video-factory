from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


YTDLP_VERSION = "2026.08.19"
YTDLP_EJS_VERSION = "0.8.0"
BGUTIL_VERSION = "1.3.2"
BGUTIL_REPOSITORY = "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git"


@dataclass(frozen=True, slots=True)
class YouTubeRuntimeSettings:
    runtime_home: Path
    python_version: str = "3.11"

    @classmethod
    def from_environment(cls) -> "YouTubeRuntimeSettings":
        configured = os.environ.get("VIDEO_FACTORY_YOUTUBE_RUNTIME_HOME", "").strip()
        root = Path(configured).expanduser() if configured else Path.home() / ".video-factory" / "youtube-runtime"
        return cls(runtime_home=root.resolve())

    @property
    def venv_dir(self) -> Path:
        return self.runtime_home / "venv"

    @property
    def python(self) -> Path:
        name = "python.exe" if os.name == "nt" else "python"
        folder = "Scripts" if os.name == "nt" else "bin"
        return self.venv_dir / folder / name

    @property
    def executable(self) -> Path:
        name = "yt-dlp.exe" if os.name == "nt" else "yt-dlp"
        folder = "Scripts" if os.name == "nt" else "bin"
        return self.venv_dir / folder / name

    @property
    def provider_source(self) -> Path:
        return self.runtime_home / "bgutil-ytdlp-pot-provider"

    @property
    def provider_server(self) -> Path:
        return self.provider_source / "server"

    @property
    def installation_path(self) -> Path:
        return self.runtime_home / "installation.json"


class ManagedYouTubeRuntime:
    """Pinned yt-dlp + EJS + local PO-token provider runtime."""

    def __init__(self, settings: YouTubeRuntimeSettings | None = None) -> None:
        self.settings = settings or YouTubeRuntimeSettings.from_environment()

    def setup(self) -> dict[str, Any]:
        uv = shutil.which("uv")
        git = shutil.which("git")
        deno = shutil.which("deno")
        if not uv or not git or not deno:
            missing = [name for name, value in (("uv", uv), ("git", git), ("deno", deno)) if not value]
            raise RuntimeError("YouTube runtime setup requires: " + ", ".join(missing))
        self.settings.runtime_home.mkdir(parents=True, exist_ok=True)
        if not self.settings.python.is_file():
            self._checked([uv, "venv", "--python", self.settings.python_version, str(self.settings.venv_dir)])
        self._checked([
            uv, "pip", "install", "--python", str(self.settings.python),
            f"yt-dlp=={YTDLP_VERSION}", f"yt-dlp-ejs=={YTDLP_EJS_VERSION}",
            f"bgutil-ytdlp-pot-provider=={BGUTIL_VERSION}",
        ])
        source = self.settings.provider_source
        if not source.exists():
            self._checked([
                git, "clone", "--filter=blob:none", "--single-branch", "--branch", BGUTIL_VERSION,
                BGUTIL_REPOSITORY, str(source),
            ])
        elif not (source / ".git").is_dir():
            raise RuntimeError(f"managed PO-token provider is not a git checkout: {source}")
        else:
            self._checked([git, "-C", str(source), "fetch", "--depth", "1", "origin", f"refs/tags/{BGUTIL_VERSION}"])
            self._checked([git, "-C", str(source), "checkout", "--detach", BGUTIL_VERSION])
        self._checked([deno, "install", "--allow-scripts=npm:canvas", "--frozen"], cwd=self.settings.provider_server)
        metadata = {
            "yt_dlp": YTDLP_VERSION,
            "yt_dlp_ejs": YTDLP_EJS_VERSION,
            "bgutil_provider": BGUTIL_VERSION,
            "provider_mode": "script",
            "provider_server": str(self.settings.provider_server),
            "deno": self._version([deno, "--version"]),
            "executable": str(self.settings.executable),
        }
        self.settings.installation_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        return metadata

    def status(self) -> dict[str, Any]:
        installed = self.settings.executable.is_file() and self.settings.installation_path.is_file()
        metadata: dict[str, Any] = {
            "installed": installed,
            "runtime_home": str(self.settings.runtime_home),
            "executable": str(self.settings.executable),
            "provider_server": str(self.settings.provider_server),
        }
        if self.settings.installation_path.is_file():
            metadata.update(json.loads(self.settings.installation_path.read_text(encoding="utf-8")))
        metadata["provider_ready"] = self.settings.provider_server.is_dir()
        metadata["version_pins_valid"] = all((
            metadata.get("yt_dlp") == YTDLP_VERSION,
            metadata.get("yt_dlp_ejs") == YTDLP_EJS_VERSION,
            metadata.get("bgutil_provider") == BGUTIL_VERSION,
            metadata.get("provider_mode") == "script",
        ))
        return metadata

    def require_executable(self) -> str:
        if not self.settings.executable.is_file():
            raise RuntimeError("YouTube runtime is not installed; run `video-factory youtube-runtime setup`")
        if not self.status().get("version_pins_valid"):
            raise RuntimeError("YouTube runtime versions drifted; run `video-factory youtube-runtime setup`")
        if not self.settings.provider_server.is_dir() and not os.environ.get("VIDEO_FACTORY_YOUTUBE_PO_TOKEN"):
            raise RuntimeError("YouTube PO-token provider is missing; run `video-factory youtube-runtime setup`")
        return str(self.settings.executable)

    def extractor_arguments(self, context: str = "gvs") -> list[str]:
        token = os.environ.get("VIDEO_FACTORY_YOUTUBE_PO_TOKEN", "").strip()
        youtube = "youtube:player_client=mweb"
        if token:
            youtube += f";po_token=mweb.{context}+{token}"
        arguments = ["--extractor-args", youtube]
        if not token:
            arguments.extend([
                "--extractor-args",
                f"youtubepot-bgutilscript:server_home={self.settings.provider_server}",
            ])
        return arguments

    def installation_metadata(self) -> dict[str, Any]:
        status = self.status()
        return {
            key: status.get(key)
            for key in ("yt_dlp", "yt_dlp_ejs", "bgutil_provider", "provider_mode")
            if status.get(key) is not None
        }

    @staticmethod
    def _checked(command: list[str], cwd: Path | None = None) -> None:
        subprocess.run(command, cwd=cwd, check=True)

    @staticmethod
    def _version(command: list[str]) -> str:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        return completed.stdout.splitlines()[0].strip()
