from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from math import ceil
from dataclasses import dataclass
from pathlib import Path

from .models import RenderManifest
from .media import probe_video


@dataclass(frozen=True, slots=True)
class MPTSettings:
    root: Path
    python: Path
    bgm_file: str = "output013.mp3"
    voice_name: str = "zh-CN-YunxiNeural-Male"
    voice_rate: float = 1.7
    video_clip_duration: int = 60
    audio_fade_seconds: float = 0.8

    @classmethod
    def from_environment(cls) -> "MPTSettings":
        root = Path(os.environ.get("MPT_ROOT", "../MoneyPrinterTurbo")).resolve()
        configured_bgm = os.environ.get("VIDEO_FACTORY_BGM_FILE", "").strip()
        if configured_bgm:
            bgm_file = str(Path(configured_bgm).expanduser().resolve())
        else:
            preferred = Path("workspace/assets/music/858ccdf31193/better-times-are-coming-mixkit-173.mp3").resolve()
            bgm_file = str(preferred) if preferred.is_file() else "output013.mp3"
        return cls(
            root=root,
            python=Path(os.environ.get("MPT_PYTHON", root / ".venv/bin/python")),
            bgm_file=bgm_file,
        )


class MPTAssemblyAdapter:
    """MPT is a BGM/encoding backend, never the owner of facts or scene planning."""

    def __init__(self, settings: MPTSettings):
        self.settings = settings

    def build_command(
        self, manifest: RenderManifest, native_track: Path, task_id: str | None = None,
        bgm_file: str | None = None,
    ) -> list[str]:
        if native_track.suffix.lower() != ".mp4":
            raise ValueError("MPT input must be a precomposed native MP4; direct images trigger unwanted zoom.")
        if not manifest.fixed_footer or manifest.footer_shows_source_url:
            raise ValueError("MPT assembly requires a fixed conclusion footer and no source URL in the picture.")
        if not all(scene.end > scene.start for scene in manifest.scenes):
            raise ValueError("all scenes need valid timing before assembly")
        # MPT currently needs text to size its no-voice silent-audio placeholder.
        # Never pass editorial text here: MPT treats the substring "Error:"
        # as a task failure, and this string must not become a second owner of
        # on-screen copy.  Each neutral sentence is ~1.29 seconds under MPT's
        # own estimator.  Leave a small tail margin: MPT rounds its silent
        # audio duration upward and otherwise loops the first browser segment.
        # A GitHub master may include a dedicated cold open before its browser
        # scenes. Size MPT's neutral silent track from the actual native video,
        # not only from the browser-scene timeline in the manifest.
        native_duration = probe_video(native_track).duration if native_track.is_file() else manifest.duration
        script = "画面节奏。" * ceil(max(native_duration - 1.5, 3.0) / 1.29)
        if task_id:
            try:
                task_id = str(uuid.UUID(task_id))
            except ValueError:
                task_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"video-factory:{task_id}"))
        else:
            task_id = str(uuid.uuid4())
        return [
            str(self.settings.python), "cli.py",
            "--video-script", script,
            "--video-language", "zh-CN",
            "--video-source", "local",
            "--video-materials", str(native_track.resolve()),
            "--video-aspect", "9:16",
            "--video-concat-mode", "sequential",
            "--video-transition-mode", "none",
            "--video-clip-duration", str(self.settings.video_clip_duration),
            "--voice-name", "no-voice",
            "--bgm-type", "custom",
            "--bgm-file", bgm_file or self.settings.bgm_file,
            "--bgm-volume", "0.05",
            "--no-subtitle-enabled",
            "--task-id", task_id,
            "--stop-at", "video",
        ]

    def assemble(self, manifest: RenderManifest, native_track: Path, task_id: str | None = None) -> Path:
        staged_bgm = self._stage_bgm()
        command = self.build_command(manifest, native_track, task_id, staged_bgm)
        subprocess.run(command, cwd=self.settings.root, check=True)
        resolved_task_id = command[command.index("--task-id") + 1]
        result = self.settings.root / "storage/tasks" / resolved_task_id / "final-1.mp4"
        if not result.is_file():
            raise RuntimeError(f"MPT reported success but did not produce {result}")
        # MPT uses a silent-audio placeholder to drive its older timeline. Its
        # rounded duration can trim the last visual second. Keep MPT's BGM mix,
        # but rebuild the final music bed from the licensed source so it spans
        # the entire native visual before fading. MPT's rounded 7.5-second mix
        # otherwise ends before an 8.4-second visual, making a tail fade inert.
        mastered = result.with_name("final-native-visual-master.mp4")
        native_duration = probe_video(native_track).duration
        music_input = Path(staged_bgm).expanduser()
        if not music_input.is_absolute():
            music_input = self.settings.root / music_input
        if not music_input.is_file():
            music_input = result
        subprocess.run([
            "ffmpeg", "-y", "-i", str(native_track),
            "-stream_loop", "-1", "-i", str(music_input),
            "-filter_complex", self._master_audio_filter(native_duration),
            "-map", "0:v:0", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(mastered),
        ], check=True)
        return mastered

    def _master_audio_filter(self, duration: float) -> str:
        fade = min(self.settings.audio_fade_seconds, max(duration / 2, 0.1))
        start = max(duration - fade, 0.0)
        return (
            f"[1:a]volume=0.05,atrim=duration={duration:.3f},asetpts=N/SR/TB,"
            f"afade=t=out:st={start:.3f}:d={fade:.3f}[a]"
        )

    def _stage_bgm(self) -> str:
        """Copy a project-owned track into MPT's managed BGM boundary."""
        source = Path(self.settings.bgm_file).expanduser()
        if not source.is_file():
            return self.settings.bgm_file
        managed_dir = self.settings.root / "storage" / "bgm"
        managed_dir.mkdir(parents=True, exist_ok=True)
        destination = managed_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return str(destination.resolve())


class NativeFFmpegAssemblyAdapter:
    """Master the native visual directly; MPT remains an emergency fallback only."""

    def __init__(self, settings: MPTSettings):
        self.settings = settings

    def build_command(
        self, manifest: RenderManifest, native_track: Path, output: Path,
        bgm_file: str | None = None,
    ) -> list[str]:
        if native_track.suffix.lower() != ".mp4":
            raise ValueError("native assembly input must be an MP4")
        if not manifest.fixed_footer or manifest.footer_shows_source_url:
            raise ValueError("native assembly requires a fixed conclusion footer and no source URL in the picture")
        if not all(scene.end > scene.start for scene in manifest.scenes):
            raise ValueError("all scenes need valid timing before assembly")
        music = Path(bgm_file or self.settings.bgm_file).expanduser()
        if not music.is_absolute():
            music = music.resolve()
        if not music.is_file():
            raise FileNotFoundError(f"native assembly BGM does not exist: {music}")
        duration = probe_video(native_track).duration
        output.parent.mkdir(parents=True, exist_ok=True)
        return [
            "ffmpeg", "-y", "-i", str(native_track),
            "-stream_loop", "-1", "-i", str(music),
            "-filter_complex", self._master_audio_filter(duration),
            "-map", "0:v:0", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(output),
        ]

    def assemble(
        self, manifest: RenderManifest, native_track: Path, output: Path,
        bgm_file: str | None = None,
    ) -> Path:
        command = self.build_command(manifest, native_track, output, bgm_file)
        subprocess.run(command, check=True)
        if not output.is_file():
            raise RuntimeError(f"FFmpeg reported success but did not produce {output}")
        return output

    def _master_audio_filter(self, duration: float) -> str:
        fade = min(self.settings.audio_fade_seconds, max(duration / 2, 0.1))
        start = max(duration - fade, 0.0)
        return (
            f"[1:a]volume=0.05,atrim=duration={duration:.3f},asetpts=N/SR/TB,"
            f"afade=t=out:st={start:.3f}:d={fade:.3f}[a]"
        )
