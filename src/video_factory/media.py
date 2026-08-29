from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VideoProbe:
    path: Path
    duration: float
    width: int
    height: int
    video_codec: str
    pixel_format: str
    audio_codec: str | None


def probe_video(path: Path) -> VideoProbe:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,pix_fmt",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    audio = next((stream for stream in payload["streams"] if stream["codec_type"] == "audio"), None)
    return VideoProbe(
        path=path,
        duration=float(payload["format"]["duration"]),
        width=int(video["width"]),
        height=int(video["height"]),
        video_codec=video["codec_name"],
        pixel_format=video["pix_fmt"],
        audio_codec=audio["codec_name"] if audio else None,
    )


def validate_wechat_mp4(
    probe: VideoProbe, max_duration: float | None = None, require_audio: bool = True,
) -> list[dict[str, object]]:
    """Validate either a silent visual track or a final WeChat-ready render."""
    checks = [
        {"name": "resolution", "passed": (probe.width, probe.height) == (1080, 1920), "detail": f"{probe.width}x{probe.height}"},
        {"name": "h264", "passed": probe.video_codec == "h264", "detail": probe.video_codec},
        {"name": "yuv420p", "passed": probe.pixel_format == "yuv420p", "detail": probe.pixel_format},
    ]
    if require_audio:
        checks.append({"name": "aac", "passed": probe.audio_codec == "aac", "detail": probe.audio_codec or "missing"})
    if max_duration is not None:
        checks.append({
            "name": "duration", "passed": probe.duration <= max_duration,
            "detail": f"{probe.duration:.2f}s / max {max_duration:.2f}s",
        })
    return checks
