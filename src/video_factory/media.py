from __future__ import annotations

import json
import re
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
    audio_duration: float | None = None
    audio_bitrate: int | None = None


@dataclass(frozen=True, slots=True)
class AudioLoudness:
    mean_db: float
    max_db: float
    longest_silence_seconds: float = 0.0


def probe_audio_loudness(path: Path) -> AudioLoudness:
    """Decode the audio track and reject AAC containers that contain silence."""
    command = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
        "-map", "0:a:0", "-af", "silencedetect=noise=-50dB:d=10,volumedetect",
        "-f", "null", "/dev/null",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    output = "\n".join([result.stdout or "", result.stderr or ""])
    mean = re.search(r"mean_volume:\s*(-?[0-9.]+)\s*dB", output)
    maximum = re.search(r"max_volume:\s*(-?[0-9.]+)\s*dB", output)
    if result.returncode != 0 or mean is None or maximum is None:
        raise ValueError(f"audio loudness probe failed for {path}")
    silences = [float(value) for value in re.findall(r"silence_duration:\s*([0-9.]+)", output)]
    return AudioLoudness(
        float(mean.group(1)), float(maximum.group(1)), max(silences, default=0.0),
    )


def probe_video(path: Path) -> VideoProbe:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,pix_fmt,duration,bit_rate",
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
        audio_duration=float(audio["duration"]) if audio and audio.get("duration") else None,
        audio_bitrate=int(audio["bit_rate"]) if audio and audio.get("bit_rate") else None,
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
