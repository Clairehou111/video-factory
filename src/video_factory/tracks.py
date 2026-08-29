from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrackSegment:
    path: Path
    duration: float


def build_crossfade_track(segments: list[TrackSegment], output: Path, fade_seconds: float = 0.12) -> Path:
    """Compose the visual timeline before MPT.

    A zero fade is an intentional hard cut. Text-heavy evidence cards must
    use it: a crossfade necessarily renders two readable text layers in the
    same frame and looks like a compositor bug on a phone screen.
    """
    if not segments:
        raise ValueError("at least one native MP4 segment is required")
    if any(segment.path.suffix.lower() != ".mp4" for segment in segments):
        raise ValueError("segments must be pre-rendered MP4 files")
    if len(segments) == 1:
        subprocess.run(["ffmpeg", "-y", "-i", str(segments[0].path), "-c", "copy", str(output)], check=True)
        return output

    inputs = [argument for segment in segments for argument in ("-i", str(segment.path))]
    if fade_seconds <= 0:
        prepared = [
            f"[{index}:v]settb=AVTB,setpts=PTS-STARTPTS,fps=25,format=yuv420p[v{index}]"
            for index in range(len(segments))
        ]
        joined = "".join(f"[v{index}]" for index in range(len(segments)))
        prepared.append(f"{joined}concat=n={len(segments)}:v=1:a=0,format=yuv420p[v]")
        subprocess.run([
            "ffmpeg", "-y", *inputs, "-filter_complex", ";".join(prepared), "-map", "[v]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", str(output),
        ], check=True)
        return output

    filters: list[str] = []
    offset = segments[0].duration - fade_seconds
    previous = "[0:v]"
    for index in range(1, len(segments)):
        label = f"xf{index}"
        filters.append(f"{previous}[{index}:v]xfade=transition=fade:duration={fade_seconds}:offset={offset}[{label}]")
        previous = f"[{label}]"
        offset += segments[index].duration - fade_seconds
    subprocess.run([
        "ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", previous,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", str(output),
    ], check=True)
    return output
