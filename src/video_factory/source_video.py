from __future__ import annotations

import hashlib
import base64
import io
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from .llm import LLMSettings
from .media import probe_video
from .youtube_runtime import ManagedYouTubeRuntime


def recommend_source_clip_duration(semantic_hint: str) -> float:
    """Choose a bounded action window from the kind of motion being proved.

    A single, instantly legible action can be short. Multi-step manipulation,
    recovery, or industrial work needs enough time to show setup and payoff.
    The vision selector may refine this recommendation from the contact sheet.
    """
    lower = semantic_hint.casefold()
    multi_step = sum(marker in lower for marker in (
        "whole-body", "full-body", "end-to-end", "industrial", "warehouse",
        "factory", "household", "pharmacy", "fold", "package", "grasp", "pick",
        "carry", "recover", "get back up", "load", "unload", "deploy", "task",
        "customer", "delivery milestone", "全身", "工业", "家务", "药房", "叠衣",
        "抓取", "搬运", "恢复", "部署", "交付", "客户", "500强",
    ))
    physical_story = any(marker in lower for marker in (
        "robot", "robotics", "humanoid", "autonomous driving", "self-driving",
        "robotaxi", "vehicle", "warehouse", "factory", "household", "pharmacy",
        "机器人", "自动驾驶", "无人车", "仓库", "工厂", "工业", "家务", "药房",
    ))
    if multi_step >= 2:
        return 12.0 if physical_story else 7.0
    if any(marker in lower for marker in (
        "jump", "kick", "sprint", "collision", "single motion", "一跃", "踢球", "碰撞",
    )):
        return 5.0 if physical_story else 4.0
    return 9.0 if physical_story else 5.0


def download_official_youtube_video(url: str, output_dir: Path) -> tuple[Path, str]:
    """Download official footage with a bounded primary/fallback route.

    The system yt-dlp route is intentionally tried once because it is cheap.
    YouTube 403/challenge failures then switch to the pinned mweb + PO-token
    runtime already managed by the factory.  Callers may still fall back to a
    browser capture if both routes fail.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:12]
    existing = next((
        path for path in output_dir.glob(f"official-source-{key}.*")
        if path.suffix.lower() in {".mp4", ".webm", ".mkv"}
    ), None)
    if existing:
        return existing, "cache"

    template = output_dir / f"official-source-{key}.%(ext)s"
    common = [
        "--no-playlist", "-f", "bv*[height<=1080]+ba/b[height<=1080]",
        "--merge-output-format", "mp4", "-o", str(template), url,
    ]
    primary_error: subprocess.CalledProcessError | None = None
    try:
        subprocess.run(["yt-dlp", *common], check=True)
        route = "system_yt_dlp"
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            primary_error = error
        runtime = ManagedYouTubeRuntime()
        try:
            subprocess.run([
                runtime.require_executable(), *runtime.extractor_arguments(), *common,
            ], check=True)
        except Exception as fallback_error:
            if primary_error is not None:
                fallback_error.add_note(
                    f"system yt-dlp exited {primary_error.returncode}; managed mweb fallback also failed"
                )
            raise
        route = "managed_mweb_po_token_fallback"

    downloaded = next((
        path for path in output_dir.glob(f"official-source-{key}.*")
        if path.suffix.lower() in {".mp4", ".webm", ".mkv"}
    ), None)
    if downloaded is None:
        raise FileNotFoundError("YouTube download completed without a source video asset")
    return downloaded, route


def _vision_action_clip(
    path: Path, duration: float, clip_duration: float, semantic_hint: str,
) -> dict[str, float | str] | None:
    """Let an available vision model rank timestamped source frames."""
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        return None
    from PIL import Image, ImageDraw, ImageFont

    interval = 4
    timestamps = list(range(0, max(1, int(duration - clip_duration)) + 1, interval))
    timestamps = timestamps[:25]
    with TemporaryDirectory(prefix="video-factory-source-contact-") as temp:
        root = Path(temp)
        frames: list[tuple[int, Image.Image]] = []
        for timestamp in timestamps:
            frame = root / f"{timestamp:03d}.jpg"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(timestamp), "-i", str(path), "-frames:v", "1",
                "-vf", "scale=320:-1", str(frame),
            ], check=True)
            with Image.open(frame).convert("RGB") as image:
                frames.append((timestamp, image.copy()))
        if not frames:
            return None
        cell_w, cell_h = 320, frames[0][1].height + 30
        columns = 5
        rows = (len(frames) + columns - 1) // columns
        sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), "#020815")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=20)
        for index, (timestamp, image) in enumerate(frames):
            x, y = (index % columns) * cell_w, (index // columns) * cell_h
            sheet.paste(image, (x, y))
            draw.text((x + 8, y + image.height + 3), f"{timestamp}s", fill="white", font=font)
        buffer = io.BytesIO()
        sheet.save(buffer, format="JPEG", quality=85)

    model = os.environ.get(
        "VIDEO_FACTORY_SOURCE_VIDEO_VISION_MODEL", "google/gemini-3.7-flash",
    ).strip()
    settings = LLMSettings.from_environment("openrouter", model)
    maximum_duration = max(8.0, min(12.0, clip_duration))
    prompt = "\n".join([
        "Choose the best start time and duration for one continuous short clip from this official source video.",
        f"The scenario-based starting recommendation is {clip_duration:.1f} seconds.",
        f"You may choose 3.5 to {maximum_duration:.1f} seconds: use less for one instantly clear motion and more for setup-action-payoff.",
        "The contact sheet labels source timestamps. Pick one labeled timestamp or up to one second before it.",
        "Choose the moment that most clearly and surprisingly proves the selected news subject performing its defining action.",
        "Avoid title cards, static group shots, a human hand blocking the subject, and a window that crosses a hard scene cut.",
        "Story context:", semantic_hint[:4000],
        'Return JSON only: {"start": number, "duration": number, "reason": "short explanation"}',
    ])
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(),
            }},
        ]}],
        "response_format": {"type": "json_object"}, "temperature": 0.1,
        "max_tokens": 300, "provider": settings.provider_preferences,
    }
    request = Request(
        settings.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), method="POST",
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/video-factory",
            "X-Title": "Video Factory Source Clip Selector",
        },
    )
    with urlopen(request, timeout=settings.timeout_seconds) as response:
        result = json.loads(response.read().decode())
    raw = result.get("choices", [{}])[0].get("message", {}).get("content")
    answer = json.loads(raw) if isinstance(raw, str) else {}
    selected_duration = max(
        3.5, min(maximum_duration, float(answer.get("duration") or clip_duration)),
    )
    selected_duration = min(selected_duration, duration)
    start = max(0.0, min(float(answer.get("start") or 0), duration - selected_duration))
    return {
        "start": round(start, 3), "end": round(start + selected_duration, 3),
        "score": 0.0, "method": "vision_ranked_timestamped_contact_sheet",
        "reason": str(answer.get("reason") or "vision-selected defining action"),
    }


def select_action_clip(
    path: Path, clip_duration: float = 5.0, semantic_hint: str = "",
) -> dict[str, float | str]:
    """Select the best continuous action clip, with a deterministic fallback."""
    from PIL import Image, ImageChops, ImageStat

    duration = probe_video(path).duration
    if duration <= clip_duration + 0.5:
        return {
            "start": 0.0, "end": round(duration, 3), "score": 0.0,
            "method": "whole_short_source_video",
        }
    if semantic_hint.strip():
        try:
            selected = _vision_action_clip(path, duration, clip_duration, semantic_hint)
            if selected is not None:
                return selected
        except Exception:
            # Vision is taste, not a hard dependency. Motion scoring remains a
            # bounded local fallback and requires no account or network.
            pass
    fps = 2.0
    with TemporaryDirectory(prefix="video-factory-source-video-") as temp:
        root = Path(temp)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
            "-vf", f"fps={fps:g},scale=320:-1,format=gray", str(root / "%05d.png"),
        ], check=True)
        frames = sorted(root.glob("*.png"))
        differences: list[float] = [0.0]
        centered_differences: list[float] = [0.0]
        brightness: list[float] = []
        previous = None
        for frame in frames:
            with Image.open(frame).convert("L") as image:
                brightness.append(float(ImageStat.Stat(image).mean[0]))
                if previous is not None:
                    delta = ImageChops.difference(previous, image)
                    differences.append(float(ImageStat.Stat(delta).mean[0]))
                    left, top = round(image.width * 0.15), round(image.height * 0.10)
                    right, bottom = round(image.width * 0.85), round(image.height * 0.95)
                    centered_differences.append(float(ImageStat.Stat(
                        delta.crop((left, top, right, bottom)),
                    ).mean[0]))
                previous = image.copy()

    window_frames = max(2, round(clip_duration * fps))
    best_score = float("-inf")
    best_start = 1.0
    latest_start = max(1, int(duration - clip_duration - 1))
    for start_second in range(1, latest_start + 1):
        first = round(start_second * fps)
        last = min(len(differences), first + window_frames)
        if last - first < window_frames - 1:
            continue
        motion = differences[first:last]
        centered_motion = centered_differences[first:last]
        light = brightness[first:last]
        hard_cuts = sum(value >= 34.0 for value in motion)
        full_motion = sum(min(value, 28.0) for value in motion) / max(len(motion), 1)
        center_motion = sum(min(value, 28.0) for value in centered_motion) / max(len(centered_motion), 1)
        # Prefer action carried by the subject in the central stage. Large
        # hands or camera wipes entering from an edge create more edge motion
        # and used to beat the robot's actual task.
        center_fraction = 0.70 * 0.85
        edge_motion = max(0.0, (full_motion - center_motion * center_fraction) / (1 - center_fraction))
        sustained_motion = center_motion - edge_motion * 0.5
        dark_penalty = max(0.0, 45.0 - sum(light) / max(len(light), 1)) * 0.25
        score = sustained_motion - hard_cuts * 18.0 - dark_penalty
        if score > best_score:
            best_score, best_start = score, float(start_second)
    return {
        "start": round(best_start, 3), "end": round(best_start + clip_duration, 3),
        "score": round(best_score, 3), "method": "sustained_motion_without_hard_cuts",
    }


def render_source_video_clip(
    source: Path, output: Path, *, start: float, duration: float,
    width: int = 1384, height: int = 1092, loop: bool = False,
) -> Path:
    """Render one uninterrupted source clip into the evidence viewport.

    Very short X animations are commonly delivered as a single MP4 loop.
    ``loop`` repeats that source for the planned proof hold instead of letting
    a sub-second asset collapse the entire scene.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    loop_args = ["-stream_loop", "-1"] if loop else []
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *loop_args, "-ss", f"{max(0.0, start):.3f}",
        "-t", f"{duration:.3f}", "-i", str(source),
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x020815,"
        "setsar=1,fps=25,format=yuv420p",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", str(output),
    ], check=True)
    return output
