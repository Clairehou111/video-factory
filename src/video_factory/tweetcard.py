from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .compositor import _font_path, _translation_only, _wrapped_lines
from .editorial import is_audience_glossary_definition
from .models import Candidate, Evidence, Scene


# These cards are rendered at 1384px then scaled into a 1080px phone frame.
# A 27px source-card translation became roughly 21px in the final video and
# was unreadable in WeChat. Keep Chinese evidence copy at mobile subtitle size.
TWEET_TRANSLATION_FONT_MIN = 46
TWEET_TRANSLATION_FONT_MAX = 52
EDITORIAL_TRANSLATION_FONT_SIZE = 46


def _contextual_excerpt(source: str, target: str, limit: int = 340) -> str:
    """Return source-owned sentence context around an exact target."""
    compact = re.sub(r"\s+", " ", source).strip()
    if not target or not compact:
        return ""
    index = compact.casefold().find(target.casefold())
    if index < 0:
        return ""
    boundaries = ".!?。！？"
    start = index
    while start > 0 and compact[start - 1] not in boundaries:
        start -= 1
    end = index + len(target)
    while end < len(compact) and compact[end] not in boundaries:
        end += 1
    if end < len(compact):
        end += 1
    excerpt = compact[start:end].strip()
    if len(excerpt) > limit:
        relative = index - start
        left = max(0, min(relative - limit // 3, len(excerpt) - limit))
        excerpt = excerpt[left:left + limit].strip()
        if left:
            excerpt = "…" + excerpt
        if left + limit < end - start:
            excerpt += "…"
    return excerpt


def _tweet_translation_copy(scene: Scene) -> str:
    translation = (scene.highlight_translation or "").strip()
    glossary = (scene.screen_interpretation or "").strip()
    if glossary and is_audience_glossary_definition(glossary) and glossary not in translation:
        return (translation + "\n" + glossary).strip()
    return translation


def render_tweet_card(
    candidate: Candidate, root: Evidence, scene: Scene, output: Path,
    size: tuple[int, int] = (1384, 1602),
) -> Path:
    """Render one complete, stable tweet2video-style source card."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = size
    canvas = Image.new("RGB", size, "#f4f6fb")
    draw = ImageDraw.Draw(canvas)
    card = (58, 54, width - 58, height - 54)
    draw.rounded_rectangle(card, radius=36, fill="#ffffff", outline="#d8dee9", width=3)
    font_file = str(_font_path())
    author_font = ImageFont.truetype(font_file, 42)
    handle_font = ImageFont.truetype(font_file, 29)
    meta_font = ImageFont.truetype(font_file, 27)

    avatar = (100, 96, 196, 192)
    draw.ellipse(avatar, fill="#165dff")
    name = str(root.metadata.get("author_name") or candidate.metadata.get("author_name") or candidate.author or "X")
    handle = str(root.metadata.get("author_handle") or candidate.author or "unknown")
    initials = "".join(part[0] for part in name.split()[:2]).upper() or "X"
    draw.text((148, 142), initials, font=ImageFont.truetype(font_file, 30), fill="white", anchor="mm")
    draw.text((220, 104), name, font=author_font, fill="#111827")
    draw.text((220, 157), "@" + handle.lstrip("@"), font=handle_font, fill="#64748b")
    draw.rounded_rectangle((width - 190, 105, width - 105, 190), radius=42, fill="#0f172a")
    draw.text((width - 147, 147), "X", font=ImageFont.truetype(font_file, 36), fill="white", anchor="mm")

    source_text = re.sub(r"https://t\.co/\S+", "", root.quote).replace("♾", "∞")
    source_text = re.sub(r"(?im)^\s*(?:learn more at|read more|details)\s*:\s*$", "", source_text).strip()
    translation = _tweet_translation_copy(scene)
    translation_font = None
    translation_lines: list[str] = []
    translation_height = 0
    if translation:
        compact_translation = translation.replace("\n", "").replace(" ", "")
        for size_px in range(TWEET_TRANSLATION_FONT_MAX, TWEET_TRANSLATION_FONT_MIN - 1, -2):
            candidate_font = ImageFont.truetype(font_file, size_px)
            candidate_lines = _wrapped_lines(draw, translation, candidate_font, width - 270, 8)
            if "".join(candidate_lines).replace(" ", "") == compact_translation:
                translation_font = candidate_font
                translation_lines = candidate_lines
                translation_height = max(120, 34 + len(candidate_lines) * (size_px + 14))
                break
        if translation_font is None:
            raise ValueError("adjacent Chinese translation does not fit the complete X card")
    top = 245
    # The complete source plus its adjacent translation already explains the
    # first beat. Repeating scene.screen_fact in a separate bottom chip makes
    # short posts look duplicated and steals space from long posts, so the X
    # card deliberately reserves only translation and source metadata.
    metadata_reserve = 28 + (translation_height + 18 if translation else 0) + 128
    max_body_height = height - 85 - metadata_reserve - top
    lines: list[str] = []
    selected_font = ImageFont.truetype(font_file, 35)
    source_compact = re.sub(r"\s+", "", source_text)
    columns: list[list[str]] = []
    for size_px in range(35, 20, -2):
        selected_font = ImageFont.truetype(font_file, size_px)
        candidate_lines = _wrapped_lines(draw, source_text, selected_font, width - 220, 120)
        line_height = draw.textbbox((0, 0), "Ag中", font=selected_font)[3] + 14
        if len(candidate_lines) * line_height <= max_body_height and re.sub(r"\s+", "", "".join(candidate_lines)) == source_compact:
            lines = candidate_lines
            break
    if not lines:
        gap = 48
        column_width = (width - 220 - gap) // 2
        for size_px in range(25, 15, -1):
            selected_font = ImageFont.truetype(font_file, size_px)
            candidate_lines = _wrapped_lines(draw, source_text, selected_font, column_width, 400)
            line_height = draw.textbbox((0, 0), "Ag中", font=selected_font)[3] + 10
            rows = max_body_height // line_height
            if (
                len(candidate_lines) <= rows * 2
                and re.sub(r"\s+", "", "".join(candidate_lines)) == source_compact
            ):
                split_at = (len(candidate_lines) + 1) // 2
                columns = [candidate_lines[:split_at], candidate_lines[split_at:]]
                break
        if not columns:
            raise ValueError("complete X post does not fit verified one-card layouts")
    line_height = draw.textbbox((0, 0), "Ag中", font=selected_font)[3] + 14
    if columns:
        line_height = draw.textbbox((0, 0), "Ag中", font=selected_font)[3] + 10
        gap = 48
        column_width = (width - 220 - gap) // 2
        for column_index, column_lines in enumerate(columns):
            left = 110 + column_index * (column_width + gap)
            for index, line in enumerate(column_lines):
                draw.text((left, top + index * line_height), line, font=selected_font, fill="#0f172a")
        body_bottom = top + max(len(column) for column in columns) * line_height
    else:
        for index, line in enumerate(lines):
            draw.text((110, top + index * line_height), line, font=selected_font, fill="#0f172a")
        body_bottom = top + len(lines) * line_height

    meta_top = body_bottom + 28
    if translation:
        draw.rounded_rectangle(
            (90, meta_top, width - 90, meta_top + translation_height), radius=22,
            fill="#eaf2ff", outline="#f0c419", width=3,
        )
        translation_line_height = translation_font.size + 14
        for index, line in enumerate(translation_lines):
            draw.text((120, meta_top + 22 + index * translation_line_height), line, font=translation_font, fill="#12366b")
        meta_top += translation_height + 18

    published = str(root.metadata.get("published_at") or candidate.published_at or "captured post")
    draw.line((110, meta_top, width - 110, meta_top), fill="#e2e8f0", width=2)
    draw.text((110, meta_top + 24), published, font=meta_font, fill="#64748b")
    metrics = root.metadata.get("metrics") or candidate.metadata.get("metrics") or {}
    metric_text = "   ".join(
        f"{label} {metrics.get(key)}" for key, label in (("replies", "回复"), ("retweets", "转发"), ("likes", "喜欢"), ("views", "浏览"))
        if metrics.get(key) is not None
    )
    if metric_text:
        draw.text((110, meta_top + 74), metric_text, font=meta_font, fill="#475569")

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def tweet_card_video(
    card: Path, duration: float, output: Path, fps: int = 25, *, motion: bool = False,
) -> Path:
    """Turn a static card into MP4; Radar V2 may opt into a slow push-in."""
    if not motion:
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1",
            "-t", f"{duration:.3f}", "-i", str(card),
            "-vf", f"fps={fps},format=yuv420p", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-r", str(fps), str(output),
        ], check=True)
        return output
    from PIL import Image

    with Image.open(card) as image:
        width, height = image.size
    zoom_step = 0.03 / max(1.0, duration * fps)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-t", f"{duration:.3f}",
        "-i", str(card), "-vf",
        f"zoompan=z='min(zoom+{zoom_step:.8f},1.03)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps={fps},"
        "format=yuv420p",
        "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-r", str(fps), str(output),
    ], check=True)
    return output


def render_editorial_card(
    scene: Scene, evidence: Evidence, output: Path, family: str = "impact_card",
    size: tuple[int, int] = (1384, 1602),
) -> Path:
    """Render an evidence-backed pacing card for sound-off short videos.

    The card never invents a new asset or claim. It changes presentation of
    the scene's already-cited fact so two screenshots from the same source do
    not masquerade as visual variety.
    """
    from PIL import Image, ImageDraw, ImageFont

    width, height = size
    palette = {
        "quote_card": ("#071426", "#ffe063", "#eaf2ff"),
        "timeline": ("#08245c", "#73a7ff", "#ffffff"),
        "impact_card": ("#121044", "#ffcf3d", "#ffffff"),
        "stat_card": ("#082c2b", "#55e6c1", "#ffffff"),
    }
    background, _, foreground = palette.get(family, palette["impact_card"])
    canvas = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(canvas)
    font_file = str(_font_path())
    fact_font = ImageFont.truetype(font_file, 61)
    interpretation_font = ImageFont.truetype(font_file, 46)

    fact = (scene.screen_fact or scene.caption).strip()
    fact_lines = _wrapped_lines(draw, fact, fact_font, width - 180, 5)
    line_height = draw.textbbox((0, 0), "中", font=fact_font)[3] + 20
    fact_top = 120
    for index, line in enumerate(fact_lines):
        draw.text((90, fact_top + index * line_height), line, font=fact_font, fill=foreground)
    rule_top = min(980, fact_top + len(fact_lines) * line_height + 54)

    excerpt = (scene.source_excerpt or "").strip()
    translation = _translation_only((scene.highlight_translation or "").strip())
    implication = (scene.screen_interpretation or "").strip()

    # Derived pacing cards must still carry evidence. Put the exact source
    # excerpt in the large middle area and its Chinese meaning immediately
    # below it. This avoids empty slides that merely repeat editorial copy.
    if excerpt:
        quote_top = rule_top + 44
        excerpt_font = ImageFont.truetype(font_file, 39)
        translation_font = ImageFont.truetype(font_file, EDITORIAL_TRANSLATION_FONT_SIZE)
        excerpt_lines = _wrapped_lines(draw, excerpt, excerpt_font, width - 250, 5)
        translation_lines = _wrapped_lines(draw, translation, translation_font, width - 250, 3) if translation else []
        quote_height = 118 + len(excerpt_lines) * 55 + len(translation_lines) * 60
        quote_bottom = min(height - 180, quote_top + max(270, quote_height))
        draw.rounded_rectangle(
            (76, quote_top, width - 76, quote_bottom), radius=30,
            fill="#0a1930", outline="#f4ca3e", width=4,
        )
        draw.rectangle((104, quote_top + 34, 116, quote_bottom - 34), fill="#f4ca3e")
        excerpt_y = quote_top + 52
        for index, line in enumerate(excerpt_lines):
            draw.text((146, excerpt_y + index * 55), line, font=excerpt_font, fill="#f8fbff")
        if translation_lines:
            translation_y = excerpt_y + len(excerpt_lines) * 55 + 34
            for index, line in enumerate(translation_lines):
                draw.text((146, translation_y + index * 60), line, font=translation_font, fill="#ffe063")
        else:
            translation_y = excerpt_y + len(excerpt_lines) * 55
        implication_top = quote_bottom + 38
    else:
        implication_top = rule_top + 62

    implication_lines = _wrapped_lines(draw, implication, interpretation_font, width - 180, 3)
    for index, line in enumerate(implication_lines):
        draw.text((90, implication_top + index * 54), line, font=interpretation_font, fill="#d7e2ff")

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def render_source_image(
    scene: Scene, evidence: Evidence, asset: Path, output: Path,
    size: tuple[int, int] = (1384, 1602),
) -> Path:
    """Render a real archived source image with dense, sound-off context."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

    width, height = size
    source = Image.open(asset).convert("RGB")
    background = ImageOps.fit(source, size).filter(ImageFilter.GaussianBlur(28))
    background = ImageEnhance.Brightness(background).enhance(0.28)
    canvas = background.convert("RGB")
    photo_box = (54, 42, width - 54, 1050)
    photo = ImageOps.contain(source, (photo_box[2] - photo_box[0], photo_box[3] - photo_box[1]))
    photo_left = (width - photo.width) // 2
    photo_top = photo_box[1] + (photo_box[3] - photo_box[1] - photo.height) // 2
    canvas.paste(photo, (photo_left, photo_top))

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((54, 1090, width - 54, height - 42), radius=28, fill="#061426")
    font_file = str(_font_path())
    fact_font = ImageFont.truetype(font_file, 45)
    implication_font = ImageFont.truetype(font_file, 40)
    fact = (scene.screen_fact or scene.caption).strip()
    implication = (scene.screen_interpretation or "").strip()
    fact_lines = _wrapped_lines(draw, fact, fact_font, width - 180, 3)
    for index, line in enumerate(fact_lines):
        draw.text((90, 1135 + index * 61), line, font=fact_font, fill="#f7f9ff")
    implication_top = 1135 + len(fact_lines) * 61 + 28
    for index, line in enumerate(_wrapped_lines(draw, implication, implication_font, width - 180, 3)):
        draw.text((90, implication_top + index * 45), line, font=implication_font, fill="#bcd0f5")

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output
