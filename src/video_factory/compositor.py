from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from .models import InformationRenderProfile, RenderManifest, Scene, TopicType
from .media import probe_video
from .radar import FACTUAL_CATEGORY_LABELS, extract_direct_identifier, factual_category_label


CANVAS = (1080, 1920)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FONT_CANDIDATES = (
    PROJECT_ROOT / "assets/fonts/NotoSansCJK-Regular.ttc",
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("C:/Windows/Fonts/msyh.ttc"),
)
DEFAULT_FONT = str(FONT_CANDIDATES[0])
# WeChat overlays the device/status chrome at the top and the post title plus
# action controls at the bottom.  These bands are intentionally content-free:
# rails may paint their background through them, but no essential glyph or
# evidence is allowed there.
WECHAT_TOP_UI_SAFE = 120
WECHAT_BOTTOM_UI_SAFE = 400
RADAR_META_RAIL = 88
def _is_radar_v2(manifest: RenderManifest) -> bool:
    return str(manifest.render_profile or "classic") == InformationRenderProfile.RADAR_V2.value

# Cold-open copy earns attention, but the repository must already feel like
# the subject—not a small preview pushed below a title card.  Each tuple is
# ``(screenshot_y, screenshot_height, crop_center_y, copy_y, repo_name_y)``.
# The walkthrough intentionally keeps even more room for the browser pane.
GITHUB_COLD_OPEN_LAYOUTS = (
    (760, 1160, 0.05, 250, 675),
    (720, 1200, 0.28, 245, 635),
    (800, 1120, 0.58, 270, 715),
)


def resolve_font_path(font_path: Path | None = None) -> Path:
    candidates: list[Path] = []
    if font_path is not None:
        candidates.append(font_path.expanduser())
    configured = os.environ.get("VIDEO_FACTORY_FONT", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(FONT_CANDIDATES)
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        "no Chinese-capable TTF/TTC was found; set VIDEO_FACTORY_FONT or install one of: "
        + searched
    )


def _font_path(font_path: Path | None = None) -> Path:
    return resolve_font_path(font_path)


def _wrapped_lines(draw: object, text: str, font: object, width: int, max_lines: int) -> list[str]:
    """Wrap CJK text without splitting an adjacent ASCII project/vendor word."""
    lines: list[str] = []
    current = ""
    for char in text.strip():
        if char == "\n":
            if current:
                lines.append(current)
                if len(lines) >= max_lines:
                    current = ""
                    break
            current = ""
            continue
        candidate = current + char
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > width:
            # Greedy character wrapping used to turn ``Claude`` into
            # ``Clau``/``de`` in a mixed Chinese headline. Move the whole
            # trailing ASCII token when it can fit on the next line.
            if char.isascii() and (char.isalnum() or char in "._-/"):
                token_start = len(current)
                while token_start and current[token_start - 1].isascii() and (
                    current[token_start - 1].isalnum() or current[token_start - 1] in "._-/"
                ):
                    token_start -= 1
                moved = current[token_start:] + char
                prefix = current[:token_start]
                if prefix and draw.textbbox((0, 0), moved, font=font)[2] <= width:
                    lines.append(prefix)
                    current = moved
                else:
                    lines.append(current)
                    current = char
            else:
                lines.append(current)
                current = char
            if len(lines) == max_lines:
                break
        else:
            current = candidate
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and len("".join(lines)) < len(text.strip()):
        lines[-1] = lines[-1].rstrip("…") + "…"
    return lines


def _splits_ascii_word(text: str, index: int) -> bool:
    """Return whether a line break would cut an adjacent ASCII identifier."""
    if not 0 < index < len(text):
        return False
    word_chars = "._-/"
    left = text[index - 1]
    right = text[index]
    return (
        left.isascii() and (left.isalnum() or left in word_chars)
        and right.isascii() and (right.isalnum() or right in word_chars)
    )


def _centered_lines(draw: object, text: str, font: object, width: int, max_lines: int) -> list[str]:
    lines = _wrapped_lines(draw, text, font, width, max_lines)
    # A greedy CJK wrap can leave a single character on line two.  Hooks are
    # read in a glance, so rebalance a complete two-line title around a natural
    # punctuation boundary (or the visual midpoint) instead of accepting an
    # orphan glyph.
    compact = text.strip().replace("\n", "")
    if max_lines == 2 and len(lines) == 2 and "".join(lines) == compact:
        candidates: list[tuple[float, int]] = []
        for index in range(2, len(compact) - 1):
            if _splits_ascii_word(compact, index):
                continue
            first, second = compact[:index], compact[index:]
            first_width = draw.textbbox((0, 0), first, font=font)[2]
            second_width = draw.textbbox((0, 0), second, font=font)[2]
            if first_width <= width and second_width <= width:
                # A natural phrase boundary is worth more than perfect pixel
                # symmetry; otherwise a balanced title can split “水印” into
                # two visually equal but linguistically broken lines.
                punctuation_bonus = -width if first[-1] in "：，；！？:、" else 0
                candidates.append((abs(first_width - second_width) + punctuation_bonus, index))
        if candidates:
            split = min(candidates)[1]
            lines = [compact[:split], compact[split:]]
    return lines


def _draw_centered(draw: object, text: str, *, box: tuple[int, int, int, int], font: object, fill: str, max_lines: int, line_gap: int = 10) -> None:
    left, top, right, bottom = box
    lines = _centered_lines(draw, text, font, right - left, max_lines)
    line_height = draw.textbbox((0, 0), "中", font=font)[3] + line_gap
    start_y = top + max(0, ((bottom - top) - len(lines) * line_height) // 2)
    for index, line in enumerate(lines):
        draw.text(((left + right) // 2, start_y + index * line_height), line, font=font, fill=fill, anchor="ma")


def _information_layout(hook: str, font_path: Path | None = None) -> tuple[int, int, int]:
    """Return top-rail height, title size, and lines without truncation.

    The rail remains fixed during playback, but its size adapts per story.
    """
    from PIL import Image, ImageDraw, ImageFont

    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    compact = hook.strip().replace("\n", "")
    font_file = str(_font_path(font_path))
    for top_height, max_lines, maximum_size in ((260, 2, 54), (300, 3, 50), (340, 3, 46), (380, 4, 42)):
        available_height = top_height - 72
        for size in range(maximum_size, 33, -2):
            font = ImageFont.truetype(font_file, size)
            lines = _wrapped_lines(draw, hook, font, 944, max_lines)
            line_height = draw.textbbox((0, 0), "中", font=font)[3] + 12
            if "".join(lines) == compact and len(lines) * line_height <= available_height:
                return top_height, size, max_lines
    raise ValueError("fixed title cannot fit the adaptive upper rail without truncation")


def _footer_layout(footer: str, font_path: Path | None = None) -> tuple[int, int, int]:
    """Return bottom-rail height, font size, and lines without ellipsis."""
    from PIL import Image, ImageDraw, ImageFont

    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    compact = footer.strip().replace("\n", "")
    font_file = str(_font_path(font_path))
    for height, max_lines, maximum_size in ((230, 2, 44), (270, 3, 42), (310, 3, 38), (350, 4, 36)):
        available_height = height - 68
        for size in range(maximum_size, 29, -2):
            font = ImageFont.truetype(font_file, size)
            lines = _wrapped_lines(draw, footer, font, 924, max_lines)
            line_height = draw.textbbox((0, 0), "中", font=font)[3] + 14
            if "".join(lines) == compact and len(lines) * line_height <= available_height:
                return height, size, max_lines
    raise ValueError("fixed conclusion cannot fit the adaptive bottom rail without truncation")


def render_fixed_footer(text: str, output: Path, font_path: Path | None = None) -> Path:
    """Compatibility helper for the former footer-only renderer."""
    from PIL import Image, ImageDraw, ImageFont

    if not text.strip():
        raise ValueError("fixed footer text is required")
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    footer_bottom = CANVAS[1] - WECHAT_BOTTOM_UI_SAFE
    footer_top = footer_bottom - 190
    draw.rounded_rectangle((80, footer_top, 1000, footer_bottom), radius=26, fill="#061a36f8")
    _draw_centered(
        draw, text, box=(120, footer_top + 20, 960, footer_bottom - 20),
        font=ImageFont.truetype(str(_font_path(font_path)), 36),
        fill="#e8f1ff", max_lines=3,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def render_information_frame(hook: str, footer: str, output: Path, font_path: Path | None = None) -> Path:
    """Opaque fixed top/bottom rails; the middle remains transparent for evidence video."""
    from PIL import Image, ImageDraw, ImageFont

    if not hook.strip() or not footer.strip():
        raise ValueError("both fixed hook and fixed footer are required")
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    top_height, title_size, title_lines = _information_layout(hook, font_path)
    bottom_height, conclusion_size, conclusion_lines = _footer_layout(footer, font_path)
    top_bottom = WECHAT_TOP_UI_SAFE + top_height
    bottom_top = 1920 - WECHAT_BOTTOM_UI_SAFE - bottom_height
    draw.rectangle((0, 0, 1080, top_bottom), fill="#031126ff")
    draw.rectangle((0, bottom_top, 1080, 1920), fill="#031126ff")
    title = ImageFont.truetype(str(_font_path(font_path)), title_size)
    conclusion = ImageFont.truetype(str(_font_path(font_path)), conclusion_size)
    _draw_centered(
        draw, hook,
        box=(68, WECHAT_TOP_UI_SAFE + 28, 1012, top_bottom - 24), font=title,
        fill="#f4f8ff", max_lines=title_lines, line_gap=12,
    )
    _draw_centered(
        draw, footer,
        box=(78, bottom_top + 28, 1002, 1920 - WECHAT_BOTTOM_UI_SAFE - 28),
        font=conclusion,
        fill="#eaf2ff", max_lines=conclusion_lines, line_gap=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _expanded_evidence_layout(manifest: RenderManifest) -> bool:
    if not _is_radar_v2(manifest):
        return False
    if manifest.github_brief or manifest.topic_type == TopicType.GITHUB_PROJECT:
        return True
    if manifest.topic_type == TopicType.RESEARCH_OR_BENCHMARK:
        return True
    if any(
        item.source_kind.casefold().startswith(("paper", "pdf", "arxiv"))
        or str(item.metadata.get("visual_role") or "").casefold() == "architecture"
        for item in manifest.evidence
    ):
        return True
    return any(
        scene.visual_family.casefold() in {"paper", "architecture", "code"}
        for scene in manifest.scenes
    )


def _single_header_layout(title: str, font_path: Path | None = None) -> tuple[int, int]:
    """Fit one stable project/action line above an expanded evidence viewport."""
    from PIL import Image, ImageDraw, ImageFont

    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    compact = re.sub(r"\s+", " ", title.strip())
    font_file = str(_font_path(font_path))
    for size in range(46, 21, -2):
        font = ImageFont.truetype(font_file, size)
        if draw.textbbox((0, 0), compact, font=font)[2] <= 944:
            return 130, size
    raise ValueError("expanded-layout title must fit one line; keep the repo/paper name and one action")


def render_single_header_frame(
    title: str, output: Path, font_path: Path | None = None,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    height, size = _single_header_layout(title, font_path)
    rail_bottom = WECHAT_TOP_UI_SAFE + height
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1080, rail_bottom), fill="#031126ff")
    draw.text(
        (540, WECHAT_TOP_UI_SAFE + height // 2), re.sub(r"\s+", " ", title.strip()),
        font=ImageFont.truetype(str(_font_path(font_path)), size), fill="#f4f8ff", anchor="mm",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _direct_identifier(manifest: RenderManifest) -> str:
    return extract_direct_identifier(manifest)


def render_direct_identifier_badge(
    identifier: str, pane_top: int, output: Path, font_path: Path | None = None,
    category_label: str = "",
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    identifier = identifier.strip()
    hint = "原帖/仓库链接见文案 ↗"
    font_file = str(_font_path(font_path))
    identifier_font = ImageFont.truetype(font_file, 23)
    hint_font = ImageFont.truetype(font_file, 18)
    identifier_width = draw.textbbox((0, 0), identifier, font=identifier_font)[2] if identifier else 0
    hint_width = draw.textbbox((0, 0), hint, font=hint_font)[2]
    width = min(940, max(identifier_width, hint_width) + 42)
    left = 1040 - width
    top = pane_top + 16
    height = 72 if identifier else 42
    draw.rounded_rectangle(
        (left, top, 1040, top + height), radius=14,
        fill="#031126e8", outline="#ffd43b", width=2,
    )
    if identifier:
        while identifier_width > width - 32 and identifier_font.size > 16:
            identifier_font = ImageFont.truetype(font_file, identifier_font.size - 1)
            identifier_width = draw.textbbox((0, 0), identifier, font=identifier_font)[2]
        draw.text((1024, top + 10), identifier, font=identifier_font, fill="#ffe063", anchor="ra")
        draw.text((1024, top + 44), hint, font=hint_font, fill="#9eabc0", anchor="ra")
    else:
        draw.text((1024, top + 11), hint, font=hint_font, fill="#9eabc0", anchor="ra")
    category_label = category_label.strip()
    if category_label in FACTUAL_CATEGORY_LABELS:
        category_font = ImageFont.truetype(font_file, 22)
        category_width = draw.textbbox((0, 0), category_label, font=category_font)[2] + 34
        draw.rounded_rectangle(
            (40, top, 40 + category_width, top + 42), radius=21,
            fill="#061a36e6", outline="#3b82f640", width=1,
        )
        draw.text((57, top + 9), category_label, font=category_font, fill="#f8fafc")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _zoom_crop(
    highlight_box: dict[str, int], source_size: tuple[int, int], pane_height: int,
    factor: float = 1.4,
) -> tuple[int, int, int, int]:
    source_width, source_height = source_size
    target_aspect = 1080 / pane_height
    maximum_width = source_width / factor
    maximum_height = source_height / factor
    if maximum_width / maximum_height > target_aspect:
        crop_height = maximum_height
        crop_width = crop_height * target_aspect
    else:
        crop_width = maximum_width
        crop_height = crop_width / target_aspect
    center_x = highlight_box["left"] + highlight_box["width"] / 2
    center_y = highlight_box["top"] + highlight_box["height"] / 2
    left = max(0.0, min(source_width - crop_width, center_x - crop_width / 2))
    top = max(0.0, min(source_height - crop_height, center_y - crop_height / 2))
    return tuple(round(value) // 2 * 2 for value in (crop_width, crop_height, left, top))


def _dense_evidence(shot: dict[str, object], source_size: tuple[int, int]) -> bool:
    highlight_box = shot.get("highlight_box")
    return bool(
        re.search(
            r"table|price|code|benchmark|terminal|chart",
            " ".join(str(shot.get(key) or "") for key in ("action", "visual_family", "id")),
            re.IGNORECASE,
        )
        or (
            isinstance(highlight_box, dict) and source_size[0] >= 1300
            and int(highlight_box.get("width") or 0) < source_size[0] * 0.65
        )
    )


def render_spotlight_overlay(
    highlight_box: dict[str, int], source_size: tuple[int, int], pane_top: int,
    pane_height: int, output: Path, font_path: Path | None = None,
    source_region: tuple[int, int, int, int] | None = None,
    dense: bool = False,
) -> Path:
    from PIL import Image, ImageDraw, ImageFilter

    del font_path
    source_width, source_height = source_size
    if source_region:
        region_width, region_height, region_left, region_top = source_region
    else:
        region_width, region_height, region_left, region_top = source_width, source_height, 0, 0
    scale = min(1080 / region_width, pane_height / region_height)
    rendered_width = region_width * scale
    rendered_height = region_height * scale
    offset_x = (1080 - rendered_width) / 2
    offset_y = pane_top + (pane_height - rendered_height) / 2
    left = offset_x + (highlight_box["left"] - region_left) * scale
    top = offset_y + (highlight_box["top"] - region_top) * scale
    right = left + highlight_box["width"] * scale
    bottom = top + highlight_box["height"] * scale
    margin = 18
    focus = (
        max(0, round(left - margin)), max(pane_top, round(top - margin)),
        min(1080, round(right + margin)), min(pane_top + pane_height, round(bottom + margin)),
    )
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    # 40% preserves page context; dense code/tables use 55% to remove noise.
    draw.rectangle(
        (0, pane_top, 1080, pane_top + pane_height),
        fill=(3, 17, 38, 140 if dense else 102),
    )
    draw.rounded_rectangle(focus, radius=18, fill=(0, 0, 0, 0))
    glow = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.rounded_rectangle(focus, radius=18, outline=(255, 216, 77, 180), width=4)
    canvas = Image.alpha_composite(canvas, glow.filter(ImageFilter.GaussianBlur(4)))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(focus, radius=18, outline=(255, 216, 77, 210), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _split_hook_fact(text: str) -> tuple[str, str]:
    compact = text.strip()
    for marker in ("，", "；", "但", "却"):
        if marker in compact:
            left, right = compact.split(marker, 1)
            return left.strip(), right.strip()
    midpoint = max(1, len(compact) // 2)
    return compact[:midpoint], compact[midpoint:]


def render_github_cold_open_frames(
    opening: str, reveal: str, verdict: str, strategy: str, project_title: str, source_frame: Path,
    output_dir: Path, font_path: Path | None = None,
) -> list[Path]:
    """Create three clean editorial beats before the walkthrough.

    The cold open deliberately uses no ornamental rules, screenshot frames, or
    production labels.  Every visible element must either identify the project
    or advance the story.  Distinction between beats comes from type hierarchy
    and colour, rather than unexplained decoration.
    """
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    output_dir.mkdir(parents=True, exist_ok=True)
    font_file = str(_font_path(font_path))
    source = Image.open(source_frame).convert("RGB")

    def common(
        top: tuple[int, int, int], bottom: tuple[int, int, int],
        *, screenshot_y: int, screenshot_height: int, centering_y: float,
    ) -> tuple[object, object]:
        canvas = Image.new("RGB", CANVAS, top)
        draw = ImageDraw.Draw(canvas)
        for y in range(CANVAS[1]):
            ratio = y / (CANVAS[1] - 1)
            color = tuple(round(a + (b - a) * ratio) for a, b in zip(top, bottom))
            draw.line((0, y, CANVAS[0], y), fill=color)
        draw.text(
            (72, 82), "OPEN SOURCE", font=ImageFont.truetype(font_file, 28),
            fill="#a9c7ff",
        )
        # The repository itself is the visual payoff.  Show it as a wide,
        # edge-to-edge preview—never as a tiny screenshot trapped in a UI
        # frame.  Different crops give the three one-second beats real motion
        # without inventing decorative graphics.
        preview = ImageOps.fit(
            source, (1080, screenshot_height), centering=(0.5, centering_y),
        )
        canvas.paste(preview, (0, screenshot_y))
        shade = Image.new("RGBA", (1080, screenshot_height), (3, 17, 48, 0))
        shade_draw = ImageDraw.Draw(shade)
        fade_height = min(220, screenshot_height)
        for offset in range(fade_height):
            alpha = round(255 * (1 - offset / max(1, fade_height - 1)))
            shade_draw.line((0, offset, 1080, offset), fill=(*top, alpha))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), Image.new("RGBA", CANVAS, (0, 0, 0, 0)))
        canvas.alpha_composite(shade, (0, screenshot_y))
        draw = ImageDraw.Draw(canvas)
        return canvas, draw

    def centered(
        draw: object, value: str, y: int, size: int, fill: str, stroke: str = "#020713",
        width: int = 820, max_lines: int = 3, minimum_size: int = 34,
    ) -> None:
        compact = value.strip().replace("\n", "")
        lines: list[str] = []
        font = None
        for candidate_size in range(size, minimum_size - 1, -2):
            candidate_font = ImageFont.truetype(font_file, candidate_size)
            candidate_lines = _wrapped_lines(draw, value, candidate_font, width, max_lines)
            last_width = (
                draw.textbbox((0, 0), candidate_lines[-1], font=candidate_font)[2]
                if candidate_lines else 0
            )
            # A single orphan glyph makes an otherwise bold hook look broken.
            # Prefer a slightly smaller face that produces a balanced final
            # line; never truncate copy to achieve the fit.
            balanced = len(candidate_lines) == 1 or last_width >= min(230, width * 0.26)
            if "".join(candidate_lines) == compact and balanced:
                font, lines = candidate_font, candidate_lines
                break
        if font is None:
            raise ValueError(f"cold-open copy cannot fit without truncation: {value}")
        line_height = draw.textbbox((0, 0), "中", font=font)[3] + 14
        for index, line in enumerate(lines):
            draw.text((540, y + index * line_height), line, font=font, fill=fill, stroke_width=4, stroke_fill=stroke, anchor="ma")

    del strategy

    repository_name = project_title.split("：", 1)[0].split(":", 1)[0].strip()

    def project_name(draw: object, y: int) -> None:
        draw.text(
            (72, y), repository_name, font=ImageFont.truetype(font_file, 34),
            fill="#d4e2ff", stroke_width=3, stroke_fill="#06142d",
        )

    frames: list[Path] = []
    screenshot_y, screenshot_height, crop_y, copy_y, name_y = GITHUB_COLD_OPEN_LAYOUTS[0]
    canvas, draw = common(
        (3, 17, 48), (8, 63, 159), screenshot_y=screenshot_y,
        screenshot_height=screenshot_height, centering_y=crop_y,
    )
    centered(draw, opening, copy_y, 86, "#ffffff", width=900, max_lines=4, minimum_size=50)
    project_name(draw, name_y)
    frames.append(output_dir / "cold-open-1.png")
    canvas.convert("RGB").save(frames[-1])

    screenshot_y, screenshot_height, crop_y, copy_y, name_y = GITHUB_COLD_OPEN_LAYOUTS[1]
    canvas, draw = common(
        (4, 29, 70), (13, 89, 187), screenshot_y=screenshot_y,
        screenshot_height=screenshot_height, centering_y=crop_y,
    )
    centered(draw, reveal, copy_y, 82, "#ffe45c", width=900, max_lines=4, minimum_size=48)
    project_name(draw, name_y)
    frames.append(output_dir / "cold-open-2.png")
    canvas.convert("RGB").save(frames[-1])

    screenshot_y, screenshot_height, crop_y, copy_y, name_y = GITHUB_COLD_OPEN_LAYOUTS[2]
    canvas, draw = common(
        (12, 25, 66), (21, 63, 151), screenshot_y=screenshot_y,
        screenshot_height=screenshot_height, centering_y=crop_y,
    )
    centered(draw, verdict, copy_y, 88, "#ffffff", width=900, max_lines=4, minimum_size=50)
    project_name(draw, name_y)
    frames.append(output_dir / "cold-open-3.png")
    canvas.convert("RGB").save(frames[-1])
    return frames


def render_github_cold_open(
    manifest: RenderManifest, visual_track: Path, output: Path,
    temp_root: Path, font_path: Path | None = None,
) -> Path:
    brief = manifest.github_brief
    if brief is None:
        raise ValueError("GitHub cold open requires github_brief")
    beats = manifest.cold_open_beats
    if len(beats) == 3:
        opening, reveal, verdict = (beat.text for beat in beats)
    else:
        opening = brief.hook_opening or brief.hook_stance
        reveal = brief.hook_reveal or _split_hook_fact(brief.hook_fact)[0]
        verdict = brief.hook_verdict or _split_hook_fact(brief.hook_fact)[1]
    durations = [beat.duration for beat in beats] if len(beats) == 3 else [1.0, 1.0, 1.2]
    source_frame = temp_root / "cold-open-source.png"
    subprocess.run([
        "ffmpeg", "-y", "-ss", "0.4", "-i", str(visual_track),
        "-frames:v", "1", "-update", "1", str(source_frame),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = render_github_cold_open_frames(
        opening, reveal, verdict, brief.hook_strategy, brief.project_title,
        source_frame, temp_root / "cold-open-frames", font_path,
    )
    total_duration = sum(durations)
    subprocess.run([
        "ffmpeg", "-y",
        "-loop", "1", "-t", f"{durations[0]:.3f}", "-i", str(frames[0]),
        "-loop", "1", "-t", f"{durations[1]:.3f}", "-i", str(frames[1]),
        "-loop", "1", "-t", f"{durations[2]:.3f}", "-i", str(frames[2]),
        "-filter_complex",
        "[0:v]fps=25,format=yuv420p[v0];[1:v]fps=25,format=yuv420p[v1];[2:v]fps=25,format=yuv420p[v2];"
        "[v0][v1][v2]concat=n=3:v=1:a=0,format=yuv420p[v]",
        "-map", "[v]", "-t", f"{total_duration:.3f}", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", str(output),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output


def _translation_only(value: str) -> str:
    for marker in ("译为：", "译为:", "→"):
        if marker in value:
            return value.split(marker, 1)[1].strip(" 。.\"”")
    return value.strip(" 。.\"”")


def render_scene_copy(scene: Scene, output: Path, font_path: Path | None = None) -> Path:
    """A small nearby Chinese gloss; never a full-screen editorial card."""
    from PIL import Image, ImageDraw, ImageFont

    translation = _translation_only((scene.highlight_translation or "").strip())
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    if not translation:
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
        return output
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(_font_path(font_path)), 27)
    lines = _wrapped_lines(draw, translation, font, 760, 2)
    line_height = draw.textbbox((0, 0), "中", font=font)[3] + 6
    height = 28 + len(lines) * line_height
    # web-scroll-video centers a highlighted target in the visible page. This
    # chip sits immediately below that center area and leaves the source text
    # unobstructed. DOM-positioned placement is added by the recorder when it
    # has a target box; this is the safe generic fallback.
    top = 1080
    draw.rounded_rectangle((100, top, 980, top + height), radius=18, fill="#071426dc", outline="#ffd84dcc", width=2)
    for index, line in enumerate(lines):
        draw.text((540, top + 14 + index * line_height), line, font=font, fill="#ffe063", anchor="ma")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def render_adjacent_gloss(
    text: str, output: Path, font_path: Path | None = None,
    highlight_box: dict[str, int] | None = None, source_size: tuple[int, int] | None = None,
    pane_top: int = 320, pane_height: int = 1250,
    source_region: tuple[int, int, int, int] | None = None,
    profile: str = InformationRenderProfile.CLASSIC.value,
) -> Path:
    """Render a modern glass chip beside the actual recorded yellow target."""
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    translation = _translation_only(text)
    if translation and profile != InformationRenderProfile.RADAR_V2.value:
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(str(_font_path(font_path)), 28)
        text_width = 820
        center_x = 540
        top = 1030
        if highlight_box and source_size:
            source_width, source_height = source_size
            scale = min(1080 / source_width, pane_height / source_height)
            pane_left = (1080 - source_width * scale) / 2
            target_left = pane_left + float(highlight_box["left"]) * scale
            target_top = pane_top + float(highlight_box["top"]) * scale
            target_width = float(highlight_box["width"]) * scale
            target_height = float(highlight_box["height"]) * scale
            text_width = min(820, max(520, int(target_width + 260)))
            center_x = int(target_left + target_width / 2)
            center_x = max(40 + text_width // 2, min(1040 - text_width // 2, center_x))
            top = int(target_top + target_height + 24)
        lines = _wrapped_lines(draw, translation, font, text_width, 3)
        line_height = draw.textbbox((0, 0), "中", font=font)[3] + 8
        height = len(lines) * line_height
        if highlight_box and source_size and top + height > 1540:
            scale = min(1080 / source_size[0], pane_height / source_size[1])
            target_top = pane_top + float(highlight_box["top"]) * scale
            top = max(pane_top + 20, int(target_top - height - 24))
        for index, line in enumerate(lines):
            draw.text(
                (center_x, top + index * line_height), line, font=font, fill="#ffe35b",
                stroke_width=5, stroke_fill="#020815", anchor="ma",
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
        return output
    if translation:
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(str(_font_path(font_path)), 28)
        text_width = 820
        center_x = 540
        top = 1030
        if highlight_box and source_size:
            source_width, source_height = source_size
            if source_region:
                region_width, region_height, region_left, region_top = source_region
            else:
                region_width, region_height, region_left, region_top = source_width, source_height, 0, 0
            scale = min(1080 / region_width, pane_height / region_height)
            pane_left = (1080 - region_width * scale) / 2
            pane_offset_top = pane_top + (pane_height - region_height * scale) / 2
            target_left = pane_left + (float(highlight_box["left"]) - region_left) * scale
            target_top = pane_offset_top + (float(highlight_box["top"]) - region_top) * scale
            target_width = float(highlight_box["width"]) * scale
            target_height = float(highlight_box["height"]) * scale
            text_width = min(820, max(520, int(target_width + 260)))
            center_x = int(target_left + target_width / 2)
            center_x = max(40 + text_width // 2, min(1040 - text_width // 2, center_x))
            top = int(target_top + target_height + 24)
        lines = _wrapped_lines(draw, translation, font, text_width, 3)
        line_height = draw.textbbox((0, 0), "中", font=font)[3] + 8
        height = len(lines) * line_height
        if highlight_box and source_size and top + height + 28 > pane_top + pane_height:
            top = max(pane_top + 20, int(target_top - height - 52))
        measured_width = max(
            (draw.textbbox((0, 0), line, font=font)[2] for line in lines), default=0,
        )
        chip_width = min(980, measured_width + 48)
        chip_left = max(24, min(1080 - chip_width - 24, center_x - chip_width / 2))
        chip_top = top - 12
        chip_bottom = top + height + 12
        draw.rounded_rectangle(
            (chip_left, chip_top, chip_left + chip_width, chip_bottom), radius=16,
            fill=(6, 26, 54, 230), outline=(59, 130, 246, 64), width=1,
        )
        emphasis = re.compile(
            r"(\$?\d+(?:\.\d+)?%?(?:/[A-Za-z])?|发布|开源|暴跌|上涨|下降|加入|推出|支持|移除|降价|离开|成立)"
        )
        for index, line in enumerate(lines):
            parts = [part for part in emphasis.split(line) if part]
            widths = [draw.textbbox((0, 0), part, font=font)[2] for part in parts]
            cursor_x = center_x - sum(widths) / 2
            for part, width in zip(parts, widths):
                fill = "#fbbf24" if emphasis.fullmatch(part) else "#f8fafc"
                draw.text((cursor_x, top + index * line_height), part, font=font, fill=fill, anchor="la")
                cursor_x += width
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def overlay_fixed_footer(visual_track: Path, footer_png: Path, output: Path) -> Path:
    """Legacy footer-only overlay kept for existing callers."""
    subprocess.run([
        "ffmpeg", "-y", "-i", str(visual_track), "-loop", "1", "-i", str(footer_png),
        "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto", "-map", "0:a?", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", "-c:a", "aac", str(output),
    ], check=True)
    return output


def compose_information_frame(
    manifest: RenderManifest, visual_track: Path, output: Path, font_path: Path | None = None,
    *, _allow_radar_effects: bool = True,
) -> Path:
    """Route flash to pinned rails and deep evidence to an expanded single-header viewport."""
    title = (manifest.fixed_title or manifest.scenes[0].caption).strip()
    radar = _is_radar_v2(manifest)
    expanded = _expanded_evidence_layout(manifest)
    if expanded:
        header_height, _ = _single_header_layout(title, font_path)
        pane_top = WECHAT_TOP_UI_SAFE + header_height + RADAR_META_RAIL
        bottom_top = 1920
        layout_profile = "expanded_single_header"
    else:
        top_height, _, _ = _information_layout(title, font_path)
        bottom_height, _, _ = _footer_layout(manifest.fixed_footer or "", font_path)
        pane_top = WECHAT_TOP_UI_SAFE + top_height + (RADAR_META_RAIL if radar else 0)
        bottom_top = 1920 - WECHAT_BOTTOM_UI_SAFE - bottom_height
        layout_profile = "radar_pinned_rails" if radar else "classic_pinned_rails"
    content_height = bottom_top - pane_top
    if content_height < 620:
        raise ValueError("selected information layout leaves less than 620px for source evidence")
    with TemporaryDirectory(prefix="video-factory-frame-") as temp:
        temp_root = Path(temp)
        static = (
            render_single_header_frame(title, temp_root / "fixed.png", font_path)
            if expanded else
            render_information_frame(title, manifest.fixed_footer or "", temp_root / "fixed.png", font_path)
        )
        inputs = ["-i", str(visual_track), "-loop", "1", "-i", str(static)]
        input_count = 2
        probe = probe_video(visual_track)
        if (probe.width, probe.height) == (1080, content_height):
            content_filter = "[0:v]setsar=1[content]"
        else:
            content_filter = f"[0:v]scale=1080:{content_height}:force_original_aspect_ratio=decrease,pad=1080:{content_height}:(ow-iw)/2:(oh-ih)/2:color=0x020815,setsar=1[content]"
        filters = [
            content_filter,
            f"[content]pad=1080:1920:0:{pane_top}:color=0x020815[base]",
            "[base][1:v]overlay=0:0:format=auto[v1]",
        ]
        previous = "[v1]"
        duration = manifest.duration
        sidecar = visual_track.with_suffix(".capture.json")
        if sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            duration = float(metadata.get("duration") or duration)
            source_size = (int(metadata.get("width") or probe.width), int(metadata.get("height") or probe.height))
            for shot in metadata.get("shots", []):
                start = float(shot["start"])
                end = float(shot["end"])
                translation = str(shot.get("translation") or "").strip()
                highlight_box = shot.get("highlight_box")
                source_region = None
                dense = _dense_evidence(shot, source_size)
                if (
                    radar and _allow_radar_effects and dense and highlight_box
                    and source_size[0] > 1080
                ):
                    source_region = _zoom_crop(highlight_box, source_size, content_height, factor=1.5)
                    crop_width, crop_height, crop_left, crop_top = source_region
                    zoom_index = input_count
                    input_count += 1
                    inputs.extend(["-i", str(visual_track)])
                    zoom_label = f"zoom{zoom_index}"
                    filters.append(
                        f"[{zoom_index}:v]crop={crop_width}:{crop_height}:{crop_left}:{crop_top},"
                        f"scale=1080:{content_height}:force_original_aspect_ratio=decrease,"
                        f"pad=1080:{content_height}:(ow-iw)/2:(oh-ih)/2:color=0x020815,setsar=1[{zoom_label}]"
                    )
                    output_label = f"[vz{zoom_index}]"
                    filters.append(
                        f"{previous}[{zoom_label}]overlay=0:{pane_top}:format=auto:"
                        f"enable='between(t,{start:.3f},{end:.3f})'{output_label}"
                    )
                    previous = output_label
                if radar and _allow_radar_effects and highlight_box:
                    spotlight = render_spotlight_overlay(
                        highlight_box, source_size, pane_top, content_height,
                        temp_root / f"spotlight-{input_count}.png", font_path,
                        source_region=source_region,
                        dense=dense,
                    )
                    spotlight_index = input_count
                    input_count += 1
                    inputs.extend(["-loop", "1", "-i", str(spotlight)])
                    output_label = f"[vs{spotlight_index}]"
                    filters.append(
                        f"{previous}[{spotlight_index}:v]overlay=0:0:format=auto:"
                        f"enable='between(t,{start:.3f},{end:.3f})'{output_label}"
                    )
                    previous = output_label
                if translation:
                    gloss = render_adjacent_gloss(
                        translation, temp_root / f"gloss-{input_count}.png", font_path,
                        highlight_box=highlight_box, source_size=source_size,
                        pane_top=pane_top, pane_height=content_height,
                        source_region=source_region,
                        profile=(InformationRenderProfile.RADAR_V2.value if radar else InformationRenderProfile.CLASSIC.value),
                    )
                    gloss_index = input_count
                    input_count += 1
                    inputs.extend(["-loop", "1", "-i", str(gloss)])
                    output_label = f"[vg{gloss_index}]"
                    filters.append(
                        f"{previous}[{gloss_index}:v]overlay=0:0:format=auto:"
                        f"enable='between(t,{start:.3f},{end:.3f})'{output_label}"
                    )
                    previous = output_label
        identifier = _direct_identifier(manifest) if radar else ""
        category_label = factual_category_label(manifest) if radar else ""
        if radar:
            badge = render_direct_identifier_badge(
                identifier, pane_top - RADAR_META_RAIL,
                temp_root / "direct-identifier.png", font_path,
                category_label=category_label,
            )
            badge_index = input_count
            inputs.extend(["-loop", "1", "-i", str(badge)])
            filters.append(f"{previous}[{badge_index}:v]overlay=0:0:format=auto[vbadge]")
            previous = "[vbadge]"
        filters.append(f"{previous}scale=in_range=pc:out_range=tv,format=pix_fmts=yuv420p[vout]")
        output.parent.mkdir(parents=True, exist_ok=True)
        browser_output = temp_root / "browser-framed.mp4"
        try:
            subprocess.run([
                "ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[vout]",
                "-t", f"{duration:.3f}", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-color_range", "tv", "-r", "25", str(browser_output),
            ], check=True)
        except subprocess.CalledProcessError as error:
            if not (radar and _allow_radar_effects):
                raise
            output.parent.mkdir(parents=True, exist_ok=True)
            output.with_suffix(".render-fallback.json").write_text(
                json.dumps({
                    "stage": "radar_visual_effects", "status": "fallback",
                    "fallback": "radar_layout_without_spotlight_or_focus_crop",
                    "error": f"{type(error).__name__}: {error}",
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return compose_information_frame(
                manifest, visual_track, output, font_path, _allow_radar_effects=False,
            )
        if manifest.github_brief and manifest.fixed_hook:
            cold_open = render_github_cold_open(
                manifest, visual_track, temp_root / "cold-open.mp4", temp_root, font_path,
            )
            subprocess.run([
                "ffmpeg", "-y", "-i", str(cold_open), "-i", str(browser_output),
                "-filter_complex",
                "[0:v]settb=AVTB,fps=25,format=yuv420p[v0];"
                "[1:v]settb=AVTB,fps=25,format=yuv420p[v1];"
                "[v0][v1]concat=n=2:v=1:a=0,format=yuv420p[v]",
                "-map", "[v]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", str(output),
            ], check=True)
        else:
            shutil.copy2(browser_output, output)
        output.with_suffix(".layout.json").write_text(
            json.dumps({
                "version": 1, "profile": layout_profile,
                "pane": {"top": pane_top, "bottom": bottom_top, "height": content_height},
                "metadata_rail": RADAR_META_RAIL if radar else 0,
                "direct_identifier": identifier,
                "category_label": category_label,
                "link_hint": "原帖/仓库链接见文案 ↗",
                "spotlight_opacity": {"normal": 0.40, "dense": 0.55} if radar else None,
                "focus_crop_factor": 1.5 if radar and _allow_radar_effects else None,
                "effects_enabled": radar and _allow_radar_effects,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return output
