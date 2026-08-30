from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .github_editor import select_github_focuses
from .models import CaptureCue, CueAction, GitHubProjectBrief, GitHubWalkthrough, RenderManifest


@dataclass(frozen=True, slots=True)
class WebCaptureRequest:
    url: str
    cues: list[CaptureCue]
    output: Path
    storyboard_dir: Path
    width: int = 1080
    height: int = 1250
    fps: int = 25
    cursor: bool = False


@dataclass(frozen=True, slots=True)
class WebScrollVideoSettings:
    root: Path
    node: str = "node"
    highlight_gap: int = 8

    @classmethod
    def from_environment(cls) -> "WebScrollVideoSettings":
        return cls(Path(os.environ.get("WEB_SCROLL_VIDEO_ROOT", "../web-scroll-video")).resolve())


class WebScrollVideoAdapter:
    """Translate director capture cues to web-scroll-video's editable cue sheet."""

    def __init__(self, settings: WebScrollVideoSettings):
        self.settings = settings

    def cue_text(self, request: WebCaptureRequest) -> str:
        lines = [
            f"out: {request.output.name}", f"width: {request.width}", f"height: {request.height}",
            f"fps: {request.fps}", f"cursor: {'on' if request.cursor else 'off'}", "", f"go {request.url}",
        ]
        for cue in request.cues:
            lines.append(self._encode_cue(cue))
        return "\n".join(lines) + "\n"

    def write_cue(self, request: WebCaptureRequest) -> Path:
        request.output.parent.mkdir(parents=True, exist_ok=True)
        path = request.output.with_suffix(".cue")
        path.write_text(self.cue_text(request), encoding="utf-8")
        return path

    def build_command(self, cue_path: Path, storyboard_dir: Path, runner_path: Path | None = None) -> list[str]:
        return [
            self.settings.node, str(runner_path or self.settings.root / "src/scroll-video.mjs"), "--script", str(cue_path),
            "--storyboard", str(storyboard_dir),
        ]

    def _write_padded_runner(self, cue_path: Path) -> Path:
        """Keep the yellow outline outside text without modifying the dependency checkout."""
        source_path = self.settings.root / "src/scroll-video.mjs"
        source = source_path.read_text(encoding="utf-8")
        marker = "border: 6px solid #ffd400; box-shadow:"
        replacement = (
            f"border: 0 !important; outline: 6px solid #ffd400 !important; "
            f"outline-offset: {self.settings.highlight_gap}px !important; box-shadow:"
        )
        if marker not in source:
            raise RuntimeError("web-scroll-video highlight CSS changed; padded runner patch must be reviewed")
        resolver_marker = """      candidates.sort((a, b) => a.area - b.area);
      element = candidates[0]?.element || null;
      return finish(element);"""
        resolver_replacement = """      const minimumReadableWidth = wanted.length <= 12 ? 24 : 100;
      candidates.sort((a, b) => {
        const aRect = a.element.getBoundingClientRect();
        const bRect = b.element.getBoundingClientRect();
        const aPenalty = aRect.width >= minimumReadableWidth ? 0 : 1;
        const bPenalty = bRect.width >= minimumReadableWidth ? 0 : 1;
        return aPenalty - bPenalty || a.area - b.area;
      });
      element = candidates[0]?.element || null;
      if (element && wanted === "in / out price") {
        let cursor = element;
        let priceCard = element;
        for (let depth = 0; depth < 6 && cursor.parentElement; depth += 1) {
          cursor = cursor.parentElement;
          const text = normalize(cursor.innerText || cursor.textContent || "");
          const rect = cursor.getBoundingClientRect();
          if (rect.width > 650 || rect.height > 220) break;
          if (text.includes(wanted) && text.includes("per 1m") && text.includes("$")) {
            priceCard = cursor;
          }
        }
        element = priceCard;
      }
      return finish(element);"""
        if resolver_marker not in source:
            raise RuntimeError("web-scroll-video text resolver changed; readable-target patch must be reviewed")
        show_marker = """      highlight.style.left = ${JSON.stringify(box.left)} + "px";
      highlight.style.top = ${JSON.stringify(box.top)} + "px";
      highlight.style.width = ${JSON.stringify(box.width)} + "px";
      highlight.style.height = ${JSON.stringify(box.height)} + "px";
      highlight.style.display = "block";"""
        show_replacement = f"""      highlight.style.left = ${{JSON.stringify(box.left)}} + "px";
      highlight.style.top = ${{JSON.stringify(box.top)}} + "px";
      highlight.style.width = ${{JSON.stringify(box.width)}} + "px";
      highlight.style.height = ${{JSON.stringify(box.height)}} + "px";
      highlight.style.setProperty("position", "fixed", "important");
      highlight.style.setProperty("display", "block", "important");
      highlight.style.setProperty("border", "0", "important");
      highlight.style.setProperty("outline", "6px solid #ffd400", "important");
      highlight.style.setProperty("outline-offset", "{self.settings.highlight_gap}px", "important");
      highlight.style.setProperty("background", "rgba(255,212,0,.10)", "important");
      highlight.style.setProperty("z-index", "2147483646", "important");"""
        if show_marker not in source:
            raise RuntimeError("web-scroll-video highlight runtime changed; inline visibility patch must be reviewed")
        runner = cue_path.with_suffix(".runner.mjs")
        runner.write_text(
            source.replace(marker, replacement, 1)
            .replace(resolver_marker, resolver_replacement, 1)
            .replace(show_marker, show_replacement, 1),
            encoding="utf-8",
        )
        return runner

    def capture(self, request: WebCaptureRequest) -> Path:
        working = request
        repairs: list[dict[str, str]] = []
        highlight_boxes: dict[str, dict[str, int]] = {}
        runner_strategy = "patched_runner"
        fallback_reason = ""
        prefer_upstream_runner = False
        for attempt in range(4):
            cue_path = self.write_cue(working)
            working.storyboard_dir.mkdir(parents=True, exist_ok=True)
            runner_path: Path | None = None
            if not prefer_upstream_runner:
                try:
                    runner_path = self._write_padded_runner(cue_path)
                except RuntimeError as error:
                    prefer_upstream_runner = True
                    runner_strategy = "upstream_unpatched_runner"
                    fallback_reason = str(error)
                    repairs.append({
                        "kind": "runner_patch_incompatible",
                        "error": str(error),
                        "repair": "run the pinned upstream capture runner without optional visual patches",
                    })
            try:
                subprocess.run(
                    self.build_command(cue_path, working.storyboard_dir, runner_path), check=True,
                )
            except subprocess.CalledProcessError as error:
                missing = self._missing_text_from_error(cue_path)
                if not missing and not prefer_upstream_runner:
                    prefer_upstream_runner = True
                    runner_strategy = "upstream_unpatched_runner"
                    fallback_reason = f"patched runner exited {error.returncode}"
                    repairs.append({
                        "kind": "patched_runner_failed",
                        "error": fallback_reason,
                        "repair": "retry once with the pinned upstream capture runner",
                    })
                    continue
                if attempt >= 3 or not missing:
                    raise
                repaired = self._repair_missing_text_request(working, missing)
                if repaired.cues == working.cues:
                    raise
                repairs.append({
                    "kind": "missing_visible_text",
                    "missing_target": missing,
                    "repair": "scroll to source-page bottom and hold without a false highlight",
                })
                working = repaired
                continue
            if not working.output.is_file():
                raise RuntimeError(f"web-scroll-video did not create {working.output}")
            try:
                highlight_boxes = self._validate_recorded_highlights(working)
                break
            except RuntimeError as error:
                shot_id = self._unreadable_highlight_shot(str(error))
                if attempt >= 3 or not shot_id:
                    raise
                repaired = self._repair_unreadable_highlight_request(working, shot_id)
                if repaired.cues == working.cues:
                    raise
                repairs.append({
                    "kind": "unreadable_highlight",
                    "shot_id": shot_id,
                    "repair": "keep the source-page position and hold without a tiny or missing highlight",
                })
                working = repaired
        else:
            raise RuntimeError("web-scroll-video exhausted deterministic capture repairs")
        self._write_capture_metadata(
            working, highlight_boxes, runner_strategy=runner_strategy,
            fallback_reason=fallback_reason,
        )
        if repairs:
            working.output.with_suffix(".capture-repairs.json").write_text(
                json.dumps(repairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
        return working.output

    @staticmethod
    def _missing_text_from_error(cue_path: Path) -> str:
        path = cue_path.with_suffix(".error.json")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        message = str(payload.get("message") or "")
        match = re.search(r'^Could not find text "(.+)"$', message, re.DOTALL)
        return match.group(1) if match else ""

    @staticmethod
    def _repair_missing_text_request(
        request: WebCaptureRequest, missing: str,
    ) -> WebCaptureRequest:
        normalized_missing = WebScrollVideoAdapter._visible_anchor(missing)
        cues: list[CaptureCue] = []
        for cue in request.cues:
            raw_target = str(cue.target or "").strip().strip('"')
            target = WebScrollVideoAdapter._visible_anchor(raw_target)
            if target != normalized_missing or cue.selector:
                cues.append(cue)
            elif cue.action == CueAction.SCROLL:
                cues.append(CaptureCue(
                    CueAction.SCROLL,
                    cue.instruction + " (missing-text fallback)",
                    target="bottom", wait_ms=cue.wait_ms, shot_id=cue.shot_id,
                    translation=cue.translation,
                ))
            elif cue.action == CueAction.HIGHLIGHT:
                cues.append(CaptureCue(
                    CueAction.WAIT,
                    cue.instruction + " (source-page hold; no false highlight)",
                    wait_ms=cue.wait_ms, shot_id=cue.shot_id,
                    translation=cue.translation,
                ))
            else:
                cues.append(cue)
        return WebCaptureRequest(
            request.url, cues, request.output, request.storyboard_dir,
            request.width, request.height, request.fps, request.cursor,
        )

    @staticmethod
    def _unreadable_highlight_shot(message: str) -> str:
        match = re.search(
            r"visual gate: (?:highlight for|yellow highlight missing for) ([^: ]+)", message,
        )
        return match.group(1) if match else ""

    @staticmethod
    def _repair_unreadable_highlight_request(
        request: WebCaptureRequest, shot_id: str,
    ) -> WebCaptureRequest:
        cues = [
            CaptureCue(
                CueAction.WAIT,
                cue.instruction + " (source-page hold; unreadable highlight removed)",
                wait_ms=cue.wait_ms, shot_id=cue.shot_id,
                translation=cue.translation,
            )
            if cue.action == CueAction.HIGHLIGHT and cue.shot_id == shot_id else cue
            for cue in request.cues
        ]
        return WebCaptureRequest(
            request.url, cues, request.output, request.storyboard_dir,
            request.width, request.height, request.fps, request.cursor,
        )

    @staticmethod
    def _cue_duration(cue: CaptureCue) -> float:
        if cue.action in {CueAction.WAIT, CueAction.SCROLL, CueAction.HIGHLIGHT, CueAction.ZOOM}:
            return (cue.wait_ms or 1000) / 1000
        return 0.0

    @classmethod
    def capture_metadata(
        cls, request: WebCaptureRequest, highlight_boxes: dict[str, dict[str, int]] | None = None,
    ) -> dict[str, object]:
        cursor = 0.0
        shots: list[dict[str, object]] = []
        for cue in request.cues:
            duration = cls._cue_duration(cue)
            if cue.shot_id:
                shot = {
                    "id": cue.shot_id,
                    "action": cue.action,
                    "start": round(cursor, 3),
                    "end": round(cursor + duration, 3),
                    "target": cue.target,
                    "selector": cue.selector,
                    "translation": cue.translation or "",
                }
                if highlight_boxes and cue.shot_id in highlight_boxes:
                    shot["highlight_box"] = highlight_boxes[cue.shot_id]
                shots.append(shot)
            cursor += duration
        return {
            "version": 1,
            "width": request.width,
            "height": request.height,
            "duration": round(cursor, 3),
            "shots": shots,
        }

    @classmethod
    def _write_capture_metadata(
        cls, request: WebCaptureRequest, highlight_boxes: dict[str, dict[str, int]] | None = None,
        *, runner_strategy: str = "patched_runner", fallback_reason: str = "",
    ) -> Path:
        sidecar = request.output.with_suffix(".capture.json")
        metadata = cls.capture_metadata(request, highlight_boxes)
        metadata["runner_strategy"] = runner_strategy
        metadata["fallback_used"] = runner_strategy != "patched_runner"
        if fallback_reason:
            metadata["fallback_reason"] = fallback_reason
        sidecar.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        return sidecar

    @staticmethod
    def _validate_recorded_highlights(request: WebCaptureRequest) -> dict[str, dict[str, int]]:
        """Sample the actual MP4; storyboard frames are written after highlights hide."""
        from PIL import Image

        metadata = WebScrollVideoAdapter.capture_metadata(request)
        highlighted = [shot for shot in metadata["shots"] if shot["action"] == CueAction.HIGHLIGHT]
        if not highlighted:
            return {}
        highlight_boxes: dict[str, dict[str, int]] = {}
        for shot in highlighted:
            frame = request.storyboard_dir / f"gate-{shot['id']}.png"
            midpoint = (float(shot["start"]) + float(shot["end"])) / 2
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{midpoint:.3f}",
                "-i", str(request.output), "-frames:v", "1", str(frame),
            ], check=True)
            with Image.open(frame).convert("RGB") as image:
                if image.size != (request.width, request.height):
                    raise RuntimeError(f"visual gate: wrong recorded size for {shot['id']}: {image.size}")
                yellow_count, highlight_box = WebScrollVideoAdapter._main_highlight_geometry(image)
                highlight_width = highlight_box["width"]
            if yellow_count < 300:
                raise RuntimeError(f"visual gate: yellow highlight missing for {shot['id']}")
            minimum_width = 35 if shot["id"] == "file_tree" else 100
            if highlight_width < minimum_width:
                raise RuntimeError(
                    f"visual gate: highlight for {shot['id']} is too small ({highlight_width}px); target a readable line, not an icon or control"
                )
            highlight_boxes[str(shot["id"])] = highlight_box
        return highlight_boxes

    @staticmethod
    def _main_highlight_width(image) -> tuple[int, int]:
        total, box = WebScrollVideoAdapter._main_highlight_geometry(image)
        return total, box["width"]

    @staticmethod
    def _main_highlight_geometry(image) -> tuple[int, dict[str, int]]:
        pixels = image.load()
        yellow_points = {
            (x, y)
            for y in range(image.height)
            for x in range(image.width)
            for red, green, blue in (pixels[x, y],)
            if red > 210 and green > 155 and blue < 100
        }
        if not yellow_points:
            return 0, {"left": 0, "top": 0, "width": 0, "height": 0}
        components: list[list[tuple[int, int]]] = []
        while yellow_points:
            seed = yellow_points.pop()
            queue = [seed]
            component = [seed]
            while queue:
                x, y = queue.pop()
                for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if neighbor in yellow_points:
                        yellow_points.remove(neighbor)
                        queue.append(neighbor)
                        component.append(neighbor)
            components.append(component)
        main_component = max(components, key=len)
        left = min(point[0] for point in main_component)
        top = min(point[1] for point in main_component)
        right = max(point[0] for point in main_component)
        bottom = max(point[1] for point in main_component)
        return len(main_component), {
            "left": left, "top": top, "width": right - left, "height": bottom - top,
        }

    @staticmethod
    def github_audit_request(
        repo_url: str, walkthrough: GitHubWalkthrough, output: Path, storyboard_dir: Path, target_duration: float | None = None,
    ) -> WebCaptureRequest:
        """Mandatory real-browser route for every GitHub project walkthrough."""
        fixed_duration = 1.8 + 1.6 + len(walkthrough.key_modules) * (1.6 + 1.8)
        readme_scroll_seconds = max(15.0, (target_duration or 0) - fixed_duration)
        cues = [
            CaptureCue(CueAction.WAIT, "hold complete repository home and file tree", wait_ms=1800),
            CaptureCue(CueAction.SCROLL, "enter README from its visible anchor", target="README", wait_ms=1600),
            CaptureCue(CueAction.SCROLL, "continuously reach README end", target="bottom", wait_ms=round(readme_scroll_seconds * 1000)),
        ]
        for module in walkthrough.key_modules:
            visible_anchor = WebScrollVideoAdapter._visible_anchor(module.anchor)
            cues.extend([
                CaptureCue(CueAction.SCROLL, f"jump to real module anchor: {visible_anchor}", target=f'"{visible_anchor}"', wait_ms=1600),
                CaptureCue(CueAction.HIGHLIGHT, f"highlight real module anchor: {visible_anchor}", target=f'"{visible_anchor}"', wait_ms=1800),
            ])
        return WebCaptureRequest(repo_url, cues, output, storyboard_dir)

    @staticmethod
    def _adjacent_translation(source: str, translation: str) -> str:
        """Chinese source text already explains itself; do not duplicate it as a gloss."""
        if re.search(r"[\u4e00-\u9fff]", source):
            return ""
        return translation.strip()

    @staticmethod
    def github_story_request(
        repo_url: str, brief: GitHubProjectBrief, output: Path, storyboard_dir: Path, target_duration: float | None = None,
    ) -> WebCaptureRequest:
        """Final-cut route: complete identity, core claim, and two ranked proofs.

        The full README traversal is retained by ``github_audit_request`` as
        an auditable source artifact. It must not turn the final video into a
        disorienting top-to-bottom screen recording.
        """
        focuses = select_github_focuses(brief)
        if len(focuses) != 2:
            raise ValueError("GitHub story route needs exactly two ranked focuses")
        target = max(8.0, min(25.0, target_duration or 20.0))
        jump = 0.3
        durations = {
            "overview": 1.0, "repo_name": 1.0, "description": 1.5,
            "file_tree": 1.0, "readme": 2.0, "focus_1": 3.0, "focus_2": 3.0,
        }
        transition_total = 0.2 + jump * 3
        base_duration = sum(durations.values())
        if target < transition_total + base_duration:
            scale = max(0.0, (target - transition_total) / base_duration)
            durations = {key: value * scale for key, value in durations.items()}
        remaining = target - transition_total - sum(durations.values())
        for key, maximum in (("focus_1", 5.0), ("focus_2", 5.0), ("readme", 5.0), ("description", 3.0), ("overview", 3.0), ("file_tree", 2.0), ("repo_name", 2.0)):
            if remaining <= 0:
                break
            addition = min(remaining, maximum - durations[key])
            durations[key] += addition
            remaining -= addition
        description_target = WebScrollVideoAdapter._visible_anchor(
            brief.repo_description_target,
        )
        description_visible = description_target.casefold() not in {
            "", "no repository description", "no description", "none", "null",
        }
        if not description_visible:
            # GitHub's API placeholder is evidence about missing metadata, not
            # text rendered on the repository page. Preserve duration without
            # asking the browser to find an element that cannot exist.
            durations["overview"] += durations["description"]
        cues = [
            CaptureCue(CueAction.SCROLL, "return to repository top", target="top", wait_ms=200),
            CaptureCue(CueAction.WAIT, "hold the complete repository overview", wait_ms=round(durations["overview"] * 1000), shot_id="repo_overview"),
            CaptureCue(CueAction.HIGHLIGHT, "identify the repository", selector='strong[itemprop="name"] a', wait_ms=round(durations["repo_name"] * 1000), shot_id="repo_name"),
        ]
        if description_visible:
            cues.append(CaptureCue(
                CueAction.HIGHLIGHT, "state the repository purpose", target=brief.repo_description_target,
                wait_ms=round(durations["description"] * 1000), shot_id="repo_description",
                translation=WebScrollVideoAdapter._adjacent_translation(
                    brief.repo_description_target, brief.repo_description_translation,
                ),
            ))
        cues.extend([
            CaptureCue(
                CueAction.HIGHLIGHT, "show a representative file in the real file tree", target=brief.file_tree_target,
                wait_ms=round(durations["file_tree"] * 1000), shot_id="file_tree",
            ),
            CaptureCue(CueAction.SCROLL, "jump to the exact README project claim", target=brief.readme_claim_target, wait_ms=round(jump * 1000)),
            CaptureCue(
                CueAction.HIGHLIGHT, "prove the core project claim", target=brief.readme_claim_target,
                wait_ms=round(durations["readme"] * 1000), shot_id="readme_claim",
                translation=WebScrollVideoAdapter._adjacent_translation(
                    brief.readme_claim_target, brief.readme_claim_translation,
                ),
            ),
        ])
        current_url = repo_url
        for index, focus in enumerate(focuses, start=1):
            capture_target = focus.browser_target or focus.target
            translation = focus.browser_translation
            if not translation and capture_target == focus.target:
                translation = focus.translation
            elif not translation and not re.search(r"[\u4e00-\u9fff]", capture_target):
                translation = focus.translation
            translation = WebScrollVideoAdapter._adjacent_translation(capture_target, translation)
            focus_url = focus.source_url or repo_url
            if focus_url != current_url:
                cues.append(CaptureCue(CueAction.OPEN, f"open selected proof source: {focus.id}", value=focus_url))
                current_url = focus_url
            cues.extend([
                CaptureCue(CueAction.SCROLL, f"jump to selected proof: {focus.id}", target=capture_target, wait_ms=round(jump * 1000)),
                CaptureCue(
                    CueAction.HIGHLIGHT, f"hold selected proof: {focus.id}", target=capture_target,
                    wait_ms=round(durations[f"focus_{index}"] * 1000), shot_id=f"focus_{index}",
                    translation=translation,
                ),
            ])
        # A larger browser viewport is the equivalent of 78% page zoom, but
        # keeps DOM highlight coordinates exact. Composition scales this
        # matching-aspect viewport into the 1080x1250 middle pane.
        return WebCaptureRequest(repo_url, cues, output, storyboard_dir, width=1384, height=1602)

    # Retained for callers that explicitly need the full README audit asset.
    github_request = github_audit_request

    @staticmethod
    def editorial_story_request(
        source_url: str, manifest: RenderManifest, output: Path, storyboard_dir: Path,
    ) -> WebCaptureRequest:
        """Compile deterministic scene cues for every non-GitHub story."""
        if not manifest.editorial_brief:
            raise ValueError("editorial browser capture requires editorial_brief")
        cues: list[CaptureCue] = []
        for scene in manifest.scenes:
            cues.extend(scene.recording_cues)
        if not cues:
            cues.append(CaptureCue(CueAction.WAIT, "hold the primary source", wait_ms=round(manifest.duration * 1000)))
        return WebCaptureRequest(source_url, cues, output, storyboard_dir, width=1384, height=1602)

    @staticmethod
    def linked_post_request(
        post_url: str, primary_url: str, primary_anchors: list[str], output: Path, storyboard_dir: Path,
        target_duration: float | None = None,
    ) -> WebCaptureRequest:
        """Record a post as the entry point, then naturally open its primary source."""
        fixed_duration = 5.0 + 1.8 + len(primary_anchors) * 3.4
        dwell = max(1.6, ((target_duration or fixed_duration) - fixed_duration) / max(len(primary_anchors), 1))
        cues = [
            CaptureCue(CueAction.WAIT, "hold the complete original post before extending it", wait_ms=5000),
            CaptureCue(CueAction.OPEN, "open the primary source mentioned by the post", value=primary_url),
            CaptureCue(CueAction.WAIT, "hold the primary page heading and opening claim", wait_ms=1800),
        ]
        for anchor in primary_anchors:
            visible_anchor = WebScrollVideoAdapter._visible_anchor(anchor)
            cues.extend([
                CaptureCue(CueAction.SCROLL, f"navigate to primary-page anchor: {visible_anchor}", target=f'"{visible_anchor}"', wait_ms=round(dwell * 1000)),
                CaptureCue(CueAction.HIGHLIGHT, f"highlight primary-page anchor: {visible_anchor}", target=f'"{visible_anchor}"', wait_ms=1800),
            ])
        return WebCaptureRequest(post_url, cues, output, storyboard_dir)

    @staticmethod
    def _visible_anchor(anchor: str) -> str:
        """README evidence is Markdown; browser targets must be rendered text."""
        rendered = re.sub(r"^\s{0,3}#{1,6}\s*", "", anchor).strip()
        # Browser text lookup is element-scoped. A model may quote several
        # adjacent rendered lines as one evidence string even though the DOM
        # exposes them as separate elements; use the first concrete line as a
        # stable, still evidence-bound anchor.
        lines = [re.sub(r"\s+", " ", line).strip() for line in rendered.splitlines()]
        return next((line for line in lines if len(line) >= 3), rendered)

    @staticmethod
    def _encode_cue(cue: CaptureCue) -> str:
        def target_expression() -> str:
            if cue.selector:
                return "selector " + json.dumps(cue.selector, ensure_ascii=False)
            raw = (cue.target or "").strip()
            if len(raw) >= 2 and raw[0] == raw[-1] == '"':
                raw = raw[1:-1]
            raw = WebScrollVideoAdapter._visible_anchor(raw)
            return "text " + json.dumps(raw, ensure_ascii=False)

        if cue.action == CueAction.WAIT:
            return f"pause {(cue.wait_ms or 1000) / 1000:g}"
        if cue.action == CueAction.WAIT_FOR:
            return f"wait {target_expression()} timeout {(cue.wait_ms or 10000) / 1000:g}"
        if cue.action == CueAction.SCROLL:
            if cue.selector:
                raise ValueError("web-scroll-video text cue format cannot scroll to a selector; use visible text")
            if (cue.target or "bottom") in {"top", "bottom"}:
                target = cue.target or "bottom"
            else:
                raw = (cue.target or "").strip()
                if len(raw) >= 2 and raw[0] == raw[-1] == '"':
                    raw = raw[1:-1]
                raw = WebScrollVideoAdapter._visible_anchor(raw)
                target = json.dumps(raw, ensure_ascii=False)
            return f"scroll to {target} over {(cue.wait_ms or 1000) / 1000:g}"
        if cue.action == CueAction.HIGHLIGHT:
            if not (cue.selector or cue.target):
                raise ValueError("highlight cue needs visible text or a selector")
            return f"highlight {target_expression()} for {(cue.wait_ms or 1000) / 1000:g}"
        if cue.action == CueAction.CLICK:
            if not (cue.selector or cue.target):
                raise ValueError("click cue needs visible text or a selector")
            return f"click {target_expression()}"
        if cue.action == CueAction.TYPE:
            if not cue.target or cue.value is None:
                raise ValueError("type cue needs a target and value")
            return f'type "{cue.value}" into "{cue.target}"'
        if cue.action == CueAction.ZOOM:
            if cue.value is None:
                raise ValueError("zoom cue needs a zoom value")
            return f"zoom to {cue.value} over {(cue.wait_ms or 1000) / 1000:g}"
        if cue.action == CueAction.OPEN:
            return f"go {cue.value or cue.target or cue.instruction}"
        raise ValueError(f"unsupported web-scroll-video cue: {cue.action}")
