from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .agent import AgentBudget, BoundedContentAgent, ContentAgentError, LinkedSourceResearchTool
from .acquisition import AcquisitionResult, URLAcquirer
from .compositor import compose_information_frame
from .editorial import canonicalize_editorial_brief, compile_evidence_shots, route_content
from .ingest import GitHubIngestor, IngestResult
from .github_context import enrich_github_context
from .github_editor import canonicalize_github_brief
from .llm import LLMSettings, OpenAICompatibleStoryWriter
from .media import probe_video, validate_wechat_mp4
from .models import ContentType, CueAction, Evidence, EvidenceShotKind, Scene, TopicType
from .mpt import MPTAssemblyAdapter, MPTSettings
from .multimodal import OpenRouterVisualAnalyst, find_high_value_visuals
from .openrouter import ModelQuote, ModelRequirements, OpenRouterCatalog
from .quality import is_publishable, validate_manifest
from .research import DirectorContextToolbox
from .storage import Workspace
from .serde import load_manifest
from .webcapture import WebCaptureRequest, WebScrollVideoAdapter, WebScrollVideoSettings
from .writer import StoryWriterPacket
from .tracks import TrackSegment, build_crossfade_track
from .tweetcard import render_editorial_card, render_source_image, render_tweet_card, tweet_card_video


@dataclass(frozen=True, slots=True)
class GenerateOptions:
    provider: str = "auto"
    model: str | None = None
    duration: float | None = None
    render: bool = True
    refresh_prices: bool = False
    topic: TopicType | None = None
    content_type: ContentType | None = None
    research: bool = True
    live_capture: bool = True
    refresh: bool = False
    youtube_media: str | None = None
    youtube_subtitles: str | None = None
    youtube_translation_plan: str | None = None
    youtube_editorial_mode: str = "auto"
    linked_sources: tuple[str, ...] = ()
    supplemental_context: str | None = None
    price_event_metadata: dict[str, object] | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class VideoFactory:
    """The actual one-URL production program; Codex is not part of this loop."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.workspace.initialize()
        self.jobs_dir = workspace.root / "jobs"
        self.cache_dir = workspace.root / "cache"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, url: str, options: GenerateOptions) -> dict[str, object]:
        source = self._classify(url)
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", url.rstrip("/").rsplit("/", 1)[-1]).strip("-") or "video"
        job_id = f"{slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        job = self.jobs_dir / job_id
        job.mkdir(parents=True)
        result: dict[str, object] = {
            "job_id": job_id, "url": url, "source_type": source,
            "started_at": _now(), "stages": [], "status": "running",
        }
        self._write_result(job, result)
        try:
            if source == "youtube":
                from .youtube import YouTubeCollectionFactory

                writer, selection = self._translation_writer(options)
                generated = YouTubeCollectionFactory(self.workspace, writer).generate(
                    url, job, render=options.render,
                    local_media=Path(options.youtube_media).resolve() if options.youtube_media else None,
                    local_subtitles=(
                        Path(options.youtube_subtitles).resolve() if options.youtube_subtitles else None
                    ),
                    translation_plan=(
                        Path(options.youtube_translation_plan).resolve()
                        if options.youtube_translation_plan else None
                    ),
                    editorial_mode=options.youtube_editorial_mode,
                )
                result.update(generated)
                result["model_selection"] = selection
            else:
                self._generate_cached_manifest(source, url, job, options, result)
            result["status"] = "completed"
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
            self._write_result(job, result)
            raise
        result["completed_at"] = _now()
        self._write_result(job, result)
        return result

    def _generate_cached_manifest(
        self, source: str, url: str, job: Path, options: GenerateOptions,
        result: dict[str, object],
    ) -> None:
        cache = self._generation_cache_path(url, options)

        def generate_fresh() -> None:
            if source == "github":
                self._generate_github(url, job, options, result)
            else:
                self._generate_editorial(url, job, options, result)

        def store_cache(status: str) -> None:
            if not result.get("manifest"):
                return
            cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(result["manifest"]), cache)
            result["stages"].append({"name": "generation_cache", "status": status, "manifest": str(cache)})

        if not cache.is_file() or options.refresh:
            generate_fresh()
            store_cache("stored")
            return

        manifest = load_manifest(cache)
        cached_checks = validate_manifest(manifest, self.workspace.root)
        blocking = [
            check for check in cached_checks if not check.passed
            and check.name not in {"music_license_record", "editorial_safety_review"}
        ]
        if blocking:
            result["stages"].append({
                "name": "generation_cache", "status": "rejected_stale",
                "manifest": str(cache), "errors": [check.detail for check in blocking],
            })
            generate_fresh()
            store_cache("refreshed")
            return

        job_manifest = job / "manifest.json"
        shutil.copy2(cache, job_manifest)
        result["stages"].append({"name": "generation_cache", "status": "hit", "manifest": str(cache)})
        result["manifest"] = str(job_manifest)
        if options.render:
            if source == "github":
                owner, repo = self._github_identity(url)
                self._render_github_manifest(
                    manifest, f"https://github.com/{owner}/{repo}", job, result,
                )
            else:
                self._render_editorial(manifest, url, job, result)
        else:
            result["checks"] = [check.to_dict() for check in cached_checks]
            result["publishable"] = is_publishable(cached_checks)

    def rerender(self, manifest_path: Path) -> dict[str, object]:
        """Retry deterministic rendering without reacquisition or LLM calls."""
        manifest = load_manifest(manifest_path.resolve())
        candidate = self.workspace.load_candidate(manifest.candidate_id)
        if manifest.github_brief:
            canonicalize_github_brief(manifest.github_brief, manifest.evidence)
        else:
            self._recompile_editorial_manifest(manifest, candidate)
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", candidate.id).strip("-") or "video"
        job_id = f"{slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        job = self.jobs_dir / job_id
        job.mkdir(parents=True)
        job_manifest = job / "manifest.json"
        job_manifest.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        result: dict[str, object] = {
            "job_id": job_id, "url": candidate.source_url, "source_type": self._classify(candidate.source_url),
            "started_at": _now(), "stages": [{
                "name": "content_reuse", "status": "ok", "manifest": str(manifest_path.resolve()),
                "llm_calls": 0, "acquisition_calls": 0,
            }], "status": "running", "manifest": str(job_manifest),
        }
        self._write_result(job, result)
        try:
            checks = validate_manifest(manifest, self.workspace.root)
            blocking = [
                item for item in checks if not item.passed
                and item.name not in {"music_license_record", "editorial_safety_review"}
            ]
            if blocking:
                raise ValueError("manifest cannot be rerendered: " + "; ".join(item.detail for item in blocking))
            if manifest.github_brief:
                self._render_github_manifest(manifest, candidate.source_url, job, result)
            else:
                self._render_editorial(manifest, candidate.source_url, job, result)
            result["status"] = "completed"
            result["completed_at"] = _now()
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
            self._write_result(job, result)
            raise
        self._write_result(job, result)
        return result

    @staticmethod
    def _recompile_editorial_manifest(manifest, candidate) -> None:
        brief = manifest.editorial_brief
        if brief is None:
            raise ValueError("rerender currently requires an editorial manifest")
        canonicalize_editorial_brief(brief, manifest.evidence)
        proposals = compile_evidence_shots(brief, candidate)
        scenes: list[Scene] = []
        cursor = 0.0
        for index, proposal in enumerate(proposals, start=1):
            duration = float(proposal.duration_hint or 2.0)
            scenes.append(Scene(
                id=f"scene-{index}", start=round(cursor, 3), end=round(cursor + duration, 3),
                narration=proposal.narration, caption=proposal.caption,
                evidence_ids=list(proposal.evidence_ids), material_role=proposal.material_role,
                visual_action=proposal.visual_action, overlay_labels=list(proposal.overlay_labels),
                stage_name=proposal.stage_name, sound_hint=proposal.sound_hint,
                recording_cues=list(proposal.recording_cues),
                screen_fact=(proposal.screen_fact or proposal.caption).strip(),
                screen_interpretation=(proposal.screen_interpretation or "").strip(),
                highlight_translation=proposal.highlight_translation,
                source_excerpt=proposal.source_excerpt, visual_family=proposal.visual_family,
                retention_job=proposal.retention_job,
            ))
            cursor += duration
        manifest.scenes = scenes

    @staticmethod
    def _canonicalize_price_event_copy(manifest, metadata: dict[str, object]) -> None:
        brief = manifest.editorial_brief
        if brief is None or len(brief.evidence_shots) < 3:
            raise ValueError("price-event story requires three evidence shots")
        price = next((
            item for item in manifest.evidence
            if item.source_kind in {"discovery:price_event", "discovery:price_context"}
        ), None)
        page = next((
            item for item in manifest.evidence if item.source_kind in {"web:primary_page", "web:page"}
        ), None)
        if price is None or page is None:
            raise ValueError("price-event story requires page and calculation evidence")

        hook = str(metadata.get("required_hook_zh") or "").strip()
        title = str(metadata.get("required_headline_zh") or "").strip()
        closing = str(metadata.get("editorial_verdict_zh") or "").strip()
        if not all((hook, title, closing)):
            raise ValueError("price-event metadata lacks required hook, headline, or closing")
        model_id = str(metadata.get("model_id") or "model")
        provider = str(metadata.get("provider") or "OpenRouter endpoint")
        prompt = float(metadata.get("page_prompt_per_m") or metadata.get("prompt_per_m") or 0)
        completion = float(metadata.get("page_completion_per_m") or metadata.get("completion_per_m") or 0)
        workload = float(metadata.get("video_workload_cost_usd") or metadata.get("workload_cost_usd") or 0)
        intelligence = float(metadata.get("intelligence_index") or 0)
        coding = float(metadata.get("coding_index") or 0)
        discount = int(metadata.get("discount_percent") or 0)
        use_case = str(metadata.get("use_case_zh") or "成本敏感的开发任务")
        description = str(metadata.get("model_description") or "").strip()
        description_excerpt = re.split(
            r"(?<=[.!?])\s+(?=[A-Z])", description, maxsplit=1,
        )[0].strip()
        if description_excerpt and description_excerpt[-1] not in ".!?":
            description_excerpt += "."
        comparison = metadata.get("official_comparison")
        comparison = comparison if isinstance(comparison, dict) else None

        manifest.fixed_hook = hook
        manifest.fixed_title = title
        manifest.fixed_footer = closing
        brief.headline = title
        brief.subheadline = hook
        brief.fixed_conclusion = closing
        strategy = brief.attention_strategy
        strategy.hook_fact = hook
        strategy.selected_hook = hook
        strategy.hook_candidates = [hook, title, closing]
        strategy.hook_evidence_ids = [price.id]
        strategy.stance = closing
        strategy.payoff = closing

        shots = brief.evidence_shots[:3]
        first, second, third = shots
        first.kind = EvidenceShotKind.BROWSER_SECTION
        first.question = "页面当前显示的真实价格异常是什么？"
        first.fact = (
            f"OpenRouter 页面显示 {model_id} 为 {discount}% off：输入 ${prompt:g}/M，输出 ${completion:g}/M。"
            if discount else
            f"OpenRouter 页面显示 {model_id} 的 {provider} 线路：输入 ${prompt:g}/M，输出 ${completion:g}/M。"
        )
        first.interpretation = "先用页面可见数字确认价格事件。"
        first.audience_copy = f"当前可靠线路：${prompt:g}/M 输入，${completion:g}/M 输出。"
        first.evidence_ids = [page.id]
        first.beat_ids = ["practical_change"]
        first.visual_family = "official_page"
        first.retention_job = "reveal"
        def display_price(value: float) -> str:
            return f"{value:.2f}" if value >= 0.01 else f"{value:.4f}".rstrip("0")
        first.target = f"${display_price(prompt)} / ${display_price(completion)} per 1M"

        second.kind = EvidenceShotKind.BROWSER_SECTION
        second.question = "同一开发任务实际要花多少钱？"
        if comparison:
            official_prompt = float(comparison.get("official_prompt_per_m") or 0)
            official_completion = float(comparison.get("official_completion_per_m") or 0)
            savings = float(comparison.get("savings_offpeak_percent") or 0)
            second.fact = (
                f"原厂谷时为 ${official_prompt:g}/M 输入、${official_completion:g}/M 输出；"
                f"同一 18K+4K 任务，{provider} 便宜 {savings:.1f}%。"
            )
            second.audience_copy = f"同模型、同任务：OpenRouter 可靠线路比原厂谷时省 {savings:.1f}%。"
            second.target = (
                f"The official-vendor endpoint costs ${official_prompt:.3f}/M input "
                f"and ${official_completion:.3f}/M output off-peak."
            )
        else:
            second.fact = f"按 18K 输入 + 4K 输出计算，一次典型开发任务约 ${workload:.4f}。"
            second.audience_copy = f"不是单价游戏：一次典型任务约 ${workload:.4f}。"
            second.target = (
                "A representative 18,000-input plus 4,000-output-token developer task costs about "
                f"${workload:.6f}."
            )
        second.interpretation = "把每百万 token 单价换成开发者能判断的任务成本。"
        second.evidence_ids = [price.id]
        second.beat_ids = ["practical_change"]
        second.visual_family = "stat_card"
        second.retention_job = "contrast"
        second.relation_to_previous = "从页面单价推进到同一工作负载的真实成本。"

        third.kind = EvidenceShotKind.BROWSER_SECTION
        third.question = "这个更便宜的选择适合放进哪些任务？"
        third.fact = f"更适合：{use_case}。"
        third.interpretation = "用模型描述给出正向使用场景与采用边界。"
        third.audience_copy = closing
        third.target = description_excerpt or (
            f"Artificial Analysis intelligence {intelligence:.1f}, and coding {coding:.1f}."
        )
        third.evidence_ids = [price.id]
        third.beat_ids = ["adoption_choice"]
        third.visual_family = "impact_card"
        third.retention_job = "payoff"
        third.relation_to_previous = "价格诱惑之后，补上能力门槛与采用结论。"
        brief.evidence_shots = shots

    @staticmethod
    def _canonicalize_price_event_visuals(manifest, source_url: str) -> None:
        """Keep API calculations off the browser-DOM path.

        The OpenRouter page remains the first visual proof. Calculated
        workload/vendor comparisons are rendered as cited editorial cards,
        because those exact sentences do not exist in the page DOM.
        """
        brief = manifest.editorial_brief
        if brief is None or not brief.evidence_shots:
            return
        price = next((
            item for item in manifest.evidence
            if item.source_kind in {"discovery:price_event", "discovery:price_context"}
        ), None)
        page = next((
            item for item in manifest.evidence
            if item.source_kind in {"web:primary_page", "web:page"}
            and item.url.rstrip("/") == source_url.rstrip("/")
        ), None)
        if price is None or page is None:
            return

        first = brief.evidence_shots[0]
        discount = re.search(r"\b\d{2,3}%\s*off\b", page.quote, re.IGNORECASE)
        title_anchor = re.search(r"^Title:\s*(.+?)\s+-\s+API Pricing", page.quote, re.MULTILINE)
        paired_price = re.search(
            r"\$\d+(?:\.\d+)?\s*/\s*\$\d+(?:\.\d+)?\s+per\s+1M",
            page.quote, re.IGNORECASE,
        )
        anchor = (
            "In / Out Price" if ("per 1M" in first.target or paired_price) else ""
            or (title_anchor.group(1).strip() if title_anchor else "")
            or (discount.group(0) if discount else "")
        )
        if anchor:
            first.kind = EvidenceShotKind.BROWSER_SECTION
            first.visual_family = "official_page"
            first.source_url = source_url
            first.target = anchor
            first.translation = "页面标出的输入 / 输出单价"
            first.full_translation = ""
            first.evidence_ids = [page.id]

        for index, shot in enumerate(brief.evidence_shots[1:], start=1):
            cited = [item for item in manifest.evidence if item.id in shot.evidence_ids]
            if shot.target and not any(shot.target.casefold() in item.quote.casefold() for item in cited):
                if shot.target.casefold() in price.quote.casefold():
                    shot.evidence_ids = list(dict.fromkeys([*shot.evidence_ids, price.id]))
            shot.kind = EvidenceShotKind.BROWSER_SECTION
            shot.visual_family = "stat_card" if index % 2 else "impact_card"

    @staticmethod
    def _price_focused_evidence(item: Evidence) -> Evidence:
        if item.source_kind not in {"web:primary_page", "web:page"}:
            return item
        if (urlparse(item.url).hostname or "").casefold() != "openrouter.ai":
            return item
        selected: list[str] = []
        kept_table_price = False
        for line in item.quote.splitlines():
            stripped = line.strip()
            if re.match(r"^##\s+Performance\b", stripped, re.IGNORECASE):
                break
            if stripped.startswith("|") and ("% off" in stripped.casefold() or "$" in stripped):
                if kept_table_price:
                    continue
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                stripped = " | ".join(cells[:4])
                kept_table_price = True
            if (
                re.search(
                    r"%\s*off|\$\d+(?:\.\d+)?\s*/\s*\$\d+(?:\.\d+)?\s+per\s+1M|"
                    r"\b(?:context|released|pricing|providers?|in / out price)\b",
                    stripped, re.IGNORECASE,
                )
                or stripped.startswith(("Title:", "# ", "## "))
            ) and not re.search(
                r"\b(?:average price|latency|throughput|ttft|tok/s|tps)\b",
                stripped, re.IGNORECASE,
            ):
                selected.append(stripped)
        excerpt = "\n".join(dict.fromkeys(line for line in selected if line))
        if len(excerpt) < 40:
            return item
        return replace(
            item, quote=excerpt,
            notes=(item.notes or "") + " Price-video writing excerpt; full captured_asset remains archived.",
        )

    def _render_github_manifest(
        self, manifest, repo_url: str, job: Path, result: dict[str, object],
    ) -> None:
        if manifest.github_brief is None:
            raise ValueError("GitHub rendering requires github_brief")
        self._apply_default_music_license(manifest)
        job_manifest = job / "manifest.json"
        browser = job / "github-browser.mp4"
        frames = job / "browser-frames"
        cold_open_duration = sum(beat.duration for beat in manifest.cold_open_beats)
        walkthrough_duration = max(0.0, manifest.duration - cold_open_duration)
        capture_request = WebScrollVideoAdapter.github_story_request(
            repo_url, manifest.github_brief, browser, frames, walkthrough_duration,
        )
        WebScrollVideoAdapter(WebScrollVideoSettings.from_environment()).capture(capture_request)
        result["stages"].append({"name": "browser_capture", "status": "ok", "video": str(browser)})
        framed = job / "framed-visual.mp4"
        compose_information_frame(manifest, browser, framed)
        result["stages"].append({"name": "compose", "status": "ok", "video": str(framed)})
        mastered = MPTAssemblyAdapter(MPTSettings.from_environment()).assemble(manifest, framed, job.name)
        final = job / "final.mp4"
        shutil.copy2(mastered, final)
        manifest.video_path = str(final.relative_to(self.workspace.root))
        self.workspace.save_manifest(manifest)
        shutil.copy2(self.workspace.manifests_dir / f"{manifest.id}.json", job_manifest)
        duration_limits = {
            ContentType.FLASH: 15.0, ContentType.EXPLAINER: 25.0, ContentType.DEEP_DIVE: 40.0,
        }
        video_checks = validate_wechat_mp4(
            probe_video(final), max_duration=duration_limits[manifest.content_type], require_audio=True,
        )
        checks = validate_manifest(manifest, self.workspace.root)
        result["stages"].append({"name": "mpt_master", "status": "ok", "video": str(final)})
        result["video"] = str(final)
        result["video_checks"] = video_checks
        result["checks"] = [check.to_dict() for check in checks]
        result["publishable"] = is_publishable(checks) and all(item["passed"] for item in video_checks)

    def _generation_cache_path(self, url: str, options: GenerateOptions) -> Path:
        payload = json.dumps({
            "url": url, "topic": options.topic.value if options.topic else "auto",
            "format": options.content_type.value if options.content_type else "auto",
            "duration": options.duration, "provider": options.provider, "model": options.model,
            "research": options.research, "linked_sources": options.linked_sources,
            "supplemental_context": options.supplemental_context,
            "price_event_metadata": options.price_event_metadata, "schema": 44,
        }, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / "generations" / f"{digest}.manifest.json"

    @staticmethod
    def _classify(url: str) -> str:
        host = (urlparse(url).hostname or "").casefold()
        if host in {"github.com", "www.github.com"}:
            return "github"
        if host in {"x.com", "twitter.com", "www.x.com", "www.twitter.com"}:
            return "x"
        if host in {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}:
            return "youtube"
        if urlparse(url).path.casefold().endswith(".pdf") or host.endswith(("arxiv.org", "openreview.net")):
            return "paper"
        return "web"

    def _generate_editorial(
        self, url: str, job: Path, options: GenerateOptions, result: dict[str, object],
    ) -> None:
        acquired = None if options.refresh else self._cached_acquisition(url)
        if acquired is None:
            acquired = URLAcquirer(self.workspace).acquire(url, job)
        candidate = acquired.ingest.candidate
        evidence = list(acquired.ingest.evidence)
        if options.supplemental_context:
            evidence = [self._price_focused_evidence(item) for item in evidence]
        if options.linked_sources:
            candidate.linked_sources = list(dict.fromkeys([
                *candidate.linked_sources, *options.linked_sources,
            ]))
            self.workspace.save_candidate(candidate)
        if options.supplemental_context:
            context_path = job / "discovery-context.md"
            context_path.write_text(options.supplemental_context.strip() + "\n", encoding="utf-8")
            asset, digest = self.workspace.archive_asset(context_path, "discovery-context")
            context_evidence = Evidence(
                id=f"{candidate.id}-discovery-context", candidate_id=candidate.id, url=url,
                quote=options.supplemental_context, source_kind="discovery:price_context",
                captured_asset=asset, sha256=digest,
                notes=(
                    "Machine-collected endpoint price comparison. Show provider, capture time, and workload "
                    "assumptions; verify claims against the linked primary pricing sources."
                ),
                metadata={
                    "price_event": dict(options.price_event_metadata or {}),
                },
            )
            self.workspace.save_evidence(context_evidence)
            evidence.append(context_evidence)
        result["stages"].append({
            "name": "ingest", "status": "ok", "candidate": candidate.id,
            "method": acquired.method, "artifact": str(acquired.artifact),
        })
        route = route_content(
            candidate, evidence, options.topic, options.content_type, options.duration,
        )
        result["routing"] = {
            "source_kind": self._classify(url), "topic_type": route.topic_type.value,
            "content_type": route.content_type.value, "target_duration": route.target_duration,
            "reason": route.reason,
        }
        writer, quote, selection = self._story_writer(options)
        copy_reviewer, reviewer_selection = self._copy_reviewer(writer, options)
        selection["copy_reviewer"] = reviewer_selection
        result["model_selection"] = selection
        x_images = [
            item for item in evidence
            if item.source_kind == "x:media_photo" and item.metadata.get("editorial_priority") == "high"
        ][:3]
        if x_images and os.environ.get("OPENROUTER_API_KEY"):
            try:
                catalog = OpenRouterCatalog(self.cache_dir / "openrouter")
                vision_quote = quote if quote and "image" in quote.input_modalities else catalog.select(
                    ModelRequirements("vision", ("text", "image")), options.refresh_prices,
                )
                vision_quotes = [vision_quote]
                if vision_quote.model_id != "google/gemini-3.7-flash":
                    vision_quotes.append(catalog.quote_for("google/gemini-3.7-flash"))
                analysis = None
                vision_errors: list[str] = []
                used_vision_quote = vision_quote
                for candidate_quote in vision_quotes:
                    try:
                        settings = LLMSettings.from_environment("openrouter", candidate_quote.model_id)
                        analysis = OpenRouterVisualAnalyst(settings, candidate_quote).analyze_x_images(url, x_images)
                        used_vision_quote = candidate_quote
                        break
                    except Exception as vision_error:
                        vision_errors.append(f"{candidate_quote.model_id}: {type(vision_error).__name__}: {vision_error}")
                if analysis is None:
                    raise RuntimeError("; ".join(vision_errors))
                analysis_path = job / "x-visual-analysis.json"
                analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                asset, digest = self.workspace.archive_asset(analysis_path, "twitter-visual-analysis")
                image_by_id = {item.id: item for item in x_images}
                added = 0
                for row in analysis.get("images", []):
                    if not isinstance(row, dict):
                        continue
                    image = image_by_id.get(str(row.get("evidence_id") or ""))
                    if image is None:
                        image = next((item for item in x_images if item.url == str(row.get("url") or "")), None)
                    if image is None:
                        continue
                    visual = Evidence(
                        id=f"{image.id}-visual-analysis", candidate_id=candidate.id, url=image.url,
                        quote=json.dumps(row, ensure_ascii=False), source_kind="x:visual_analysis",
                        captured_asset=asset, sha256=digest,
                        notes="Multimodal transcription of archived X media; cite with its parent image and do not infer beyond visible pixels.",
                        metadata={"parent_image_id": image.id, "parent_source_url": image.metadata.get("parent_source_url")},
                    )
                    self.workspace.save_evidence(visual)
                    evidence.append(visual)
                    added += 1
                result["stages"].append({
                    "name": "x_visual_analysis", "status": "ok" if added else "empty",
                    "model": used_vision_quote.model_id, "attempts": vision_errors,
                    "images": added, "artifact": str(analysis_path),
                    "provenance": analysis.get("provenance"),
                })
            except Exception as error:
                result["stages"].append({
                    "name": "x_visual_analysis", "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                })
        packet = StoryWriterPacket(
            candidate, evidence, route.topic_type, route.content_type, route.target_duration,
            editorial_direction=(
                "先发现可验证的冲突、反差或高风险变化。第一镜必须承担钩子并同时展示来源证据；"
                "结尾必须回答开头。X 原帖完整放在第一镜，不拆段。"
                + self._causal_uncertainty_direction(evidence)
                + self._source_identity_direction(evidence)
                + (
                    "这是价格异常线索：用每百万 token 实价、指定工作负载总价和可靠端点做钩子；"
                    "若与原厂比较，必须同时交代峰谷价、缓存条件、供应商和抓取时间。"
                    "不要把折扣写成无条件的模型能力推荐。若 intelligence 低于内部 55 分门槛，"
                    "结论必须明确它只值得低成本测试，禁止使用‘强劲、强大、值得立即切换/尝试’等推荐。"
                    "价格视频严格使用三幕：①页面真实折扣/endpoint 实价；②18K 输入+4K 输出的任务总价，"
                    "若有原厂比价则同时展示原厂峰谷价；③intelligence/coding 分数和是否过 55 分门槛的结论。"
                    "必须使用 supplemental price context；不要使用平均客户支付价、延迟、吞吐或其他页面杂项。"
                    "fixed_hook、fixed_title、fixed_conclusion 必须分别逐字使用 metadata.required_hook_zh、"
                    "metadata.required_headline_zh、metadata.editorial_verdict_zh，不得扩写。"
                    if options.supplemental_context else ""
                )
            ),
        )
        try:
            run = self._run_editorial_agent_with_fallback(
                packet, writer, copy_reviewer, options, job, selection,
            )
        except ContentAgentError as error:
            trace_path = next((
                path for path in (
                    job / "content-agent-error.json", job / "content-agent-primary-error.json",
                ) if path.is_file()
            ), job / "content-agent-error.json")
            if not trace_path.is_file():
                trace_path.write_text(
                    json.dumps({"error": str(error), "trace": error.trace}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            result["content_agent_error"] = str(trace_path)
            raise
        manifest = run.manifest
        if options.price_event_metadata:
            retained_ids = {item.id for item in manifest.evidence}
            manifest.evidence.extend(
                item for item in evidence
                if item.id not in retained_ids
                and item.source_kind in {
                    "web:primary_page", "web:page", "discovery:price_event", "discovery:price_context",
                }
            )
            self._canonicalize_price_event_copy(manifest, options.price_event_metadata)
        if options.supplemental_context:
            self._canonicalize_price_event_visuals(manifest, url)
            self._recompile_editorial_manifest(manifest, candidate)
        self._reconcile_x_display_name(candidate, manifest.evidence)
        context_actions = [item for item in run.trace if item.get("step") == "context_research"]
        result["stages"].append({
            "name": "context_research", "status": "ok" if context_actions else ("disabled" if not options.research else "skipped"),
            "sources": context_actions,
        })
        manifest_path = self.workspace.save_manifest(manifest)
        job_manifest = job / "manifest.json"
        shutil.copy2(manifest_path, job_manifest)
        result["stages"].append({
            "name": "content_agent", "status": "ok", "llm_calls": run.llm_calls,
            "attention_strategy": asdict(manifest.editorial_brief.attention_strategy) if manifest.editorial_brief else None,
            "opportunity": asdict(manifest.editorial_brief.opportunity) if manifest.editorial_brief and manifest.editorial_brief.opportunity else None,
            "director_brief": asdict(manifest.editorial_brief.director_brief) if manifest.editorial_brief and manifest.editorial_brief.director_brief else None,
        })
        result["manifest"] = str(job_manifest)
        checks = validate_manifest(manifest, self.workspace.root)
        if not options.render:
            result["checks"] = [check.to_dict() for check in checks]
            result["publishable"] = is_publishable(checks)
            return

        self._render_editorial(manifest, candidate.source_url, job, result)

    @staticmethod
    def _causal_uncertainty_direction(evidence: list[Evidence]) -> str:
        corpus = "\n".join(item.quote for item in evidence).casefold()
        explicit_uncertainty = any(marker in corpus for marker in (
            "don't know yet whether", "do not know yet whether", "not sure whether",
            "unclear whether", "不确定是否", "尚不确定", "说不清是不是", "原因不明",
        ))
        trigger_language = any(marker in corpus for marker in (
            "trigger", "caused", "suspend", "banned", "封禁", "封号", "触发",
        ))
        if not (explicit_uncertainty and trigger_language):
            return ""
        return (
            "证据明确说因果关系尚未确定。任何可见字段只要同时提到前一操作与后一结果，"
            "必须在同一句或同一屏明确写出‘当事人说不确定两者是否相关/是否由此触发尚无定论’；"
            "禁止用‘导致、触发、秒封、随即被封、照做就被封’偷渡确定因果。可以写时间顺序，"
            "但不能让观众离开这一语义单元后才看到范围限定。"
        )

    @staticmethod
    def _source_identity_direction(evidence: list[Evidence]) -> str:
        corpus = "\n".join(item.quote for item in evidence).casefold()
        if not any(marker in corpus for marker in ("employee", "员工", "staff member")):
            return ""
        return (
            "来源身份必须逐级准确：员工个人账号、个人评论或个人回复不等于公司官方账号或公司声明；"
            "除非证据明确标注为公司官方回应，否则禁止写‘官方回应、官方表态、第一条官方回应’，"
            "必须点名为‘某公司员工的个人回复/评论’。"
        )

    def _editorial_agent(
        self, writer: OpenAICompatibleStoryWriter, copy_reviewer: OpenAICompatibleStoryWriter,
        options: GenerateOptions, job: Path, max_llm_calls: int = 12,
    ) -> BoundedContentAgent:
        return BoundedContentAgent(
            writer, research_tool=LinkedSourceResearchTool(self.workspace),
            context_tool=DirectorContextToolbox(self.workspace, job) if options.research else None,
            copy_reviewer=copy_reviewer,
            # Plan + structural write/repair + independent copy review/repair/
            # verification must all fit. A repaired draft never bypasses the
            # critic merely because it consumed the original low-cost budget.
            budget=AgentBudget(max_llm_calls=max_llm_calls, max_research_sources=3, max_repairs=2, max_escalations=0),
        )

    def _run_editorial_agent_with_fallback(
        self, packet: StoryWriterPacket, writer: OpenAICompatibleStoryWriter,
        copy_reviewer: OpenAICompatibleStoryWriter, options: GenerateOptions,
        job: Path, selection: dict[str, object],
    ):
        fallback_model = "google/gemini-3.7-flash"
        # A cheap writer gets one complete semantic round (review, repair,
        # verify). If that still fails, open the job-scoped semantic circuit
        # and escalate instead of spending several more rounds on the same
        # model. The stronger writer keeps the bounded convergence budget.
        primary_call_budget = 14 if writer.settings.model == fallback_model else 6
        try:
            return self._editorial_agent(
                writer, copy_reviewer, options, job, max_llm_calls=primary_call_budget,
            ).run(packet)
        except ContentAgentError as primary_error:
            primary_trace_path = job / "content-agent-primary-error.json"
            primary_trace_path.write_text(
                json.dumps(
                    {"error": str(primary_error), "trace": primary_error.trace},
                    ensure_ascii=False, indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            can_fallback = (
                bool(os.environ.get("OPENROUTER_API_KEY"))
                and writer.settings.model != fallback_model
            )
            if not can_fallback:
                raise
            fallback_writer = OpenAICompatibleStoryWriter(
                LLMSettings.from_environment("openrouter", fallback_model)
            )
            fallback_reviewer, fallback_reviewer_selection = self._copy_reviewer(
                fallback_writer, options,
            )
            selection["fallback"] = {
                "provider": "openrouter", "model": fallback_model,
                "reason": "low-cost primary model exhausted bounded semantic-copy repairs",
                "primary_error": str(primary_trace_path),
                "copy_reviewer": fallback_reviewer_selection,
            }
            try:
                return self._editorial_agent(
                    fallback_writer, fallback_reviewer, options, job, max_llm_calls=14,
                ).run(packet)
            except ContentAgentError as fallback_error:
                trace_path = job / "content-agent-error.json"
                trace_path.write_text(
                    json.dumps(
                        {"error": str(fallback_error), "trace": fallback_error.trace},
                        ensure_ascii=False, indent=2,
                    ) + "\n",
                    encoding="utf-8",
                )
                raise

    def _cached_acquisition(self, url: str) -> AcquisitionResult | None:
        """Reuse immutable archived source evidence when only the edit schema changed."""
        candidate = self.workspace.candidate_for_source_url(url)
        if candidate is None:
            return None
        evidence = [
            item for item in self.workspace.evidence_for_candidate(candidate.id)
            if not item.metadata.get("context_role")
            and not any(marker in item.source_kind for marker in (
                "related", "context", "search_result", "agent_primary", "source_image",
            ))
        ]
        root = next((item for item in evidence if item.url.rstrip("/") == url.rstrip("/")), None)
        if root is None:
            return None
        self._reconcile_x_display_name(candidate, evidence)
        artifact = self.workspace.root / root.captured_asset
        if not artifact.is_file():
            return None
        return AcquisitionResult(
            IngestResult(candidate=candidate, evidence=evidence, linked_candidates=[]),
            artifact, "workspace-acquisition-cache",
        )

    def _reconcile_x_display_name(self, candidate, evidence: list[Evidence]) -> None:
        root = next((
            item for item in evidence
            if item.url.rstrip("/") == candidate.source_url.rstrip("/")
        ), None)
        if root is None:
            return
        root_handle = str(root.metadata.get("author_handle") or candidate.author or "").lstrip("@").casefold()
        display_name = next((
            str(item.metadata.get("author_name") or "").strip()
            for item in evidence
            if str(item.metadata.get("author_handle") or "").lstrip("@").casefold() == root_handle
            and str(item.metadata.get("author_name") or "").strip().lstrip("@").casefold() != root_handle
        ), "")
        if display_name and str(root.metadata.get("author_name") or "").strip() != display_name:
            root.metadata["author_name"] = display_name
            candidate.metadata["author_name"] = display_name
            self.workspace.save_evidence(root)
            self.workspace.save_candidate(candidate)

    def _render_editorial(
        self, manifest, source_url: str, job: Path, result: dict[str, object],
    ) -> None:
        self._apply_default_music_license(manifest)
        job_manifest = job / "manifest.json"
        browser = job / "evidence-browser.mp4"
        frames = job / "browser-frames"
        brief = manifest.editorial_brief
        segmented_families = {"quote_card", "timeline", "impact_card", "stat_card"}
        needs_segmented_track = bool(brief and any(
            shot.kind in {EvidenceShotKind.TWEET_CARD, EvidenceShotKind.IMAGE}
            or shot.visual_family in segmented_families
            for shot in brief.evidence_shots
        ))
        if needs_segmented_track:
            self._render_editorial_track(manifest, source_url, browser, frames, job)
        else:
            capture_request = WebScrollVideoAdapter.editorial_story_request(
                source_url, manifest, browser, frames,
            )
            WebScrollVideoAdapter(WebScrollVideoSettings.from_environment()).capture(capture_request)
        result["stages"].append({"name": "browser_capture", "status": "ok", "video": str(browser)})
        framed = job / "framed-visual.mp4"
        compose_information_frame(manifest, browser, framed)
        result["stages"].append({"name": "compose", "status": "ok", "video": str(framed)})
        mastered = MPTAssemblyAdapter(MPTSettings.from_environment()).assemble(manifest, framed, job.name)
        final = job / "final.mp4"
        shutil.copy2(mastered, final)
        manifest.video_path = str(final.relative_to(self.workspace.root))
        self.workspace.save_manifest(manifest)
        shutil.copy2(self.workspace.manifests_dir / f"{manifest.id}.json", job_manifest)
        video_checks = validate_wechat_mp4(
            probe_video(final), max_duration=15 if manifest.content_type == ContentType.FLASH else None,
            require_audio=True,
        )
        checks = validate_manifest(manifest, self.workspace.root)
        result["stages"].append({"name": "mpt_master", "status": "ok", "video": str(final)})
        result["video"] = str(final)
        result["video_checks"] = video_checks
        result["checks"] = [check.to_dict() for check in checks]
        result["publishable"] = is_publishable(checks) and all(item["passed"] for item in video_checks)

    @staticmethod
    def _apply_default_music_license(manifest) -> None:
        settings = MPTSettings.from_environment()
        selected = Path(settings.bgm_file).resolve()
        if not selected.is_file():
            return
        digest = hashlib.sha256()
        with selected.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != "858ccdf311934f64bc9d76642cbcd0f04d8d282db5c87279a58d63379f0698d0":
            return
        manifest.music_license_status = "royalty_free_verified"
        manifest.license_records = [{
            "track": "Better Times Are Coming — Mixkit 173",
            "source": "Mixkit / Envato",
            "license": "Mixkit Free License",
            "audio_asset": "assets/music/858ccdf31193/better-times-are-coming-mixkit-173.mp3",
            "license_archive": "assets/licenses/bc0e5363e51c/mixkit-music-free-license-20260827.md",
            "track_page_archive": "assets/licenses/e10df15cb6a4/mixkit-better-times-track-page-20260827.md",
        }]

    def _render_editorial_track(
        self, manifest, source_url: str, output: Path, frames: Path, job: Path,
    ) -> None:
        candidate = self.workspace.load_candidate(manifest.candidate_id)
        brief = manifest.editorial_brief
        if brief is None or len(brief.evidence_shots) != len(manifest.scenes):
            raise ValueError("segmented editorial rendering needs one evidence_shot per compiled scene")
        evidence_by_id = {item.id: item for item in manifest.evidence}
        derived_families = {"quote_card", "timeline", "impact_card", "stat_card"}
        segments: list[TrackSegment] = []
        sidecar_parts: list[dict[str, object]] = []
        # Evidence and tweet cards contain dense text. Direct crossfades make
        # both cards readable at once for several frames, so editorial tracks
        # use clean cuts; motion remains inside real browser captures.
        fade = 0.0
        for index, (scene, shot) in enumerate(zip(manifest.scenes, brief.evidence_shots), start=1):
            cited = next((evidence_by_id[item] for item in scene.evidence_ids if item in evidence_by_id), None)
            if cited is None:
                raise ValueError(f"scene {scene.id} has no archived evidence")
            family = (scene.visual_family or shot.visual_family or "").strip()
            if cited.source_kind == "web:reported_context" and family not in derived_families:
                family = "timeline" if shot.retention_job in {"background", "contrast", "turn"} else "quote_card"
            if shot.kind == EvidenceShotKind.IMAGE:
                image_evidence = next((
                    evidence_by_id[item] for item in scene.evidence_ids
                    if item in evidence_by_id
                    and evidence_by_id[item].source_kind in {"web:source_image", "x:media_photo"}
                ), cited)
                if not image_evidence.captured_asset:
                    raise ValueError(f"source image {image_evidence.id} has no archived asset")
                asset = self.workspace.root / image_evidence.captured_asset
                frame = render_source_image(scene, image_evidence, asset, job / f"source-image-{index}.png")
                part = tweet_card_video(frame, scene.duration, job / f"source-image-{index}.mp4")
                duration = scene.duration
                local_shots = [{
                    "id": scene.id, "action": "source_image", "start": 0, "end": duration,
                    "translation": "",
                }]
            elif family in derived_families or (
                family == "code" and cited.source_kind == "x:referenced_context_post"
            ):
                frame = render_editorial_card(scene, cited, job / f"editorial-card-{index}.png", family)
                part = tweet_card_video(frame, scene.duration, job / f"editorial-card-{index}.mp4")
                duration = scene.duration
                local_shots = [{
                    "id": scene.id, "action": family, "start": 0, "end": duration,
                    "translation": "",
                }]
            elif shot.kind == EvidenceShotKind.TWEET_CARD:
                if not cited.source_kind.startswith("x:"):
                    raise ValueError(f"tweet_card {scene.id} must cite archived X evidence")
                frame = render_tweet_card(candidate, cited, scene, job / f"tweet-card-{index}.png")
                part = tweet_card_video(frame, scene.duration, job / f"tweet-card-{index}.mp4")
                duration = scene.duration
                local_shots = [{
                    "id": scene.id, "action": "tweet_card", "start": 0, "end": duration,
                    "translation": "",
                }]
            else:
                capture_url = shot.source_url or cited.url or source_url
                cues = list(scene.recording_cues)
                if cues and cues[0].action == CueAction.OPEN:
                    capture_url = cues[0].value or cues[0].target or capture_url
                    cues = cues[1:]
                part = job / f"evidence-browser-{index}.mp4"
                request = WebCaptureRequest(
                    capture_url, cues, part, frames / f"scene-{index}", width=1384, height=1602,
                )
                WebScrollVideoAdapter(WebScrollVideoSettings.from_environment()).capture(request)
                metadata = json.loads(part.with_suffix(".capture.json").read_text(encoding="utf-8"))
                duration = float(metadata.get("duration") or scene.duration)
                local_shots = list(metadata.get("shots", []))
            segments.append(TrackSegment(part, duration))
            offset = sum(item.duration for item in segments[:-1]) - fade * max(0, len(segments) - 1)
            for local in local_shots:
                shifted = dict(local)
                shifted["start"] = round(float(local.get("start", 0)) + offset, 3)
                shifted["end"] = round(float(local.get("end", duration)) + offset, 3)
                sidecar_parts.append(shifted)
        build_crossfade_track(segments, output, fade_seconds=fade)
        sidecar = {
            "version": 1, "width": 1384, "height": 1602,
            "duration": round(sum(item.duration for item in segments) - fade * max(0, len(segments) - 1), 3),
            "shots": sidecar_parts,
        }
        output.with_suffix(".capture.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

    def _generate_github(
        self, url: str, job: Path, options: GenerateOptions, result: dict[str, object],
    ) -> None:
        owner, repo = self._github_identity(url)
        repo_url = f"https://github.com/{owner}/{repo}"
        repo_json = job / "repo.json"
        readme_path = job / "README.md"
        repo_payload = self._github_json(f"https://api.github.com/repos/{owner}/{repo}")
        repo_json.write_text(json.dumps(repo_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        readme_bytes = self._github_bytes(
            f"https://api.github.com/repos/{owner}/{repo}/readme", "application/vnd.github.raw+json",
        )
        readme_path.write_bytes(readme_bytes)
        ingest = GitHubIngestor().ingest(repo_json, readme_path, self.workspace)
        result["stages"].append({"name": "ingest", "status": "ok", "candidate": ingest.candidate.id})

        writer, quote, selection = self._story_writer(options)
        copy_reviewer, reviewer_selection = self._copy_reviewer(writer, options)
        selection["copy_reviewer"] = reviewer_selection
        result["model_selection"] = selection
        evidence = list(ingest.evidence)
        visual_context = ""
        default_branch = str(repo_payload.get("default_branch") or "main")
        raw_base = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/README.md"
        readme_text = readme_bytes.decode("utf-8", errors="replace")
        context_evidence, linked_sources, context_actions = enrich_github_context(
            ingest.candidate, readme_text, owner, repo, default_branch,
            self.workspace, job, self._github_bytes,
        )
        evidence.extend(context_evidence)
        ingest.candidate.linked_sources = list(dict.fromkeys([
            *ingest.candidate.linked_sources, *linked_sources,
        ]))
        self.workspace.save_candidate(ingest.candidate)
        result["stages"].append({
            "name": "background_context", "status": "ok" if context_evidence else "skipped",
            "documents": context_actions, "linked_primary_sources": linked_sources,
        })
        visuals = find_high_value_visuals(readme_text, raw_base)
        if visuals and os.environ.get("OPENROUTER_API_KEY"):
            catalog = OpenRouterCatalog(self.cache_dir / "openrouter")
            vision_quote = catalog.select(ModelRequirements("vision", ("text", "image")), options.refresh_prices)
            settings = LLMSettings.from_environment("openrouter", vision_quote.model_id)
            analysis = OpenRouterVisualAnalyst(settings, vision_quote).analyze(repo_url, readme_text, visuals)
            analysis_path = job / "visual-analysis.json"
            analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            asset, digest = self.workspace.archive_asset(analysis_path, "github-visual-analysis")
            visual_evidence = Evidence(
                id=f"{ingest.candidate.id}-visual-analysis", candidate_id=ingest.candidate.id,
                url=repo_url + "#readme", quote=json.dumps(analysis, ensure_ascii=False),
                source_kind="github:visual_analysis", captured_asset=asset, sha256=digest,
                notes="Multimodal interpretation only; never a browser-target or sole factual source.",
            )
            self.workspace.save_evidence(visual_evidence)
            evidence.append(visual_evidence)
            visual_context = "多模态模型已读 README 中的架构/Benchmark 图；仅用于解释，浏览器目标与事实必须回到原始 README。"
            result["stages"].append({
                "name": "visual_analysis", "status": "ok", "images": [asdict(item) for item in visuals],
                "model": vision_quote.to_dict(), "artifact": str(analysis_path),
                "provenance": analysis.get("provenance"),
            })
        else:
            reason = "no high-value README image" if not visuals else "OPENROUTER_API_KEY is not configured"
            result["stages"].append({"name": "visual_analysis", "status": "skipped", "reason": reason})

        packet = StoryWriterPacket(
            ingest.candidate, evidence, TopicType.GITHUB_PROJECT, ContentType.EXPLAINER,
            options.duration or 20.0, editorial_direction=visual_context,
        )
        def github_agent(
            active_writer: OpenAICompatibleStoryWriter,
            active_reviewer: OpenAICompatibleStoryWriter,
        ) -> BoundedContentAgent:
            return BoundedContentAgent(
                active_writer, research_tool=LinkedSourceResearchTool(self.workspace),
                copy_reviewer=active_reviewer,
                # Normal GitHub jobs still finish in two or three calls.  The
                # higher ceiling is only available when deterministic validation
                # or the independent copy critic explicitly rejects a draft; a
                # semantic round requires review + repair + verification, and up
                # to three bounded rounds must be possible without Codex stepping
                # in to finish a URL job manually.
                budget=AgentBudget(max_llm_calls=12, max_research_sources=2, max_repairs=3, max_escalations=0),
            )
        try:
            run = github_agent(writer, copy_reviewer).run(packet)
        except ContentAgentError as primary_error:
            primary_trace_path = job / "content-agent-primary-error.json"
            primary_trace_path.write_text(
                json.dumps({"error": str(primary_error), "trace": primary_error.trace}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            can_fallback = (
                options.provider in {"auto", "openrouter"}
                and bool(os.environ.get("OPENROUTER_API_KEY"))
                and writer.settings.model != "google/gemini-3.7-flash"
            )
            if not can_fallback:
                result["content_agent_error"] = str(primary_trace_path)
                raise
            fallback_model = "google/gemini-3.7-flash"
            fallback_writer = OpenAICompatibleStoryWriter(
                LLMSettings.from_environment("openrouter", fallback_model)
            )
            fallback_reviewer, fallback_reviewer_selection = self._copy_reviewer(
                fallback_writer, options,
            )
            selection["fallback"] = {
                "provider": "openrouter", "model": fallback_model,
                "reason": "low-cost primary model exhausted grounded-structure repairs",
                "primary_error": str(primary_trace_path),
                "copy_reviewer": fallback_reviewer_selection,
            }
            try:
                run = github_agent(fallback_writer, fallback_reviewer).run(packet)
            except ContentAgentError as fallback_error:
                trace_path = job / "content-agent-error.json"
                trace_path.write_text(
                    json.dumps({"error": str(fallback_error), "trace": fallback_error.trace}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                result["content_agent_error"] = str(trace_path)
                raise
        manifest = run.manifest
        manifest_path = self.workspace.save_manifest(manifest)
        job_manifest = job / "manifest.json"
        shutil.copy2(manifest_path, job_manifest)
        result["stages"].append({"name": "content_agent", "status": "ok", "llm_calls": run.llm_calls})
        result["manifest"] = str(job_manifest)
        if not options.render:
            checks = validate_manifest(manifest, self.workspace.root)
            result["checks"] = [check.to_dict() for check in checks]
            result["publishable"] = is_publishable(checks)
            return

        self._render_github_manifest(manifest, repo_url, job, result)

    def _story_writer(self, options: GenerateOptions) -> tuple[OpenAICompatibleStoryWriter, ModelQuote | None, dict[str, object]]:
        provider = options.provider
        if provider == "auto" and os.environ.get("OPENROUTER_API_KEY"):
            try:
                quote = OpenRouterCatalog(self.cache_dir / "openrouter").select(
                    ModelRequirements("story", ("text",)), options.refresh_prices,
                )
                settings = LLMSettings.from_environment("openrouter", options.model or quote.model_id)
                return OpenAICompatibleStoryWriter(settings), quote, {
                    "provider": "openrouter", "quote": quote.to_dict(), "daily_catalog": True,
                }
            except Exception as error:
                fallback_reason = f"OpenRouter selection failed: {type(error).__name__}: {error}"
        else:
            fallback_reason = "OPENROUTER_API_KEY is not configured" if provider == "auto" else "explicit provider"
        selected = "deepseek" if provider == "auto" else provider
        if selected == "openrouter":
            if options.model:
                settings = LLMSettings.from_environment("openrouter", options.model)
                return OpenAICompatibleStoryWriter(settings), None, {"provider": "openrouter", "model": options.model}
            quote = OpenRouterCatalog(self.cache_dir / "openrouter").select(
                ModelRequirements("story", ("text",)), options.refresh_prices,
            )
            settings = LLMSettings.from_environment("openrouter", quote.model_id)
            return OpenAICompatibleStoryWriter(settings), quote, {"provider": "openrouter", "quote": quote.to_dict()}
        settings = LLMSettings.from_environment(selected, options.model)
        return OpenAICompatibleStoryWriter(settings), None, {
            "provider": selected, "model": settings.model, "fallback_reason": fallback_reason,
        }

    def _copy_reviewer(
        self, writer: OpenAICompatibleStoryWriter, options: GenerateOptions,
    ) -> tuple[OpenAICompatibleStoryWriter, dict[str, object]]:
        """Choose an independent low-cost critic; never let the writer grade itself when avoidable."""
        if os.environ.get("OPENROUTER_API_KEY"):
            try:
                quote = OpenRouterCatalog(self.cache_dir / "openrouter").select(
                    ModelRequirements(
                        "review", ("text",), excluded_models=(writer.settings.model,),
                    ),
                    options.refresh_prices,
                )
                settings = LLMSettings.from_environment("openrouter", quote.model_id)
                return OpenAICompatibleStoryWriter(settings), {
                    "provider": "openrouter", "quote": quote.to_dict(),
                    "reason": "cheapest capability-qualified independent critic",
                }
            except Exception:
                pass
        if os.environ.get("DEEPSEEK_API_KEY") and writer.settings.provider != "deepseek":
            settings = LLMSettings.from_environment("deepseek", None)
            return OpenAICompatibleStoryWriter(settings), {
                "provider": "deepseek", "model": settings.model,
                "reason": "economical independent critic fallback",
            }
        return writer, {
            "provider": writer.settings.provider, "model": writer.settings.model,
            "reason": "no independent reviewer configured",
        }

    def _translation_writer(self, options: GenerateOptions) -> tuple[OpenAICompatibleStoryWriter, dict[str, object]]:
        if options.provider == "auto" and os.environ.get("OPENROUTER_API_KEY"):
            try:
                quote = OpenRouterCatalog(self.cache_dir / "openrouter").select(
                    ModelRequirements("translation", ("text",)), options.refresh_prices,
                )
                settings = LLMSettings.from_environment("openrouter", options.model or quote.model_id)
                return OpenAICompatibleStoryWriter(settings), {
                    "provider": "openrouter", "quote": quote.to_dict(),
                    "purpose": "translation", "daily_catalog": True,
                }
            except Exception as error:
                fallback_reason = f"OpenRouter translation selection failed: {type(error).__name__}: {error}"
        else:
            fallback_reason = "OPENROUTER_API_KEY is not configured" if options.provider == "auto" else "explicit provider"
        selected = "deepseek" if options.provider == "auto" else options.provider
        settings = LLMSettings.from_environment(selected, options.model)
        return OpenAICompatibleStoryWriter(settings), {
            "provider": selected, "model": settings.model, "purpose": "translation",
            "fallback_reason": fallback_reason,
        }

    @staticmethod
    def _github_identity(url: str) -> tuple[str, str]:
        parts = [part for part in urlparse(url).path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("GitHub URL must identify owner/repository")
        return parts[0], parts[1].removesuffix(".git")

    @staticmethod
    def _headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
        headers = {"Accept": accept, "User-Agent": "video-factory/0.1", "X-GitHub-Api-Version": "2022-11-28"}
        if token := os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @classmethod
    def _github_json(cls, url: str) -> dict[str, object]:
        with urlopen(Request(url, headers=cls._headers()), timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    @classmethod
    def _github_bytes(cls, url: str, accept: str) -> bytes:
        with urlopen(Request(url, headers=cls._headers(accept)), timeout=30) as response:
            return response.read()

    @staticmethod
    def _write_result(job: Path, result: dict[str, object]) -> None:
        (job / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
