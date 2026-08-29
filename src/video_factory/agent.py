from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
from io import BytesIO
from dataclasses import asdict, dataclass, replace
from html.parser import HTMLParser
from typing import Callable, Protocol
from urllib.parse import urljoin, urlparse
from urllib.error import URLError
from urllib.request import Request, urlopen

from .director import StoryboardDirector, StoryboardRequest
from .github_editor import compose_github_hook, validate_github_brief
from .llm import EditorialPlan, StoryDraftError
from .models import ContextEvent, ContextGraph, EditorialOpportunity, Evidence, RenderManifest, SelectionReason
from .quality import validate_manifest
from .safety import review_evidence
from .storage import Workspace
from .writer import StoryWriterPacket


class StoryModel(Protocol):
    """The small surface required by the bounded agent loop."""

    def plan(self, packet: StoryWriterPacket) -> tuple[EditorialPlan, dict[str, object]]: ...

    def generate(
        self, packet: StoryWriterPacket,
    ) -> tuple[StoryboardRequest, dict[str, object], dict[str, object]]: ...

    def repair(
        self, packet: StoryWriterPacket, invalid_draft: dict[str, object], validation_error: str,
    ) -> tuple[StoryboardRequest, dict[str, object], dict[str, object]]: ...

    def review_visible_copy(
        self, packet: StoryWriterPacket, draft: dict[str, object],
    ) -> tuple[list[dict[str, object]], dict[str, object]]: ...


@dataclass(frozen=True, slots=True)
class AgentBudget:
    """Hard limits keep the content agent cheap and reproducible."""

    max_llm_calls: int = 12
    max_research_sources: int = 3
    max_repairs: int = 1
    max_escalations: int = 1

    def __post_init__(self) -> None:
        if self.max_llm_calls < 2:
            raise ValueError("content agent needs at least two LLM calls: plan and write")
        if self.max_research_sources < 0 or self.max_repairs < 0 or self.max_escalations < 0:
            raise ValueError("agent budgets cannot be negative")


@dataclass(frozen=True, slots=True)
class ResearchOutcome:
    evidence: list[Evidence]
    actions: list[dict[str, object]]


class EvidenceResearchTool(Protocol):
    name: str

    def run(
        self, packet: StoryWriterPacket, requested_urls: list[str], limit: int,
    ) -> ResearchOutcome: ...


class ContextPlanningTool(Protocol):
    name: str

    def run(self, packet: StoryWriterPacket, plan: EditorialPlan, limit: int) -> ResearchOutcome: ...


class ArchivedEvidenceTool:
    """Default zero-network tool: report which requested sources are archived."""

    name = "inspect_archived_evidence"

    def run(self, packet: StoryWriterPacket, requested_urls: list[str], limit: int) -> ResearchOutcome:
        evidence = list(packet.evidence)
        actions: list[dict[str, object]] = []
        for url in requested_urls[:limit]:
            present = any(_same_document(item.url, url) for item in evidence)
            actions.append({"tool": self.name, "url": url, "status": "available" if present else "not_archived"})
        return ResearchOutcome(evidence, actions)


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "noscript"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


class _SourceImageParser(HTMLParser):
    """Collect source-owned editorial images without treating icons as art."""

    def __init__(self) -> None:
        super().__init__()
        self.images: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.casefold(): value or "" for name, value in attrs}
        if tag.casefold() == "img":
            source = values.get("src") or values.get("data-src") or values.get("data-lazy-src")
            if not source and values.get("srcset"):
                candidates = [item.strip().split()[0] for item in values["srcset"].split(",") if item.strip()]
                source = candidates[-1] if candidates else ""
            if source:
                self.images.append((source, values.get("alt", "").strip()))
        elif tag.casefold() == "meta" and values.get("property", "").casefold() in {
            "og:image", "twitter:image",
        }:
            source = values.get("content", "").strip()
            if source:
                self.images.append((source, values.get("property", "")))


class LinkedSourceResearchTool:
    """Open only model-requested links already attached to the candidate.

    This is intentionally not an unconstrained crawler.  The planner cannot
    invent destinations, the tool has a source budget, and every fetched body
    is archived before it can become evidence.
    """

    name = "open_linked_primary_source"

    def __init__(
        self, workspace: Workspace, fetcher: Callable[[str], tuple[bytes, str]] | None = None,
        media_fetcher: Callable[[str], tuple[bytes, str] | tuple[bytes, str, str]] | None = None,
        timeout_seconds: int = 30, max_bytes: int = 1_500_000,
    ) -> None:
        self.workspace = workspace
        self.fetcher = fetcher or self._fetch
        self.media_fetcher = media_fetcher or self._fetch
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes

    def run(self, packet: StoryWriterPacket, requested_urls: list[str], limit: int) -> ResearchOutcome:
        evidence = list(packet.evidence)
        actions: list[dict[str, object]] = []
        allowed = set(packet.candidate.linked_sources)
        for url in requested_urls[:limit]:
            if any(_same_document(item.url, url) for item in evidence):
                actions.append({"tool": self.name, "url": url, "status": "already_archived"})
                continue
            if url not in allowed:
                actions.append({"tool": self.name, "url": url, "status": "rejected_not_linked"})
                continue
            try:
                fetched = self.fetcher(url)
                body, content_type = fetched[:2]
                resolved_url = str(fetched[2]) if len(fetched) > 2 else url
                if len(body) > self.max_bytes:
                    raise ValueError(f"source exceeds {self.max_bytes} bytes")
                text = _body_to_text(body, content_type)
                if not text.strip():
                    raise ValueError("source contains no readable text")
                digest = hashlib.sha256(body).hexdigest()
                download_dir = self.workspace.root / "agent-sources"
                download_dir.mkdir(parents=True, exist_ok=True)
                raw = download_dir / f"{digest[:12]}.source"
                raw.write_bytes(body)
                archived_path, archived_hash = self.workspace.archive_asset(raw, "agent-web-sources")
                item = Evidence(
                    id=f"agent-web-{digest[:16]}", candidate_id=packet.candidate.id,
                    url=url, quote=text, source_kind="web:agent_primary_source",
                    captured_asset=archived_path, sha256=archived_hash,
                    notes="Opened by the bounded content agent from a candidate-linked URL.",
                )
                self.workspace.save_evidence(item)
                evidence.append(item)
                image_evidence, image_actions = self._archive_source_images(
                    packet, url, resolved_url, body, content_type,
                )
                evidence.extend(image_evidence)
                actions.append({
                    "tool": self.name, "url": url, "status": "archived", "evidence_id": item.id,
                    "source_images": [image.id for image in image_evidence],
                })
                actions.extend(image_actions)
            except Exception as error:  # Keep a failed source from aborting an otherwise evidence-bound story.
                actions.append({"tool": self.name, "url": url, "status": "failed", "error": str(error)})
        return ResearchOutcome(evidence, actions)

    def _archive_source_images(
        self, packet: StoryWriterPacket, source_url: str, resolved_url: str,
        body: bytes, content_type: str,
    ) -> tuple[list[Evidence], list[dict[str, object]]]:
        decoded = body.decode("utf-8", errors="replace")
        discovered: list[tuple[str, str]] = []
        if "html" in content_type.casefold() or decoded.lstrip().startswith("<"):
            parser = _SourceImageParser()
            parser.feed(decoded)
            discovered.extend(parser.images)
        discovered.extend(
            (match.group(2), match.group(1).strip())
            for match in re.finditer(r"!\[([^\]]*)\]\((https?://[^)\s]+)\)", decoded)
        )

        def priority(item: tuple[str, str]) -> tuple[int, int]:
            raw_url, alt = item
            text = (raw_url + " " + alt).casefold()
            high_value = any(marker in text for marker in (
                "founding team", "team", "founders", "architecture", "benchmark", "chart",
                "diagram", "screenshot", "demo", "result", "comparison",
            ))
            noisy = any(marker in text for marker in ("favicon", "logo", "icon", "badge", "avatar", "infinity"))
            return (2 if high_value else 1 if alt else 0, -1 if noisy else 0)

        unique: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw_url, alt in sorted(discovered, key=priority, reverse=True):
            image_url = urljoin(resolved_url, html.unescape(raw_url.strip()))
            if image_url in seen or urlparse(image_url).scheme not in {"http", "https"}:
                continue
            seen.add(image_url)
            if priority((image_url, alt))[1] < 0:
                continue
            unique.append((image_url, alt))
            if len(unique) == 4:
                break

        archived: list[Evidence] = []
        actions: list[dict[str, object]] = []
        for image_url, alt in unique:
            try:
                fetched = self.media_fetcher(image_url)
                image_body, image_type = fetched[:2]
                final_image_url = str(fetched[2]) if len(fetched) > 2 else image_url
                if not image_type.casefold().startswith("image/") or len(image_body) > 8_000_000:
                    raise ValueError("source image has an unsupported type or exceeds 8 MB")
                from PIL import Image

                with Image.open(BytesIO(image_body)) as opened:
                    width, height = opened.size
                    image_format = (opened.format or "png").casefold()
                if min(width, height) < 240 or width * height < 200_000:
                    raise ValueError(f"source image is too small for video: {width}x{height}")
                digest = hashlib.sha256(image_body).hexdigest()
                download_dir = self.workspace.root / "agent-sources"
                download_dir.mkdir(parents=True, exist_ok=True)
                raw = download_dir / f"{digest[:12]}.{image_format}"
                raw.write_bytes(image_body)
                archived_path, archived_hash = self.workspace.archive_asset(raw, "agent-source-images")
                role_text = (alt + " " + final_image_url).casefold()
                visual_role = next((
                    role for role, markers in (
                        ("team", ("team", "founder")),
                        ("benchmark", ("benchmark", "chart", "result", "comparison")),
                        ("architecture", ("architecture", "diagram")),
                        ("product", ("screenshot", "demo", "product")),
                    ) if any(marker in role_text for marker in markers)
                ), "source_image")
                image = Evidence(
                    id=f"agent-image-{digest[:16]}", candidate_id=packet.candidate.id,
                    url=final_image_url, quote=f"Official source image: {alt or visual_role}",
                    source_kind="web:source_image", captured_asset=archived_path, sha256=archived_hash,
                    notes="Image discovered on and downloaded from an opened candidate-linked primary source.",
                    metadata={
                        "parent_source_url": source_url, "resolved_parent_url": resolved_url,
                        "alt": alt, "width": width, "height": height, "visual_role": visual_role,
                        "editorial_priority": "high" if visual_role != "source_image" else "normal",
                    },
                )
                self.workspace.save_evidence(image)
                archived.append(image)
                actions.append({
                    "tool": self.name, "url": final_image_url, "status": "archived_source_image",
                    "evidence_id": image.id, "visual_role": visual_role,
                })
            except Exception as error:
                actions.append({
                    "tool": self.name, "url": image_url, "status": "source_image_skipped", "error": str(error),
                })
        return archived, actions

    def _fetch(self, url: str) -> tuple[bytes, str, str]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("only absolute HTTP(S) sources are supported")
        if parsed.hostname.casefold() in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("local network sources are not allowed")
        try:
            if ipaddress.ip_address(parsed.hostname).is_private:
                raise ValueError("private network sources are not allowed")
        except ValueError as error:
            if "private network" in str(error):
                raise
        request = Request(url, headers={"User-Agent": "video-factory-content-agent/1.0"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read(self.max_bytes + 1)
                return body, response.headers.get_content_type(), response.geturl()
        except (URLError, OSError):
            # Some official help centers reject non-browser TLS clients. Jina's
            # read-only text gateway is a bounded fallback; evidence still
            # retains the original URL and never treats the gateway as source.
            gateway = "https://r.jina.ai/http://" + parsed.netloc + parsed.path
            if parsed.query:
                gateway += "?" + parsed.query
            fallback = Request(gateway, headers={"User-Agent": "video-factory-content-agent/1.0"})
            with urlopen(fallback, timeout=self.timeout_seconds) as response:
                return response.read(self.max_bytes + 1), response.headers.get_content_type(), url


@dataclass(slots=True)
class AgentRunResult:
    manifest: RenderManifest
    plan: EditorialPlan
    trace: list[dict[str, object]]
    llm_calls: int
    used_escalation: bool


class ContentAgentError(RuntimeError):
    def __init__(self, message: str, trace: list[dict[str, object]]):
        super().__init__(message)
        self.trace = trace


class BoundedContentAgent:
    """A small, explicit loop: plan → research tool → write → repair/escalate.

    The model owns editorial choices.  It never controls browser cues,
    rendering, publication, or arbitrary network destinations.
    """

    def __init__(
        self, primary: StoryModel, research_tool: EvidenceResearchTool | None = None,
        context_tool: ContextPlanningTool | None = None,
        escalation: StoryModel | None = None, copy_reviewer: StoryModel | None = None,
        budget: AgentBudget | None = None,
        director: StoryboardDirector | None = None,
    ) -> None:
        self.primary = primary
        self.research_tool = research_tool or ArchivedEvidenceTool()
        self.context_tool = context_tool
        self.escalation = escalation
        self.copy_reviewer = copy_reviewer or primary
        self.budget = budget or AgentBudget()
        self.director = director or StoryboardDirector()

    def run(self, packet: StoryWriterPacket) -> AgentRunResult:
        trace: list[dict[str, object]] = []
        calls = 0
        try:
            plan, provenance = self.primary.plan(packet)
            calls += 1
            trace.append({"step": "plan", "status": "ok", "provenance": provenance, "result": asdict(plan)})
        except StoryDraftError as error:
            calls += 1
            plan = EditorialPlan(
                angle="从现有证据选择最具体、最可验证的价值与边界",
                audience_value="帮助开发者判断是否值得试用",
                selected_evidence_ids=[item.id for item in packet.evidence],
                requested_urls=[], unresolved_questions=[f"规划 JSON 无效：{error}"], ready_to_write=True,
            )
            trace.append({"step": "plan", "status": "fallback", "error": str(error)})

        context_outcome = ResearchOutcome(list(packet.evidence), [])
        if self.context_tool and self.budget.max_research_sources:
            context_outcome = self.context_tool.run(packet, plan, self.budget.max_research_sources)
            trace.extend({"step": "context_research", **action} for action in context_outcome.actions)
        context_packet = replace(packet, evidence=context_outcome.evidence)
        outcome = self.research_tool.run(context_packet, plan.requested_urls, self.budget.max_research_sources)
        trace.extend({"step": "research", **action} for action in outcome.actions)
        selected = set(plan.selected_evidence_ids)
        selected.update(item.id for item in outcome.evidence if item.id not in {old.id for old in packet.evidence})
        # High-value source visuals are execution material, not optional prose
        # context. Keep them available even when a cheap planning model only
        # selected the surrounding text evidence.
        selected.update(
            item.id for item in outcome.evidence
            if item.source_kind in {"web:source_image", "x:media_photo"}
            and item.metadata.get("editorial_priority") == "high"
        )
        # Deterministic discovery calculations (for example endpoint-price
        # comparisons) are required story evidence. A planner may rank the
        # prettier source page first, but it cannot discard the calculation
        # that justified automatic adoption in the first place.
        selected.update(
            item.id for item in outcome.evidence if item.source_kind.startswith("discovery:")
        )
        writing_evidence = [item for item in outcome.evidence if item.id in selected]
        if not writing_evidence:
            writing_evidence = list(outcome.evidence)
        context_graph = _context_graph_from_evidence(plan, writing_evidence)
        discarded_context_evidence = {
            evidence_id
            for event in context_graph.events
            if event.id in set(context_graph.discarded_context_ids)
            for evidence_id in event.evidence_ids
        }
        # Research is deliberately broader than the final story. Never put a
        # context item rejected by the relationship gate back in front of the
        # writer, where a cheap model can turn two separately true events into
        # one unsupported narrative.
        writing_evidence = [
            item for item in writing_evidence if item.id not in discarded_context_evidence
        ]
        safety_review = review_evidence(writing_evidence)
        direction = "；".join(filter(None, [
            packet.editorial_direction,
            f"核心角度：{plan.angle}", f"观众价值：{plan.audience_value}",
            "尚未证实，必须明确写成未知：" + "；".join(plan.unresolved_questions) if plan.unresolved_questions else "",
            "敏感内容边界：" + safety_review.allowed_angle if safety_review.requires_human_review else "",
            "禁止角度：" + (safety_review.prohibited_angle or "") if safety_review.requires_human_review else "",
        ]))
        opportunity = _opportunity_from_plan(plan, writing_evidence)
        writing_packet = replace(
            packet, evidence=writing_evidence, editorial_direction=direction,
            opportunity=opportunity, context_graph=context_graph,
        )

        manifest, raw_draft, calls, failure = self._write_with_one_repair(
            self.primary, writing_packet, calls, trace, tier="primary",
        )
        review_method = getattr(self.copy_reviewer, "review_visible_copy", None)
        if (
            manifest is not None and raw_draft is not None and callable(review_method)
            # A rejecting critic consumes three calls as one atomic quality
            # round: review, writer repair, and independent verification.
            # Never start that round when the budget can only pay for the
            # review and repair but not the verification.
            and calls <= self.budget.max_llm_calls - 3
        ):
            try:
                calls += 1
                issues, review_provenance = review_method(writing_packet, raw_draft)
                trace.append({
                    "step": "copy_review", "status": "approved" if not issues else "repair_required",
                    "issues": issues, "provenance": review_provenance,
                })
                if issues:
                    calls += 1
                    review_error = "semantic copy critic issues: " + json.dumps(issues, ensure_ascii=False)
                    request, repair_provenance, repaired = self.primary.repair(
                        writing_packet, raw_draft, review_error,
                    )
                    try:
                        preflight = _request_preflight_errors(request)
                        if preflight:
                            raise ValueError("; ".join(preflight))
                        manifest = self.director.direct(request)
                        blocking = _blocking_quality_errors(manifest)
                        if blocking:
                            raise ValueError("; ".join(blocking))
                    except ValueError as repair_validation_error:
                        # Semantic repairs can accidentally violate a render
                        # bound (for example translation length). Give the
                        # writer one content-neutral structural cleanup, while
                        # reserving the final call for independent verification.
                        if calls >= self.budget.max_llm_calls - 1:
                            raise
                        calls += 1
                        request, cleanup_provenance, repaired = self.primary.repair(
                            writing_packet, repaired, str(repair_validation_error),
                        )
                        preflight = _request_preflight_errors(request)
                        if preflight:
                            raise ValueError("; ".join(preflight))
                        manifest = self.director.direct(request)
                        blocking = _blocking_quality_errors(manifest)
                        if blocking:
                            raise ValueError("; ".join(blocking))
                        trace.append({
                            "step": "copy_review_repair_cleanup", "status": "ok",
                            "provenance": cleanup_provenance,
                            "reason": str(repair_validation_error),
                        })
                    raw_draft = repaired
                    trace.append({
                        "step": "copy_review_repair", "status": "ok",
                        "provenance": repair_provenance,
                    })
                    if calls >= self.budget.max_llm_calls:
                        raise ValueError("copy-review repair could not be independently verified within the LLM call budget")
                    calls += 1
                    remaining_issues, verify_provenance = review_method(writing_packet, raw_draft)
                    trace.append({
                        "step": "copy_review_verify",
                        "status": "approved" if not remaining_issues else "failed",
                        "issues": remaining_issues,
                        "provenance": verify_provenance,
                    })
                    if remaining_issues:
                        if calls >= self.budget.max_llm_calls - 1:
                            raise ValueError(
                                "semantic copy critic still rejects repaired draft: "
                                + json.dumps(remaining_issues, ensure_ascii=False)
                            )
                        calls += 1
                        second_review_error = (
                            "semantic copy critic issues after first repair: "
                            + json.dumps(remaining_issues, ensure_ascii=False)
                        )
                        request, second_repair_provenance, second_repaired = self.primary.repair(
                            writing_packet, raw_draft, second_review_error,
                        )
                        try:
                            preflight = _request_preflight_errors(request)
                            if preflight:
                                raise ValueError("; ".join(preflight))
                            manifest = self.director.direct(request)
                            blocking = _blocking_quality_errors(manifest)
                            if blocking:
                                raise ValueError("; ".join(blocking))
                        except ValueError as second_validation_error:
                            if calls >= self.budget.max_llm_calls - 1:
                                raise
                            calls += 1
                            request, cleanup_provenance, second_repaired = self.primary.repair(
                                writing_packet, second_repaired, str(second_validation_error),
                            )
                            preflight = _request_preflight_errors(request)
                            if preflight:
                                raise ValueError("; ".join(preflight))
                            manifest = self.director.direct(request)
                            blocking = _blocking_quality_errors(manifest)
                            if blocking:
                                raise ValueError("; ".join(blocking))
                            trace.append({
                                "step": "copy_review_second_repair_cleanup", "status": "ok",
                                "provenance": cleanup_provenance,
                                "reason": str(second_validation_error),
                            })
                        raw_draft = second_repaired
                        trace.append({
                            "step": "copy_review_second_repair", "status": "ok",
                            "provenance": second_repair_provenance,
                        })
                        if calls >= self.budget.max_llm_calls:
                            raise ValueError("second copy-review repair could not be independently verified")
                        calls += 1
                        final_issues, final_provenance = review_method(writing_packet, raw_draft)
                        trace.append({
                            "step": "copy_review_final_verify",
                            "status": "approved" if not final_issues else "failed",
                            "issues": final_issues,
                            "provenance": final_provenance,
                        })
                        if final_issues:
                            if calls >= self.budget.max_llm_calls - 1:
                                raise ValueError(
                                    "semantic copy critic still rejects second repaired draft: "
                                    + json.dumps(final_issues, ensure_ascii=False)
                                )
                            calls += 1
                            third_review_error = (
                                "semantic copy critic issues after second repair: "
                                + json.dumps(final_issues, ensure_ascii=False)
                            )
                            request, third_repair_provenance, third_repaired = self.primary.repair(
                                writing_packet, raw_draft, third_review_error,
                            )
                            preflight = _request_preflight_errors(request)
                            if preflight:
                                raise ValueError("; ".join(preflight))
                            manifest = self.director.direct(request)
                            blocking = _blocking_quality_errors(manifest)
                            if blocking:
                                raise ValueError("; ".join(blocking))
                            raw_draft = third_repaired
                            trace.append({
                                "step": "copy_review_third_repair", "status": "ok",
                                "provenance": third_repair_provenance,
                            })
                            if calls >= self.budget.max_llm_calls:
                                raise ValueError("third copy-review repair could not be independently verified")
                            calls += 1
                            last_issues, last_provenance = review_method(writing_packet, raw_draft)
                            trace.append({
                                "step": "copy_review_last_verify",
                                "status": "approved" if not last_issues else "failed",
                                "issues": last_issues,
                                "provenance": last_provenance,
                            })
                            if last_issues:
                                raise ValueError(
                                    "semantic copy critic still rejects third repaired draft: "
                                    + json.dumps(last_issues, ensure_ascii=False)
                                )
            except (StoryDraftError, ValueError, RuntimeError) as error:
                manifest = None
                failure = f"copy review failed: {error}"
                trace.append({"step": "copy_review", "status": "failed", "error": str(error)})
        used_escalation = False
        if manifest is None and self.escalation and self.budget.max_escalations and calls < self.budget.max_llm_calls:
            used_escalation = True
            trace.append({"step": "escalate", "status": "started", "reason": failure})
            manifest, raw_draft, calls, failure = self._write_once(
                self.escalation, writing_packet, calls, trace, tier="escalation",
            )
        if manifest is None:
            raise ContentAgentError(failure or "content agent could not create a valid manifest", trace)

        manifest.quality_checks.append({
            "name": "content_agent", "passed": True,
            "detail": {
                "mode": "contextual_director_loop", "llm_calls": calls, "max_llm_calls": self.budget.max_llm_calls,
                "research_tool": self.research_tool.name, "used_escalation": used_escalation,
                "context_tool": self.context_tool.name if self.context_tool else None,
                "plan": asdict(plan), "usage": _usage_totals(trace), "trace": trace,
            },
        })
        visible_evidence_ids = set(
            manifest.editorial_brief.attention_strategy.hook_evidence_ids
            if manifest.editorial_brief else []
        )
        visible_evidence_ids.update(item for scene in manifest.scenes for item in scene.evidence_ids)
        director_evidence_ids: set[str] = set()
        if manifest.editorial_brief and manifest.editorial_brief.director_brief:
            director_evidence_ids.update(
                item for beat in manifest.editorial_brief.director_brief.story_arc for item in beat.evidence_ids
            )
        director_only_evidence_ids = director_evidence_ids - visible_evidence_ids
        graph = manifest.editorial_brief.context_graph if manifest.editorial_brief else None
        discarded_context = set(graph.discarded_context_ids if graph else [])
        context_by_evidence = {
            evidence_id: event.id
            for event in (graph.events if graph else [])
            for evidence_id in event.evidence_ids
        }
        omitted = []
        for item in writing_evidence:
            if item.id in visible_evidence_ids:
                continue
            omitted.append({
                "evidence_id": item.id, "source_kind": item.source_kind,
                "reason": (
                    "director_only_not_visible" if item.id in director_only_evidence_ids
                    else "discarded_context" if context_by_evidence.get(item.id) in discarded_context
                    else "not_selected_within_duration_budget"
                ),
            })
        manifest.quality_checks.append({
            "name": "editorial_evidence_coverage", "passed": True,
            "detail": {
                "available_evidence_ids": [item.id for item in writing_evidence],
                "used_evidence_ids": sorted(visible_evidence_ids),
                "director_only_evidence_ids": sorted(director_only_evidence_ids),
                "researched_but_omitted": omitted,
            },
        })
        manifest.quality_checks.append({
            "name": "editorial_safety_review", "passed": not safety_review.requires_human_review,
            "detail": {
                "requires_human_review": safety_review.requires_human_review,
                "reasons": safety_review.reasons,
                "allowed_angle": safety_review.allowed_angle,
                "prohibited_angle": safety_review.prohibited_angle,
            },
        })
        return AgentRunResult(manifest, plan, trace, calls, used_escalation)

    def _write_with_one_repair(
        self, model: StoryModel, packet: StoryWriterPacket, calls: int,
        trace: list[dict[str, object]], tier: str,
    ) -> tuple[RenderManifest | None, dict[str, object] | None, int, str | None]:
        manifest, draft, calls, failure = self._write_once(model, packet, calls, trace, tier)
        repairs = 0
        while manifest is None and calls < self.budget.max_llm_calls and draft is not None:
            visible_copy_cleanup = _visible_copy_only_failure(failure or "")
            if repairs >= self.budget.max_repairs and not (
                repairs == self.budget.max_repairs and visible_copy_cleanup
            ):
                break
            try:
                calls += 1
                repairs += 1
                request, provenance, repaired = model.repair(packet, draft, failure or "invalid storyboard")
                preflight = _request_preflight_errors(request)
                if preflight:
                    raise ValueError("; ".join(preflight))
                manifest = self.director.direct(request)
                errors = _blocking_quality_errors(manifest)
                if errors:
                    raise ValueError("; ".join(errors))
                trace.append({"step": "repair", "attempt": repairs, "tier": tier, "status": "ok", "provenance": provenance})
                return manifest, repaired, calls, None
            except (StoryDraftError, ValueError, RuntimeError) as error:
                draft = getattr(error, "draft", locals().get("repaired", draft))
                failure = str(error)
                trace.append({
                    "step": "repair", "attempt": repairs, "tier": tier, "status": "failed", "error": failure,
                    "provenance": locals().get("provenance"), "draft": draft,
                })
        return manifest, draft, calls, failure

    def _write_once(
        self, model: StoryModel, packet: StoryWriterPacket, calls: int,
        trace: list[dict[str, object]], tier: str,
    ) -> tuple[RenderManifest | None, dict[str, object] | None, int, str | None]:
        if calls >= self.budget.max_llm_calls:
            return None, None, calls, "LLM call budget exhausted"
        try:
            calls += 1
            request, provenance, draft = model.generate(packet)
            preflight = _request_preflight_errors(request)
            if preflight:
                raise ValueError("; ".join(preflight))
            manifest = self.director.direct(request)
            errors = _blocking_quality_errors(manifest)
            if errors:
                raise ValueError("; ".join(errors))
            trace.append({"step": "write", "tier": tier, "status": "ok", "provenance": provenance})
            return manifest, draft, calls, None
        except StoryDraftError as error:
            trace.append({"step": "write", "tier": tier, "status": "invalid", "error": str(error), "draft": error.draft})
            return None, error.draft, calls, str(error)
        except (ValueError, RuntimeError) as error:
            trace.append({
                "step": "write", "tier": tier, "status": "failed", "error": str(error),
                "provenance": locals().get("provenance"), "draft": locals().get("draft"),
            })
            return None, locals().get("draft"), calls, str(error)


def _blocking_quality_errors(manifest: RenderManifest) -> list[str]:
    # Sensitive subjects may still receive an evidence-bound editorial draft,
    # but the separate publish gate remains false until human review.
    return [
        f"{item.name}: {item.detail}" for item in validate_manifest(manifest)
        if not item.passed and item.name not in {"editorial_safety_review", "music_license_record"}
    ]


def _visible_copy_only_failure(value: str) -> bool:
    clauses = [item.strip() for item in value.split(";") if item.strip()]
    return bool(clauses) and all(
        "root-post Chinese translation" in item or "fixed conclusion must fit" in item
        for item in clauses
    )


def _request_preflight_errors(request: StoryboardRequest) -> list[str]:
    """Collect independent model errors before the one allowed repair call."""
    errors: list[str] = []
    if request.github_brief:
        errors.extend(validate_github_brief(request.github_brief, request.evidence))
        expected_hook = compose_github_hook(request.github_brief)
        if request.fixed_hook != expected_hook:
            errors.append("GitHub fixed_hook must be composed from hook_stance and hook_fact")
    visible_copy = "\n".join(filter(None, [
        request.fixed_hook, request.github_brief.project_title if request.github_brief else "", request.footer,
        *(scene.caption for scene in request.scenes),
        *(scene.screen_fact for scene in request.scenes),
        *(scene.screen_interpretation for scene in request.scenes),
    ]))
    if request.editorial_brief:
        strategy = request.editorial_brief.attention_strategy
        visible_copy += "\n" + "\n".join(filter(None, [
            strategy.hook_fact, strategy.conflict, strategy.surprise, strategy.stakes,
            strategy.stance, strategy.payoff, *strategy.hook_candidates,
        ]))
    internal_labels = [
        label for label in ("仓库描述", "README声明", "证据带读", "关键结论", "翻译：", "译为：")
        if label.casefold() in visible_copy.casefold()
    ]
    hook = request.fixed_hook.strip()
    if any(hook.startswith(prefix) for prefix in ("一个工具", "这个项目", "这个仓库", "一款工具")):
        errors.append("viewer hook must name the project or lead with a concrete result, not: " + hook)
    if internal_labels:
        errors.append("viewer-facing scenes must remove internal labels: " + "、".join(internal_labels))
    generic_advice = [
        phrase for phrase in ("替你决定", "仍是你的责任", "最终还是要你", "不能代替你", "仍需你负责")
        if phrase in visible_copy
    ]
    if generic_advice:
        errors.append("viewer-facing conclusion must replace generic advice: " + "、".join(generic_advice))
    unsupported_downgrades = [
        phrase for phrase in (
            "适合快速原型而非成品交付", "只适合原型", "仅适合原型", "不适合成品",
            "不适合生产", "不能用于正式项目", "质量仍需人工把关", "原创性和质量控制仍需人工",
        ) if phrase in visible_copy
    ]
    if unsupported_downgrades:
        errors.append(
            "do not downgrade a GitHub project to prototype-only without an explicit source limitation: "
            + "、".join(unsupported_downgrades)
        )
    source_text = "\n".join(item.quote for item in request.evidence)
    ritual_uncertainty = [
        phrase for phrase in (
            "仅停留在概念层面", "只是一个概念", "不宜过度期待", "影响有待验证", "尚未成熟", "仍是未知数",
        ) if phrase in visible_copy and phrase not in source_text
    ]
    if ritual_uncertainty:
        errors.append(
            "do not end with unsupported ritual uncertainty; state implemented capability or a specific unknown instead: "
            + "、".join(ritual_uncertainty)
        )
    if re.search(
        r"(?:但|不过|然而).{0,40}(?:未知|未公开|有待|需关注|待披露|尚未.{0,8}公开)|"
        r"(?:尚未.{0,8}公开|影响范围.{0,10}(?:未知|未公开)|值得关注.{0,12}(?:后续|影响)|"
        r"需(?:持续)?关注|仍待.{0,8}(?:公开|验证|研究)|有待验证|进一步研究)[。！]?$",
        request.footer.strip(), re.IGNORECASE,
    ):
        errors.append("fixed conclusion must end on impact/payoff/action, not an unknown or ritual caveat")
    visible_manual_claim = re.search(r"(?:仍|还|必须|需要|需).{0,10}人工.{0,6}(?:检查|审核|把关|复核)", visible_copy)
    sourced_manual_boundary = re.search(
        r"人工.{0,12}(?:检查|审核|把关|复核)|(?:manual|human).{0,12}(?:check|review)|review.{0,12}(?:before|manually)",
        source_text, re.IGNORECASE,
    )
    if visible_manual_claim and not sourced_manual_boundary:
        errors.append("manual checking/review is not an automatic GitHub limitation; cite an explicit source or remove it")
    if request.topic_type.value == "github_project" and request.github_brief:
        brief = request.github_brief
        repo_name = request.candidate.title.rsplit("/", 1)[-1].strip()
        opening = brief.hook_opening.casefold()
        reveal = brief.hook_reveal.casefold()
        has_repo_context = any(item.source_kind == "github:linked_context" for item in request.evidence)
        if has_repo_context and not brief.background_actor.strip():
            errors.append(
                "README-linked vendor/background evidence exists: fill background_actor/action/consequence/evidence_ids "
                "and tell that causal event before the repository response"
            )
        if brief.background_actor.strip():
            if brief.background_actor.casefold() not in opening:
                errors.append("hook_opening must name the evidenced background actor/company/person")
            if repo_name.casefold() not in reveal:
                errors.append("hook_reveal must name the repository that responds to the background event")
        elif repo_name.casefold() not in opening:
            errors.append("hook_opening must name the repository; ownerless capability openings are not allowed")
        closing_copy = "\n".join((request.github_brief.hook_verdict, request.footer))
        if re.search(
            r"但|不过|然而|无法|不能|尽力|权衡|依赖外部|部署.{0,4}门槛|不适合|而非|仍需人工|需要人工",
            closing_copy,
        ):
            errors.append(
                "keep GitHub hook_verdict/footer value-forward; do not end on a boundary. "
                "If README states a limitation, keep it in boundary or the relevant evidence scene, not the closing"
            )
        strong_attack = re.search(r"绕过|破解|攻破|击破|干碎|失效", visible_copy)
        opening_source = "\n".join(
            item.quote if item.source_kind in {"github:metadata", "github:repository"}
            else "\n".join(item.quote.splitlines()[:80])
            for item in request.evidence
            if item.source_kind in {"github:metadata", "github:repository", "github:readme"}
        )
        explicit_attack = re.search(
            r"绕过|破解|攻破|击破|干碎|(?:^|\W)(?:bypass|crack|defeat|evade)(?:\W|$)",
            opening_source, re.IGNORECASE,
        )
        if strong_attack and not explicit_attack:
            errors.append(
                "do not upgrade strip/remove/clean/detect into bypass/crack/defeat; preserve the source verb strength"
            )
    return errors


def _opportunity_from_plan(plan: EditorialPlan, evidence: list[Evidence]) -> EditorialOpportunity:
    known = {item.id for item in evidence}
    reasons: list[SelectionReason] = []
    for index, raw in enumerate(plan.selection_reasons, start=1):
        ids = [str(item) for item in raw.get("evidence_ids", []) if str(item) in known]
        reasons.append(SelectionReason(
            id=str(raw.get("id") or f"selection-{index}"),
            dimension=str(raw.get("dimension") or "audience_relevance"),
            rationale=str(raw.get("rationale") or plan.audience_value),
            evidence_ids=ids or list(plan.selected_evidence_ids[:1]),
        ))
    if not reasons:
        reasons.append(SelectionReason(
            "selection-core", "audience_relevance", plan.audience_value,
            [item for item in plan.selected_evidence_ids if item in known][:2],
        ))
    return EditorialOpportunity(
        event_claim=plan.angle,
        why_now=plan.why_now or plan.angle,
        why_audience=plan.why_audience or plan.audience_value,
        audience_pain_or_desire=plan.audience_pain_or_desire or plan.audience_value,
        selection_reasons=reasons,
        context_hypotheses=list(plan.context_questions),
        context_gaps=list(plan.unresolved_questions),
        story_archetype=plan.story_archetype,
    )


def _context_graph_from_evidence(plan: EditorialPlan, evidence: list[Evidence]) -> ContextGraph:
    events: list[ContextEvent] = []
    required_ids: list[str] = []
    pattern_ids: list[str] = []
    discarded_ids: list[str] = []
    event_confirmation_seen = False
    chronology_is_story = plan.story_archetype in {
        "event_chain", "people_change", "price_competition", "official_update",
    }
    for index, item in enumerate(evidence, start=1):
        if not any(marker in item.source_kind for marker in ("related", "context", "search_result", "quoted")):
            continue
        event_id = f"context-{index}"
        role = str(item.metadata.get("context_role") or "")
        actor = str(item.metadata.get("author_name") or item.metadata.get("publisher") or "source context")
        relation = {
            "same_author_setup": "earlier same-author setup that explains the current post",
            "referenced_setup": "exact earlier post that explains the configuration or event shown in the screenshot",
            "incumbent_history": "earlier related move that tests whether the current event is part of a pattern",
            "event_context": "independent confirmation or direct background for the current event",
        }.get(role, "background")
        events.append(ContextEvent(
            actor=actor, action=next((line.strip() for line in item.quote.splitlines() if line.strip()), item.url)[:240],
            occurred_at=str(item.metadata.get("published_at") or item.captured_at),
            relation=relation, evidence_ids=[item.id], id=event_id,
        ))
        if role == "incumbent_history" and plan.story_archetype == "people_change":
            pattern_ids.append(event_id)
        elif role == "incumbent_history":
            discarded_ids.append(event_id)
        elif item.source_kind == "x:quoted_post":
            required_ids.append(event_id)
        elif role == "same_author_setup" and chronology_is_story:
            required_ids.append(event_id)
        # A referenced setup is not generic background. It is the exact post
        # recovered because the visible screenshot says "I followed this
        # setup" (or equivalent). Without it the audience only sees vague
        # phrases such as "another model" and cannot understand the event.
        # Keep this requirement independent of the planner's story-archetype
        # label: a weak/mistaken archetype must not discard causal evidence.
        elif role == "referenced_setup":
            required_ids.append(event_id)
        elif role == "event_context" and chronology_is_story and not event_confirmation_seen:
            required_ids.append(event_id)
            event_confirmation_seen = True
        else:
            discarded_ids.append(event_id)
    return ContextGraph(
        events=events, required_context_ids=required_ids, pattern_context_ids=pattern_ids,
        discarded_context_ids=discarded_ids,
        expansion_dimensions=list(plan.expansion_dimensions),
    )


def _usage_totals(trace: list[dict[str, object]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for item in trace:
        provenance = item.get("provenance") or {}
        usage = provenance.get("usage") or {} if isinstance(provenance, dict) else {}
        if isinstance(usage, dict):
            for name in totals:
                value = usage.get(name)
                if isinstance(value, int):
                    totals[name] += value
    return totals


def _same_document(left: str, right: str) -> bool:
    return left.rstrip("/").split("#", 1)[0] == right.rstrip("/").split("#", 1)[0]


def _body_to_text(body: bytes, content_type: str) -> str:
    decoded = body.decode("utf-8", errors="replace")
    if "html" not in content_type.casefold() and not decoded.lstrip().startswith("<"):
        return decoded[:250_000]
    parser = _VisibleTextParser()
    parser.feed(decoded)
    text = "\n".join(parser.parts)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text)[:250_000]
