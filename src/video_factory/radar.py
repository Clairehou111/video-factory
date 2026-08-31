from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from .models import RenderManifest, TopicType


FACTUAL_CATEGORY_LABELS = {
    "模型发布", "价格变化", "开源项目", "论文结果", "工具更新", "行业公告",
}


class AdvisoryRecoveryAction(StrEnum):
    RETRY_BACKOFF = "retry_backoff"
    SPLIT_ATTACHMENTS = "split_attachments"
    USE_STORYBOARD = "use_storyboard"
    START_CLEAN_SESSION = "start_clean_session"
    NEEDS_HUMAN_AUTHORIZATION = "needs_human_authorization"
    STOP = "stop"


@dataclass(frozen=True)
class AdvisoryRecoveryDecision:
    action: AdvisoryRecoveryAction
    delay_seconds: int = 0
    next_attachment_count: int = 0
    reason: str = ""


class TechnicalArtifactKind(StrEnum):
    HUGGING_FACE_MODEL = "hugging_face_model"
    PYTHON_PACKAGE = "python_package"
    CONTAINER_IMAGE = "container_image"
    GITHUB_REPOSITORY = "github_repository"
    MODEL_ID = "model_id"
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class TechnicalArtifact:
    kind: TechnicalArtifactKind
    value: str
    display: str
    source: str


def plan_advisory_recovery(
    error: str, *, attachment_count: int, media_kind: str, attempt: int,
) -> AdvisoryRecoveryDecision:
    """Choose a bounded fallback for an external visual-review provider.

    This deliberately never enables billing, links a paid key, or retries an
    ambiguous chargeable action.  UI integrations can execute the returned
    action and persist it in their run trace without embedding provider-specific
    click logic in the editorial pipeline.
    """
    detail = error.casefold()
    kind = media_kind.casefold().strip()
    delay = min(30, 2 ** max(1, attempt + 1))
    if any(marker in detail for marker in (
        "paid api key", "set up billing", "billing required", "permission denied",
    )):
        return AdvisoryRecoveryDecision(
            AdvisoryRecoveryAction.NEEDS_HUMAN_AUTHORIZATION,
            reason="provider requires an explicitly authorized paid credential",
        )
    if any(marker in detail for marker in ("file types are not supported", "unsupported file type")):
        if kind in {"video", "gif", "animated_image"}:
            return AdvisoryRecoveryDecision(
                AdvisoryRecoveryAction.USE_STORYBOARD,
                reason="provider rejected motion media; preserve shot order in a contact sheet",
            )
        return AdvisoryRecoveryDecision(
            AdvisoryRecoveryAction.STOP,
            reason="provider rejected the already-degraded review artifact",
        )
    if "internal error" in detail or "unexpected error" in detail:
        if attachment_count > 1:
            return AdvisoryRecoveryDecision(
                AdvisoryRecoveryAction.SPLIT_ATTACHMENTS,
                delay_seconds=delay,
                next_attachment_count=max(1, (attachment_count + 1) // 2),
                reason="reduce a failing multimodal payload without dropping review coverage",
            )
        if attachment_count == 1 and kind == "video":
            return AdvisoryRecoveryDecision(
                AdvisoryRecoveryAction.USE_STORYBOARD,
                delay_seconds=delay,
                next_attachment_count=1,
                reason="single-video review failed; preserve visual sequence as a storyboard",
            )
        if attachment_count == 0 and attempt < 2:
            return AdvisoryRecoveryDecision(
                AdvisoryRecoveryAction.START_CLEAN_SESSION,
                delay_seconds=delay,
                reason="text-only failure after media errors indicates contaminated session state",
            )
    if attempt < 2:
        return AdvisoryRecoveryDecision(
            AdvisoryRecoveryAction.RETRY_BACKOFF,
            delay_seconds=delay,
            next_attachment_count=max(0, attachment_count),
            reason="transient provider failure within the bounded retry budget",
        )
    return AdvisoryRecoveryDecision(
        AdvisoryRecoveryAction.STOP,
        reason="bounded advisory recovery budget exhausted",
    )


def _explicit_artifact(value: str) -> TechnicalArtifact:
    normalized = value.strip()
    for prefix, kind in (
        ("HF: ", TechnicalArtifactKind.HUGGING_FACE_MODEL),
        ("pip install ", TechnicalArtifactKind.PYTHON_PACKAGE),
        ("docker pull ", TechnicalArtifactKind.CONTAINER_IMAGE),
        ("github.com/", TechnicalArtifactKind.GITHUB_REPOSITORY),
        ("Model: ", TechnicalArtifactKind.MODEL_ID),
    ):
        if normalized.casefold().startswith(prefix.casefold()):
            return TechnicalArtifact(
                kind, normalized[len(prefix):].strip(), normalized,
                "editorial_brief.direct_identifier",
            )
    return TechnicalArtifact(
        TechnicalArtifactKind.EXPLICIT, normalized, normalized,
        "editorial_brief.direct_identifier",
    )


def extract_technical_artifact(manifest: RenderManifest) -> TechnicalArtifact | None:
    """Extract one typed, source-backed developer artifact from archived material."""
    if manifest.editorial_brief and manifest.editorial_brief.direct_identifier.strip():
        return _explicit_artifact(manifest.editorial_brief.direct_identifier)
    urls = [
        *manifest.source_urls,
        *(item.url for item in manifest.evidence),
        *(
            str(item.metadata.get("resolved_parent_url") or "")
            for item in manifest.evidence
        ),
        *(
            value for scene in manifest.scenes for cue in scene.recording_cues
            for value in (cue.value or "", cue.target or "") if value.startswith("http")
        ),
        *(
            shot.source_url for shot in (
                manifest.editorial_brief.evidence_shots if manifest.editorial_brief else []
            ) if shot.source_url
        ),
    ]
    reserved_hf_roots = {
        "api", "assets", "datasets", "docs", "models", "settings", "social-thumbnails", "spaces",
    }
    for url in urls:
        parsed = urlparse(url)
        if not (parsed.hostname or "").casefold().endswith("huggingface.co"):
            continue
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0].casefold() not in reserved_hf_roots:
            value = "/".join(parts[:2])
            return TechnicalArtifact(
                TechnicalArtifactKind.HUGGING_FACE_MODEL, value, "HF: " + value, url,
            )
    texts = [
        *(item.quote for item in manifest.evidence),
        *(scene.source_excerpt or "" for scene in manifest.scenes),
        *(scene.screen_fact or "" for scene in manifest.scenes),
    ]
    joined = "\n".join(texts)
    for pattern, prefix in (
        (r"\bpip\s+install\s+([A-Za-z0-9_.-]+)", "pip install "),
        (r"\bdocker\s+pull\s+([^\s`]+)", "docker pull "),
    ):
        match = re.search(pattern, joined, re.IGNORECASE)
        if match:
            value = match.group(1)
            kind = (
                TechnicalArtifactKind.PYTHON_PACKAGE
                if prefix.startswith("pip") else TechnicalArtifactKind.CONTAINER_IMAGE
            )
            return TechnicalArtifact(kind, value, prefix + value, "archived_evidence_text")
    for url in urls:
        match = re.search(r"github\.com/([^/?#]+/[^/?#]+)", url, re.IGNORECASE)
        if match:
            value = match.group(1).removesuffix(".git")
            return TechnicalArtifact(
                TechnicalArtifactKind.GITHUB_REPOSITORY, value,
                "github.com/" + value, url,
            )
    for item in manifest.evidence:
        price_event = item.metadata.get("price_event")
        if isinstance(price_event, dict) and str(price_event.get("model_id") or "").strip():
            value = str(price_event["model_id"]).strip()
            return TechnicalArtifact(
                TechnicalArtifactKind.MODEL_ID, value, "Model: " + value,
                f"evidence:{item.id}:price_event.model_id",
            )
    return None


def extract_direct_identifier(manifest: RenderManifest) -> str:
    """Return the presentation form of a typed, source-backed artifact."""
    artifact = extract_technical_artifact(manifest)
    return artifact.display if artifact else ""


def factual_category_label(manifest: RenderManifest) -> str:
    explicit = (
        manifest.editorial_brief.category_label.strip()
        if manifest.editorial_brief else ""
    )
    if explicit in FACTUAL_CATEGORY_LABELS:
        return explicit
    if manifest.github_brief or manifest.topic_type == TopicType.GITHUB_PROJECT:
        return "开源项目"
    if manifest.topic_type == TopicType.RESEARCH_OR_BENCHMARK:
        return "论文结果"
    if any(isinstance(item.metadata.get("price_event"), dict) for item in manifest.evidence):
        return "价格变化"
    return ""


def build_tencent_radar_copy(
    manifest: RenderManifest, *, fallback_title: str, publisher: str, source_url: str,
) -> tuple[str, str]:
    """Build factual external packaging; callers may still override it."""
    hook = (manifest.fixed_title or manifest.fixed_hook or fallback_title).strip()
    title = hook
    if len(title) > 30:
        title = title[:30]
        # Never leave half of an adjacent ASCII model/project token at the
        # platform boundary. A shorter complete title is preferable.
        if len(hook) > 30 and title[-1].isascii() and hook[30:31] and hook[30].isascii():
            boundary = max(title.rfind(mark) for mark in (" ", "｜", "：", ":", "，"))
            if boundary >= 12:
                title = title[:boundary].rstrip()
    conclusion = (manifest.fixed_footer or "").strip()
    identifier = extract_direct_identifier(manifest)
    rows = [value for value in (conclusion, identifier) if value]
    rows.append(f"来源：{publisher.strip() or '原始来源'}｜{source_url}")
    return title, "\n".join(rows)
