from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from .models import RenderManifest, TopicType


FACTUAL_CATEGORY_LABELS = {
    "模型发布", "价格变化", "开源项目", "论文结果", "工具更新", "行业公告",
}


def extract_direct_identifier(manifest: RenderManifest) -> str:
    """Return only an identifier copied from archived source material."""
    if manifest.editorial_brief and manifest.editorial_brief.direct_identifier.strip():
        return manifest.editorial_brief.direct_identifier.strip()
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
            return "HF: " + "/".join(parts[:2])
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
            return prefix + match.group(1)
    for url in urls:
        match = re.search(r"github\.com/([^/?#]+/[^/?#]+)", url, re.IGNORECASE)
        if match:
            return "github.com/" + match.group(1).removesuffix(".git")
    for item in manifest.evidence:
        price_event = item.metadata.get("price_event")
        if isinstance(price_event, dict) and str(price_event.get("model_id") or "").strip():
            return "Model: " + str(price_event["model_id"]).strip()
    return ""


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
    label = factual_category_label(manifest)
    prefix = f"【{label}】" if label else ""
    title = prefix + hook
    if len(title) > 30 and prefix and len(hook) <= 30:
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
