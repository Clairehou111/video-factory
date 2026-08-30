from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .llm import LLMSettings
from .models import Evidence
from .openrouter import ModelQuote
from .translation import IT_TRANSLATION_CONTRACT


VISUAL_SIGNAL = re.compile(
    r"architecture|benchmark|performance|comparison|workflow|pipeline|results?|chart|diagram|"
    r"架构|基准|跑分|性能|对比|工作流|流程|结果|图表", re.IGNORECASE,
)
MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^\s)]+)(?:\s+['\"][^)]*['\"])?\)")
HTML_IMAGE = re.compile(r"<img[^>]+src=['\"]([^'\"]+)['\"][^>]*?(?:alt=['\"]([^'\"]*)['\"])?[^>]*>", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class VisualCandidate:
    url: str
    alt: str
    context: str


def find_high_value_visuals(readme: str, base_url: str, limit: int = 3) -> list[VisualCandidate]:
    candidates: list[VisualCandidate] = []
    matches = [(match.start(), match.group(2), match.group(1)) for match in MARKDOWN_IMAGE.finditer(readme)]
    matches.extend((match.start(), match.group(1), match.group(2) or "") for match in HTML_IMAGE.finditer(readme))
    for start, raw_url, alt in sorted(matches):
        before = readme[:start]
        line_start = before.rfind("\n") + 1
        local_end = readme.find("\n", start)
        local = readme[line_start:local_end if local_end >= 0 else start + 300]
        headings = list(re.finditer(r"(?m)^#{1,6}\s+(.+)$", before[max(0, len(before) - 3000):]))
        heading = headings[-1].group(1) if headings else ""
        context = re.sub(r"\s+", " ", f"{heading} {local}").strip()
        if not VISUAL_SIGNAL.search(f"{alt} {heading} {local}"):
            continue
        url = urljoin(base_url, raw_url)
        lower_url = url.casefold()
        if (
            url.startswith("data:") or any(item.url == url for item in candidates)
            or any(noise in lower_url for noise in ("shields.io/", "badge", "star-history", "/sponsor/", "/sponsors/"))
        ):
            continue
        candidates.append(VisualCandidate(url, alt, context))
        if len(candidates) >= limit:
            break
    return candidates


class OpenRouterVisualAnalyst:
    """Read charts/architecture images with the cheapest capability-qualified vision model."""

    def __init__(self, settings: LLMSettings, quote: ModelQuote):
        if settings.provider != "openrouter":
            raise ValueError("visual analyst requires OpenRouter")
        if "image" not in quote.input_modalities:
            raise ValueError("selected OpenRouter model does not accept images")
        self.settings, self.quote = settings, quote

    def analyze(self, repo_url: str, readme: str, visuals: list[VisualCandidate]) -> dict[str, object]:
        content: list[dict[str, object]] = [{"type": "text", "text": "\n".join([
            "You are reading a GitHub README for Chinese developers and vibe coders.",
            "Analyze only what the supplied README context and images visibly support.",
            "For every image return: visible_text, technical_explanation, chinese_gloss, what_it_proves, limitations.",
            "Do not infer benchmark conditions, architecture behavior, or superiority that is not visible.",
            "These results are editorial context only. Never invent browser-visible README wording.",
            IT_TRANSLATION_CONTRACT,
            "Repository: " + repo_url,
            "README text: " + readme[:30_000],
            "Image-local excerpts: " + json.dumps([item.context for item in visuals], ensure_ascii=False),
            "Return JSON: {\"images\":[{\"url\":\"\",\"visible_text\":\"\",\"technical_explanation\":\"\",\"chinese_gloss\":\"\",\"what_it_proves\":\"\",\"limitations\":\"\"}],\"editorial_takeaway\":\"\"}",
        ])}]
        for visual in visuals:
            content.append({"type": "image_url", "image_url": {"url": visual.url}})
        payload: dict[str, object] = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"}, "temperature": 0.1, "max_tokens": 2200,
            "provider": self.settings.provider_preferences,
        }
        request = Request(
            self.settings.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/video-factory", "X-Title": "Video Factory",
            },
        )
        with urlopen(request, timeout=self.settings.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        raw = result.get("choices", [{}])[0].get("message", {}).get("content")
        if not isinstance(raw, str):
            raise RuntimeError("OpenRouter vision model returned no JSON content")
        analysis = json.loads(raw)
        analysis["provenance"] = {
            "provider": "openrouter", "requested_model": self.settings.model,
            "actual_model": result.get("model", self.settings.model), "quote": self.quote.to_dict(),
            "usage": result.get("usage"),
        }
        return analysis

    def analyze_x_images(self, root_url: str, images: list[Evidence]) -> dict[str, object]:
        """Read decisive images attached to root or quoted X posts."""
        content: list[dict[str, object]] = [{"type": "text", "text": "\n".join([
            "You are extracting evidence from images attached to an X post for a Chinese technical editor.",
            "Transcribe only clearly visible text. Preserve names, handles, vendors, products, actions and chronology.",
            "Explain what conflict/event the screenshot visibly proves, and state what remains unknown.",
            "Do not infer motives, causality, employment changes, policy scope or technical behavior beyond the pixels and supplied attachment context.",
            IT_TRANSLATION_CONTRACT,
            "Root post: " + root_url,
            "Attachment index: " + json.dumps([
                {"evidence_id": item.id, "url": item.url, "context": item.quote}
                for item in images
            ], ensure_ascii=False),
            "Return JSON: {\"images\":[{\"evidence_id\":\"\",\"url\":\"\",\"visible_text\":\"\",\"chinese_translation\":\"\",\"named_entities\":[\"\"],\"what_it_proves\":\"\",\"unknowns\":\"\"}],\"editorial_takeaway\":\"\"}",
        ])}]
        for item in images:
            content.append({"type": "image_url", "image_url": {"url": item.url}})
        payload: dict[str, object] = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"}, "temperature": 0.1, "max_tokens": 2200,
            "provider": self.settings.provider_preferences,
        }
        request = Request(
            self.settings.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/video-factory", "X-Title": "Video Factory",
            },
        )
        with urlopen(request, timeout=self.settings.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        raw = result.get("choices", [{}])[0].get("message", {}).get("content")
        if not isinstance(raw, str):
            raise RuntimeError("OpenRouter X-image analyst returned no JSON content")
        analysis = json.loads(raw)
        analysis["provenance"] = {
            "provider": "openrouter", "requested_model": self.settings.model,
            "actual_model": result.get("model", self.settings.model), "quote": self.quote.to_dict(),
            "usage": result.get("usage"),
        }
        return analysis
