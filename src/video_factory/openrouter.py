from __future__ import annotations

import html
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen


MODELS_API = "https://openrouter.ai/api/v1/models?limit=1000"
DISCOUNTS_PAGE = "https://openrouter.ai/models?discount=true"
DISCOUNTS_READER = "https://r.jina.ai/https://openrouter.ai/models?discount=true"
ENDPOINTS_API = "https://openrouter.ai/api/v1/models/{model_id}/endpoints"


@dataclass(frozen=True, slots=True)
class ModelRequirements:
    purpose: str = "story"
    input_modalities: tuple[str, ...] = ("text",)
    minimum_context: int = 32_000
    require_structured_output: bool = True


@dataclass(frozen=True, slots=True)
class ModelQuote:
    model_id: str
    prompt_price: float
    completion_price: float
    context_length: int
    input_modalities: tuple[str, ...]
    discount_percent: int | None
    quality_score: float
    estimated_cost: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_discounted_models(page: str) -> dict[str, int]:
    """Extract current discounts from OpenRouter's own model-list page.

    The Models API is authoritative for effective prices but currently has no
    stable discount field.  The public page does, so we keep the two sources
    separate and fail open when its markup changes.
    """
    decoded = html.unescape(page).replace("\\u002F", "/").replace("\\/", "/")
    results: dict[str, int] = {}
    # The read-only rendering is stable Markdown.  Keep this parser before the
    # raw Next.js fallback because the page now renders model URLs as absolute
    # links and may contain no relative href attributes at all.
    markdown_model = re.compile(
        r"https://openrouter\.ai/((?:~)?[^/\s)]+/[^\s)?#]+)\)[^\n]{0,700}?(\d{1,3})%\s*off",
        re.IGNORECASE,
    )
    for match in markdown_model.finditer(decoded):
        results[match.group(1)] = int(match.group(2))
    model_pattern = re.compile(r'(?:href=["\']|"href":")/([^/"\'?]+/[^/"\'?#]+)')
    discount_pattern = re.compile(r"(\d{1,3})%\s*off", re.IGNORECASE)
    for match in model_pattern.finditer(decoded):
        window = decoded[match.start():match.start() + 1800]
        discount = discount_pattern.search(window)
        if discount:
            results[match.group(1)] = int(discount.group(1))
    # Next.js payloads sometimes put the discount before the model URL.
    for discount in discount_pattern.finditer(decoded):
        window = decoded[discount.start():discount.start() + 1800]
        model = model_pattern.search(window)
        if model:
            results.setdefault(model.group(1), int(discount.group(1)))
    return results


class OpenRouterCatalog:
    """Daily price/capability snapshot and deterministic cheapest-model picker."""

    def __init__(
        self, cache_dir: Path, fetch: Callable[[str], bytes] | None = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self.cache_dir = cache_dir
        self.fetch = fetch or self._fetch
        self.today = today

    @staticmethod
    def _fetch(url: str) -> bytes:
        request = Request(url, headers={"User-Agent": "video-factory/0.1"})
        with urlopen(request, timeout=20) as response:
            return response.read()

    def snapshot(self, refresh: bool = False) -> dict[str, object]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = self.cache_dir / f"models-{self.today().isoformat()}.json"
        if cache.is_file() and not refresh:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            # Refresh snapshots created by the old raw-HTML parser, which can
            # silently see only one of many current discount cards.
            if cached.get("discounts_fetch_url") == DISCOUNTS_READER:
                return cached
        models = json.loads(self.fetch(MODELS_API).decode("utf-8")).get("data", [])
        try:
            discounts = parse_discounted_models(self.fetch(DISCOUNTS_READER).decode("utf-8", errors="replace"))
            discount_status = "ok" if discounts else "page_parsed_no_entries"
        except Exception as error:  # pricing must remain usable if UI markup changes
            discounts, discount_status = {}, f"unavailable:{type(error).__name__}"
        payload = {
            "date": self.today().isoformat(), "models_api": MODELS_API,
            "discounts_page": DISCOUNTS_PAGE, "discounts_fetch_url": DISCOUNTS_READER,
            "discount_status": discount_status,
            "discounts": discounts, "models": models,
        }
        cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload

    def select(self, requirements: ModelRequirements, refresh: bool = False) -> ModelQuote:
        snapshot = self.snapshot(refresh=refresh)
        discounts = snapshot.get("discounts", {})
        allowlist = self._allowlist(requirements.purpose)
        eligible: list[ModelQuote] = []
        for item in snapshot.get("models", []):
            model_id = str(item.get("id", ""))
            if not model_id or model_id.endswith(":batch") or model_id == "openrouter/auto":
                continue
            if model_id.endswith(":free") and os.environ.get("OPENROUTER_ALLOW_FREE", "0") != "1":
                continue
            if allowlist and model_id not in allowlist:
                continue
            architecture = item.get("architecture") or {}
            modalities = tuple(architecture.get("input_modalities") or [])
            output_modalities = tuple(architecture.get("output_modalities") or [])
            if any(modality not in modalities for modality in requirements.input_modalities):
                continue
            if "text" not in output_modalities:
                continue
            context = int(item.get("context_length") or 0)
            if context < requirements.minimum_context:
                continue
            supported = set(item.get("supported_parameters") or [])
            if requirements.require_structured_output and "response_format" not in supported:
                continue
            pricing = item.get("pricing") or {}
            try:
                prompt, completion = float(pricing.get("prompt", -1)), float(pricing.get("completion", -1))
            except (TypeError, ValueError):
                continue
            if prompt < 0 or completion < 0:
                continue
            benchmark = (item.get("benchmarks") or {}).get("artificial_analysis") or {}
            quality_score = float(benchmark.get("intelligence_index") or 0)
            minimum_quality = float(os.environ.get(
                "OPENROUTER_MIN_INTELLIGENCE",
                "30" if requirements.purpose == "translation" else (
                    "45" if requirements.purpose == "vision" else "55"
                ),
            ))
            # A configured allowlist is an explicit operator quality decision.
            if not allowlist and quality_score < minimum_quality:
                continue
            # Compare a realistic story call: roughly 18K input + 4K output.
            estimated = prompt * 18_000 + completion * 4_000
            discount = discounts.get(model_id) if isinstance(discounts, dict) else None
            eligible.append(ModelQuote(
                model_id, prompt, completion, context, modalities,
                int(discount) if discount is not None else None, quality_score, estimated,
                "daily OpenRouter price/capability snapshot",
            ))
        if not eligible:
            raise RuntimeError(
                f"no OpenRouter model satisfies purpose={requirements.purpose}, "
                f"modalities={requirements.input_modalities}; set the matching OPENROUTER_*_MODELS allowlist"
            )
        discounted = [quote for quote in eligible if quote.discount_percent is not None]
        pool = discounted if discounted and os.environ.get("OPENROUTER_PREFER_DISCOUNTS", "1") != "0" else eligible
        chosen = min(pool, key=lambda quote: (quote.estimated_cost, -quote.context_length, quote.model_id))
        reason = (
            f"cheapest eligible {'discounted ' if chosen.discount_percent is not None else ''}model; "
            f"effective catalog estimate ${chosen.estimated_cost:.6f}/request"
        )
        return ModelQuote(**{**chosen.to_dict(), "reason": reason})

    def quote_for(self, model_id: str, refresh: bool = False) -> ModelQuote:
        """Return current capability/pricing metadata for an exact fallback model."""
        snapshot = self.snapshot(refresh=refresh)
        item = next((row for row in snapshot.get("models", []) if row.get("id") == model_id), None)
        if not isinstance(item, dict):
            raise RuntimeError(f"OpenRouter model is absent from today's catalog: {model_id}")
        architecture = item.get("architecture") or {}
        modalities = tuple(architecture.get("input_modalities") or [])
        pricing = item.get("pricing") or {}
        prompt, completion = float(pricing.get("prompt", 0)), float(pricing.get("completion", 0))
        benchmark = (item.get("benchmarks") or {}).get("artificial_analysis") or {}
        discounts = snapshot.get("discounts") or {}
        discount = discounts.get(model_id) if isinstance(discounts, dict) else None
        return ModelQuote(
            model_id=model_id, prompt_price=prompt, completion_price=completion,
            context_length=int(item.get("context_length") or 0), input_modalities=modalities,
            discount_percent=int(discount) if discount is not None else None,
            quality_score=float(benchmark.get("intelligence_index") or 0),
            estimated_cost=prompt * 18_000 + completion * 4_000,
            reason="exact multimodal reliability fallback from daily OpenRouter catalog",
        )

    @staticmethod
    def _allowlist(purpose: str) -> set[str]:
        name = {
            "vision": "OPENROUTER_VISION_MODELS",
            "translation": "OPENROUTER_TRANSLATION_MODELS",
        }.get(purpose, "OPENROUTER_TEXT_MODELS")
        return {item.strip() for item in os.environ.get(name, "").split(",") if item.strip()}
