from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from video_factory.multimodal import find_high_value_visuals
from video_factory.openrouter import (
    DISCOUNTS_PAGE, DISCOUNTS_READER, MODELS_API, ModelRequirements, OpenRouterCatalog,
    parse_discounted_models,
)


class OpenRouterCatalogTests(unittest.TestCase):
    def test_discount_page_and_api_are_joined_by_model_id(self) -> None:
        page = '<a href="/vendor/vision-fast">Vision Fast</a><div>75% off</div>'
        models = {"data": [{
            "id": "vendor/vision-fast", "context_length": 100_000,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "supported_parameters": ["response_format"],
            "pricing": {"prompt": "0.0000002", "completion": "0.000001"},
            "benchmarks": {"artificial_analysis": {"intelligence_index": 45}},
        }]}
        payloads = {MODELS_API: json.dumps(models).encode(), DISCOUNTS_READER: page.encode()}
        with tempfile.TemporaryDirectory() as temp:
            catalog = OpenRouterCatalog(Path(temp), fetch=lambda url: payloads[url], today=lambda: date(2026, 8, 21))
            quote = catalog.select(ModelRequirements("vision", ("text", "image")))
            self.assertEqual(quote.model_id, "vendor/vision-fast")
            self.assertEqual(quote.discount_percent, 75)
            self.assertTrue((Path(temp) / "models-2026-08-21.json").is_file())

    def test_discount_parser_handles_live_html_shape(self) -> None:
        self.assertEqual(parse_discounted_models(
            '<a class="x" href="/google/gemini-flash">Gemini</a><div>90% off</div>'
        ), {"google/gemini-flash": 90})

    def test_daily_cache_prevents_repeated_fetches(self) -> None:
        calls: list[str] = []
        models = {"data": []}
        def fetch(url: str) -> bytes:
            calls.append(url)
            return json.dumps(models).encode() if url == MODELS_API else b""
        with tempfile.TemporaryDirectory() as temp:
            catalog = OpenRouterCatalog(Path(temp), fetch=fetch, today=lambda: date(2026, 8, 21))
            catalog.snapshot()
            catalog.snapshot()
        self.assertEqual(calls, [MODELS_API, DISCOUNTS_READER])

    def test_visual_detector_rejects_badges_and_star_history(self) -> None:
        readme = """# Demo\n## Architecture\n![pipeline](docs/architecture.png)\n## Star History\n![](https://star-history.com/chart?a=1)"""
        visuals = find_high_value_visuals(readme, "https://raw.githubusercontent.com/a/b/main/README.md")
        self.assertEqual([item.url for item in visuals], [
            "https://raw.githubusercontent.com/a/b/main/docs/architecture.png",
        ])


if __name__ == "__main__":
    unittest.main()
