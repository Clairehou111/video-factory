from __future__ import annotations

import unittest
import json
import re
import subprocess
from unittest.mock import MagicMock, patch
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from video_factory.discovery import (
    ChannelConfig, DiscoveryCandidate, DiscoveryChannel, ResourceDiscoveryConfig,
    OpenRouterDiscountDiscoveryAdapter, RSSDiscoveryAdapter, ResourceDiscoveryService,
    assign_event_clusters, evaluate_candidate,
    select_parallel_candidates, XDiscoveryAdapter,
)
from video_factory.openrouter import DISCOUNTS_READER, ENDPOINTS_API, MODELS_API, parse_discounted_models
from video_factory.quality import CheckResult
from video_factory.storage import Workspace


NOW = datetime(2026, 8, 28, 6, 0, tzinfo=UTC)


def x_candidate(identifier: str, title: str, hours_ago: int = 1) -> DiscoveryCandidate:
    body = (
        f"{title}. The team released an AI agent API today with three concrete tools. "
        "It searches documentation, reads exact sections, and returns cited results. "
        "Developers can use the API now in production work."
    )
    return DiscoveryCandidate(
        id=identifier, channel=DiscoveryChannel.X,
        url=f"https://x.com/example/status/{identifier.rsplit('-', 1)[-1]}", title=title,
        author="example", publisher="X", published_at=(NOW - timedelta(hours=hours_ago)).isoformat(),
        summary=body, body_text=body, stable_id=f"x:{identifier}", discovered_at=NOW.isoformat(),
    )


class StaticAdapter:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = 0

    def search(self, config, now):
        self.calls += 1
        return self.candidates


class FakeFactory:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.generate_calls = []
        self.rerender_calls = []

    def generate(self, url, options):
        self.generate_calls.append((url, options))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def rerender(self, manifest):
        self.rerender_calls.append(manifest)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class DiscoveryTest(unittest.TestCase):
    def test_x_adapter_retries_ok_false_then_uses_opencli(self) -> None:
        commands = []

        def runner(command, **kwargs):
            commands.append(command)
            if command[0] == "twitter":
                return subprocess.CompletedProcess(
                    command, 0, json.dumps({"ok": False, "error": {"message": "HTTP 404"}}), "",
                )
            payload = [{
                "id": "123", "text": "A concrete AI agent SDK release with API tools and benchmark results for developers.",
                "author": {"screenName": "builder"}, "created_at": NOW.isoformat(),
                "url": "https://x.com/builder/status/123",
            }]
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

        config = ChannelConfig.from_dict(DiscoveryChannel.X, {
            "queries": ["AI agent"], "seed_accounts": [],
        })
        rows = XDiscoveryAdapter(runner).search(config, NOW)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].stable_id, "x:123")
        self.assertEqual([command[0] for command in commands], ["twitter", "twitter", "opencli"])

    def test_default_channel_cadences_are_independent(self) -> None:
        config = ResourceDiscoveryConfig()

        self.assertEqual(config.channels[DiscoveryChannel.X].cadence_hours, 2)
        self.assertEqual(config.channels[DiscoveryChannel.NEWS].cadence_hours, 2)
        self.assertEqual(config.channels[DiscoveryChannel.NEWS_ZH].cadence_hours, 2)
        self.assertEqual(config.channels[DiscoveryChannel.OFFICIAL].cadence_hours, 2)
        self.assertEqual(config.channels[DiscoveryChannel.OFFICIAL_ZH].cadence_hours, 2)
        self.assertEqual(config.channels[DiscoveryChannel.PAPER].cadence_hours, 24)
        self.assertEqual(config.channels[DiscoveryChannel.GITHUB].cadence_hours, 48)
        self.assertEqual(config.channels[DiscoveryChannel.YOUTUBE].cadence_hours, 48)
        self.assertEqual(config.channels[DiscoveryChannel.OPENROUTER].cadence_hours, 2)

    def test_chinese_llm_official_and_news_sources_are_in_defaults(self) -> None:
        config = ResourceDiscoveryConfig()
        official = config.channels[DiscoveryChannel.OFFICIAL_ZH]
        news = config.channels[DiscoveryChannel.NEWS_ZH]

        for domain in (
            "deepseek.com", "zhipuai.cn", "kimi.com", "qwen.ai", "volcengine.com",
            "hunyuan.tencent.com", "qianfan.cloud.baidu.com", "minimaxi.com", "stepfun.com",
            "baichuan-ai.com", "01.ai", "sensenova.cn", "xfyun.cn", "huaweicloud.com",
        ):
            self.assertIn(domain, official.seed_domains)
        for domain in ("36kr.com", "caixin.com", "jiemian.com", "cls.cn", "qbitai.com"):
            self.assertIn(domain, news.seed_domains)
        self.assertTrue(any("价格战" in query for query in news.queries))
        self.assertTrue(any("开放权重" in query for query in official.queries))
        self.assertFalse(any(re.search(r"[\u3400-\u9fff]", query) for query in config.channels[DiscoveryChannel.NEWS].queries))
        self.assertFalse(any(re.search(r"[\u3400-\u9fff]", query) for query in config.channels[DiscoveryChannel.OFFICIAL].queries))

    def test_chinese_queries_use_chinese_google_news_locale(self) -> None:
        url = RSSDiscoveryAdapter._google_news_url("智谱 新模型 发布")

        self.assertIn("hl=zh-CN", url)
        self.assertIn("gl=CN", url)
        self.assertIn("ceid=CN:zh-Hans", url)

    def test_rss_download_falls_back_to_curl_after_tls_failure(self) -> None:
        with patch("video_factory.discovery.urlopen", side_effect=OSError("TLS EOF")), patch(
            "video_factory.discovery.subprocess.run",
            return_value=subprocess.CompletedProcess(["curl"], 0, b"<rss/>", b""),
        ):
            payload = RSSDiscoveryAdapter._download("https://news.example/rss")

        self.assertEqual(payload, b"<rss/>")

    def test_news_filters_trusted_publishers_before_probe_limit(self) -> None:
        rows = []
        for index in range(8):
            rows.append(
                f"<item><title>Untrusted {index}</title><link>https://news.example/u{index}</link>"
                f"<pubDate>Fri, 28 Aug 2026 05:{59-index:02d}:00 GMT</pubDate>"
                "<description>untrusted story</description>"
                '<source url="https://untrusted.example">Untrusted</source></item>'
            )
        rows.append(
            "<item><title>Trusted model launch</title><link>https://news.example/trusted</link>"
            "<pubDate>Fri, 28 Aug 2026 05:40:00 GMT</pubDate>"
            "<description>trusted story</description>"
            '<source url="https://www.reuters.com">Reuters</source></item>'
        )
        payload = ("<rss><channel>" + "".join(rows) + "</channel></rss>").encode()
        adapter = RSSDiscoveryAdapter(
            DiscoveryChannel.NEWS,
            fetcher=lambda url: ("AI model launch with API details. " * 30, "https://www.reuters.com/p/1"),
        )
        adapter._download = lambda url: payload
        config = ChannelConfig.from_dict(DiscoveryChannel.NEWS, {
            "queries": ["AI model"], "seed_domains": ["reuters.com"], "probe_limit": 1,
        })

        found = adapter.search(config, NOW)

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].publisher, "Reuters")

    def test_english_and_chinese_news_are_independent_channels(self) -> None:
        english = (
            "<rss><channel><item><title>English model launch</title>"
            "<link>https://news.example/en</link><pubDate>Fri, 28 Aug 2026 05:59:00 GMT</pubDate>"
            "<description>launch</description>"
            '<source url="https://www.reuters.com">Reuters</source></item></channel></rss>'
        ).encode()
        chinese = (
            "<rss><channel><item><title>智谱发布新模型</title>"
            "<link>https://news.example/zh</link><pubDate>Fri, 28 Aug 2026 05:58:00 GMT</pubDate>"
            "<description>发布</description>"
            '<source url="https://www.36kr.com">36Kr</source></item></channel></rss>'
        ).encode()
        english_adapter = RSSDiscoveryAdapter(
            DiscoveryChannel.NEWS,
            fetcher=lambda url: ("AI launch. " * 80, "https://www.reuters.com/ai-launch"),
        )
        chinese_adapter = RSSDiscoveryAdapter(
            DiscoveryChannel.NEWS_ZH,
            fetcher=lambda url: ("智谱发布新模型。" * 80, "https://www.36kr.com/p/ai-launch"),
        )
        english_adapter._download = lambda url: english
        chinese_adapter._download = lambda url: chinese

        english_found = english_adapter.search(ChannelConfig.from_dict(DiscoveryChannel.NEWS, {
            "queries": ["AI model launch"], "seed_domains": ["reuters.com"], "probe_limit": 1,
        }), NOW)
        chinese_found = chinese_adapter.search(ChannelConfig.from_dict(DiscoveryChannel.NEWS_ZH, {
            "queries": ["智谱 新模型 发布"], "seed_domains": ["36kr.com"], "probe_limit": 1,
        }), NOW)

        self.assertEqual(english_found[0].channel, DiscoveryChannel.NEWS)
        self.assertEqual(english_found[0].publisher, "Reuters")
        self.assertEqual(chinese_found[0].channel, DiscoveryChannel.NEWS_ZH)
        self.assertEqual(chinese_found[0].publisher, "36Kr")

    def test_chinese_official_query_excludes_non_chinese_vendor_domains(self) -> None:
        requested = []
        adapter = RSSDiscoveryAdapter(
            DiscoveryChannel.OFFICIAL_ZH,
            fetcher=lambda url: ("", url),
        )
        adapter._download = lambda url: requested.append(url) or b"<rss><channel/></rss>"
        config = ChannelConfig.from_dict(DiscoveryChannel.OFFICIAL_ZH, {
            "queries": ["新模型 发布"],
            "feeds": [],
        })

        adapter.search(config, NOW)

        self.assertEqual(len(requested), 1)
        self.assertIn("site%3Adeepseek.com", requested[0])
        self.assertIn("site%3Akimi.com", requested[0])
        self.assertNotIn("site%3Aopenai.com", requested[0])
        self.assertNotIn("site%3Amicrosoft.com", requested[0])

    def test_chinese_official_model_launch_passes_event_gate(self) -> None:
        body = (
            "智谱正式发布新模型 GLM-6，并开放 API。新模型支持更长上下文、工具调用和多模态输入。"
            "官方页面给出了三个开发示例、模型能力说明、上线范围和迁移时间，开发者今天即可使用。"
        ) * 10
        item = DiscoveryCandidate(
            id="glm-launch", channel=DiscoveryChannel.OFFICIAL_ZH,
            url="https://www.zhipuai.cn/news/glm-6", title="智谱正式发布 GLM-6 新模型",
            publisher="智谱", published_at=NOW.isoformat(), summary=body, body_text=body,
            metadata={"image_count": 2},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.OFFICIAL_ZH, {}), NOW)

        self.assertTrue(item.eligible)
        self.assertEqual(item.topic_type.value, "model_or_product")

    def test_glm_model_card_is_attributed_to_chinese_official_channel(self) -> None:
        payload = (
            "<rss><channel><item><title>GLM-5.3-Flash - 智谱AI开放文档</title>"
            "<link>https://news.example/glm-5-3-flash</link>"
            "<pubDate>Fri, 28 Aug 2026 05:58:00 GMT</pubDate>"
            "<description>智谱正式发布并开源 GLM-5.3-Flash</description>"
            '<source url="https://docs.bigmodel.cn">智谱AI开放文档</source>'
            "</item></channel></rss>"
        ).encode()
        adapter = RSSDiscoveryAdapter(
            DiscoveryChannel.OFFICIAL_ZH,
            fetcher=lambda url: (
                "智谱正式发布并开源 GLM-5.3-Flash。模型采用重新训练的基础模型。" * 20,
                "https://docs.bigmodel.cn/cn/guide/models/glm-5-3-flash",
            ),
        )
        adapter._download = lambda url: payload

        found = adapter.search(ChannelConfig.from_dict(DiscoveryChannel.OFFICIAL_ZH, {
            "queries": ["GLM-5.3-Flash 发布 开源"], "probe_limit": 1,
        }), NOW)

        self.assertEqual(found[0].channel, DiscoveryChannel.OFFICIAL_ZH)
        self.assertEqual(found[0].metadata["source_class"], "official")
        self.assertEqual(found[0].metadata["language"], "zh")
        self.assertEqual(found[0].url, "https://docs.bigmodel.cn/cn/guide/models/glm-5-3-flash")

    def test_official_channel_rejects_unresolved_google_news_wrapper(self) -> None:
        payload = (
            "<rss><channel><item><title>Official model launch</title>"
            "<link>https://news.google.com/rss/articles/wrapper</link>"
            "<pubDate>Fri, 28 Aug 2026 05:58:00 GMT</pubDate>"
            "<description>Official launch</description>"
            '<source url="https://x.ai">X.ai</source>'
            "</item></channel></rss>"
        ).encode()
        adapter = RSSDiscoveryAdapter(
            DiscoveryChannel.OFFICIAL,
            fetcher=lambda url: ("Google News wrapper", url),
        )
        adapter._download = lambda url: payload

        found = adapter.search(ChannelConfig.from_dict(DiscoveryChannel.OFFICIAL, {
            "queries": ["model launch"], "seed_domains": ["x.ai"], "probe_limit": 1,
        }), NOW)

        self.assertEqual(found, [])

    def test_small_chinese_llm_promotion_is_rejected(self) -> None:
        body = (
            "Kimi API 推出限时优惠，调用价格折扣 5%。活动页面说明参与方式、套餐范围和结束时间。"
            "这是一次常规促销，模型能力、上下文、API 功能和产品可用范围均没有变化。"
        ) * 12
        item = DiscoveryCandidate(
            id="kimi-small-sale", channel=DiscoveryChannel.OFFICIAL_ZH,
            url="https://platform.kimi.com/promotion", title="Kimi API 限时优惠 5%",
            publisher="Moonshot AI", published_at=NOW.isoformat(), summary=body, body_text=body,
            metadata={"image_count": 1},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.OFFICIAL_ZH, {}), NOW)

        self.assertFalse(item.eligible)
        self.assertIn("routine_chinese_llm_promotion", item.rejection_reasons)

    def test_openrouter_markdown_discount_parser_handles_absolute_links(self) -> None:
        page = (
            "[Solar Pro 4](https://openrouter.ai/upstage/solar-pro4)90% off 524K context"
            "$0.03/M input tokens$0.12/M output tokens\n"
            "[Small sale](https://openrouter.ai/acme/model)15% off"
        )

        self.assertEqual(parse_discounted_models(page), {
            "upstage/solar-pro4": 90, "acme/model": 15,
        })

    def test_openrouter_gate_rejects_routine_promotion(self) -> None:
        item = DiscoveryCandidate(
            id="openrouter-routine", channel=DiscoveryChannel.OPENROUTER,
            url="https://openrouter.ai/acme/model", title="Acme model is 20% off",
            author="OpenRouter", publisher="OpenRouter", published_at=NOW.isoformat(),
            summary="A routine endpoint promotion with exact token prices and a stable provider. " * 5,
            body_text="A routine endpoint promotion with exact token prices and a stable provider. " * 5,
            metadata={"compelling": False, "endpoint_uptime": 99.99, "visual_path": "model_page"},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.OPENROUTER, {}), NOW)

        self.assertFalse(item.eligible)
        self.assertIn("promotion_not_compelling", item.rejection_reasons)

    def test_openrouter_adapter_finds_deepseek_cheaper_than_official(self) -> None:
        model_id = "deepseek/deepseek-v4-flash-0731"
        models = {"data": [{
            "id": model_id, "name": "DeepSeek: DeepSeek V4 Flash 0731",
            "created": int((NOW - timedelta(days=28)).timestamp()), "context_length": 1_310_720,
            "architecture": {"output_modalities": ["text"]},
            "benchmarks": {"artificial_analysis": {"intelligence_index": 51.8, "coding_index": 69.1}},
        }]}
        endpoints = {"data": {"endpoints": [
            {"provider_name": "OpenInference", "status": 0, "uptime_last_30m": 99.99,
             "pricing": {"prompt": "0.00000003", "completion": "0.0000001", "discount": 0}},
            {"provider_name": "DeepSeek", "status": 0, "uptime_last_30m": 99.98,
             "pricing": {"prompt": "0.00000022", "completion": "0.00000066", "discount": 0,
                         "overrides": [{"prompt": "0.00000044", "completion": "0.00000132"}]}},
        ]}}
        payloads = {
            MODELS_API: json.dumps(models).encode(), DISCOUNTS_READER: b"# no listed discount",
            ENDPOINTS_API.format(model_id=model_id): json.dumps(endpoints).encode(),
        }
        adapter = OpenRouterDiscountDiscoveryAdapter(lambda url: payloads[url])
        config = ChannelConfig.from_dict(DiscoveryChannel.OPENROUTER, {"probe_limit": 8})

        rows = adapter.search(config, NOW)
        evaluate_candidate(rows[0], config, NOW)

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].eligible)
        self.assertIn("cheaper_than_official_vendor", rows[0].metadata["attraction_reasons"])
        self.assertAlmostEqual(
            rows[0].metadata["official_comparison"]["savings_offpeak_percent"], 85.8, places=1,
        )
        self.assertEqual(rows[0].content_type.value, "flash")

    def test_openrouter_events_do_not_drip_into_later_promo_videos(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            item = DiscoveryCandidate(
                id="openrouter-price-event-1", channel=DiscoveryChannel.OPENROUTER,
                url="https://openrouter.ai/deepseek/deepseek-v4-flash-0731",
                title="DeepSeek V4 Flash is 86% cheaper than the official off-peak endpoint",
                author="OpenRouter", publisher="OpenRouter", published_at=NOW.isoformat(),
                summary="Exact endpoint pricing, provider, uptime, and workload comparison. " * 7,
                body_text="Exact endpoint pricing, provider, uptime, and workload comparison. " * 7,
                metadata={
                    "compelling": True, "endpoint_uptime": 99.99, "visual_path": "model_page",
                    "linked_sources": ["https://api-docs.deepseek.com/quick_start/pricing"],
                },
            )
            adapter = StaticAdapter([item])
            factory = FakeFactory([{"status": "completed", "publishable": True, "video": "final.mp4"}])
            service = ResourceDiscoveryService(
                workspace, adapters={DiscoveryChannel.OPENROUTER: adapter}, factory=factory,
                clock=lambda: NOW, sleeper=lambda _: None,
            )
            config = ResourceDiscoveryConfig()
            for channel in DiscoveryChannel:
                config.channels[channel].enabled = channel == DiscoveryChannel.OPENROUTER

            first = service.run(config, scheduled=False)
            second = service.run(config, scheduled=False)

            self.assertEqual(first.channels["openrouter"].status, "generated")
            self.assertEqual(second.channels["openrouter"].status, "no_selection")
            self.assertEqual(len(factory.generate_calls), 1)
            self.assertIn("price_event_already_seen", item.rejection_reasons)

    def test_channel_and_topic_are_independent(self) -> None:
        item = x_candidate("x-1", "Acme raises a Series B for its AI product")
        config = ChannelConfig.from_dict(DiscoveryChannel.X, {})

        evaluate_candidate(item, config, NOW)

        self.assertEqual(item.channel, DiscoveryChannel.X)
        self.assertEqual(item.topic_type.value, "company_or_team")
        self.assertTrue(item.eligible)

    def test_github_gate_requires_trial_and_concrete_demo(self) -> None:
        weak = DiscoveryCandidate(
            id="github-a-b", channel=DiscoveryChannel.GITHUB,
            url="https://github.com/a/b", title="a/b", author="a", publisher="GitHub",
            published_at=(NOW - timedelta(days=1)).isoformat(), body_text="Architecture notes. " * 80,
        )

        evaluate_candidate(weak, ChannelConfig.from_dict(DiscoveryChannel.GITHUB, {}), NOW)

        self.assertFalse(weak.eligible)
        self.assertIn("missing_trial_path", weak.rejection_reasons)
        self.assertIn("missing_concrete_io_or_demo", weak.rejection_reasons)

    def test_youtube_gate_rejects_video_without_transcript(self) -> None:
        item = DiscoveryCandidate(
            id="youtube-demo", channel=DiscoveryChannel.YOUTUBE,
            url="https://youtube.com/watch?v=demo", title="AI agent engineering interview",
            author="Original Channel", publisher="Original Channel", published_at=NOW.isoformat(),
            summary="A detailed AI agent engineering interview with concrete systems and lessons. " * 6,
            body_text="A detailed AI agent engineering interview with concrete systems and lessons. " * 6,
            metadata={"duration_seconds": 1800, "transcript_available": False},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.YOUTUBE, {}), NOW)

        self.assertFalse(item.eligible)
        self.assertIn("transcript_unavailable", item.rejection_reasons)

    def test_parallel_matching_makes_one_video_per_event_and_advances_channel(self) -> None:
        x_launch = x_candidate("x-1", "Mistral launches Agentic Search")
        official_launch = DiscoveryCandidate(
            id="official-1", channel=DiscoveryChannel.OFFICIAL,
            url="https://mistral.ai/news/agentic-search", title="Introducing Mistral Agentic Search",
            publisher="Mistral", published_at=NOW.isoformat(), eligible=True, score=96,
        )
        official_other = DiscoveryCandidate(
            id="official-2", channel=DiscoveryChannel.OFFICIAL,
            url="https://openai.com/news/new-api", title="OpenAI releases a new API toolkit",
            publisher="OpenAI", published_at=NOW.isoformat(), eligible=True, score=88,
        )
        x_launch.eligible, x_launch.score = True, 94
        candidates = [x_launch, official_launch, official_other]
        assign_event_clusters(candidates)

        selected = select_parallel_candidates({
            DiscoveryChannel.X: [x_launch],
            DiscoveryChannel.OFFICIAL: [official_launch, official_other],
        })

        self.assertEqual(selected[DiscoveryChannel.X].id, "x-1")
        self.assertEqual(selected[DiscoveryChannel.OFFICIAL].id, "official-2")

    def test_service_auto_adopts_and_obeys_next_run(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            item = x_candidate("x-1", "Acme launches an agent SDK")
            adapter = StaticAdapter([item])
            factory = FakeFactory([{"status": "completed", "publishable": True, "video": "final.mp4"}])
            service = ResourceDiscoveryService(
                workspace, adapters={DiscoveryChannel.X: adapter}, factory=factory,
                clock=lambda: NOW, sleeper=lambda _: None,
            )
            config = ResourceDiscoveryConfig()
            for channel in DiscoveryChannel:
                config.channels[channel].enabled = channel == DiscoveryChannel.X

            first = service.run(config)
            second = service.run(config)

            self.assertEqual(first.channels["x"].status, "generated")
            self.assertEqual(len(factory.generate_calls), 1)
            self.assertEqual(second.channels["x"].status, "not_due")
            self.assertEqual(adapter.calls, 1)

    def test_adoption_retries_same_candidate_three_times(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            item = x_candidate("x-1", "Acme launches an agent SDK")
            adapter = StaticAdapter([item])
            factory = FakeFactory([
                RuntimeError("browser failed"), RuntimeError("browser failed again"),
                {"status": "completed", "publishable": True, "video": "final.mp4"},
            ])
            service = ResourceDiscoveryService(
                workspace, adapters={DiscoveryChannel.X: adapter}, factory=factory,
                clock=lambda: NOW, sleeper=lambda _: None,
            )
            config = ResourceDiscoveryConfig()
            for channel in DiscoveryChannel:
                config.channels[channel].enabled = channel == DiscoveryChannel.X

            result = service.run(config)

            adoption = result.channels["x"].adoption
            self.assertEqual(adoption["status"], "generated")
            self.assertEqual(len(adoption["attempts"]), 3)
            self.assertEqual(len(factory.generate_calls), 3)

    def test_x_canonical_author_url_reuses_failed_manifest_without_llm_retry(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            job = workspace.root / "jobs" / "failed-x-capture"
            job.mkdir(parents=True)
            manifest = job / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            (job / "result.json").write_text(json.dumps({
                "url": "https://x.com/Builder/status/2093612518396871075",
                "status": "failed", "manifest": str(manifest),
            }), encoding="utf-8")
            item = x_candidate("x-2093612518396871075", "Acme launches an agent SDK")
            item.url = "https://x.com/i/status/2093612518396871075"
            factory = FakeFactory([RuntimeError("capture failed")])
            factory.rerender = MagicMock(return_value={
                "status": "completed", "publishable": True, "video": "final.mp4",
                "manifest": str(manifest),
            })
            delays: list[int] = []
            service = ResourceDiscoveryService(
                workspace, factory=factory, clock=lambda: NOW, sleeper=delays.append,
            )

            result = service._adopt(
                item, ResourceDiscoveryConfig(retry_backoff_seconds=[0, 30]), "auto", None,
            )

            self.assertEqual(result["status"], "generated")
            self.assertEqual(len(factory.generate_calls), 1)
            factory.rerender.assert_called_once_with(manifest)
            self.assertEqual(delays, [])
            self.assertEqual(result["attempts"][1]["mode"], "deterministic_rerender")

    def test_needs_human_candidate_resumes_from_manifest_before_any_llm_call(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            job = workspace.root / "jobs" / "prior-x-attempt"
            job.mkdir(parents=True)
            manifest = job / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            (job / "result.json").write_text(json.dumps({
                "url": "https://x.com/Builder/status/42", "status": "completed",
                "manifest": str(manifest),
            }), encoding="utf-8")
            item = x_candidate("x-42", "Acme launches an agent SDK")
            item.url = "https://x.com/i/status/42"
            item.status = "needs_human"
            factory = FakeFactory([])
            factory.rerender = MagicMock(return_value={
                "status": "completed", "publishable": True, "video": "final.mp4",
                "manifest": str(manifest),
            })
            service = ResourceDiscoveryService(
                workspace, factory=factory, clock=lambda: NOW, sleeper=lambda _: None,
            )

            result = service._adopt(
                item, ResourceDiscoveryConfig(retry_backoff_seconds=[0]), "auto", None,
            )

            self.assertEqual(result["status"], "generated")
            self.assertEqual(factory.generate_calls, [])
            factory.rerender.assert_called_once_with(manifest)
            self.assertEqual(result["attempts"][0]["mode"], "deterministic_rerender")

    def test_scheduled_blocked_candidate_uses_cost_cooldown_then_needs_human(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            item = x_candidate("x-cost", "Acme launches an agent SDK")
            adapter = StaticAdapter([item])
            factory = FakeFactory([RuntimeError("model failed")] * 6)
            current = [NOW]
            service = ResourceDiscoveryService(
                workspace, adapters={DiscoveryChannel.X: adapter}, factory=factory,
                clock=lambda: current[0], sleeper=lambda _: None,
            )
            config = ResourceDiscoveryConfig(
                retry_backoff_seconds=[0, 0, 0], blocked_retry_delay_hours=6,
                max_blocked_retry_runs=2,
            )
            for channel in DiscoveryChannel:
                config.channels[channel].enabled = channel == DiscoveryChannel.X

            first = service.run(config)
            current[0] = NOW + timedelta(hours=3)
            cooldown = service.run(config)
            current[0] = NOW + timedelta(hours=7)
            exhausted = service.run(config)

            self.assertEqual(first.channels["x"].status, "blocked")
            self.assertEqual(cooldown.channels["x"].status, "blocked_retry_wait")
            self.assertEqual(exhausted.channels["x"].status, "needs_human")
            self.assertEqual(len(factory.generate_calls), 6)
            state = workspace.load_discovery_state()
            self.assertNotIn("blocked_candidate", state["channels"]["x"])
            self.assertEqual(state["needs_human_candidates"][-1]["candidate_id"], "x-cost")

    def test_adoption_retries_when_final_video_checks_fail(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            item = x_candidate("x-1", "Acme launches an agent SDK")
            adapter = StaticAdapter([item])
            failed_result = {
                "status": "completed", "publishable": False, "video": "final.mp4",
                "checks": [{"name": "manifest", "passed": True, "detail": "ok"}],
                "video_checks": [{"name": "resolution", "passed": False, "detail": "1920x1080"}],
            }
            factory = FakeFactory([failed_result, failed_result, failed_result])
            service = ResourceDiscoveryService(
                workspace, adapters={DiscoveryChannel.X: adapter}, factory=factory,
                clock=lambda: NOW, sleeper=lambda _: None,
            )
            config = ResourceDiscoveryConfig()
            for channel in DiscoveryChannel:
                config.channels[channel].enabled = channel == DiscoveryChannel.X

            result = service.run(config)

            adoption = result.channels["x"].adoption
            self.assertEqual(adoption["status"], "blocked")
            self.assertEqual(len(adoption["attempts"]), 3)

    def test_youtube_adoption_reuses_complete_assets_from_failed_job(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            old_job = workspace.root / "jobs" / "old-youtube-attempt"
            old_job.mkdir(parents=True)
            source_url = "https://www.youtube.com/watch?v=tech123"
            (old_job / "result.json").write_text(json.dumps({
                "url": source_url, "status": "failed",
            }), encoding="utf-8")
            media = old_job / "tech123.mkv"
            subtitles = old_job / "tech123.en.json3"
            translation_plan = old_job / "translation-plan.json"
            media.write_bytes(b"video")
            subtitles.write_text("{}", encoding="utf-8")
            translation_plan.write_text("{}", encoding="utf-8")
            item = DiscoveryCandidate(
                id="youtube-tech123", channel=DiscoveryChannel.YOUTUBE,
                url=source_url, title="Agent SDK architecture tutorial",
                author="Builder", publisher="Builder", published_at=NOW.isoformat(),
                summary="technical tutorial", body_text="technical tutorial",
                stable_id="youtube:tech123", discovered_at=NOW.isoformat(), eligible=True,
            )
            factory = FakeFactory([{
                "status": "completed", "publishable": True,
                "collection_manifest": "collection.json",
            }])
            service = ResourceDiscoveryService(
                workspace, factory=factory, clock=lambda: NOW, sleeper=lambda _: None,
            )
            config = ResourceDiscoveryConfig(retry_backoff_seconds=[0])

            result = service._adopt(item, config, "deepseek", None)

            self.assertEqual(result["status"], "generated")
            options = factory.generate_calls[0][1]
            self.assertEqual(options.youtube_media, str(media))
            self.assertEqual(options.youtube_subtitles, str(subtitles))
            self.assertEqual(options.youtube_translation_plan, str(translation_plan))

    def test_youtube_audio_failure_is_repaired_and_revalidated_in_same_attempt(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            collection_path = workspace.root / "job-collection.json"
            collection_path.write_text("{}", encoding="utf-8")
            item = DiscoveryCandidate(
                id="youtube-audio", channel=DiscoveryChannel.YOUTUBE,
                url="https://www.youtube.com/watch?v=audio", title="Agent engineering talk",
                eligible=True, discovered_at=NOW.isoformat(),
            )
            factory = FakeFactory([{
                "status": "completed", "publishable": False,
                "collection_manifest": str(collection_path),
                "checks": [{
                    "name": "render:item:wechat_vertical:audible_audio",
                    "passed": False, "detail": "silent",
                }],
            }])
            collection = MagicMock()
            collection.id = "collection-audio"
            collection.to_dict.return_value = {"id": "collection-audio"}
            renderer = MagicMock()
            renderer.repair_silent_audio.return_value = ["renders/fixed.mp4"]
            service = ResourceDiscoveryService(
                workspace, factory=factory, clock=lambda: NOW, sleeper=lambda _: None,
            )
            with patch("video_factory.discovery.load_collection_manifest", return_value=collection), patch(
                "video_factory.discovery.YouTubeCollectionRenderer", return_value=renderer,
            ), patch(
                "video_factory.discovery.validate_collection",
                return_value=[CheckResult("audio", True, "audible")],
            ):
                result = service._adopt(
                    item, ResourceDiscoveryConfig(retry_backoff_seconds=[0]), "deepseek", None,
                )

            self.assertEqual(result["status"], "generated")
            self.assertEqual(len(result["attempts"]), 1)
            repair = result["attempts"][0]["result"]["automatic_repairs"][0]
            self.assertEqual(repair["kind"], "silent_or_truncated_audio")
            renderer.repair_silent_audio.assert_called_once_with(collection)

    def test_three_failures_block_channel_until_skip(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            item = x_candidate("x-1", "Acme launches an agent SDK")
            service = ResourceDiscoveryService(
                workspace, adapters={DiscoveryChannel.X: StaticAdapter([item])},
                factory=FakeFactory([RuntimeError("fail")] * 3), clock=lambda: NOW,
                sleeper=lambda _: None,
            )
            config = ResourceDiscoveryConfig()
            for channel in DiscoveryChannel:
                config.channels[channel].enabled = channel == DiscoveryChannel.X

            result = service.run(config)
            skipped = service.skip("x-1", "source cannot be rendered")
            state = workspace.load_discovery_state()

            self.assertEqual(result.channels["x"].status, "blocked")
            self.assertEqual(skipped["status"], "skipped")
            self.assertNotIn("blocked_candidate", state["channels"]["x"])

    def test_forced_blocked_retry_skips_a_redundant_channel_search(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            item = x_candidate("x-1", "Acme launches an agent SDK")
            adapter = StaticAdapter([item])
            factory = FakeFactory([
                RuntimeError("fail one"), RuntimeError("fail two"), RuntimeError("fail three"),
                {"status": "completed", "publishable": True, "video": "final.mp4"},
            ])
            service = ResourceDiscoveryService(
                workspace, adapters={DiscoveryChannel.X: adapter}, factory=factory,
                clock=lambda: NOW, sleeper=lambda _: None,
            )
            config = ResourceDiscoveryConfig(retry_backoff_seconds=[0, 0, 0])
            for channel in DiscoveryChannel:
                config.channels[channel].enabled = channel == DiscoveryChannel.X

            first = service.run(config, scheduled=False)
            second = service.run(config, scheduled=False)

            self.assertEqual(first.channels["x"].status, "blocked")
            self.assertEqual(second.channels["x"].status, "generated")
            self.assertEqual(adapter.calls, 1)
            self.assertEqual(len(factory.generate_calls), 4)


if __name__ == "__main__":
    unittest.main()
