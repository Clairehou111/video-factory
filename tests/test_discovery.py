from __future__ import annotations

import unittest
import json
import re
import subprocess
import threading
import time
from unittest.mock import MagicMock, patch
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from video_factory.discovery import (
    AdoptionPolicy, ChannelConfig, DiscoveryCandidate, DiscoveryChannel, ResourceDiscoveryConfig,
    GitHubDiscoveryAdapter, OpenRouterDiscountDiscoveryAdapter, RSSDiscoveryAdapter,
    ResourceDiscoveryService,
    assign_event_clusters, atomize_robotics_roundup, evaluate_adoption_candidate,
    evaluate_candidate, extract_source_video_url, select_adoption_candidates, select_parallel_candidates,
    XDiscoveryAdapter, _roundup_primary_source,
)
from video_factory.openrouter import DISCOUNTS_READER, ENDPOINTS_API, MODELS_API, parse_discounted_models
from video_factory.models import ContentType, TopicType
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
    def test_adoption_policy_is_separate_from_source_quality(self) -> None:
        body = (
            "Acme AI industry weekly digest contains background, history, and several references. "
            "The article summarizes general market commentary without announcing a material change. "
        ) * 8
        item = DiscoveryCandidate(
            id="routine-digest", channel=DiscoveryChannel.X,
            url="https://x.com/acme/status/100", title="Acme GPT weekly digest",
            author="Acme", publisher="X", published_at=NOW.isoformat(),
            summary=body, body_text=body,
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.X, {}), NOW)
        evaluate_adoption_candidate(item, AdoptionPolicy())

        decision = item.metadata["adoption_decision"]
        self.assertGreaterEqual(item.score, 90)
        self.assertFalse(decision["passed"])
        self.assertIn("weak_audience_consequence", decision["reasons"])
        self.assertIn("llm_intelligence", decision["category_flags"])
        self.assertEqual(decision["breakdown"]["llm_priority_bonus"], 0)
        self.assertFalse(decision["llm_priority_qualified"])

    def test_meaningful_llm_story_receives_configured_priority_bonus(self) -> None:
        body = (
            "OpenAI released a new GPT reasoning model today through its production API. "
            "The model cuts developer inference cost by 70% and returns benchmark results faster. "
        ) * 8
        item = DiscoveryCandidate(
            id="llm-release", channel=DiscoveryChannel.X,
            url="https://x.com/openai/status/101", title="OpenAI cuts GPT inference cost by 70%",
            author="OpenAI", publisher="X", published_at=NOW.isoformat(),
            summary=body, body_text=body,
        )
        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.X, {}), NOW)
        evaluate_adoption_candidate(item, AdoptionPolicy(llm_priority_bonus=0))
        without_bonus = item.metadata["adoption_decision"]["score"]
        evaluate_adoption_candidate(item, AdoptionPolicy(llm_priority_bonus=6))
        decision = item.metadata["adoption_decision"]

        self.assertIn("llm_intelligence", decision["category_flags"])
        self.assertEqual(decision["score"], min(100, without_bonus + 6))

    def test_physical_and_hardware_candidates_use_the_higher_threshold(self) -> None:
        physical = DiscoveryCandidate(
            id="robot", channel=DiscoveryChannel.ROBOTICS, url="https://example.com/robot",
            title="Robot begins warehouse deployment", publisher="Robot Co",
            published_at=NOW.isoformat(), eligible=True, score=96,
            summary="A robot deployed in a warehouse today and completed 100 production tasks for users.",
            body_text="A robot deployed in a warehouse today and completed 100 production tasks for users.",
            metadata={"real_world_impact_tier": 2, "image_count": 2},
        )
        hardware = DiscoveryCandidate(
            id="gpu", channel=DiscoveryChannel.NEWS, url="https://example.com/gpu",
            title="Nvidia ships a new AI inference GPU", publisher="Example",
            published_at=NOW.isoformat(), eligible=True, score=96,
            summary="Nvidia ships an AI inference GPU server today with 40% more model throughput.",
            body_text="Nvidia ships an AI inference GPU server today with 40% more model throughput.",
            metadata={"image_count": 2},
        )

        for item in (physical, hardware):
            evaluate_adoption_candidate(item, AdoptionPolicy())
            self.assertEqual(item.metadata["adoption_decision"]["threshold"], 82)

    def test_ordinary_consumer_hardware_specs_and_release_schedule_are_rejected(self) -> None:
        items = [
            DiscoveryCandidate(
                id="mac-memory", channel=DiscoveryChannel.NEWS,
                url="https://example.com/mac-memory",
                title="New Mac has 32GB less memory than M5 Max", publisher="Example",
                published_at=NOW.isoformat(), eligible=True, score=98,
                summary="The laptop comes with less RAM and three colors.",
                body_text="The laptop comes with less RAM and three colors.",
            ),
            DiscoveryCandidate(
                id="apple-schedule", channel=DiscoveryChannel.NEWS,
                url="https://example.com/apple-september",
                title="Apple plans several hardware launches in September", publisher="Example",
                published_at=NOW.isoformat(), eligible=True, score=98,
                summary="Apple plans a phone, laptop, and display product refresh in September.",
                body_text="Apple plans a phone, laptop, and display product refresh in September.",
            ),
        ]

        for item in items:
            evaluate_adoption_candidate(item, AdoptionPolicy())
            decision = item.metadata["adoption_decision"]
            self.assertFalse(decision["passed"])
            self.assertIn("hardware_not_ai_related", decision["reasons"])

    def test_ai_label_alone_does_not_admit_routine_hardware_launch(self) -> None:
        item = DiscoveryCandidate(
            id="ai-pc-colors", channel=DiscoveryChannel.NEWS,
            url="https://example.com/ai-pc-colors",
            title="AI laptop arrives in silver, cherry, and blue", publisher="Example",
            published_at=NOW.isoformat(), eligible=True, score=98,
            summary="The new AI laptop launches next month in three colors with a product roadmap.",
            body_text="The new AI laptop launches next month in three colors with a product roadmap.",
        )

        evaluate_adoption_candidate(item, AdoptionPolicy())

        decision = item.metadata["adoption_decision"]
        self.assertFalse(decision["passed"])
        self.assertIn("routine_ai_hardware_product_news", decision["reasons"])

    def test_youtube_competes_in_an_independent_pool(self) -> None:
        general = x_candidate("x-101", "OpenAI launches a practical agent SDK")
        evaluate_candidate(general, ChannelConfig.from_dict(DiscoveryChannel.X, {}), NOW)
        youtube = DiscoveryCandidate(
            id="youtube-101", channel=DiscoveryChannel.YOUTUBE,
            url="https://youtube.com/watch?v=101", title=general.title,
            publisher="Engineering Channel", published_at=NOW.isoformat(),
            eligible=True, score=95, metadata={"youtube_score": 88},
        )
        general.event_key = youtube.event_key = "event:same-subject"

        selected = select_adoption_candidates([general, youtube], AdoptionPolicy())

        self.assertEqual({item.id for item in selected}, {general.id, youtube.id})
        self.assertEqual(
            {item.metadata["adoption_decision"]["pool"] for item in selected},
            {"general", "youtube"},
        )

    def test_youtube_history_does_not_dedupe_against_general_pool(self) -> None:
        youtube = DiscoveryCandidate(
            id="youtube-102", channel=DiscoveryChannel.YOUTUBE,
            url="https://youtube.com/watch?v=102", title="OpenAI agent SDK architecture",
            publisher="Engineering Channel", published_at=NOW.isoformat(),
        )
        state = {"generated_events": [{
            "candidate_id": "news-102", "channel": "news", "pool": "general",
            "url": "https://example.com/openai-agent-sdk",
            "title": youtube.title, "published_at": NOW.isoformat(),
            "generated_at": NOW.isoformat(),
        }]}

        duplicate = ResourceDiscoveryService._historical_duplicate(
            youtube, state, ResourceDiscoveryConfig(), NOW,
        )

        self.assertFalse(duplicate)

    def test_published_youtube_source_is_never_selected_for_a_second_angle(self) -> None:
        youtube = DiscoveryCandidate(
            id="youtube-second-angle", channel=DiscoveryChannel.YOUTUBE,
            url="https://youtube.com/watch?v=1u5dMAKl_ks",
            title="A different angle from the same interview",
            publisher="All-In Podcast", published_at=NOW.isoformat(),
        )
        state = {
            "generated_events": [],
            "published_youtube_source_ids": ["1u5dMAKl_ks"],
        }

        duplicate = ResourceDiscoveryService._historical_duplicate(
            youtube, state, ResourceDiscoveryConfig(), NOW,
        )

        self.assertTrue(duplicate)

    def test_remote_collection_publish_builds_durable_youtube_source_ledger(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Workspace(Path(directory))
            workspace.initialize()
            manifest_id = "youtube-published-source"
            (workspace.collections_dir / f"{manifest_id}.json").write_text(json.dumps({
                "id": manifest_id,
                "candidate_id": "youtube-source",
                "source_url": "https://youtube.com/watch?v=published123",
                "source_video_id": "published123",
                "source_title": "Published interview",
                "source_channel": "Example",
                "collection_title": "Published interview",
                "transcript": [], "terminology": [], "items": [],
            }), encoding="utf-8")
            batch_dir = workspace.publish_dir / "published-batch"
            batch_dir.mkdir()
            (batch_dir / "batch.json").write_text(json.dumps({
                "id": "published-batch", "batch_type": "collection",
                "manifest_id": manifest_id, "state": "approved",
                "items": [{"state": "submitted"}],
            }), encoding="utf-8")

            service = ResourceDiscoveryService(workspace)

            self.assertEqual(service._published_youtube_source_ids(), {"published123"})

    def test_all_passing_unique_candidates_from_one_channel_are_selected(self) -> None:
        first = x_candidate("x-201", "Acme launches an agent SDK")
        second = x_candidate("x-202", "Mistral releases a model API for developers")
        for item in (first, second):
            evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.X, {}), NOW)
        assign_event_clusters([first, second])

        selected = select_adoption_candidates([first, second], AdoptionPolicy())

        self.assertEqual({item.id for item in selected}, {first.id, second.id})

    def test_same_pool_duplicate_keeps_higher_adoption_score(self) -> None:
        stronger = x_candidate("x-211", "OpenAI cuts GPT inference cost by 70%")
        weaker = x_candidate("x-212", "OpenAI releases an agent SDK")
        stronger.summary = stronger.body_text = (
            "OpenAI released GPT through its API today and cut developer inference cost by 70%. "
            "The production model returns benchmark results faster than the previous version. "
        ) * 6
        for item in (stronger, weaker):
            evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.X, {}), NOW)
            item.event_key = "event:duplicate"

        selected = select_adoption_candidates([weaker, stronger], AdoptionPolicy())

        self.assertEqual([item.id for item in selected], [stronger.id])

    def test_service_generates_every_passing_candidate_from_same_channel(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            items = [
                x_candidate("x-301", "Acme launches an agent SDK"),
                x_candidate("x-302", "Mistral releases a model API for developers"),
            ]
            factory = FakeFactory([
                {"status": "completed", "publishable": True, "video": "one.mp4"},
                {"status": "completed", "publishable": True, "video": "two.mp4"},
            ])
            service = ResourceDiscoveryService(
                workspace, adapters={DiscoveryChannel.X: StaticAdapter(items)},
                factory=factory, clock=lambda: NOW, sleeper=lambda _: None,
            )
            config = ResourceDiscoveryConfig(retry_backoff_seconds=[0])
            for channel in DiscoveryChannel:
                config.channels[channel].enabled = channel == DiscoveryChannel.X

            result = service.run(config, scheduled=False)

            entry = result.channels["x"]
            self.assertEqual(entry.status, "generated")
            self.assertEqual(len(entry.selections), 2)
            self.assertEqual(len(entry.adoptions), 2)
            self.assertEqual(len(factory.generate_calls), 2)

    def test_higher_adoption_score_keeps_the_bounded_retry_slot(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            service = ResourceDiscoveryService(workspace, clock=lambda: NOW)
            stronger = x_candidate("x-401", "OpenAI cuts GPT inference cost by 70%")
            weaker = x_candidate("x-402", "Acme launches an agent SDK")
            stronger.metadata["adoption_decision"] = {"score": 92}
            weaker.metadata["adoption_decision"] = {"score": 78}
            state = workspace.load_discovery_state()

            self.assertEqual(
                service._record_blocked_candidate(
                    state, stronger, ResourceDiscoveryConfig(), NOW,
                ),
                "blocked",
            )
            self.assertEqual(
                service._record_blocked_candidate(
                    state, weaker, ResourceDiscoveryConfig(), NOW,
                ),
                "needs_human",
            )

            channel_state = state["channels"]["x"]
            self.assertEqual(channel_state["blocked_candidate"]["id"], stronger.id)
            self.assertEqual(state["needs_human_candidates"][0]["candidate_id"], weaker.id)

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

    def test_x_auth_failure_is_recorded_as_high_severity_source_alert(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            adapter = MagicMock()
            adapter.search.side_effect = RuntimeError(
                "AUTH_REQUIRED: Not logged into x.com (no ct0 cookie)"
            )
            channels = {
                channel: ChannelConfig(enabled=channel == DiscoveryChannel.X)
                for channel in DiscoveryChannel
            }
            service = ResourceDiscoveryService(
                workspace, adapters={DiscoveryChannel.X: adapter},
                factory=FakeFactory([]), clock=lambda: NOW,
            )

            run = service.run(
                ResourceDiscoveryConfig(channels=channels), scheduled=False,
                channels=[DiscoveryChannel.X],
            )

            self.assertEqual(run.channels["x"].status, "search_failed")
            problems = json.loads(
                (Path(temp) / "automation" / "self-audit" / "problems.json").read_text()
            )["problems"]
            problem = next(iter(problems.values()))
            self.assertEqual(problem["category"], "source_auth_unavailable")
            self.assertEqual(problem["severity"], "high")

    def test_default_channel_cadences_are_independent(self) -> None:
        config = ResourceDiscoveryConfig()

        self.assertEqual(config.channels[DiscoveryChannel.X].cadence_hours, 2)
        self.assertEqual(config.channels[DiscoveryChannel.PROJECTS].cadence_hours, 4)
        self.assertEqual(config.channels[DiscoveryChannel.ROBOTICS].cadence_hours, 4)
        self.assertEqual(config.channels[DiscoveryChannel.NEWS].cadence_hours, 2)
        self.assertEqual(config.channels[DiscoveryChannel.NEWS_ZH].cadence_hours, 2)
        self.assertEqual(config.channels[DiscoveryChannel.OFFICIAL].cadence_hours, 2)
        self.assertEqual(config.channels[DiscoveryChannel.OFFICIAL_ZH].cadence_hours, 2)
        self.assertEqual(config.channels[DiscoveryChannel.PAPER].cadence_hours, 24)
        self.assertEqual(config.channels[DiscoveryChannel.GITHUB].cadence_hours, 48)
        self.assertEqual(config.channels[DiscoveryChannel.YOUTUBE].cadence_hours, 2)
        self.assertEqual(config.channels[DiscoveryChannel.OPENROUTER].cadence_hours, 2)

    def test_adoption_policy_can_be_configured_independently(self) -> None:
        config = ResourceDiscoveryConfig.from_dict({
            "adoption_policy": {
                "general_minimum_score": 77,
                "high_standard_minimum_score": 86,
                "youtube_minimum_score": 83,
                "minimum_audience_consequence": 14,
                "llm_priority_bonus": 5,
            },
        })

        self.assertEqual(config.adoption_policy.general_minimum_score, 77)
        self.assertEqual(config.adoption_policy.high_standard_minimum_score, 86)
        self.assertEqual(config.adoption_policy.youtube_minimum_score, 83)

    def test_projects_and_robotics_sources_are_in_defaults(self) -> None:
        config = ResourceDiscoveryConfig()
        projects = config.channels[DiscoveryChannel.PROJECTS]
        robotics = config.channels[DiscoveryChannel.ROBOTICS]
        autonomous = config.channels[DiscoveryChannel.AUTONOMOUS_DRIVING]

        for domain in ("reuters.com", "techcrunch.com", "sifted.eu"):
            self.assertIn(domain, projects.seed_domains)
        for feed in ("https://techcrunch.com/category/startups/feed/", "https://sifted.eu/feed"):
            self.assertIn(feed, projects.feeds)
        for domain in (
            "therobotreport.com", "spectrum.ieee.org", "waymo.com", "bostondynamics.com",
            "figure.ai", "unitree.com", "sunday.ai", "zoox.com", "nuro.ai",
        ):
            self.assertIn(domain, robotics.seed_domains)
        self.assertEqual(robotics.lookback_hours, 24 * 7)
        self.assertTrue(any("physical AI" in query for query in robotics.queries))
        self.assertTrue(any("first ever" in query for query in robotics.queries))
        self.assertTrue(autonomous.enabled)
        self.assertEqual(autonomous.lookback_hours, 24 * 7)
        self.assertTrue(any("autonomous driving" in query for query in autonomous.queries))
        self.assertIn("waymo.com", autonomous.seed_domains)
        self.assertTrue(any("autonomous driving" in query for query in config.channels[DiscoveryChannel.X].queries))
        self.assertTrue(any("embodied AI" in query for query in config.channels[DiscoveryChannel.PAPER].queries))
        self.assertIn("typesafe.ai", config.channels[DiscoveryChannel.OFFICIAL].seed_domains)

    def test_news_defaults_cover_reported_ai_hardware_procurement(self) -> None:
        news = ResourceDiscoveryConfig().channels[DiscoveryChannel.NEWS]

        self.assertIn("theinformation.com", news.seed_domains)
        self.assertNotIn("macrumors.com", news.seed_domains)
        self.assertFalse(any("macrumors.com" in feed for feed in news.feeds))
        self.assertTrue(any("Mac OR GPU OR chip OR compute" in query for query in news.queries))

    def test_disabled_news_source_is_removed_from_explicit_config(self) -> None:
        news = ChannelConfig.from_dict(DiscoveryChannel.NEWS, {
            "seed_domains": ["reuters.com", "macrumors.com"],
            "feeds": ["https://feeds.macrumors.com/MacRumors-All"],
        })

        self.assertEqual(news.seed_domains, ["reuters.com"])
        self.assertEqual(news.feeds, [])

    def test_rss_trace_preserves_source_query_and_funnel_counts(self) -> None:
        payload = (
            "<rss><channel><item><title>OpenAI buys Macs for agent training</title>"
            "<link>https://www.reuters.com/technology/openai-macs</link>"
            "<pubDate>Mon, 31 Aug 2026 10:00:00 GMT</pubDate>"
            "<description>OpenAI bought Macs while Anthropic rents them.</description>"
            "</item></channel></rss>"
        ).encode()
        adapter = RSSDiscoveryAdapter(
            DiscoveryChannel.NEWS,
            fetcher=lambda url: (
                "OpenAI bought tens of thousands of Macs for reinforcement learning and computer-use agents. "
                "Anthropic rents Mac mini capacity through AWS for similar work. " * 8,
                "https://www.reuters.com/technology/openai-macs",
            ),
        )
        adapter._download = lambda url: payload
        config = ChannelConfig.from_dict(DiscoveryChannel.NEWS, {
            "queries": [], "feeds": ["https://feeds.reuters.com/technology"],
            "seed_domains": ["reuters.com"], "probe_limit": 2,
        })

        found = adapter.search(config, NOW)

        self.assertEqual(len(found), 1)
        self.assertEqual(adapter.last_trace["sources"][0]["kind"], "curated_feed")
        self.assertEqual(adapter.last_trace["funnel"]["rows_raw"], 1)
        self.assertEqual(adapter.last_trace["funnel"]["rows_trusted"], 1)
        self.assertEqual(adapter.last_trace["funnel"]["candidates_emitted"], 1)

    def test_rss_adapter_records_repetition_across_independent_publishers(self) -> None:
        feeds = {
            "https://alpha.example/feed": (
                "https://alpha.example/jev",
                "Jev model launch draws developer demand",
            ),
            "https://beta.example/feed": (
                "https://beta.example/jev",
                "Jev model launch draws heavy developer demand",
            ),
            "https://gamma.example/feed": (
                "https://gamma.example/jev",
                "Jev model launch draws developer demand worldwide",
            ),
        }

        def download(url: str) -> bytes:
            link, title = feeds[url]
            return (
                f"<rss><channel><item><title>{title}</title><link>{link}</link>"
                f"<pubDate>Fri, 28 Aug 2026 05:59:00 GMT</pubDate>"
                f"<description>Jev is a new AI model for developers.</description>"
                f"</item></channel></rss>"
            ).encode()

        adapter = RSSDiscoveryAdapter(
            DiscoveryChannel.NEWS,
            fetcher=lambda url: (
                "Jev is a new AI model released through an API for developer automation. " * 20,
                url,
            ),
        )
        adapter._download = download
        config = ChannelConfig.from_dict(DiscoveryChannel.NEWS, {
            "queries": [], "feeds": list(feeds),
            "seed_domains": ["alpha.example", "beta.example", "gamma.example"],
            "probe_limit": 3,
        })

        found = adapter.search(config, NOW)

        self.assertEqual(len(found), 3)
        self.assertTrue(all(item.metadata["cross_source_mentions"] == 3 for item in found))
        for item in found:
            evaluate_candidate(item, config, NOW)
            self.assertIn("cross_source_repetition", item.metadata["viral_attention_signals"])

    def test_structural_discovery_funnel_gap_is_added_to_nightly_audit(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            adapter = StaticAdapter([])
            adapter.last_trace = {
                "adapter": "rss",
                "funnel": {
                    "sources_planned": 4, "sources_succeeded": 4,
                    "sources_failed": 0, "rows_raw": 18,
                    "rows_trusted": 0, "candidates_emitted": 0,
                },
            }
            channels = {
                channel: ChannelConfig(enabled=channel == DiscoveryChannel.NEWS)
                for channel in DiscoveryChannel
            }
            config = ResourceDiscoveryConfig(channels=channels)
            service = ResourceDiscoveryService(
                workspace, adapters={DiscoveryChannel.NEWS: adapter},
                factory=FakeFactory([]), clock=lambda: NOW,
            )

            run = service.run(config, scheduled=False, channels=[DiscoveryChannel.NEWS])

            self.assertEqual(run.channels["news"].trace["funnel"]["rows_raw"], 18)
            problems = json.loads(
                (Path(temp) / "automation" / "self-audit" / "problems.json").read_text()
            )["problems"]
            problem = next(iter(problems.values()))
            self.assertEqual(problem["stage"], "discovery")
            self.assertEqual(problem["category"], "coverage_gap")
            self.assertIn("reduced to zero", problem["observed"])

    def test_curated_direct_feed_keeps_its_source_provenance(self) -> None:
        payload = (
            "<rss><channel><item><title>AI startup passes one million users</title>"
            "<link>https://techcrunch.com/example-ai</link>"
            "<pubDate>Fri, 28 Aug 2026 05:59:00 GMT</pubDate>"
            "<description>A new builder product with a working demo.</description>"
            "</item></channel></rss>"
        ).encode()
        adapter = RSSDiscoveryAdapter(
            DiscoveryChannel.PROJECTS,
            fetcher=lambda url: (
                "The startup reached 1 million users and launched a working AI product. " * 30,
                "https://techcrunch.com/example-ai",
            ),
        )
        adapter._download = lambda url: payload
        config = ChannelConfig.from_dict(DiscoveryChannel.PROJECTS, {
            "queries": [], "feeds": ["https://techcrunch.com/category/startups/feed/"],
            "seed_domains": ["techcrunch.com"], "probe_limit": 1,
        })

        found = adapter.search(config, NOW)

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].channel, DiscoveryChannel.PROJECTS)
        self.assertEqual(found[0].publisher, "techcrunch.com")
        self.assertEqual(found[0].metadata["publisher_url"], "https://techcrunch.com")

    def test_project_channel_rejects_unquantified_small_launch(self) -> None:
        body = (
            "A small startup team launched a new AI product today. "
            "The founders published a demo, product page, API details, and an early-access form. "
            "The tool is available now, but the story gives no user, revenue, funding, or growth milestone. "
        ) * 8
        item = DiscoveryCandidate(
            id="small-launch", channel=DiscoveryChannel.PROJECTS,
            url="https://techcrunch.com/small-launch", title="Small AI product launches",
            publisher="TechCrunch", published_at=NOW.isoformat(), summary=body,
            body_text=body, metadata={"image_count": 2},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.PROJECTS, {}), NOW)

        self.assertFalse(item.eligible)
        self.assertIn("missing_breakout_traction", item.rejection_reasons)

    def test_project_channel_accepts_lovable_class_breakout_signal(self) -> None:
        body = (
            "The AI product became one of the fastest-growing startups after reaching 1 million users. "
            "The company reported $12 million ARR, 300% growth, and a $25 million funding round. "
            "The founding team explained the product workflow, customer adoption, launch timeline, and API. "
        ) * 8
        item = DiscoveryCandidate(
            id="breakout-startup", channel=DiscoveryChannel.PROJECTS,
            url="https://techcrunch.com/breakout-startup", title="AI startup reaches one million users",
            publisher="TechCrunch", published_at=NOW.isoformat(), summary=body,
            body_text=body, metadata={"image_count": 2},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.PROJECTS, {}), NOW)

        self.assertTrue(item.eligible)
        self.assertNotIn("missing_breakout_traction", item.rejection_reasons)

    def test_project_feed_model_launch_bypasses_startup_traction_gate(self) -> None:
        summary = (
            "Jev is a new transformer-based AI model for fast structured software decisions. "
            "TypeSafe released the public model in early access through its API today."
        )
        body = (
            summary + " Developers use it for automation workflows with lower latency and cost. "
            "The company briefly lost the ability to serve users from its API because demand was so high. "
            "A founder contrasted this software release with building a god in a data center. "
        ) * 8
        item = DiscoveryCandidate(
            id="jev-launch", channel=DiscoveryChannel.PROJECTS,
            url="https://techcrunch.com/jev-launch",
            title="A new kind of AI model is thrilling developers",
            publisher="TechCrunch", published_at=NOW.isoformat(),
            summary=summary, body_text=body, metadata={"image_count": 1},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.PROJECTS, {}), NOW)
        evaluate_adoption_candidate(item, AdoptionPolicy())

        self.assertTrue(item.eligible)
        self.assertNotIn("missing_breakout_traction", item.rejection_reasons)
        self.assertEqual(
            item.metadata["viral_attention_signals"],
            ["demand_exceeded_service_capacity"],
        )
        decision = item.metadata["adoption_decision"]
        self.assertTrue(decision["passed"])
        self.assertNotIn("hardware", decision["category_flags"])
        self.assertNotIn("routine_ai_hardware_product_news", decision["reasons"])

    def test_financing_article_cannot_borrow_ai_scope_from_publisher_chrome(self) -> None:
        title = "Y Combinator insurance tech alum Angle Health hits $2.7B valuation"
        body = f"""Title: {title}

URL Source: https://techcrunch.com/angle-health

Markdown Content:
[AI](https://techcrunch.com/category/artificial-intelligence/)
[Cloud Computing](https://techcrunch.com/tag/cloud-computing/)
[Robotics](https://techcrunch.com/category/robotics/)

# {title}

These days it is rare for a startup not focused on AI agents to raise a hefty Series C.
Angle Health announced a $200 million Series C and a $400 million tender offer
at a $2.7 billion valuation. The health insurance company expects the deal to close this month.

Angle Health helps small businesses obtain and manage level-funded health plans.
It is an AI-powered platform that integrates with payroll and HR systems.
The startup serves over 5,000 businesses and is profitable.

Topics

Biotech & Health, Fundraising, insurance tech

## Related

OpenAI launches an AI model. A physical AI robotics startup publishes a benchmark.
"""
        item = DiscoveryCandidate(
            id="angle-health-financing", channel=DiscoveryChannel.PROJECTS,
            url="https://techcrunch.com/angle-health", title=title,
            publisher="TechCrunch", published_at=NOW.isoformat(),
            summary=(
                "Angle Health has grown to 5,000 customers and become profitable "
                "by helping small businesses obtain health insurance."
            ),
            body_text=body, metadata={"image_count": 14},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.PROJECTS, {}), NOW)
        evaluate_adoption_candidate(item, AdoptionPolicy())

        self.assertFalse(item.eligible)
        self.assertIn("finance_only_without_technology_consequence", item.rejection_reasons)
        self.assertNotIn("viral_attention_signals", item.metadata)
        self.assertEqual(item.metadata["page_image_count"], 14)
        self.assertEqual(item.metadata["image_count"], 0)
        decision = item.metadata["adoption_decision"]
        self.assertFalse(decision["passed"])
        self.assertNotIn("llm_intelligence", decision["category_flags"])
        self.assertIn("source_quality_gate_failed", decision["reasons"])

    def test_github_adapter_records_rapid_integration_repository_growth(self) -> None:
        rows = [{
            "fullName": f"builder{index}/jev-tool-{index}",
            "url": f"https://github.com/builder{index}/jev-tool-{index}",
            "description": "A Jev integration for typed workflow decisions",
            "createdAt": NOW.isoformat(), "updatedAt": NOW.isoformat(),
            "pushedAt": NOW.isoformat(), "stargazersCount": index,
            "isArchived": False, "owner": {"login": f"builder{index}"},
        } for index in range(3)]

        def runner(command, **kwargs):
            if command[:3] == ["gh", "search", "repos"]:
                return subprocess.CompletedProcess(command, 0, json.dumps(rows), "")
            return subprocess.CompletedProcess(
                command, 0,
                "# Jev tool\n\nInstall with pip. Includes an API example, input, output, and workflow demo.",
                "",
            )

        found = GitHubDiscoveryAdapter(runner).search(
            ChannelConfig.from_dict(DiscoveryChannel.GITHUB, {
                "queries": ["Jev created:>{date}"], "probe_limit": 3,
            }),
            NOW,
        )

        self.assertEqual(len(found), 3)
        for item in found:
            velocity = item.metadata["related_repo_velocity"]
            self.assertEqual(velocity["subject"], "jev")
            self.assertEqual(velocity["repository_count"], 3)
            evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.GITHUB, {}), NOW)
            self.assertIn(
                "rapid_integration_repository_growth",
                item.metadata["viral_attention_signals"],
            )

    def test_non_it_business_or_aerospace_story_is_rejected(self) -> None:
        body = (
            "The organization announced a new strategy today with a large budget. "
            "Leaders discussed markets, investment, hiring, and future expansion. "
            "The report includes several concrete figures and a detailed timeline. "
        ) * 8
        item = DiscoveryCandidate(
            id="off-scope-strategy", channel=DiscoveryChannel.NEWS,
            url="https://reuters.com/off-scope-strategy",
            title="NASA outlines a new Moon and Mars strategy", publisher="Reuters",
            published_at=NOW.isoformat(), summary=body, body_text=body,
            metadata={"image_count": 2},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.NEWS, {}), NOW)

        self.assertFalse(item.eligible)
        self.assertIn("outside_it_software_ai_scope", item.rejection_reasons)

    def test_raw_html_fallback_cannot_pass_as_article_evidence(self) -> None:
        body = (
            "<!DOCTYPE html><html><head><meta name='description' content='AI lab raised $50 million'>"
            "</head><body><nav>Subscribe Markets Startups AI Software</nav>"
            "<script>window.__DATA__ = 'AI API funding users developers';</script></body></html>"
        ) * 20
        item = DiscoveryCandidate(
            id="raw-html-project", channel=DiscoveryChannel.PROJECTS,
            url="https://sifted.eu/raw-html-project", title="AI lab raises $50 million",
            publisher="Sifted", published_at=NOW.isoformat(), body_text=body,
            metadata={"image_count": 20},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.PROJECTS, {}), NOW)

        self.assertFalse(item.eligible)
        self.assertIn("unextractable_source_page", item.rejection_reasons)

    def test_news_channel_rejects_uneventful_sixth_os_beta(self) -> None:
        body = (
            "Apple released the sixth public beta of iOS 27 one week after beta five. "
            "No notable new features were found in today's developer beta. "
            "Compatible devices can download it from Software Update. "
        ) * 12
        item = DiscoveryCandidate(
            id="ios-27-beta-6", channel=DiscoveryChannel.NEWS,
            url="https://9to5mac.com/2026/08/31/apple-ios-27-public-beta-6/",
            title="Apple releases iOS 27 public beta 6", publisher="9to5Mac",
            published_at=NOW.isoformat(), summary=body, body_text=body,
            metadata={"image_count": 1},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.NEWS, {}), NOW)

        self.assertFalse(item.eligible)
        self.assertIn("routine_iteration_without_material_change", item.rejection_reasons)

    def test_disabled_source_is_rejected_even_if_runtime_config_bypasses_config_loader(self) -> None:
        feed = (
            "<rss><channel><item><title>Apple and OpenAI lawsuit</title>"
            "<link>https://forums.macrumors.com/threads/apple-openai.2488249/</link>"
            "<pubDate>Mon, 31 Aug 2026 21:00:00 GMT</pubDate>"
            "<description>Apple says a former engineer used trade secrets at OpenAI.</description>"
            "</item></channel></rss>"
        ).encode()
        body = (
            "Apple alleges that a former engineer downloaded a circuit schematic and used it at OpenAI. "
            "The filing describes LTspice simulation work, a request for expedited discovery, and a hearing date. "
        ) * 12
        adapter = RSSDiscoveryAdapter(
            DiscoveryChannel.NEWS,
            fetcher=lambda url: (body, "https://forums.macrumors.com/threads/apple-openai.2488249/"),
        )
        adapter._download = lambda url: feed
        config = ChannelConfig(
            queries=[], feeds=["https://feeds.macrumors.com/MacRumors-All"],
            seed_domains=["macrumors.com"], probe_limit=1,
        )

        found = adapter.search(config, NOW)

        self.assertEqual(found, [])
        self.assertTrue(any(
            row.get("reason") == "source_domain_disabled"
            for row in adapter.last_trace["rejections"]
        ))

    def test_news_channel_keeps_versioned_release_with_material_security_change(self) -> None:
        body = (
            "Apple released iOS 27.1 with a security update for an actively exploited zero-day. "
            "The release fixes CVE-2026-12345 and introduces a new device protection capability. "
            "Apple recommends that compatible devices install the update today. "
        ) * 12
        item = DiscoveryCandidate(
            id="ios-27-security", channel=DiscoveryChannel.NEWS,
            url="https://9to5mac.com/2026/09/01/ios-27-security-update/",
            title="Apple fixes exploited iOS 27 zero-day", publisher="9to5Mac",
            published_at=NOW.isoformat(), summary=body, body_text=body,
            metadata={"image_count": 1},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.NEWS, {}), NOW)

        self.assertTrue(item.eligible)
        self.assertNotIn("routine_iteration_without_material_change", item.rejection_reasons)

    def test_robotics_and_autonomous_driving_story_passes_audience_gate(self) -> None:
        body = (
            "A robotics startup launched a physical AI humanoid for warehouse deployment today. "
            "The robot completed 120 field-test hours and demonstrated perception, manipulation, and safe motion planning. "
            "The founding team published a product demo, pilot schedule, and autonomous-driving technology details. "
        ) * 8
        item = DiscoveryCandidate(
            id="robotics-launch", channel=DiscoveryChannel.ROBOTICS,
            url="https://www.therobotreport.com/new-humanoid-launch/",
            title="Startup launches physical AI robot after field tests",
            publisher="The Robot Report", published_at=NOW.isoformat(), summary=body,
            body_text=body, metadata={
                "image_count": 3,
                "source_video_url": "https://www.youtube.com/watch?v=ROBOT01",
            },
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.ROBOTICS, {}), NOW)

        self.assertTrue(item.eligible)
        self.assertEqual(item.topic_type.value, "company_or_team")
        self.assertEqual(item.score_breakdown["audience_value"], 20.0)
        self.assertGreater(item.score_breakdown["attention_value"], 0.0)

    def test_robotics_simulation_only_claim_cannot_pose_as_real_world_result(self) -> None:
        body = (
            "The proposed planner reports an 8 to 11 times reduction in a simulation. "
            "The results have been validated only within a simplified 2D simulation. "
            "The method has not yet been verified on physical robots in the real world. "
        ) * 12
        item = DiscoveryCandidate(
            id="robotics-simulation", channel=DiscoveryChannel.ROBOTICS,
            url="https://www.therobotreport.com/simulation-theory/",
            title="A new mathematical theory for embodied AI",
            publisher="The Robot Report", published_at=NOW.isoformat(),
            summary=body, body_text=body, metadata={"image_count": 3},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.ROBOTICS, {}), NOW)

        self.assertFalse(item.eligible)
        self.assertIn("simulation_only_without_real_world_evidence", item.rejection_reasons)

    def test_physical_technology_without_source_video_is_rejected_even_with_images(self) -> None:
        body = (
            "A robot remained online with zero hardware faults but misclassified ordinary objects. "
            "Researchers report measured failures, normal system health, and a reproducible test. "
        ) * 10
        item = DiscoveryCandidate(
            id="robot-no-video", channel=DiscoveryChannel.ROBOTICS,
            url="https://example.com/robot-study", title="Daily objects confuse a healthy robot",
            publisher="Example Robotics", published_at=NOW.isoformat(), summary=body,
            body_text=body, metadata={"image_count": 8},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.ROBOTICS, {}), NOW)
        evaluate_adoption_candidate(item, AdoptionPolicy())

        self.assertFalse(item.eligible)
        self.assertIn("missing_source_video", item.rejection_reasons)
        self.assertIn("missing_source_video", item.metadata["adoption_decision"]["reasons"])

    def test_routine_autonomous_driving_city_expansion_is_not_video_worthy(self) -> None:
        body = (
            "Waymo begins welcoming first public riders in Denver, San Diego, and Tampa, "
            "marking 14 cities where its existing fully autonomous service is available. "
            "More riders will receive access over time through the same Waymo app. "
        ) * 12
        item = DiscoveryCandidate(
            id="waymo-more-cities", channel=DiscoveryChannel.AUTONOMOUS_DRIVING,
            url="https://waymo.com/blog/more-cities",
            title="Welcoming our first riders in Denver, San Diego, and Tampa",
            publisher="Waymo", published_at=NOW.isoformat(), summary=body,
            body_text=body,
            metadata={"source_video_url": "https://youtube.com/watch?v=WaymoCities"},
        )

        evaluate_candidate(
            item, ChannelConfig.from_dict(DiscoveryChannel.AUTONOMOUS_DRIVING, {}), NOW,
        )
        evaluate_adoption_candidate(item, AdoptionPolicy())

        self.assertFalse(item.eligible)
        self.assertIn("routine_physical_geographic_expansion", item.rejection_reasons)
        self.assertIn(
            "routine_physical_geographic_expansion",
            item.metadata["adoption_decision"]["reasons"],
        )

    def test_first_international_driverless_service_is_not_treated_as_routine_expansion(self) -> None:
        body = (
            "Waymo launches its first international paid fully driverless service in Japan today. "
            "The launch removes the safety driver and opens public rides on city roads. "
            "The company published an uncut video showing customers completing real trips. "
        ) * 12
        item = DiscoveryCandidate(
            id="waymo-new-boundary", channel=DiscoveryChannel.AUTONOMOUS_DRIVING,
            url="https://waymo.com/blog/first-international-driverless-service",
            title="Waymo launches first international driverless service in Japan",
            publisher="Waymo", published_at=NOW.isoformat(), summary=body,
            body_text=body,
            metadata={"source_video_url": "https://youtube.com/watch?v=WaymoJapan"},
        )

        evaluate_candidate(
            item, ChannelConfig.from_dict(DiscoveryChannel.AUTONOMOUS_DRIVING, {}), NOW,
        )

        self.assertNotIn("routine_physical_geographic_expansion", item.rejection_reasons)

    def test_supplier_certification_alone_is_not_video_worthy(self) -> None:
        body = (
            "E-Con Systems earned IATF 16949:2016 certification for its automotive cameras. "
            "The quality-management standard covers defect prevention and manufacturing controls. "
            "Its existing camera portfolio continues to support four automotive camera types. "
        ) * 12
        item = DiscoveryCandidate(
            id="camera-certification", channel=DiscoveryChannel.AUTONOMOUS_DRIVING,
            url="https://example.com/iatf-certification",
            title="E-Con Systems earns IATF 16949:2016 certification",
            publisher="Example", published_at=NOW.isoformat(), summary=body,
            body_text=body,
            metadata={"source_video_url": "https://youtube.com/watch?v=Certificate"},
        )

        evaluate_candidate(
            item, ChannelConfig.from_dict(DiscoveryChannel.AUTONOMOUS_DRIVING, {}), NOW,
        )
        evaluate_adoption_candidate(item, AdoptionPolicy())

        self.assertFalse(item.eligible)
        self.assertIn("standalone_certification", item.rejection_reasons)
        self.assertIn("standalone_certification", item.metadata["adoption_decision"]["reasons"])

    def test_finance_only_startup_deal_without_technology_consequence_is_rejected(self) -> None:
        body = (
            "The education company sold to a rival for $206 million, 94% below its peak valuation. "
            "It still had $94.8 million cash in the bank and shareholders received stock. "
            "The deal follows pandemic growth, cost cuts, and a return to physical classes. "
        ) * 12
        item = DiscoveryCandidate(
            id="finance-only-deal", channel=DiscoveryChannel.PROJECTS,
            url="https://example.com/education-company-sale",
            title="Education startup sells for 94% below peak valuation",
            publisher="Example", published_at=NOW.isoformat(), summary=body,
            body_text=body, metadata={"image_count": 3},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.PROJECTS, {}), NOW)
        evaluate_adoption_candidate(item, AdoptionPolicy())

        self.assertFalse(item.eligible)
        self.assertIn(
            "finance_only_without_technology_consequence", item.rejection_reasons,
        )
        self.assertIn(
            "finance_only_without_technology_consequence",
            item.metadata["adoption_decision"]["reasons"],
        )

    def test_funding_story_with_concrete_ai_infrastructure_consequence_remains_eligible(self) -> None:
        body = (
            "An AI infrastructure startup raised $50 million to ship a new inference chip today. "
            "The production GPU alternative cuts model API latency by 40% for developers. "
            "Customers can deploy the open-source software stack in cloud infrastructure now. "
        ) * 12
        item = DiscoveryCandidate(
            id="ai-infrastructure-funding", channel=DiscoveryChannel.PROJECTS,
            url="https://example.com/ai-infrastructure-funding",
            title="AI infrastructure startup raises $50M to ship faster inference chips",
            publisher="Example", published_at=NOW.isoformat(), summary=body,
            body_text=body, metadata={"image_count": 3},
        )

        evaluate_candidate(item, ChannelConfig.from_dict(DiscoveryChannel.PROJECTS, {}), NOW)

        self.assertNotIn(
            "finance_only_without_technology_consequence", item.rejection_reasons,
        )

    def test_rss_video_extractor_normalizes_youtube_embeds(self) -> None:
        body = '<iframe src="https://www.youtube.com/embed/Action123?start=15"></iframe>'

        self.assertEqual(
            extract_source_video_url(body),
            "https://www.youtube.com/watch?v=Action123",
        )

    def test_robotics_roundup_routes_generation_to_subject_primary_page(self) -> None:
        body = """
        [Video 1](https://www.youtube.com/watch?v=one)
        Meet Microduck, a 25 cm robot.
        [Video 2](https://www.youtube.com/watch?v=two)
        [Microduck](https://pollen-robotics.com/microduck/)
        [Unrelated](https://generalistai.com/blog/gen-1-5)
        [Video 3](https://www.youtube.com/watch?v=three)
        """
        item = DiscoveryCandidate(
            id="microduck-roundup", channel=DiscoveryChannel.ROBOTICS,
            url="https://spectrum.ieee.org/video-friday-microduck-robot",
            title="Video Friday: Meet Microduck", publisher="IEEE Spectrum",
            published_at=NOW.isoformat(), body_text=body,
        )

        self.assertEqual(
            _roundup_primary_source(item),
            "https://pollen-robotics.com/microduck",
        )

        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            factory = FakeFactory([{
                "status": "completed", "publishable": True, "video": "microduck.mp4",
            }])
            service = ResourceDiscoveryService(workspace, factory=factory, sleeper=lambda _: None)
            item.eligible = True

            result = service._adopt(item, ResourceDiscoveryConfig(), "auto", None)

            self.assertEqual(result["status"], "generated")
            called_url, options = factory.generate_calls[0]
            self.assertEqual(called_url, "https://pollen-robotics.com/microduck")
            self.assertIn(item.url, options.linked_sources)
            self.assertIn("Preferred subject-specific primary source", options.discovery_context)
            self.assertEqual(options.discovery_published_at, item.published_at)
            self.assertEqual(options.render_profile, "radar_v2")

    def test_robotics_roundup_is_atomized_with_each_video_paired_to_its_subject(self) -> None:
        roundup_html = """
        <h2>Sunday household worker</h2>
        <p>A full-size household robot folds laundry and loads a dishwasher in a real home.</p>
        <iframe src="https://www.youtube.com/embed/SUNDAY01"></iframe>
        <a href="https://www.sunday.ai/home-worker">Sunday home worker</a>
        <h2>Microduck research robot</h2>
        <iframe src="https://www.youtube.com/embed/MICROD02"></iframe>
        <p>A 25 cm biped demonstrates open-source reinforcement learning on a desk.</p>
        <a href="https://pollen-robotics.com/microduck/">Pollen Microduck</a>
        <h2>Pharmacy worker</h2>
        <a href="https://spectrum.ieee.org/tag/robotics">Robotics tag</a>
        <p>TRON 2 × Wuji Hand 2 handles TCM pharmacy work: picking, weighing, grinding and
        packaging. An omnidirectional base and dual-arm force control perform the task.</p>
        <iframe src="https://youtu.be/PHARMA03"></iframe>
        <a href="https://example-robotics.com/pharmacy-worker">Pharmacy worker</a>
        """
        parent = DiscoveryCandidate(
            id="video-friday", channel=DiscoveryChannel.ROBOTICS,
            url="https://spectrum.ieee.org/video-friday-robots", title="Video Friday: Robots at Work",
            publisher="IEEE Spectrum", published_at=NOW.isoformat(), summary=roundup_html,
            body_text=roundup_html, stable_id="web:video-friday", discovered_at=NOW.isoformat(),
        )

        children = atomize_robotics_roundup(parent)

        self.assertEqual(len(children), 3)
        by_url = {child.url: child for child in children}
        expected = {
            "https://www.sunday.ai/home-worker": "https://www.youtube.com/watch?v=SUNDAY01",
            "https://pollen-robotics.com/microduck": "https://www.youtube.com/watch?v=MICROD02",
            "https://example-robotics.com/pharmacy-worker": "https://www.youtube.com/watch?v=PHARMA03",
        }
        self.assertEqual(set(by_url), set(expected))
        for primary_url, video_url in expected.items():
            child = by_url[primary_url]
            self.assertEqual(child.metadata["source_video_url"], video_url)
            self.assertEqual(child.metadata["roundup_parent_url"], parent.url)
            self.assertTrue(child.metadata["atomized_roundup_event"])
            self.assertIn(parent.url, child.metadata["linked_sources"])
        self.assertIn(
            "folds laundry and loads a dishwasher",
            by_url["https://www.sunday.ai/home-worker"].summary,
        )
        config = ChannelConfig.from_dict(DiscoveryChannel.ROBOTICS, {"minimum_score": 0})
        for child in children:
            evaluate_candidate(child, config, NOW)
            self.assertNotIn("missing_visual_path", child.rejection_reasons)
        self.assertEqual(
            by_url["https://www.sunday.ai/home-worker"].metadata["real_world_impact_tier"], 2,
        )
        self.assertEqual(
            by_url["https://pollen-robotics.com/microduck"].metadata["real_world_impact_tier"], 1,
        )
        self.assertEqual(
            by_url["https://example-robotics.com/pharmacy-worker"].metadata["real_world_impact_tier"], 2,
        )
        self.assertTrue(by_url["https://www.sunday.ai/home-worker"].eligible)
        self.assertTrue(by_url["https://pollen-robotics.com/microduck"].eligible)
        self.assertTrue(by_url["https://example-robotics.com/pharmacy-worker"].eligible)

    def test_robot_social_study_is_not_mistaken_for_a_useful_physical_task(self) -> None:
        config = ChannelConfig.from_dict(DiscoveryChannel.ROBOTICS, {"minimum_score": 0})
        study_text = (
            "Fifty people talked with the commercial humanoid robot Pepper while researchers "
            "measured brain activity. People move and talk normally, and expressive errors reduce "
            "trust. Robots are becoming more autonomous and may later enter homes and hospitals. "
            "The paper reports participant measurements, hormone readings, and study results. "
        ) * 8
        study = DiscoveryCandidate(
            id="pepper-study", channel=DiscoveryChannel.ROBOTICS,
            url="https://robohub.org/pepper-brain-study", title="Pepper errors reduce trust",
            publisher="Robohub", published_at=NOW.isoformat(), summary=study_text,
            body_text=study_text, stable_id="web:pepper-study", metadata={"image_count": 2},
        )

        evaluate_candidate(study, config, NOW)

        self.assertEqual(study.metadata["real_world_impact_tier"], 1)

    def test_useful_worker_robot_ranks_above_eligible_toy_demo(self) -> None:
        config = ChannelConfig.from_dict(DiscoveryChannel.ROBOTICS, {"minimum_score": 0})
        worker_text = (
            "A physical AI household robot folds shirts, loads dishes, and cleans a real home. "
            "The full-size robot completed the useful task without teleoperation during a public demo. "
            "Engineers published measured manipulation results, safety details, and an uncut video. "
        ) * 8
        toy_text = (
            "A 25 cm physical robot prototype walks on a desk in a research demonstration. "
            "The toy-sized biped uses open-source reinforcement learning, picks up tiny objects, "
            "and ships before Christmas. "
            "Engineers published a video, hardware measurements, and source code for the prototype. "
        ) * 8
        worker = DiscoveryCandidate(
            id="household-worker", channel=DiscoveryChannel.ROBOTICS,
            url="https://www.sunday.ai/household-worker", title="Robot folds laundry in a real home",
            publisher="Sunday Robotics", published_at=NOW.isoformat(), summary=worker_text,
            body_text=worker_text, stable_id="web:household-worker", metadata={
                "image_count": 3,
                "source_video_url": "https://www.youtube.com/watch?v=WORKER01",
            },
        )
        toy = DiscoveryCandidate(
            id="desk-toy", channel=DiscoveryChannel.ROBOTICS,
            url="https://pollen-robotics.com/microduck", title="Tiny biped walks across a desk",
            publisher="Pollen Robotics", published_at=NOW.isoformat(), summary=toy_text,
            body_text=toy_text, stable_id="web:desk-toy", metadata={
                "image_count": 4,
                "source_video_url": "https://www.youtube.com/watch?v=TOY0001",
            },
        )
        evaluate_candidate(worker, config, NOW)
        evaluate_candidate(toy, config, NOW)
        assign_event_clusters([worker, toy])

        selected = select_parallel_candidates({DiscoveryChannel.ROBOTICS: [toy, worker]})

        self.assertTrue(worker.eligible)
        self.assertTrue(toy.eligible)
        self.assertEqual(worker.metadata["real_world_impact_tier"], 2)
        self.assertEqual(toy.metadata["real_world_impact_tier"], 1)
        self.assertIs(selected[DiscoveryChannel.ROBOTICS], worker)

    def test_robotics_and_autonomous_candidates_are_rejected_from_wrong_channel(self) -> None:
        robot_text = (
            "A physical AI household robot folds laundry, loads a dishwasher, and cleans a real home. "
            "The robot completed three useful manipulation tasks during an uncut public demonstration. "
            "The team published measurements, safety results, availability, and field-test details. "
        ) * 8
        driving_text = (
            "A fully autonomous robotaxi carried paid riders on public roads without a safety driver. "
            "The self-driving vehicle completed 12,000 rides and published highway safety results. "
            "The operator released an uncut field-test video, deployment map, and service details. "
        ) * 8
        robot_in_driving = DiscoveryCandidate(
            id="robot-in-driving", channel=DiscoveryChannel.AUTONOMOUS_DRIVING,
            url="https://example.com/household-robot", title="Robot folds laundry at home",
            publisher="Example", published_at=NOW.isoformat(), summary=robot_text,
            body_text=robot_text, metadata={"image_count": 3},
        )
        driving_in_robotics = DiscoveryCandidate(
            id="driving-in-robotics", channel=DiscoveryChannel.ROBOTICS,
            url="https://example.com/robotaxi", title="Robotaxi launches without safety driver",
            publisher="Example", published_at=NOW.isoformat(), summary=driving_text,
            body_text=driving_text, metadata={"image_count": 3},
        )

        evaluate_candidate(
            robot_in_driving, ChannelConfig.from_dict(DiscoveryChannel.AUTONOMOUS_DRIVING, {}), NOW,
        )
        evaluate_candidate(
            driving_in_robotics, ChannelConfig.from_dict(DiscoveryChannel.ROBOTICS, {}), NOW,
        )

        self.assertIn("belongs_to_robotics_bucket", robot_in_driving.rejection_reasons)
        self.assertIn("belongs_to_autonomous_driving_bucket", driving_in_robotics.rejection_reasons)
        self.assertFalse(robot_in_driving.eligible)
        self.assertFalse(driving_in_robotics.eligible)

    def test_robotics_and_autonomous_driving_each_generate_in_one_service_run(self) -> None:
        robot_text = (
            "A physical AI household robot folds laundry, loads dishes, and cleans a real home. "
            "The robot completed three useful manipulation tasks during an uncut public demonstration. "
            "The team published measured results, safety details, a pilot schedule, and availability. "
        ) * 8
        driving_text = (
            "A fully autonomous robotaxi carried paid riders on public roads without a safety driver. "
            "The self-driving vehicle completed 12,000 rides and published highway safety results. "
            "The operator released an uncut field-test video, deployment map, and service details. "
        ) * 8
        robot = DiscoveryCandidate(
            id="useful-home-robot", channel=DiscoveryChannel.ROBOTICS,
            url="https://www.sunday.ai/home-worker", title="Household robot folds laundry in a real home",
            publisher="Sunday Robotics", published_at=NOW.isoformat(), summary=robot_text,
            body_text=robot_text, stable_id="web:useful-home-robot",
            metadata={
                "image_count": 3,
                "source_video_url": "https://www.youtube.com/watch?v=HOME123",
            }, discovered_at=NOW.isoformat(),
        )
        driving = DiscoveryCandidate(
            id="public-road-robotaxi", channel=DiscoveryChannel.AUTONOMOUS_DRIVING,
            url="https://waymo.com/blog/paid-rides", title="Robotaxi carries riders without safety driver",
            publisher="Waymo", published_at=NOW.isoformat(), summary=driving_text,
            body_text=driving_text, stable_id="web:public-road-robotaxi",
            metadata={
                "image_count": 3,
                "source_video_url": "https://www.youtube.com/watch?v=DRIVE123",
            }, discovered_at=NOW.isoformat(),
        )
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            factory = FakeFactory([
                {"status": "completed", "publishable": True, "video": "first.mp4"},
                {"status": "completed", "publishable": True, "video": "second.mp4"},
            ])
            service = ResourceDiscoveryService(
                workspace,
                adapters={
                    DiscoveryChannel.ROBOTICS: StaticAdapter([robot]),
                    DiscoveryChannel.AUTONOMOUS_DRIVING: StaticAdapter([driving]),
                },
                factory=factory, clock=lambda: NOW, sleeper=lambda _: None,
            )
            config = ResourceDiscoveryConfig(retry_backoff_seconds=[0])
            for channel in DiscoveryChannel:
                config.channels[channel].enabled = channel in {
                    DiscoveryChannel.ROBOTICS, DiscoveryChannel.AUTONOMOUS_DRIVING,
                }

            result = service.run(config, scheduled=False)

            self.assertEqual(result.channels["robotics"].status, "generated")
            self.assertEqual(result.channels["autonomous_driving"].status, "generated")
            self.assertEqual({url for url, _ in factory.generate_calls}, {robot.url, driving.url})
            self.assertEqual(len(factory.generate_calls), 2)
            generated = workspace.load_discovery_state()["generated_events"]
            self.assertEqual({event["candidate_id"] for event in generated}, {robot.id, driving.id})

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

    def test_rss_search_reports_when_every_source_failed(self) -> None:
        adapter = RSSDiscoveryAdapter(DiscoveryChannel.ROBOTICS)
        adapter._download = MagicMock(side_effect=OSError("DNS unavailable"))
        config = ChannelConfig.from_dict(DiscoveryChannel.ROBOTICS, {
            "feeds": ["https://a.example/feed", "https://b.example/feed"],
            "queries": [],
        })

        with self.assertRaisesRegex(RuntimeError, "failed for all 2 sources"):
            adapter.search(config, NOW)

    def test_rss_search_round_robins_probe_budget_across_sources(self) -> None:
        crowded = (
            "<rss><channel>"
            + "".join(
                f"<item><title>Crowded {index}</title>"
                f"<link>https://a.example/{index}</link>"
                f"<pubDate>Fri, 28 Aug 2026 05:{59-index:02d}:00 GMT</pubDate>"
                "<description>robotics demo</description></item>"
                for index in range(5)
            )
            + "</channel></rss>"
        ).encode()
        independent = (
            "<rss><channel><item><title>First robot field test</title>"
            "<link>https://b.example/field-test</link>"
            "<pubDate>Fri, 28 Aug 2026 05:40:00 GMT</pubDate>"
            "<description>first autonomous field test video</description>"
            "</item></channel></rss>"
        ).encode()
        adapter = RSSDiscoveryAdapter(
            DiscoveryChannel.ROBOTICS,
            fetcher=lambda url: ("robot field test demo with concrete results. " * 30, url),
        )
        adapter._download = lambda url: crowded if "a.example" in url else independent
        config = ChannelConfig.from_dict(DiscoveryChannel.ROBOTICS, {
            "feeds": ["https://a.example/feed", "https://b.example/feed"],
            "queries": [], "seed_domains": ["a.example", "b.example"], "probe_limit": 2,
        })

        found = adapter.search(config, NOW)

        self.assertEqual({item.url for item in found}, {
            "https://a.example/0", "https://b.example/field-test",
        })

    def test_rss_search_probes_sources_concurrently(self) -> None:
        active = 0
        maximum_active = 0
        lock = threading.Lock()

        def fetcher(url):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.04)
            with lock:
                active -= 1
            return "robot field test demo with concrete results. " * 30, url

        adapter = RSSDiscoveryAdapter(DiscoveryChannel.ROBOTICS, fetcher=fetcher)
        payloads = {
            "https://a.example/feed": (
                "<rss><channel><item><title>A robot demo</title><link>https://a.example/a</link>"
                "<pubDate>Fri, 28 Aug 2026 05:59:00 GMT</pubDate>"
                "<description>autonomous demo video</description></item></channel></rss>"
            ).encode(),
            "https://b.example/feed": (
                "<rss><channel><item><title>B robot demo</title><link>https://b.example/b</link>"
                "<pubDate>Fri, 28 Aug 2026 05:58:00 GMT</pubDate>"
                "<description>autonomous demo video</description></item></channel></rss>"
            ).encode(),
        }
        adapter._download = lambda url: payloads[url]
        config = ChannelConfig.from_dict(DiscoveryChannel.ROBOTICS, {
            "feeds": list(payloads), "queries": [],
            "seed_domains": ["a.example", "b.example"], "probe_limit": 2,
        })

        found = adapter.search(config, NOW)

        self.assertEqual(len(found), 2)
        self.assertGreaterEqual(maximum_active, 2)

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

    def test_unified_youtube_does_not_reapply_the_ten_probe_cap(self) -> None:
        with TemporaryDirectory() as temp:
            items = [
                DiscoveryCandidate(
                    id=f"youtube-pool-{index}", channel=DiscoveryChannel.YOUTUBE,
                    url=f"https://youtube.com/watch?v=pool{index}",
                    title=f"AI engineering source {index}", publisher="Engineering Channel",
                    published_at=NOW.isoformat(),
                    summary="A technical AI engineering discussion with architecture and deployment details. " * 4,
                    body_text="A technical AI engineering discussion with architecture and deployment details. " * 4,
                    metadata={
                        "duration_seconds": 1800, "transcript_available": False,
                        "youtube_editorial_mode": "technical_coverage", "youtube_score": 85,
                    },
                )
                for index in range(13)
            ]
            workspace = Workspace(Path(temp))
            service = ResourceDiscoveryService(
                workspace, adapters={DiscoveryChannel.YOUTUBE: StaticAdapter(items)},
                factory=FakeFactory([]), clock=lambda: NOW, sleeper=lambda _: None,
            )
            config = ResourceDiscoveryConfig()
            for channel in DiscoveryChannel:
                config.channels[channel].enabled = channel == DiscoveryChannel.YOUTUBE
            config.channels[DiscoveryChannel.YOUTUBE].probe_limit = 10
            config.channels[DiscoveryChannel.YOUTUBE].max_candidates = 20

            result = service.run(config, scheduled=False)

            entry = result.channels["youtube"]
            self.assertEqual(len(entry.candidates), 13)
            self.assertEqual(entry.trace["funnel"]["evaluated"], 13)

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

    def test_new_channels_run_end_to_end_from_discovery_to_generation_audit(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            project_body = (
                "The AI startup reached 1 million users and reported $12 million ARR. "
                "The team grew 300% and raised $25 million after launching its production API. "
                "The report explains customer adoption, the workflow, product availability, and pricing. "
            ) * 8
            project = DiscoveryCandidate(
                id="projects-breakout", channel=DiscoveryChannel.PROJECTS,
                url="https://techcrunch.com/breakout-ai", title="Breakout AI product reaches one million users",
                publisher="TechCrunch", published_at=NOW.isoformat(), summary=project_body,
                body_text=project_body, stable_id="web:projects-breakout",
                metadata={"image_count": 3}, discovered_at=NOW.isoformat(),
            )
            robotics_body = (
                "A physical AI humanoid product launched for warehouse deployment today. "
                "The robot completed 120 field-test hours with perception, manipulation, and motion planning. "
                "The engineering team published a demo, pilot schedule, safety results, and availability details. "
            ) * 8
            robotics = DiscoveryCandidate(
                id="robotics-field-test", channel=DiscoveryChannel.ROBOTICS,
                url="https://www.therobotreport.com/humanoid-field-test/",
                title="Physical AI humanoid begins warehouse deployment",
                publisher="The Robot Report", published_at=NOW.isoformat(), summary=robotics_body,
                body_text=robotics_body, stable_id="web:robotics-field-test",
                metadata={
                    "image_count": 4,
                    "source_video_url": "https://www.youtube.com/watch?v=ROBOT99",
                }, discovered_at=NOW.isoformat(),
            )
            factory = FakeFactory([
                {"status": "completed", "publishable": True, "video": "project-final.mp4"},
                {"status": "completed", "publishable": True, "video": "robotics-final.mp4"},
            ])
            service = ResourceDiscoveryService(
                workspace,
                adapters={
                    DiscoveryChannel.PROJECTS: StaticAdapter([project]),
                    DiscoveryChannel.ROBOTICS: StaticAdapter([robotics]),
                },
                factory=factory, clock=lambda: NOW, sleeper=lambda _: None,
            )
            config = ResourceDiscoveryConfig(retry_backoff_seconds=[0])
            for channel in DiscoveryChannel:
                config.channels[channel].enabled = channel in {
                    DiscoveryChannel.PROJECTS, DiscoveryChannel.ROBOTICS,
                }

            result = service.run(config, scheduled=False, provider="auto")

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.channels["projects"].status, "generated")
            self.assertEqual(result.channels["robotics"].status, "generated")
            self.assertEqual(len(factory.generate_calls), 2)
            calls = {url: options for url, options in factory.generate_calls}
            self.assertEqual(calls[project.url].topic.value, "company_or_team")
            self.assertEqual(calls[project.url].content_type.value, "flash")
            self.assertEqual(calls[robotics.url].topic.value, "model_or_product")
            self.assertIn("BINDING STORY SCOPE", calls[robotics.url].discovery_context)
            self.assertIn(robotics.title, calls[robotics.url].discovery_context)
            self.assertEqual(calls[robotics.url].render_profile, "radar_v2")
            self.assertTrue(all(options.research for options in calls.values()))
            self.assertTrue((workspace.root / "discovery" / "runs" / f"{result.id}.json").is_file())
            state = workspace.load_discovery_state()
            self.assertEqual(
                {item["candidate_id"] for item in state["generated_events"]},
                {project.id, robotics.id},
            )

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

    def test_invalid_cached_image_manifest_falls_back_to_full_generation(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            job = workspace.root / "jobs" / "failed-image-manifest"
            job.mkdir(parents=True)
            manifest = job / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            source_url = "https://www.therobotreport.com/physical-ai"
            (job / "result.json").write_text(json.dumps({
                "url": source_url, "status": "failed", "manifest": str(manifest),
            }), encoding="utf-8")
            item = DiscoveryCandidate(
                id="robotics-image", channel=DiscoveryChannel.ROBOTICS,
                url=source_url, title="Physical AI field test", publisher="The Robot Report",
                published_at=NOW.isoformat(), eligible=True, status="blocked",
                topic_type=TopicType.MODEL_OR_PRODUCT, content_type=ContentType.EXPLAINER,
            )
            factory = FakeFactory([{
                "status": "completed", "publishable": True, "video": "final.mp4",
            }])
            factory.rerender = MagicMock(side_effect=ValueError(
                "source image scene scene-4 does not cite an archived image asset",
            ))
            service = ResourceDiscoveryService(
                workspace, factory=factory, clock=lambda: NOW, sleeper=lambda _: None,
            )

            result = service._adopt(
                item, ResourceDiscoveryConfig(retry_backoff_seconds=[0, 0]), "auto", None,
            )

            self.assertEqual(result["status"], "generated")
            self.assertEqual(
                [attempt["mode"] for attempt in result["attempts"]],
                ["deterministic_rerender", "full_generation"],
            )
            self.assertEqual(
                result["attempts"][0]["recovery"],
                "discard_invalid_manifest_and_regenerate",
            )
            factory.rerender.assert_called_once_with(manifest)
            self.assertEqual(len(factory.generate_calls), 1)

    def test_internal_programming_error_is_not_retried_three_times(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            candidate = DiscoveryCandidate(
                id="robotics-code-error", channel=DiscoveryChannel.ROBOTICS,
                url="https://example.com/robot-demo", title="First robot demo",
                publisher="Example", published_at=NOW.isoformat(),
                summary="robot demo " * 80, body_text="robot demo " * 80,
                stable_id="web:robotics-code-error", eligible=True,
            )
            factory = MagicMock()
            factory.generate.side_effect = NameError("validator variable is undefined")
            service = ResourceDiscoveryService(workspace, factory=factory)

            result = service._adopt(candidate, ResourceDiscoveryConfig(), "auto", None)

            self.assertEqual(result["status"], "blocked")
            self.assertEqual(len(result["attempts"]), 1)
            self.assertEqual(
                result["attempts"][0]["recovery"],
                "stop_non_retryable_internal_error",
            )

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
