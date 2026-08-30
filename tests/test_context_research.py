import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.agent import _context_graph_from_evidence
from video_factory.factory import VideoFactory
from video_factory.llm import EditorialPlan, OpenAICompatibleStoryWriter
from video_factory.models import Candidate, ContentType, Evidence, SourceType, TopicType
from video_factory.research import (
    _archive_same_author_setup, _archive_visual_actor_context, _incumbent_history_query, _is_prior_distinct_event,
    _reported_context_matches_root,
)
from video_factory.storage import Workspace
from video_factory.writer import StoryWriterPacket


class ContextResearchTests(unittest.TestCase):
    def _packet(self, workspace: Workspace) -> StoryWriterPacket:
        candidate = Candidate(
            "tweet-tibo", SourceType.TWEET,
            "https://x.com/thsottiaux/status/2090964822422949999",
            "The banked reset has landed", author="thsottiaux",
        )
        root = Evidence(
            "root", candidate.id, candidate.source_url,
            "The banked reset has landed. Have an amazing weekend.", "x:thread_post",
            metadata={
                "author_handle": "thsottiaux", "author_name": "Tibo",
                "published_at": "Sat Aug 22 00:50:36 +0000 2026",
            },
        )
        quoted = Evidence(
            "quoted", candidate.id, "https://x.com/thsottiaux/status/2090947196107764189",
            "The banked reset will be there by 8pm PST for all paid users of ChatGPT Work and Codex.",
            "x:quoted_post", metadata={"published_at": "Fri Aug 21 23:40:34 +0000 2026"},
        )
        return StoryWriterPacket(
            candidate, [root, quoted], TopicType.OFFICIAL_ANNOUNCEMENT, ContentType.FLASH, 12,
        )

    def test_same_author_timeline_recovers_the_setup_not_an_unrelated_post(self) -> None:
        rows = [{
            "id": "2090766694897619318", "author": "thsottiaux", "name": "Tibo",
            "text": "We hit 20M active users for Codex and every Codex and ChatGPT Work user gets a BANKED reset.",
            "created_at": "Fri Aug 21 11:43:19 +0000 2026",
            "url": "https://x.com/thsottiaux/status/2090766694897619318",
            "likes": 19768, "retweets": 955, "replies": 2628, "views": 3072806,
        }, {
            "id": "unrelated", "author": "thsottiaux", "name": "Tibo",
            "text": "Transparent images are available today.",
            "created_at": "Fri Aug 21 02:46:59 +0000 2026",
            "url": "https://x.com/thsottiaux/status/unrelated",
        }]
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            packet = self._packet(workspace)
            completed = SimpleNamespace(stdout=json.dumps(rows))
            with patch("video_factory.research.subprocess.run", return_value=completed):
                evidence, actions = _archive_same_author_setup(workspace, Path(temp), packet)
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0].metadata["context_role"], "same_author_setup")
            self.assertIn("20M", evidence[0].quote)
            self.assertEqual(actions[0]["status"], "archived")

    def test_screenshot_account_dispute_recovers_the_exact_setup_posts(self) -> None:
        rows = [{
            "id": "alex-root", "author": "alexgetmancom",
            "text": "I followed the setup almost exactly. Shortly afterward, Anthropic suspended my account. I filed an appeal.",
            "created_at": "Sat Aug 08 14:11:36 +0000 2026",
            "url": "https://x.com/alexgetmancom/status/alex-root",
            "quoted_tweet": {
                "author": "thsottiaux",
                "text": "Install CLIProxyAPI and run Claude Code with gpt-5.6-sol. If this gets blocked, I owe you a reset.",
            },
            "card": {"title": "claudex", "description": "Run Claude Code with a different model behind it"},
        }, {
            "id": "alex-update", "author": "alexgetmancom",
            "text": "UPDATE: actual model was gpt-5.6-luna. Requests were routed to the Codex backend via Codex OAuth. The Anthropic suspension email arrived seconds later.",
            "created_at": "Sat Aug 08 16:46:30 +0000 2026",
            "url": "https://x.com/alexgetmancom/status/alex-update",
        }]
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            candidate = Candidate(
                "tweet-sam", SourceType.TWEET, "https://x.com/sama/status/1",
                "Sam mentions Tibo", author="sama",
            )
            root = Evidence(
                "root", candidate.id, candidate.source_url, "openai is tibo", "x:thread_post",
                metadata={"author_handle": "sama", "published_at": "Sun Aug 09 15:09:52 +0000 2026"},
            )
            analysis = Evidence(
                "visual", candidate.id, "https://pbs.twimg.com/context.jpg",
                '{"visible_text":"alex getman @alexgetmancom followed the setup; Anthropic suspended the account and he filed an appeal. Tibo @thsottiaux asked why another model was banned."}',
                "x:visual_analysis",
            )
            packet = StoryWriterPacket(
                candidate, [root, analysis], TopicType.PRACTICE_POST, ContentType.FLASH, 12,
            )
            with patch(
                "video_factory.research.subprocess.run",
                return_value=SimpleNamespace(stdout=json.dumps(rows)),
            ):
                evidence, actions = _archive_visual_actor_context(
                    workspace, Path(temp), packet, limit=2,
                )
            self.assertEqual(len(evidence), 2)
            combined = "\n".join(item.quote for item in evidence)
            self.assertIn("gpt-5.6-luna", combined)
            self.assertIn("Claude Code with gpt-5.6-sol", combined)
            self.assertTrue(all(item.metadata["context_role"] == "referenced_setup" for item in evidence))
            self.assertEqual(actions[-1]["status"], "archived")

    def test_people_move_adds_prior_incumbent_query_and_rejects_same_day_headlines(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            base = self._packet(workspace)
            base.evidence[0].notes = "Author bio: Co-founder. Former Chief Scientist, Google."
            base.evidence[0].metadata["published_at"] = "Wed Aug 05 16:06:02 +0000 2026"
            base.candidate.author = "JeffDean"
            packet = StoryWriterPacket(
                base.candidate, base.evidence, TopicType.COMPANY_OR_TEAM, ContentType.FLASH, 12,
            )
            plan = EditorialPlan(
                "Jeff Dean 创业", "技术人才流动", ["root"], [], [], True,
                story_archetype="people_change",
            )
            query = _incumbent_history_query(packet, plan)
            self.assertIn("Google", query or "")
            self.assertIn("Jeff Dean", query or "")
            self.assertFalse(_is_prior_distinct_event(
                packet.candidate, packet.evidence, "Four Google researchers launch a startup",
                "Wed, 05 Aug 2026 07:00:00 GMT",
            ))
            self.assertTrue(_is_prior_distinct_event(
                packet.candidate, packet.evidence, "Another Google AI leader left for a new lab",
                "Mon, 20 Jul 2026 07:00:00 GMT",
            ))

    def test_context_graph_separates_setup_pattern_and_duplicate_confirmation(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            packet = self._packet(workspace)
            prior = Evidence(
                "prior", packet.candidate.id, "https://x.com/a/status/prior", "Earlier setup",
                "x:related_prior_post", metadata={"context_role": "same_author_setup", "author_name": "Tibo"},
            )
            history = Evidence(
                "history", packet.candidate.id, "https://example.com/history", "Earlier incumbent move",
                "web:reported_context", metadata={"context_role": "incumbent_history"},
            )
            one = Evidence(
                "confirm-1", packet.candidate.id, "https://example.com/one", "Current event confirmed",
                "web:reported_context", metadata={"context_role": "event_context"},
            )
            two = Evidence(
                "confirm-2", packet.candidate.id, "https://example.com/two", "Same current event again",
                "web:reported_context", metadata={"context_role": "event_context"},
            )
            plan = EditorialPlan(
                "event", "audience", ["root"], [], [], True,
                story_archetype="event_chain",
            )
            graph = _context_graph_from_evidence(plan, [*packet.evidence, prior, history, one, two])
            ids_by_evidence = {event.evidence_ids[0]: event.id for event in graph.events}
            self.assertIn(ids_by_evidence["prior"], graph.required_context_ids)
            self.assertIn(ids_by_evidence["history"], graph.discarded_context_ids)
            self.assertIn(ids_by_evidence["confirm-1"], graph.required_context_ids)
            self.assertIn(ids_by_evidence["confirm-2"], graph.discarded_context_ids)

            people_plan = EditorialPlan(
                "person left incumbent", "developers", ["root"], [], [], True,
                story_archetype="people_change",
            )
            people_graph = _context_graph_from_evidence(
                people_plan, [*packet.evidence, history],
            )
            people_ids = {event.evidence_ids[0]: event.id for event in people_graph.events}
            self.assertIn(people_ids["history"], people_graph.pattern_context_ids)

            technical_plan = EditorialPlan(
                "architecture analysis", "developers", ["root"], [], [], True,
                story_archetype="research_disclosure",
            )
            referenced = Evidence(
                "referenced", packet.candidate.id, "https://x.com/a/status/referenced",
                "I followed this setup and the account was suspended",
                "x:referenced_context_post",
                metadata={"context_role": "referenced_setup", "author_name": "Alex"},
            )
            technical_graph = _context_graph_from_evidence(
                technical_plan, [*packet.evidence, prior, one, referenced],
            )
            technical_ids = {
                event.evidence_ids[0]: event.id for event in technical_graph.events
            }
            self.assertIn(technical_ids["prior"], technical_graph.discarded_context_ids)
            self.assertIn(technical_ids["confirm-1"], technical_graph.discarded_context_ids)
            self.assertIn(technical_ids["referenced"], technical_graph.required_context_ids)

    def test_visual_actor_context_reuses_archived_exact_post(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            packet = self._packet(workspace)
            referenced = Evidence(
                "referenced", packet.candidate.id,
                "https://x.com/alex/status/referenced",
                "I followed this setup and Anthropic suspended my account.",
                "x:referenced_context_post",
                metadata={
                    "context_role": "referenced_setup", "author_name": "Alex",
                    "published_at": "Sat Aug 08 14:11:36 +0000 2026",
                },
            )
            workspace.save_evidence(referenced)

            recovered, actions = _archive_visual_actor_context(
                workspace, Path(temp), packet, limit=2,
            )

            self.assertEqual([item.id for item in recovered], ["referenced"])
            self.assertEqual(actions[0]["status"], "workspace_cache")

    def test_planner_cannot_turn_a_compliment_into_a_people_move(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            packet = self._packet(workspace)
            packet.candidate.title = "lol another one of the things i like most about openai is tibo"
            packet.evidence[0].quote = packet.candidate.title
            packet.evidence[1].quote = "@sama How much they celebrate with Anthropic"
            draft = {
                "angle": "Altman 点名 Tibo",
                "audience_value": "观察公司竞争",
                "selected_evidence_ids": ["root", "quoted"],
                "requested_urls": [], "unresolved_questions": ["Tibo 指代待确认"],
                "ready_to_write": True,
                "why_now": "刚刚发生", "why_audience": "技术圈关注公司竞争",
                "audience_pain_or_desire": "identity",
                "selection_reasons": [{
                    "id": "reply", "dimension": "competition", "rationale": "CEO 公开回应",
                    "evidence_ids": ["root", "quoted"],
                }],
                "expansion_dimensions": ["people", "historical_pattern", "competition"],
                "context_questions": ["Tibo 是谁"],
                "search_queries": ["Google AI talent departures", "Tibo OpenAI Anthropic"],
                "story_archetype": "people_change",
            }
            plan = OpenAICompatibleStoryWriter._parse_plan(packet, draft)
            self.assertEqual(plan.story_archetype, "other")
            self.assertNotIn("people", plan.expansion_dimensions)
            self.assertEqual(plan.search_queries, ["Tibo OpenAI Anthropic"])

    def test_recruiting_reply_inside_a_quoted_dispute_is_an_event_chain_not_people_move(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            packet = self._packet(workspace)
            packet.candidate.title = "lol another one of the things i like most about openai is tibo"
            packet.evidence[0].quote = packet.candidate.title
            packet.evidence[1].quote = (
                "A user appealed an account suspension. Tibo replied that the ban seemed odd. "
                "Boris replied: We are hiring if you would like to work at Anthropic."
            )
            draft = {
                "angle": "Sam 点名 Tibo，引用一串封号申诉与招聘回复",
                "audience_value": "看懂多方公开互动",
                "selected_evidence_ids": ["root", "quoted"],
                "requested_urls": [], "unresolved_questions": ["封号原因未知"],
                "ready_to_write": True, "why_now": "刚刚发生",
                "why_audience": "开发者关心账号风险与公司互动",
                "audience_pain_or_desire": "risk",
                "selection_reasons": [{
                    "id": "interaction", "dimension": "competition",
                    "rationale": "多方公开回复形成完整事件链",
                    "evidence_ids": ["root", "quoted"],
                }],
                "expansion_dimensions": ["people", "cause", "later_update"],
                "context_questions": ["封号原因是否确认"],
                "search_queries": ["OpenAI Anthropic talent departures"],
                "story_archetype": "people_change",
            }

            plan = OpenAICompatibleStoryWriter._parse_plan(packet, draft)

            self.assertEqual(plan.story_archetype, "event_chain")
            self.assertNotIn("people", plan.expansion_dimensions)

    def test_reported_context_needs_a_headline_entity_bridge(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            packet = self._packet(workspace)
            packet.candidate.title = "OpenAI banked reset has landed"
            self.assertTrue(_reported_context_matches_root(
                packet.candidate, packet.evidence, "OpenAI rolls out banked reset to paid users",
            ))
            self.assertFalse(_reported_context_matches_root(
                packet.candidate, packet.evidence, "Google takes the hit in AI talent war",
            ))

    def test_reported_context_rejects_generic_discovery_overlap(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            packet = self._packet(workspace)
            packet.candidate.title = "AWS Agentic Resource Discovery open spec"
            packet.evidence[0].quote = (
                "Agentic Resource Discovery federates agent catalogs like DNS."
            )

            self.assertFalse(_reported_context_matches_root(
                packet.candidate, packet.evidence,
                "Amazon launches AI research tool to speed early-stage drug discovery",
            ))

    def test_factory_reuses_archived_root_without_reusing_old_research_context(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            candidate = Candidate("tweet-1", SourceType.TWEET, "https://x.com/dev/status/1", "Root")
            source = Path(temp) / "capture.json"
            source.write_text("{}", encoding="utf-8")
            asset, digest = workspace.archive_asset(source, "twitter-captures")
            root = Evidence(
                "root", candidate.id, candidate.source_url, "Root post", "x:thread_post",
                captured_asset=asset, sha256=digest,
            )
            old_context = Evidence(
                "old-context", candidate.id, "https://example.com/context", "Old report", "web:reported_context",
                captured_asset=asset, sha256=digest, metadata={"context_role": "event_context"},
            )
            old_source_image = Evidence(
                "old-image", candidate.id, "https://example.com/team.jpg", "Official team image",
                "web:source_image", captured_asset=asset, sha256=digest,
                metadata={"visual_role": "team", "editorial_priority": "high"},
            )
            workspace.save_candidate(candidate)
            workspace.save_evidence(root)
            workspace.save_evidence(old_context)
            workspace.save_evidence(old_source_image)
            cached = VideoFactory(workspace)._cached_acquisition(candidate.source_url)
            self.assertIsNotNone(cached)
            self.assertEqual(cached.method, "workspace-acquisition-cache")
            self.assertEqual([item.id for item in cached.ingest.evidence], ["root"])


if __name__ == "__main__":
    unittest.main()
