import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from video_factory.agent import AgentBudget, BoundedContentAgent, LinkedSourceResearchTool, _visible_copy_only_failure
from video_factory.director import NarrativeAnswer, SceneProposal, StoryboardRequest
from video_factory.llm import EditorialPlan, StoryDraftError
from video_factory.models import Candidate, ContentType, Evidence, MaterialRole, SourceType, TopicType
from video_factory.storage import Workspace
from video_factory.writer import StoryWriterPacket


def packet_with_link() -> StoryWriterPacket:
    candidate = Candidate(
        "tweet-1", SourceType.TWEET, "https://x.com/example/status/1", "一个具体实践",
        linked_sources=["https://example.com/primary"],
    )
    evidence = Evidence("tweet-evidence", candidate.id, candidate.source_url, "作者展示了一次具体实践。", "x:thread_post")
    return StoryWriterPacket(candidate, [evidence], TopicType.PRACTICE_POST, ContentType.FLASH, 8)


def valid_request(packet: StoryWriterPacket) -> StoryboardRequest:
    evidence_id = packet.evidence[0].id
    return StoryboardRequest(
        "story-1", packet.candidate, packet.topic_type, packet.content_type, packet.evidence,
        "这是一次可验证的实践，不是普遍规律",
        [
            NarrativeAnswer("author_claim", "作者展示了一次具体实践", [evidence_id]),
            NarrativeAnswer("evidence_context", "证据来自已归档原帖", [evidence_id]),
            NarrativeAnswer("scope", "结果只适用于作者展示的条件", [evidence_id]),
        ],
        [
            SceneProposal(
                "event", "internal", "作者展示了什么", MaterialRole.PROOF, "show source",
                [evidence_id], ["author_claim"], duration_hint=4,
                screen_fact="作者展示了一次具体实践", screen_interpretation="先确认事实，不扩大结论",
            ),
            SceneProposal(
                "scope", "internal", "它能说明什么", MaterialRole.EXPLANATION, "show source scope",
                [evidence_id], ["scope"], duration_hint=4,
                screen_fact="这是作者的一次结果", screen_interpretation="值得试，但不能推成普遍规律",
            ),
        ],
        8, fixed_hook="一次实践能证明多少？",
    )


class RepairingModel:
    def __init__(self, fail_repair: bool = False) -> None:
        self.generate_calls = 0
        self.repair_calls = 0
        self.fail_repair = fail_repair

    def plan(self, packet):
        return EditorialPlan(
            "验证具体结果与适用范围", "判断是否值得复现", [packet.evidence[0].id], [], [], True,
        ), {"model": "cheap"}

    def generate(self, packet):
        self.generate_calls += 1
        raise StoryDraftError({"broken": True}, ValueError("missing scenes"))

    def repair(self, packet, invalid_draft, validation_error):
        self.repair_calls += 1
        if self.fail_repair:
            raise StoryDraftError(invalid_draft, ValueError("still invalid"))
        return valid_request(packet), {"model": "cheap"}, {"repaired": True}


class ValidModel:
    def __init__(self) -> None:
        self.generate_calls = 0

    def plan(self, packet):
        raise AssertionError("escalation model must not repeat planning")

    def generate(self, packet):
        self.generate_calls += 1
        return valid_request(packet), {"model": "strong"}, {"valid": True}

    def repair(self, packet, invalid_draft, validation_error):
        raise AssertionError("escalation gets one bounded attempt")


class ValidPrimaryModel(ValidModel):
    def plan(self, packet):
        return EditorialPlan(
            "验证具体结果与适用范围", "判断是否值得复现", [packet.evidence[0].id], [], [], True,
        ), {"model": "cheap"}


class ReviewingPrimaryModel(ValidPrimaryModel):
    def __init__(self, issues=None) -> None:
        super().__init__()
        self.issues = issues or []
        self.repair_calls = 0
        self.review_calls = 0

    def review_visible_copy(self, packet, draft):
        self.review_calls += 1
        return (self.issues if self.review_calls == 1 else []), {"model": "critic"}

    def repair(self, packet, invalid_draft, validation_error):
        self.repair_calls += 1
        self.asserted_error = validation_error
        return valid_request(packet), {"model": "writer"}, {"review_repaired": True}


class ContentAgentTest(unittest.TestCase):
    def test_only_render_size_failures_receive_the_bounded_extra_cleanup(self) -> None:
        self.assertTrue(_visible_copy_only_failure(
            "root-post Chinese translation must fit beside the source in at most 140 characters; "
            "fixed conclusion must fit the persistent bottom rail"
        ))
        self.assertFalse(_visible_copy_only_failure(
            "people-change story must visibly use context; root-post Chinese translation must fit"
        ))

    def test_valid_first_draft_is_returned_without_a_repair(self) -> None:
        model = ValidPrimaryModel()
        result = BoundedContentAgent(model).run(packet_with_link())
        self.assertEqual(result.llm_calls, 2)
        self.assertEqual(result.manifest.id, "story-1")

    def test_invalid_story_is_repaired_once_by_the_cheap_model(self) -> None:
        model = RepairingModel()
        result = BoundedContentAgent(model).run(packet_with_link())
        self.assertEqual(result.llm_calls, 3)
        self.assertFalse(result.used_escalation)
        self.assertEqual(model.repair_calls, 1)
        agent_check = next(item for item in result.manifest.quality_checks if item["name"] == "content_agent")
        self.assertEqual(agent_check["detail"]["mode"], "contextual_director_loop")

    def test_semantic_copy_critic_can_request_one_bounded_repair(self) -> None:
        model = ReviewingPrimaryModel([{
            "field_path": "editorial_brief.fixed_conclusion",
            "category": "entity_relation",
            "problem": "the recipient is ambiguous",
            "evidence_ids": ["tweet-evidence"],
            "repair_instruction": "name the recipient from evidence",
        }])
        result = BoundedContentAgent(model).run(packet_with_link())
        self.assertEqual(result.llm_calls, 5)
        self.assertEqual(model.repair_calls, 1)
        self.assertEqual(model.review_calls, 2)
        self.assertIn("semantic copy critic issues", model.asserted_error)
        steps = [item["step"] for item in next(
            check for check in result.manifest.quality_checks if check["name"] == "content_agent"
        )["detail"]["trace"]]
        self.assertIn("copy_review", steps)
        self.assertIn("copy_review_repair", steps)
        self.assertIn("copy_review_verify", steps)

    def test_copy_critic_is_not_started_without_repair_and_verify_budget(self) -> None:
        model = ReviewingPrimaryModel([{
            "field_path": "editorial_brief.fixed_conclusion",
            "category": "payoff",
            "problem": "the conclusion needs a concrete payoff",
            "evidence_ids": ["tweet-evidence"],
            "repair_instruction": "state the supported payoff",
        }])
        result = BoundedContentAgent(
            model,
            budget=AgentBudget(max_llm_calls=4, max_repairs=1, max_escalations=0),
        ).run(packet_with_link())
        self.assertEqual(result.llm_calls, 2)
        self.assertEqual(model.review_calls, 0)
        self.assertEqual(model.repair_calls, 0)

    def test_strong_model_is_only_used_after_primary_and_repair_fail(self) -> None:
        primary = RepairingModel(fail_repair=True)
        strong = ValidModel()
        result = BoundedContentAgent(
            primary, escalation=strong,
            budget=AgentBudget(max_llm_calls=4, max_repairs=1, max_escalations=1),
        ).run(packet_with_link())
        self.assertTrue(result.used_escalation)
        self.assertEqual(result.llm_calls, 4)
        self.assertEqual(strong.generate_calls, 1)

    def test_linked_source_tool_archives_before_returning_evidence(self) -> None:
        packet = packet_with_link()
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            tool = LinkedSourceResearchTool(
                workspace, fetcher=lambda _: (b"<html><body><h1>Primary proof</h1><p>Visible result</p></body></html>", "text/html"),
            )
            outcome = tool.run(packet, ["https://example.com/primary", "https://untrusted.example/"], 3)
            self.assertEqual(outcome.actions[0]["status"], "archived")
            self.assertEqual(outcome.actions[1]["status"], "rejected_not_linked")
            archived = outcome.evidence[-1]
            self.assertIn("Primary proof", archived.quote)
            self.assertTrue((workspace.root / archived.captured_asset).is_file())
            self.assertIsNotNone(archived.sha256)

    def test_linked_source_tool_promotes_official_team_photo_to_evidence(self) -> None:
        from PIL import Image

        image_bytes = BytesIO()
        Image.new("RGB", (1200, 700), "navy").save(image_bytes, format="JPEG")
        page = (
            b'<html><body><h1>The Team</h1><img src="assets/team.jpg" '
            b'alt="The founding team of Example"></body></html>'
        )
        packet = packet_with_link()
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            tool = LinkedSourceResearchTool(
                workspace,
                fetcher=lambda _: (page, "text/html", "https://example.com/primary"),
                media_fetcher=lambda _: (
                    image_bytes.getvalue(), "image/jpeg", "https://example.com/assets/team.jpg",
                ),
            )
            outcome = tool.run(packet, ["https://example.com/primary"], 1)
            photo = next(item for item in outcome.evidence if item.source_kind == "web:source_image")
            self.assertEqual(photo.metadata["visual_role"], "team")
            self.assertEqual(photo.metadata["editorial_priority"], "high")
            self.assertTrue((workspace.root / photo.captured_asset).is_file())


if __name__ == "__main__":
    unittest.main()
