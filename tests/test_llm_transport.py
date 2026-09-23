import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request
from unittest.mock import patch

from video_factory.llm import (
    LLMSettings, OpenAICompatibleStoryWriter, TransportFallbackStoryWriter,
    _coerce_model_float,
)
from video_factory.llm_transport import LLMBudgetExceeded, LLMTransport
from video_factory.models import (
    Candidate, ContentType, EditorialOpportunity, Evidence, SelectionReason,
    SourceType, TopicType,
)
from video_factory.writer import StoryWriterPacket


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class LLMTransportTests(unittest.TestCase):
    def test_transport_ledgers_every_billable_retry_without_prompt_text(self) -> None:
        with TemporaryDirectory() as temp:
            transport = LLMTransport(Path(temp))
            writer = OpenAICompatibleStoryWriter(LLMSettings(
                "openrouter", "https://openrouter.example/api/v1", "test-key", "critic",
            ), transport)
            responses = [
                _Response({
                    "model": "critic", "choices": [{
                        "message": {"content": ""}, "finish_reason": "error",
                    }], "usage": {"total_tokens": 100, "cost": 0.01},
                }),
                _Response({
                    "model": "critic", "choices": [{
                        "message": {"content": '{"ok":true}'}, "finish_reason": "stop",
                    }], "usage": {"total_tokens": 50, "cost": 0.005},
                }),
            ]
            with (
                transport.scope(
                    job_id="job-1", candidate_id="candidate-1", max_requests=4,
                    max_cost_usd=0.10, max_openrouter_semantic_reviews=2,
                ),
                patch("video_factory.llm.urlopen", side_effect=responses),
            ):
                draft, _ = writer._request_json([
                    {"role": "user", "content": "SECRET PROMPT TEXT"},
                ], 100)

            self.assertEqual(draft, {"ok": True})
            rows = [
                json.loads(line) for line in transport.ledger_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["status"] for row in rows], ["error", "ok"])
            self.assertEqual(sum(row["cost_usd"] for row in rows), 0.015)
            self.assertEqual(rows[0]["job_id"], "job-1")
            self.assertNotIn("SECRET PROMPT TEXT", transport.ledger_path.read_text(encoding="utf-8"))

    def test_transport_enforces_two_openrouter_semantic_review_posts(self) -> None:
        transport = LLMTransport()
        request = Request(
            "https://openrouter.example/api/v1/chat/completions", data=b"{}", method="POST",
        )
        response = _Response({"model": "critic", "usage": {"cost": 0.001}})
        with transport.scope(
            job_id="job-1", max_requests=10, max_cost_usd=0.10,
            max_openrouter_semantic_reviews=2,
        ):
            with transport.stage("semantic_review"):
                transport.request_json(
                    request, timeout=30, provider="openrouter", requested_model="critic",
                    opener=lambda *_args, **_kwargs: response,
                )
                transport.request_json(
                    request, timeout=30, provider="openrouter", requested_model="critic",
                    opener=lambda *_args, **_kwargs: response,
                )
                with self.assertRaises(LLMBudgetExceeded):
                    transport.request_json(
                        request, timeout=30, provider="openrouter", requested_model="critic",
                        opener=lambda *_args, **_kwargs: response,
                    )

    def test_kimi3_alias_uses_monthly_coding_plan_endpoint(self) -> None:
        with patch.dict(os.environ, {"KIMI_CODE_API": "test-plan-key"}, clear=True):
            settings = LLMSettings.from_environment("kimi", "kimi/kimi3")

        self.assertEqual(settings.provider, "kimi")
        self.assertEqual(settings.model, "k3")
        self.assertEqual(settings.base_url, "https://api.kimi.com/coding/v1")
        self.assertEqual(settings.api_key, "test-plan-key")
        self.assertEqual(settings.reasoning_effort, "high")

    def test_kimi3_request_uses_supported_temperature_and_reasoning_effort(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "kimi", "https://api.kimi.com/coding/v1", "test-key", "k3",
        ))
        response = _Response({
            "model": "k3",
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
            "usage": {},
        })
        with patch("video_factory.llm.urlopen", return_value=response) as opened:
            writer._request_json([{"role": "user", "content": "return json"}], 100)

        payload = json.loads(opened.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["temperature"], 1)
        self.assertEqual(payload["reasoning_effort"], "high")

    def test_model_numeric_fields_ignore_trailing_sentence_punctuation(self) -> None:
        self.assertEqual(_coerce_model_float("0.605.", 3.0), 0.605)
        self.assertEqual(_coerce_model_float("about 2.8 seconds", 3.0), 2.8)
        self.assertEqual(_coerce_model_float("unknown", 3.0), 3.0)

    def test_visible_copy_review_returns_structured_semantic_issues(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "cheap-model",
        ))
        candidate = Candidate(
            "tweet-1", SourceType.TWEET, "https://x.com/vendor/status/1", "Agent update",
        )
        evidence = Evidence(
            "evidence-1", candidate.id, candidate.source_url,
            "A named actor performed a concrete action.", "x:thread_post",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.PRACTICE_POST, ContentType.FLASH, 12,
        )
        issue = {
            "field_path": "editorial_brief.fixed_conclusion",
            "category": "natural_chinese", "problem": "reads like literal translation",
            "evidence_ids": [evidence.id], "repair_instruction": "rewrite in natural Chinese",
        }
        reviews = [
            {
                **issue, "verdict": "fail", "actor_action_object_recipient": "",
                "certainty": "inference", "naturalness_score": 2,
            },
            {
                "field_path": "editorial_brief.evidence_shots[0].fact", "verdict": "pass",
                "actor_action_object_recipient": "actor did action", "certainty": "fact",
                "naturalness_score": 4, "evidence_ids": [evidence.id], "category": "none",
                "problem": "", "repair_instruction": "",
            },
        ]
        with patch.object(
            writer, "_request_json",
            return_value=({"approved": False, "field_reviews": reviews}, {"model": "critic"}),
        ) as requested:
            issues, provenance = writer.review_visible_copy(packet, {
                "editorial_brief": {
                    "fixed_conclusion": "抽象结论", "evidence_shots": [{
                        "id": "shot-1", "fact": "抽象事实", "evidence_ids": [evidence.id],
                    }],
                },
            })
        self.assertEqual(issues, [issue])
        self.assertEqual(provenance["model"], "critic")
        critic_prompt = requested.call_args.args[0][-1]["content"]
        self.assertIn("Apply a read-aloud speech test", critic_prompt)
        self.assertIn("did not discuss or message each other", critic_prompt)

    def test_copy_critic_receives_late_page_target_context(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "critic",
        ))
        candidate = Candidate("waymo", SourceType.WEB, "https://waymo.example/lessons", "Lessons")
        target = "Pure end-to-end neural architectures risk black box failures."
        evidence = Evidence(
            "page", candidate.id, candidate.source_url,
            "Opening material. " + ("x" * 5000) + target + " More source text.",
            "web:primary_page",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.RESEARCH_OR_BENCHMARK, ContentType.FLASH, 20,
        )
        reviews = [{
            "field_path": "editorial_brief.evidence_shots[0].fact", "verdict": "pass",
            "naturalness_score": 5, "attention_score": 0, "evidence_ids": [evidence.id],
            "category": "none", "problem": "", "repair_instruction": "",
        }, {
            "field_path": "editorial_brief.evidence_shots[0].target", "verdict": "pass",
            "naturalness_score": 5, "attention_score": 0, "evidence_ids": [evidence.id],
            "category": "none", "problem": "", "repair_instruction": "",
        }]
        with patch.object(writer, "_request_json", return_value=(
            {"approved": True, "field_reviews": reviews}, {"model": "critic"},
        )) as requested:
            issues, _ = writer.review_visible_copy(packet, {"editorial_brief": {
                "evidence_shots": [{
                    "id": "shot-1", "fact": "Waymo says black boxes can fail",
                    "target": target, "evidence_ids": [evidence.id],
                }],
            }})

        self.assertEqual(issues, [])
        prompt = requested.call_args.args[0][-1]["content"]
        self.assertIn(target, prompt)
        self.assertIn("cited target context", prompt)

    def test_spoken_chinese_reviewer_is_copy_only_not_a_second_director(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "deepseek", "https://deepseek.example/v1", "test-key", "deepseek-chat",
        ))
        candidate = Candidate(
            "paper-1", SourceType.PAPER, "https://arxiv.org/pdf/1", "Paper",
        )
        evidence = Evidence(
            "paper", candidate.id, candidate.source_url,
            "Agents sustain high prices without communication.", "paper:pdf",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.RESEARCH_OR_BENCHMARK, ContentType.FLASH, 12,
        )

        def review(messages, max_tokens):
            prompt = messages[-1]["content"]
            fields = json.loads(prompt.split("Fields: ", 1)[1].split("\nEvidence:", 1)[0])
            rows = [{
                "field_path": path, "verdict": "pass", "naturalness_score": 5,
                "problem": "", "repair_instruction": "", "evidence_ids": [evidence.id],
            } for path in fields]
            return {"field_reviews": rows}, {"model": "deepseek-chat"}

        with patch.object(writer, "_request_json", side_effect=review) as requested:
            issues, _ = writer.review_spoken_chinese(packet, {
                "editorial_brief": {
                    "headline": "AI 没有交流，却一起维持高价",
                    "subheadline": "电力市场实验",
                    "fixed_conclusion": "只看通信记录，可能发现不了这种抬价",
                    "attention_strategy": {"selected_hook": "AI 没商量，为什么还会一起抬价？"},
                    "evidence_shots": [{
                        "fact": "几个 AI 各自出价，却慢慢学会一起维持高价",
                        "translation": "它们没有收到合谋指令，也没有互相通信",
                    }],
                },
            })

        self.assertEqual(issues, [])
        prompt = requested.call_args.args[0][-1]["content"]
        self.assertIn("Do not review story choice, directing", prompt)
        self.assertIn("Do not request a chart", prompt)
        self.assertIn("as its grammatical actor", prompt)

    def test_spoken_chinese_review_cannot_approve_unexplained_tacit_collusion_label(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "deepseek", "https://deepseek.example/v1", "test-key", "deepseek-chat",
        ))
        candidate = Candidate(
            "paper-1", SourceType.PAPER, "https://arxiv.org/pdf/1", "Paper",
        )
        evidence = Evidence(
            "paper", candidate.id, candidate.source_url,
            "Agents learned tacit collusion and sustained high prices without communicating.",
            "paper:pdf",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.RESEARCH_OR_BENCHMARK, ContentType.FLASH, 12,
        )

        def approve_everything(messages, max_tokens):
            prompt = messages[-1]["content"]
            fields = json.loads(prompt.split("Fields: ", 1)[1].split("\nEvidence:", 1)[0])
            return {"field_reviews": [{
                "field_path": path, "verdict": "pass", "naturalness_score": 5,
                "problem": "", "repair_instruction": "", "evidence_ids": [evidence.id],
            } for path in fields]}, {"model": "deepseek-chat"}

        with patch.object(writer, "_request_json", side_effect=approve_everything):
            issues, _ = writer.review_spoken_chinese(packet, {
                "editorial_brief": {
                    "headline": "AI自己学会合谋抬价",
                    "subheadline": "电力市场强化学习实验",
                    "fixed_conclusion": "只查通信记录，可能抓不到这种抬价",
                    "attention_strategy": {"selected_hook": "AI自己学会合谋抬价"},
                    "evidence_shots": [{
                        "fact": "它们没商量、也没互发消息，却慢慢学会一起维持高价",
                        "translation": "这种现象叫默契合谋",
                    }],
                },
            })

        failed_paths = {item["field_path"] for item in issues}
        self.assertEqual(failed_paths, {
            "editorial_brief.headline",
            "editorial_brief.attention_strategy.selected_hook",
        })

    def test_semantic_copy_repair_cannot_rewrite_clean_targets_or_other_shots(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/v1", "test-key", "writer",
        ))
        candidate = Candidate("paper-1", SourceType.PAPER, "https://example.com/paper", "Paper")
        evidence = Evidence(
            "paper", candidate.id, candidate.source_url, "Exact source sentence.", "paper:pdf",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.RESEARCH_OR_BENCHMARK, ContentType.FLASH, 12,
        )
        draft = {"editorial_brief": {
            "headline": "Clean headline", "subheadline": "Stiff subtitle",
            "fixed_conclusion": "Clean conclusion",
            "attention_strategy": {
                "hook_fact": "clean", "conflict": "", "surprise": "", "stakes": "clean",
                "stance": "clean", "payoff": "clean", "hook_candidates": ["Clean headline"] * 3,
                "hook_evidence_ids": [evidence.id], "selected_hook": "Clean headline",
            },
            "evidence_shots": [{
                "id": "shot-1", "question": "q", "fact": "Clean fact",
                "interpretation": "internal", "audience_copy": "Clean copy",
                "target": "Exact source sentence.", "translation": "Clean translation",
                "full_translation": "", "relation_to_previous": "start",
                "evidence_ids": [evidence.id], "beat_ids": ["proof"],
            }],
        }}
        # A real sparse repair is allowed to return only the rejected field.
        model_patch = {"subheadline": "Natural subtitle"}
        error = (
            'semantic copy critic issues: [{"field_path":"editorial_brief.subheadline",'
            '"category":"natural_chinese","problem":"stiff"}]'
        )

        with (
            patch.object(writer, "_request_json", return_value=(model_patch, {"model": "writer"})),
            patch.object(writer, "_to_storyboard_request", return_value=object()),
        ):
            _, _, repaired = writer._repair_editorial_copy(packet, draft, error)

        brief = repaired["editorial_brief"]
        self.assertEqual(brief["subheadline"], "Natural subtitle")
        self.assertEqual(brief["headline"], "Clean headline")
        self.assertEqual(brief["fixed_conclusion"], "Clean conclusion")
        self.assertEqual(brief["evidence_shots"][0]["target"], "Exact source sentence.")
        self.assertEqual(brief["evidence_shots"][0]["fact"], "Clean fact")
        self.assertEqual(brief["attention_strategy"]["selected_hook"], "Clean headline")

    def test_holistic_story_review_rejects_a_grounded_but_wrong_story_axis(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "critic",
        ))
        candidate = Candidate(
            "tweet-jeff", SourceType.TWEET, "https://x.com/example/status/1",
            "Google infrastructure leaders leave to start a company",
        )
        evidence = Evidence(
            "evidence-1", candidate.id, candidate.source_url,
            "Four long-time Google infrastructure leaders founded Discovery Loop.",
            "x:thread_post",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.COMPANY_OR_TEAM, ContentType.FLASH, 15,
        )
        field_review = {
            "field_path": "editorial_brief.headline", "verdict": "pass",
            "actor_action_object_recipient": "Discovery Loop automates research",
            "certainty": "fact", "naturalness_score": 5, "attention_score": 4,
            "evidence_ids": [evidence.id], "category": "none", "problem": "",
            "repair_instruction": "",
        }
        story_review = {
            "verdict": "fail",
            "primary_promise": "Google loses four infrastructure leaders",
            "draft_axis": "Discovery Loop's experiment automation workflow",
            "narrative_tension_score": 2, "audience_relevance_score": 3,
            "chronology_score": 4, "evidence_hierarchy_score": 2,
            "decision_value_score": 4, "pacing_score": 4,
            "failure_modes": [{
                "category": "story_axis",
                "problem": "the product mechanism replaces the Google talent-loss event",
                "repair_instruction": "lead with Google's loss, then use the product as payoff",
                "evidence_ids": [evidence.id],
            }],
        }
        with patch.object(writer, "_request_json", return_value=(
            {"approved": False, "story_review": story_review, "field_reviews": [field_review]},
            {"model": "critic"},
        )):
            issues, provenance = writer.review_visible_copy(packet, {
                "editorial_brief": {"headline": "Discovery Loop自动化科研循环"},
            })

        self.assertEqual(issues[0]["category"], "story_axis")
        self.assertEqual(issues[0]["field_path"], "editorial_brief.director_brief")
        self.assertEqual(provenance["narrative_score"], 2)
        self.assertEqual(provenance["pacing_score"], 4)

    def test_holistic_story_failure_uses_full_rebuild_not_copy_patch(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "critic",
        ))
        candidate = Candidate(
            "tweet-1", SourceType.TWEET, "https://x.com/vendor/status/1", "event",
        )
        evidence = Evidence(
            "evidence-1", candidate.id, candidate.source_url, "grounded event", "x:thread_post",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.COMPANY_OR_TEAM, ContentType.FLASH, 15,
        )
        error = (
            'semantic copy critic issues: [{"field_path":"editorial_brief.director_brief",'
            '"category":"story_axis","problem":"wrong axis"}]'
        )
        sentinel = (object(), {"model": "writer"}, {"editorial_brief": {}})
        with patch.object(writer, "_generate_from_messages", return_value=sentinel) as rebuild, patch.object(
            writer, "_repair_editorial_copy",
        ) as copy_patch:
            result = writer.repair(packet, {"editorial_brief": {}}, error)

        self.assertIs(result, sentinel)
        rebuild.assert_called_once()
        copy_patch.assert_not_called()

    def test_editorial_value_failure_uses_full_rebuild_not_copy_patch(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "critic",
        ))
        candidate = Candidate(
            "web-1", SourceType.WEB, "https://example.com/news", "announcement",
        )
        evidence = Evidence(
            "evidence-1", candidate.id, candidate.source_url, "grounded event", "web:primary_page",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.OFFICIAL_ANNOUNCEMENT, ContentType.FLASH, 15,
        )
        error = (
            'semantic copy critic issues: [{"field_path":"editorial_brief.director_brief",'
            '"category":"editorial_value","problem":"source paraphrase only"}]'
        )
        sentinel = (object(), {"model": "writer"}, {"editorial_brief": {}})
        with patch.object(writer, "_generate_from_messages", return_value=sentinel) as rebuild, patch.object(
            writer, "_repair_editorial_copy",
        ) as copy_patch:
            result = writer.repair(packet, {"editorial_brief": {}}, error)

        self.assertIs(result, sentinel)
        rebuild.assert_called_once()
        copy_patch.assert_not_called()

    def test_locked_editorial_opportunity_requires_holistic_review(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "critic",
        ))
        candidate = Candidate(
            "tweet-1", SourceType.TWEET, "https://x.com/vendor/status/1", "event",
        )
        evidence = Evidence(
            "evidence-1", candidate.id, candidate.source_url, "grounded event", "x:thread_post",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.COMPANY_OR_TEAM, ContentType.FLASH, 15,
            opportunity=EditorialOpportunity(
                "Google loses a core team", "current", "developer relevance", "talent shift",
                [SelectionReason("primary", "company_stake", "Google loses the team", [evidence.id])],
                story_archetype="people_change",
            ),
        )
        field_review = {
            "field_path": "editorial_brief.headline", "verdict": "pass",
            "naturalness_score": 5, "attention_score": 4, "evidence_ids": [evidence.id],
            "category": "none", "problem": "", "repair_instruction": "",
        }
        with patch.object(writer, "_request_json", return_value=(
            {"approved": True, "field_reviews": [field_review]}, {"model": "critic"},
        )):
            issues, _ = writer.review_visible_copy(packet, {
                "editorial_brief": {"headline": "Google核心团队离开"},
            })

        self.assertEqual(issues[0]["category"], "story_axis")
        self.assertIn("omitted", issues[0]["problem"])

    def test_visible_copy_review_uses_field_verdicts_when_summary_boolean_disagrees(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "cheap-model",
        ))
        candidate = Candidate(
            "tweet-1", SourceType.TWEET, "https://x.com/vendor/status/1", "Agent update",
        )
        evidence = Evidence(
            "evidence-1", candidate.id, candidate.source_url,
            "A named actor performed a concrete action.", "x:thread_post",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.PRACTICE_POST, ContentType.FLASH, 12,
        )
        reviews = [{
            "field_path": "editorial_brief.headline", "verdict": "pass",
            "actor_action_object_recipient": "actor did action", "certainty": "fact",
            "naturalness_score": 5, "attention_score": 4,
            "evidence_ids": [evidence.id], "category": "none",
            "problem": "", "repair_instruction": "",
        }]
        with patch.object(writer, "_request_json", return_value=(
            {"approved": False, "field_reviews": reviews}, {"model": "critic"},
        )):
            issues, _ = writer.review_visible_copy(packet, {
                "editorial_brief": {"headline": "具体事件标题"},
            })
        self.assertEqual(issues, [])

    def test_visible_copy_review_allows_hook_to_keep_the_headline_story_axis(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "cheap-model",
        ))
        candidate = Candidate(
            "tweet-1", SourceType.TWEET, "https://x.com/vendor/status/1", "Agent update",
        )
        evidence = Evidence(
            "evidence-1", candidate.id, candidate.source_url,
            "A research team disclosed an API issue that exposes a hidden trace.", "x:thread_post",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.PRACTICE_POST, ContentType.FLASH, 12,
        )
        paths = [
            "editorial_brief.headline",
            "editorial_brief.attention_strategy.selected_hook",
        ]
        reviews = [{
            "field_path": path, "verdict": "pass",
            "actor_action_object_recipient": "team disclosed issue", "certainty": "fact",
            "naturalness_score": 5, "attention_score": 4,
            "evidence_ids": [evidence.id], "category": "none",
            "problem": "", "repair_instruction": "",
        } for path in paths]
        with patch.object(writer, "_request_json", return_value=(
            {"approved": True, "field_reviews": reviews}, {"model": "critic"},
        )):
            issues, _ = writer.review_visible_copy(packet, {
                "editorial_brief": {
                    "headline": "研究团队披露模型 API 漏洞，可提取隐藏思维链",
                    "attention_strategy": {
                        "selected_hook": "研究团队披露大模型 API 漏洞：隐藏思维链可被完整提取",
                    },
                },
            })
        self.assertEqual(issues, [])

    def test_visible_copy_review_does_not_demand_harness_definition_in_every_field(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "critic",
        ))
        candidate = Candidate(
            "tweet-1", SourceType.TWEET, "https://x.com/vendor/status/1", "Harness dispute",
        )
        evidence = Evidence(
            "evidence-1", candidate.id, candidate.source_url,
            "A user used the Anthropic harness with another model and appealed a suspension.",
            "x:thread_post",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.PRACTICE_POST, ContentType.FLASH, 12,
        )
        fields = [
            "editorial_brief.headline",
            "editorial_brief.attention_strategy.selected_hook",
        ]
        reviews = [{
            "field_path": path, "verdict": "fail",
            "actor_action_object_recipient": "user used harness", "certainty": "reported_claim",
            "naturalness_score": 4, "attention_score": 4,
            "evidence_ids": [evidence.id], "category": "technical_specificity",
            "problem": "harness 未解释", "repair_instruction": "重复写成测试框架（harness）",
        } for path in fields]
        with patch.object(writer, "_request_json", return_value=(
            {"approved": False, "field_reviews": reviews}, {"model": "critic"},
        )):
            issues, _ = writer.review_visible_copy(packet, {
                "editorial_brief": {
                    "headline": "Anthropic harness 跨模型封号争议",
                    "attention_strategy": {
                        "selected_hook": "用户称 Anthropic harness 接其他模型后被封",
                    },
                },
            })
        self.assertEqual(issues, [])

    def test_visible_copy_review_includes_browser_target_as_proof(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "cheap-model",
        ))
        candidate = Candidate(
            "web-1", SourceType.OFFICIAL_ANNOUNCEMENT,
            "https://example.com/news", "Funding news",
        )
        evidence = Evidence(
            "evidence-1", candidate.id, candidate.source_url,
            "The company raised $21 million. It currently has negative gross margins.", "web:page",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.COMPANY_OR_TEAM, ContentType.FLASH, 12,
        )
        observed_paths = set()

        def review(messages, max_tokens):
            nonlocal observed_paths
            fields_line = next(
                line for line in messages[-1]["content"].splitlines() if line.startswith("Fields: ")
            )
            observed_paths = set(json.loads(fields_line.removeprefix("Fields: ")))
            rows = [{
                "field_path": path, "verdict": "pass",
                "actor_action_object_recipient": "", "certainty": "fact",
                "naturalness_score": 5, "attention_score": 0,
                "evidence_ids": [evidence.id], "category": "none",
                "problem": "", "repair_instruction": "",
            } for path in observed_paths]
            return {"approved": True, "field_reviews": rows}, {"model": "critic"}

        with patch.object(writer, "_request_json", side_effect=review):
            issues, _ = writer.review_visible_copy(packet, {
                "editorial_brief": {"evidence_shots": [{
                    "id": "shot-1", "fact": "公司融资2100万美元",
                    "target": "The company raised $21 million.",
                    "evidence_ids": [evidence.id],
                }]},
            })

        self.assertEqual(issues, [])
        self.assertIn("editorial_brief.evidence_shots[0].target", observed_paths)

    def test_github_review_expands_every_rendered_hook_and_translation_field(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "cheap-model",
        ))
        candidate = Candidate(
            "github-acme-tool", SourceType.GITHUB, "https://github.com/acme/tool", "acme/tool",
        )
        evidence = Evidence(
            "readme", candidate.id, candidate.source_url,
            "Free to use assets. Run tool --topic demo.", "github:readme",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.GITHUB_PROJECT, ContentType.EXPLAINER, 20,
        )
        observed_paths = set()

        def review(messages, max_tokens):
            nonlocal observed_paths
            fields_line = next(
                line for line in messages[-1]["content"].splitlines() if line.startswith("Fields: ")
            )
            observed_paths = set(json.loads(fields_line.removeprefix("Fields: ")))
            rows = [{
                "field_path": "fixed_conclusion" if path == "github_brief.footer" else path,
                "verdict": "pass",
                "actor_action_object_recipient": "", "certainty": "fact",
                "naturalness_score": 5,
                "attention_score": 3 if path == "github_brief.hook_verdict" else 4,
                "evidence_ids": [evidence.id], "category": "none",
                "problem": "", "repair_instruction": "",
            } for path in observed_paths]
            return {"approved": True, "field_reviews": rows}, {"model": "critic"}

        draft = {
            "footer": "工具把主题变成成片",
            "github_brief": {
                "hook_opening": "acme/tool 开始自动做视频",
                "hook_reveal": "输入主题就能运行完整流程",
                "hook_verdict": "这条工作流值得直接试",
                "project_title": "acme/tool｜主题生成视频",
                "hook_evidence_ids": [evidence.id],
                "repo_description_translation": "输入主题生成视频",
                "readme_claim_translation": "一条命令运行流程",
                "selected_focus_ids": ["trial"],
                "focus_candidates": [{
                    "id": "trial", "translation": "运行命令生成视频",
                    "browser_translation": "命令行直接生成视频",
                    "evidence_ids": [evidence.id],
                }],
            },
        }
        with patch.object(writer, "_request_json", side_effect=review):
            issues, _ = writer.review_visible_copy(packet, draft)
        self.assertEqual(issues, [])
        self.assertEqual(observed_paths, {
            "github_brief.hook_opening", "github_brief.hook_reveal",
            "github_brief.hook_verdict", "github_brief.project_title",
            "github_brief.footer", "github_brief.repo_description_translation",
            "github_brief.readme_claim_translation",
            "github_brief.focus_candidates[0].browser_translation",
        })

    def test_visible_copy_review_fails_closed_without_hidden_coverage_retry(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "cheap-model",
        ))
        candidate = Candidate(
            "tweet-1", SourceType.TWEET, "https://x.com/vendor/status/1", "Update",
        )
        evidence = Evidence(
            "evidence-1", candidate.id, candidate.source_url,
            "Vendor shipped a concrete update.", "x:thread_post",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.PRACTICE_POST, ContentType.FLASH, 12,
        )
        complete = [{
            "field_path": path, "verdict": "pass",
            "actor_action_object_recipient": "vendor shipped update", "certainty": "fact",
            "naturalness_score": 5, "attention_score": 4,
            "evidence_ids": [evidence.id], "category": "none",
            "problem": "", "repair_instruction": "",
        } for path in (
            "editorial_brief.headline", "editorial_brief.fixed_conclusion",
        )]
        with patch.object(
            writer, "_request_json",
            return_value=({"approved": True, "field_reviews": complete[:1]}, {"model": "critic"}),
        ) as requested:
            issues, provenance = writer.review_visible_copy(packet, {
                "editorial_brief": {
                    "headline": "厂商交付具体更新",
                    "fixed_conclusion": "这项更新已经可以使用",
                },
            })
        self.assertEqual(requested.call_count, 1)
        self.assertEqual(provenance["missing_fields"], ["editorial_brief.fixed_conclusion"])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["category"], "review_coverage")

    def test_visible_copy_review_accepts_short_evidence_shot_field_aliases(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "cheap-model",
        ))
        candidate = Candidate(
            "tweet-1", SourceType.TWEET, "https://x.com/vendor/status/1", "Update",
        )
        evidence = Evidence(
            "evidence-1", candidate.id, candidate.source_url,
            "Vendor shipped a concrete update.", "x:thread_post",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.PRACTICE_POST, ContentType.FLASH, 12,
        )

        def review(messages, max_tokens):
            fields_line = next(
                line for line in messages[-1]["content"].splitlines() if line.startswith("Fields: ")
            )
            paths = json.loads(fields_line.removeprefix("Fields: "))
            rows = []
            for path in paths:
                if path.endswith(".translation"):
                    returned_path = "shot-1.translation"
                else:
                    returned_path = path.removeprefix("editorial_brief.") if "evidence_shots[" in path else path
                rows.append({
                    "field_path": returned_path, "verdict": "pass",
                    "actor_action_object_recipient": "", "certainty": "fact",
                    "naturalness_score": 5, "attention_score": 4,
                    "evidence_ids": [evidence.id], "category": "none",
                    "problem": "", "repair_instruction": "",
                })
            return {"approved": True, "field_reviews": rows}, {"model": "critic"}

        with patch.object(writer, "_request_json", side_effect=review) as requested:
            issues, _ = writer.review_visible_copy(packet, {
                "editorial_brief": {"evidence_shots": [{
                    "id": "shot-1", "fact": "厂商交付具体更新",
                    "translation": "厂商已经交付这项更新", "evidence_ids": [evidence.id],
                }]},
            })

        self.assertEqual(issues, [])
        self.assertEqual(requested.call_count, 1)

    def test_http_200_invalid_json_is_retried(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "cheap-model",
        ))
        invalid = _Response({
            "choices": [{"message": {"content": "provider overloaded"}, "finish_reason": "error"}],
        })
        valid = _Response({
            "model": "cheap-model", "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
            "usage": {},
        })
        with patch("video_factory.llm.urlopen", side_effect=[invalid, valid]) as opened, patch("video_factory.llm.time.sleep"):
            draft, _ = writer._request_json([{"role": "user", "content": "return json"}], 100)
        self.assertEqual(draft, {"ok": True})
        self.assertEqual(opened.call_count, 2)

    def test_transport_stops_after_two_bounded_attempts(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "cheap-model",
            timeout_seconds=17,
        ))
        invalid = _Response({
            "choices": [{"message": {"content": "provider overloaded"}, "finish_reason": "error"}],
        })
        with (
            patch("video_factory.llm.urlopen", side_effect=[invalid, invalid, AssertionError("third attempt")]) as opened,
            patch("video_factory.llm.time.sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "after 2 attempts"),
        ):
            writer._request_json([{"role": "user", "content": "return json"}], 100)

        self.assertEqual(opened.call_count, 2)
        self.assertEqual([call.kwargs["timeout"] for call in opened.call_args_list], [17, 17])
        sleep.assert_called_once_with(0.6)

    def test_http_429_uses_retry_after_and_records_recovery(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "cheap-model",
        ))
        limited = HTTPError(
            "https://openrouter.example/api/v1/chat/completions", 429,
            "Too Many Requests", {"Retry-After": "3"}, None,
        )
        valid = _Response({
            "model": "fallback-model",
            "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
            "usage": {},
        })

        with (
            patch("video_factory.llm.urlopen", side_effect=[limited, valid]) as opened,
            patch("video_factory.llm.time.sleep") as sleep,
        ):
            draft, provenance = writer._request_json(
                [{"role": "user", "content": "return json"}], 100,
            )

        self.assertEqual(draft, {"ok": True})
        self.assertEqual(opened.call_count, 2)
        sleep.assert_called_once_with(3.0)
        self.assertEqual(provenance["request_attempts"], 2)
        self.assertEqual(provenance["retry_events"][0]["reason"], "HTTP 429")

    def test_transport_exhaustion_falls_back_to_secondary_provider(self) -> None:
        primary = OpenAICompatibleStoryWriter(LLMSettings(
            "openrouter", "https://openrouter.example/api/v1", "test-key", "cheap-model",
        ))
        fallback = OpenAICompatibleStoryWriter(LLMSettings(
            "deepseek", "https://deepseek.example/v1", "test-key", "deepseek-chat",
        ))
        writer = TransportFallbackStoryWriter(primary, fallback)

        with (
            patch.object(
                primary, "_request_json",
                side_effect=RuntimeError("openrouter story request failed after 4 attempts: HTTP 429"),
            ),
            patch.object(
                fallback, "_request_json",
                return_value=({"ok": True}, {"provider": "deepseek", "model": "deepseek-chat"}),
            ) as used_fallback,
        ):
            draft, provenance = writer._request_json(
                [{"role": "user", "content": "return json"}], 100,
            )

        self.assertEqual(draft, {"ok": True})
        used_fallback.assert_called_once()
        self.assertEqual(provenance["transport_fallback"]["from_provider"], "openrouter")
        self.assertEqual(provenance["transport_fallback"]["to_provider"], "deepseek")

        with (
            patch.object(primary, "_request_json") as skipped_primary,
            patch.object(
                fallback, "_request_json",
                return_value=({"second": True}, {"provider": "deepseek", "model": "deepseek-chat"}),
            ),
        ):
            second, second_provenance = writer._request_json(
                [{"role": "user", "content": "return another json"}], 100,
            )

        skipped_primary.assert_not_called()
        self.assertEqual(second, {"second": True})
        self.assertTrue(second_provenance["transport_fallback"]["circuit_open"])

    def test_final_unknown_shot_uses_bounded_editorial_copy_repair(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "deepseek", "https://example.invalid", "test-key", "test-model",
        ))
        candidate = Candidate(
            "tweet-1", SourceType.TWEET, "https://x.com/vendor/status/1", "Agent update",
        )
        evidence = Evidence(
            "evidence-1", candidate.id, candidate.source_url,
            "The agent directory is available under Apache 2.0.", "x:thread_post",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.TOOL_SDK_AGENT, ContentType.EXPLAINER, 28,
        )
        sentinel = (object(), {"repair": "bounded"}, {"editorial_brief": {}})
        error = (
            "invalid editorial brief: final changing shot must end on "
            "verified capability/impact, not an unknown or ritual caution"
        )

        with patch.object(writer, "_repair_editorial_copy", return_value=sentinel) as repair:
            result = writer.repair(packet, {"editorial_brief": {}}, error)

        self.assertIs(result, sentinel)
        repair.assert_called_once_with(packet, {"editorial_brief": {}}, error)

    def test_unsupported_release_wording_uses_bounded_editorial_copy_repair(self) -> None:
        writer = OpenAICompatibleStoryWriter(LLMSettings(
            "deepseek", "https://example.invalid", "test-key", "test-model",
        ))
        candidate = Candidate(
            "web-1", SourceType.WEB, "https://vendor.example/docs", "Agent docs",
        )
        evidence = Evidence(
            "evidence-1", candidate.id, candidate.source_url,
            "The agent directory is available under Apache 2.0.", "web:page",
        )
        packet = StoryWriterPacket(
            candidate, [evidence], TopicType.TOOL_SDK_AGENT, ContentType.EXPLAINER, 28,
        )
        sentinel = (object(), {"repair": "bounded"}, {"editorial_brief": {}})
        error = (
            "invalid editorial brief: release/launch wording needs explicit release evidence; "
            "documentation alone proves availability and capability"
        )

        with patch.object(writer, "_repair_editorial_copy", return_value=sentinel) as repair:
            result = writer.repair(packet, {"editorial_brief": {}}, error)

        self.assertIs(result, sentinel)
        repair.assert_called_once_with(packet, {"editorial_brief": {}}, error)


if __name__ == "__main__":
    unittest.main()
