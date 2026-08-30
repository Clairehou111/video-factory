import json
import unittest
from unittest.mock import patch

from video_factory.llm import LLMSettings, OpenAICompatibleStoryWriter, _coerce_model_float
from video_factory.models import Candidate, ContentType, Evidence, SourceType, TopicType
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
        with patch.object(writer, "_request_json", return_value=({"approved": False, "field_reviews": reviews}, {"model": "critic"})):
            issues, provenance = writer.review_visible_copy(packet, {
                "editorial_brief": {
                    "fixed_conclusion": "抽象结论", "evidence_shots": [{
                        "id": "shot-1", "fact": "抽象事实", "evidence_ids": [evidence.id],
                    }],
                },
            })
        self.assertEqual(issues, [issue])
        self.assertEqual(provenance["model"], "critic")

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

    def test_visible_copy_review_rejects_hook_that_only_rephrases_headline(self) -> None:
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
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["category"], "retention_hook")

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

    def test_visible_copy_review_retries_when_critic_omits_a_field(self) -> None:
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
        with patch.object(writer, "_request_json", side_effect=[
            ({"approved": True, "field_reviews": complete[:1]}, {"model": "critic"}),
            ({"approved": True, "field_reviews": complete}, {"model": "critic"}),
        ]) as requested:
            issues, provenance = writer.review_visible_copy(packet, {
                "editorial_brief": {
                    "headline": "厂商交付具体更新",
                    "fixed_conclusion": "这项更新已经可以使用",
                },
            })
        self.assertEqual(issues, [])
        self.assertEqual(requested.call_count, 2)
        self.assertTrue(provenance["review_retried"])

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
