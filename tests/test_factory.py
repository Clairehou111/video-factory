from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from video_factory.agent import ContentAgentError
from video_factory.editorial import repair_fragmented_radar_hook
from video_factory.factory import (
    CompositeCopyReviewer, GenerateOptions, VideoFactory,
    _browser_capture_needs_card_fallback, static_radar_target_duration,
)
from video_factory.models import (
    AttentionStrategy, Candidate, ColdOpenBeat, ContentType, EditorialBrief, Evidence, EvidenceShot,
    EvidenceShotKind, InformationRenderProfile, MaterialRole, RenderManifest, Scene,
    SourceType,
)
from video_factory.multimodal import VisualCandidate
from video_factory.serde import load_manifest
from video_factory.storage import Workspace


def basic_manifest() -> RenderManifest:
    evidence = Evidence("e-1", "candidate-1", "https://github.com/acme/demo", "claim", "github:readme")
    return RenderManifest(
        id="render-1", candidate_id="candidate-1", content_type=ContentType.EXPLAINER,
        scenes=[Scene(
            "scene-1", 0, 20, "claim", "claim", [evidence.id],
            MaterialRole.PROOF, "show source",
        )],
        evidence=[evidence], source_urls=[evidence.url], fixed_hook="hook", fixed_footer="conclusion",
    )


class VideoFactoryTest(unittest.TestCase):
    def test_github_visual_analysis_failure_is_optional(self) -> None:
        with TemporaryDirectory() as temp:
            factory = VideoFactory(Workspace(Path(temp) / "workspace"))
            candidate = Candidate(
                "github-acme-demo", SourceType.GITHUB,
                "https://github.com/acme/demo", "acme/demo", "acme",
            )
            evidence = Evidence(
                "github-acme-demo-readme", candidate.id, candidate.source_url,
                "A grounded README with an install path, API example, input, output, and benchmark.",
                "github:readme",
            )
            ingest = SimpleNamespace(candidate=candidate, evidence=[evidence])
            writer = SimpleNamespace(settings=SimpleNamespace(model="story-model"))
            agent_run = SimpleNamespace(
                manifest=basic_manifest(), trace=[], llm_calls=1,
            )
            vision_quote = SimpleNamespace(
                model_id="vision-model", to_dict=lambda: {"model_id": "vision-model"},
            )
            result = {"stages": []}
            job = Path(temp) / "job"
            job.mkdir()

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
                patch.object(factory, "_github_json", return_value={"default_branch": "main"}),
                patch.object(factory, "_github_bytes", return_value=b"# Demo\n\nInstall and benchmark example."),
                patch("video_factory.factory.GitHubIngestor") as ingestor_class,
                patch("video_factory.factory.enrich_github_context", return_value=([], [], [])),
                patch("video_factory.factory.find_high_value_visuals", return_value=[
                    VisualCandidate("https://example.com/benchmark.png", "benchmark", "Benchmark"),
                ]),
                patch.object(factory, "_story_writer", return_value=(writer, None, {})),
                patch.object(factory, "_copy_reviewer", return_value=(writer, {})),
                patch.object(factory, "_active_agent_policy", return_value={}),
                patch("video_factory.factory.OpenRouterCatalog") as catalog_class,
                patch("video_factory.factory.LLMSettings.from_environment", return_value=MagicMock()),
                patch("video_factory.factory.OpenRouterVisualAnalyst") as analyst_class,
                patch("video_factory.factory.BoundedContentAgent") as agent_class,
            ):
                ingestor_class.return_value.ingest.return_value = ingest
                catalog_class.return_value.select.return_value = vision_quote
                analyst_class.return_value.analyze.side_effect = RuntimeError(
                    "OpenRouter vision model returned no JSON content"
                )
                agent_class.return_value.run.return_value = agent_run

                factory._generate_github(
                    candidate.source_url, job,
                    GenerateOptions(provider="openrouter", render=False), result,
                )

            stage = next(item for item in result["stages"] if item["name"] == "visual_analysis")
            self.assertEqual(stage["status"], "failed_optional")
            self.assertIn("returned no JSON content", stage["reason"])
            self.assertIn("manifest", result)

    def test_translation_model_alias_routes_to_kimi_coding_plan_before_openrouter(self) -> None:
        with TemporaryDirectory() as temp:
            factory = VideoFactory(Workspace(Path(temp) / "workspace"))
            with patch.dict(os.environ, {
                "KIMI_CODE_API": "test-plan-key",
                "OPENROUTER_API_KEY": "test-openrouter-key",
            }, clear=True):
                writer, selection = factory._translation_writer(
                    GenerateOptions(provider="auto", model="kimi/kimi3"),
                )

        self.assertEqual(writer.settings.provider, "kimi")
        self.assertEqual(writer.settings.model, "k3")
        self.assertEqual(writer.settings.reasoning_effort, "low")
        self.assertEqual(selection["billing"], "kimi_coding_plan")

    def test_youtube_subtitle_review_uses_independent_deepseek_for_kimi_translation(self) -> None:
        with TemporaryDirectory() as temp:
            factory = VideoFactory(Workspace(Path(temp) / "workspace"))
            writer = MagicMock()
            writer.settings.provider = "kimi"
            directing_writer = MagicMock()
            directing_writer.settings.provider = "openrouter"
            directing_writer.settings.model = "google/gemini-3.7-flash"
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
                reviewer, selection = factory._youtube_subtitle_reviewer(
                    writer, directing_writer,
                )

        self.assertEqual(reviewer.settings.provider, "deepseek")
        self.assertEqual(reviewer.settings.model, "deepseek-chat")
        self.assertEqual(selection["purpose"], "youtube_subtitle_fidelity_and_naturalness")
        self.assertIn("native-Chinese", selection["reason"])

    def test_static_news_radar_targets_eight_seconds_but_other_formats_keep_room(self) -> None:
        self.assertEqual(static_radar_target_duration(ContentType.FLASH, "news"), 8.5)
        self.assertEqual(static_radar_target_duration(ContentType.EXPLAINER, "news_zh"), 8.5)
        self.assertEqual(static_radar_target_duration(ContentType.FLASH, "x"), 14.0)
        self.assertEqual(static_radar_target_duration(ContentType.DEEP_DIVE, "news"), 20.0)

    def test_ready_hook_incumbent_survives_a_weaker_policy_regeneration(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            batch_dir = workspace.root / "publish" / "publish-story-candidate-1-golden"
            batch_dir.mkdir(parents=True)
            incumbent = "Smart都迁去中国生产了，这家西班牙公司偏要在欧洲造微型车"
            accepted_job = workspace.root / "jobs" / "accepted"
            accepted_job.mkdir(parents=True)
            accepted = basic_manifest()
            accepted.fixed_hook = incumbent
            accepted.fixed_title = incumbent
            (accepted_job / "manifest.json").write_text(
                json.dumps(accepted.to_dict(), ensure_ascii=False), encoding="utf-8",
            )
            (batch_dir / "batch.json").write_text(json.dumps({
                "manifest_id": "story-candidate-1",
                "state": "ready_for_review",
                "video_path": str(accepted_job / "final.mp4"),
                "targets": [{
                    "platform": "tencent", "title": incumbent[:18], "state": "pending",
                }],
            }), encoding="utf-8")
            manifest = basic_manifest()
            manifest.fixed_hook = "面对中国微型车竞争，西班牙公司推出新车"
            manifest.fixed_title = manifest.fixed_hook
            manifest.editorial_brief = EditorialBrief(
                manifest.fixed_hook, "", "欧洲公司尝试本地制造",
                AttentionStrategy(
                    "新车", "竞争", "认证", "市场", "本地制造", "开售",
                    [manifest.fixed_hook, "Liux获得认证", "欧洲微型车竞争"],
                    [manifest.evidence[0].id], manifest.fixed_hook,
                ), [], [], [], 12.0, opening_mode="conflict",
            )
            result: dict[str, object] = {"stages": []}

            VideoFactory(workspace)._retain_hook_incumbent(manifest, result)

            self.assertEqual(manifest.fixed_hook, incumbent)
            self.assertEqual(manifest.editorial_brief.attention_strategy.selected_hook, incumbent)
            self.assertEqual(result["stages"][0]["status"], "incumbent_retained")

    def test_missing_live_quote_requests_grounded_card_visual_fallback(self) -> None:
        with TemporaryDirectory() as temp:
            video = Path(temp) / "evidence-browser-1.mp4"
            video.with_suffix(".capture-repairs.json").write_text(json.dumps([{
                "kind": "missing_visible_text",
                "missing_target": "an archived quote no longer in the live DOM",
                "repair": "scroll to source-page bottom and hold without a false highlight",
            }]), encoding="utf-8")

            self.assertTrue(_browser_capture_needs_card_fallback(video))

    def test_non_missing_capture_repair_keeps_real_source_visual(self) -> None:
        with TemporaryDirectory() as temp:
            video = Path(temp) / "evidence-browser-1.mp4"
            video.with_suffix(".capture-repairs.json").write_text(json.dumps([{
                "kind": "unreadable_highlight",
                "repair": "hold the real source page without the tiny outline",
            }]), encoding="utf-8")

            self.assertFalse(_browser_capture_needs_card_fallback(video))

    def test_official_video_flash_becomes_one_clip_plus_two_readable_cards(self) -> None:
        page = Evidence(
            "page", "candidate-1", "https://vendor.example/robot", "Exact source proof",
            "web:primary_page",
        )
        source_video = Evidence(
            "video", "candidate-1", "https://youtube.com/watch?v=official", "Official video",
            "web:source_video", captured_asset="assets/video.mp4",
            metadata={"clip_start": 12.0, "clip_end": 17.0},
        )
        shots = [
            EvidenceShot(
                id=f"shot-{index}", kind=EvidenceShotKind.BROWSER_SECTION,
                question="发生了什么？", fact="机器人在真实房间完成抓取并自主恢复动作",
                interpretation="这是可直接看到的实体动作证据", evidence_ids=[page.id],
                beat_ids=["proof"], source_url=page.url, target="Exact source proof",
                translation="准确证据", duration=2.8, visual_family="official_page",
                audience_copy="真实动作发生在实体环境，不是仿真画面",
            )
            for index in range(1, 4)
        ]
        brief = EditorialBrief(
            "机器人开始执行真实任务", "一段原片加两张证据卡", "先看动作，再看价格与开源范围",
            AttentionStrategy(
                "实体动作", "真实任务对比玩具", "可以连续完成", "进入日常环境",
                "先看它做事", "再判断是否可用", ["机器人开始执行真实任务"] * 3,
                [page.id], "机器人开始执行真实任务",
            ), [], [], shots, 9.0, opening_mode="direct_fact",
        )
        manifest = RenderManifest(
            "render-video", "candidate-1", ContentType.FLASH, [], [page], [page.url],
            editorial_brief=brief, fixed_hook=brief.headline, fixed_title=brief.headline,
            fixed_footer=brief.fixed_conclusion,
            render_profile=InformationRenderProfile.RADAR_V2.value,
        )
        candidate = MagicMock(source_url=page.url)

        VideoFactory._canonicalize_source_video_visuals(
            manifest, candidate, source_video,
        )

        self.assertEqual(brief.evidence_shots[0].kind, EvidenceShotKind.VIDEO)
        self.assertEqual(
            [shot.visual_family for shot in brief.evidence_shots],
            ["source_video", "stat_card", "impact_card"],
        )
        self.assertLessEqual(sum(shot.duration for shot in brief.evidence_shots), 15.0)
        self.assertGreaterEqual(sum(shot.duration for shot in brief.evidence_shots), 12.0)
        self.assertEqual(len(manifest.scenes), 3)
        self.assertEqual(manifest.scenes[0].evidence_ids[0], source_video.id)

    def test_x_attached_video_follows_complete_root_post(self) -> None:
        root = Evidence(
            "root", "candidate-1", "https://x.com/example/status/1",
            "The original X post announcing a KV-cache visualization.",
            "x:thread_post",
        )
        source_video = Evidence(
            "video", "candidate-1", "https://video.twimg.com/tweet_video/example.mp4",
            "Attached animation", "web:source_video", captured_asset="assets/attached.mp4",
            metadata={"clip_start": 0.0, "clip_end": 5.0},
        )
        shots = [
            EvidenceShot(
                "shot-1", EvidenceShotKind.TWEET_CARD, "", "原帖先说明发生了什么", "",
                [root.id], ["opening"], source_url=root.url, duration=4.0,
                visual_family="tweet", full_translation="原帖中文翻译",
            ),
            EvidenceShot(
                "shot-2", EvidenceShotKind.BROWSER_SECTION, "", "动画展示缓存变化", "",
                [root.id], ["proof"], source_url=root.url, duration=4.0,
                visual_family="quote_card",
            ),
            EvidenceShot(
                "shot-3", EvidenceShotKind.BROWSER_SECTION, "", "变化会影响显存占用", "",
                [root.id], ["takeaway"], source_url=root.url, duration=4.0,
                visual_family="impact_card",
            ),
        ]
        brief = EditorialBrief(
            "KV 缓存动画展示显存变化", "先看原帖，再看动画", "动画让抽象机制更直观",
            AttentionStrategy(
                "缓存变化", "静态图与动画", "可视化过程", "推理开发者",
                "先交代来源", "再看原动画", ["KV 缓存动画展示显存变化"] * 3,
                [root.id], "KV 缓存动画展示显存变化",
            ), [], [], shots, 12.0, opening_mode="direct_fact",
        )
        manifest = RenderManifest(
            "render-x-video", "candidate-1", ContentType.FLASH, [], [root], [root.url],
            editorial_brief=brief, fixed_hook=brief.headline, fixed_title=brief.headline,
            fixed_footer=brief.fixed_conclusion,
            render_profile=InformationRenderProfile.RADAR_V2.value,
        )
        candidate = MagicMock(source_type=SourceType.TWEET, source_url=root.url)

        VideoFactory._canonicalize_source_video_visuals(manifest, candidate, source_video)

        self.assertEqual(brief.evidence_shots[0].kind, EvidenceShotKind.TWEET_CARD)
        self.assertEqual(brief.evidence_shots[0].visual_family, "tweet")
        self.assertEqual(brief.evidence_shots[1].kind, EvidenceShotKind.VIDEO)
        self.assertEqual(brief.evidence_shots[1].visual_family, "source_video")
        self.assertEqual(manifest.scenes[0].evidence_ids[0], root.id)
        self.assertEqual(manifest.scenes[1].evidence_ids[0], source_video.id)

    def test_multi_step_source_action_gets_more_time_than_cards(self) -> None:
        page = Evidence(
            "page", "candidate-1", "https://vendor.example/robot", "Exact source proof",
            "web:primary_page",
        )
        source_video = Evidence(
            "video", "candidate-1", "https://youtube.com/watch?v=official", "Official video",
            "web:source_video", captured_asset="assets/video.mp4",
            metadata={"clip_start": 12.0, "clip_end": 24.0},
        )
        shots = [
            EvidenceShot(
                id=f"shot-{index}", kind=EvidenceShotKind.BROWSER_SECTION,
                question="What happened?", fact="A robot completes a multi-step industrial task",
                interpretation="The action reaches a visible payoff", evidence_ids=[page.id],
                beat_ids=["proof"], source_url=page.url, target="Exact source proof",
                translation="", duration=5.0, visual_family="official_page",
            )
            for index in range(1, 4)
        ]
        brief = EditorialBrief(
            "Robot enters industrial work", "One action clip plus two cards", "Deployment is visible",
            AttentionStrategy(
                "Industrial deployment", "Real work versus demo", "Task completes", "Factory users",
                "Watch the task", "Then verify deployment", ["Robot enters industrial work"] * 3,
                [page.id], "Robot enters industrial work",
            ), [], [], shots, 15.0, opening_mode="direct_fact",
        )
        stale_video = Evidence(
            "video", "candidate-1", source_video.url, "Old official video selection",
            "web:source_video", captured_asset="assets/video.mp4",
            metadata={"clip_start": 12.0, "clip_end": 17.0},
        )
        manifest = RenderManifest(
            "render-video-long", "candidate-1", ContentType.FLASH, [], [page, stale_video], [page.url],
            editorial_brief=brief, fixed_hook=brief.headline, fixed_title=brief.headline,
            fixed_footer=brief.fixed_conclusion,
            render_profile=InformationRenderProfile.RADAR_V2.value,
        )

        candidate = MagicMock(source_url=page.url)
        candidate.metadata = {"discovery_channel": "robotics"}
        VideoFactory._canonicalize_source_video_visuals(manifest, candidate, source_video)

        self.assertEqual([shot.duration for shot in brief.evidence_shots], [12.0, 3.5, 3.5])
        self.assertEqual(sum(scene.end - scene.start for scene in manifest.scenes), 19.0)
        self.assertTrue(all(not shot.audience_copy for shot in brief.evidence_shots[1:]))
        self.assertEqual(manifest.fixed_footer, brief.fixed_conclusion)
        replaced = next(item for item in manifest.evidence if item.id == source_video.id)
        self.assertEqual(replaced.metadata["clip_end"], 24.0)

    def test_rerender_reuses_archived_source_video_asset(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            archived = workspace.root / "assets" / "official.mp4"
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes(b"video")
            video = Evidence(
                "video", "candidate-1", "https://youtube.com/watch?v=official", "Official",
                "web:source_video", captured_asset="assets/official.mp4",
            )
            manifest = RenderManifest(
                "render-reuse", "candidate-1", ContentType.FLASH, [], [video], [video.url],
            )

            selected = VideoFactory(workspace)._archived_source_video_asset(manifest, video.url)

            self.assertEqual(selected, archived)
            self.assertEqual(str(selected.relative_to(workspace.root)), video.captured_asset)

    def test_native_media_master_is_primary_and_does_not_invoke_mpt(self) -> None:
        with TemporaryDirectory() as temp:
            job = Path(temp)
            framed = job / "framed.mp4"
            mastered = job / "native-ffmpeg-master.mp4"
            result: dict[str, object] = {"stages": []}
            with (
                patch("video_factory.factory.MPTSettings.from_environment", return_value=MagicMock()),
                patch(
                    "video_factory.factory.NativeFFmpegAssemblyAdapter.assemble",
                    return_value=mastered,
                ) as native,
                patch("video_factory.factory.MPTAssemblyAdapter.assemble") as mpt,
            ):
                selected = VideoFactory._assemble_master(basic_manifest(), framed, job, result)

            self.assertEqual(selected, mastered)
            native.assert_called_once()
            mpt.assert_not_called()
            self.assertEqual(result["stages"][0]["backend"], "native_ffmpeg")
            self.assertFalse(result["stages"][0]["fallback_used"])

    def test_media_master_records_native_failure_before_mpt_fallback(self) -> None:
        with TemporaryDirectory() as temp:
            job = Path(temp)
            framed = job / "framed.mp4"
            fallback = job / "mpt-master.mp4"
            result: dict[str, object] = {"stages": []}
            with (
                patch("video_factory.factory.MPTSettings.from_environment", return_value=MagicMock()),
                patch(
                    "video_factory.factory.NativeFFmpegAssemblyAdapter.assemble",
                    side_effect=RuntimeError("native encoder failed"),
                ),
                patch(
                    "video_factory.factory.MPTAssemblyAdapter.assemble",
                    return_value=fallback,
                ) as mpt,
            ):
                selected = VideoFactory._assemble_master(basic_manifest(), framed, job, result)

            self.assertEqual(selected, fallback)
            mpt.assert_called_once()
            failure = json.loads((job / "assembly-primary-error.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["fallback"], "money_printer_turbo")
            self.assertEqual([stage["status"] for stage in result["stages"]], ["fallback", "ok"])

    def test_editorial_agent_routes_failed_final_verification_to_human_review(self) -> None:
        with TemporaryDirectory() as temp:
            factory = VideoFactory(Workspace(Path(temp) / "workspace"))
            job = Path(temp) / "job"
            job.mkdir()
            primary_writer = MagicMock()
            primary_writer.settings.model = "z-ai/glm-5.3-flash"
            primary_reviewer = MagicMock()
            primary_agent = MagicMock()
            primary_agent.run.side_effect = ContentAgentError(
                "critic rejected duplicate hook", [{"step": "copy_review", "status": "failed"}],
            )
            selection: dict[str, object] = {}
            packet = MagicMock()
            packet.candidate.id = "candidate-1"

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False),
                patch.object(
                    factory, "_editorial_agent", return_value=primary_agent,
                ) as editorial_agent,
            ):
                with self.assertRaises(ContentAgentError):
                    factory._run_editorial_agent(
                        packet, primary_writer, primary_reviewer,
                        GenerateOptions(provider="deepseek"), job, selection,
                    )

            self.assertTrue((job / "content-agent-error.json").is_file())
            self.assertEqual(selection["semantic_failure"]["action"], "human_review")
            self.assertEqual(editorial_agent.call_args.kwargs["max_llm_calls"], 7)

    def test_openrouter_writer_uses_independent_gemini_critic(self) -> None:
        with TemporaryDirectory() as temp:
            factory = VideoFactory(Workspace(Path(temp) / "workspace"))
            writer = MagicMock()
            writer.settings.provider = "openrouter"
            writer.settings.model = "google/gemini-3.7-flash"
            critic_settings = MagicMock()
            critic_settings.provider = "openrouter"
            critic_settings.model = "google/gemini-3.6-flash"
            chinese_settings = MagicMock()
            chinese_settings.provider = "deepseek"
            chinese_settings.model = "deepseek-chat"
            gemini_reviewer = MagicMock()
            chinese_reviewer = MagicMock()

            with (
                patch.dict(os.environ, {
                    "OPENROUTER_API_KEY": "test-key", "DEEPSEEK_API_KEY": "test-key",
                }, clear=False),
                patch(
                    "video_factory.factory.LLMSettings.from_environment",
                    side_effect=[critic_settings, chinese_settings],
                ) as settings,
                patch(
                    "video_factory.factory.OpenAICompatibleStoryWriter",
                    side_effect=[gemini_reviewer, chinese_reviewer],
                ),
                patch("video_factory.factory.OpenRouterCatalog.select") as select,
            ):
                reviewer, selection = factory._copy_reviewer(writer, GenerateOptions())

            self.assertIsInstance(reviewer, CompositeCopyReviewer)
            self.assertEqual(selection["provider"], "composite")
            self.assertEqual(
                [item["model"] for item in selection["reviewers"]],
                ["google/gemini-3.6-flash", "deepseek-chat"],
            )
            self.assertIn("spoken-Chinese", selection["reason"])
            self.assertEqual(settings.call_count, 2)
            select.assert_not_called()

    def test_composite_copy_reviewer_requires_both_taste_and_spoken_chinese(self) -> None:
        taste = MagicMock()
        chinese = MagicMock()
        taste.review_visible_copy.return_value = ([], {"model": "gemini"})
        chinese_issue = {
            "field_path": "editorial_brief.evidence_shots[0].translation",
            "category": "natural_chinese",
            "problem": "像论文直译，不像开发者口头解释",
        }
        chinese.review_spoken_chinese.return_value = ([chinese_issue], {"model": "deepseek"})
        reviewer = CompositeCopyReviewer([
            ("story_and_directing_taste", taste),
            ("spoken_chinese_copy", chinese),
        ])

        issues, provenance = reviewer.review_visible_copy(object(), {})

        self.assertEqual(issues, [chinese_issue])
        self.assertEqual(provenance["provider"], "composite")
        self.assertEqual([item["role"] for item in provenance["reviews"]], [
            "story_and_directing_taste", "spoken_chinese_copy",
        ])

    def test_explicit_trigger_uncertainty_requires_adjacent_visible_qualification(self) -> None:
        evidence = [Evidence(
            "e-uncertain", "tweet-1", "https://x.com/example/status/1",
            "Shortly afterward my account was suspended. I don't know yet whether this setup was the trigger.",
            "x:thread_post",
        )]

        direction = VideoFactory._causal_uncertainty_direction(evidence)

        self.assertIn("同一句或同一屏", direction)
        self.assertIn("是否由此触发尚无定论", direction)
        self.assertIn("禁止用‘导致、触发、秒封、随即被封、照做就被封’", direction)
        self.assertIn("不能取代故事本身", direction)
        self.assertIn("不得把‘截图只能证明/不能读成因果’写成整条视频的 Hook 或结论", direction)

    def test_no_causal_uncertainty_rule_without_explicit_source_boundary(self) -> None:
        evidence = [Evidence(
            "e-causal", "tweet-1", "https://x.com/example/status/1",
            "The vendor confirmed that the policy caused the suspension.", "x:thread_post",
        )]
        self.assertEqual(VideoFactory._causal_uncertainty_direction(evidence), "")

    def test_openrouter_discount_index_skips_routine_fifteen_percent_promotion(self) -> None:
        evidence = [Evidence(
            "discount-page", "openrouter", "https://openrouter.ai/models?discount=true",
            "[Wan 3.0](https://openrouter.ai/alibaba/wan-3) 15% off",
            "web:primary_page",
        )]

        gate = VideoFactory._openrouter_discount_story_gate(
            "https://openrouter.ai/models?discount=true", evidence, None,
        )

        self.assertIsNotNone(gate)
        self.assertFalse(gate["eligible"])
        self.assertEqual(gate["threshold_percent"], 75.0)
        self.assertEqual(gate["effective_discount_percent"], 15.0)

    def test_openrouter_discount_index_accepts_exceptional_effective_savings(self) -> None:
        evidence = [Evidence(
            "discount-page", "openrouter", "https://openrouter.ai/models?discount=true",
            "OpenRouter discounted models", "web:primary_page",
        )]
        metadata = {
            "discount_percent": 0,
            "official_comparison": {"savings_offpeak_percent": 85.8},
        }

        gate = VideoFactory._openrouter_discount_story_gate(
            "https://openrouter.ai/models?discount=true", evidence, metadata,
        )

        self.assertTrue(gate["eligible"])
        self.assertEqual(gate["effective_discount_percent"], 85.8)

    def test_ui_rerender_recompiles_scenes_without_rewriting_editorial_story(self) -> None:
        manifest = basic_manifest()
        manifest.editorial_brief = MagicMock()
        original_hook = manifest.fixed_hook
        original_footer = manifest.fixed_footer

        with (
            patch("video_factory.factory.canonicalize_editorial_brief") as canonicalize,
            patch("video_factory.factory.compile_evidence_shots", return_value=[]),
        ):
            VideoFactory._recompile_editorial_manifest(
                manifest, MagicMock(), normalize_story=False,
            )

        canonicalize.assert_not_called()
        self.assertEqual(manifest.fixed_hook, original_hook)
        self.assertEqual(manifest.fixed_footer, original_footer)

    def test_fragmented_hook_repair_selects_existing_complete_alternative(self) -> None:
        manifest = basic_manifest()
        manifest.render_profile = InformationRenderProfile.RADAR_V2.value
        manifest.editorial_brief = EditorialBrief(
            "Noble Machines 成立18个月", "首批机器人已交付", "机器人进入工业现场",
            AttentionStrategy(
                "已交付", "从演示到部署", "18个月", "工业客户", "已部署", "完成交付",
                [
                    "成立18个月，Noble Machines 把首批通用机器人交付给财富全",
                    "Noble Machines 成立18个月即达成首个交付里程碑",
                    "一家成立18个月的机器人公司",
                ],
                [manifest.evidence[0].id],
                "成立18个月，Noble Machines 把首批通用机器人交付给财富全",
            ),
            [], [], [], 15.0, opening_mode="direct_fact",
        )

        repaired = repair_fragmented_radar_hook(manifest.editorial_brief)

        self.assertEqual(repaired, "Noble Machines 成立18个月即达成首个交付里程碑")

    def test_ui_rerender_load_freezes_approved_editorial_copy(self) -> None:
        manifest = basic_manifest()
        manifest.editorial_brief = EditorialBrief(
            "标题", "副标题", "结论",
            AttentionStrategy(
                "事实", "冲突", "意外", "影响", "判断", "回报",
                ["候选一", "候选二", "候选三"], [manifest.evidence[0].id], "候选一",
            ),
            [], [], [EvidenceShot(
                "shot-1", EvidenceShotKind.BROWSER_SECTION, "发生了什么？", "原事实", "解释",
                [manifest.evidence[0].id], ["proof"],
            )], 10.0,
        )
        manifest.editorial_brief.evidence_shots[0].fact = (
            "alex getman：照 Tibo 配置后账号被停用，是否由此触发未定，已申诉"
        )
        manifest.fixed_hook = "已审核 Hook"
        manifest.fixed_title = "已审核标题"
        manifest.fixed_footer = "已审核结论"

        with TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text(
                json.dumps(manifest.to_dict(), ensure_ascii=False), encoding="utf-8",
            )
            loaded = load_manifest(path, normalize_story=False)

        self.assertEqual(
            loaded.editorial_brief.evidence_shots[0].fact,
            "alex getman：照 Tibo 配置后账号被停用，是否由此触发未定，已申诉",
        )
        self.assertEqual(loaded.fixed_hook, "已审核 Hook")
        self.assertEqual(loaded.fixed_title, "已审核标题")
        self.assertEqual(loaded.fixed_footer, "已审核结论")

    def test_employee_reply_cannot_be_promoted_to_official_company_response(self) -> None:
        evidence = [Evidence(
            "e-employee", "tweet-1", "https://x.com/employee/status/1",
            "An Anthropic employee replied from a personal account: We are hiring.",
            "x:visual_analysis",
        )]

        direction = VideoFactory._source_identity_direction(evidence)

        self.assertIn("个人回复不等于公司官方账号或公司声明", direction)
        self.assertIn("禁止写‘官方回应、官方表态、第一条官方回应’", direction)

    def test_github_generation_cache_is_reused_and_refreshable(self) -> None:
        with TemporaryDirectory() as temp:
            factory = VideoFactory(Workspace(Path(temp) / "workspace"))
            manifest = basic_manifest()

            def generate_github(url, job, options, result):
                path = job / "manifest.json"
                path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
                result["manifest"] = str(path)
                result["checks"] = []
                result["publishable"] = True

            url = "https://github.com/acme/demo"
            with (
                patch.object(factory, "_generate_github", side_effect=generate_github) as generate,
                patch("video_factory.factory.validate_manifest", return_value=[]),
            ):
                first = factory.generate(url, GenerateOptions(render=False))
                second = factory.generate(url, GenerateOptions(render=False))
                refreshed = factory.generate(url, GenerateOptions(render=False, refresh=True))

            self.assertEqual(generate.call_count, 2)
            self.assertEqual(
                next(stage for stage in second["stages"] if stage["name"] == "generation_cache")["status"],
                "hit",
            )
            self.assertEqual(
                next(stage for stage in first["stages"] if stage["name"] == "generation_cache")["status"],
                "stored",
            )
            self.assertEqual(
                next(stage for stage in refreshed["stages"] if stage["name"] == "generation_cache")["status"],
                "stored",
            )

    def test_generation_cache_isolated_by_render_profile(self) -> None:
        with TemporaryDirectory() as temp:
            factory = VideoFactory(Workspace(Path(temp) / "workspace"))
            classic = factory._generation_cache_path(
                "https://x.com/example/status/1", GenerateOptions(render=False),
            )
            radar = factory._generation_cache_path(
                "https://x.com/example/status/1",
                GenerateOptions(render=False, render_profile="radar_v2"),
            )
            self.assertNotEqual(classic, radar)

    def test_generation_cache_hit_persists_loaded_manifest_migrations(self) -> None:
        with TemporaryDirectory() as temp:
            factory = VideoFactory(Workspace(Path(temp) / "workspace"))
            factory.workspace.initialize()
            options = GenerateOptions(render=False)
            url = "https://github.com/acme/demo"
            cache = factory._generation_cache_path(url, options)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text('{"stale": true}\n', encoding="utf-8")
            job = Path(temp) / "job"
            job.mkdir()
            result: dict[str, object] = {"stages": []}

            with (
                patch("video_factory.factory.load_manifest", return_value=basic_manifest()),
                patch("video_factory.factory.validate_manifest", return_value=[]),
            ):
                factory._generate_cached_manifest("github", url, job, options, result)

            self.assertEqual(
                json.loads((job / "manifest.json").read_text(encoding="utf-8"))["fixed_footer"],
                "conclusion",
            )
            self.assertEqual(
                json.loads(cache.read_text(encoding="utf-8"))["fixed_footer"],
                "conclusion",
            )

    def test_github_render_applies_license_and_budgets_cold_open(self) -> None:
        with TemporaryDirectory() as temp:
            factory = VideoFactory(Workspace(Path(temp) / "workspace"))
            manifest = basic_manifest()
            manifest.github_brief = object()  # The capture request is intercepted before inspecting the brief.
            manifest.cold_open_beats = [
                ColdOpenBeat("one", "one", "event_hook", 1.1, ["e-1"]),
                ColdOpenBeat("two", "two", "capability_reveal", 1.2, ["e-1"]),
                ColdOpenBeat("three", "three", "editorial_verdict", 1.3, ["e-1"]),
            ]
            requested_durations: list[float] = []

            def stop_after_request(*args):
                requested_durations.append(float(args[-1]))
                raise RuntimeError("capture intercepted")

            bgm = Path("workspace/assets/music/858ccdf31193/better-times-are-coming-mixkit-173.mp3").resolve()
            with (
                patch.dict(os.environ, {"VIDEO_FACTORY_BGM_FILE": str(bgm)}, clear=False),
                patch(
                    "video_factory.factory.WebScrollVideoAdapter.github_story_request",
                    side_effect=stop_after_request,
                ),
                self.assertRaisesRegex(RuntimeError, "capture intercepted"),
            ):
                factory._render_github_manifest(
                    manifest, "https://github.com/acme/demo", Path(temp), {"stages": []},
                )

            self.assertEqual(manifest.music_license_status, "royalty_free_verified")
            self.assertTrue(manifest.license_records)
            self.assertAlmostEqual(requested_durations[0], 16.4)

    def test_archive_asset_hashes_without_reading_entire_file(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = Workspace(root / "workspace")
            workspace.initialize()
            source = root / "large.mp4"
            payload = b"a" * (2 * 1024 * 1024 + 17)
            source.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()

            with patch.object(Path, "read_bytes", side_effect=AssertionError("unbounded read")):
                archived, digest = workspace.archive_asset(source, "youtube-video")

            self.assertEqual(digest, expected)
            self.assertEqual((workspace.root / archived).stat().st_size, len(payload))

    def test_archive_asset_reuses_an_already_archived_source(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            source = Path(temp) / "captions.json3"
            source.write_text("reviewed subtitles", encoding="utf-8")
            archived, digest = workspace.archive_asset(
                source, "youtube-subtitles", source.name,
            )

            reused, reused_digest = workspace.archive_asset(
                workspace.root / archived, "youtube-subtitles", source.name,
            )

            self.assertEqual(reused, archived)
            self.assertEqual(reused_digest, digest)

    def test_rerender_dispatches_youtube_collection_without_llm_or_acquisition(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            manifest_path = Path(temp) / "collection-manifest.json"
            manifest_path.write_text(
                json.dumps({"source_video_id": "video-1", "items": []}), encoding="utf-8",
            )
            collection = SimpleNamespace(
                id="youtube-video-1-test", source_video_id="video-1",
                source_url="https://youtube.com/watch?v=video-1", items=[], quality_checks=[],
                to_dict=lambda: {
                    "id": "youtube-video-1-test", "source_video_id": "video-1", "items": [],
                },
            )

            with patch(
                "video_factory.factory.load_collection_manifest", return_value=collection,
            ), patch(
                "video_factory.youtube.YouTubeCollectionRenderer.render", return_value=collection,
            ) as render, patch(
                "video_factory.youtube.validate_collection", return_value=[],
            ):
                result = VideoFactory(workspace).rerender(manifest_path)

            render.assert_called_once_with(collection)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["renders"], [])
            self.assertEqual(result["stages"][0]["llm_calls"], 0)
            self.assertEqual(result["stages"][0]["acquisition_calls"], 0)


if __name__ == "__main__":
    unittest.main()
