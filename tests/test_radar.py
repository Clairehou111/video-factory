from __future__ import annotations

import subprocess
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from video_factory.compositor import (
    RADAR_META_RAIL, _dense_evidence, _expanded_evidence_layout,
    _radar_metadata, compose_information_frame, render_direct_identifier_badge,
    render_spotlight_overlay,
)
from video_factory.editorial import (
    _validate_radar_contract, canonicalize_editorial_brief, validate_editorial_structure,
)
from video_factory.github_editor import copy_width
from video_factory.models import (
    AttentionStrategy, Candidate, ContentType, ContextGraph, DirectorBrief, EditorialBrief,
    Evidence, EvidenceShot, EvidenceShotKind, InformationRenderProfile, MaterialRole,
    RenderManifest, Scene, SourceType, StorySubject, TopicType,
)
from video_factory.radar import (
    AdvisoryRecoveryAction, TechnicalArtifactKind, build_tencent_radar_copy,
    extract_direct_identifier, extract_technical_artifact, plan_advisory_recovery,
)
from video_factory.quality import validate_manifest
from video_factory.tracks import TrackSegment, build_dip_to_color_track
from video_factory.tweetcard import tweet_card_video
from video_factory.writer import StoryWriterPacket


def _brief(**overrides) -> EditorialBrief:
    evidence_shots = [
        EvidenceShot(
            "shot-1", EvidenceShotKind.TWEET_CARD, "", "GLM 发布 FP8 权重", "",
            ["e-1"], ["beat-1"], full_translation="GLM 发布 FP8 权重并公布拒绝率变化",
        ),
        EvidenceShot(
            "shot-2", EvidenceShotKind.BROWSER_SECTION, "", "拒绝率从96%降到11%", "",
            ["e-1"], ["beat-2"], translation="模型直接拒绝回答的请求占比从96%降到11%",
        ),
        EvidenceShot(
            "shot-3", EvidenceShotKind.BROWSER_SECTION, "", "开发者可下载权重测试", "",
            ["e-1"], ["beat-3"], translation="开发者可下载 FP8 权重进行安全测试",
        ),
    ]
    values = {
        "headline": "GLM-5.3-Flash拒绝率降至11%",
        "subheadline": "OrcaRouter发布FP8权重",
        "fixed_conclusion": "开发者可下载FP8权重做安全测试",
        "attention_strategy": AttentionStrategy(
            "GLM发布权重", "旧版与新版拒绝率", "", "本地测试", "直接快报", "可下载测试",
            ["GLM-5.3-Flash拒绝率降至11%", "GLM发布FP8权重", "拒绝率降至11%"],
            ["e-1"], "GLM-5.3-Flash拒绝率降至11%",
        ),
        "subjects": [StorySubject("GLM-5.3-Flash", "model", "发布", "可测试", ["e-1"])],
        "context_events": [], "evidence_shots": evidence_shots, "duration_target": 10,
        "opening_mode": "direct_fact", "category_label": "模型发布",
        "direct_identifier": "HF: OrcaRouter/GLM-5.3-Flash", "editorial_inference": "",
    }
    values.update(overrides)
    return EditorialBrief(**values)


def _manifest(profile: str = "classic", topic: TopicType = TopicType.MODEL_OR_PRODUCT) -> RenderManifest:
    evidence = Evidence(
        "e-1", "c-1", "https://huggingface.co/OrcaRouter/GLM-5.3-Flash",
        "OrcaRouter published GLM-5.3-Flash. pip install glm-tools", "web:primary_page",
    )
    return RenderManifest(
        "m-1", "c-1", ContentType.FLASH,
        [Scene("s-1", 0, 10, "", "GLM更新", ["e-1"], MaterialRole.PROOF, "show")],
        [evidence], [evidence.url], topic_type=topic, editorial_brief=_brief(),
        fixed_title="GLM-5.3-Flash拒绝率降至11%", fixed_footer="开发者可下载FP8权重做安全测试",
        render_profile=profile,
    )


class RadarV2Test(unittest.TestCase):
    def test_classic_profile_never_enables_expanded_radar_layout(self) -> None:
        classic = _manifest("classic", TopicType.RESEARCH_OR_BENCHMARK)
        radar = _manifest(InformationRenderProfile.RADAR_V2.value, TopicType.RESEARCH_OR_BENCHMARK)
        self.assertFalse(_expanded_evidence_layout(classic))
        self.assertTrue(_expanded_evidence_layout(radar))

    def test_unknown_render_profile_is_publish_blocking(self) -> None:
        manifest = _manifest("radar_future")
        check = next(item for item in validate_manifest(manifest) if item.name == "render_profile")
        self.assertFalse(check.passed)

    def test_radar_contract_uses_direct_fact_without_forcing_shock(self) -> None:
        brief = _brief()
        evidence = [_manifest().evidence[0]]
        canonicalize_editorial_brief(brief, evidence)
        self.assertEqual(
            [shot.narrative_beat for shot in brief.evidence_shots],
            ["opening", "proof", "takeaway"],
        )
        self.assertEqual(_validate_radar_contract(brief, evidence), [])

    def test_radar_canonicalizer_repairs_mechanical_contract_errors_without_llm(self) -> None:
        brief = _brief(category_label="开源神器", editorial_inference="not a question")
        brief.attention_strategy.hook_candidates = [
            "GLM-5.3-Flash 发布了一个非常非常长而且不适合手机首屏阅读的模型更新说明",
            "GLM-5.3-Flash 的第二个同样非常长的候选标题需要被机械压缩",
            "GLM-5.3-Flash 的第三个候选标题也不能突破手机画面预算",
        ]
        brief.attention_strategy.selected_hook = "不在候选列表里的标题"
        brief.fixed_conclusion = "开发者现在可以下载模型权重并在本地进行红蓝对抗和安全测试验证完整工作流"
        brief.evidence_shots[0].narrative_beat = "shock"
        brief.evidence_shots[0].fact = "GLM-5.3-Flash 发布了非常长的事实描述并且继续堆叠不必要的解释"
        brief.evidence_shots[0].audience_copy = "这段附加说明也远远超过单屏预算"
        brief.evidence_shots[0].full_translation = "模型发布说明" * 20
        canonicalize_editorial_brief(brief, [_manifest().evidence[0]])
        self.assertEqual(brief.category_label, "")
        self.assertIn(brief.attention_strategy.selected_hook, brief.attention_strategy.hook_candidates)
        self.assertLessEqual(copy_width(brief.attention_strategy.selected_hook), 28)
        self.assertLessEqual(copy_width(brief.fixed_conclusion), 40)

        self.assertLessEqual(
            copy_width(brief.evidence_shots[0].fact + brief.evidence_shots[0].audience_copy), 40,
        )
        self.assertLessEqual(copy_width(brief.evidence_shots[0].full_translation), 120)
        self.assertFalse(any(
            "Chinese gloss" in error for error in _validate_radar_contract(
                brief, [_manifest().evidence[0]],
            )
        ))
        self.assertEqual(
            [shot.narrative_beat for shot in brief.evidence_shots],
            ["opening", "proof", "takeaway"],
        )
        self.assertEqual(brief.editorial_inference, "")

    def test_radar_conclusion_uses_complete_existing_stance_instead_of_severing_name(self) -> None:
        brief = _brief()
        brief.fixed_conclusion = (
            "截图能证明的只有三件事：封号申诉公开发生、Tibo 否认自己是 Anthropic 员工、"
            "Boris Cherny 发布招聘回复，但三者没有已证实的因果关系"
        )
        brief.attention_strategy.stance = (
            "截图只证明申诉、否认和招聘回复同时出现，不代表三者有因果"
        )

        canonicalize_editorial_brief(brief, [_manifest().evidence[0]])

        self.assertEqual(
            brief.fixed_conclusion,
            "截图只证明申诉、否认和招聘回复同时出现，不代表三者有因果",
        )
        self.assertLessEqual(copy_width(brief.fixed_conclusion), 40)

        brief.fixed_conclusion = (
            "截图能证明的只有三件事：封号申诉公开发生、Tibo 否认自己是 Anthropic 员工、Boris"
        )
        canonicalize_editorial_brief(brief, [_manifest().evidence[0]])
        self.assertEqual(
            brief.fixed_conclusion,
            "截图只证明申诉、否认和招聘回复同时出现，不代表三者有因果",
        )

    def test_radar_reapplies_width_budget_after_glossary_injection(self) -> None:
        brief = _brief()
        brief.evidence_shots[0].fact = "GLM-5.3-Flash 拒答率从 96% 降至 11%"
        brief.evidence_shots[0].audience_copy = ""

        canonicalize_editorial_brief(brief, [_manifest().evidence[0]])

        self.assertIn("拒绝回答", brief.evidence_shots[0].audience_copy)
        self.assertLessEqual(
            copy_width(brief.evidence_shots[0].fact + brief.evidence_shots[0].audience_copy),
            40,
        )

        fp8_brief = _brief()
        fp8_brief.evidence_shots[0].fact = "GLM-5.3-Flash 发布原生 FP8 权重"
        fp8_brief.evidence_shots[0].audience_copy = ""
        fp8_brief.evidence_shots[0].translation = ""
        fp8_brief.evidence_shots[0].full_translation = ""
        fp8_brief.evidence_shots = fp8_brief.evidence_shots[:1]
        fp8_brief.attention_strategy.hook_candidates = [
            "GLM-5.3-Flash 发布 FP8 权重",
            "OrcaRouter 开放模型权重",
            "模型权重进入安全研究",
        ]
        fp8_brief.attention_strategy.selected_hook = fp8_brief.attention_strategy.hook_candidates[0]
        canonicalize_editorial_brief(fp8_brief, [_manifest().evidence[0]])
        self.assertIn("低数值精度", fp8_brief.evidence_shots[0].audience_copy)
        self.assertLessEqual(
            copy_width(
                fp8_brief.evidence_shots[0].fact + fp8_brief.evidence_shots[0].audience_copy
            ),
            40,
        )

    def test_radar_drops_optional_second_line_instead_of_showing_a_fragment(self) -> None:
        brief = _brief()
        brief.evidence_shots[0].fact = (
            "OpenRouter 模型库上线 Discounted 筛选标签，支持按折扣状态查找模型"
        )
        brief.evidence_shots[0].audience_copy = "OpenRouter 模型库可直接筛选折扣线路"
        brief.evidence_shots[0].translation = ""
        brief.evidence_shots[0].full_translation = ""
        brief.evidence_shots = brief.evidence_shots[:1]
        brief.attention_strategy.hook_candidates = [
            "OpenRouter 新增折扣筛选",
            "模型库直接显示优惠线路",
            "调用前先查折扣状态",
        ]
        brief.attention_strategy.selected_hook = brief.attention_strategy.hook_candidates[0]

        canonicalize_editorial_brief(brief, [_manifest().evidence[0]])

        self.assertEqual(brief.evidence_shots[0].audience_copy, "")

        for fragment in ("OpenRouter 模", "调用单价降至每秒", "视频生成模型直接"):
            with self.subTest(fragment=fragment):
                cached = _brief()
                cached.evidence_shots[0].fact = "OpenRouter 显示折扣模型"
                cached.evidence_shots[0].audience_copy = fragment
                cached.evidence_shots[0].translation = ""
                cached.evidence_shots[0].full_translation = ""
                cached.evidence_shots = cached.evidence_shots[:1]
                cached.attention_strategy.hook_candidates = [
                    "OpenRouter 新增折扣筛选", "模型库显示优惠线路", "调用前查看折扣状态",
                ]
                cached.attention_strategy.selected_hook = cached.attention_strategy.hook_candidates[0]
                canonicalize_editorial_brief(cached, [_manifest().evidence[0]])
                self.assertEqual(cached.evidence_shots[0].audience_copy, "")

    def test_radar_model_hook_accepts_evidence_preserving_artifact_base_name(self) -> None:
        brief = _brief()
        brief.subjects = [
            StorySubject(
                "GLM-5.3-Flash-Uncensored-FP8", "model", "发布权重", "开放研究", ["e-1"],
            )
        ]
        brief.attention_strategy.hook_candidates = [
            "GLM-5.3-Flash 拒答率降至 11%",
            "OrcaRouter 发布原生 FP8 权重",
            "拒答未归零暴露更复杂对齐机制",
        ]
        brief.attention_strategy.selected_hook = brief.attention_strategy.hook_candidates[0]
        canonicalize_editorial_brief(brief, [_manifest().evidence[0]])
        candidate = Candidate("c-1", SourceType.WEB, "https://example.com/model", "Model")
        errors = validate_editorial_structure(
            brief, candidate, [_manifest().evidence[0]], TopicType.MODEL_OR_PRODUCT,
            ContentType.FLASH,
        )
        self.assertFalse(any("concrete model subject" in error for error in errors))

    def test_radar_direct_fact_does_not_invent_a_conflict(self) -> None:
        brief = _brief(opening_mode="direct_fact")
        brief.attention_strategy.conflict = ""
        candidate = Candidate("c-1", SourceType.WEB, "https://example.com/update", "Update")
        errors = validate_editorial_structure(
            brief, candidate, [_manifest().evidence[0]], TopicType.MODEL_OR_PRODUCT,
            ContentType.FLASH,
        )
        self.assertFalse(any("attention_strategy.conflict" in error for error in errors))

    def test_multi_model_price_monitor_does_not_require_one_model_in_hook(self) -> None:
        brief = _brief(opening_mode="direct_fact")
        brief.subjects = [
            StorySubject("Gemini 3.7 Flash (batch)", "model", "参与折扣", "价格变化", ["e-1"]),
        ]
        brief.attention_strategy.hook_candidates = [
            "OpenRouter 折扣页列出多条低价线路",
            "同一模型不同线路价格拉开",
            "调用前先核对折扣线路",
        ]
        brief.attention_strategy.selected_hook = brief.attention_strategy.hook_candidates[0]
        evidence = _manifest().evidence[0]
        evidence.url = "https://openrouter.ai/models?discount=true"
        candidate = Candidate("c-1", SourceType.WEB, evidence.url, "Discount models")
        canonicalize_editorial_brief(brief, [evidence])
        errors = validate_editorial_structure(
            brief, candidate, [evidence], TopicType.MODEL_OR_PRODUCT, ContentType.FLASH,
        )
        self.assertFalse(any("concrete model subject" in error for error in errors))
        self.assertIn("OpenRouter", brief.attention_strategy.selected_hook)

    def test_radar_required_context_wins_over_conflicting_discarded_label(self) -> None:
        brief = _brief()
        brief.context_graph = ContextGraph(
            required_context_ids=["ctx-1"], discarded_context_ids=["ctx-1"],
        )
        brief.director_brief = DirectorBrief(
            "thesis", "tension", "neutral", 1, [], [], 10, "trigger",
        )

        canonicalize_editorial_brief(brief, [_manifest().evidence[0]])

        self.assertIn("ctx-1", brief.director_brief.selected_context_ids)
        self.assertNotIn("ctx-1", brief.context_graph.discarded_context_ids)

    def test_radar_web_opening_does_not_get_tweet_translation_budget(self) -> None:
        brief = _brief()
        brief.evidence_shots[0].kind = EvidenceShotKind.BROWSER_SECTION
        brief.evidence_shots[0].full_translation = "折扣页当前展示多个模型与供应商价格变化" * 10
        canonicalize_editorial_brief(brief, [_manifest().evidence[0]])
        self.assertLessEqual(copy_width(brief.evidence_shots[0].full_translation), 40)
        self.assertFalse(any(
            "Chinese gloss" in error for error in _validate_radar_contract(
                brief, [_manifest().evidence[0]],
            )
        ))

    def test_radar_hook_only_gets_extended_budget_when_subject_survives_clipping(self) -> None:
        brief = _brief()
        brief.attention_strategy.hook_candidates = [
            "同一折扣页的价格差距为什么会突然拉开一个数量级",
            "开发者调用模型前应该先检查真实输入输出价格",
            "批量线路把调用成本直接压低但适用条件不同",
        ]
        brief.attention_strategy.selected_hook = brief.attention_strategy.hook_candidates[0]
        canonicalize_editorial_brief(brief, [_manifest().evidence[0]])
        self.assertLessEqual(copy_width(brief.attention_strategy.selected_hook), 20)
        self.assertFalse(any(
            "Radar headline" in error for error in _validate_radar_contract(
                brief, [_manifest().evidence[0]],
            )
        ))

    def test_radar_rejects_marketing_category_and_unsupported_identifier(self) -> None:
        brief = _brief(category_label="开源神器", direct_identifier="pip install invented")
        errors = _validate_radar_contract(brief, [_manifest().evidence[0]])
        self.assertTrue(any("category_label" in error for error in errors))
        self.assertTrue(any("direct_identifier" in error for error in errors))

    def test_spotlight_uses_adaptive_40_and_55_percent_dimming(self) -> None:
        with TemporaryDirectory() as temp:
            normal = Path(temp) / "normal.png"
            dense = Path(temp) / "dense.png"
            box = {"left": 300, "top": 300, "width": 400, "height": 100}
            render_spotlight_overlay(box, (1080, 1000), 300, 1000, normal, dense=False)
            render_spotlight_overlay(box, (1080, 1000), 300, 1000, dense, dense=True)
            with Image.open(normal) as image:
                self.assertEqual(image.getpixel((10, 350))[3], 102)
            with Image.open(dense) as image:
                self.assertEqual(image.getpixel((10, 350))[3], 140)

    def test_density_detection_handles_real_capture_metadata(self) -> None:
        self.assertTrue(_dense_evidence(
            {"id": "price-table", "action": "highlight", "highlight_box": {
                "left": 100, "top": 100, "width": 400, "height": 80,
            }},
            (1384, 1602),
        ))
        self.assertFalse(_dense_evidence({"id": "hero", "action": "hold"}, (1080, 1200)))

    def test_identifier_and_category_stay_in_reserved_metadata_rail(self) -> None:
        with TemporaryDirectory() as temp:
            output = Path(temp) / "badge.png"
            pane_top = 500
            render_direct_identifier_badge(
                "Model: deepseek/deepseek-v4-flash-0731",
                pane_top - RADAR_META_RAIL, output, category_label="价格变化",
            )
            with Image.open(output) as image:
                alpha = image.getchannel("A")
                self.assertIsNone(alpha.crop((0, pane_top + 1, 1080, 1920)).getbbox())

    def test_story_without_identifier_or_fact_label_gets_no_empty_metadata_rail(self) -> None:
        manifest = _manifest(InformationRenderProfile.RADAR_V2.value)
        manifest.editorial_brief.direct_identifier = ""
        manifest.editorial_brief.category_label = ""
        manifest.source_urls = ["https://x.com/example/status/1"]
        manifest.evidence[0].url = manifest.source_urls[0]
        manifest.evidence[0].quote = "Jeff Dean announced a new research organization."
        self.assertEqual(_radar_metadata(manifest), ("", ""))

    def test_static_card_motion_is_opt_in_for_radar(self) -> None:
        with TemporaryDirectory() as temp:
            card = Path(temp) / "card.png"
            Image.new("RGB", (1080, 1920), "navy").save(card)
            with patch("video_factory.tweetcard.subprocess.run") as run:
                tweet_card_video(card, 2, Path(temp) / "classic.mp4")
                self.assertNotIn("zoompan", " ".join(run.call_args.args[0]))
                tweet_card_video(card, 2, Path(temp) / "radar.mp4", motion=True)
                self.assertIn("zoompan", " ".join(run.call_args.args[0]))

    def test_transition_failure_records_hard_cut_fallback(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            segments = [TrackSegment(root / "a.mp4", 2), TrackSegment(root / "b.mp4", 2)]
            output = root / "joined.mp4"
            failure = subprocess.CalledProcessError(1, ["ffmpeg"])
            with patch("video_factory.tracks.subprocess.run", side_effect=[failure, None]) as run:
                build_dip_to_color_track(segments, output)
            self.assertEqual(run.call_count, 2)
            self.assertTrue(output.with_suffix(".transition-fallback.json").is_file())

    def test_compositor_failure_retries_without_complex_radar_effects(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            visual = root / "visual.mp4"
            visual.write_bytes(b"video")
            visual.with_suffix(".capture.json").write_text(json.dumps({
                "width": 1384, "height": 1602, "duration": 2,
                "shots": [{
                    "id": "price-table", "action": "highlight", "start": 0, "end": 2,
                    "translation": "同一模型线路价格从每百万0.22美元降至0.03美元",
                    "highlight_box": {"left": 100, "top": 100, "width": 400, "height": 80},
                }],
            }), encoding="utf-8")
            output = root / "radar.mp4"
            failure = subprocess.CalledProcessError(1, ["ffmpeg"])
            with patch(
                "video_factory.compositor.probe_video",
                return_value=SimpleNamespace(width=1384, height=1602),
            ), patch(
                "video_factory.compositor.subprocess.run", side_effect=[failure, None],
            ), patch("video_factory.compositor.shutil.copy2"):
                compose_information_frame(
                    _manifest(InformationRenderProfile.RADAR_V2.value), visual, output,
                )
            trace = json.loads(output.with_suffix(".render-fallback.json").read_text(encoding="utf-8"))
            self.assertEqual(trace["fallback"], "radar_layout_without_spotlight_or_focus_crop")
            layout = json.loads(output.with_suffix(".layout.json").read_text(encoding="utf-8"))
            self.assertFalse(layout["effects_enabled"])

    def test_publish_copy_uses_optional_factual_label_and_source_identifier(self) -> None:
        manifest = _manifest(InformationRenderProfile.RADAR_V2.value)
        title, description = build_tencent_radar_copy(
            manifest, fallback_title="fallback", publisher="OrcaRouter",
            source_url=manifest.source_urls[0],
        )
        self.assertTrue(title.startswith("【模型发布】"))
        self.assertIn("HF: OrcaRouter/GLM-5.3-Flash", description)
        self.assertIn(manifest.source_urls[0], description)
        self.assertEqual(extract_direct_identifier(manifest), "HF: OrcaRouter/GLM-5.3-Flash")

    def test_identifier_skips_huggingface_social_thumbnail_paths(self) -> None:
        manifest = _manifest(InformationRenderProfile.RADAR_V2.value)
        manifest.editorial_brief.direct_identifier = ""
        manifest.source_urls = ["https://x.com/orcarouter/status/1"]
        manifest.evidence.insert(0, Evidence(
            "thumb", "c-1",
            "https://huggingface.co/social-thumbnails/models/OrcaRouter/GLM-5.3-Flash.png",
            "thumbnail", "web:source_image",
        ))
        self.assertEqual(extract_direct_identifier(manifest), "HF: OrcaRouter/GLM-5.3-Flash")
        artifact = extract_technical_artifact(manifest)
        self.assertIsNotNone(artifact)
        assert artifact is not None
        self.assertEqual(artifact.kind, TechnicalArtifactKind.HUGGING_FACE_MODEL)
        self.assertEqual(artifact.value, "OrcaRouter/GLM-5.3-Flash")
        self.assertEqual(artifact.source, manifest.evidence[1].url)

    def test_technical_artifact_preserves_type_and_provenance(self) -> None:
        manifest = _manifest(InformationRenderProfile.RADAR_V2.value)
        manifest.editorial_brief.direct_identifier = ""
        manifest.source_urls = ["https://x.com/example/status/1"]
        manifest.evidence[0].url = manifest.source_urls[0]
        manifest.evidence[0].quote = "Install it with pip install radar-kit today."
        artifact = extract_technical_artifact(manifest)
        self.assertIsNotNone(artifact)
        assert artifact is not None
        self.assertEqual(artifact.kind, TechnicalArtifactKind.PYTHON_PACKAGE)
        self.assertEqual(artifact.value, "radar-kit")
        self.assertEqual(artifact.source, "archived_evidence_text")
        self.assertEqual(extract_direct_identifier(manifest), "pip install radar-kit")

    def test_writer_prompt_exposes_four_opening_modes_and_optional_factual_label(self) -> None:
        candidate = Candidate("c-1", SourceType.TWEET, "https://x.com/a/status/1", "GLM")
        packet = StoryWriterPacket(
            candidate, [_manifest().evidence[0]], TopicType.MODEL_OR_PRODUCT,
            ContentType.FLASH, 10, render_profile=InformationRenderProfile.RADAR_V2.value,
        )
        prompt = packet.prompt()
        self.assertIn("direct_fact|conflict|counter_intuitive|developer_roi", prompt)
        self.assertIn("category_label is optional", prompt)
        self.assertIn("do not force shock", prompt)

    def test_advisory_recovery_splits_payload_then_degrades_video(self) -> None:
        batch = plan_advisory_recovery(
            "An internal error occurred", attachment_count=4, media_kind="video", attempt=0,
        )
        self.assertEqual(batch.action, AdvisoryRecoveryAction.SPLIT_ATTACHMENTS)
        self.assertEqual(batch.next_attachment_count, 2)
        single = plan_advisory_recovery(
            "An internal error occurred", attachment_count=1, media_kind="video", attempt=1,
        )
        self.assertEqual(single.action, AdvisoryRecoveryAction.USE_STORYBOARD)

    def test_advisory_recovery_never_enables_paid_key_implicitly(self) -> None:
        decision = plan_advisory_recovery(
            "permission denied; link a paid API key and set up billing",
            attachment_count=1, media_kind="video", attempt=0,
        )
        self.assertEqual(decision.action, AdvisoryRecoveryAction.NEEDS_HUMAN_AUTHORIZATION)
        self.assertEqual(decision.delay_seconds, 0)

    def test_advisory_recovery_uses_clean_session_then_stops_boundedly(self) -> None:
        clean = plan_advisory_recovery(
            "An unexpected error occurred", attachment_count=0, media_kind="text", attempt=1,
        )
        self.assertEqual(clean.action, AdvisoryRecoveryAction.START_CLEAN_SESSION)
        stopped = plan_advisory_recovery(
            "An unexpected error occurred", attachment_count=1, media_kind="storyboard", attempt=2,
        )
        self.assertEqual(stopped.action, AdvisoryRecoveryAction.STOP)


if __name__ == "__main__":
    unittest.main()
