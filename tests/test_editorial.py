import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.director import NarrativeAnswer, StoryboardDirector
from video_factory.editorial import (
    canonicalize_editorial_brief, compile_evidence_shots, route_content,
    validate_editorial_brief, validate_editorial_structure,
)
from video_factory.compositor import (
    GITHUB_COLD_OPEN_LAYOUTS, WECHAT_BOTTOM_UI_SAFE, WECHAT_TOP_UI_SAFE,
    _centered_lines, _footer_layout, _information_layout, _sequential_rail_windows,
    _wrapped_lines, compose_information_frame, render_github_cold_open_frames,
    render_information_frame,
)
from video_factory.media import VideoProbe
from video_factory.llm import OpenAICompatibleStoryWriter, _compile_evidence_shot_kind
from video_factory.models import (
    AttentionStrategy, Candidate, ContentType, EditorialBrief, Evidence, EvidenceShot,
    ContextGraph, DirectorBrief, EditorialOpportunity, EvidenceShotKind, MaterialRole, Scene,
    RenderManifest, SelectionReason, SourceType, StoryArcBeat, StorySubject, TopicType,
)
from video_factory.quality import _quantity_supported, validate_manifest
from video_factory.storage import Workspace
from video_factory.tweetcard import render_editorial_card, render_tweet_card
from video_factory.webcapture import WebScrollVideoAdapter
from video_factory.writer import StoryWriterPacket
from video_factory.acquisition import URLAcquirer


def evidence(candidate, text="A concrete verified capability"):
    return Evidence("e-1", candidate.id, candidate.source_url, text, "web:primary_page")


def brief_for(candidate, topic=TopicType.COMPANY_OR_TEAM, content_type=ContentType.FLASH):
    item = evidence(candidate, "Jeff Dean is leaving Google and founded Discovery Loop. A concrete verified capability")
    shots = [
        EvidenceShot(
            "source", EvidenceShotKind.TWEET_CARD if candidate.source_type == SourceType.TWEET else EvidenceShotKind.BROWSER_SECTION,
            "发生了什么", "Jeff Dean成立Discovery Loop", "四位长期合作者一起离开原有组织",
            [item.id], ["identity"], candidate.source_url,
            "" if candidate.source_type == SourceType.TWEET else "A concrete verified capability",
            "一项可验证的具体能力", 4.0, "",
        ),
        EvidenceShot(
            "impact", EvidenceShotKind.BROWSER_SECTION, "为什么重要", "团队要自动化机器学习实验循环",
            "竞争焦点从单个模型转向自动化科研系统", [item.id], ["product_direction", "impact"],
            candidate.source_url, "A concrete verified capability", "一项可验证的具体能力", 4.0, "能力揭示",
        ),
    ]
    return EditorialBrief(
        "Jeff Dean携三位AI老将创业", "目标直指自动化机器学习实验",
        "这次出走带走的是模型、系统和芯片的全栈协作能力",
        AttentionStrategy(
            "Jeff Dean与三位长期合作者成立新公司", "Google核心团队与新公司之间的人才反差", "四人合作时间跨越多年",
            "影响自动化科研竞争", "这不是普通创业，而是完整技术组合迁移", "真正值得看的是他们能否让AI自己推进实验",
            ["Jeff Dean带着三位老战友创业", "Google一口气失去四位AI老将", "Discovery Loop要让AI自己做实验"],
            [item.id], "Google一口气失去四位AI老将",
        ),
        [StorySubject("Jeff Dean", "person", "联合创立Discovery Loop", "把全栈AI经验带入科研自动化", [item.id])],
        [], shots, 12.0,
    ), item


class EditorialContractTests(unittest.TestCase):
    def test_internal_interpretation_is_not_compiled_into_visible_scene_copy(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/dev/status/1", "Post", author="dev")
        brief, item = brief_for(candidate)
        shot = brief.evidence_shots[1]
        shot.interpretation = "正确读法是降低预期权重，值得跟进学习"
        shot.audience_copy = ""
        shot.target = "A concrete verified capability"
        proposals = compile_evidence_shots(brief, candidate)
        self.assertEqual(proposals[1].screen_interpretation, "")
        self.assertEqual(proposals[1].source_excerpt, "A concrete verified capability")

    def test_english_browser_target_requires_adjacent_chinese_translation(self):
        candidate = Candidate("official", SourceType.WEB, "https://vendor.example/news", "News")
        brief, item = brief_for(candidate)
        brief.evidence_shots[1].target = "A concrete verified capability"
        brief.evidence_shots[1].translation = ""

        errors = validate_editorial_brief(
            brief, candidate, [item], TopicType.COMPANY_OR_TEAM, ContentType.FLASH,
        )

        self.assertTrue(any("English browser target" in error for error in errors))

    def test_x_screenshot_model_detail_does_not_reclassify_an_account_dispute_as_model_launch(self):
        candidate = Candidate(
            "tweet", SourceType.TWEET, "https://x.com/sama/status/1",
            "one of the things i like most about openai is tibo", author="sama",
        )
        root = Evidence("root", candidate.id, candidate.source_url, candidate.title, "x:thread_post")
        quoted = Evidence(
            "quoted", candidate.id, "https://x.com/stats/status/2",
            "How much they celebrate with Anthropic", "x:quoted_post",
        )
        screenshot = Evidence(
            "visual", candidate.id, "https://pbs.twimg.com/context.jpg",
            "A user configured Claude Code with another model and the account was suspended.",
            "x:visual_analysis",
        )
        route = route_content(candidate, [root, quoted, screenshot])
        self.assertEqual(route.topic_type, TopicType.PRACTICE_POST)
        self.assertEqual(route.content_type, ContentType.FLASH)

    def test_audience_copy_rejects_internal_reading_directions(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/dev/status/1", "Post", author="dev")
        brief, item = brief_for(candidate)
        brief.evidence_shots[1].audience_copy = "解读时应降低预期权重"
        errors = validate_editorial_brief(
            brief, candidate, [item], TopicType.COMPANY_OR_TEAM, ContentType.FLASH,
        )
        self.assertTrue(any("audience_copy contains internal" in error for error in errors))

    def test_fixed_upper_rail_expands_to_keep_long_title_complete(self):
        title = "最大的开源权重模型 Kimi K3 发布次日，Sebastian Raschka 直接说破：它不是新架构，而是 Kimi Linear 从 48B 放大到 2.8T。"
        top_height, _, lines = _information_layout(title)
        self.assertLessEqual(top_height, 340)
        self.assertGreaterEqual(lines, 3)
        with TemporaryDirectory() as directory:
            output = render_information_frame(title, "K3 的结构变化可以直接回溯到原帖证据。", Path(directory) / "frame.png")
            from PIL import Image
            with Image.open(output) as image:
                self.assertEqual(image.size, (1080, 1920))

    def test_fixed_bottom_rail_fits_long_conclusion_without_ellipsis(self):
        footer = "K3 的价值在于把 Kimi Linear 成功放大投产：MoE 到 LatentMoE 和 MLA/KDA 都是效率向替换，NoPE 全替代 RoPE 是已知首个前沿级案例，并新增原生多模态支持。"
        height, _, lines = _footer_layout(footer)
        self.assertLessEqual(height, 310)
        self.assertGreaterEqual(lines, 3)

    def test_information_frame_keeps_wechat_ui_safe_bands_free_of_text(self):
        from PIL import Image

        title = "OpenRouter Cheaper Choice｜DeepSeek V4 Flash 0731"
        footer = "开发者现在可以直接比较真实线路价格，再决定是否接入。"
        with TemporaryDirectory() as directory:
            output = render_information_frame(title, footer, Path(directory) / "frame.png")
            with Image.open(output).convert("RGBA") as image:
                background = (3, 17, 38, 255)
                top_colors = set(image.crop((0, 0, 1080, WECHAT_TOP_UI_SAFE)).getdata())
                bottom_colors = set(image.crop((0, 1920 - WECHAT_BOTTOM_UI_SAFE, 1080, 1920)).getdata())
                self.assertEqual(top_colors, {background})
                self.assertEqual(bottom_colors, {background})

                title_band = image.crop((0, WECHAT_TOP_UI_SAFE, 1080, WECHAT_TOP_UI_SAFE + 260))
                footer_band = image.crop((0, 1920 - WECHAT_BOTTOM_UI_SAFE - 230, 1080, 1920 - WECHAT_BOTTOM_UI_SAFE))
                self.assertGreater(len(set(title_band.getdata())), 1)
                self.assertGreater(len(set(footer_band.getdata())), 1)

    def test_directed_composition_shows_hook_and_conclusion_sequentially(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            visual = root / "visual.mp4"
            output = root / "directed.mp4"
            visual.write_bytes(b"video")
            evidence = Evidence("e-1", "candidate-1", "https://example.com", "claim", "web")
            manifest = RenderManifest(
                "render-1", "candidate-1", ContentType.FLASH,
                [Scene("s-1", 0, 8, "n", "caption", [evidence.id], MaterialRole.PROOF, "show")],
                [evidence], [evidence.url], fixed_title="开场观点", fixed_footer="最终结论",
            )

            def create_output(command, **_kwargs):
                Path(command[-1]).write_bytes(b"rendered")

            with (
                patch("video_factory.compositor.probe_video", return_value=VideoProbe(
                    visual, 8.0, 1384, 1602, "h264", "yuv420p", None,
                )),
                patch("video_factory.compositor.subprocess.run", side_effect=create_output) as run,
            ):
                compose_information_frame(manifest, visual, output)

            command = run.call_args.args[0]
            filters = command[command.index("-filter_complex") + 1]
            hook_end, conclusion_start = _sequential_rail_windows(8.0)
            self.assertIn(f"between(t,0,{hook_end:.3f})", filters)
            self.assertIn(f"between(t,{conclusion_start:.3f},8.000)", filters)
            self.assertIn(f"pad=1080:1920:0:{WECHAT_TOP_UI_SAFE}", filters)
            direction = json.loads(output.with_suffix(".direction.json").read_text(encoding="utf-8"))
            self.assertEqual(direction["mode"], "sequential_single_focus")
            self.assertEqual(direction["source_pane"]["height"], 1400)
            self.assertLess(hook_end, conclusion_start)

    def test_balanced_footer_never_splits_an_ascii_word(self):
        from PIL import Image, ImageDraw, ImageFont

        text = "本地部署即可批量出片，还能接入 Agent 做自动化内容生产，落地门槛很低"
        image = Image.new("RGB", (1080, 300))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(Path("/Users/clairehou/pyProjects/MoneyPrinterTurbo/resource/fonts/STHeitiMedium.ttc")), 42)
        lines = _wrapped_lines(draw, text, font, 924, 3)
        self.assertFalse(any(line.endswith("Ag") for line in lines))
        self.assertFalse(any(line.startswith("ent") for line in lines))

        # Exercise the two-line balancing path that previously re-split a
        # token after the greedy wrapper had kept it intact.
        balanced = _centered_lines(draw, text, font, 924, 2)
        self.assertNotIn("Ag\nent", "\n".join(balanced))

    def test_github_cold_open_contains_no_old_ornamental_lines_or_frames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            from PIL import Image
            source = root / "source.png"
            Image.new("RGB", (1080, 1200), "white").save(source)
            frames = render_github_cold_open_frames(
                "Anthropic 在 Claude 内容里加入溯源标记",
                "watermarks-remover 直接清理隐藏字符和元数据",
                "Claude Code 生成后可以自动清理",
                "conflict", "watermarks-remover：清理 AI 溯源标记",
                source, root / "frames",
            )
            forbidden = {(217, 45, 76), (22, 93, 255), (29, 77, 255), (124, 45, 255)}
            for frame in frames:
                with Image.open(frame).convert("RGB") as image:
                    self.assertTrue(forbidden.isdisjoint(set(image.getdata())))

    def test_github_cold_open_gives_repository_most_of_the_frame(self):
        for screenshot_y, screenshot_height, _, _, project_name_y in GITHUB_COLD_OPEN_LAYOUTS:
            self.assertLessEqual(screenshot_y, 800)
            self.assertGreaterEqual(screenshot_height, 1120)
            self.assertLess(project_name_y, screenshot_y)

    def test_one_source_image_promotion_does_not_erase_later_text_evidence(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/dev/status/1", "Architecture", author="dev")
        brief, root = brief_for(candidate)
        root.quote = "The architecture uses NoPE everywhere. Native multimodal support is available."
        image = Evidence(
            "arch-image", candidate.id, "https://pbs.twimg.com/architecture.jpg", "Official architecture diagram",
            "x:media_photo", metadata={
                "editorial_priority": "high", "visual_role": "architecture",
                "parent_source_url": candidate.source_url,
            },
        )
        brief.evidence_shots[1].fact = "架构图展示 NoPE"
        brief.evidence_shots[1].target = "The architecture uses NoPE everywhere."
        brief.evidence_shots.append(EvidenceShot(
            "payoff", EvidenceShotKind.TWEET_CARD, "架构结论", "原生多模态补齐架构能力",
            "内部收束说明", [root.id], ["impact"], source_url=candidate.source_url,
            target="Native multimodal support is available.", visual_family="impact_card", retention_job="payoff",
        ))
        canonicalize_editorial_brief(brief, [root, image])
        promoted = [shot for shot in brief.evidence_shots if image.id in shot.evidence_ids]
        self.assertEqual(len(promoted), 1)
        self.assertEqual(brief.evidence_shots[-1].target, "Native multimodal support is available.")

    def test_duplicate_image_evidence_url_cannot_steal_later_text_shot(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/dev/status/1", "Architecture", author="dev")
        brief, root = brief_for(candidate)
        root.quote = "The architecture replaces components with efficiency-tweaked versions."
        image_metadata = {
            "editorial_priority": "high", "visual_role": "architecture",
            "parent_source_url": candidate.source_url,
        }
        selected_image = Evidence(
            "arch-media", candidate.id, "https://pbs.twimg.com/architecture.jpg", "Architecture diagram",
            "x:media_photo", metadata=image_metadata,
        )
        duplicate_image = Evidence(
            "arch-agent-copy", candidate.id, "https://pbs.twimg.com/architecture.jpg?format=jpg", "Architecture diagram",
            "web:source_image", metadata=image_metadata,
        )
        brief.evidence_shots[1].kind = EvidenceShotKind.IMAGE
        brief.evidence_shots[1].visual_family = "source_image"
        brief.evidence_shots[1].evidence_ids = [selected_image.id]
        brief.evidence_shots[1].target = ""
        brief.evidence_shots.append(EvidenceShot(
            "text-proof", EvidenceShotKind.TWEET_CARD, "效率替换", "架构变化集中在效率替换",
            "内部说明", [root.id], ["impact"], source_url=candidate.source_url,
            target="replaces components with efficiency-tweaked versions", visual_family="quote_card",
            retention_job="payoff",
        ))
        canonicalize_editorial_brief(brief, [root, selected_image, duplicate_image])
        self.assertEqual(
            brief.evidence_shots[-1].target,
            "replaces components with efficiency-tweaked versions",
        )
        self.assertNotIn(duplicate_image.id, brief.evidence_shots[-1].evidence_ids)

    def test_repeated_x_image_analysis_becomes_a_derived_card_not_a_raw_image_browser(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/dev/status/1", "Chart", author="dev")
        brief, root = brief_for(candidate)
        image = Evidence(
            "chart-media", candidate.id, "https://pbs.twimg.com/chart.jpg", "Attached chart",
            "x:media_photo", metadata={"parent_source_url": candidate.source_url},
        )
        analysis = Evidence(
            "chart-analysis", candidate.id, image.url,
            '{"visible_text":"decoded tokens align with billed tokens"}', "x:visual_analysis",
            metadata={"parent_image_id": image.id, "parent_source_url": candidate.source_url},
        )
        brief.evidence_shots[1].kind = EvidenceShotKind.IMAGE
        brief.evidence_shots[1].visual_family = "source_image"
        brief.evidence_shots[1].evidence_ids = [image.id, analysis.id]
        brief.evidence_shots[1].target = ""
        brief.evidence_shots.append(EvidenceShot(
            "chart-detail", EvidenceShotKind.IMAGE, "图表细节", "数据落在直线上", "内部说明",
            [analysis.id], ["impact"], source_url=image.url,
            target="decoded tokens align with billed tokens", visual_family="chart", retention_job="impact",
        ))
        canonicalize_editorial_brief(brief, [root, image, analysis])
        detail = brief.evidence_shots[-1]
        self.assertEqual(detail.visual_family, "impact_card")
        self.assertNotIn(image.id, detail.evidence_ids)
        self.assertEqual(detail.source_url, "")

    def test_external_paper_target_is_routed_off_x_to_nearby_source_url(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/dev/status/1", "Paper thread", author="dev")
        brief, root = brief_for(candidate)
        linked = Evidence(
            "linked", candidate.id, "https://t.co/short", "Paper: arxiv.org/abs/2608.09867\nStealing Reasoning Traces from Proprietary LLM APIs",
            "web:agent_primary_source",
        )
        brief.evidence_shots[1].evidence_ids = [linked.id]
        brief.evidence_shots[1].visual_family = "paper"
        brief.evidence_shots[1].target = "Stealing Reasoning Traces from Proprietary LLM APIs"
        brief.evidence_shots[1].source_url = candidate.source_url
        canonicalize_editorial_brief(brief, [root, linked])
        self.assertEqual(brief.evidence_shots[1].source_url, "https://arxiv.org/abs/2608.09867")

    def test_distant_paper_link_does_not_steal_an_earlier_thread_quote(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/dev/status/1", "Paper thread", author="dev")
        brief, root = brief_for(candidate)
        target = "they have already patched several issues caused by this vulnerability"
        linked = Evidence(
            "linked", candidate.id, "https://t.co/short",
            target + "\n" + ("unrelated thread detail " * 80) + "\nPaper: arxiv.org/abs/2608.09867",
            "web:agent_primary_source",
        )
        brief.evidence_shots[1].evidence_ids = [linked.id]
        brief.evidence_shots[1].visual_family = "paper"
        brief.evidence_shots[1].kind = EvidenceShotKind.PDF_PAGE
        brief.evidence_shots[1].target = target
        brief.evidence_shots[1].source_url = candidate.source_url
        canonicalize_editorial_brief(brief, [root, linked])
        self.assertEqual(brief.evidence_shots[1].source_url, candidate.source_url)
        self.assertEqual(brief.evidence_shots[1].visual_family, "quote_card")
        self.assertEqual(brief.evidence_shots[1].kind, EvidenceShotKind.BROWSER_SECTION)

    def test_embedded_partner_link_does_not_steal_official_page_prose(self):
        candidate = Candidate(
            "official", SourceType.WEB, "https://vendor.example/announcement", "Announcement",
        )
        brief, item = brief_for(candidate)
        target = "The evaluator cannot see the model weights"
        item.quote = (
            "The evaluator [cannot see](https://partner.example/protocol) the model weights "
            "and the provider cannot see test prompts."
        )
        brief.evidence_shots[1].evidence_ids = [item.id]
        brief.evidence_shots[1].target = target
        brief.evidence_shots[1].source_url = "https://cdn.vendor.example/diagram.svg"
        brief.evidence_shots[1].visual_family = "stat_card"

        canonicalize_editorial_brief(brief, [item])

        self.assertEqual(brief.evidence_shots[1].source_url, candidate.source_url)

    def test_translated_quantity_matches_value_and_unit_but_not_derived_sum(self):
        source = "The four of us have worked together for 14 to 30 years."
        self.assertTrue(_quantity_supported("30 年", source))
        self.assertFalse(_quantity_supported("90 年", source))

    def test_translated_funding_and_word_duration_quantities_are_supported(self):
        source = "The startup raised $21 million and reached the milestone within three weeks."
        self.assertTrue(_quantity_supported("2100万美元", source))
        self.assertTrue(_quantity_supported("3周", source))
        self.assertFalse(_quantity_supported("6500万美元", source))

    def test_english_calendar_date_matches_chinese_year_and_month(self):
        source = "New pricing takes effect at 16:00 UTC, Aug 16, 2026."
        self.assertTrue(_quantity_supported("2026 年", source))
        self.assertTrue(_quantity_supported("8 月", source))

    def test_opencli_flat_tweet_shape_is_normalized(self):
        payload = [{
            "id": "2085034604172603724", "author": "JeffDean", "name": "Jeff Dean",
            "text": "Announcing Discovery Loop @DiscoLoopAI https://t.co/example", "likes": 10,
            "created_at": "Wed Aug 05 16:06:02 +0000 2026",
            "media_urls": ["https://pbs.twimg.com/media/team.jpg"],
        }, {
            "id": "2085083442669318443", "author": "JeffDean", "name": "Jeff Dean",
            "text": "My last day at Google; now starting @DiscoLoopAI", "created_at": "Wed Aug 05 19:20:06 +0000 2026",
        }]
        normalized = URLAcquirer._normalize_x_payload(payload, "2085034604172603724", "JeffDean")
        root = normalized["data"][0]
        self.assertEqual(root["author"]["screenName"], "JeffDean")
        self.assertEqual(root["author"]["name"], "Jeff Dean")
        self.assertEqual(root["metrics"]["likes"], 10)
        self.assertEqual(root["media"][0]["type"], "photo")
        self.assertTrue(normalized["data"][1]["context_only"])

    def test_root_display_name_is_recovered_from_same_author_quote(self):
        payload = [{
            "id": "2090964822422949999", "author": "thsottiaux", "name": "thsottiaux",
            "text": "The banked reset has landed.",
            "quoted_tweet": {
                "id": "2090947196107764189", "author": "thsottiaux", "name": "Tibo",
                "text": "The banked reset will be there by 8pm PST.",
                "media_urls": ["https://pbs.twimg.com/media/context.jpg"],
            },
        }]
        normalized = URLAcquirer._normalize_x_payload(payload, "2090964822422949999", "thsottiaux")
        self.assertEqual(normalized["data"][0]["author"]["name"], "Tibo")
        self.assertEqual(
            normalized["data"][0]["quoted_tweet"]["media"],
            [{"type": "photo", "url": "https://pbs.twimg.com/media/context.jpg"}],
        )

    def test_x_capture_failure_preserves_opencli_exit_diagnostics(self):
        failure = subprocess.CalledProcessError(
            69, ["opencli", "twitter", "thread"],
            output="trace retained", stderr="browser automation permission denied",
        )
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            with (
                patch("video_factory.acquisition.subprocess.run", side_effect=failure),
                patch.object(URLAcquirer, "_fetch_readable", side_effect=RuntimeError("reader offline")),
                self.assertRaisesRegex(
                    RuntimeError,
                    r"opencli-twitter-thread: exit 69: browser automation permission denied trace retained",
                ),
            ):
                URLAcquirer(workspace)._acquire_x(
                    "https://x.com/JeffDean/status/2085034604172603724", Path(temp),
                )

    def test_x_capture_falls_back_to_public_snapshot_without_browser_session(self):
        snapshot = '''Title: Jeff Dean (@JeffDean) on X
Published Time: 2026-08-05T16:06:02.000Z
Markdown Content:
## Post
* [Jeff Dean](https://x.com/JeffDean) [@JeffDean](https://x.com/JeffDean)  Announcing Discovery Loop with Sanjay, Oriol and Quoc. Learn more at [discoveryloop.com](https://discoveryloop.com/) [![Image 3](https://pbs.twimg.com/media/team.jpg)](https://x.com/JeffDean/status/2085034604172603724/photo/1) [4:06 PM · Aug 5, 2026](https://x.com/JeffDean/status/2085034604172603724)
'''
        failure = subprocess.CalledProcessError(69, ["opencli"], stderr="permission denied")
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            acquirer = URLAcquirer(workspace)
            with (
                patch("video_factory.acquisition.subprocess.run", side_effect=failure),
                patch.object(acquirer, "_fetch_readable", return_value=(snapshot.encode(), "text/plain", "jina-reader")),
                patch.object(acquirer, "_archive_x_media"),
            ):
                result = acquirer._acquire_x(
                    "https://x.com/JeffDean/status/2085034604172603724", Path(temp),
                )

            self.assertEqual(result.method, "jina-reader-x-fallback")
            self.assertEqual(result.ingest.candidate.author, "JeffDean")
            self.assertIn("Announcing Discovery Loop", result.ingest.evidence[0].quote)
            self.assertIn("discoveryloop.com", result.ingest.evidence[0].quote)

    def test_x_capture_backs_off_once_before_switching_backend(self):
        failure = subprocess.CalledProcessError(69, ["opencli"], stderr="extension busy")
        success = subprocess.CompletedProcess(
            ["opencli"], 0,
            stdout='[{"id":"2086470022772457950","author":"sama","name":"Sam Altman",'
                   '"text":"tibo is great","created_at":"Sun Aug 09 15:09:52 +0000 2026"}]',
            stderr="",
        )
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            acquirer = URLAcquirer(workspace)
            with (
                patch("video_factory.acquisition.subprocess.run", side_effect=[failure, success]) as run,
                patch("video_factory.acquisition.time.sleep") as sleep,
                patch.object(acquirer, "_fetch_readable") as public_reader,
                patch.object(acquirer, "_archive_x_media"),
            ):
                result = acquirer._acquire_x(
                    "https://x.com/sama/status/2086470022772457950", Path(temp),
                )

            self.assertEqual(result.method, "opencli-twitter-thread")
            self.assertEqual(run.call_count, 2)
            sleep.assert_called_once_with(0.75)
            public_reader.assert_not_called()
            capture = json.loads(result.artifact.read_text(encoding="utf-8"))
            self.assertEqual(capture["selected_backend"], "opencli-twitter-thread")
            self.assertEqual(
                [item["status"] for item in capture["acquisition_trace"]],
                ["retryable_failure", "backoff", "succeeded"],
            )

    def test_source_kind_and_story_topic_are_routed_independently(self):
        cases = [
            (Candidate("x-company", SourceType.TWEET, "https://x.com/JeffDean/status/1", "We are founding Discovery Loop", author="JeffDean"), "founding a company and leaving Google", TopicType.COMPANY_OR_TEAM, ContentType.FLASH),
            (Candidate("x-practice", SourceType.TWEET, "https://x.com/dev/status/2", "My coding workflow"), "I tried this workflow once", TopicType.PRACTICE_POST, ContentType.FLASH),
            (Candidate("x-research", SourceType.TWEET, "https://x.com/researcher/status/3", "We found an API vulnerability"), "We verified the experiment across frontier models", TopicType.RESEARCH_OR_BENCHMARK, ContentType.DEEP_DIVE),
            (Candidate("x-update", SourceType.TWEET, "https://x.com/employee/status/4", "The banked reset has landed"), "For all paid users", TopicType.OFFICIAL_ANNOUNCEMENT, ContentType.FLASH),
            (Candidate("x-model", SourceType.TWEET, "https://x.com/lab/status/5", "GLM-5.3-Flash: we released open-source native FP8 model weights"), "320B parameters on Hugging Face", TopicType.MODEL_OR_PRODUCT, ContentType.EXPLAINER),
            (Candidate("tool", SourceType.WEB, "https://developers.cloudflare.com/agents/", "Agents SDK docs"), "Quick Start SDK API reference", TopicType.TOOL_SDK_AGENT, ContentType.EXPLAINER),
            (Candidate("model", SourceType.WEB, "https://www.anthropic.com/product", "Introducing Claude Opus"), "The new model is available today", TopicType.MODEL_OR_PRODUCT, ContentType.EXPLAINER),
            (Candidate("company", SourceType.WEB, "https://example.com/team", "New founding team"), "The seed round backs a new company", TopicType.COMPANY_OR_TEAM, ContentType.FLASH),
            (Candidate("paper", SourceType.PAPER, "https://example.com/report.pdf", "Technical report"), "Benchmark methodology and dataset", TopicType.RESEARCH_OR_BENCHMARK, ContentType.DEEP_DIVE),
            (Candidate("announcement", SourceType.WEB, "https://vendor.com/news/change", "API migration"), "Old endpoint is retired; migration effective tomorrow", TopicType.OFFICIAL_ANNOUNCEMENT, ContentType.FLASH),
        ]
        for candidate, text, topic, content_type in cases:
            with self.subTest(candidate=candidate.id):
                decision = route_content(candidate, [evidence(candidate, text)])
                self.assertEqual(decision.topic_type, topic)
                self.assertEqual(decision.content_type, content_type)
                if content_type == ContentType.FLASH:
                    self.assertLessEqual(decision.target_duration, 15)

    def test_news_funding_amount_routes_to_company_story_before_agent_terms(self):
        candidate = Candidate(
            "runable", SourceType.WEB,
            "https://techcrunch.com/2026/08/26/runable-hits-21m/",
            "Runable hits $21M to bet AI agents can grow businesses",
        )
        item = Evidence(
            "runable-page", candidate.id, candidate.source_url,
            "The startup raised $21 million in a Series A at a $65 million post-money valuation. "
            "Its AI agent builds websites and runs customer-growth workflows.",
            "web:primary_page",
        )

        route = route_content(candidate, [item])

        self.assertEqual(route.topic_type, TopicType.COMPANY_OR_TEAM)
        self.assertEqual(route.content_type, ContentType.FLASH)

    def test_official_double_blind_evaluation_routes_to_research(self):
        candidate = Candidate(
            "deepmind-eval", SourceType.WEB,
            "https://deepmind.google/blog/piloting-the-worlds-first-double-blind-ai-evaluations",
            "Piloting the world's first double-blind AI evaluations",
        )
        item = Evidence(
            "deepmind-page", candidate.id, candidate.source_url,
            "The evaluation addresses benchmark contamination with a double-blind methodology. "
            "A technical report explains the experiment and cryptographic environment.",
            "web:primary_page",
        )

        route = route_content(candidate, [item])

        self.assertEqual(route.topic_type, TopicType.RESEARCH_OR_BENCHMARK)
        self.assertEqual(route.content_type, ContentType.DEEP_DIVE)

    def test_non_github_prompt_never_exposes_low_level_scene_schema(self):
        candidate = Candidate("tool", SourceType.WEB, "https://example.com/docs", "Tool docs")
        packet = StoryWriterPacket(candidate, [evidence(candidate)], TopicType.TOOL_SDK_AGENT, ContentType.EXPLAINER, 28)
        prompt = packet.prompt()
        self.assertIn('"editorial_brief"', prompt)
        self.assertNotIn('"material_role"', prompt)
        self.assertNotIn('"recording_cues"', prompt)
        self.assertNotIn('"scenes"', prompt)
        self.assertIn('"director_brief"', prompt)
        self.assertIn("Investigate like an editor", prompt)

    def test_model_prompt_requires_exact_name_and_plain_metric_explanation(self):
        candidate = Candidate("model", SourceType.WEB, "https://example.com/model", "GLM-5.3-Flash")
        prompt = StoryWriterPacket(
            candidate, [evidence(candidate)], TopicType.MODEL_OR_PRODUCT, ContentType.EXPLAINER, 28,
        ).prompt()

        self.assertIn("persistent title and selected hook must name the exact model", prompt)
        self.assertIn("vibe coder", prompt)
        self.assertIn("refusal rate means", prompt)
        self.assertIn("renderer owns title fitting", prompt)

    def test_model_brief_rejects_hook_that_names_vendor_but_not_model(self):
        candidate = Candidate("model", SourceType.WEB, "https://example.com/model", "GLM-5.3-Flash")
        brief, item = brief_for(candidate, TopicType.MODEL_OR_PRODUCT, ContentType.EXPLAINER)
        brief.subjects = [
            StorySubject("OrcaRouter", "vendor", "发布权重", "开放研究", [item.id]),
            StorySubject("GLM-5.3-Flash", "model", "开放权重", "可供研究", [item.id]),
        ]
        brief.attention_strategy.selected_hook = "OrcaRouter 把拒绝率从 96% 降到 11%"
        brief.attention_strategy.hook_candidates[0] = brief.attention_strategy.selected_hook

        errors = validate_editorial_structure(
            brief, candidate, [item], TopicType.MODEL_OR_PRODUCT, ContentType.EXPLAINER,
        )

        self.assertTrue(any("must name the concrete model subject" in error for error in errors))

    def test_model_opening_explains_refusal_rate_for_vibe_coders(self):
        candidate = Candidate("model", SourceType.WEB, "https://example.com/model", "GLM-5.3-Flash")
        brief, item = brief_for(candidate, TopicType.MODEL_OR_PRODUCT, ContentType.EXPLAINER)
        brief.subjects = [StorySubject(
            "GLM-5.3-Flash", "model", "开放权重", "拒绝率下降", [item.id],
        )]
        brief.attention_strategy.selected_hook = "GLM-5.3-Flash 拒绝率从 96% 降到 11%"
        brief.attention_strategy.hook_candidates[0] = brief.attention_strategy.selected_hook
        brief.evidence_shots[0].fact = "GLM-5.3-Flash 拒绝率降到 11%"
        brief.evidence_shots[0].audience_copy = ""

        errors = validate_editorial_structure(
            brief, candidate, [item], TopicType.MODEL_OR_PRODUCT, ContentType.EXPLAINER,
        )
        self.assertTrue(any("must explain specialist metric 拒绝率" in error for error in errors))

        brief.evidence_shots[0].audience_copy = "拒绝率就是模型直接拒绝回答的问题占比。"
        errors = validate_editorial_structure(
            brief, candidate, [item], TopicType.MODEL_OR_PRODUCT, ContentType.EXPLAINER,
        )
        self.assertFalse(any("must explain specialist metric 拒绝率" in error for error in errors))

    def test_model_canonicalization_selects_existing_model_named_hook(self):
        candidate = Candidate("model", SourceType.WEB, "https://example.com/model", "GLM-5.3-Flash")
        brief, item = brief_for(candidate, TopicType.MODEL_OR_PRODUCT, ContentType.EXPLAINER)
        brief.subjects = [
            StorySubject("OrcaRouter", "vendor", "发布权重", "开放研究", [item.id]),
            StorySubject("GLM-5.3-Flash", "model", "开放权重", "可供研究", [item.id]),
        ]
        brief.attention_strategy.hook_candidates = [
            "OrcaRouter 把拒答率从 96% 降到 11%",
            "GLM-5.3-Flash 拒答率从 96% 降到 11%",
            "GLM-5.3-Flash 的部分拒绝仍未归零",
        ]
        brief.attention_strategy.selected_hook = brief.attention_strategy.hook_candidates[0]

        canonicalize_editorial_brief(brief, [item])

        self.assertIn("GLM-5.3-Flash", brief.attention_strategy.selected_hook)

    def test_canonicalization_injects_plain_refusal_metric_glossary(self):
        candidate = Candidate("model", SourceType.WEB, "https://example.com/model", "GLM-5.3-Flash")
        brief, item = brief_for(candidate, TopicType.MODEL_OR_PRODUCT, ContentType.EXPLAINER)
        brief.attention_strategy.selected_hook = "GLM-5.3-Flash 拒答率从 96% 降到 11%"
        brief.attention_strategy.hook_candidates[0] = brief.attention_strategy.selected_hook
        brief.evidence_shots[0].audience_copy = "320B 参数，原生 FP8。"

        canonicalize_editorial_brief(brief, [item])

        self.assertEqual(
            brief.evidence_shots[0].audience_copy,
            "拒答率：模型直接拒绝回答的请求占比。",
        )

    def test_common_metrics_receive_one_plain_definition_at_first_use(self):
        candidate = Candidate("model", SourceType.WEB, "https://example.com/model", "Model")
        cases = {
            "幻觉率降到 3%": "幻觉率：模型生成错误或无依据内容的占比。",
            "激活参数只有 18B": "激活参数：模型每次生成时实际参与计算的参数规模。",
            "原生 FP8 权重": "FP8 量化：用更低数值精度减少模型的显存和计算占用。",
            "TTFT 降到 100ms": "TTFT：发出请求到看到第一个 Token 的等待时间。",
            "Arena 分上升": "Arena 分：模型对战评测按胜负换算的相对分数。",
            "SWE-bench pass@1 达到 50%": "pass@1：代码只生成一次就通过测试的比例。",
        }
        for visible, expected in cases.items():
            with self.subTest(term=visible):
                brief, item = brief_for(candidate, TopicType.MODEL_OR_PRODUCT, ContentType.EXPLAINER)
                brief.evidence_shots[0].fact = visible
                brief.evidence_shots[0].audience_copy = ""
                canonicalize_editorial_brief(brief, [item])
                self.assertEqual(brief.evidence_shots[0].audience_copy, expected)

    def test_common_developer_terms_do_not_consume_the_single_glossary_slot(self):
        candidate = Candidate("model", SourceType.WEB, "https://example.com/model", "Model")
        brief, item = brief_for(candidate, TopicType.MODEL_OR_PRODUCT, ContentType.EXPLAINER)
        brief.evidence_shots[0].fact = "上下文窗口扩大到 1M，吞吐量达到 200 TPS"
        brief.evidence_shots[0].audience_copy = "开发者可直接比较长文本和并发性能。"

        canonicalize_editorial_brief(brief, [item])

        self.assertEqual(brief.evidence_shots[0].audience_copy, "开发者可直接比较长文本和并发性能。")

    def test_video_explains_only_the_hardest_term_unless_hook_names_one(self):
        candidate = Candidate("model", SourceType.WEB, "https://example.com/model", "Model")
        brief, item = brief_for(candidate, TopicType.MODEL_OR_PRODUCT, ContentType.EXPLAINER)
        brief.evidence_shots[0].fact = "原生 FP8 权重，拒绝率降到 11%"
        brief.evidence_shots[0].audience_copy = ""
        brief.evidence_shots[1].fact = "激活参数只有 18B"
        brief.evidence_shots[1].audience_copy = ""
        canonicalize_editorial_brief(brief, [item])
        explanations = [shot.audience_copy for shot in brief.evidence_shots if "：" in shot.audience_copy]
        self.assertEqual(explanations, ["FP8 量化：用更低数值精度减少模型的显存和计算占用。"])

        brief.evidence_shots[0].audience_copy = ""
        brief.attention_strategy.selected_hook = "模型拒绝率降到 11%"
        brief.attention_strategy.hook_candidates[0] = brief.attention_strategy.selected_hook
        canonicalize_editorial_brief(brief, [item])
        self.assertEqual(
            brief.evidence_shots[0].audience_copy,
            "拒绝率：模型直接拒绝回答的请求占比。",
        )

    def test_company_and_research_prompts_include_channel_specific_editorial_contracts(self):
        candidate = Candidate("web", SourceType.WEB, "https://example.com/story", "Story")
        item = evidence(candidate)

        company_prompt = StoryWriterPacket(
            candidate, [item], TopicType.COMPANY_OR_TEAM, ContentType.FLASH, 12,
        ).prompt()
        research_prompt = StoryWriterPacket(
            candidate, [item], TopicType.RESEARCH_OR_BENCHMARK, ContentType.DEEP_DIVE, 50,
        ).prompt()

        self.assertIn("funding is evidence of a bet, not the whole story", company_prompt)
        self.assertIn("Distinguish a protocol, pilot, or proposed method", research_prompt)

    def test_contextual_flash_is_blocked_when_it_has_no_retention_cadence(self):
        candidate = Candidate("x-company", SourceType.TWEET, "https://x.com/JeffDean/status/1", "Discovery Loop", author="JeffDean")
        brief, item = brief_for(candidate)
        reason = SelectionReason("important-person", "important_person", "Jeff Dean离开影响技术圈", [item.id])
        brief.opportunity = EditorialOpportunity(
            "Jeff Dean创业", "关键人物发生组织变化", "开发者关心AI人才流动", "行业格局", [reason], story_archetype="people_change",
        )
        brief.context_graph = ContextGraph()
        brief.evidence_shots[0].translation = ""
        brief.evidence_shots[0].full_translation = ""
        brief.director_brief = DirectorBrief(
            "Google失去长期协作团队", "另外三个人是谁", "conflict", 2, [],
            [StoryArcBeat("event", "Jeff Dean创业", "先建立事件", [item.id], [reason.id])], 12, "关键人物离开Google",
        )
        errors = validate_editorial_brief(brief, candidate, [item], TopicType.COMPANY_OR_TEAM, ContentType.FLASH)
        self.assertIn("high-retention flash needs at least three semantic visual changes", errors)
        self.assertIn("high-retention flash needs at least three visual families", errors)
        self.assertIn("a non-Chinese root post requires adjacent readable Chinese translation in the first shot", errors)

    def test_flash_rejects_a_fixed_conclusion_that_will_be_visually_truncated(self):
        candidate = Candidate("x-company", SourceType.TWEET, "https://x.com/dev/status/1", "Post", author="dev")
        brief, item = brief_for(candidate)
        brief.fixed_conclusion = "这是一条故意写得非常非常长的固定结论，它不断重复事件、背景、能力、影响和判断，最终一定会超过底部固定栏能够完整显示的范围，因此程序必须在渲染之前拒绝它。"
        errors = validate_editorial_brief(
            brief, candidate, [item], TopicType.COMPANY_OR_TEAM, ContentType.FLASH,
        )
        self.assertTrue(any("persistent bottom rail" in error for error in errors))

    def test_docs_page_cannot_be_upgraded_to_launch_news(self):
        candidate = Candidate("tool", SourceType.WEB, "https://example.com/docs", "Agents docs")
        brief, item = brief_for(candidate, TopicType.TOOL_SDK_AGENT, ContentType.EXPLAINER)
        item.quote = "Build and deploy AI Agents that autonomously perform tasks."
        brief.attention_strategy.hook_fact = "Cloudflare Agents 正式上线"
        brief.attention_strategy.conflict = "开发者不再手写会话状态"
        brief.attention_strategy.stakes = "Agent 开发工作流发生变化"
        brief.attention_strategy.stance = "先用官方能力替换重复基础设施"
        brief.attention_strategy.payoff = "Agent 状态管理可以交给平台"
        brief.attention_strategy.hook_candidates = [
            "Cloudflare Agents 正式上线", "Cloudflare 推出 Agents 平台", "Agent 会话基础设施可以少写一层",
        ]
        brief.attention_strategy.selected_hook = "Cloudflare Agents 正式上线"
        errors = validate_editorial_brief(
            brief, candidate, [item], TopicType.TOOL_SDK_AGENT, ContentType.EXPLAINER,
        )
        self.assertTrue(any("release/launch wording" in error for error in errors))

    def test_landed_is_explicit_rollout_evidence_for_chinese_online_wording(self):
        candidate = Candidate("tool", SourceType.TWEET, "https://x.com/dev/status/1", "Feature landed", author="dev")
        brief, item = brief_for(candidate, TopicType.OFFICIAL_ANNOUNCEMENT, ContentType.FLASH)
        item.quote = "The banked reset has landed."
        brief.attention_strategy.hook_fact = "banked reset 已上线"
        brief.attention_strategy.hook_candidates = ["banked reset 已上线", "banked reset 已落地", "banked reset 已到账"]
        brief.attention_strategy.selected_hook = "banked reset 已上线"
        brief.attention_strategy.hook_evidence_ids = [item.id]
        errors = validate_editorial_brief(
            brief, candidate, [item], TopicType.OFFICIAL_ANNOUNCEMENT, ContentType.FLASH,
        )
        self.assertFalse(any("release/launch wording" in error for error in errors))

    def test_complete_tweet_card_renders_as_one_frame(self):
        candidate = Candidate(
            "tweet", SourceType.TWEET, "https://x.com/dev/status/1", "Post", author="dev",
            published_at="2026-08-21", metadata={"author_name": "Developer", "metrics": {"likes": 42}},
        )
        root = Evidence("tweet-root", candidate.id, candidate.source_url, "One complete original post with its full claim and context.", "x:post")
        scene = Scene(
            "scene-1", 0, 4, "", "完整原帖一次呈现", [root.id], MaterialRole.PROOF,
            "hold complete post", screen_fact="作者给出了一个可验证的完整主张",
        )
        with TemporaryDirectory() as directory:
            output = render_tweet_card(candidate, root, scene, Path(directory) / "tweet.png")
            from PIL import Image
            with Image.open(output) as image:
                self.assertEqual(image.size, (1384, 1602))
                # The complete post and its adjacent translation are the
                # first-screen explanation; do not repeat scene copy in a
                # separate dark subtitle chip at the bottom.
                self.assertEqual(image.getpixel((700, 1450)), (255, 255, 255))

    def test_program_compiles_semantic_card_family_into_material_kind(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/dev/status/1", "Post", author="dev")
        root = Evidence("root", candidate.id, candidate.source_url, "Feature landed", "x:thread_post")
        raw = {"kind": "impact_card", "visual_family": "impact_card", "evidence_ids": [root.id]}
        self.assertEqual(_compile_evidence_shot_kind(raw, {root.id: root}), EvidenceShotKind.TWEET_CARD)

    def test_root_tweet_family_wins_over_attached_image_material(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/dev/status/1", "Post", author="dev")
        root = Evidence("root", candidate.id, candidate.source_url, "Complete root post", "x:thread_post")
        image = Evidence("image", candidate.id, "https://pbs.twimg.com/context.jpg", "Attached context", "x:media_photo")
        raw = {"visual_family": "tweet", "evidence_ids": [root.id, image.id]}
        self.assertEqual(
            _compile_evidence_shot_kind(raw, {root.id: root, image.id: image}),
            EvidenceShotKind.TWEET_CARD,
        )

    def test_flash_presentation_compiler_preserves_real_source_then_adds_payoff_card(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/dev/status/1", "Post", author="dev")
        brief, root = brief_for(candidate)
        brief.evidence_shots.append(EvidenceShot(
            "payoff", EvidenceShotKind.BROWSER_SECTION, "最后得到什么", "并行执行数千实验",
            "实验循环成为真正的产品方向", [root.id], ["impact"], candidate.source_url,
            "A concrete verified capability", "具体能力", 4, "落到结果", "official_page", "payoff",
        ))
        brief.evidence_shots[0].visual_family = "tweet"
        brief.evidence_shots[0].retention_job = "hook_proof"
        brief.evidence_shots[1].visual_family = "official_page"
        brief.evidence_shots[1].retention_job = "reveal"
        canonicalize_editorial_brief(brief, [root])
        self.assertEqual(brief.evidence_shots[1].visual_family, "official_page")
        self.assertEqual(brief.evidence_shots[2].visual_family, "impact_card")
        self.assertLessEqual(sum(item.duration for item in brief.evidence_shots), 12)

    def test_evidence_backed_impact_card_has_native_vertical_size(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/dev/status/1", "Post", author="dev")
        root = Evidence("root", candidate.id, candidate.source_url, "Feature landed", "x:thread_post")
        scene = Scene(
            "scene", 0, 2.8, "", "覆盖所有付费用户", [root.id], MaterialRole.PROOF,
            "compiled impact card", screen_fact="覆盖所有付费用户",
            screen_interpretation="此前预告，随后确认落地", visual_family="impact_card",
        )
        with TemporaryDirectory() as directory:
            output = render_editorial_card(scene, root, Path(directory) / "impact.png")
            from PIL import Image
            with Image.open(output) as image:
                self.assertEqual(image.size, (1384, 1602))

    def test_flash_rejects_brand_translation_limit_invention_and_duplicate_payoff(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/dev/status/1", "banked reset has landed", author="dev")
        item = Evidence("root", candidate.id, candidate.source_url, "The banked reset has landed. For all paid users.", "x:thread_post")
        reason = SelectionReason("change", "capability_shift", "功能已落地", [item.id])
        shots = [
            EvidenceShot("s1", EvidenceShotKind.TWEET_CARD, "发生什么", "banked reset 已落地", "功能已经可用", [item.id], ["event"], visual_family="tweet", retention_job="hook_proof", selection_reason_ids=[reason.id], full_translation="banked reset 已落地，覆盖所有付费用户。"),
            EvidenceShot("s2", EvidenceShotKind.TWEET_CARD, "覆盖谁", "银行级重置覆盖所有付费用户", "此前预告，随后落地", [item.id], ["effective_scope"], relation_to_previous="此前先预告，随后落地", visual_family="quote_card", retention_job="impact", selection_reason_ids=[reason.id]),
            EvidenceShot("s3", EvidenceShotKind.TWEET_CARD, "最后结论", "银行级重置已覆盖全部付费用户", "功能没有时间限制", [item.id], ["effective_scope"], relation_to_previous="最终确认范围", visual_family="impact_card", retention_job="payoff", selection_reason_ids=[reason.id]),
        ]
        brief = EditorialBrief(
            "banked reset 已落地", "覆盖所有付费用户", "所有付费用户现已覆盖",
            AttentionStrategy("功能落地", "此前预告随后落地", "", "覆盖付费用户", "按时兑现", "所有付费用户可用", ["banked reset 已落地", "付费用户全部覆盖", "此前预告如今兑现"], [item.id], "banked reset 已落地"),
            [StorySubject("开发团队", "team", "确认功能落地", "覆盖付费用户", [item.id])], [], shots, 12,
            opportunity=EditorialOpportunity("功能落地", "刚刚发生", "覆盖技术用户", "确认可用范围", [reason]),
            context_graph=ContextGraph(),
            director_brief=DirectorBrief("功能落地", "覆盖谁", "relief", 2, [], [StoryArcBeat("event", "功能落地", "开场", [item.id], [reason.id])], 12, "功能落地"),
        )
        errors = validate_editorial_brief(brief, candidate, [item], TopicType.OFFICIAL_ANNOUNCEMENT, ContentType.FLASH)
        self.assertTrue(any("branded feature name" in error for error in errors))
        self.assertTrue(any("absence-of-limit" in error for error in errors))
        self.assertTrue(any("near-duplicates" in error for error in errors))

    def test_retention_compiler_selects_the_stronger_model_written_hook(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/dev/status/1", "banked reset", author="dev")
        brief, item = brief_for(candidate)
        item.quote = "The banked reset has landed."
        brief.attention_strategy.hook_candidates = [
            "OpenAI banked reset正式可用",
            "banked reset已落地，付费用户可以使用",
            "OpenAI兑现承诺：banked reset准时落地",
        ]
        brief.attention_strategy.selected_hook = brief.attention_strategy.hook_candidates[0]
        canonicalize_editorial_brief(brief, [item])
        self.assertEqual(brief.attention_strategy.selected_hook, "OpenAI兑现承诺：banked reset准时落地")

    def test_x_root_actor_hook_beats_a_detached_background_hook(self):
        candidate = Candidate("tweet", SourceType.TWEET, "https://x.com/sama/status/1", "Sam reply", author="sama")
        brief, item = brief_for(candidate)
        item.source_kind = "x:thread_post"
        item.metadata.update({"author_name": "Sam Altman", "author_handle": "sama"})
        brief.attention_strategy.hook_candidates = [
            "Sam Altman 点名 Tibo，Anthropic 招聘回应突然入场",
            "用户把 Anthropic harness 接其他模型后账号被停用",
            "封号争议下，评论区出现了一句招聘邀请",
        ]
        brief.attention_strategy.selected_hook = brief.attention_strategy.hook_candidates[1]
        canonicalize_editorial_brief(brief, [item])
        self.assertEqual(brief.attention_strategy.selected_hook, brief.attention_strategy.hook_candidates[0])

    def test_low_level_non_github_model_output_is_rejected(self):
        candidate = Candidate("tool", SourceType.WEB, "https://example.com/docs", "Tool docs")
        packet = StoryWriterPacket(candidate, [evidence(candidate)], TopicType.TOOL_SDK_AGENT, ContentType.EXPLAINER, 28)
        with self.assertRaisesRegex(ValueError, "editorial_brief"):
            OpenAICompatibleStoryWriter._to_storyboard_request(packet, {"footer": "结论", "scenes": [{"material_role": "trial"}]})

    def test_editorial_brief_compiles_to_deterministic_roles_and_cues(self):
        candidate = Candidate("x-company", SourceType.TWEET, "https://x.com/JeffDean/status/1", "Discovery Loop", author="JeffDean")
        brief, item = brief_for(candidate)
        self.assertEqual(validate_editorial_brief(brief, candidate, [item], TopicType.COMPANY_OR_TEAM, ContentType.FLASH), [])
        scenes = compile_evidence_shots(brief, candidate)
        self.assertEqual(scenes[0].material_role.value, "proof")
        self.assertEqual(scenes[0].recording_cues[0].instruction, "hold the complete original post")
        self.assertNotIn("trial", scenes[0].visual_action)
        self.assertNotIn("boundary", scenes[0].visual_action)

    def test_related_primary_post_is_attached_to_a_supported_departure_hook(self):
        candidate = Candidate("x-company", SourceType.TWEET, "https://x.com/JeffDean/status/1", "Discovery Loop", author="JeffDean")
        brief, root = brief_for(candidate)
        root.quote = "Announcing Discovery Loop"
        related = Evidence("e-related", candidate.id, "https://x.com/JeffDean/status/2", "Tomorrow is my last day at Google after 27 years", "x:related_post")
        brief.attention_strategy.hook_evidence_ids = [root.id]
        canonicalize_editorial_brief(brief, [root, related])
        self.assertEqual(brief.attention_strategy.hook_evidence_ids, [root.id, related.id])

    def test_billing_correlation_is_not_rewritten_as_vulnerability_mechanism(self):
        candidate = Candidate("x-security", SourceType.TWEET, "https://x.com/dev/status/1", "API research", author="dev")
        brief, item = brief_for(candidate)
        item.quote = "We used a vulnerability in frontier APIs. Reasoning token count matches billed thinking tokens 1:1."
        brief.attention_strategy.hook_fact = "研究者利用 API 计费漏洞提取隐藏推理"
        brief.attention_strategy.hook_candidates = [
            "研究者利用 API 计费漏洞提取隐藏推理", "前沿模型 API 暴露隐藏推理", "计费 token 1:1 验证了提取结果",
        ]
        brief.attention_strategy.selected_hook = brief.attention_strategy.hook_candidates[0]
        errors = validate_editorial_brief(brief, candidate, [item], TopicType.RESEARCH_OR_BENCHMARK, ContentType.EXPLAINER)
        self.assertTrue(any("billing-token correlation" in error for error in errors))

    def test_ritual_unknown_footer_uses_existing_evidence_backed_payoff(self):
        candidate = Candidate("x-company", SourceType.TWEET, "https://x.com/JeffDean/status/1", "Discovery Loop", author="JeffDean")
        brief, item = brief_for(candidate)
        brief.fixed_conclusion = "这件事很重要，但具体技术细节和影响范围尚未完全公开。"
        errors = validate_editorial_brief(brief, candidate, [item], TopicType.COMPANY_OR_TEAM, ContentType.FLASH)
        self.assertTrue(any("fixed conclusion is bureaucratic" in error or "deliver impact/payoff" in error for error in errors))

    def test_ritual_payoff_falls_back_to_evidence_bound_subject_impact(self):
        candidate = Candidate("x-company", SourceType.TWEET, "https://x.com/JeffDean/status/1", "Discovery Loop", author="JeffDean")
        brief, item = brief_for(candidate)
        brief.fixed_conclusion = "具体技术细节和影响范围尚未完全公开。"
        brief.attention_strategy.payoff = "完整影响未知，需持续关注。"
        brief.attention_strategy.stance = "值得关注后续影响。"
        errors = validate_editorial_brief(brief, candidate, [item], TopicType.COMPANY_OR_TEAM, ContentType.FLASH)
        self.assertTrue(any("fixed conclusion is bureaucratic" in error for error in errors))

    def test_director_keeps_fixed_rails_and_structured_attention(self):
        candidate = Candidate("x-company", SourceType.TWEET, "https://x.com/JeffDean/status/1", "Discovery Loop", author="JeffDean")
        brief, item = brief_for(candidate)
        scenes = compile_evidence_shots(brief, candidate)
        answers = [
            NarrativeAnswer("identity", "Jeff Dean成立Discovery Loop", [item.id]),
            NarrativeAnswer("product_direction", "自动化机器学习实验", [item.id]),
            NarrativeAnswer("impact", "自动化科研竞争转向系统能力", [item.id]),
        ]
        from video_factory.director import StoryboardRequest
        request = StoryboardRequest(
            "story-x", candidate, TopicType.COMPANY_OR_TEAM, ContentType.FLASH,
            [item], brief.fixed_conclusion, answers, scenes, 12,
            editorial_brief=brief, fixed_hook=brief.attention_strategy.selected_hook,
        )
        manifest = StoryboardDirector().direct(request)
        self.assertEqual(manifest.fixed_title, brief.attention_strategy.selected_hook)
        self.assertEqual(manifest.fixed_footer, brief.fixed_conclusion)
        self.assertIs(manifest.editorial_brief, brief)
        request = WebScrollVideoAdapter.editorial_story_request(
            candidate.source_url, manifest, __import__("pathlib").Path("out.mp4"), __import__("pathlib").Path("frames"),
        )
        self.assertEqual(request.url, candidate.source_url)
        failed = {check.name for check in validate_manifest(manifest) if not check.passed}
        self.assertNotIn("attention_strategy", failed)
        self.assertNotIn("duration_band", failed)


if __name__ == "__main__":
    unittest.main()
