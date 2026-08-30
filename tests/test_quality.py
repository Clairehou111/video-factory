import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PIL import Image, ImageDraw

from video_factory.capture import BrowserCaptureImporter, BrowserCaptureRequest, CaptureKind, CapturedBrowserArtifact
from video_factory.director import NarrativeAnswer, SceneProposal, StoryboardDirector, StoryboardRequest
from video_factory.ingest import GitHubIngestor, TwitterCliIngestor
from video_factory.models import CaptureCue, CueAction
from video_factory.models import (
    Candidate, ContentType, Evidence, GitHubFocusCandidate, GitHubModuleFocus,
    GitHubProjectBrief, GitHubWalkthrough, MaterialRole, RenderManifest, Scene,
    SourceType, StoryBeat, TopicType,
)
from video_factory.mpt import MPTAssemblyAdapter, MPTSettings, NativeFFmpegAssemblyAdapter
from video_factory.compositor import resolve_font_path
from video_factory.media import VideoProbe
from video_factory.publish import PublicationState, PublishDraft, WeChatVideoAccountSelectors, WeChatVideoAccountUploader, prepare_publish_draft
from video_factory.quality import is_publishable, validate_manifest
from video_factory.storage import Workspace
from video_factory.narrative import extract_external_urls, route_external_source
from video_factory.webcapture import WebCaptureRequest, WebScrollVideoAdapter, WebScrollVideoSettings
from video_factory.writer import StoryWriterPacket
from video_factory.safety import review_evidence
from video_factory.llm import OpenAICompatibleStoryWriter
from video_factory.github_editor import canonicalize_github_brief, select_github_focuses, validate_github_brief
from video_factory.tweetcard import _tweet_translation_copy


class QualityTest(unittest.TestCase):
    @staticmethod
    def github_brief(metadata_id: str, readme_id: str) -> GitHubProjectBrief:
        return GitHubProjectBrief(
            project_kind="cli_sdk",
            core_job="把一个重复任务自动化",
            input_output="输入主题，输出结果",
            adoption_path="运行 Quick Start 命令",
            unique_edge="一个命令覆盖完整链路",
            boundary="输出仍需人工检查",
            verdict="适合需要批量初稿的人",
            repo_description_target="Automate a repeated task",
            readme_claim_target="Turn one topic into a finished result",
            file_tree_target="README.md",
            hook_strategy="conflict",
            hook_stance="太敢了",
            hook_fact="重复任务现在能直接自动化",
            hook_evidence_ids=[readme_id],
            project_title="acme-demo｜重复任务自动化",
            hook_opening="重复任务终于能一条命令自动化",
            hook_reveal="输入主题就能直接输出完整结果",
            hook_verdict="省掉重复劳动，才是它真正的价值",
            subject_name="acme-demo",
            subject_type="project",
            subject_action="把重复任务自动化",
            subject_consequence="让开发者省掉重复劳动",
            focus_candidates=[
                GitHubFocusCandidate(
                    "trial", "trial", "Quick Start command", "最小试用", "能马上验证",
                    [readme_id], viewer_value=3, visual_proof=3, distinctiveness=2, actionability=3,
                ),
                GitHubFocusCandidate(
                    "boundary", "boundary", "Review output before publishing", "人工边界", "避免盲用",
                    [readme_id], viewer_value=2, visual_proof=3, distinctiveness=2, risk_importance=3,
                ),
                GitHubFocusCandidate(
                    "edge", "technical_edge", "Plugin architecture", "插件机制", "解释扩展性",
                    [readme_id], viewer_value=1, visual_proof=2, distinctiveness=2,
                ),
            ],
            selected_focus_ids=["trial", "edge"],
        )

    def test_github_file_tree_target_must_be_visible_on_repo_home(self) -> None:
        metadata = Evidence(
            "meta", "github-acme-demo", "https://github.com/acme/demo",
            "Automate a repeated task", "github:metadata",
        )
        readme = Evidence(
            "readme", "github-acme-demo", "https://github.com/acme/demo#readme",
            "Turn one topic into a finished result. Quick Start command. "
            "Review output before publishing. Plugin architecture.",
            "github:readme",
        )
        brief = self.github_brief(metadata.id, readme.id)
        brief.file_tree_target = "service/scripts/clean_file.py"
        errors = validate_github_brief(brief, [metadata, readme])
        self.assertTrue(any("top-level file or directory" in error for error in errors))

    def test_github_table_row_uses_one_real_browser_cell(self) -> None:
        metadata = Evidence(
            "meta", "github-acme-demo", "https://github.com/acme/demo",
            "Automate a repeated task", "github:metadata",
        )
        readme = Evidence(
            "readme", "github-acme-demo", "https://github.com/acme/demo#readme",
            "Turn one topic into a finished result. Quick Start command.\n"
            "| mode | behavior |\n| --- | --- |\n"
            "| clean | Strips marks in place and reports that the file changed. |\n"
            "Review output before publishing. Plugin architecture.",
            "github:readme",
        )
        brief = self.github_brief(metadata.id, readme.id)
        brief.focus_candidates[0].target = (
            "clean | Strips marks in place and reports that the file changed."
        )
        brief.focus_candidates[0].browser_target = (
            "clean | Strips marks in place and reports that the file changed."
        )
        canonicalize_github_brief(brief, [metadata, readme])
        self.assertEqual(
            brief.focus_candidates[0].browser_target,
            "Strips marks in place and reports that the file changed.",
        )
    def test_flash_manifest_is_traceable_and_respects_footer_policy(self) -> None:
        candidate = Candidate(
            id="candidate-1", source_type=SourceType.TWEET, source_url="https://x.com/example/status/1", title="example"
        )
        evidence = Evidence(
            id="e-1", candidate_id=candidate.id, url=candidate.source_url, quote="A source-backed claim", source_kind="tweet"
        )
        manifest = RenderManifest(
            id="render-1",
            candidate_id=candidate.id,
            content_type=ContentType.FLASH,
            source_urls=[candidate.source_url],
            evidence=[evidence],
            fixed_footer="Codex 不只回答，开始接手真实任务",
            scenes=[
                Scene("s-1", 0, 4, "发生了什么", "事实", [evidence.id], MaterialRole.PROOF, "show tweet"),
                Scene("s-2", 4, 8, "为什么重要", "结论", [evidence.id], MaterialRole.EXPLANATION, "highlight conclusion", highlight_translation="结论重点"),
            ],
            music_license_status="verified", license_records=[{"track": "test", "license": "original"}],
        )
        self.assertTrue(is_publishable(validate_manifest(manifest)))

    def test_director_pads_deep_dive_holds_to_format_floor(self) -> None:
        candidate = Candidate(
            "research-1", SourceType.OFFICIAL_ANNOUNCEMENT,
            "https://example.com/research", "Research evaluation",
        )
        evidence = Evidence("e-1", candidate.id, candidate.source_url, "Verified protocol", "web:page")
        beat_ids = [
            "research_question", "method", "primary_artifact", "conditions",
            "scope", "unproven", "recommendation",
        ]
        proposals = [
            SceneProposal(
                f"proof-{index}", "note", f"fact-{index}", MaterialRole.PROOF,
                "show source", [evidence.id], [beat_ids[min(index - 1, len(beat_ids) - 1)]],
                recording_cues=[CaptureCue(CueAction.WAIT, "hold", wait_ms=4000)],
                duration_hint=4,
            )
            for index in range(1, 6)
        ]
        request = StoryboardRequest(
            "m-1", candidate, TopicType.RESEARCH_OR_BENCHMARK, ContentType.DEEP_DIVE,
            [evidence], "这是评测协议，不是模型排名",
            [NarrativeAnswer(beat, beat, [evidence.id]) for beat in beat_ids],
            proposals, 50,
        )

        manifest = StoryboardDirector().direct(request)

        self.assertEqual(manifest.duration, 25)
        self.assertTrue(all(scene.duration == 5 for scene in manifest.scenes))
        self.assertTrue(all(scene.recording_cues[-1].wait_ms == 5000 for scene in manifest.scenes))
        duration_check = next(item for item in validate_manifest(manifest) if item.name == "duration_band")
        self.assertTrue(duration_check.passed)

    def test_github_quality_rejects_unsupported_prototype_downgrade(self) -> None:
        candidate = Candidate("c", SourceType.GITHUB, "https://github.com/a/b", "a/b")
        evidence = Evidence("e", "c", candidate.source_url, "A useful production workflow", "github:readme")
        manifest = RenderManifest(
            "m", "c", ContentType.EXPLAINER,
            [Scene("s", 0, 5, "n", "MoneyPrinterTurbo", ["e"], MaterialRole.PROOF, "show")],
            [evidence], [candidate.source_url], fixed_hook="MoneyPrinterTurbo 自动生成视频",
            fixed_title="MoneyPrinterTurbo 自动生成视频",
            fixed_footer="适合快速原型而非成品交付",
        )
        check = next(item for item in validate_manifest(manifest) if item.name == "unsupported_editorial_downgrade")
        self.assertFalse(check.passed)

    def test_browser_capture_archives_asset_without_browser_state(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            raw_asset = Path(temp) / "source.png"
            raw_asset.write_bytes(b"proof")
            candidate = Candidate("candidate-1", SourceType.TWEET, "https://x.com/example/status/1", "example")
            evidence = BrowserCaptureImporter(Workspace(root)).import_capture(
                BrowserCaptureRequest(CaptureKind.TWEET, candidate.source_url, candidate, "source-backed quote"),
                CapturedBrowserArtifact(raw_asset, "visible page text"),
            )
            self.assertTrue((root / evidence.captured_asset).is_file())
            self.assertIsNotNone(evidence.sha256)

    def test_publish_draft_never_auto_submits(self) -> None:
        with TemporaryDirectory() as temp:
            video = Path(temp) / "video.mp4"
            video.write_bytes(b"video")
            candidate = Candidate("candidate-1", SourceType.TWEET, "https://x.com/example/status/1", "example")
            evidence = Evidence("e-1", candidate.id, candidate.source_url, "claim", "tweet")
            manifest = RenderManifest(
                "render-1", candidate.id, ContentType.FLASH,
                [
                    Scene("s-1", 0, 4, "narration", "caption", [evidence.id], MaterialRole.PROOF, "show proof"),
                    Scene("s-2", 4, 8, "conclusion", "conclusion", [evidence.id], MaterialRole.EXPLANATION, "highlight conclusion", highlight_translation="重点结论"),
                ],
                [evidence], [candidate.source_url], fixed_footer="清楚结论", video_path=str(video),
                music_license_status="verified", license_records=[{"track": "test", "license": "original"}],
            )
            probe = VideoProbe(video, 8.0, 1080, 1920, "h264", "yuv420p", "aac")
            with patch("video_factory.publish.probe_video", return_value=probe):
                draft = prepare_publish_draft(manifest, "title", "description")
            self.assertEqual(draft.state, PublicationState.READY_FOR_HUMAN_REVIEW)
            self.assertTrue(draft.final_publish_requires_human)

    def test_mpt_adapter_requires_a_native_track_and_safe_footer(self) -> None:
        candidate = Candidate("candidate-1", SourceType.TWEET, "https://x.com/example/status/1", "example")
        evidence = Evidence("e-1", candidate.id, candidate.source_url, "claim", "tweet")
        manifest = RenderManifest(
            "render-1", candidate.id, ContentType.FLASH,
            [Scene("s-1", 0, 8, "narration", "caption", [evidence.id], MaterialRole.PROOF, "show proof")],
            [evidence], [candidate.source_url], fixed_footer="清楚结论",
        )
        adapter = MPTAssemblyAdapter(MPTSettings(Path("/mpt"), Path("/python")))
        command = adapter.build_command(manifest, Path("track.mp4"), task_id="task-1")
        self.assertIn("--video-transition-mode", command)
        self.assertEqual(command[command.index("--video-transition-mode") + 1], "none")
        self.assertIn("--no-subtitle-enabled", command)
        self.assertEqual(command[command.index("--voice-name") + 1], "no-voice")
        import uuid
        uuid.UUID(command[command.index("--task-id") + 1])
        self.assertEqual(
            adapter._master_audio_filter(8.0),
            "[1:a]volume=0.05,atrim=duration=8.000,asetpts=N/SR/TB,"
            "afade=t=out:st=7.200:d=0.800[a]",
        )

    def test_native_ffmpeg_master_skips_the_mpt_render_pipeline(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            native = root / "native.mp4"
            music = root / "music.mp3"
            output = root / "master.mp4"
            native.write_bytes(b"video")
            music.write_bytes(b"music")
            manifest = RenderManifest(
                "render-1", "candidate-1", ContentType.FLASH,
                [Scene("s-1", 0, 8, "n", "c", ["e-1"], MaterialRole.PROOF, "show")],
                [Evidence("e-1", "candidate-1", "https://example.com", "claim", "web")],
                ["https://example.com"], fixed_footer="结论",
            )
            adapter = NativeFFmpegAssemblyAdapter(MPTSettings(root, root / "python", str(music)))
            with patch("video_factory.mpt.probe_video", return_value=VideoProbe(
                native, 8.0, 1080, 1920, "h264", "yuv420p", None,
            )):
                command = adapter.build_command(manifest, native, output)

            self.assertEqual(command[0], "ffmpeg")
            self.assertEqual(command[command.index("-i") + 1], str(native))
            self.assertNotIn("cli.py", command)
            self.assertIn(str(music), command)

    def test_capture_uses_unpatched_runner_when_optional_patch_is_incompatible(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "capture.mp4"
            request = WebCaptureRequest(
                "https://example.com",
                [CaptureCue(CueAction.WAIT, "hold", wait_ms=1000, shot_id="hold")],
                output, root / "frames",
            )
            adapter = WebScrollVideoAdapter(WebScrollVideoSettings(root / "web-scroll-video"))

            def create_capture(*_args, **_kwargs):
                output.write_bytes(b"video")

            with (
                patch.object(adapter, "_write_padded_runner", side_effect=RuntimeError("patch marker changed")),
                patch("video_factory.webcapture.subprocess.run", side_effect=create_capture) as run,
            ):
                adapter.capture(request)

            command = run.call_args.args[0]
            self.assertEqual(command[1], str(root / "web-scroll-video/src/scroll-video.mjs"))
            metadata = json.loads(output.with_suffix(".capture.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["runner_strategy"], "upstream_unpatched_runner")
            self.assertTrue(metadata["fallback_used"])
            repairs = json.loads(output.with_suffix(".capture-repairs.json").read_text(encoding="utf-8"))
            self.assertEqual(repairs[0]["kind"], "runner_patch_incompatible")

    def test_font_resolution_falls_back_after_invalid_local_configuration(self) -> None:
        with TemporaryDirectory() as temp:
            fallback = Path(temp) / "portable-font.ttc"
            fallback.write_bytes(b"font")
            with (
                patch.dict("os.environ", {"VIDEO_FACTORY_FONT": str(Path(temp) / "missing.ttc")}),
                patch("video_factory.compositor.FONT_CANDIDATES", (fallback,)),
            ):
                self.assertEqual(resolve_font_path(), fallback.resolve())

    def test_github_capture_does_not_translate_chinese_source_again(self) -> None:
        self.assertEqual(WebScrollVideoAdapter._adjacent_translation("输入主题即可生成视频", "输入主题即可生成视频"), "")
        self.assertEqual(WebScrollVideoAdapter._adjacent_translation("Generate a video", "生成视频"), "生成视频")

    def test_topic_contract_blocks_unscoped_practice_claim(self) -> None:
        candidate = Candidate("candidate-1", SourceType.TWEET, "https://x.com/example/status/1", "example")
        evidence = Evidence("e-1", candidate.id, candidate.source_url, "claim", "tweet")
        manifest = RenderManifest(
            "render-1", candidate.id, ContentType.FLASH,
            [Scene("s-1", 0, 8, "narration", "caption", [evidence.id], MaterialRole.PROOF, "show proof")],
            [evidence], [candidate.source_url], topic_type=TopicType.PRACTICE_POST,
            story_beats=[
                StoryBeat("author_claim", "作者分享一个经验", [evidence.id]),
                StoryBeat("evidence_context", "原帖已采集", [evidence.id]),
            ],
            fixed_footer="清楚结论",
        )
        messages = [item.detail for item in validate_manifest(manifest)]
        self.assertIn("缺少叙事问题：scope", messages)

    def test_external_urls_are_routed_to_primary_source_templates(self) -> None:
        urls = extract_external_urls("Read https://github.com/acme/tool and https://docs.example.com/x.")
        self.assertEqual(urls, ["https://github.com/acme/tool", "https://docs.example.com/x."])
        self.assertEqual(route_external_source(SourceType.GITHUB), TopicType.GITHUB_PROJECT)

    def test_director_makes_an_evidence_bound_practice_storyboard(self) -> None:
        candidate = Candidate("candidate-1", SourceType.TWEET, "https://x.com/example/status/1", "example")
        evidence = Evidence("e-1", candidate.id, candidate.source_url, "claim", "tweet")
        storyboard = StoryboardDirector().direct(StoryboardRequest(
            "render-1", candidate, TopicType.PRACTICE_POST, ContentType.FLASH, [evidence], "固定结论",
            [
                NarrativeAnswer("author_claim", "作者分享了一个具体实践", [evidence.id]),
                NarrativeAnswer("evidence_context", "原帖和截图均已保存", [evidence.id]),
                NarrativeAnswer("scope", "这是个案，不等于所有用户都会得到同样结果", [evidence.id]),
            ],
            [
                SceneProposal("事实", "作者分享了一个实践。", "发生了什么", MaterialRole.PROOF, "show source", [evidence.id], ["author_claim"], duration_hint=3),
                SceneProposal("边界", "这是个案，仍要看真实条件。", "不能推成普遍规律", MaterialRole.EXPLANATION, "highlight scope", [evidence.id], ["scope"], duration_hint=4, highlight_translation="适用边界"),
            ], 8,
        ))
        self.assertEqual(storyboard.scenes[0].start, 0)
        self.assertEqual(storyboard.fixed_footer, "固定结论")

    def test_director_accepts_decimal_schedule_equal_to_target(self) -> None:
        candidate = Candidate("candidate-decimal", SourceType.TWEET, "https://x.com/example/status/2", "example")
        evidence = Evidence("e-decimal", candidate.id, candidate.source_url, "claim", "tweet")
        proposals = [
            SceneProposal(
                f"事实 {index}", "", f"第 {index} 个事实", MaterialRole.PROOF,
                "show source", [evidence.id], ["author_claim"],
                duration_hint=3.2 if index == 1 else 2.2,
            )
            for index in range(1, 6)
        ]
        storyboard = StoryboardDirector().direct(StoryboardRequest(
            "render-decimal", candidate, TopicType.PRACTICE_POST, ContentType.FLASH,
            [evidence], "固定结论",
            [
                NarrativeAnswer("author_claim", "作者给出事实", [evidence.id]),
                NarrativeAnswer("evidence_context", "原帖提供上下文", [evidence.id]),
                NarrativeAnswer("scope", "结论只覆盖原帖个案", [evidence.id]),
            ],
            proposals, 12.0,
        ))
        self.assertEqual(storyboard.scenes[-1].end, 12.0)

    def test_twitter_ingestion_preserves_thread_and_creates_link_queue(self) -> None:
        with TemporaryDirectory() as temp:
            capture = Path(temp) / "tweet.json"
            capture.write_text('{"data":[{"id":"1","text":"Claim https://github.com/acme/demo","author":{"screenName":"a","verified":true},"createdAtISO":"2026-08-01T00:00:00Z","urls":[]},{"id":"2","text":"Context","author":{"screenName":"a"},"urls":[]},{"id":"3","text":"Reply","author":{"screenName":"another"},"urls":[]}]}', encoding="utf-8")
            result = TwitterCliIngestor().ingest(capture, Workspace(Path(temp) / "workspace"), lambda _: "https://github.com/acme/demo")
            self.assertEqual(len(result.evidence), 2)
            self.assertEqual(result.linked_candidates[0].source_type, SourceType.GITHUB)
            self.assertEqual(result.linked_candidates[0].metadata["original_url"], "https://github.com/acme/demo")

    def test_github_ingestion_keeps_readme_as_evidence(self) -> None:
        with TemporaryDirectory() as temp:
            repo = Path(temp) / "repo.json"
            readme = Path(temp) / "README.md"
            repo.write_text('{"full_name":"acme/demo","html_url":"https://github.com/acme/demo","owner":{"login":"acme"},"description":"Demo","license":{"spdx_id":"MIT"}}', encoding="utf-8")
            readme.write_text("# Demo\nInstall with a real command.", encoding="utf-8")
            result = GitHubIngestor().ingest(repo, readme, Workspace(Path(temp) / "workspace"))
            self.assertEqual(result.evidence[1].source_kind, "github:readme")

    def test_web_capture_cues_become_editable_scroll_video_sheet(self) -> None:
        request = WebCaptureRequest(
            "https://example.com", [
                CaptureCue(CueAction.WAIT, "settle", wait_ms=500),
                CaptureCue(CueAction.HIGHLIGHT, "show claim", target='"Claim"', wait_ms=1500),
                CaptureCue(CueAction.SCROLL, "find team", target='"Team"', wait_ms=2000),
            ], Path("/tmp/track.mp4"), Path("/tmp/frames"),
        )
        cue = WebScrollVideoAdapter(WebScrollVideoSettings(Path("/web-scroll-video"))).cue_text(request)
        self.assertIn('highlight text "Claim" for 1.5', cue)
        self.assertIn('scroll to "Team" over 2', cue)

    def test_github_capture_uses_required_real_browser_route(self) -> None:
        walkthrough = GitHubWalkthrough(1, 2, 3, 4, [
            GitHubModuleFocus(5, "Quick Start", "minimal_usable_example", ["e-1"]),
            GitHubModuleFocus(6, "Limitations", "limitation", ["e-1"]),
        ])
        request = WebScrollVideoAdapter.github_request("https://github.com/acme/demo", walkthrough, Path("/tmp/out.mp4"), Path("/tmp/frames"))
        cue = WebScrollVideoAdapter(WebScrollVideoSettings(Path("/web-scroll-video"))).cue_text(request)
        self.assertIn('go https://github.com/acme/demo', cue)
        self.assertIn('scroll to bottom over 15', cue)
        self.assertIn('highlight text "Quick Start" for 1.8', cue)
        self.assertEqual(WebScrollVideoAdapter._visible_anchor("## 快速开始 🚀"), "快速开始 🚀")

    def test_github_final_cut_reads_readme_top_then_selected_modules(self) -> None:
        brief = self.github_brief("meta", "readme")
        request = WebScrollVideoAdapter.github_story_request("https://github.com/acme/demo", brief, Path("/tmp/out.mp4"), Path("/tmp/frames"), 20)
        cue = WebScrollVideoAdapter(WebScrollVideoSettings(Path("/web-scroll-video"))).cue_text(request)
        self.assertIn("width: 1384", cue)
        self.assertIn("height: 1602", cue)
        self.assertIn('highlight selector "strong[itemprop=\\"name\\"] a"', cue)
        self.assertIn('scroll to "Turn one topic into a finished result"', cue)
        self.assertIn('highlight text "Quick Start command"', cue)
        self.assertIn('highlight text "Plugin architecture"', cue)
        self.assertNotIn('scroll to bottom', cue)

    def test_multiline_browser_evidence_uses_first_visible_line_as_anchor(self) -> None:
        request = WebCaptureRequest(
            "https://example.com", [CaptureCue(
                CueAction.SCROLL, "find evidence",
                target='"8x H100 / H200\nTested and recommended\nvLLM + Tensor Parallelism"',
                wait_ms=500,
            )], Path("/tmp/out.mp4"), Path("/tmp/frames"),
        )

        cue = WebScrollVideoAdapter(WebScrollVideoSettings(Path("/web-scroll-video"))).cue_text(request)

        self.assertIn('scroll to "8x H100 / H200"', cue)
        self.assertNotIn("Tested and recommended", cue)

    def test_github_final_cut_skips_nonvisible_empty_description_placeholder(self) -> None:
        brief = self.github_brief("meta", "readme")
        brief.repo_description_target = "No repository description"
        request = WebScrollVideoAdapter.github_story_request(
            "https://github.com/acme/demo", brief,
            Path("/tmp/out.mp4"), Path("/tmp/frames"), 20,
        )

        cue = WebScrollVideoAdapter(WebScrollVideoSettings(Path("/web-scroll-video"))).cue_text(request)

        self.assertNotIn("No repository description", cue)
        self.assertAlmostEqual(WebScrollVideoAdapter.capture_metadata(request)["duration"], 20.0)

    def test_missing_browser_text_repairs_to_real_page_hold_without_false_highlight(self) -> None:
        request = WebCaptureRequest(
            "https://example.com", [
                CaptureCue(CueAction.SCROLL, "find claim", target='"Missing claim"', wait_ms=500),
                CaptureCue(
                    CueAction.HIGHLIGHT, "show claim", target='"Missing claim"',
                    wait_ms=1500, shot_id="claim",
                ),
            ], Path("/tmp/out.mp4"), Path("/tmp/frames"),
        )

        repaired = WebScrollVideoAdapter._repair_missing_text_request(request, "Missing claim")

        self.assertEqual(repaired.cues[0].action, CueAction.SCROLL)
        self.assertEqual(repaired.cues[0].target, "bottom")
        self.assertEqual(repaired.cues[1].action, CueAction.WAIT)
        self.assertEqual(repaired.cues[1].shot_id, "claim")
        self.assertEqual(
            WebScrollVideoAdapter.capture_metadata(repaired)["duration"],
            WebScrollVideoAdapter.capture_metadata(request)["duration"],
        )

    def test_tiny_browser_highlight_repairs_to_source_page_hold(self) -> None:
        request = WebCaptureRequest(
            "https://example.com", [
                CaptureCue(CueAction.SCROLL, "find license", target="License:", wait_ms=500),
                CaptureCue(
                    CueAction.HIGHLIGHT, "show license", target="License:",
                    wait_ms=1500, shot_id="license",
                ),
            ], Path("/tmp/out.mp4"), Path("/tmp/frames"),
        )

        shot_id = WebScrollVideoAdapter._unreadable_highlight_shot(
            "visual gate: highlight for license is too small (74px); target a readable line"
        )
        repaired = WebScrollVideoAdapter._repair_unreadable_highlight_request(request, shot_id)

        self.assertEqual(shot_id, "license")
        self.assertEqual(repaired.cues[0], request.cues[0])
        self.assertEqual(repaired.cues[1].action, CueAction.WAIT)
        self.assertEqual(repaired.cues[1].shot_id, "license")
        self.assertEqual(
            WebScrollVideoAdapter.capture_metadata(repaired)["duration"],
            WebScrollVideoAdapter.capture_metadata(request)["duration"],
        )

    def test_x_card_renders_the_single_selected_glossary_beside_translation(self) -> None:
        scene = Scene(
            "scene-1", 0, 3, "internal", "fact", ["root"], MaterialRole.PROOF,
            "show tweet", screen_fact="fact",
            screen_interpretation="拒绝率：模型直接拒绝回答的请求占比。",
            highlight_translation="完整根帖中文摘要。",
        )
        self.assertEqual(
            _tweet_translation_copy(scene),
            "完整根帖中文摘要。\n拒绝率：模型直接拒绝回答的请求占比。",
        )

        scene.screen_interpretation = "普通补充信息不应重复显示。"
        self.assertEqual(_tweet_translation_copy(scene), "完整根帖中文摘要。")

    def test_github_focus_selection_keeps_one_proof_and_one_decision(self) -> None:
        brief = self.github_brief("meta", "readme")
        brief.focus_candidates[2].viewer_value = 3
        brief.focus_candidates[2].risk_importance = 3
        selected = select_github_focuses(brief)
        self.assertEqual(selected[0].editorial_role, "trial")
        self.assertEqual(selected[1].editorial_role, "technical_edge")

    def test_github_brief_rejects_an_invented_browser_target(self) -> None:
        metadata = Evidence("meta", "c", "https://github.com/acme/demo", "Automate a repeated task", "github:metadata")
        readme = Evidence(
            "readme", "c", "https://github.com/acme/demo",
            "Turn one topic into a finished result. Quick Start command. Review output before publishing. Plugin architecture.",
            "github:readme",
        )
        brief = self.github_brief(metadata.id, readme.id)
        brief.focus_candidates[0].target = "A capability absent from the README"
        errors = validate_github_brief(brief, [metadata, readme])
        self.assertTrue(any("not present" in error for error in errors))

    def test_github_capture_metadata_matches_the_middle_pane_and_shot_timing(self) -> None:
        request = WebScrollVideoAdapter.github_story_request(
            "https://github.com/acme/demo", self.github_brief("meta", "readme"),
            Path("/tmp/out.mp4"), Path("/tmp/frames"), 20,
        )
        metadata = WebScrollVideoAdapter.capture_metadata(request)
        self.assertEqual((metadata["width"], metadata["height"]), (1384, 1602))
        self.assertAlmostEqual(metadata["width"] / metadata["height"], 1080 / 1250, places=3)
        self.assertAlmostEqual(metadata["duration"], 20.0)
        shortened = WebScrollVideoAdapter.github_story_request(
            "https://github.com/acme/demo", self.github_brief("meta", "readme"),
            Path("/tmp/out-short.mp4"), Path("/tmp/frames-short"), 11.4,
        )
        self.assertAlmostEqual(
            WebScrollVideoAdapter.capture_metadata(shortened)["duration"], 11.4, places=2,
        )
        shot_ids = {shot["id"] for shot in metadata["shots"]}
        self.assertTrue({"repo_overview", "repo_name", "repo_description", "file_tree", "readme_claim", "focus_1", "focus_2"} <= shot_ids)
        boxed = WebScrollVideoAdapter.capture_metadata(
            request, {"focus_1": {"left": 10, "top": 20, "width": 300, "height": 40}},
        )
        focus = next(shot for shot in boxed["shots"] if shot["id"] == "focus_1")
        self.assertEqual(focus["highlight_box"]["width"], 300)

    def test_linked_post_capture_keeps_the_post_before_opening_the_primary_page(self) -> None:
        request = WebScrollVideoAdapter.linked_post_request(
            "https://x.com/JeffDean/status/1", "https://example.com", ["The Approach", "The Team"],
            Path("/tmp/out.mp4"), Path("/tmp/frames"), 24,
        )
        cue = WebScrollVideoAdapter(WebScrollVideoSettings(Path("/web-scroll-video"))).cue_text(request)
        self.assertLess(cue.index("go https://x.com/JeffDean/status/1"), cue.index("go https://example.com"))
        self.assertIn('highlight text "The Approach"', cue)

    def test_writer_packet_exposes_only_topic_required_questions_and_evidence(self) -> None:
        candidate = Candidate("c-1", SourceType.GITHUB, "https://github.com/acme/demo", "demo")
        evidence = Evidence("e-1", candidate.id, candidate.source_url, "README supports this", "github:readme")
        prompt = StoryWriterPacket(candidate, [evidence], TopicType.GITHUB_PROJECT, ContentType.FLASH, 10).prompt()
        self.assertIn('"problem"', prompt)
        self.assertNotIn('"human_check"', prompt)
        self.assertIn('README supports this', prompt)
        self.assertIn('"editorial_role"', prompt)
        self.assertIn('"github_scenes"', prompt)
        self.assertNotIn('"material_role"', prompt)

    def test_research_plan_accepts_a_fragment_of_the_same_source_page(self) -> None:
        candidate = Candidate("c-1", SourceType.GITHUB, "https://github.com/acme/demo", "demo")
        evidence = Evidence("e-1", candidate.id, candidate.source_url, "claim", "github:repository")
        packet = StoryWriterPacket(candidate, [evidence], TopicType.GITHUB_PROJECT, ContentType.FLASH, 12)
        plan = OpenAICompatibleStoryWriter._parse_plan(packet, {
            "angle": "具体角度", "audience_value": "试用判断", "selected_evidence_ids": ["e-1"],
            "requested_urls": ["https://github.com/acme/demo#readme"], "unresolved_questions": [],
            "ready_to_write": True,
        })
        self.assertEqual(plan.requested_urls, ["https://github.com/acme/demo#readme"])

    def test_markdown_target_is_canonicalized_to_browser_visible_spacing(self) -> None:
        metadata = Evidence("meta", "c", "https://github.com/acme/demo", "Automate a repeated task", "github:repository")
        readme = Evidence(
            "readme", "c", "https://github.com/acme/demo#readme",
            'Turn one topic into a finished result. Plugin architecture.\n- [x] 支持一键 **跨平台发布**，生成后请 review output before publishing', "github:readme",
        )
        brief = self.github_brief(metadata.id, readme.id)
        brief.readme_claim_target = "支持一键跨平台发布"
        brief.focus_candidates[0].target = "支持一键跨平台发布"
        brief.focus_candidates[1].target = "review output before publishing"
        canonicalize_github_brief(brief, [metadata, readme])
        self.assertEqual(brief.focus_candidates[0].target, "支持一键 跨平台发布")
        self.assertFalse(validate_github_brief(brief, [metadata, readme]))

    def test_joined_code_lines_are_reduced_to_one_real_browser_line(self) -> None:
        metadata = Evidence("meta", "c", "https://github.com/acme/demo", "Automate a repeated task", "github:repository")
        readme = Evidence(
            "readme", "c", "https://github.com/acme/demo#readme",
            "Turn one topic into a finished result. Quick Start command. Review output before publishing. Plugin architecture.\n"
            "warning: PDF edits are incremental — original bytes\nremain recoverable; install qpdf",
            "github:readme",
        )
        brief = self.github_brief(metadata.id, readme.id)
        brief.focus_candidates[1].target = (
            "warning: PDF edits are incremental — original bytes remain recoverable; install qpdf"
        )
        canonicalize_github_brief(brief, [metadata, readme])
        self.assertEqual(brief.focus_candidates[1].target, "warning: PDF edits are incremental — original bytes")

    def test_code_proof_gets_a_separate_prose_browser_target(self) -> None:
        metadata = Evidence("meta", "c", "https://github.com/acme/demo", "Automate a repeated task", "github:repository")
        readme = Evidence(
            "readme", "c", "https://github.com/acme/demo#readme",
            "Turn one topic into a finished result. Plugin architecture. Review output before publishing.\n"
            "最简单的完整命令如下：\n```shell\nuv run python cli.py --video-subject demo\n```",
            "github:readme",
        )
        brief = self.github_brief(metadata.id, readme.id)
        brief.readme_claim_target = "Turn one topic into a finished result"
        brief.focus_candidates[0].target = "uv run python cli.py --video-subject demo"
        canonicalize_github_brief(brief, [metadata, readme])
        self.assertEqual(brief.focus_candidates[0].browser_target, "最简单的完整命令如下：")
        request = WebScrollVideoAdapter.github_story_request(
            "https://github.com/acme/demo", brief, Path("/tmp/out.mp4"), Path("/tmp/frames"), 20,
        )
        focus_cue = next(cue for cue in request.cues if cue.shot_id == "focus_1")
        self.assertEqual(focus_cue.target, "最简单的完整命令如下：")

    def test_visual_gate_measures_one_yellow_component_not_global_pixels(self) -> None:
        image = Image.new("RGB", (300, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 24, 24), fill=(255, 212, 0))
        draw.rectangle((260, 60, 274, 74), fill=(255, 212, 0))
        count, width = WebScrollVideoAdapter._main_highlight_width(image)
        self.assertEqual(count, 225)
        self.assertLess(count, 300)
        self.assertEqual(width, 14)

    def test_highlight_runner_draws_an_outside_outline_with_text_gap(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp) / "web-scroll-video"
            source = root / "src" / "scroll-video.mjs"
            source.parent.mkdir(parents=True)
            source.write_text(
                'style.textContent = "border: 6px solid #ffd400; box-shadow: glow";\n'
                '      candidates.sort((a, b) => a.area - b.area);\n'
                '      element = candidates[0]?.element || null;\n'
                '      return finish(element);\n'
                '      highlight.style.left = ${JSON.stringify(box.left)} + "px";\n'
                '      highlight.style.top = ${JSON.stringify(box.top)} + "px";\n'
                '      highlight.style.width = ${JSON.stringify(box.width)} + "px";\n'
                '      highlight.style.height = ${JSON.stringify(box.height)} + "px";\n'
                '      highlight.style.display = "block";',
                encoding="utf-8",
            )
            cue = Path(temp) / "story.cue"
            runner = WebScrollVideoAdapter(WebScrollVideoSettings(root, highlight_gap=8))._write_padded_runner(cue)
            patched = runner.read_text(encoding="utf-8")
            self.assertIn("border: 0", patched)
            self.assertIn("outline-offset: 8px", patched)
            self.assertIn('setProperty("display", "block", "important")', patched)
            self.assertIn("minimumReadableWidth", patched)
            self.assertIn('wanted === "in / out price"', patched)
            self.assertIn('text.includes("per 1m")', patched)

    def test_quality_rejects_an_invented_visible_time_saving(self) -> None:
        candidate = Candidate("c-1", SourceType.TWEET, "https://x.com/a/status/1", "test")
        evidence = Evidence("e-1", candidate.id, candidate.source_url, "自动完成重复流程", "tweet")
        manifest = RenderManifest(
            "m-1", candidate.id, ContentType.FLASH,
            [Scene("s-1", 0, 8, "internal", "自动完成重复流程", [evidence.id], MaterialRole.PROOF, "show source")],
            [evidence], [candidate.source_url], fixed_hook="自动完成重复流程",
            fixed_footer="把数小时工作压缩到几分钟",
        )
        failed = {item.name for item in validate_manifest(manifest) if not item.passed}
        self.assertIn("quantified_claims", failed)

    def test_sensitive_security_material_requires_editorial_review(self) -> None:
        evidence = [Evidence("e-1", "c-1", "https://example.com", "A vulnerability could expose credentials.", "research")]
        review = review_evidence(evidence)
        self.assertTrue(review.requires_human_review)
        self.assertIn("Do not generate reproduction steps", review.prohibited_angle)

        manifest = RenderManifest(
            "m-1", "c-1", ContentType.FLASH,
            [Scene("s-1", 0, 8, "internal", "漏洞披露", ["e-1"], MaterialRole.PROOF, "show source")],
            evidence, ["https://example.com"], fixed_hook="漏洞披露", fixed_footer="等待修复",
        )
        failed = {item.name for item in validate_manifest(manifest) if not item.passed}
        self.assertIn("editorial_safety_review", failed)

    def test_provenance_removal_tool_requires_editorial_review(self) -> None:
        evidence = [Evidence("e-1", "c-1", "https://example.com", "Strip AI provenance marks and C2PA metadata.", "github:readme")]
        self.assertTrue(review_evidence(evidence).requires_human_review)

    def test_quality_rejects_overlapping_scenes_and_unknown_beat_evidence(self) -> None:
        candidate = Candidate("c-1", SourceType.TWEET, "https://x.com/a/status/1", "test")
        evidence = Evidence("e-1", candidate.id, candidate.source_url, "claim", "tweet")
        manifest = RenderManifest(
            "m-1", candidate.id, ContentType.FLASH,
            [
                Scene("a", 0, 5, "a", "a", [evidence.id], MaterialRole.PROOF, "show"),
                Scene("b", 4, 8, "b", "b", [evidence.id], MaterialRole.PROOF, "show"),
            ], [evidence], [candidate.source_url], topic_type=TopicType.PRACTICE_POST,
            story_beats=[
                StoryBeat("author_claim", "claim", [evidence.id]), StoryBeat("evidence_context", "context", [evidence.id]),
                StoryBeat("scope", "scope", ["unknown"]),
            ], fixed_footer="footer",
        )
        names = {item.name for item in validate_manifest(manifest) if not item.passed}
        self.assertIn("timeline_order", names)
        self.assertIn("story_beat_evidence", names)

    def test_video_account_upload_stops_before_final_publish(self) -> None:
        class FakeDriver:
            def __init__(self): self.calls = []
            def open(self, url): self.calls.append(("open", url))
            def upload(self, selector, file_path): self.calls.append(("upload", selector, file_path))
            def fill(self, selector, value): self.calls.append(("fill", selector, value))
            def wait_until_visible(self, selector): self.calls.append(("wait", selector))

        with TemporaryDirectory() as temp:
            video = Path(temp) / "video.mp4"
            video.write_bytes(b"video")
            draft = PublishDraft("p-1", "m-1", "title", "description", str(video), PublicationState.READY_FOR_HUMAN_REVIEW)
            driver = FakeDriver()
            result = WeChatVideoAccountUploader().prepare_for_human_final_click(
                draft, driver, WeChatVideoAccountSelectors("https://channels.weixin.qq.com/platform", "#upload", "#title", "#description", "#publish"),
            )
            self.assertEqual(result.state, PublicationState.AWAITING_FINAL_PUBLISH_CLICK)
            self.assertEqual(driver.calls[-1], ("wait", "#publish"))

    def test_llm_draft_parses_an_evidence_ranked_github_brief(self) -> None:
        candidate = Candidate("github-acme-demo", SourceType.GITHUB, "https://github.com/acme/demo", "acme/demo")
        metadata = Evidence("meta", candidate.id, candidate.source_url, "Automate a repeated task", "github:metadata")
        readme = Evidence(
            "readme", candidate.id, candidate.source_url,
            "Turn one topic into a finished result. Quick Start command. Review output before publishing. Plugin architecture.",
            "github:readme",
        )
        draft = {
            "footer": "结论", "answers": [
                {"beat_id": "problem", "answer": "解决一个问题", "evidence_ids": ["readme"]},
                {"beat_id": "human_check", "answer": "仍要人工检查", "evidence_ids": ["readme"]},
            ],
            "scenes": [
                {"stage_name": "repo", "narration": "a", "caption": "a", "material_role": "proof", "visual_action": "a", "evidence_ids": ["meta"], "beat_ids": ["problem"], "duration_hint": 3},
                {"stage_name": "claim", "narration": "b", "caption": "b", "material_role": "proof", "visual_action": "b", "evidence_ids": ["readme"], "beat_ids": ["problem"], "duration_hint": 3},
                {"stage_name": "proof", "narration": "c", "caption": "c", "material_role": "proof", "visual_action": "c", "evidence_ids": ["readme"], "beat_ids": ["problem"], "duration_hint": 3},
                {"stage_name": "boundary", "narration": "d", "caption": "d", "material_role": "proof", "visual_action": "d", "evidence_ids": ["readme"], "beat_ids": ["human_check"], "duration_hint": 3}
            ],
            "github_brief": json.loads(json.dumps(asdict(self.github_brief("meta", "readme"))))
            ,"github_scenes": [
                {"stage": "repo_identity", "message": "项目做什么", "interpretation": "先确认用途", "focus_id": None, "evidence_ids": ["meta"], "beat_ids": ["problem"], "duration_hint": 5},
                {"stage": "readme_claim", "message": "输入输出", "interpretation": "确认完整链路", "focus_id": None, "evidence_ids": ["readme"], "beat_ids": ["input_output"], "duration_hint": 5},
                {"stage": "selected_focus", "message": "最小试用", "interpretation": "可以立即验证", "focus_id": "trial", "evidence_ids": ["readme"], "beat_ids": ["trial_task"], "duration_hint": 5},
                {"stage": "selected_focus", "message": "插件机制", "interpretation": "解释扩展能力", "focus_id": "edge", "evidence_ids": ["readme"], "beat_ids": ["workflow_fit"], "duration_hint": 5}
            ]
        }
        packet = StoryWriterPacket(candidate, [metadata, readme], TopicType.GITHUB_PROJECT, ContentType.EXPLAINER, 20)
        request = OpenAICompatibleStoryWriter._to_storyboard_request(packet, draft)
        self.assertEqual(request.github_brief.project_kind, "cli_sdk")
        self.assertEqual(request.fixed_hook, "重复任务终于能一条命令自动化")
        self.assertEqual(request.github_brief.focus_candidates[0].target, "Quick Start command")
        self.assertEqual([scene.stage_name for scene in request.scenes], ["repo_identity", "readme_claim", "selected_focus", "selected_focus"])
        self.assertEqual(request.scenes[-1].material_role, MaterialRole.EXPLANATION)

    def test_director_accepts_github_only_when_recording_contract_is_complete(self) -> None:
        candidate = Candidate("github-acme-demo", SourceType.GITHUB, "https://github.com/acme/demo", "acme/demo")
        metadata = Evidence("meta", candidate.id, candidate.source_url, "Automate a repeated task", "github:metadata")
        readme = Evidence(
            "readme", candidate.id, candidate.source_url,
            "Turn one topic into a finished result. Quick Start command. Review output before publishing. Plugin architecture.",
            "github:readme",
        )
        proposal = lambda name, evidence_id: SceneProposal(name, name, name, MaterialRole.PROOF, "browser action", [evidence_id], ["problem"], duration_hint=3)
        request = StoryboardRequest(
            "m-1", candidate, TopicType.GITHUB_PROJECT, ContentType.FLASH, [metadata, readme], "footer",
            [NarrativeAnswer("problem", "problem", [readme.id]), NarrativeAnswer("human_check", "check", [readme.id])],
            [
                proposal("repo", metadata.id), proposal("claim", readme.id),
                proposal("proof", readme.id), proposal("boundary", readme.id),
            ], 12,
            github_brief=self.github_brief(metadata.id, readme.id),
        )
        manifest = StoryboardDirector().direct(request)
        manifest.music_license_status = "verified"
        manifest.license_records = [{"track": "test", "license": "original"}]
        self.assertTrue(is_publishable(validate_manifest(manifest)))
