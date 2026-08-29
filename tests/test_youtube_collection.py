import json
import subprocess
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.models import (
    Candidate, CollectionItem, CollectionItemKind, FramingMode, HookSpec, HookStrategy,
    PlatformRender, RenderProfile, RightsReview, SlideTranslation, SourceMediaInfo, SourceRange, SourceType,
    TerminologyEntry, TerminologyStrategy, TranscriptCue,
)
from video_factory.media import VideoProbe
from video_factory.serde import collection_manifest_from_dict
from video_factory.storage import Workspace
from video_factory.youtube import (
    DiscoveryConfig, NaturalSubtitleTranslator, YouTubeAcquirer, YouTubeCandidate,
    YouTubeCollectionRenderer, YouTubeDiscoveryService, SourceBelow1080Error, YouTubeAcquisitionError,
    build_collection_manifest, build_hook_candidates, normalize_chinese_subtitle,
    _coerce_range, editorial_plan_contract_errors, parse_youtube_json3,
    rebalance_source_cues, render_source_ranges, terminology_contract_errors,
    validate_collection, wrap_subtitle, write_item_subtitle_files, _headline_fragment,
    _required_short_source_ranges, _slide_translation_rows, _write_hook_overlay_concat,
)


class FakeYouTubeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if "--flat-playlist" in command:
            payload = {
                "entries": [
                    {
                        "id": "karpathy-1", "title": "Andrej Karpathy: Agentic Engineering Systems",
                        "channel": "Sequoia Capital", "duration": 1320, "view_count": 500000,
                        "url": "https://www.youtube.com/watch?v=karpathy-1",
                    },
                    {
                        "id": "weak-1", "title": "Funny cats", "channel": "Cats",
                        "duration": 600, "view_count": 999999,
                        "url": "https://www.youtube.com/watch?v=weak-1",
                    },
                ],
            }
        else:
            video_id = "karpathy-1" if "karpathy-1" in command[-1] else "weak-1"
            if video_id == "karpathy-1":
                payload = {
                    "id": video_id, "title": "Andrej Karpathy: Agentic Engineering Systems",
                    "channel": "Sequoia Capital", "description": "How developers build and scale AI agent systems",
                    "duration": 1320, "view_count": 500000, "upload_date": "20260826",
                    "chapters": [{"start_time": 0, "end_time": 300, "title": "How teams build"}],
                    "formats": [{"format_id": "137", "width": 1920, "height": 1080}],
                    "automatic_captions": {"en": [{"ext": "json3"}]},
                }
            else:
                payload = {
                    "id": video_id, "title": "Funny cats", "channel": "Cats",
                    "description": "pets", "duration": 600, "view_count": 999999,
                    "upload_date": "20260826", "chapters": [],
                }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")


class YouTubeCollectionTest(unittest.TestCase):
    def test_metadata_failure_does_not_accept_null_json(self) -> None:
        runner = lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, "null\n", "network lookup failed",
        )
        with TemporaryDirectory() as temp, self.assertRaisesRegex(
            YouTubeAcquisitionError, "network lookup failed",
        ):
            YouTubeAcquirer(Workspace(Path(temp)), runner=runner)._metadata(
                "https://youtube.com/watch?v=failed",
            )

    def test_wechat_hook_headline_persists_for_full_video(self) -> None:
        hook = HookSpec(
            "hook-1", HookStrategy.CONTRARIAN, "暗工厂不会自己到来",
            "解释组织与工具链为何重要", SourceRange(10, 18),
            ["cue-1"], ["cue-2"], selected=True,
        )
        with TemporaryDirectory() as temp:
            concat, _ = _write_hook_overlay_concat(hook, 300, Path(temp) / "episode.mp4")
            content = concat.read_text(encoding="utf-8")

        self.assertIn("duration 7.000000", content)
        self.assertIn("duration 293.000000", content)
        self.assertIn("hook-compact.png", content)
        self.assertNotIn("blank.png", content)

    def test_bilibili_hook_only_appears_during_cold_open(self) -> None:
        hook = HookSpec(
            "hook-1", HookStrategy.CONTRARIAN, "AI 越强，基本功越重要",
            "解释基本功为何成为 AI 时代的杠杆", SourceRange(10, 18),
            ["cue-1"], ["cue-1"], selected=True,
        )
        with TemporaryDirectory() as temp:
            concat, _ = _write_hook_overlay_concat(
                hook, 1200, Path(temp) / "chapter.mp4",
                RenderProfile.BILIBILI_LANDSCAPE,
            )
            content = concat.read_text(encoding="utf-8")

        self.assertIn("duration 8.000000", content)
        self.assertIn("duration 1192.000000", content)
        self.assertIn("blank.png", content)
        self.assertNotIn("hook-compact.png'", content)

    def test_translation_retries_only_missing_cue_ids(self) -> None:
        class PartialWriter:
            def __init__(self):
                self.calls = 0

            def _request_json(self, messages, max_tokens):
                self.calls += 1
                if self.calls == 1:
                    return ({
                        "terminology": [], "main_ranges": [{"start": 0, "end": 900}],
                        "themes": [
                            {"title": f"完整主题 {index}", "thesis": "完整观点", "start": index * 300, "end": (index + 1) * 300}
                            for index in range(3)
                        ],
                    }, {"call": 1})
                if self.calls == 2:
                    return ({"translations": [{"id": "c1", "text": "第一句。"}]}, {"call": 2})
                return ({"translations": [{"id": "c2", "text": "第二句。"}]}, {"call": 3})

        writer = PartialWriter()
        cues = [
            TranscriptCue("c1", 0, 3, "First sentence."),
            TranscriptCue("c2", 3, 6, "Second sentence."),
        ]

        _, _, traces = NaturalSubtitleTranslator(writer).translate({"duration": 1200}, cues)

        self.assertEqual(writer.calls, 3)
        self.assertEqual([item.translation for item in cues], ["第一句。", "第二句。"])
        chunk_traces = [item for item in traces if item["step"] == "translate_chunk"]
        self.assertEqual([item["requested"] for item in chunk_traces], [2, 1])

    def test_editorial_ranges_accept_natural_timecodes(self) -> None:
        source_range = _coerce_range(
            {
                "start": "00:45", "end": "05:30", "framing": "slide",
                "crop": {"x": 500, "y": 80, "width": 1320, "height": 742},
            },
            1325,
        )

        self.assertIsNotNone(source_range)
        self.assertEqual(source_range.start, 45)
        self.assertEqual(source_range.end, 330)
        self.assertEqual(source_range.framing, FramingMode.SLIDE)
        self.assertTrue(source_range.has_explicit_crop)

    def test_editorial_ranges_reject_crop_outside_normalized_frame(self) -> None:
        self.assertIsNone(_coerce_range({
            "start": 0, "end": 10,
            "crop": {"x": 1800, "y": 0, "width": 500, "height": 500},
        }, 60))
        self.assertIsNone(_coerce_range({
            "start": 0, "end": 10,
            "crop": {"x": 0, "y": 900, "width": 500, "height": 300},
        }, 60))
        edge = _coerce_range({
            "start": 0, "end": 10,
            "crop": {"x": 1420, "y": 780, "width": 500, "height": 300},
        }, 60)
        self.assertIsNotNone(edge)
        self.assertTrue(edge.has_explicit_crop)

    def test_translation_repairs_only_terms_omitted_by_first_pass(self) -> None:
        class RepairingWriter:
            def __init__(self):
                self.calls = 0

            def _request_json(self, messages, max_tokens):
                self.calls += 1
                if self.calls == 1:
                    return ({
                        "terminology": [{"source": "RAG", "strategy": "preserve"}],
                        "main_ranges": [{"start": 0, "end": 900}],
                        "themes": [
                            {"title": f"完整主题 {index}", "thesis": "完整观点", "start": index * 300, "end": (index + 1) * 300}
                            for index in range(3)
                        ],
                    }, {"call": 1})
                if self.calls == 2:
                    return ({
                        "translations": [{"id": "c1", "text": "这是检索增强流程。"}],
                    }, {"call": 2})
                return ({
                    "translations": [{"id": "c1", "text": "这是 RAG 检索增强流程。"}],
                }, {"call": 3})

        cues = [TranscriptCue("c1", 0, 3, "This is a RAG pipeline.")]
        terms, _, traces = NaturalSubtitleTranslator(RepairingWriter()).translate({"duration": 1200}, cues)

        self.assertEqual(terms[0].source, "RAG")
        self.assertIn("RAG", cues[0].translation)
        self.assertEqual(traces[-1]["step"], "terminology_repair")

    def test_deterministic_term_enforcement_handles_model_noncompliance(self) -> None:
        cues = [TranscriptCue("c1", 0, 4, "The LLM uses RAG.", "模型使用检索增强生成。")]
        terminology = [
            TerminologyEntry("LLM", TerminologyStrategy.PRESERVE),
            TerminologyEntry("RAG", TerminologyStrategy.PRESERVE),
        ]

        enforced = NaturalSubtitleTranslator._enforce_terminology_contract(cues, terminology)

        self.assertEqual(enforced, ["LLM", "RAG"])
        self.assertIn("（LLM）", cues[0].translation)
        self.assertIn("（RAG）", cues[0].translation)
        self.assertEqual(terminology_contract_errors(cues, terminology), [])

    def test_acronyms_do_not_match_inside_ordinary_words(self) -> None:
        cues = [TranscriptCue(
            "c1", 0, 5,
            "How will changing your organization affect the team?",
            "组织变化会如何影响团队？",
        )]
        terminology = [
            TerminologyEntry("RAG", TerminologyStrategy.PRESERVE),
            TerminologyEntry("LLM", TerminologyStrategy.PRESERVE),
        ]

        enforced = NaturalSubtitleTranslator._enforce_terminology_contract(cues, terminology)

        self.assertEqual(enforced, [])
        self.assertNotIn("RAG", cues[0].translation)
        self.assertNotIn("LLM", cues[0].translation)
        self.assertEqual(terminology_contract_errors(cues, terminology), [])

    def test_local_media_and_json3_subtitles_skip_remote_caption_download(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = Workspace(root / "workspace")
            workspace.initialize()
            media = root / "source.mp4"
            media.write_bytes(b"local-video")
            subtitle = root / "source.en-orig.json3"
            subtitle.write_text(json.dumps({
                "events": [{
                    "tStartMs": 0, "dDurationMs": 2000,
                    "segs": [{"utf8": "Agent systems are changing."}],
                }],
            }), encoding="utf-8")
            commands = []

            def metadata_runner(command, **kwargs):
                commands.append(command)
                payload = {
                    "id": "local-demo", "title": "Agent Systems", "channel": "AI Engineer",
                    "duration": 1200, "upload_date": "20260827", "chapters": [],
                }
                return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

            with patch("video_factory.youtube.probe_video", return_value=VideoProbe(
                path=str(media), duration=1200, width=1920, height=1080, video_codec="h264",
                audio_codec="aac", pixel_format="yuv420p",
            )):
                result = YouTubeAcquirer(workspace, runner=metadata_runner).acquire(
                    "https://youtube.com/watch?v=local-demo", root / "job",
                    local_media=media, local_subtitles=subtitle, download_media=False,
                )

            self.assertEqual(len(result[3]), 1)
            self.assertEqual(len(commands), 1)
            self.assertNotIn("--write-sub", commands[0])

    def test_local_media_below_1080_is_rejected_without_upscaling(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = Workspace(root / "workspace")
            workspace.initialize()
            media = root / "source.mp4"
            media.write_bytes(b"low-resolution-video")
            subtitle = root / "source.en.json3"
            subtitle.write_text(json.dumps({"events": [{
                "tStartMs": 0, "dDurationMs": 4000,
                "segs": [{"utf8": "A complete source caption."}],
            }]}), encoding="utf-8")

            def metadata_runner(command, **kwargs):
                payload = {"id": "low-demo", "title": "Low source", "duration": 1200}
                return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

            with patch("video_factory.youtube.probe_video", return_value=VideoProbe(
                path=str(media), duration=1200, width=640, height=360, video_codec="h264",
                audio_codec="aac", pixel_format="yuv420p",
            )), self.assertRaisesRegex(SourceBelow1080Error, "source_below_1080"):
                YouTubeAcquirer(workspace, runner=metadata_runner).acquire(
                    "https://youtube.com/watch?v=low-demo", root / "job",
                    local_media=media, local_subtitles=subtitle, download_media=False,
                )

    def test_scheduled_discovery_selects_at_most_one_and_waits_48_hours(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            now = [datetime(2026, 8, 27, 0, 0, tzinfo=UTC)]
            runner = FakeYouTubeRunner()
            service = YouTubeDiscoveryService(workspace, runner=runner, clock=lambda: now[0])
            selected: list[str] = []
            config = DiscoveryConfig(query_pools={"karpathy": ["Andrej Karpathy AI"]})

            first = service.run(config, on_selected=lambda item: selected.append(item.video_id) or {"ok": True})
            self.assertEqual(first.status, "selected")
            self.assertEqual(selected, ["karpathy-1"])
            self.assertGreaterEqual(first.selected.score, 70)

            now[0] += timedelta(hours=47, minutes=59)
            second = service.run(config)
            self.assertEqual(second.status, "not_due")

            now[0] += timedelta(minutes=1)
            third = service.run(config)
            self.assertEqual(third.status, "no_selection")
            self.assertEqual(selected, ["karpathy-1"])
            self.assertTrue(all("player_client=mweb" in " ".join(command) for command in runner.commands))

    def test_low_quality_search_never_calls_generation(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            runner = FakeYouTubeRunner()
            service = YouTubeDiscoveryService(
                workspace, runner=runner,
                clock=lambda: datetime(2026, 8, 27, tzinfo=UTC),
            )
            called = []
            result = service.run(
                DiscoveryConfig(minimum_score=101, query_pools={"popular_ai": ["AI"]}),
                on_selected=lambda item: called.append(item.video_id),
            )
            self.assertEqual(result.status, "no_selection")
            self.assertEqual(called, [])

    def test_obvious_secondary_repost_is_rejected_even_when_hot(self) -> None:
        repost = YouTubeCandidate(
            video_id="repost", url="https://youtube.com/watch?v=repost",
            title="Andrej Karpathy: Software Is Changing Again", channel="Tech Clips Daily",
            description="Source: @stanfordonline. Andrej Karpathy explains AI agents.",
            published_at="20260826", duration_seconds=3600, view_count=2_000_000,
            transcript_available=True,
        )
        trusted = YouTubeCandidate(
            video_id="official", url="https://youtube.com/watch?v=official",
            title="Andrej Karpathy: Software Is Changing Again", channel="Stanford Online",
            description="A Stanford seminar for developers building AI systems.",
            published_at="20260826", duration_seconds=3600, view_count=20_000,
            transcript_available=True,
        )
        config = DiscoveryConfig()
        now = datetime(2026, 8, 27, tzinfo=UTC)

        YouTubeDiscoveryService._score(repost, config, now)
        YouTubeDiscoveryService._score(trusted, config, now)

        self.assertFalse(repost.eligible)
        self.assertIn("secondary_repost_source", repost.rejection_reasons)
        self.assertEqual(repost.score_breakdown["source_authority"], 9.0)
        self.assertTrue(trusted.eligible)
        self.assertEqual(trusted.score_breakdown["source_authority"], 20.0)

    def test_metadata_probe_budget_is_diversified_across_source_pools(self) -> None:
        candidates = [
            YouTubeCandidate(
                video_id=f"karpathy-{index}", url=f"https://youtube.com/watch?v=k{index}",
                title="Andrej Karpathy AI", channel="Stanford Online",
                view_count=1_000_000 - index, matched_pools=["karpathy"],
            )
            for index in range(5)
        ]
        candidates.append(YouTubeCandidate(
            video_id="yc-1", url="https://youtube.com/watch?v=yc1",
            title="AI Startup School", channel="Y Combinator",
            view_count=100, matched_pools=["yc"],
        ))
        config = DiscoveryConfig(
            metadata_probe_limit=2,
            query_pools={"karpathy": ["Karpathy"], "yc": ["YC AI"]},
        )

        selected = YouTubeDiscoveryService._choose_probe_candidates(candidates, config)

        self.assertEqual({item.matched_pools[0] for item in selected}, {"karpathy", "yc"})

    def test_json3_parser_uses_word_timing_preserves_fillers_and_removes_duplicates(self) -> None:
        payload = {
            "events": [
                {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "Um"}]},
                {"tStartMs": 1000, "dDurationMs": 3000, "segs": [
                    {"utf8": "Stop"}, {"utf8": " fixing", "tOffsetMs": 400},
                    {"utf8": " the", "tOffsetMs": 800}, {"utf8": " code.", "tOffsetMs": 1100},
                ]},
                {"tStartMs": 1000, "dDurationMs": 3000, "segs": [
                    {"utf8": "Stop"}, {"utf8": " fixing", "tOffsetMs": 400},
                ]},
            ],
        }
        with TemporaryDirectory() as temp:
            path = Path(temp) / "captions.json3"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cues = parse_youtube_json3(path)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].source_text, "Um")
        self.assertEqual(cues[1].source_text, "Stop fixing the code.")
        self.assertAlmostEqual(cues[1].start, 1.0)

    def test_json3_parser_restores_spaces_across_caption_events(self) -> None:
        payload = {"events": [
            {"tStartMs": 0, "dDurationMs": 1000, "segs": [
                {"utf8": "So,"}, {"utf8": " if", "tOffsetMs": 400},
            ]},
            {"tStartMs": 1000, "dDurationMs": 1800, "segs": [
                {"utf8": "you"}, {"utf8": " have"}, {"utf8": " a"},
                {"utf8": " story."},
            ]},
        ]}
        with TemporaryDirectory() as temp:
            path = Path(temp) / "captions.json3"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cues = parse_youtube_json3(path)

        self.assertEqual(cues[0].source_text, "So, if you have a story.")

    def test_terminology_preserves_english_when_chinese_is_not_natural(self) -> None:
        cues = [TranscriptCue(
            "c1", 0, 3, "Build a Harness and Skill registry.",
            "建立 Harness（Agent 的执行与反馈框架）和 Skill 注册中心。",
        )]
        terms = [
            TerminologyEntry("Harness", TerminologyStrategy.BILINGUAL_ONCE, first_use_explanation="Agent 的执行与反馈框架"),
            TerminologyEntry("Skill", TerminologyStrategy.PRESERVE),
        ]
        self.assertEqual(terminology_contract_errors(cues, terms), [])
        cues[0].translation = "建立挽具和技能登记处。"
        errors = terminology_contract_errors(cues, terms)
        self.assertTrue(any("Harness" in item for item in errors))
        self.assertTrue(any("挽具" in item for item in errors))

    def test_collection_plan_must_not_fall_back_to_mechanical_slices(self) -> None:
        cues = [
            TranscriptCue(f"c{index}", index * 5, index * 5 + 5, "Agent systems must scale.", "Agent 系统必须扩展。")
            for index in range(240)
        ]
        candidate = Candidate(
            "youtube-demo", SourceType.YOUTUBE, "https://youtube.com/watch?v=demo", "Agent Teams",
            author="AI Engineer", metadata={"video_id": "demo"},
        )
        weak_plan = {
            "collection_title": "AI 工程团队升级", "main_title": "团队升级",
            "main_ranges": [{"start": 0, "end": 1200}],
            "themes": [{"title": "片段", "thesis": "零散观点", "start": 0, "end": 120}],
        }
        with self.assertRaisesRegex(ValueError, "3–5 complete episodes"):
            build_collection_manifest(candidate, {"duration": 1200}, cues, [], weak_plan, "", "")

    def test_collection_manifest_round_trip_and_quality_contract(self) -> None:
        cues = [
            TranscriptCue(f"c{index}", index * 5, index * 5 + 5, "Agent systems must scale.", "Agent 系统必须扩展。")
            for index in range(240)
        ]
        candidate = Candidate(
            "youtube-demo", SourceType.YOUTUBE, "https://youtube.com/watch?v=demo", "Agent Teams",
            author="AI Engineer", metadata={"video_id": "demo"},
        )
        plan = {
            "collection_title": "AI 工程团队升级",
            "main_title": "AI Coding 真正难的是团队升级",
            "main_ranges": [{"start": 0, "end": 1200}],
            "themes": [
                {
                    "title": f"AI 团队扩展主题 {index}",
                    "thesis": "完整解释 AI 团队如何扩展",
                    "start": (index - 1) * 300,
                    "end": index * 300,
                    "hook_headlines": [
                        "团队扩展不能只靠工具",
                        "平台所有权应该归谁？",
                        "Harness 决定扩展上限",
                    ],
                }
                for index in range(1, 5)
            ],
        }
        manifest = build_collection_manifest(
            candidate, {"duration": 1200}, cues,
            [TerminologyEntry("Agent", TerminologyStrategy.PRESERVE)], plan, "", "",
            SourceMediaInfo(1920, 1080, 1200, "h264", "aac", "137", "mweb"),
        )
        manifest.rights_review = RightsReview(status="reviewed", reviewed_by="editor")
        checks = validate_collection(manifest)
        self.assertTrue(all(item.passed for item in checks), [item.detail for item in checks if not item.passed])
        restored = collection_manifest_from_dict(manifest.to_dict())
        self.assertEqual(restored.collection_title, "AI 工程团队升级")
        self.assertEqual(restored.items[1].source_ranges[0].framing, FramingMode.AUTO)
        self.assertEqual(len(restored.items), 5)
        wechat = next(
            render for render in restored.items[1].renders
            if render.profile == RenderProfile.WECHAT_VERTICAL
        )
        self.assertEqual(len(wechat.hook_candidates), 3)
        self.assertEqual(wechat.hook_candidates[0].headline_zh, "团队扩展不能只靠工具")
        self.assertAlmostEqual(
            sum(item.duration for item in render_source_ranges(restored.items[1], wechat)),
            restored.items[1].duration,
        )

        with TemporaryDirectory() as temp:
            source, chinese, bilingual = write_item_subtitle_files(
                restored, restored.items[1], wechat, Path(temp) / "episode",
            )
            self.assertIn("Agent systems must scale.", source.read_text(encoding="utf-8"))
            self.assertIn("Agent 系统必须扩展。", chinese.read_text(encoding="utf-8"))
            combined = bilingual.read_text(encoding="utf-8")
            self.assertLess(combined.index("Agent systems must scale."), combined.index("Agent 系统必须扩展。"))

    def test_short_source_creates_complete_bilibili_and_wechat_editions(self) -> None:
        duration = 1500
        cues = [
            TranscriptCue(
                f"c{index}", index * 5, index * 5 + 5,
                "Agent systems must scale through the team.", "Agent 系统必须通过团队扩展。",
            )
            for index in range(duration // 5)
        ]
        lessons = [
            {
                "title": f"完整短课主题 {index}", "thesis": "完整解释团队扩展路径。",
                "start": (index - 1) * 300, "end": index * 300,
                "hook_headlines": [
                    f"团队扩展不能只靠工具{index}",
                    f"平台责任应该归谁{index}？",
                    f"Harness 决定扩展上限{index}",
                ],
            }
            for index in range(1, 6)
        ]
        plan = {
            "collection_title": "完整 AI 学习合集", "story_start": 0, "story_end": duration,
            "bilibili_chapters": [{
                "title": "完整故事", "thesis": "完整保留原视频的学习路径。",
                "start": 0, "end": duration,
                "hook_headlines": [
                    "AI 越强，基本功越重要", "先写代码会放大什么风险？", "共同设计能减少返工",
                ],
            }],
            "wechat_lessons": lessons,
        }
        candidate = Candidate(
            "youtube-short", SourceType.YOUTUBE, "https://youtube.com/watch?v=short",
            "Complete Agent Story", author="AI Teacher", metadata={"video_id": "short"},
        )

        manifest = build_collection_manifest(
            candidate, {"duration": duration}, cues, [], plan, "", "",
            SourceMediaInfo(1920, 1080, duration, "h264", "aac", "137", "mweb"),
        )
        manifest.rights_review = RightsReview(status="reviewed", reviewed_by="editor")
        checks = validate_collection(manifest)

        self.assertTrue(all(item.passed for item in checks), [
            item.detail for item in checks if not item.passed
        ])
        chapters = [item for item in manifest.items if item.kind == CollectionItemKind.BILIBILI_CHAPTER]
        shorts = [item for item in manifest.items if item.kind == CollectionItemKind.WECHAT_SHORT]
        self.assertEqual((len(chapters), len(shorts)), (1, 5))
        self.assertEqual({row.profile for row in chapters[0].renders}, {RenderProfile.BILIBILI_LANDSCAPE})
        self.assertTrue(all(
            {row.profile for row in item.renders} == {RenderProfile.WECHAT_VERTICAL}
            for item in shorts
        ))
        self.assertTrue(next(item for item in checks if item.name == "wechat_story_coverage").passed)

    def test_long_source_requires_bilibili_coverage_but_allows_selected_wechat_lessons(self) -> None:
        duration = 7200
        cues = [
            TranscriptCue(
                f"c{index}", index * 5, index * 5 + 5,
                "The platform team must improve the Agent system.",
                "平台团队必须改进 Agent 系统。",
            )
            for index in range(duration // 5)
        ]
        plan = {
            "collection_title": "两小时 AI 深度学习合集",
            "story_start": 0, "story_end": duration,
            "bilibili_chapters": [
                {
                    "title": f"学习章节 {index}", "thesis": "完整保留这一阶段的论证。",
                    "start": (index - 1) * 1800, "end": index * 1800,
                    "hook_headlines": [
                        f"工具升级不等于团队升级{index}",
                        f"平台团队该先解决什么{index}？",
                        f"系统能力决定 Agent 上限{index}",
                    ],
                }
                for index in range(1, 5)
            ],
            "wechat_lessons": [
                {
                    "title": f"精选短课 {index}", "thesis": "提取一个可独立学习的关键观点。",
                    "start": index * 1000, "end": index * 1000 + 300,
                    "hook_headlines": [
                        f"工具升级不等于团队升级{index}",
                        f"平台团队应该负责什么{index}？",
                        f"系统能力决定 Agent 上限{index}",
                    ],
                }
                for index in range(1, 6)
            ],
        }
        candidate = Candidate(
            "youtube-long", SourceType.YOUTUBE, "https://youtube.com/watch?v=long",
            "Two Hour Agent Course", author="AI Teacher", metadata={"video_id": "long"},
        )

        manifest = build_collection_manifest(
            candidate, {"duration": duration}, cues, [], plan, "", "",
            SourceMediaInfo(1920, 1080, duration, "h264", "aac", "137", "mweb"),
        )
        manifest.rights_review = RightsReview(status="reviewed", reviewed_by="editor")
        checks = validate_collection(manifest)

        self.assertTrue(all(item.passed for item in checks), [
            item.detail for item in checks if not item.passed
        ])
        self.assertEqual(
            len([item for item in manifest.items if item.kind == CollectionItemKind.BILIBILI_CHAPTER]),
            4,
        )
        self.assertNotIn("wechat_story_coverage", {item.name for item in checks})

    def test_short_source_rejects_incomplete_wechat_coverage(self) -> None:
        plan = {
            "story_start": 0, "story_end": 1500,
            "bilibili_chapters": [{
                "title": "完整故事", "thesis": "完整论证", "start": 0, "end": 1500,
                "hook_headlines": ["工具升级不等于团队升级", "平台责任应该如何划分？", "系统能力决定 Agent 上限"],
            }],
            "wechat_lessons": [
                {
                    "title": f"短课主题 {index}", "thesis": "独立观点",
                    "start": index * 300, "end": index * 300 + 180,
                    "hook_headlines": ["团队不能只靠工具", "平台责任应该归谁？", "系统能力决定上限"],
                }
                for index in range(3)
            ],
        }

        errors = editorial_plan_contract_errors(plan, 1500)

        self.assertTrue(any("WeChat lessons must cover" in item for item in errors), errors)

    def test_short_source_boundaries_are_deterministic_and_complete(self) -> None:
        cues = [
            TranscriptCue(f"c{index}", index * 5, index * 5 + 5, "Complete thought.")
            for index in range(222)
        ]

        ranges = _required_short_source_ranges(cues, 0, 1106)

        self.assertEqual(len(ranges), 4)
        self.assertEqual(ranges[0]["start"], 0)
        self.assertEqual(ranges[-1]["end"], 1106)
        self.assertTrue(all(180 <= row["end"] - row["start"] <= 360 for row in ranges))
        self.assertTrue(all(
            ranges[index]["end"] == ranges[index + 1]["start"]
            for index in range(len(ranges) - 1)
        ))

    def test_dangling_source_fragments_merge_before_translation(self) -> None:
        cues = [
            TranscriptCue("c1", 0, 5.7, "You can write a specification about how an"),
            TranscriptCue("c2", 5.7, 7.4, "application is supposed to work."),
            TranscriptCue("c3", 8, 8.8, "And"),
            TranscriptCue("c4", 8.8, 14.5, "the agent will pick it up."),
        ]

        balanced = rebalance_source_cues(cues)

        self.assertEqual(len(balanced), 2)
        self.assertEqual(
            balanced[0].source_text,
            "You can write a specification about how an application is supposed to work.",
        )
        self.assertEqual(balanced[1].source_text, "And the agent will pick it up.")

    def test_subtitle_wrap_keeps_english_term(self) -> None:
        wrapped = wrap_subtitle("不要逐条修代码，要持续改进 Harness 和上下文。", 12)
        self.assertIn("Harness", wrapped)
        self.assertEqual(len(wrapped.splitlines()), 2)

    def test_chinese_subtitle_normalizes_spaces_around_punctuation(self) -> None:
        normalized = normalize_chinese_subtitle(
            "所以 ， 如果AI可行 ， 交给你的AFK Agent ； 我很乐意 。",
        )

        self.assertEqual(normalized, "所以，如果 AI 可行，交给你的 AFK Agent；我很乐意。")

    def test_headline_shortening_never_bisects_an_english_term(self) -> None:
        source = "Engineering platform ownership changes when Skill registry scales"
        shortened = _headline_fragment(source, 18)

        self.assertEqual(shortened, "Engineering")
        self.assertNotEqual(shortened, source[:18])

    def test_wechat_hook_keeps_intro_and_removes_later_duplicate(self) -> None:
        hook = HookSpec(
            "hook-1", HookStrategy.CONTRARIAN, "团队扩展不能只靠工具",
            "解释组织与工具链为何重要", SourceRange(40, 45),
            ["cue-1"], ["cue-1"], selected=True,
        )
        render = PlatformRender(
            RenderProfile.WECHAT_VERTICAL, 1080, 1920,
            hook_candidates=[hook], selected_hook=hook,
        )
        item = CollectionItem(
            "episode-1", CollectionItemKind.EPISODE, 1, "完整主题", "完整观点",
            [SourceRange(0, 300, FramingMode.SLIDE, "slide layout", 500, 80, 1320, 742)],
            [render],
        )

        ranges = render_source_ranges(item, render)

        self.assertEqual([(row.start, row.end) for row in ranges], [
            (40, 45), (0, 40), (45, 300),
        ])
        self.assertEqual(ranges[0].framing, FramingMode.SLIDE)
        self.assertTrue(ranges[0].has_explicit_crop)
        self.assertEqual((ranges[0].crop_x, ranges[0].crop_width), (500, 1320))
        self.assertEqual(sum(row.duration for row in ranges), item.duration)

    def test_wechat_split_layout_preserves_complete_composite_frame(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            source_subtitle = root / "source.srt"
            translation_subtitle = root / "translation.srt"
            source_subtitle.write_text("", encoding="utf-8")
            translation_subtitle.write_text("", encoding="utf-8")
            output = root / "split.mp4"
            source_range = SourceRange(
                0, 300, FramingMode.SPLIT, "speaker and slide composite",
                425, 0, 1495, 840,
            )
            item = CollectionItem(
                "lesson-1", CollectionItemKind.WECHAT_SHORT, 1,
                "完整主题", "完整观点", [source_range],
                [PlatformRender(RenderProfile.WECHAT_VERTICAL, 1080, 1920)],
            )
            runner = lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", "")

            YouTubeCollectionRenderer(Workspace(root), runner=runner)._render_one(
                source, item, item.renders[0], source_subtitle,
                translation_subtitle, output,
            )
            filters = output.with_suffix(".filters.txt").read_text(encoding="utf-8")

        self.assertIn("[src0]split=2[bg0][fg0]", filters)
        self.assertIn(
            "[fg0]scale=1080:720:force_original_aspect_ratio=decrease,"
            "pad=1080:720:(ow-iw)/2:(oh-ih)/2:black[fit0]",
            filters,
        )
        self.assertNotIn("[fg0]crop=", filters)

    def test_slide_translations_follow_reordered_hook_timeline(self) -> None:
        rows = _slide_translation_rows(
            [SlideTranslation(10, 20, "Deep Modules", "deep modules（深模块）")],
            [SourceRange(15, 20), SourceRange(0, 15), SourceRange(20, 30)],
        )

        self.assertEqual(rows, [
            (0, 5, "deep modules（深模块）", None, None),
            (15, 20, "deep modules（深模块）", None, None),
        ])

    def test_hook_ranking_prefers_concrete_engineering_language(self) -> None:
        cues = [
            TranscriptCue("weak", 0, 8, "I'm not the only one saying this in this event."),
            TranscriptCue("concrete", 10, 18, "Your Harness must automate the coding system."),
        ]

        hooks = build_hook_candidates(
            "团队自动化需要完整系统", "Harness 必须覆盖编码之外的自动化。",
            SourceRange(0, 300), cues, "episode-1",
            ["Harness 不止于编码", "该修代码还是改系统？", "自动化决定团队效率"],
        )

        self.assertEqual(hooks[0].source_range.start, 10)


if __name__ == "__main__":
    unittest.main()
