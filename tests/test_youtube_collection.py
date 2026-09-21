import json
import re
import subprocess
import unittest
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.models import (
    Candidate, CollectionItem, CollectionItemKind, Evidence, FramingMode, HookSpec, HookStrategy,
    PlatformRender, RenderProfile, RightsReview, SlideTranslation, SourceMediaInfo, SourceRange, SourceType,
    TerminologyEntry, TerminologyStrategy, TranscriptCue,
    VideoCollectionManifest,
)
from video_factory.media import AudioLoudness, VideoProbe, probe_audio_loudness, probe_video
from video_factory.serde import collection_manifest_from_dict
from video_factory.storage import Workspace
from video_factory.youtube import (
    DiscoveryConfig, NaturalSubtitleTranslator, YouTubeAcquirer, YouTubeCandidate,
    YouTubeCollectionFactory, YouTubeCollectionRenderer, YouTubeDiscoveryService,
    SourceBelow1080Error, YouTubeAcquisitionError,
    build_collection_manifest, build_hook_candidates, normalize_chinese_subtitle,
    audience_relevance, classify_youtube_editorial,
    _coerce_range, editorial_plan_contract_errors, normalize_editorial_plan_structure,
    _editorial_planning_transcript,
    _add_guidance_ranges_to_planning_transcript,
    _apply_supported_requested_title,
    _matching_completed_directing_audit,
    _requested_exact_range_from_editorial_guidance,
    parse_youtube_json3,
    rebalance_source_cues, render_source_ranges, terminology_contract_errors,
    validate_collection, wrap_subtitle, write_item_subtitle_files, _headline_fragment,
    _required_short_source_ranges, _slide_translation_rows, _write_hook_overlay_concat,
    _write_slide_translation_overlay_concat,
    _write_subtitle_overlay_concat, _fit_text_by_pixels,
    _interview_hook_context_fits_overlay,
    _fit_interview_hook_headline,
    _fit_subtitle_by_pixels,
    rebase_interview_clip_timeline, _local_interview_media_window,
    _previous_clip_for_local_timeline,
    enforce_cached_terminology_contract, omit_spoken_fillers_from_translation,
    omit_non_speech_directions, source_is_non_speech_only, source_is_spoken_filler_only,
    split_bilingual_subtitle_display,
    select_caption_incumbent,
    merge_dependent_subtitle_cues,
    fast_translation_cues,
    _resolve_chinese_subtitle_font_path,
    _resolve_headline_font_path,
    _semantic_english_parts,
    _coalesce_short_semantic_parts,
    _split_overlong_semantic_parts,
    cached_interview_caption_pipeline_complete,
    interview_caption_duration_errors,
    INTERVIEW_CAPTION_POLICY_VERSION,
    INTERVIEW_CAPTION_POLICY_FINGERPRINT,
    INTERVIEW_CAPTION_HARD_MAX_SECONDS,
    INTERVIEW_CAPTION_MAX_ENGLISH_WORDS,
    INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS,
    interview_asr_suspicions,
    interview_chinese_style_errors,
    targeted_whisper_caption_audit,
    _whisper_text_for_range,
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
    def test_human_accepted_caption_plan_beats_same_clip_challenger(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            incumbent = (
                workspace.root / "editorial" / "caption-incumbents" / "video-1.json"
            )
            incumbent.parent.mkdir(parents=True)
            incumbent.write_text(json.dumps({
                "source_video_id": "video-1",
                "editorial_mode": "known_tech_interview_clip",
                "source_clip": {"original_start": 100, "original_end": 190},
            }), encoding="utf-8")
            challenger = Path(temp) / "challenger.json"
            challenger.write_text(json.dumps({
                "source_video_id": "video-1",
                "editorial_mode": "known_tech_interview_clip",
                "source_clip": {"original_start": 100.2, "original_end": 190.2},
            }), encoding="utf-8")

            selected, trace = select_caption_incumbent(
                workspace, "video-1", challenger,
                "known_tech_interview_clip",
            )

        self.assertEqual(selected, incumbent)
        self.assertEqual(trace["step"], "caption_incumbent_retained")

    def test_caption_incumbent_never_overrides_a_different_requested_clip(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            incumbent = (
                workspace.root / "editorial" / "caption-incumbents" / "video-1.json"
            )
            incumbent.parent.mkdir(parents=True)
            incumbent.write_text(json.dumps({
                "source_video_id": "video-1",
                "editorial_mode": "known_tech_interview_clip",
                "source_clip": {"original_start": 100, "original_end": 190},
            }), encoding="utf-8")
            challenger = Path(temp) / "challenger.json"
            challenger.write_text(json.dumps({
                "source_video_id": "video-1",
                "editorial_mode": "known_tech_interview_clip",
                "source_clip": {"original_start": 500, "original_end": 590},
            }), encoding="utf-8")

            selected, trace = select_caption_incumbent(
                workspace, "video-1", challenger,
                "known_tech_interview_clip",
            )

        self.assertEqual(selected, challenger)
        self.assertIsNone(trace)

    def test_audio_verified_successor_of_incumbent_is_reused_without_repair_loop(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            incumbent = (
                workspace.root / "editorial" / "caption-incumbents" / "video-1.json"
            )
            incumbent.parent.mkdir(parents=True)
            base = {
                "source_video_id": "video-1",
                "editorial_mode": "known_tech_interview_clip",
                "source_clip": {"original_start": 100, "original_end": 190},
            }
            incumbent.write_text(json.dumps(base), encoding="utf-8")
            successor = Path(temp) / "successor.json"
            successor.write_text(json.dumps({
                **base,
                "trace": [
                    {"step": "caption_incumbent_retained", "incumbent_plan": str(incumbent)},
                    {"step": "targeted_whisper_caption_audit", "corrections": []},
                    {"step": "audio_verified_caption_repair", "cue_ids": ["card"]},
                ],
            }), encoding="utf-8")

            selected, trace = select_caption_incumbent(
                workspace, "video-1", successor,
                "known_tech_interview_clip",
            )

        self.assertEqual(selected, successor)
        self.assertEqual(trace["step"], "caption_incumbent_verified_successor_reused")

    def test_asr_suspicion_gate_is_text_only_and_targets_noisy_spans(self) -> None:
        cues = [
            TranscriptCue(
                "repeat", 10, 15,
                "constraints constraints extraordinary companies being built", "",
            ),
            TranscriptCue(
                "truncated", 20, 26,
                "The compet cycle moves earnings toward applications", "",
            ),
            TranscriptCue(
                "clean", 30, 36,
                "The compute cycle moves earnings toward applications", "",
            ),
        ]

        findings = interview_asr_suspicions(cues)

        self.assertEqual([row["cue_id"] for row in findings], ["repeat", "truncated"])
        self.assertIn("repeated_content_word:constraints", findings[0]["reasons"])
        self.assertIn("unknown_cycle_modifier:compet", findings[1]["reasons"])

    def test_asr_suspicion_gate_ignores_repeated_spoken_contractions(self) -> None:
        cues = [TranscriptCue(
            "stutter", 10, 12, "And it's it's so good at that.", "",
        )]

        self.assertEqual(interview_asr_suspicions(cues), [])

    def test_bilingual_enforcement_does_not_duplicate_existing_explanation(self) -> None:
        cues = [TranscriptCue(
            "card", 0, 4, "The system of record owns the data.",
            "业务数据主记录拥有这些数据。",
        )]
        terms = [TerminologyEntry(
            "system of record", TerminologyStrategy.BILINGUAL_ONCE,
            first_use_explanation="业务数据主记录",
        )]

        NaturalSubtitleTranslator._enforce_terminology_contract(cues, terms)

        self.assertEqual(cues[0].translation.count("业务数据主记录"), 1)
        self.assertIn("system of record：业务数据主记录", cues[0].translation)
        self.assertEqual(terminology_contract_errors(cues, terms), [])

    def test_translated_subterm_does_not_destroy_preserved_phrase(self) -> None:
        cues = [TranscriptCue(
            "card", 0, 5,
            "The model still depends on legacy infrastructure.",
            "模型仍然依赖 legacy infrastructure。",
        )]
        terms = [
            TerminologyEntry(
                "legacy infrastructure", TerminologyStrategy.PRESERVE,
            ),
            TerminologyEntry(
                "infrastructure", TerminologyStrategy.TRANSLATE, target="基础设施",
            ),
        ]

        NaturalSubtitleTranslator._enforce_terminology_contract(cues, terms)

        self.assertIn("legacy infrastructure", cues[0].translation)
        self.assertEqual(terminology_contract_errors(cues, terms), [])

    def test_whisper_words_are_sliced_to_only_the_suspicious_card_range(self) -> None:
        payload = {"segments": [{"words": [
            {"start": 0.0, "end": 1.0, "word": " outside"},
            {"start": 2.0, "end": 2.5, "word": " The"},
            {"start": 2.5, "end": 3.0, "word": " compute"},
            {"start": 3.0, "end": 3.6, "word": " cycle."},
            {"start": 5.0, "end": 6.0, "word": " later"},
        ]}]}

        self.assertEqual(
            _whisper_text_for_range(payload, 1.8, 3.8), "The compute cycle.",
        )

    def test_targeted_whisper_audit_runs_once_then_uses_its_cache(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(command, **kwargs):
            calls.append(command)
            output_dir = Path(command[command.index("--output_dir") + 1])
            media = Path(command[1])
            (output_dir / f"{media.stem}.json").write_text(json.dumps({
                "segments": [{"words": [
                    {"start": 1.0, "end": 1.5, "word": " constraints"},
                    {"start": 1.5, "end": 2.0, "word": ","},
                    {"start": 2.0, "end": 2.5, "word": " extraordinary"},
                    {"start": 2.5, "end": 3.0, "word": " companies."},
                ]}],
            }), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        with TemporaryDirectory() as temp:
            root = Path(temp)
            media = root / "clip.mkv"
            media.write_bytes(b"media")
            cues = [TranscriptCue(
                "card", 1, 3, "constraints constraints extraordinary companies", "",
            )]
            with patch("video_factory.youtube.shutil.which", return_value="/usr/bin/whisper"):
                first = targeted_whisper_caption_audit(
                    media, cues, root / "job", runner=fake_runner,
                )
                second = targeted_whisper_caption_audit(
                    media, cues, root / "job", runner=fake_runner,
                )

        self.assertEqual(len(calls), 1)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(
            first["corrections"][0]["verified_source"],
            "constraints, extraordinary companies.",
        )

    def test_targeted_whisper_keeps_context_for_proportionally_timed_cards(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(command, **kwargs):
            calls.append(command)
            output_dir = Path(command[command.index("--output_dir") + 1])
            media = Path(command[1])
            words = "like like all obviously an insane amount of value went into the text".split()
            (output_dir / f"{media.stem}.json").write_text(json.dumps({
                "segments": [{"words": [
                    {"start": 21.8 + index * 0.28,
                     "end": 22.05 + index * 0.28,
                     "word": " " + word}
                    for index, word in enumerate(words)
                ]}],
            }), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        with TemporaryDirectory() as temp:
            root = Path(temp)
            media = root / "clip.mkv"
            media.write_bytes(b"media")
            cues = [TranscriptCue(
                "card", 13.867, 16.061,
                "like like all obviously an insane amount of value went into the text", "",
            )]
            with patch("video_factory.youtube.shutil.which", return_value="/usr/bin/whisper"):
                audit = targeted_whisper_caption_audit(
                    media, cues, root / "job", runner=fake_runner,
                )

        timestamps = calls[0][calls[0].index("--clip_timestamps") + 1]
        self.assertEqual(timestamps, "0.000,31.061")
        self.assertEqual(
            audit["corrections"][0]["verified_source"],
            "like like all obviously an insane amount of value went into the text",
        )

    def test_default_discovery_tracks_selected_investor_and_operator_channels(self) -> None:
        config = DiscoveryConfig()

        self.assertEqual(
            config.query_pools["all_in"],
            ['"All-In Podcast" AI', '"All-In Podcast" SaaS'],
        )
        self.assertIn("Sequoia Capital AI startup engineering", config.query_pools["sequoia"])
        self.assertIn("a16z AI enterprise software", config.query_pools["a16z"])
        self.assertIn(
            "Lightspeed Venture Partners AI startup engineering",
            config.query_pools["lightspeed"],
        )
        self.assertIn('"The MAD Podcast" enterprise AI', config.query_pools["mad_podcast"])
        self.assertIn('"SaaStr AI" SaaS', config.query_pools["saastr"])

        configured = DiscoveryConfig.from_path(Path("examples/youtube_discovery.json"))
        self.assertEqual(
            configured.channel_sources["all_in"],
            ["https://www.youtube.com/channel/UCESLZhusAkFfsNsApnjF_Cg"],
        )
        self.assertEqual(configured.minimum_audience_score, 12)

    def test_audience_relevance_uses_weighted_exact_markers(self) -> None:
        accidental, _, accidental_matches = audience_relevance(
            "The chair said this is a general team conversation.",
        )
        generic, _, _ = audience_relevance(
            "An AI startup team has a general founder conversation.",
        )
        technical, breakdown, matches = audience_relevance(
            "Nvidia released a new inference GPU with 40% higher throughput for production systems.",
        )

        self.assertEqual(accidental, 0)
        self.assertNotIn("ai", accidental_matches["topic_markers"])
        self.assertLess(generic, 12)
        self.assertGreaterEqual(technical, 12)
        self.assertGreater(breakdown["technical_specificity"], 0)
        self.assertIn("hardware_compute", matches["topic_groups"])

    def test_long_interview_planning_uses_bounded_nonpolitical_technical_windows(self) -> None:
        cues = []
        for index in range(80):
            start = float(index * 15)
            text = (
                "The election and government debate dominates this political segment."
                if index < 24 else
                "Nvidia engineers explain inference GPU throughput, production systems, and deployment."
            )
            cues.append(TranscriptCue(
                id=f"cue-{index}", start=start, end=start + 15, source_text=text,
            ))
        metadata = {
            "duration": 1200,
            "chapters": [
                {"start_time": 0, "end_time": 360, "title": "Election debate"},
                {"start_time": 360, "end_time": 1200, "title": "Nvidia inference systems"},
            ],
        }

        rows, trace = _editorial_planning_transcript(
            metadata, cues, "known_tech_interview_clip", maximum_characters=2400,
        )

        combined = " ".join(str(row["text"]) for row in rows).casefold()
        self.assertEqual(trace["mode"], "ranked_nonpolitical_windows")
        self.assertLessEqual(trace["planning_characters"], 2400)
        self.assertIn("inference gpu throughput", combined)
        self.assertNotIn("election", combined)

    def test_timestamp_guidance_adds_omitted_source_range_to_planning_input(self) -> None:
        cues = [
            TranscriptCue("early", 0, 15, "GPU throughput and inference systems."),
            TranscriptCue("target", 2049, 2060, "This industry is not just models; it is mostly applications."),
            TranscriptCue("payoff", 2127, 2138, "The competitive cycle moves earnings toward the application layer."),
        ]

        rows, ranges = _add_guidance_ranges_to_planning_transcript(
            [{"id": "early", "start": 0, "end": 15, "chapter": "", "text": cues[0].source_text}],
            cues,
            "Use the continuous passage around 34:04–35:38 and include its payoff.",
        )

        self.assertEqual(ranges, [{"start": 2044.0, "end": 2138.0}])
        self.assertEqual([row["id"] for row in rows], ["early", "target", "payoff"])

    def test_structural_interview_plan_is_repaired_before_another_llm_call(self) -> None:
        class NoCallWriter:
            def _request_json(self, *args, **kwargs):
                raise AssertionError("deterministic structure repair should run first")

        cues = [
            TranscriptCue(
                id=f"cue-{index}", start=float(index * 10), end=float(index * 10 + 10),
                source_text="Nvidia explains inference systems and production throughput.",
            )
            for index in range(18)
        ]
        plan = {
            "editorial_mode": "known_tech_interview_clip",
            "collection_title": "Nvidia 系统判断",
            "bilibili_chapters": [],
            "wechat_lessons": [{
                "start": 20, "end": 120, "title": "", "thesis": "",
                "speaker_label": "Nvidia 高管", "framing": "speaker",
                "hook_headlines": [
                    "推理需求正在改写软件市场",
                    "GPU吞吐决定生产系统成本",
                    "SaaS并未被AI直接淘汰",
                ],
            }],
        }

        repaired, trace = NaturalSubtitleTranslator(NoCallWriter()).ensure_editorial_plan(
            {"duration": 180}, cues, plan, "known_tech_interview_clip",
        )

        self.assertEqual(trace[0]["step"], "deterministic_editorial_structure_pre_repair")
        self.assertTrue(repaired["wechat_lessons"][0]["title"])
        self.assertTrue(repaired["wechat_lessons"][0]["thesis"])

    def test_interview_speaker_label_falls_back_to_metadata_participants(self) -> None:
        class NoCallWriter:
            def _request_json(self, *args, **kwargs):
                raise AssertionError("metadata-grounded speaker repair should not call the LLM")

        cues = [
            TranscriptCue(
                id=f"cue-{index}", start=float(index * 10), end=float(index * 10 + 10),
                source_text="AI demand is outrunning compute supply and changing infrastructure.",
            )
            for index in range(18)
        ]
        plan = {
            "editorial_mode": "known_tech_interview_clip",
            "collection_title": "AI 需求与算力",
            "bilibili_chapters": [],
            "wechat_lessons": [{
                "start": 20, "end": 120, "title": "AI需求跑赢算力供应",
                "thesis": "AI重度用户增长快于算力扩张。", "framing": "speaker",
                "hook_headlines": [
                    "AI短缺比泡沫更值得警惕",
                    "重度用户正在推高算力需求",
                    "算力供应决定AI扩散速度",
                ],
            }],
        }
        metadata = {
            "duration": 180, "channel": "a16z",
            "description": "a16z’s David George sits down with Gavin Baker to unpack AI demand.",
        }

        repaired, trace = NaturalSubtitleTranslator(NoCallWriter()).ensure_editorial_plan(
            metadata, cues, plan, "known_tech_interview_clip",
        )

        self.assertEqual(
            repaired["wechat_lessons"][0]["speaker_label"],
            "David George × Gavin Baker",
        )
        self.assertEqual(editorial_plan_contract_errors(repaired, 180, cues), [])
        self.assertEqual(trace[0]["step"], "deterministic_editorial_structure_pre_repair")

    def test_overlong_interview_speaker_label_falls_back_to_metadata_identity(self) -> None:
        class NoCallWriter:
            def _request_json(self, *args, **kwargs):
                raise AssertionError("metadata fallback should repair the invalid label")

        cues = [TranscriptCue(
            "claim", 0, 60,
            "Marketing dashboards are dead; this becomes high frequency trading, AI driven and human supervised.",
        )]
        plan = {
            "editorial_mode": "known_tech_interview_clip",
            "collection_title": "营销自动化", "bilibili_chapters": [],
            "wechat_lessons": [{
                "start": 0, "end": 60,
                "title": "营销仪表盘将死：AI把营销变成高频交易",
                "thesis": "AI执行，人类监督。",
                "speaker_label": "一个明显超过移动端身份栏长度限制的冗长嘉宾介绍文本",
                "framing": "speaker",
                "hook_headlines": [
                    "营销仪表盘正在消失", "AI接管营销执行", "人类只负责监督",
                ],
            }],
        }

        repaired, _ = NaturalSubtitleTranslator(NoCallWriter()).ensure_editorial_plan(
            {"duration": 60, "channel": "a16z"}, cues, plan,
            "known_tech_interview_clip",
        )

        self.assertEqual(repaired["wechat_lessons"][0]["speaker_label"], "a16z 对谈")

    def test_interview_speaker_fallback_extracts_guest_who_joins_show(self) -> None:
        from video_factory.youtube import _metadata_speaker_label

        label = _metadata_speaker_label({
            "channel": "Lightspeed Venture Partners",
            "description": "Profound co-founder and CEO, James Cadwallader, joins Lightspeed’s Claire Zau.",
        })

        self.assertEqual(label, "James Cadwallader")

    def test_incomplete_terminology_rows_downgrade_to_source_preservation(self) -> None:
        cues = [TranscriptCue(
            "cue-1", 0, 5,
            "The headless agent writes to the system of record.",
        )]

        entries = NaturalSubtitleTranslator._parse_terminology([
            {"source": "system of record", "strategy": "bilingual_once"},
            {"source": "headless", "strategy": "bilingual_once", "target": ""},
            {"source": "agent", "strategy": "translate"},
        ], cues)

        self.assertEqual(
            [entry.strategy for entry in entries],
            [
                TerminologyStrategy.PRESERVE,
                TerminologyStrategy.PRESERVE,
                TerminologyStrategy.TRANSLATE,
            ],
        )
        self.assertEqual(entries[2].target, "智能体")

    def test_established_technical_terms_use_natural_chinese_not_forced_english(self) -> None:
        cues = [TranscriptCue(
            "cue-1", 0, 8,
            "Open-source competition lowers infrastructure costs for the application layer.",
        )]

        entries = NaturalSubtitleTranslator._parse_terminology([
            {"source": "open-source", "strategy": "preserve"},
            {"source": "infrastructure", "strategy": "preserve"},
            {"source": "application layer", "strategy": "preserve"},
            {"source": "KV cache", "strategy": "preserve"},
        ], cues)

        self.assertEqual(
            [(entry.source, entry.strategy, entry.target) for entry in entries],
            [
                ("open-source", TerminologyStrategy.TRANSLATE, "开源"),
                ("infrastructure", TerminologyStrategy.TRANSLATE, "基础设施"),
                ("application layer", TerminologyStrategy.TRANSLATE, "应用层"),
                ("KV cache", TerminologyStrategy.PRESERVE, ""),
            ],
        )

    def test_ordinary_interview_terms_cannot_be_marked_english_preserving(self) -> None:
        cues = [
            TranscriptCue(
                "vertical", 0, 4, "It is a high-paying vertical.",
                "这是个高薪 vertical：垂直行业。",
            ),
            TranscriptCue(
                "pricing", 4, 8, "They can manage token pricing.",
                "他们能靠 token pricing 来调节。",
            ),
        ]

        terms = NaturalSubtitleTranslator._parse_terminology([
            {
                "source": "vertical", "strategy": "bilingual_once",
                "target": "垂直行业", "first_use_explanation": "垂直行业",
            },
            {"source": "token pricing", "strategy": "preserve"},
        ], cues)
        NaturalSubtitleTranslator._enforce_terminology_contract(cues, terms)

        self.assertEqual(
            [(term.source, term.strategy, term.target) for term in terms],
            [
                ("vertical", TerminologyStrategy.TRANSLATE, "垂直行业"),
                ("token pricing", TerminologyStrategy.TRANSLATE, "token 定价"),
            ],
        )
        self.assertEqual(cues[0].translation, "这是个高薪的垂直行业。")
        self.assertEqual(cues[1].translation, "他们能靠 token 定价来调节。")
        self.assertEqual(terminology_contract_errors(cues, terms), [])

    def test_latin_term_matching_does_not_treat_agentic_as_agent(self) -> None:
        cues = [TranscriptCue(
            "card", 0, 4, "They deploy an agentic system.", "他们部署智能体系统。",
        )]
        terms = [TerminologyEntry("Agent", TerminologyStrategy.PRESERVE)]

        enforced = NaturalSubtitleTranslator._enforce_terminology_contract(cues, terms)

        self.assertEqual(enforced, [])
        self.assertNotIn("Agent", cues[0].translation)
        self.assertEqual(terminology_contract_errors(cues, terms), [])

    def test_bilingual_explanation_is_kept_only_on_first_source_use(self) -> None:
        cues = [
            TranscriptCue(
                "cue-1", 0, 5, "The system of record owns the data.",
                "system of record（业务数据的权威存储系统）拥有数据。",
            ),
            TranscriptCue(
                "cue-2", 5, 10, "A new system of record may emerge.",
                "新的system of record：业务数据的权威存储系统可能出现。",
            ),
        ]
        terminology = [TerminologyEntry(
            "system of record", TerminologyStrategy.BILINGUAL_ONCE,
            first_use_explanation="业务数据的权威存储系统",
        )]

        enforced = NaturalSubtitleTranslator._enforce_terminology_contract(cues, terminology)

        self.assertIn("system of record", enforced)
        self.assertEqual(
            "\n".join(cue.translation for cue in cues).count("业务数据的权威存储系统"),
            1,
        )
        self.assertEqual(terminology_contract_errors(cues, terminology), [])

    def test_channel_sources_are_primary_and_search_is_recent_filtered(self) -> None:
        commands = []

        def runner(command, **kwargs):
            commands.append(command)
            if command[-1].endswith("/videos"):
                payload = {"entries": [{
                    "id": "channel-new", "title": "Fresh Nvidia AI systems episode",
                    "channel": "All-In Podcast", "url": "https://youtube.com/watch?v=channel-new",
                }]}
            else:
                payload = {"entries": [{
                    "id": "search-new", "title": "Recent AI engineering interview",
                    "channel": "Engineering", "url": "https://youtube.com/watch?v=search-new",
                }]}
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

        with TemporaryDirectory() as temp:
            service = YouTubeDiscoveryService(
                Workspace(Path(temp)), runner=runner,
                clock=lambda: datetime(2026, 8, 27, tzinfo=UTC),
            )
            found = service._search(DiscoveryConfig(
                lookback_days=30, results_per_query=3,
                query_pools={"all_in": ["All-In Podcast AI"]},
                channel_sources={"all_in": ["https://youtube.com/channel/all-in"]},
            ))

        by_id = {item.video_id: item for item in found}
        self.assertIn("channel:all_in", by_id["channel-new"].discovery_routes)
        self.assertEqual(by_id["channel-new"].channel_recency_rank, 0)
        search_command = next(command for command in commands if command[-1].startswith("ytsearch"))
        self.assertIn("after:2026-07-28", search_command[-1])
        self.assertEqual(
            {row["route"] for row in service.last_trace["sources"]},
            {"channel", "search"},
        )

    def test_failed_channel_feed_falls_back_to_semantic_search(self) -> None:
        def runner(command, **kwargs):
            if command[-1].endswith("/videos"):
                return subprocess.CompletedProcess(command, 1, "", "channel unavailable")
            payload = {"entries": [{
                "id": "fallback", "title": "AI deployment architecture",
                "channel": "Engineering", "url": "https://youtube.com/watch?v=fallback",
            }]}
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

        with TemporaryDirectory() as temp:
            service = YouTubeDiscoveryService(
                Workspace(Path(temp)), runner=runner,
                clock=lambda: datetime(2026, 8, 27, tzinfo=UTC),
            )
            found = service._search(DiscoveryConfig(
                query_pools={"source": ["AI deployment"]},
                channel_sources={"source": ["https://youtube.com/channel/source"]},
            ))

        self.assertEqual([item.video_id for item in found], ["fallback"])
        self.assertEqual(service.last_trace["funnel"]["sources_failed"], 1)
        self.assertEqual(service.last_trace["sources"][0]["status"], "failed")

    def test_unknown_robotics_startup_team_talk_is_technical_coverage(self) -> None:
        mode, people, political = classify_youtube_editorial(
            "How our robotics startup built an autonomous humanoid",
            "Field Robotics Lab",
            "The founding team explains perception, motion planning, sim-to-real, and field tests.",
            [{"title": "Robot perception architecture"}],
            ["Startup CTO", "Robotics Lead"],
        )

        self.assertEqual(mode, "technical_coverage")
        self.assertEqual(people, [])
        self.assertEqual(political, [])

    def test_curated_investor_channel_interview_routes_to_one_clip(self) -> None:
        mode, people, political = classify_youtube_editorial(
            "Atlassian CEO on the SaaS Apocalypse, AI Agents & What Comes Next",
            "a16z",
            "Alex Rampell and Erik Torenberg speak with the Atlassian cofounder.",
            [],
            ["Alex Rampell", "Mike Cannon-Brookes"],
        )

        self.assertEqual(mode, "known_tech_interview_clip")
        self.assertEqual(people, [])
        self.assertEqual(political, [])

    def test_curated_investor_channel_non_interview_routes_to_one_clip(self) -> None:
        mode, _, _ = classify_youtube_editorial(
            "Three Predictions for Enterprise AI",
            "Sequoia Capital",
            "A solo presentation about enterprise software and AI infrastructure.",
            [],
            ["Investor"],
        )

        self.assertEqual(mode, "known_tech_interview_clip")

    def test_curated_channel_nontechnical_video_is_rejected(self) -> None:
        mode, _, political = classify_youtube_editorial(
            "Election Roundtable",
            "All-In Podcast",
            "A discussion of candidates, polling, and government policy.",
            [],
            ["Host A", "Host B"],
        )

        self.assertEqual(mode, "political_rejected")
        self.assertTrue(political)

    def test_curated_channel_ai_infrastructure_investment_thesis_routes_to_clip(self) -> None:
        mode, _, political = classify_youtube_editorial(
            "Dan Dreyfus: The Next AI Bottleneck is Copper",
            "All-In Podcast",
            "A presentation on critical minerals, copper demand, the power grid, and investing in AI infrastructure.",
            [
                {"title": "Copper's rise"},
                {"title": "The grid is dying: blackouts and bottlenecks"},
            ],
            ["Dan Dreyfus"],
        )

        self.assertEqual(mode, "known_tech_interview_clip")
        self.assertEqual(political, [])

    def test_curated_channel_tech_company_market_interview_is_outside_it_scope(self) -> None:
        mode, _, _ = classify_youtube_editorial(
            "Why Secondary Markets Are Eating the IPO",
            "All-In Podcast",
            "Founders and investors discuss why technology companies stay private and how VC liquidity is changing.",
            [],
            ["Founder", "Investor"],
        )

        self.assertEqual(mode, "rejected")

    def test_all_in_sponsor_copy_cannot_qualify_off_scope_episodes(self) -> None:
        cases = [
            (
                "Bill Gurley: Searching for Feynman",
                "A talk about the Challenger inquiry and the O-ring ice-water demonstration.",
            ),
            (
                "Jared Isaacman: A New Era for NASA and American Space Exploration",
                "NASA strategy, Moon missions, Mars, new vehicles, and deep-space research.",
            ),
            (
                "Nick Shirley: Exposing Government Fraud & Taking on the Media",
                "Investigative journalism, welfare fraud, and criticism of mainstream outlets.",
            ),
        ]
        sponsor_copy = (
            " Thanks to our partners for making this possible! "
            "IREN is an AI Cloud platform delivering compute and software for inference. "
            "Google helps startups build faster with artificial intelligence."
        )
        now = datetime(2026, 9, 19, tzinfo=UTC)

        for index, (title, description) in enumerate(cases):
            with self.subTest(title=title):
                item = YouTubeCandidate(
                    video_id=f"off-scope-{index}",
                    url=f"https://youtube.com/watch?v=offscope{index}",
                    title=title, channel="All-In Podcast",
                    description=description + sponsor_copy,
                    published_at="20260919", duration_seconds=1800,
                    view_count=200_000, transcript_available=True,
                )

                YouTubeDiscoveryService._score(item, DiscoveryConfig(), now)

                self.assertFalse(item.eligible)
                self.assertEqual(item.scope_markers, [])
                self.assertIn("outside_it_software_ai_scope", item.rejection_reasons)

    def test_episode_subject_still_qualifies_when_sponsor_copy_is_removed(self) -> None:
        item = YouTubeCandidate(
            video_id="ai-subject", url="https://youtube.com/watch?v=aisubject",
            title="How developers deploy AI agents", channel="All-In Podcast",
            description=(
                "A production software architecture discussion with API and inference details. "
                "Thanks to our partners for making this possible! Generic sponsor copy follows."
            ),
            published_at="20260919", duration_seconds=1800,
            view_count=200_000, transcript_available=True,
        )

        YouTubeDiscoveryService._score(
            item, DiscoveryConfig(), datetime(2026, 9, 19, tzinfo=UTC),
        )

        self.assertTrue(item.eligible)
        self.assertIn("software", item.scope_markers)

    def test_known_tech_interview_survives_politics_elsewhere_in_full_source(self) -> None:
        mode, people, political = classify_youtube_editorial(
            "Riding AGI, AI Anxiety, Who Funded COVID, Defending Taiwan",
            "Naval", "A long conversation about technology and society", [], ["Naval", "Nivi"],
        )

        self.assertEqual(mode, "known_tech_interview_clip")
        self.assertIn("naval", people)
        self.assertTrue(political)

    def test_country_technology_and_education_comparison_is_allowed(self) -> None:
        cues = [
            TranscriptCue("political-outside", 0, 8, "The election and government debate came first.", "前面谈到其他话题。"),
            TranscriptCue("safe-1", 100, 108, "China and the United States teach software engineering differently.", "中美的软件工程教育方式不同。"),
            TranscriptCue("safe-2", 108, 116, "Students should learn to inspect what AI generated.", "学生要学会检查 AI 生成的代码。"),
            TranscriptCue("safe-3", 116, 124, "That skill matters more when code becomes cheap.", "代码越便宜，这项能力越重要。"),
            TranscriptCue("safe-4", 124, 209, "Engineering education must emphasize judgment and verification.", "工程教育更要强调判断与验证。"),
        ]
        plan = {
            "editorial_mode": "known_tech_interview_clip",
            "collection_title": "科技人物高光",
            "story_start": 100, "story_end": 209, "bilibili_chapters": [],
            "wechat_lessons": [{
                "speaker_label": "C++之父", "title": "AI 越会写，越要学会验",
                "thesis": "AI 降低代码成本后，判断与验证更重要。",
                "start": 100, "end": 209, "framing": "speaker",
                "hook_headlines": [
                    "AI 写得越快，人越要会验", "代码便宜后，判断力更贵", "不同国家都在补同一课",
                ],
            }],
        }
        candidate = Candidate(
            "youtube-interview", SourceType.YOUTUBE, "https://youtube.com/watch?v=interview",
            "Known technologist interview", author="InfoQ", metadata={"video_id": "interview"},
        )
        manifest = build_collection_manifest(
            candidate, {"duration": 600, "known_tech_people": ["Bjarne Stroustrup"]},
            cues, [], plan, "", "",
            SourceMediaInfo(1920, 1080, 600, "h264", "aac", "137", "mweb"),
        )
        manifest.rights_review = RightsReview(status="reviewed", reviewed_by="editor")

        checks = validate_collection(manifest)

        self.assertEqual(manifest.editorial_mode, "known_tech_interview_clip")
        self.assertEqual(len(manifest.items), 1)
        self.assertEqual(manifest.items[0].renders[0].selected_hook.speaker_label, "C++之父")
        self.assertTrue(next(item for item in checks if item.name.endswith(":non_political")).passed)

    def test_selected_interview_clip_rejects_political_words(self) -> None:
        cues = [TranscriptCue("p", 100, 170, "The election changed the government's war policy.")]
        plan = {
            "editorial_mode": "known_tech_interview_clip", "story_start": 100, "story_end": 170,
            "bilibili_chapters": [], "wechat_lessons": [{
                "speaker_label": "Naval", "title": "技术判断", "thesis": "讨论技术判断。",
                "start": 100, "end": 170,
                "hook_headlines": ["技术判断改变路径", "真正的系统代价", "团队应该如何选择"],
            }],
        }

        errors = editorial_plan_contract_errors(plan, 600, cues)

        self.assertTrue(any("political content" in error for error in errors), errors)

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

    def test_acquirer_falls_back_to_archived_metadata_on_network_failure(self) -> None:
        runner = lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, "", "SSL EOF",
        )
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            previous = workspace.root / "jobs" / "previous"
            previous.mkdir(parents=True)
            metadata = {
                "id": "cached123", "title": "Cached technical lesson",
                "channel": "Teacher", "duration": 900, "upload_date": "20260820",
                "chapters": [],
            }
            (previous / "cached123.metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8",
            )
            subtitles = previous / "cached123.en.json3"
            subtitles.write_text(json.dumps({"events": [{
                "tStartMs": 0, "dDurationMs": 2000,
                "segs": [{"utf8": "Technical lesson."}],
            }]}), encoding="utf-8")

            candidate, _, loaded, cues, _, _, _ = YouTubeAcquirer(
                workspace, runner=runner,
            ).acquire(
                "https://youtube.com/watch?v=cached123", workspace.root / "jobs" / "retry",
                local_subtitles=subtitles, download_media=False,
            )

            self.assertEqual(candidate.id, "youtube-cached123")
            self.assertEqual(loaded["title"], "Cached technical lesson")
            self.assertEqual(cues[0].source_text, "Technical lesson.")

    def test_acquirer_falls_back_to_archived_subtitle_on_network_failure(self) -> None:
        calls: list[list[str]] = []

        def runner(command, **kwargs):
            calls.append(command)
            if "--dump-single-json" in command:
                return subprocess.CompletedProcess(command, 0, json.dumps({
                    "id": "cached123", "title": "Cached technical lesson",
                    "channel": "Teacher", "duration": 900,
                    "upload_date": "20260820", "chapters": [],
                }), "")
            return subprocess.CompletedProcess(command, 1, "", "SSL EOF")

        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            previous = workspace.root / "jobs" / "previous"
            previous.mkdir(parents=True)
            (previous / "cached123.en-orig.json3").write_text(json.dumps({
                "events": [{
                    "tStartMs": 1000, "dDurationMs": 2500,
                    "segs": [{"utf8": "Archived transcript survives refresh."}],
                }],
            }), encoding="utf-8")

            candidate, _, _, cues, _, subtitle_asset, _ = YouTubeAcquirer(
                workspace, runner=runner,
            ).acquire(
                "https://youtube.com/watch?v=cached123",
                workspace.root / "jobs" / "retry", download_media=False,
            )

            self.assertEqual(candidate.id, "youtube-cached123")
            self.assertEqual(cues[0].source_text, "Archived transcript survives refresh.")
            self.assertIn("youtube-subtitles", subtitle_asset)
            self.assertTrue(any("--write-auto-sub" in command for command in calls))

    def test_interview_download_command_uses_only_padded_selected_interval(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = Workspace(root / "workspace")
            workspace.initialize()
            job = root / "job"
            job.mkdir()
            commands: list[list[str]] = []

            def runner(command, **kwargs):
                commands.append(command)
                (job / "clip123.mkv").write_bytes(b"video")
                return subprocess.CompletedProcess(command, 0, "", "")

            path = YouTubeAcquirer(workspace, runner=runner)._download_media(
                "https://youtube.com/watch?v=clip123", "clip123", job,
                download_window=(98.0, 202.0),
            )

            self.assertEqual(path.name, "clip123.mkv")
            command = commands[0]
            self.assertEqual(command[command.index("--download-sections") + 1], "*98.000-202.000")
            self.assertIn("--force-keyframes-at-cuts", command)

    def test_remote_interval_retries_when_audio_ends_before_video(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = Workspace(root / "workspace")
            workspace.initialize()
            job = root / "job"
            job.mkdir()
            media = job / "clip123.mkv"
            attempts = [0]

            def download(*args, **kwargs):
                attempts[0] += 1
                media.write_bytes(f"attempt-{attempts[0]}".encode())
                return media

            incomplete = VideoProbe(
                media, 180, 1920, 1080, "h264", "yuv420p", "aac",
                audio_duration=85,
            )
            complete = VideoProbe(
                media, 180, 1920, 1080, "h264", "yuv420p", "aac",
                audio_duration=180,
            )
            acquirer = YouTubeAcquirer(workspace, runner=lambda *args, **kwargs: None)
            candidate = Candidate(
                "youtube-clip123", SourceType.YOUTUBE,
                "https://youtube.com/watch?v=clip123", "Clip",
                metadata={"video_id": "clip123"},
            )

            with patch.object(acquirer, "_download_media", side_effect=download), patch(
                "video_factory.youtube.probe_video", side_effect=[incomplete, complete],
            ):
                _, media_info, _, _ = acquirer.acquire_remote_media(
                    candidate, {"id": "clip123", "duration": 600},
                    candidate.source_url, job, source_range=SourceRange(100, 276),
                )

            self.assertEqual(attempts[0], 2)
            self.assertEqual(media_info.duration, 180)

    def test_remote_interval_retries_after_bounded_acquisition_timeout(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = Workspace(root / "workspace")
            workspace.initialize()
            job = root / "job"
            job.mkdir()
            media = job / "clip123.mkv"
            media.write_bytes(b"complete")
            acquirer = YouTubeAcquirer(workspace, runner=lambda *args, **kwargs: None)
            candidate = Candidate(
                "youtube-clip123", SourceType.YOUTUBE,
                "https://youtube.com/watch?v=clip123", "Clip",
                metadata={"video_id": "clip123"},
            )

            with patch.object(
                acquirer, "_download_media",
                side_effect=[YouTubeAcquisitionError("timed out"), media],
            ) as download, patch(
                "video_factory.youtube.probe_video",
                return_value=VideoProbe(
                    media, 120, 1920, 1080, "h264", "yuv420p", "aac",
                    audio_duration=120,
                ),
            ):
                acquirer.acquire_remote_media(
                    candidate, {"id": "clip123", "duration": 600},
                    candidate.source_url, job, source_range=SourceRange(100, 216),
                )

            self.assertEqual(download.call_count, 2)

    def test_probe_video_reads_matroska_audio_duration_tag(self) -> None:
        payload = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920,
                 "height": 1080, "pix_fmt": "yuv420p"},
                {"codec_type": "audio", "codec_name": "aac", "bit_rate": "192000",
                 "tags": {"DURATION": "00:01:24.946000000"}},
            ],
            "format": {"duration": "179.700000"},
        }
        completed = subprocess.CompletedProcess(
            ["ffprobe"], 0, json.dumps(payload), "",
        )

        with patch("video_factory.media.subprocess.run", return_value=completed):
            probe = probe_video(Path("bounded.mkv"))

        self.assertAlmostEqual(probe.audio_duration or 0, 84.946, places=3)

    def test_complete_source_download_omits_download_sections(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = Workspace(root / "workspace")
            workspace.initialize()
            job = root / "job"
            job.mkdir()
            commands: list[list[str]] = []

            def runner(command, **kwargs):
                commands.append(command)
                (job / "technical123.mkv").write_bytes(b"video")
                return subprocess.CompletedProcess(command, 0, "", "")

            YouTubeAcquirer(workspace, runner=runner)._download_media(
                "https://youtube.com/watch?v=technical123", "technical123", job,
            )

            self.assertNotIn("--download-sections", commands[0])

    def test_interview_rebase_keeps_original_provenance_and_stays_in_bounds(self) -> None:
        cues = [
            TranscriptCue("c1", 99, 108, "A concrete engineering claim.", "一个工程判断。"),
            TranscriptCue("c2", 108, 198, "The complete useful explanation.", "完整解释。"),
            TranscriptCue("c3", 198, 203, "The final consequence.", "最终影响。"),
        ]
        plan = {
            "story_start": 100, "story_end": 200,
            "wechat_lessons": [{"start": 100, "end": 200, "title": "工程判断"}],
        }

        provenance = rebase_interview_clip_timeline(cues, plan, {
            "original_start": 100, "original_end": 200,
            "download_start": 98, "download_end": 202,
        }, 104)

        self.assertEqual((plan["story_start"], plan["story_end"]), (2.0, 102.0))
        self.assertEqual(plan["wechat_lessons"][0]["original_start"], 100.0)
        self.assertTrue(all(0 <= cue.start < cue.end <= 104 for cue in cues))
        self.assertEqual((cues[0].original_start, cues[-1].original_end), (99, 203))
        self.assertTrue(provenance["rebased"])

        # A reviewed cached plan is already clip-local. Its original fields,
        # rather than its local seconds, drive the next bounded acquisition.
        cached_cues = [TranscriptCue(**asdict(cue)) for cue in cues]
        cached_plan = json.loads(json.dumps(plan))
        second = rebase_interview_clip_timeline(
            cached_cues, cached_plan, {
                "original_start": 100, "original_end": 200,
                "download_start": 98, "download_end": 202,
            }, 104, previous_clip=provenance,
        )
        self.assertEqual(
            [(cue.start, cue.end) for cue in cached_cues],
            [(cue.start, cue.end) for cue in cues],
        )
        self.assertEqual(second["original_start"], 100)

        full_source_cues = [TranscriptCue(**asdict(cue)) for cue in cues]
        full_source_plan = json.loads(json.dumps(plan))
        rebase_interview_clip_timeline(
            full_source_cues, full_source_plan, {
                "original_start": 100, "original_end": 200,
                "download_start": 0, "download_end": 600,
            }, 600, previous_clip=provenance,
        )
        self.assertEqual((full_source_plan["story_start"], full_source_plan["story_end"]), (100.0, 200.0))
        self.assertEqual((full_source_cues[0].start, full_source_cues[-1].end), (99.0, 203.0))

    def test_interview_rebase_drops_tiny_boundary_caption_overlap(self) -> None:
        cues = [
            TranscriptCue("inside", 190, 199, "Complete claim.", "完整观点。"),
            TranscriptCue("barely-after", 199.95, 205, "A new sentence.", "新句子。"),
        ]
        plan = {
            "story_start": 100, "story_end": 200,
            "wechat_lessons": [{"start": 100, "end": 200, "title": "工程判断"}],
        }

        rebase_interview_clip_timeline(cues, plan, {
            "original_start": 100, "original_end": 200,
            "download_start": 98, "download_end": 202,
        }, 104)

        self.assertEqual([cue.id for cue in cues], ["inside"])

    def test_cached_bounded_interview_media_is_reused_as_a_clip(self) -> None:
        cached = {
            "original_start": 176.8, "original_end": 272.24,
            "download_start": 174.8, "download_end": 274.24,
            "media_duration": 99.516, "rebased": True,
        }

        window, bounded = _local_interview_media_window(cached, 99.52, 4466.0)

        self.assertTrue(bounded)
        self.assertEqual(window["download_start"], 174.8)
        self.assertEqual(window["download_end"], 274.24)

        full_window, full_is_bounded = _local_interview_media_window(cached, 4466.0, 4466.0)
        self.assertFalse(full_is_bounded)
        self.assertEqual(full_window["download_start"], 0.0)

        planned = {
            "original_start": 690.0, "original_end": 870.0, "rebased": False,
        }
        copied_window, copied_is_bounded = _local_interview_media_window(
            planned, 187.42, 3191.0,
        )
        self.assertTrue(copied_is_bounded)
        self.assertAlmostEqual(copied_window["download_start"], 684.58)
        self.assertEqual(copied_window["download_end"], 872.0)

    def test_no_render_plan_with_local_cues_is_not_rebased_as_original_seconds(self) -> None:
        cached = {
            "original_start": 2044.32, "original_end": 2138.0,
            "rebased": False,
        }
        window = {
            "original_start": 2044.32, "original_end": 2138.0,
            "download_start": 2042.269, "download_end": 2140.0,
        }
        cues = [
            TranscriptCue("local-1", 2.0, 9.0, "first", "第一句。"),
            TranscriptCue("local-2", 85.0, 95.0, "last", "最后一句。"),
        ]
        plan = {
            "story_start": 2.0, "story_end": 95.0,
            "wechat_lessons": [{"start": 2.0, "end": 95.0, "title": "标题"}],
        }

        previous = _previous_clip_for_local_timeline(
            cues, cached, window, 97.731,
        )
        rebase_interview_clip_timeline(
            cues, plan, window, 97.731, previous_clip=previous,
        )

        self.assertEqual([cue.id for cue in cues], ["local-1", "local-2"])
        self.assertAlmostEqual(cues[0].start, 2.0)
        self.assertAlmostEqual(cues[-1].end, 95.0)
        self.assertAlmostEqual(cues[0].original_start, 2044.269)

    def test_interview_selection_precedes_partial_media_download(self) -> None:
        events: list[str] = []
        download_flags: list[bool] = []
        testcase = self
        metadata = {
            "id": "interview123", "title": "Known engineer interview",
            "channel": "AI Engineer", "description": "AI engineering interview",
            "duration": 600, "creators": ["Andrej Karpathy", "Host"],
        }
        cues = [
            TranscriptCue(f"c{index}", 100 + index * 7.5, 107.5 + index * 7.5,
                          "A concrete AI engineering system changes team workflow.", "工程系统改变团队工作流。")
            for index in range(12)
        ]
        plan = {
            "editorial_mode": "known_tech_interview_clip", "collection_title": "人物高光",
            "story_start": 100, "story_end": 190, "bilibili_chapters": [],
            "wechat_lessons": [{
                "speaker_label": "Andrej Karpathy", "title": "工程系统改变团队协作",
                "thesis": "一个完整且可验证的工程判断。", "start": 100, "end": 190,
                "framing": "speaker", "hook_headlines": [
                    "AI 工程真正卡在协作", "系统选择会改变团队", "这套方法减少返工成本",
                ],
            }],
        }

        class FakeAcquirer:
            def __init__(self, workspace):
                self.workspace = workspace

            def acquire(self, url, job, **kwargs):
                events.append("metadata_transcript")
                download_flags.append(bool(kwargs["download_media"]))
                candidate = Candidate(
                    "youtube-interview123", SourceType.YOUTUBE, url, metadata["title"],
                    author=metadata["channel"], metadata={"video_id": metadata["id"]},
                )
                return candidate, [], dict(metadata), list(cues), "", "subtitles.json3", None

            def acquire_remote_media(self, candidate, acquired_metadata, url, job, source_range=None, **kwargs):
                events.append("partial_download")
                testcase.assertIsNotNone(source_range)
                testcase.assertEqual((source_range.start, source_range.end), (100, 190))
                info = SourceMediaInfo(1920, 1080, 94, "h264", "aac")
                evidence = Evidence("video", candidate.id, url, "selected interval", "youtube:video")
                return "clip.mkv", info, evidence, {
                    "original_start": 100, "original_end": 190,
                    "download_start": 98, "download_end": 192,
                }

        def select(self, acquired_metadata, acquired_cues, editorial_mode):
            events.append("highlight_selected")
            acquired_cues[:] = [cue for cue in acquired_cues if cue.end > 100 and cue.start < 190]
            return [], dict(plan), []

        def audit(self, editorial_plan, acquired_cues, duration):
            events.append("directing_audited")
            return {"step": "interview_directing_audit", "provenance": {"provider": "test"}}

        with TemporaryDirectory() as temp, patch(
            "video_factory.youtube.YouTubeAcquirer", FakeAcquirer,
        ), patch.object(
            NaturalSubtitleTranslator, "translate", select,
        ), patch.object(
            NaturalSubtitleTranslator, "audit_interview_directing", audit,
        ), patch.object(
            YouTubeCollectionRenderer, "render", return_value=[],
        ):
            workspace = Workspace(Path(temp))
            workspace.initialize()
            (Path(temp) / "job").mkdir()
            result = YouTubeCollectionFactory(workspace, object()).generate(
                "https://youtube.com/watch?v=interview123", Path(temp) / "job",
                render=True, editorial_mode="known_tech_interview_clip",
            )
            generated = json.loads(Path(result["collection_manifest"]).read_text(encoding="utf-8"))

        self.assertEqual(events, [
            "metadata_transcript", "highlight_selected", "directing_audited", "partial_download",
        ])
        self.assertEqual(download_flags, [False])
        self.assertEqual(result["editorial_mode"], "known_tech_interview_clip")
        source_range = generated["items"][0]["source_ranges"][0]
        hook_range = generated["items"][0]["renders"][0]["selected_hook"]["source_range"]
        self.assertEqual((source_range["start"], source_range["end"]), (2.0, 92.0))
        self.assertEqual((source_range["original_start"], source_range["original_end"]), (100.0, 190.0))
        self.assertTrue(0 <= hook_range["start"] < hook_range["end"] <= 94)
        self.assertIsNotNone(hook_range["original_start"])
        self.assertTrue(all(0 <= cue["start"] < cue["end"] <= 94 for cue in generated["transcript"]))

    def test_technical_coverage_factory_requests_complete_source(self) -> None:
        requested_ranges: list[SourceRange | None] = []
        metadata = {
            "id": "technical123", "title": "AI engineering lecture", "channel": "AI Engineer",
            "description": "technical systems", "duration": 600, "creators": [],
        }
        cues = [
            TranscriptCue(f"c{index}", index * 8, min(600, index * 8 + 8),
                          "A concrete AI engineering system changes team workflow.", "工程系统改变团队工作流。")
            for index in range(75)
        ]
        plan = {
            "editorial_mode": "technical_coverage", "collection_title": "技术精讲",
            "story_start": 0, "story_end": 600, "bilibili_chapters": [],
            "wechat_lessons": [
                {"title": "系统设计第一部分", "thesis": "完整观点一。", "start": 0, "end": 300,
                 "framing": "speaker", "hook_headlines": ["系统瓶颈不在模型", "工具链决定团队速度", "验证流程减少返工"]},
                {"title": "系统设计第二部分", "thesis": "完整观点二。", "start": 300, "end": 600,
                 "framing": "speaker", "hook_headlines": ["扩展之后问题变了", "评估必须进入流程", "系统最终稳定交付"]},
            ],
        }

        class FakeAcquirer:
            def __init__(self, workspace):
                pass

            def acquire(self, url, job, **kwargs):
                candidate = Candidate("youtube-technical123", SourceType.YOUTUBE, url, metadata["title"],
                                      author=metadata["channel"], metadata={"video_id": metadata["id"]})
                return candidate, [], dict(metadata), list(cues), "", "subtitles.json3", None

            def acquire_remote_media(self, candidate, acquired_metadata, url, job, source_range=None, **kwargs):
                requested_ranges.append(source_range)
                info = SourceMediaInfo(1920, 1080, 600, "h264", "aac")
                evidence = Evidence("video", candidate.id, url, "complete source", "youtube:video")
                return "complete.mkv", info, evidence, None

        def select(self, acquired_metadata, acquired_cues, editorial_mode):
            return [], dict(plan), []

        with TemporaryDirectory() as temp, patch(
            "video_factory.youtube.YouTubeAcquirer", FakeAcquirer,
        ), patch.object(
            NaturalSubtitleTranslator, "translate", select,
        ), patch.object(
            YouTubeCollectionRenderer, "render", return_value=[],
        ):
            workspace = Workspace(Path(temp))
            workspace.initialize()
            (Path(temp) / "job").mkdir()
            YouTubeCollectionFactory(workspace, object()).generate(
                "https://youtube.com/watch?v=technical123", Path(temp) / "job",
                render=True, editorial_mode="technical_coverage",
            )

        self.assertEqual(requested_ranges, [None])

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

    def test_interview_hook_persists_with_causal_payoff_below_title(self) -> None:
        hook = HookSpec(
            "hook-1", HookStrategy.CONTRARIAN,
            "席位减少、Agent 用量上升，SaaS API 涨价是在自掘坟墓？",
            "厂商想靠 API 涨价补收入；客户只愿为结果付费，否则就把数据搬走。",
            SourceRange(10, 18), ["cue-1"], ["cue-1"],
            speaker_label="SaaStr联合创始人Jason", selected=True,
        )
        with TemporaryDirectory() as temp:
            concat, height = _write_hook_overlay_concat(
                hook, 58.3, Path(temp) / "interview.mp4",
                RenderProfile.WECHAT_VERTICAL,
            )
            content = concat.read_text(encoding="utf-8")
            from PIL import Image
            frame = Image.open(Path(temp) / "interview.hook-frames" / "hook.png")
            lower_alpha = frame.getchannel("A").crop((0, 285, 1080, 420))
            former_accent = frame.crop((55, 110, 70, 280))

        self.assertEqual(height, 430)
        self.assertIsNotNone(lower_alpha.getbbox())
        self.assertNotIn((255, 216, 77, 235), list(former_accent.get_flattened_data()))
        self.assertIn("duration 58.300000", content)
        self.assertIn("hook.png", content)
        self.assertNotIn("hook-compact.png", content)

    def test_interview_hook_uses_actual_pixel_width_for_long_mixed_title(self) -> None:
        from video_factory.compositor import resolve_font_path

        font, wrapped = _fit_text_by_pixels(
            "AI不是零和：开源、云、应用都能赢", resolve_font_path(), 924, 64, 48,
        )

        self.assertTrue(all(font.getlength(line) <= 924 for line in wrapped.splitlines()))
        self.assertGreaterEqual(font.getlength(wrapped.splitlines()[0]), 924 * 0.58)
        if len(wrapped.splitlines()) > 1:
            self.assertFalse(wrapped.splitlines()[1].startswith(("、", "，", "。", "；", "：")))
        self.assertEqual(
            "".join(wrapped.splitlines()).replace(" ", ""),
            "AI不是零和：开源、云、应用都能赢",
        )

    def test_interview_hook_preserves_prominent_two_line_incumbent_wrap(self) -> None:
        font, wrapped = _fit_interview_hook_headline(
            "AI 不是泡沫，但 2026 年必须靠收入说话",
            _resolve_headline_font_path(),
        )

        self.assertEqual(font.size, 64)
        self.assertEqual(len(wrapped.splitlines()), 2)
        self.assertEqual(
            "".join(wrapped.split()),
            "AI不是泡沫，但2026年必须靠收入说话",
        )

    def test_interview_hook_prefers_clause_boundary_over_split_chinese_phrase(self) -> None:
        from video_factory.compositor import resolve_font_path

        _, wrapped = _fit_text_by_pixels(
            "黄仁勋：AI竞赛不是模型竞赛，而是应用竞赛",
            resolve_font_path(), 924, 64, 48,
        )

        lines = wrapped.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].endswith("，"))
        self.assertFalse(lines[0].endswith("竞"))
        self.assertFalse(lines[1].startswith("赛"))

    def test_wechat_bilingual_subtitle_card_is_wide_and_left_aligned(self) -> None:
        from PIL import Image

        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.srt"
            translation = root / "translation.srt"
            source.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nOpen source wins, and inference clouds win.\n",
                encoding="utf-8",
            )
            translation.write_text(
                "1\n00:00:00,000 --> 00:00:02,000\n开源能赢，推理云也能赢。\n",
                encoding="utf-8",
            )
            _write_subtitle_overlay_concat(
                source, translation, RenderProfile.WECHAT_VERTICAL, 2.0, root / "clip.mp4",
            )
            frame = Image.open(root / "clip.subtitle-frames" / "subtitle-0001.png")
            alpha_bbox = frame.getchannel("A").getbbox()

        self.assertIsNotNone(alpha_bbox)
        self.assertLessEqual(alpha_bbox[0], 70)
        self.assertGreaterEqual(alpha_bbox[2], 1010)

    def test_long_bilingual_subtitles_reflow_by_actual_pixel_width(self) -> None:
        from video_factory.compositor import resolve_font_path

        english = (
            "Yeah. And it's like the pie is growing so fast but what's happening is "
            "there's an interesting duality that is not even just like rising tide."
        )
        chinese = "对。就像这块饼涨得太快了，但实际发生的是，这里有个有趣的双重性。它甚至不只是水涨船高。"

        english_font, wrapped_english = _fit_subtitle_by_pixels(
            english, resolve_font_path(), 880, 28, 20, 3,
        )
        chinese_font, wrapped_chinese = _fit_subtitle_by_pixels(
            chinese, resolve_font_path(), 880, 44, 34, 3,
        )

        self.assertLessEqual(len(wrapped_english.splitlines()), 3)
        self.assertLessEqual(len(wrapped_chinese.splitlines()), 3)
        self.assertTrue(all(english_font.getlength(line) <= 880 for line in wrapped_english.splitlines()))
        self.assertTrue(all(chinese_font.getlength(line) <= 880 for line in wrapped_chinese.splitlines()))
        self.assertEqual(" ".join(wrapped_english.split()), english)
        self.assertEqual("".join(wrapped_chinese.split()), "".join(chinese.split()))

    def test_subtitle_pixel_fit_reserves_width_for_visible_stroke(self) -> None:
        from PIL import Image, ImageDraw
        from video_factory.compositor import resolve_font_path

        chinese = "代码的价值几乎完全由最终生成的文本量决定，开发者还能自己排查失败。"
        font, wrapped = _fit_subtitle_by_pixels(
            chinese, resolve_font_path(), 880, 44, 34, 3, stroke_width=2,
        )
        draw = ImageDraw.Draw(Image.new("RGBA", (1080, 330), (0, 0, 0, 0)))
        bbox = draw.multiline_textbbox(
            (0, 0), wrapped, font=font, spacing=8, stroke_width=2,
        )

        self.assertLessEqual(bbox[2] - bbox[0], 880)
        self.assertEqual("".join(wrapped.split()), chinese)

    def test_subtitle_closing_punctuation_never_starts_its_own_line(self) -> None:
        from video_factory.compositor import resolve_font_path

        font, wrapped = _fit_subtitle_by_pixels(
            "但基础设施建成后，问题就变成美国其他各个层面该怎么办。这个行业不只是关于模型。",
            resolve_font_path(), 880, 44, 30, 4, stroke_width=2,
        )

        self.assertTrue(all(
            not line.startswith(("，", "。", "；", "：", "！", "？"))
            for line in wrapped.splitlines()
        ))
        self.assertTrue(all(font.getlength(line) <= 876 for line in wrapped.splitlines()))

    def test_subtitle_uses_bounded_emergency_font_before_overflow(self) -> None:
        from video_factory.compositor import resolve_font_path

        # A long indivisible token can be only a few pixels wider than the
        # safe area at the normal minimum size. It must shrink slightly rather
        # than overflow the card or be clipped by the frame.
        font, wrapped = _fit_subtitle_by_pixels(
            "W" * 31, resolve_font_path(), 880, 44, 30, 4, stroke_width=2,
        )

        self.assertGreaterEqual(font.size, 22)
        self.assertLess(font.size, 30)
        self.assertTrue(all(font.getlength(line) <= 876 for line in wrapped.splitlines()))

    def test_renderer_never_invents_alignment_by_splitting_two_languages(self) -> None:
        english = (
            "The model layer captures all the value, while product companies "
            "cannot build sustainable margins from applications."
        )
        chinese = "模型层拿走了所有价值，产品公司就很难靠应用建立可持续利润。"

        parts = split_bilingual_subtitle_display(english, chinese, 6.2)

        self.assertEqual(parts, [(english, chinese)])

    def test_interview_caption_does_not_slice_one_source_thought_into_cards(self) -> None:
        english = (
            "But once you get the infrastructure built, the question is what about "
            "all of the other layers across the United States?"
        )
        chinese = "可一旦把 infrastructure 建好，问题就来了：其他那些层怎么办？"

        parts = split_bilingual_subtitle_display(english, chinese, 7.7)

        self.assertEqual(len(parts), 1)
        self.assertIn("infrastructure built", parts[0][0])
        self.assertIn("其他那些层怎么办", parts[0][1])

    def test_reviewed_semantic_card_is_not_partitioned_again_by_renderer(self) -> None:
        english = (
            "so that when we're ready to deploy compute that they'll be ready "
            "for us land power shell"
        )
        chinese = "我们部署算力时，土地、电力和厂房才能准备就绪。"

        parts = split_bilingual_subtitle_display(
            english, chinese, 5.6, semantic_locked=True,
        )

        self.assertEqual(parts, [(english, chinese)])

    def test_clause_split_uses_dependent_boundaries_to_avoid_overlong_cards(self) -> None:
        ecosystem = (
            "And so I look across the entire ecosystem and look for bottlenecks "
            "and if there are places where extraordinary companies are being built"
        )
        causal = (
            "I think about the long-term supply chain more than most because our "
            "company is really large and in order for us to succeed many companies support me."
        )

        ecosystem_parts = _semantic_english_parts(ecosystem, 3)
        causal_parts = _semantic_english_parts(causal, 3)

        self.assertEqual(" ".join(ecosystem_parts), ecosystem)
        self.assertEqual(" ".join(causal_parts), causal)
        self.assertEqual(len(ecosystem_parts), 3)
        self.assertEqual(len(causal_parts), 3)
        self.assertTrue(any(part.casefold().startswith("because ") for part in causal_parts))

    def test_noisy_asr_condition_keeps_bottleneck_predicate_in_same_card(self) -> None:
        source = (
            "And so I look across the entire ecosystem and look for bottlenecks "
            "and if there are places where extraordinary companies are being built "
            "constraints constraints extraordinary companies being built uh maybe "
            "it's uh uh uh supply chain that has to uh get scaled up so that when "
            "we're ready to deploy compute that they'll be ready for us land power "
            "shell and so this is no different than looking at the supply chain upstream."
        )

        parts = _semantic_english_parts(source, 5)

        self.assertEqual(" ".join(parts), source)
        self.assertEqual(parts[0], "And so I look across the entire ecosystem")
        self.assertTrue(parts[1].startswith("and look for bottlenecks and if "))
        self.assertFalse(any(part.startswith("and if ") for part in parts))

    def test_corrupt_compet_cycle_fragment_is_not_stranded_before_its_payoff(self) -> None:
        source = (
            "The compet cycle tends to be though that the earnings over time over "
            "long stretches of time tends to move up the stack right towards the "
            "application layer where you can over earn for larger periods of time."
        )

        parts = _semantic_english_parts(source, 3)

        self.assertEqual(" ".join(parts), source)
        self.assertFalse(any(part.rstrip().endswith("though") for part in parts[:-1]))
        self.assertIn("application layer", parts[0])
        self.assertTrue(parts[-1].startswith("where you can over earn"))

    def test_semantic_split_does_not_create_short_asr_or_idiom_fragments(self) -> None:
        coding = (
            "Um so in coding essenti and this is back to the kind of utility point "
            "like the utility of code is represented by the text you can generate"
        )
        legal = (
            "And it's like okay well lo and behold legal is interesting because "
            "reviewing and writing legal documents creates a lot of value"
        )

        coding_parts = _semantic_english_parts(coding, 4)
        legal_parts = _semantic_english_parts(legal, 4)

        self.assertFalse(any(part.endswith("coding essenti") for part in coding_parts))
        self.assertFalse(any(part.endswith("well lo") for part in legal_parts))
        self.assertTrue(any("lo and behold" in part for part in legal_parts))

    def test_business_model_uses_contextual_chinese_term(self) -> None:
        cues = [TranscriptCue(
            "cue", 0, 4,
            "As CEO, what is the right business model?",
            "作为 CEO，什么才是正确的商业模式？",
        )]

        errors = terminology_contract_errors(cues, [TerminologyEntry(
            "model", TerminologyStrategy.TRANSLATE, target="模型",
        )])

        self.assertEqual(errors, [])

    def test_ai_slop_has_explicit_chinese_terminology_contract(self) -> None:
        cues = [TranscriptCue(
            "cue", 0, 4,
            "The utility point about AI slop is different.",
            "AI 低质内容的效用逻辑不同。",
        )]

        terminology = NaturalSubtitleTranslator._parse_terminology([], cues)

        slop = next(item for item in terminology if item.source == "slop")
        self.assertEqual(slop.target, "低质内容")
        self.assertEqual(terminology_contract_errors(cues, terminology), [])

    def test_asr_topic_setup_becomes_complete_card_without_borrowing_next_claim(self) -> None:
        class Translator:
            def _request_json(self, messages, max_tokens):
                prompt = messages[-1]["content"]
                self.assert_topic_rule(prompt)
                rows = json.loads(prompt.split("Cards: ", 1)[1].split(
                    "\nPrevious validation error:", 1,
                )[0])
                copy = [
                    "说回低质内容的效用问题，编程领域情况特殊。",
                    "代码效用几乎完全取决于可生成的文本量。",
                    "这些文本本身凝结了巨大价值。",
                ]
                return ({"translations": [
                    {"id": row["id"], "text": copy[index]}
                    for index, row in enumerate(rows)
                ]}, {"model": "topic-translator"})

            @staticmethod
            def assert_topic_rule(prompt):
                if "complete topic-setting Chinese sentence" not in prompt:
                    raise AssertionError(prompt)

        class Reviewer:
            def _request_json(self, messages, max_tokens):
                prompt = messages[-1]["content"]
                if "complete topic-setting Chinese sentence" not in prompt:
                    raise AssertionError(prompt)
                rows = json.loads(prompt.split("Rows: ", 1)[1])
                return ({"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5,
                    "errors": [],
                } for row in rows]}, {"model": "topic-reviewer"})

        source = (
            "um so in coding essentially and this is back to the kind of utility point "
            "on on you know slop like the utility of code is almost 100 represented by "
            "the amount of text that you can generate like like all obviously an insane "
            "amount of value went into the text"
        )
        cues = [TranscriptCue(
            "cue-1-card-1", 0, 16.414, source,
            "旧字幕。",
        )]
        terminology = [TerminologyEntry(
            "slop", TerminologyStrategy.TRANSLATE, target="低质内容",
        )]

        trace = NaturalSubtitleTranslator(
            Translator(), None, Reviewer(),
        ).segment_interview_subtitle_cards(cues, terminology)

        self.assertFalse(trace["fallback_used"])
        self.assertEqual(len(cues), 3)
        self.assertIn("低质内容", cues[0].translation)
        self.assertTrue(all(not cue.translation.endswith("——") for cue in cues))

    def test_open_source_check_accepts_natural_contextual_chinese(self) -> None:
        cues = [TranscriptCue(
            "cue", 0, 5,
            "There was an open-source check on closed source.",
            "这体现了开源对闭源的制衡。",
        )]

        errors = terminology_contract_errors(cues, [TerminologyEntry(
            "open-source check", TerminologyStrategy.TRANSLATE, target="开源制衡",
        )])

        self.assertEqual(errors, [])

    def test_clause_split_may_create_so_that_card_for_translation_to_localize(self) -> None:
        source = (
            "so we started working with all of these companies long before the growth came "
            "so that the growth could happen."
        )

        parts = _semantic_english_parts(source, 4)

        self.assertTrue(any(part.casefold().startswith("so that ") for part in parts[1:]))
        self.assertEqual(" ".join(parts), source)

    def test_semantic_split_never_leaves_nonfinal_card_on_connector(self) -> None:
        source = (
            "Corning has to support us and Lumentum and TSMC and memory companies "
            "and and so we started working with them before demand arrived."
        )

        parts = _semantic_english_parts(source, 5)

        self.assertEqual(" ".join(parts), source)
        self.assertTrue(all(
            not part.casefold().rstrip(" ,.;").endswith((" and", " so", " uh", " um"))
            for part in parts[:-1]
        ))

    def test_satya_legacy_asr_keeps_to_your_point_with_its_object(self) -> None:
        source = (
            "because otherwise we'll be back to some mainframe lock-in that's not a thing "
            "to your point about if anything given that we will hopefully continue "
            "to have a richer choice in every layer."
        )

        parts = _semantic_english_parts(source, 4)

        self.assertEqual(" ".join(parts), source)
        self.assertFalse(any(part.rstrip(" ,.;").endswith("about") for part in parts))
        self.assertTrue(any(
            "to your point about if anything given that" in part for part in parts
        ))

    def test_tiny_supplier_fragment_coalesces_before_translation(self) -> None:
        parts = [
            "Corning and Lumentum have to support our very large long-term infrastructure expansion plans",
            "and TSMC of course",
            "and memory companies",
            "and so we started working early",
        ]

        merged = _coalesce_short_semantic_parts(parts, 9)

        self.assertEqual(len(merged), 3)
        self.assertIn("TSMC", merged[1])
        self.assertIn("memory companies", merged[1])

    def test_caption_timing_gate_rejects_avoidable_seventeen_second_card(self) -> None:
        cues = [
            TranscriptCue(
                "good-card", 0, 7.24,
                "But once infrastructure is built, what happens to the other layers?",
                "基础设施建成后，其他产业环节怎么办？",
            ),
            TranscriptCue(
                "bad-card", 7.24, 24.19,
                "I look for bottlenecks and maybe the supply chain must scale so that resources are ready.",
                "我会寻找瓶颈，供应链也许必须扩容，确保资源及时就绪。",
            ),
        ]

        errors = interview_caption_duration_errors(cues)

        self.assertTrue(any(
            "bad-card lasts 16.95s" in error for error in errors
        ), errors)

    def test_caption_policy_has_no_unsplittable_hard_limit_exemption(self) -> None:
        cue = TranscriptCue(
            "one-clause", 0, 7.51,
            "Infrastructure economics remain difficult.",
            "基础设施经济性仍然很难。",
        )

        errors = interview_caption_duration_errors([cue])

        self.assertTrue(any("hard maximum" in error for error in errors), errors)

    def test_caption_policy_rejects_subsecond_and_dense_cards(self) -> None:
        cue = TranscriptCue(
            "dense", 0, 0.9,
            "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four twenty-five twenty-six twenty-seven twenty-eight twenty-nine.",
            "这是一条明显过于密集而且来不及阅读的中文字幕，任何观众都无法在不到一秒内读完。",
        )

        errors = interview_caption_duration_errors([cue])

        self.assertTrue(any("minimum" in error for error in errors), errors)
        self.assertTrue(any("English words" in error for error in errors), errors)
        self.assertTrue(any("Chinese characters" in error for error in errors), errors)

    def test_cached_semantic_cards_are_immutable_during_rerender(self) -> None:
        cues = [TranscriptCue(
            "cue-1-card-1", 0, 4.5, "The application layer matters.", "应用层很重要。",
        )]
        trace = [{
            "step": "interview_semantic_subtitle_cards",
            "cue_ids": [cues[0].id],
            "reviewed_cue_ids": [cues[0].id],
            "policy_version": INTERVIEW_CAPTION_POLICY_VERSION,
            "policy_fingerprint": INTERVIEW_CAPTION_POLICY_FINGERPRINT,
        }]

        self.assertTrue(cached_interview_caption_pipeline_complete(cues, trace))
        self.assertFalse(cached_interview_caption_pipeline_complete(cues, []))

        stale = [{**trace[0], "policy_version": "legacy"}]
        self.assertFalse(cached_interview_caption_pipeline_complete(cues, stale))

        mixed = [*cues, TranscriptCue(
            "cue-2", 4.5, 14.5,
            "This stale parent remains much too long to publish safely.",
            "这个过期的父级字幕仍然太长，不能安全发布。",
        )]
        self.assertFalse(cached_interview_caption_pipeline_complete(mixed, trace))

    def test_stale_semantic_siblings_are_retimed_without_realigning_text(self) -> None:
        cues = [
            TranscriptCue(
                "cue-1-card-1", 0, 7.4,
                "The developer creates value at the computer.",
                "开发者在电脑前创造价值。",
                original_start=100, original_end=107.4,
            ),
            TranscriptCue(
                "cue-1-card-2", 7.4, 8.2,
                "and types code.", "并敲下代码。",
                original_start=107.4, original_end=108.2,
            ),
        ]
        bilingual_before = [
            (cue.source_text, cue.translation) for cue in cues
        ]

        trace = NaturalSubtitleTranslator(object()).segment_interview_subtitle_cards(
            cues, [],
        )

        self.assertEqual(
            [(cue.source_text, cue.translation) for cue in cues], bilingual_before,
        )
        self.assertAlmostEqual(cues[0].start, 0)
        self.assertAlmostEqual(cues[-1].end, 8.2)
        self.assertAlmostEqual(cues[0].original_start, 100)
        self.assertAlmostEqual(cues[-1].original_end, 108.2)
        self.assertAlmostEqual(cues[0].original_end, cues[1].original_start)
        self.assertLessEqual(cues[0].original_end - cues[0].original_start, 7.5)
        self.assertTrue(all(cue.duration >= 1.2 for cue in cues))
        self.assertEqual(trace["policy_version"], INTERVIEW_CAPTION_POLICY_VERSION)

    def test_hard_compliant_stale_card_is_not_resplit_to_chase_target(self) -> None:
        class MustNotTranslate:
            def _request_json(self, *args, **kwargs):
                raise AssertionError("hard-compliant reviewed card must remain immutable")

        cue = TranscriptCue(
            "cue-1-card-2", 0, 5.6,
            "so that when we're ready to deploy compute that they'll be ready for us land power shell",
            "我们部署算力时，土地、电力和厂房才能准备就绪。",
        )
        before = asdict(cue)

        trace = NaturalSubtitleTranslator(
            MustNotTranslate(),
        ).segment_interview_subtitle_cards([cue], [])

        self.assertEqual(asdict(cue), before)
        self.assertEqual(trace["cue_ids"], [])

    def test_semantic_cards_retry_when_uppercase_entity_moves_between_rows(self) -> None:
        class GeminiCritic:
            def __init__(self):
                self.calls = 0

            def _request_json(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    rows = [
                        {"id": "cue-1-card-1", "text": "Corning 和台积电都支持我们"},
                        {"id": "cue-1-card-2", "text": "这家公司也支持我们"},
                    ]
                else:
                    rows = [
                        {"id": "cue-1-card-1", "text": "Corning 支持我们"},
                        {"id": "cue-1-card-2", "text": "台积电也支持我们"},
                    ]
                return ({"translations": rows}, {"model": "gemini-review"})

        critic = GeminiCritic()
        cues = [TranscriptCue(
            "cue-1", 0, 8,
            "Corning strongly supports our work and TSMC also strongly supports our work.",
            "Corning 和 TSMC 都支持我们。",
        )]

        trace = NaturalSubtitleTranslator(
            object(), critic,
        ).segment_interview_subtitle_cards(cues, [])

        self.assertFalse(trace["fallback_used"])
        self.assertEqual(critic.calls, 2)
        self.assertEqual([cue.translation for cue in cues], [
            "Corning 支持我们。", "台积电也支持我们。",
        ])

    def test_semantic_fallback_rejects_entity_shifted_chinese_partition(self) -> None:
        class InvalidCritic:
            def _request_json(self, *args, **kwargs):
                return ({"translations": []}, {"model": "unavailable-copy"})

        source = (
            "Corning strongly supports our work and TSMC also strongly supports our work."
        )
        cues = [TranscriptCue(
            "cue-1", 0, 8, source,
            "Corning 和台积电都支持我们，所以合作才能继续。",
        )]

        trace = NaturalSubtitleTranslator(
            object(), InvalidCritic(),
        ).segment_interview_subtitle_cards(cues, [])

        self.assertTrue(trace["fallback_used"])
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].id, "cue-1")

    def test_failed_batch_recovers_each_semantic_card_in_isolation(self) -> None:
        class ParagraphProneCritic:
            def _request_json(self, messages, max_tokens):
                prompt = messages[-1]["content"]
                if "Translate exactly one fixed English interview caption" not in prompt:
                    return ({"translations": []}, {"mode": "bad-batch"})
                if "Source: Corning" in prompt:
                    text = "Corning 支持我们。"
                elif "Source: and TSMC" in prompt:
                    text = "台积电也支持我们。"
                else:
                    raise AssertionError(prompt)
                return ({"text": text}, {"mode": "isolated"})

        cues = [TranscriptCue(
            "cue-1", 0, 8,
            "Corning strongly supports our work and TSMC also strongly supports our work.",
            "Corning 和台积电都支持我们。",
        )]

        trace = NaturalSubtitleTranslator(
            object(), ParagraphProneCritic(),
        ).segment_interview_subtitle_cards(cues, [])

        self.assertFalse(trace["fallback_used"])
        self.assertEqual(trace["provenance"]["mode"], "isolated_card_recovery")
        self.assertEqual([cue.translation for cue in cues], [
            "Corning 支持我们。", "台积电也支持我们。",
        ])

    def test_semantic_cards_retry_translation_that_exceeds_its_time_budget(self) -> None:
        class GeminiCritic:
            def __init__(self):
                self.calls = 0

            def _request_json(self, *args, **kwargs):
                self.calls += 1
                first = (
                    "我会非常仔细并且全面系统地审视整个生态系统中的全部情况和每一个潜在问题"
                    if self.calls == 1 else "我会审视生态。"
                )
                return ({"translations": [
                    {"id": "cue-1-card-1", "text": first},
                    {"id": "cue-1-card-2", "text": "再逐一检查所有基础设施瓶颈"},
                ]}, {"model": "gemini-review"})

        critic = GeminiCritic()
        cues = [TranscriptCue(
            "cue-1", 0, 8,
            "I scan the ecosystem and then I carefully inspect every infrastructure bottleneck across the system.",
            "我会审视整个生态，再逐一检查所有基础设施瓶颈。",
        )]
        terms = [TerminologyEntry(
            "infrastructure", TerminologyStrategy.TRANSLATE, target="基础设施",
        )]

        trace = NaturalSubtitleTranslator(
            object(), critic,
        ).segment_interview_subtitle_cards(cues, terms)

        self.assertFalse(trace["fallback_used"])
        self.assertEqual(critic.calls, 2)
        self.assertEqual(cues[0].translation, "我会审视生态。")

    def test_unsplit_interview_parent_fails_closed_before_renderer(self) -> None:
        cue = TranscriptCue(
            "cue-parent", 0, 20,
            "I inspect the ecosystem and identify bottlenecks and find where suppliers must scale.",
            "我会检查整个生态系统，找出瓶颈，并确定哪些供应商必须扩容。",
        )
        item = CollectionItem(
            "short", CollectionItemKind.WECHAT_SHORT, 1, "title", "thesis",
            [SourceRange(0, 20)],
        )
        render = PlatformRender(RenderProfile.WECHAT_VERTICAL, 1080, 1920)
        manifest = VideoCollectionManifest(
            "collection", "candidate", "https://youtube.com/watch?v=test", "test",
            "source", "channel", "title", [cue], [], [item],
            editorial_mode="known_tech_interview_clip",
        )

        with TemporaryDirectory() as temp, self.assertRaisesRegex(
            ValueError, "regenerate the cached translation plan",
        ):
            write_item_subtitle_files(
                manifest, item, render, Path(temp) / "locked",
            )

    def test_long_relative_payoff_uses_three_publishable_cards(self) -> None:
        source = (
            "The competitive cycle tends to be that earnings over long stretches "
            "move up the stack towards the application layer where you can over earn "
            "for larger periods of time."
        )

        parts = _semantic_english_parts(source, 3)

        self.assertEqual(len(parts), 3)
        self.assertIn("competitive cycle", parts[0])
        self.assertIn("application layer", parts[1])
        self.assertTrue(parts[2].startswith("where you can over earn"))

    def test_gemini_semantic_cards_are_shorter_and_losslessly_timed(self) -> None:
        class GeminiCritic:
            def _request_json(self, *args, **kwargs):
                return ({"translations": [
                    {"id": "cue-1-card-1", "text": "供应链需要提前扩容。"},
                    {"id": "cue-1-card-2", "text": "部署算力时，土地、电力和厂房都要就绪。"},
                    {"id": "cue-1-card-3", "text": "这与考察上游供应链的逻辑一致。"},
                ]}, {"model": "gemini-review"})

        source = (
            "it's the supply chain that has to get scaled up so that when we're ready "
            "to deploy compute they'll be ready for us land power shell and so this is "
            "no different than looking at the supply chain upstream."
        )
        cues = [TranscriptCue(
            "cue-1", 0, 14.2, source,
            "供应链需要扩大规模，部署计算前要准备好土地、电力和机房。这和审视上游供应链一样。",
        )]
        translator = NaturalSubtitleTranslator(object(), GeminiCritic())

        trace = translator.segment_interview_subtitle_cards(cues, [])

        self.assertEqual(trace["step"], "interview_semantic_subtitle_cards")
        self.assertFalse(trace["fallback_used"])
        self.assertEqual(len(cues), 3)
        self.assertEqual(" ".join(cue.source_text for cue in cues), source)
        self.assertAlmostEqual(cues[0].start, 0)
        self.assertAlmostEqual(cues[-1].end, 14.2)
        self.assertTrue(all(
            cue.duration <= INTERVIEW_CAPTION_HARD_MAX_SECONDS for cue in cues
        ))
        self.assertTrue(all(
            len(cue.source_text.split()) <= INTERVIEW_CAPTION_MAX_ENGLISH_WORDS
            for cue in cues
        ))
        self.assertTrue(all(
            len(re.sub(r"\s+", "", cue.translation))
            <= INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS
            for cue in cues
        ))

    def test_semantic_card_review_retries_dependent_translated_chinese(self) -> None:
        class ChineseTranslator:
            def __init__(self):
                self.calls = 0

            def _request_json(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    rows = [
                        {"id": "cue-1-card-1", "text": "这可能是需要扩大规模的芯片供应链"},
                        {"id": "cue-1-card-2", "text": "以便应用可以上线"},
                    ]
                else:
                    rows = [
                        {"id": "cue-1-card-1", "text": "芯片供应链需要提前扩容"},
                        {"id": "cue-1-card-2", "text": "扩容完成后，应用才能上线"},
                    ]
                return ({"translations": rows}, {"model": "chinese-translator"})

        class FidelityReviewer:
            def _request_json(self, messages, max_tokens):
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                reviews = []
                for row in rows:
                    text = row["chinese"]
                    reviews.append({
                        "id": row["id"],
                        "pass": not text.startswith("这可能是") and not text.startswith("以便"),
                        "fidelity_score": 5,
                        "naturalness_score": (
                            5 if not text.startswith(("这可能是", "以便")) else 2
                        ),
                        "errors": [] if not text.startswith(("这可能是", "以便")) else ["not standalone"],
                    })
                return ({"reviews": reviews}, {"model": "fidelity-reviewer"})

        translator = ChineseTranslator()
        cues = [TranscriptCue(
            "cue-1", 0, 10,
            "the chips supply chain has to scale up and then applications can launch.",
            "芯片供应链要扩容，应用才能上线。",
        )]
        terms = [
            TerminologyEntry("chips", TerminologyStrategy.TRANSLATE, target="芯片"),
            TerminologyEntry("applications", TerminologyStrategy.TRANSLATE, target="应用"),
        ]

        trace = NaturalSubtitleTranslator(
            translator, FidelityReviewer(),
        ).segment_interview_subtitle_cards(cues, terms)

        self.assertEqual(translator.calls, 2)
        self.assertFalse(trace["fallback_used"])
        self.assertEqual([cue.translation for cue in cues], [
            "芯片供应链需要提前扩容。", "扩容完成后，应用才能上线。",
        ])

    def test_semantic_fallback_keeps_parent_when_clause_terms_are_misaligned(self) -> None:
        class InvalidCritic:
            def _request_json(self, *args, **kwargs):
                return ({"translations": []}, {"model": "unavailable-copy"})

        source = (
            "infrastructure capacity must scale and supply chain capacity must grow."
        )
        cues = [TranscriptCue(
            "cue-1", 0, 12, source,
            "供应链必须扩容，基础设施容量也必须增长。",
        )]
        terms = [
            TerminologyEntry(
                "infrastructure", TerminologyStrategy.TRANSLATE, target="基础设施",
            ),
            TerminologyEntry(
                "supply chain", TerminologyStrategy.TRANSLATE, target="供应链",
            ),
        ]

        trace = NaturalSubtitleTranslator(
            object(), InvalidCritic(),
        ).segment_interview_subtitle_cards(cues, terms)

        self.assertTrue(trace["fallback_used"])
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].id, "cue-1")
        self.assertEqual(cues[0].source_text, source)

    def test_independent_fidelity_review_corrects_temporal_meaning(self) -> None:
        class ChineseTranslator:
            def __init__(self):
                self.calls = 0
                self.requested_ids = []

            def _request_json(self, messages, max_tokens):
                self.calls += 1
                second = (
                    "我们需要随时部署算力"
                    if self.calls == 1 else "准备部署算力时，配套资源必须就绪"
                )
                prompt = messages[-1]["content"]
                requested, _ = json.JSONDecoder().raw_decode(prompt.split("Cards: ", 1)[1])
                requested_ids = {row["id"] for row in requested}
                self.requested_ids.append(requested_ids)
                rows = [
                    {"id": "cue-1-card-1", "text": "供应链需要提前扩容"},
                    {"id": "cue-1-card-2", "text": second},
                ]
                return ({"translations": [
                    row for row in rows if row["id"] in requested_ids
                ]}, {"model": "chinese-translator"})

        class FidelityReviewer:
            def _request_json(self, messages, max_tokens):
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                return ({"reviews": [{
                    "id": row["id"],
                    "pass": "随时" not in row["chinese"],
                    "fidelity_score": 2 if "随时" in row["chinese"] else 5,
                    "naturalness_score": 5,
                    "errors": ["temporal meaning changed"] if "随时" in row["chinese"] else [],
                } for row in rows]}, {"model": "native-fidelity"})

        class DirectingMustNotReview:
            def _request_json(self, *args, **kwargs):
                raise AssertionError("directing critic must not audit subtitle copy")

        cues = [TranscriptCue(
            "cue-1", 0, 10,
            "the supply chain has to scale up so when we're ready to deploy compute resources are ready.",
            "供应链要扩容，部署算力时配套资源要就绪。",
        )]

        translator = ChineseTranslator()
        trace = NaturalSubtitleTranslator(
            translator, DirectingMustNotReview(), FidelityReviewer(),
        ).segment_interview_subtitle_cards(cues, [])

        self.assertEqual(cues[1].translation, "准备部署算力时，配套资源必须就绪。")
        self.assertNotIn("随时", cues[1].translation)
        self.assertEqual(translator.requested_ids, [
            {"cue-1-card-1", "cue-1-card-2"}, {"cue-1-card-2"},
        ])
        self.assertEqual(
            trace["fidelity_review_provenance"]["model"], "native-fidelity",
        )

    def test_reviewer_cannot_approve_merely_understandable_chinese_at_score_three(self) -> None:
        class ChineseTranslator:
            def __init__(self):
                self.calls = 0

            def _request_json(self, messages, max_tokens):
                self.calls += 1
                first = "利润沿技术栈向上移动" if self.calls == 1 else "利润会向应用层转移"
                prompt = messages[-1]["content"]
                requested, _ = json.JSONDecoder().raw_decode(prompt.split("Cards: ", 1)[1])
                requested_ids = {row["id"] for row in requested}
                rows = [
                    {"id": "cue-1-card-1", "text": first},
                    {"id": "cue-1-card-2", "text": "企业能长期获得超额利润"},
                ]
                return ({"translations": [
                    row for row in rows if row["id"] in requested_ids
                ]}, {"model": "chinese-translator"})

        class StrictReviewer:
            def _request_json(self, messages, max_tokens):
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                return ({"reviews": [{
                    "id": row["id"], "pass": True, "fidelity_score": 5,
                    "naturalness_score": 3 if "技术栈" in row["chinese"] else 5,
                    "errors": ["literal metaphor"] if "技术栈" in row["chinese"] else [],
                } for row in rows]}, {"model": "strict-reviewer"})

        translator = ChineseTranslator()
        cues = [TranscriptCue(
            "cue-1", 0, 10,
            "earnings move up the stack toward applications and companies can over earn for longer.",
            "利润向应用层转移，企业能长期获得超额利润。",
        )]

        NaturalSubtitleTranslator(
            translator, StrictReviewer(),
        ).segment_interview_subtitle_cards(cues, [])

        self.assertEqual(translator.calls, 2)
        self.assertEqual(cues[0].translation, "利润会向应用层转移。")

    def test_fidelity_reviewer_cannot_turn_maybe_into_frequency(self) -> None:
        class ChineseTranslator:
            def _request_json(self, *args, **kwargs):
                return ({"translations": [
                    {"id": "cue-1-card-1", "text": "我会寻找产业瓶颈"},
                    {"id": "cue-1-card-2", "text": "供应链可能需要扩容"},
                ]}, {"model": "chinese-translator"})

        class RegressingReviewer:
            def _request_json(self, messages, max_tokens):
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                return ({"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5,
                    "errors": [],
                    # Even if an auditor suggests replacement copy, the
                    # verdict-only contract must ignore it.
                    "text": (
                        "供应链往往需要扩容"
                        if row["id"] == "cue-1-card-2" else row["chinese"]
                    ),
                } for row in rows]}, {"model": "native-fidelity"})

        cues = [TranscriptCue(
            "cue-1", 0, 10,
            "I look for bottlenecks and maybe the supply chain needs to scale up.",
            "我会寻找瓶颈，供应链可能需要扩容。",
        )]

        NaturalSubtitleTranslator(
            ChineseTranslator(), None, RegressingReviewer(),
        ).segment_interview_subtitle_cards(cues, [])

        self.assertIn("可能", cues[1].translation)
        self.assertNotIn("往往", cues[1].translation)

    def test_configured_fidelity_reviewer_failure_never_publishes_unreviewed_cards(self) -> None:
        class ChineseTranslator:
            def __init__(self):
                self.calls = 0

            def _request_json(self, *args, **kwargs):
                self.calls += 1
                return ({"translations": [
                    {"id": "cue-1-card-1", "text": "我会寻找产业瓶颈"},
                    {"id": "cue-1-card-2", "text": "供应链可能需要扩容"},
                ]}, {"model": "chinese-translator"})

        class UnavailableReviewer:
            def _request_json(self, *args, **kwargs):
                raise RuntimeError("review service unavailable")

        cues = [TranscriptCue(
            "cue-1", 0, 10,
            "I look for bottlenecks and maybe the supply chain needs to scale up.",
            "我会寻找瓶颈，供应链可能需要扩容。",
        )]
        translator = ChineseTranslator()

        with self.assertRaisesRegex(
            ValueError, "reviewed subtitle translation exhausted",
        ):
            NaturalSubtitleTranslator(
                translator, None, UnavailableReviewer(),
            ).segment_interview_subtitle_cards(cues, [])

        self.assertEqual(translator.calls, 3)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].id, "cue-1")

    def test_audio_verified_card_is_retranslated_and_independently_reviewed(self) -> None:
        class Translator:
            def _request_json(self, messages, max_tokens):
                self.assert_audio_source(messages[-1]["content"])
                return ({"translations": [{
                    "id": "cycle-card", "text": "算力产业的利润会逐渐向应用层转移",
                }]}, {"provider": "kimi", "model": "k3"})

            @staticmethod
            def assert_audio_source(prompt):
                if "The compute cycle" not in prompt:
                    raise AssertionError(prompt)

        class Reviewer:
            def _request_json(self, messages, max_tokens):
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                return ({"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 0.9, "naturalness_score": 0.9, "errors": [],
                } for row in rows]}, {"provider": "deepseek", "model": "deepseek-chat"})

        cues = [TranscriptCue(
            "cycle-card", 0, 8,
            "The compet cycle moves earnings toward the application layer",
            "计算周期把利润推向应用层。",
        )]
        audit = {"corrections": [{
            "cue_id": "cycle-card",
            "source": cues[0].source_text,
            "verified_source": (
                "The compute cycle moves earnings toward the application layer"
            ),
            "reasons": ["unknown_cycle_modifier:compet"],
        }]}

        trace = NaturalSubtitleTranslator(
            Translator(), None, Reviewer(),
        ).repair_audio_verified_cards(cues, [], audit)

        self.assertEqual(
            cues[0].source_text,
            "The compute cycle moves earnings toward the application layer",
        )
        self.assertEqual(cues[0].translation, "算力产业的利润会逐渐向应用层转移。")
        self.assertEqual(trace["cue_ids"], ["cycle-card"])

    def test_audio_verified_unchanged_card_keeps_reviewed_translation(self) -> None:
        class MustNotCall:
            def _request_json(self, *args, **kwargs):
                raise AssertionError("unchanged audio evidence must not trigger a rewrite")

        cues = [TranscriptCue(
            "benchmark-card", 0, 5,
            "AI labs use coding as as a competitive benchmark",
            "AI 实验室把编程作为竞争性基准测试。",
        )]
        audit = {"corrections": [{
            "cue_id": "benchmark-card",
            "source": cues[0].source_text,
            "verified_source": cues[0].source_text,
            "reasons": ["repeated_content_word:as"],
            "changed": False,
        }]}

        trace = NaturalSubtitleTranslator(
            MustNotCall(), None, MustNotCall(),
        ).repair_audio_verified_cards(cues, [], audit)

        self.assertIsNone(trace)
        self.assertEqual(cues[0].translation, "AI 实验室把编程作为竞争性基准测试。")

    def test_audio_verified_dangling_fragment_merges_into_accepted_payoff(self) -> None:
        class MustNotCall:
            def _request_json(self, *args, **kwargs):
                raise AssertionError("dangling fragment merge must remain model-free")

        preferred = "长期来看，行业利润会逐渐向应用层转移。"
        cues = [
            TranscriptCue(
                "cycle-card-1", 85.0, 87.0,
                "The compet cycle tends to be though", "竞争周期通常如此。",
            ),
            TranscriptCue(
                "cycle-card-2", 87.0, 92.8,
                "that earnings over long stretches move toward the application layer",
                preferred,
            ),
        ]
        audit = {"corrections": [{
            "cue_id": "cycle-card-1",
            "source": cues[0].source_text,
            "verified_source": "cycle tends to be",
            "reasons": ["unknown_cycle_modifier:compet"],
        }]}

        trace = NaturalSubtitleTranslator(
            MustNotCall(), None, MustNotCall(),
        ).repair_audio_verified_cards(cues, [], audit)

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].id, "cycle-card-2")
        self.assertEqual(cues[0].start, 85.0)
        self.assertEqual(cues[0].translation, preferred)
        self.assertTrue(cues[0].source_text.startswith("cycle tends to be that earnings"))
        self.assertEqual(trace["merged_fragments"][0]["removed_cue_id"], "cycle-card-1")

    def test_final_chinese_style_gate_does_not_rewrite_without_specific_error(self) -> None:
        class MustNotRun:
            def _request_json(self, *args, **kwargs):
                raise AssertionError("global style rewrite must not run")

        cues = [
            TranscriptCue(
                "country", 0, 5, "what about the other layers across the United States?",
                "全美其他层面怎么办。",
            ),
            TranscriptCue(
                "downstream", 5, 8, "Now I'm doing downstream.",
                "我现在往下游做。",
            ),
            TranscriptCue(
                "cycle", 8, 12, "The competitive cycle tends to move earnings up the stack.",
                "竞争周期往往是，利润向上转移。",
            ),
            TranscriptCue(
                "cycle_owner", 12, 16, "Earnings move during the competitive cycle.",
                "竞争周期的利润会向应用层转移。",
            ),
        ]
        trace = NaturalSubtitleTranslator(
            MustNotRun(), MustNotRun(),
        ).repair_interview_chinese_style(cues, [])

        self.assertIsNone(trace)
        self.assertEqual(cues[1].translation, "我现在往下游做。")

    def test_final_chinese_style_gate_repairs_post_terminology_density(self) -> None:
        class CapturingWriter:
            def __init__(self) -> None:
                self.prompt = ""

            def _request_json(self, messages, **kwargs):
                self.prompt = messages[-1]["content"]
                return {
                    "translations": [{
                        "id": "data-center-card-1",
                        "text": "data center：数据中心开放了。",
                    }],
                }, {"provider": "test"}

        writer = CapturingWriter()
        cues = [TranscriptCue(
            "data-center-card-1", 0, 4,
            "the data center opened",
            "这是一个为了触发术语补全后密度门而故意写得特别特别长的中文测试句子。",
        )]
        terminology = [TerminologyEntry(
            "data center", TerminologyStrategy.BILINGUAL_ONCE,
            first_use_explanation="数据中心",
        )]

        trace = NaturalSubtitleTranslator(
            writer, None,
        ).repair_interview_chinese_style(cues, terminology)

        self.assertEqual(trace["step"], "interview_chinese_style_repair")
        self.assertEqual(cues[0].translation, "data center：数据中心开放了。")
        self.assertLessEqual(
            len(re.sub(r"\s+", "", cues[0].translation)),
            INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS,
        )
        self.assertIn("first_use_explanation", writer.prompt)
        self.assertIn(
            f"within {INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS} visible characters",
            writer.prompt,
        )

    def test_headline_font_places_fullwidth_comma_near_baseline(self) -> None:
        from PIL import ImageFont

        font = ImageFont.truetype(str(_resolve_headline_font_path()), 64)
        ideograph = font.getbbox("模")
        comma = font.getbbox("，")

        self.assertGreaterEqual(comma[1], ideograph[3] - 14)
        self.assertGreaterEqual(comma[3], ideograph[3])

    def test_chinese_subtitle_font_places_fullwidth_comma_near_baseline(self) -> None:
        from PIL import ImageFont

        font = ImageFont.truetype(str(_resolve_chinese_subtitle_font_path()), 44)
        ideograph = font.getbbox("模")
        comma = font.getbbox("，")

        self.assertGreaterEqual(comma[1], ideograph[3] - 10)
        self.assertGreaterEqual(comma[3], ideograph[3])

    def test_interview_hook_context_uses_chinese_punctuation_font(self) -> None:
        hook = HookSpec(
            "hook-1", HookStrategy.CONTRARIAN,
            "AI 写代码一夜普及，其他知识工作却卡住了",
            "代码产出能被文本完整承载、实验室每天拿它刷榜自测。",
            SourceRange(0, 12), ["cue-1"], ["cue-1"],
            speaker_label="Sequoia Capital 对谈", selected=True,
        )
        with TemporaryDirectory() as temp, patch(
            "video_factory.youtube._fit_text_by_pixels", wraps=_fit_text_by_pixels,
        ) as fit_text:
            _write_hook_overlay_concat(
                hook, 12, Path(temp) / "interview.mp4",
                RenderProfile.WECHAT_VERTICAL,
            )

        context_call = next(
            call for call in fit_text.call_args_list if call.args[0] == hook.promise
        )
        self.assertEqual(
            Path(context_call.args[1]), _resolve_chinese_subtitle_font_path(),
        )

    def test_dependent_caption_fragments_merge_into_one_complete_display_phrase(self) -> None:
        cues = [
            TranscriptCue("c1", 0, 3.9, "I think the concept of", "我觉得这个概念"),
            TranscriptCue(
                "c2", 3.95, 9.9,
                "a human being logging into an analytics dashboard and squinting at line charts",
                "是人登录 analytics dashboard，眯着眼看 line charts。",
            ),
            TranscriptCue("c3", 10, 12, "That is already changing.", "这已经在改变。"),
        ]

        merged = merge_dependent_subtitle_cues(cues)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].start, 0)
        self.assertEqual(merged[0].end, 9.9)
        self.assertIn("concept of a human", merged[0].source_text)
        self.assertIn("人登录", merged[0].translation)

    def test_dangling_fragment_can_cross_soft_merge_duration_limit(self) -> None:
        cues = [
            TranscriptCue(
                "c1", 0, 29.5, "The royalty of", "收入来自。",
            ),
            TranscriptCue(
                "c2", 29.55, 34.0, "an AI product goes to the model layer.",
                "AI 产品的收入流向模型层。",
            ),
        ]

        merged = merge_dependent_subtitle_cues(cues)

        self.assertEqual(len(merged), 1)
        self.assertIn("royalty of an AI product", merged[0].source_text)

    def test_hedge_maybe_stays_with_the_claim_that_follows(self) -> None:
        cues = [
            TranscriptCue(
                "c1", 0, 12,
                "I look for constraints around extraordinary companies, maybe",
                "我会寻找卓越公司周围的制约因素，也许",
            ),
            TranscriptCue(
                "c2", 12.1, 25,
                "it's the supply chain that has to scale up.",
                "需要扩容的是供应链。",
            ),
        ]

        merged = merge_dependent_subtitle_cues(cues)

        self.assertEqual(len(merged), 1)
        self.assertIn("maybe it's the supply chain", merged[0].source_text)

    def test_long_merged_thought_splits_before_maybe_not_after_it(self) -> None:
        source = (
            "I look across the ecosystem and look for bottlenecks and if there are places "
            "where extraordinary companies are being built constraints may appear uh "
            "maybe it's the supply chain that has to scale up so that compute can deploy "
            "and so this is like looking upstream."
        )

        parts = _semantic_english_parts(source, 5)

        self.assertGreaterEqual(len(parts), 2)
        self.assertLessEqual(len(parts), 5)
        self.assertTrue(any("maybe it's" in part for part in parts))
        self.assertFalse(any(part.rstrip(" ,.;").endswith("maybe") for part in parts))
        self.assertTrue(all(
            not part.casefold().rstrip(" ,.;").endswith((" and", " so", " uh", " um"))
            for part in parts[:-1]
        ))

    def test_semantic_split_keeps_defining_relative_clause_with_its_noun(self) -> None:
        source = (
            "Like lines of code, ideally good code is the thing "
            "that will be correlated to whether you produced software people wanted."
        )

        parts = _semantic_english_parts(source, 2)

        self.assertEqual(parts, [source])

    def test_semantic_split_does_not_cut_lists_or_open_relative_clauses(self) -> None:
        list_source = (
            "knowledge and expertise and meetings and everything but ultimately "
            "the text is the thing that produces the program people want."
        )
        relative_source = (
            "It is a technical audience where when they deploy an agentic system "
            "and they run into a bug, they fix it. They know how to triage it."
        )

        list_parts = _semantic_english_parts(list_source, 4)
        relative_parts = _semantic_english_parts(relative_source, 3)

        self.assertFalse(any(part.startswith("and expertise") for part in list_parts))
        self.assertFalse(any(part.endswith("agentic system") for part in relative_parts))
        self.assertFalse(any(part.casefold() == "but but like" for part in list_parts))

    def test_overlong_semantic_part_is_refined_before_card_translation(self) -> None:
        source = (
            "That is good because without it we will not have broad diffusion "
            "because otherwise we will return to a mainframe ecosystem."
        )

        parts = _split_overlong_semantic_parts([source], 16.0)
        weights = [len(part.split()) for part in parts]

        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(all(
            16.0 * weight / sum(weights) <= 7.5
            or len(_semantic_english_parts(part, 2)) == 1
            for part, weight in zip(parts, weights)
        ))

    def test_merged_caption_repair_batches_large_interview_without_dropping_ids(self) -> None:
        class CopyWriter:
            def __init__(self) -> None:
                self.batch_sizes = []

            def _request_json(self, messages, max_tokens):
                prompt = messages[-1]["content"]
                prefix = "Cues: " if "Cues: " in prompt else "Rows: "
                line = next(row for row in prompt.splitlines() if row.startswith(prefix))
                rows = json.loads(line[len(prefix):])
                self.batch_sizes.append(len(rows))
                return ({
                    "translations": [
                        {"id": row["id"], "text": f"第{row['id'][1:]}条完整字幕。"}
                        for row in rows
                    ],
                }, {"provider": "deepseek", "model": "copy"})

        copy_writer = CopyWriter()
        wrapper = type("Wrapper", (), {"fallback": copy_writer})()
        translator = NaturalSubtitleTranslator(wrapper)
        before = [
            TranscriptCue(f"c{index}", index * 3, index * 3 + 3, f"fragment {index}", "旧字幕。")
            for index in range(9)
        ]
        merged = [
            TranscriptCue(
                cue.id, cue.start, cue.end,
                cue.source_text + " completed", cue.translation,
            ) for cue in before
        ]

        trace = translator.repair_merged_interview_translations(before, merged, [])

        self.assertEqual(copy_writer.batch_sizes, [8, 8, 1, 1])
        self.assertEqual(len(trace["batches"]), 2)
        self.assertEqual([cue.translation for cue in merged], [
            f"第{index}条完整字幕。" for index in range(9)
        ])

    def test_interview_directing_audit_uses_primary_model_not_transport_fallback(self) -> None:
        class PrimaryCritic:
            def __init__(self) -> None:
                self.calls = 0

            def _request_json(self, messages, max_tokens):
                self.calls += 1
                self.assert_prompt = messages[-1]["content"]
                return ({
                    "critique": "现有文案只在复述主题，没有把模型层与应用公司的利益冲突放到第一眼。",
                    "hook_options": [
                        {
                            "headline": "模型层拿走利润，产品公司还怎么活？",
                            "context": "模型成本必须被竞争压低，应用公司才有空间建立可持续利润。",
                        },
                        {
                            "headline": "微软CEO：开源才能让应用公司赚到钱",
                            "context": "开放竞争会压低模型层租金，让价值和利润逐步转移到应用层。",
                        },
                        {
                            "headline": "只有模型赚钱，AI生态就不可持续",
                            "context": "如果模型层拿走全部价值，产品公司就无法形成利润，生态也无法持续。",
                        },
                    ],
                }, {"provider": "openrouter", "model": "editorial-primary"})

        class FallbackWrapper:
            def __init__(self) -> None:
                self.primary = PrimaryCritic()
                self.fallback = self.Judge()
                self.calls = 0

            class Judge:
                def __init__(self) -> None:
                    self.calls = 0

                def _request_json(self, messages, max_tokens):
                    self.calls += 1
                    return ({
                        "rationale": "现有标题太平，新候选把利益冲突说清楚了。",
                        "ranked_ids": ["proposal-1", "proposal-2", "proposal-3"],
                    }, {"provider": "deepseek", "model": "judge"})

            def _request_json(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("focused directing must not use the sticky fallback wrapper")

        writer = FallbackWrapper()
        translator = NaturalSubtitleTranslator(writer)
        cues = [TranscriptCue(
            "cue-1", 10, 70,
            "If the model layer captures all the value, product companies cannot live. "
            "Open source competition must lower model costs so applications can build margins.",
        )]
        plan = {"wechat_lessons": [{
            "title": "模型层独吞AI利润，微软CEO说这不可持续",
            "thesis": "模型层不能拿走所有价值。",
            "start": 10, "end": 70,
            "hook_headlines": ["模型层与应用层", "AI生态需要发展", "微软CEO谈开源"],
            "hook_context": "他谈到了模型与应用。",
        }]}

        trace = translator.audit_interview_directing(plan, cues, 90)

        self.assertEqual(writer.primary.calls, 1)
        self.assertEqual(writer.fallback.calls, 1)
        self.assertEqual(writer.calls, 0)
        self.assertEqual(trace["provenance"]["provider"], "openrouter")
        self.assertEqual(
            plan["wechat_lessons"][0]["hook_headlines"][0],
            "模型层拿走利润，产品公司还怎么活？",
        )
        self.assertEqual(
            plan["wechat_lessons"][0]["title"],
            "模型层拿走利润，产品公司还怎么活？",
        )
        self.assertEqual(
            plan["collection_title"],
            "模型层拿走利润，产品公司还怎么活？",
        )
        self.assertEqual(
            trace["winning_title"],
            "模型层拿走利润，产品公司还怎么活？",
        )
        self.assertEqual(
            plan["wechat_lessons"][0]["hook_context"],
            "模型成本必须被竞争压低，应用公司才有空间建立可持续利润。",
        )

    def test_interview_directing_preserves_complete_incumbent_pair(self) -> None:
        approved_context = (
            "针对居民对数据中心抢占水电的抵制，Meta 推出社区契约："
            "在自负电费之余降电价、用AI节水，并将税收盈余直接拨付给当地教师与急救人员；"
            "承诺不接受的社区可直接拒绝"
        )

        class Critic:
            def _request_json(self, messages, max_tokens):
                return ({
                    "critique": "标题还可以更直接。",
                    "hook_options": [
                        {
                            "headline": "社区不要数据中心，Meta说可以拒绝",
                            "context": "Meta承诺自付电费并用AI节水，社区若仍不接受项目就可以直接拒绝。",
                        },
                        {
                            "headline": "Meta承诺自己承担数据中心电费",
                            "context": "公司还会降低当地电价并把税收盈余拨给教师和急救人员。",
                        },
                        {
                            "headline": "Meta用社区契约回应数据中心抵制",
                            "context": "这份契约覆盖水电、税收和拒绝权，让当地社区决定项目是否进入。",
                        },
                    ],
                }, {"provider": "test", "model": "critic"})

        class Judge:
            def _request_json(self, messages, max_tokens):
                return ({
                    "rationale": "现有标题和完整解释合在一起最强。",
                    "ranked_ids": ["incumbent", "proposal-1", "proposal-2"],
                }, {"provider": "test", "model": "judge"})

        critic = Critic()
        wrapper = type("Wrapper", (), {
            "primary": critic,
            "fallback": Judge(),
        })()
        translator = NaturalSubtitleTranslator(wrapper)
        cues = [TranscriptCue(
            "cue-1", 0, 60,
            "Meta was criticized over data-center water and power use. Communities can say no. "
            "Meta will pay its electricity costs, lower rates, use AI to save water, and direct "
            "tax surplus to local teachers and first responders.",
        )]
        plan = {"wechat_lessons": [{
            "title": "Meta 数据中心被骂惨",
            "thesis": "Meta提出社区契约回应水电争议。",
            "start": 0, "end": 60,
            "hook_headlines": [
                "Meta 数据中心被骂惨",
                "社区可以拒绝Meta数据中心",
                "Meta承诺自付数据中心电费",
            ],
            "hook_context": approved_context,
        }]}

        trace = translator.audit_interview_directing(plan, cues, 60)

        self.assertEqual(plan["wechat_lessons"][0]["hook_context"], approved_context)
        self.assertEqual(
            plan["wechat_lessons"][0]["hook_headlines"][0],
            "Meta 数据中心被骂惨",
        )
        self.assertEqual(trace["winning_pair_id"], "incumbent")
        self.assertEqual(trace["winning_context"], approved_context)

    def test_old_directing_policy_cannot_reopen_unchanged_accepted_pair(self) -> None:
        hooks = [
            "按席位收费活不下去，纯用量定价三周没人碰",
            "把 CRM 工作拆成四类，核心收平台费",
            "定价按工作类型拆开，客户才愿意开始用",
        ]
        context = (
            "Keith 先试按席位收费，发现活不长；再试纯用量定价，结果三周没人碰产品。"
            "最后他把 Lightfield 的工作拆成四类：核心 CRM 收平台费，管道生成和"
            "工作流自动化按用量，智能预测按价值收费。"
        )
        old_trace = [{
            "step": "interview_directing_audit",
            "policy_version": "2026-09-17-good-render",
            "hook_headlines": hooks,
        }]

        matched = _matching_completed_directing_audit(old_trace, hooks, context)

        self.assertIs(matched, old_trace[0])

    def test_approved_queue_manifest_restores_exact_first_screen_pair(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            manifest_id = "youtube-PJrntzMA4iQ-golden"
            (workspace.collections_dir / f"{manifest_id}.json").write_text(json.dumps({
                "source_video_id": "PJrntzMA4iQ",
                "items": [{
                    "source_ranges": [{
                        "start": 2.0, "end": 120.4,
                        "original_start": 477.76, "original_end": 596.16,
                    }],
                    "renders": [{
                        "selected_hook": {
                            "headline_zh": "AI 不是泡沫，但 2026 年必须靠收入说话",
                            "promise": (
                                "Brad 算了一笔账：云厂商建数据中心是为了出租，不是自己买单。"
                                "如果实验室收入停滞，租金就付不起，资本开支循环就会断。"
                            ),
                        },
                        "hook_candidates": [
                            {"headline_zh": "AI 不是泡沫，但 2026 年必须靠收入说话"},
                            {"headline_zh": "1.5 万亿 capex 最终要靠收入买单"},
                            {"headline_zh": "没人付租金，AI 基建循环就会断"},
                        ],
                    }],
                }],
            }, ensure_ascii=False), encoding="utf-8")
            batch_dir = workspace.publish_dir / "approved-golden"
            batch_dir.mkdir()
            (batch_dir / "batch.json").write_text(json.dumps({
                "manifest_id": manifest_id,
                "approved_at": "2026-09-21T12:00:00Z",
                "queue_hidden": False,
            }), encoding="utf-8")
            factory = YouTubeCollectionFactory(workspace, object())

            pair = factory._approved_interview_hook_pair(
                "PJrntzMA4iQ",
                {"original_start": 477.76, "original_end": 596.16},
            )

        self.assertEqual(
            pair["headline"], "AI 不是泡沫，但 2026 年必须靠收入说话",
        )
        self.assertIn("资本开支循环就会断", pair["context"])
        self.assertEqual(pair["manifest_id"], manifest_id)

    def test_atomic_hook_package_never_takes_context_from_another_candidate(self) -> None:
        class Critic:
            def _request_json(self, *args, **kwargs):
                return ({
                    "critique": "compare complete packages",
                    "hook_options": [
                        {
                            "headline": "AI 不是泡沫，但 2026 年必须靠收入说话",
                            "context": "头部实验室若没有足够收入，就付不起云厂商建设数据中心所需的租金。",
                        },
                        {
                            "headline": "1.5 万亿 capex，最终要由谁来买单？",
                            "context": "微软、谷歌和亚马逊建设算力是为了出租，资本循环必须靠客户收入维持。",
                        },
                        {
                            "headline": "AI 数据中心的循环可能在租金这里断掉",
                            "context": "如果实验室月收入停滞，云厂商就无法用租金覆盖持续扩张的基础设施投入。",
                        },
                    ],
                }, {"provider": "test", "model": "critic"})

        class Judge:
            def _request_json(self, *args, **kwargs):
                return ({
                    "rationale": "第二组问题和解释配合最好。",
                    "ranked_ids": ["proposal-2", "proposal-1", "proposal-3"],
                }, {"provider": "test", "model": "judge"})

        wrapper = type("Wrapper", (), {
            "primary": Critic(), "fallback": Judge(),
        })()
        translator = NaturalSubtitleTranslator(wrapper)
        cues = [TranscriptCue(
            "cue-1", 0, 60,
            "Microsoft, Google, and Amazon build data centers to rent them. "
            "If lab revenue stalls, rent cannot cover 1.5 trillion dollars of capex.",
        )]
        plan = {"wechat_lessons": [{
            "title": "AI 基础设施投入",
            "thesis": "收入必须覆盖数据中心租金。",
            "start": 0, "end": 60,
            "hook_headlines": ["AI 基础设施投入", "云厂商建设数据中心", "实验室需要收入"],
            "hook_context": "一段关于数据中心投入的讨论。",
        }]}

        trace = translator.audit_interview_directing(plan, cues, 60)

        self.assertEqual(trace["winning_pair_id"], "proposal-2")
        self.assertEqual(
            plan["wechat_lessons"][0]["hook_context"],
            "微软、谷歌和亚马逊建设算力是为了出租，资本循环必须靠客户收入维持。",
        )

    def test_historical_fixed_subtitles_fit_real_hook_panel(self) -> None:
        contexts = [
            (
                "Keith 先试按席位收费，发现活不长；再试纯用量定价，结果三周没人碰产品。"
                "最后他把 Lightfield 的工作拆成四类：核心 CRM 收平台费，管道生成和"
                "工作流自动化按用量，智能预测按价值收费。"
            ),
            (
                "Brad 算了一笔账：微软、谷歌、亚马逊建数据中心是为了出租，不是自己买单。"
                "如果头部实验室月收入停在 40 亿，租金就付不起，1.5 万亿 capex 的循环就会断。"
            ),
            (
                "针对居民对数据中心抢占水电的抵制，Meta 推出社区契约：在自负电费之余降电价、"
                "用 AI 节水，并将税收盈余直接拨付给当地教师与急救人员；承诺不接受的社区可直接拒绝。"
            ),
        ]

        self.assertTrue(all(
            _interview_hook_context_fits_overlay(context)
            for context in contexts
        ))

    def test_interview_directing_audit_blocks_invalid_primary_output(self) -> None:
        class InvalidPrimary:
            def _request_json(self, *args, **kwargs):
                return ({
                    "critique": "plain",
                    "hook_options": [{
                        "headline": "这是一段访谈",
                        "context": "总结了AI行业。",
                    }] * 3,
                }, {"provider": "openrouter", "model": "editorial-primary"})

        class FallbackWrapper:
            primary = InvalidPrimary()

            def _request_json(self, *args, **kwargs):
                return ({}, {"provider": "deepseek", "model": "fallback"})

        translator = NaturalSubtitleTranslator(FallbackWrapper())
        cues = [TranscriptCue(
            "cue-1", 0, 60,
            "The model layer captures all the value and product companies cannot live.",
        )]
        plan = {"wechat_lessons": [{
            "title": "模型层拿走所有利润",
            "thesis": "产品公司无法持续。",
            "start": 0, "end": 60,
            "hook_headlines": ["模型层拿走利润", "产品公司无法持续", "开源竞争改变利润"],
            "hook_context": "模型与应用在争夺价值。",
        }]}

        with self.assertRaisesRegex(RuntimeError, "refusing fallback-quality hooks"):
            translator.audit_interview_directing(plan, cues, 60)

    def test_translation_density_repair_shortens_only_overfast_cues(self) -> None:
        class ConciseWriter:
            def _request_json(self, *args, **kwargs):
                return ({"translations": [{
                    "id": "cue-0743",
                    "text": "knowledge work。也许能画出分布图，但不知是否有人发表。",
                }]}, {"provider": "test", "model": "concise"})

        cues = [TranscriptCue(
            "cue-0743", 0, 3.04,
            "knowledge work, and you would probably have a histogram; I do not know if anyone published it.",
            "knowledge work。你大概会得到一个 histogram，我不知道有没有人发表过，也许",
        )]
        translator = NaturalSubtitleTranslator(ConciseWriter())

        self.assertEqual([cue.id for cue in fast_translation_cues(cues)], ["cue-0743"])
        trace = translator._repair_reading_speed(cues)

        self.assertEqual(trace["step"], "subtitle_reading_speed_repair")
        self.assertEqual(fast_translation_cues(cues), [])
        self.assertIn("knowledge work", cues[0].translation)

    def test_translation_density_repair_retries_with_exact_limit_feedback(self) -> None:
        class RetryWriter:
            def __init__(self):
                self.prompts = []

            def _request_json(self, messages, **kwargs):
                self.prompts.append(messages[-1]["content"])
                text = (
                    "这是一条仍然明显超过限制而且没有认真压缩的中文翻译字幕文本"
                    if len(self.prompts) == 1 else "可读的短翻译"
                )
                return ({"translations": [{"id": "dense", "text": text}]}, {
                    "attempt": len(self.prompts),
                })

        writer = RetryWriter()
        cue = TranscriptCue("dense", 0, 1.5, "A dense sentence.", "一段非常长的翻译文本用于触发修复。")

        trace = NaturalSubtitleTranslator(writer)._repair_reading_speed([cue])

        self.assertEqual(len(writer.prompts), 2)
        self.assertIn("maximum is 18", writer.prompts[1])
        self.assertEqual(trace["step"], "subtitle_reading_speed_repair")

    def test_translation_density_repair_retries_when_required_term_is_dropped(self) -> None:
        class TerminologyWriter:
            def __init__(self):
                self.calls = 0

            def _request_json(self, *args, **kwargs):
                self.calls += 1
                text = "知识工作能画成分布图" if self.calls == 1 else "knowledge work 能画成分布图"
                return ({"translations": [{"id": "dense", "text": text}]}, {
                    "attempt": self.calls,
                })

        writer = TerminologyWriter()
        cue = TranscriptCue("dense", 0, 3, "knowledge work can form a histogram.", "很长的翻译")
        terminology = [TerminologyEntry(
            "knowledge work", TerminologyStrategy.PRESERVE,
        )]

        NaturalSubtitleTranslator(writer)._repair_reading_speed(
            [cue], terminology=terminology,
        )

        self.assertEqual(writer.calls, 2)
        self.assertIn("knowledge work", cue.translation)

    def test_translation_density_repair_keeps_bilingual_first_use_explanation(self) -> None:
        class ExplanationWriter:
            def __init__(self):
                self.calls = 0

            def _request_json(self, *args, **kwargs):
                self.calls += 1
                text = "harness 提升可靠性" if self.calls == 1 else "harness（智能体运行框架）更可靠"
                return ({"translations": [{"id": "dense", "text": text}]}, {
                    "attempt": self.calls,
                })

        writer = ExplanationWriter()
        cue = TranscriptCue("dense", 0, 3, "The harness is more reliable.", "很长的翻译")
        terminology = [TerminologyEntry(
            "harness", TerminologyStrategy.BILINGUAL_ONCE,
            first_use_explanation="智能体运行框架",
        )]

        NaturalSubtitleTranslator(writer)._repair_reading_speed(
            [cue], terminology=terminology,
        )

        self.assertEqual(writer.calls, 2)
        self.assertEqual(cue.translation.count("智能体运行框架"), 1)

    def test_translation_retries_when_model_drops_cue_id_prefix(self) -> None:
        class PrefixRetryWriter:
            def __init__(self):
                self.translation_calls = 0

            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                if "senior Chinese editor" in prompt:
                    return ({
                        "editorial_mode": "known_tech_interview_clip",
                        "collection_title": "AI经济学",
                        "story_start": 0, "story_end": 60,
                        "terminology": [], "bilibili_chapters": [],
                        "wechat_lessons": [{
                            "start": 0, "end": 60,
                            "title": "应用层必须能赚钱",
                            "thesis": "应用公司需要可持续利润。",
                            "speaker_label": "Satya Nadella",
                            "framing": "speaker",
                            "hook_headlines": [
                                "应用层必须能赚钱", "模型定价决定利润", "开源竞争压低成本",
                            ],
                        }],
                    }, {"step": "plan"})
                self.translation_calls += 1
                cue_id = "0001" if self.translation_calls == 1 else "cue-0001"
                return ({"translations": [{
                    "id": cue_id, "text": "应用公司需要可持续利润。",
                }]}, {"step": self.translation_calls})

        writer = PrefixRetryWriter()
        cues = [TranscriptCue(
            "cue-0001", 0, 60,
            "Application companies need sustainable margins.",
        )]

        _, _, trace = NaturalSubtitleTranslator(writer).translate(
            {"duration": 60, "title": "Satya on AI economics"},
            cues, "known_tech_interview_clip",
        )

        self.assertEqual(writer.translation_calls, 2)
        self.assertEqual(cues[0].translation, "应用公司需要可持续利润。")
        self.assertTrue(any(row["step"] == "translate_chunk_rejected" for row in trace))

    def test_spoken_fillers_are_omitted_without_dropping_meaning(self) -> None:
        source = "Um, open source wins, you know, inference cloud, uh."
        translation = "嗯，开源赢了，你知道，推理云，呃。"

        self.assertEqual(
            omit_spoken_fillers_from_translation(source, translation),
            "开源赢了，推理云。",
        )
        self.assertTrue(source_is_spoken_filler_only("Um, uh, you know."))
        self.assertEqual(omit_spoken_fillers_from_translation("Um, uh.", "嗯，呃。"), "")

    def test_non_speech_directions_are_omitted_from_both_visible_languages(self) -> None:
        source = "Um [clears throat] and then I would hypothesize they executed well"
        translation = "[清喉咙]，然后我推测他们执行得很好"

        self.assertEqual(
            omit_spoken_fillers_from_translation(source, translation),
            "然后我推测他们执行得很好",
        )
        self.assertEqual(
            omit_non_speech_directions(source),
            "Um and then I would hypothesize they executed well",
        )
        self.assertTrue(source_is_non_speech_only("[clears throat]"))
        self.assertEqual(omit_spoken_fillers_from_translation("[coughs]", "[咳嗽]"), "")

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

    def test_terminology_repair_falls_back_after_directing_provider_failure(self) -> None:
        class FailedDirector:
            def _request_json(self, *args, **kwargs):
                raise RuntimeError("transient SSL EOF")

        class RecoveryWriter:
            def _request_json(self, *args, **kwargs):
                return ({"translations": [{
                    "id": "c1", "text": "芯片供应链需要扩容。",
                }]}, {"provider": "deepseek", "model": "recovery"})

        cues = [TranscriptCue(
            "c1", 0, 4, "The chips supply chain must scale.",
            "供应网络需要扩容。",
        )]
        terms = [TerminologyEntry(
            "supply chain", TerminologyStrategy.TRANSLATE, target="供应链",
        )]
        errors = terminology_contract_errors(cues, terms)

        trace = NaturalSubtitleTranslator(
            RecoveryWriter(), FailedDirector(),
        )._repair_terminology(cues, terms, errors)

        self.assertEqual(cues[0].translation, "芯片供应链需要扩容。")
        self.assertEqual(trace["provenance"]["provider"], "deepseek")
        self.assertTrue(trace["provider_failures"])

    def test_cached_reviewed_translation_is_reaudited_before_render(self) -> None:
        cues = [TranscriptCue(
            "c1", 0, 4, "Agents are replacing seats.",
            "Agent 正在取代人工席位。",
        )]
        terminology = [TerminologyEntry(
            "agent", TerminologyStrategy.BILINGUAL_ONCE,
            first_use_explanation="智能体",
        )]

        trace = enforce_cached_terminology_contract(cues, terminology)

        self.assertEqual(trace["step"], "cached_terminology_enforcement")
        self.assertIn("智能体", cues[0].translation)
        self.assertEqual(terminology_contract_errors(cues, terminology), [])

    def test_established_translate_terms_are_enforced_per_cue(self) -> None:
        cues = [
            TranscriptCue(
                "c1", 0, 4, "Models and chips both matter.",
                "模型和 chips 都很重要。",
            ),
            TranscriptCue(
                "c2", 4, 8, "Applications create durable value.",
                "applications 能创造长期价值。",
            ),
        ]
        terminology = NaturalSubtitleTranslator._parse_terminology([], cues)

        before = terminology_contract_errors(cues, terminology)
        enforced = NaturalSubtitleTranslator._enforce_terminology_contract(cues, terminology)
        after = terminology_contract_errors(cues, terminology)

        self.assertTrue(any("chips:c1" in error for error in before))
        self.assertTrue(any("applications:c2" in error for error in before))
        self.assertIn("chips", enforced)
        self.assertIn("applications", enforced)
        self.assertIn("芯片", cues[0].translation)
        self.assertIn("应用", cues[1].translation)
        self.assertNotIn("chips", cues[0].translation.casefold())
        self.assertNotIn("applications", cues[1].translation.casefold())
        self.assertEqual(after, [])

    def test_translate_term_cannot_be_masked_by_a_different_good_cue(self) -> None:
        cues = [
            TranscriptCue("c1", 0, 3, "chips scale", "芯片可以扩展。"),
            TranscriptCue("c2", 3, 6, "chips improve", "chips 继续改进。"),
        ]
        terminology = [TerminologyEntry(
            "chips", TerminologyStrategy.TRANSLATE, target="芯片",
        )]

        errors = terminology_contract_errors(cues, terminology)

        self.assertTrue(any("chips:c2" in error for error in errors))

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

    def test_scheduled_discovery_selects_at_most_one_and_waits_2_hours(self) -> None:
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

            now[0] += timedelta(hours=1, minutes=59)
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

    def test_probe_budget_expands_to_cover_every_pool_with_candidates(self) -> None:
        pools = {f"pool-{index}": [f"query {index}"] for index in range(13)}
        candidates = [
            YouTubeCandidate(
                video_id=f"video-{index}", url=f"https://youtube.com/watch?v={index}",
                title=f"AI engineering {index}", channel="Engineering",
                matched_pools=[pool],
            )
            for index, pool in enumerate(pools)
        ]

        selected = YouTubeDiscoveryService._choose_probe_candidates(
            candidates, DiscoveryConfig(metadata_probe_limit=10, query_pools=pools),
        )

        self.assertEqual(len(selected), 13)
        self.assertEqual({item.matched_pools[0] for item in selected}, set(pools))

    def test_direct_channel_receives_a_second_probe_for_next_newest_upload(self) -> None:
        candidates = [
            YouTubeCandidate(
                video_id=f"direct-{index}", url=f"https://youtube.com/watch?v=direct{index}",
                title=f"Direct source episode {index}", channel="Direct Source",
                matched_pools=["direct"], discovery_routes=["channel:direct"],
                channel_recency_rank=index,
            )
            for index in range(3)
        ]

        selected = YouTubeDiscoveryService._choose_probe_candidates(
            candidates,
            DiscoveryConfig(
                metadata_probe_limit=1, query_pools={},
                channel_sources={"direct": ["https://youtube.com/channel/direct"]},
            ),
        )

        self.assertEqual([item.video_id for item in selected], ["direct-0", "direct-1"])

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
            TranscriptCue(
                f"c{index}", index * 5, index * 5 + 5,
                "Agent systems must scale.", "智能体系统必须扩展。",
            )
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
            [TerminologyEntry(
                "Agent", TerminologyStrategy.TRANSLATE, target="智能体",
            )], plan, "", "",
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
            self.assertIn("智能体系统必须扩展。", chinese.read_text(encoding="utf-8"))
            combined = bilingual.read_text(encoding="utf-8")
            self.assertLess(
                combined.index("Agent systems must scale."),
                combined.index("智能体系统必须扩展。"),
            )

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

    def test_deterministic_structure_repair_preserves_long_source_quality_contract(self) -> None:
        duration = 4182.0
        cues = [
            TranscriptCue(f"c{index}", index * 10, min(duration, index * 10 + 10), "Agent systems lesson.")
            for index in range(419)
        ]
        plan = {
            "collection_title": "自改进 Agent 课程",
            "story_start": 0,
            "story_end": duration,
            "terminology": [],
            "bilibili_chapters": [
                {"title": "规模化规律", "thesis": "解释模型规模化。", "start": 0, "end": 900,
                 "hook_headlines": ["太短", "规模化的工程取舍", "规模化带来的系统变化"]},
                {"title": "Agent 工作流", "thesis": "解释 Agent 工作流。", "start": 2500, "end": 3300,
                 "hook_headlines": ["聊天不是 Agent", "工作流的工程取舍", "工作流带来的系统变化"]},
            ],
            "wechat_lessons": [
                {"title": f"技术精讲 {index}", "thesis": "独立技术观点。", "start": index * 500,
                 "end": index * 500 + 120, "hook_headlines": ["太短"]}
                for index in range(4)
            ],
        }

        repaired = normalize_editorial_plan_structure(plan, cues, duration)

        self.assertEqual(editorial_plan_contract_errors(repaired, duration), [])
        self.assertEqual(repaired["story_start"], 0)
        self.assertEqual(repaired["story_end"], duration)
        self.assertTrue(all(
            480 <= row["end"] - row["start"] <= 1800
            for row in repaired["bilibili_chapters"]
        ))
        self.assertTrue(all(
            180 <= row["end"] - row["start"] <= 360
            for row in repaired["wechat_lessons"]
        ))

    def test_oversized_interview_range_is_not_center_clamped_to_180_seconds(self) -> None:
        cues = [
            TranscriptCue(f"c{index}", index * 10, index * 10 + 10, "Complete thought.")
            for index in range(40)
        ]
        plan = {
            "editorial_mode": "known_tech_interview_clip",
            "collection_title": "访谈高光", "bilibili_chapters": [],
            "wechat_lessons": [{
                "speaker_label": "Speaker", "title": "一段完整的工程判断",
                "thesis": "解释原因与结果。", "start": 80, "end": 290,
                "framing": "speaker", "hook_headlines": [
                    "这个工程判断改变团队", "为什么旧方法失效", "系统选择决定结果",
                ],
            }],
        }

        repaired = normalize_editorial_plan_structure(plan, cues, 400)

        row = repaired["wechat_lessons"][0]
        self.assertEqual((row["start"], row["end"]), (80, 290))
        self.assertIn(
            "interview highlight must be a complete 45–180 second source range",
            editorial_plan_contract_errors(repaired, 400, cues),
        )

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

    def test_hook_normalizes_saas_typo_and_matches_cold_open_to_headline(self) -> None:
        cues = [
            TranscriptCue("generic", 0, 7, "Here is Fable 5.1, a model.", "这是 Fable 5.1 模型。"),
            TranscriptCue(
                "proof", 12, 19,
                "Open source models, inference clouds, and applications can all win.",
                "开源模型、推理云和应用公司都能赢。",
            ),
            TranscriptCue("outcome", 24, 31, "Nvidia supplies compute to every layer.", "Nvidia 向每一层供应算力。"),
        ]

        hooks = build_hook_candidates(
            "AI不是零和游戏", "开源、云和应用公司都能赢。",
            SourceRange(0, 31), cues, "episode", [
                "AI不是零和：开源、云、应用都能赢",
                "SAS 末日论被Nvidia财报证伪",
                "Nvidia算力支撑每一层",
            ],
        )

        self.assertEqual(hooks[0].source_cue_ids, ["proof"])
        self.assertIn("SaaS 末日论", hooks[1].headline_zh)

    def test_versioned_product_reveal_is_not_translated_as_literal_deixis(self) -> None:
        self.assertEqual(
            omit_spoken_fillers_from_translation(
                "Here is Fable 5.1.", "这是 Fable 5.1。",
            ),
            "Fable 5.1 来了。",
        )
        self.assertEqual(
            omit_spoken_fillers_from_translation(
                "Here is Fable 5.1, a model.", "这是 Fable 5.1 模型。",
            ),
            "Fable 5.1 来了。",
        )

    def test_ordinary_here_is_identification_remains_literal(self) -> None:
        self.assertEqual(
            omit_spoken_fillers_from_translation(
                "Here is the chart we discussed.", "这是我们刚才讨论的图表。",
            ),
            "这是我们刚才讨论的图表。",
        )

    def test_wechat_hook_keeps_interview_source_chronological(self) -> None:
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

        self.assertEqual([(row.start, row.end) for row in ranges], [(0, 300)])
        self.assertEqual(ranges[0].framing, FramingMode.SLIDE)
        self.assertTrue(ranges[0].has_explicit_crop)
        self.assertEqual((ranges[0].crop_x, ranges[0].crop_width), (500, 1320))
        self.assertEqual(sum(row.duration for row in ranges), item.duration)

    def test_interview_title_must_be_entailed_by_selected_source(self) -> None:
        plan = {
            "editorial_mode": "known_tech_interview_clip",
            "bilibili_chapters": [],
            "wechat_lessons": [{
                "start": 0, "end": 60,
                "title": "为什么 AI 写代码普及，写文档却没人用？",
                "thesis": "比较两类工作的采用速度。",
                "speaker_label": "Aaron Levie",
                "hook_headlines": [
                    "AI 写代码先普及", "文档工作为何没跟上", "开发者先吃到红利",
                ],
            }],
        }
        cues = [
            TranscriptCue(
                "routing", 0, 60,
                "An orchestration agent routes cheap open models and expensive frontier models.",
            ),
        ]

        errors = editorial_plan_contract_errors(plan, 60, cues)

        self.assertTrue(any("concepts absent" in error for error in errors), errors)

    def test_interview_title_entailment_accepts_matching_source(self) -> None:
        plan = {
            "editorial_mode": "known_tech_interview_clip",
            "bilibili_chapters": [],
            "wechat_lessons": [{
                "start": 0, "end": 60,
                "title": "营销仪表盘将死：AI 把营销变成高频交易",
                "thesis": "AI 驱动执行，人类负责监督。",
                "speaker_label": "James Cadwallader",
                "hook_headlines": [
                    "营销仪表盘正在消失", "AI 接管营销执行", "人类只负责监督",
                ],
            }],
        }
        cues = [
            TranscriptCue(
                "claim", 0, 60,
                "Marketing dashboards are dead. It becomes high frequency trading, AI driven and human supervised.",
            ),
        ]

        self.assertEqual(editorial_plan_contract_errors(plan, 60, cues), [])

    def test_interview_competition_title_accepts_explicit_competitive_cycle(self) -> None:
        plan = {
            "editorial_mode": "known_tech_interview_clip",
            "bilibili_chapters": [],
            "wechat_lessons": [{
                "start": 0, "end": 90,
                "title": "AI竞赛不是模型竞赛，而是应用竞赛",
                "thesis": "产业价值最终流向应用层。",
                "speaker_label": "黄仁勋",
                "hook_headlines": [
                    "AI竞赛不只看模型", "应用层才是长期赢家", "利润最终向上层迁移",
                ],
            }],
        }
        cues = [TranscriptCue(
            "claim", 0, 90,
            "This industry is not just about the model. It is mostly about applications. "
            "The competitive cycle moves earnings toward the application layer.",
        )]

        self.assertEqual(editorial_plan_contract_errors(plan, 90, cues), [])

    def test_interview_competition_title_accepts_complete_industry_cycle_argument(self) -> None:
        plan = {
            "editorial_mode": "known_tech_interview_clip",
            "bilibili_chapters": [],
            "wechat_lessons": [{
                "start": 0, "end": 90,
                "title": "AI竞赛不是模型竞赛，而是应用竞赛",
                "thesis": "产业价值最终流向应用层。",
                "speaker_label": "黄仁勋",
                "hook_headlines": [
                    "AI竞赛不只看模型", "应用层才是长期赢家", "利润最终向上层迁移",
                ],
            }],
        }
        cues = [TranscriptCue(
            "claim", 0, 90,
            "This industry is not just about the model. The cycle tends to move "
            "earnings over long stretches of time toward the application layer.",
        )]

        self.assertEqual(editorial_plan_contract_errors(plan, 90, cues), [])

    def test_supported_requested_title_is_preserved_after_model_weakens_it(self) -> None:
        cues = [TranscriptCue(
            "claim", 0, 60,
            "Marketing dashboards are dead. This becomes high frequency trading, AI driven and human supervised.",
        )]
        plan = {"wechat_lessons": [{
            "start": 0, "end": 60, "title": "营销仪表盘将死",
        }]}

        trace = _apply_supported_requested_title(
            plan, cues, 60,
            "Preserve the strong angle/title: 营销仪表盘将死：AI 把营销变成高频交易，人类只负责监督. Select one passage.",
        )

        self.assertTrue(trace and trace["applied"])
        self.assertEqual(
            plan["wechat_lessons"][0]["title"],
            "营销仪表盘将死：AI 把营销变成高频交易，人类只负责监督",
        )
        self.assertEqual(
            plan["collection_title"],
            "营销仪表盘将死：AI 把营销变成高频交易，人类只负责监督",
        )

    def test_unsupported_requested_title_is_not_forced_onto_clip(self) -> None:
        cues = [TranscriptCue("claim", 0, 60, "An orchestration agent routes models.")]
        plan = {"wechat_lessons": [{"start": 0, "end": 60, "title": "模型路由"}]}

        trace = _apply_supported_requested_title(
            plan, cues, 60,
            "Preserve title: 营销仪表盘将死：AI 把营销变成高频交易. Select one passage.",
        )

        self.assertFalse(trace and trace["applied"])
        self.assertEqual(plan["wechat_lessons"][0]["title"], "模型路由")

    def test_explicit_exact_guidance_range_is_parsed(self) -> None:
        self.assertEqual(
            _requested_exact_range_from_editorial_guidance(
                "Select exactly one answer from 16:24 to 17:14."
            ),
            (984.0, 1034.0),
        )

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
            commands = []

            def runner(command, **kwargs):
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

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
        self.assertIn("[1:a]atrim", filters)
        self.assertIn(
            "asetpts=PTS-0.000/TB,aresample=async=1:first_pts=0", filters,
        )
        self.assertIn("apad=whole_dur=300.000,atrim=duration=300.000", filters)
        self.assertIn("[2:v]format=rgba", filters)
        self.assertEqual(commands[0].count(str(source)), 2)

    def test_audio_loudness_probe_reads_volumedetect_levels(self) -> None:
        completed = subprocess.CompletedProcess(
            ["ffmpeg"], 0, "",
            "[Parsed_volumedetect] mean_volume: -14.2 dB\n"
            "[Parsed_volumedetect] max_volume: -1.8 dB\n"
            "[Parsed_silencedetect] silence_duration: 45.2\n",
        )
        with patch("video_factory.media.subprocess.run", return_value=completed) as run:
            loudness = probe_audio_loudness(
                Path("audible.mp4"), minimum_silence_seconds=3.0,
            )

        self.assertEqual(loudness.mean_db, -14.2)
        self.assertEqual(loudness.max_db, -1.8)
        self.assertEqual(loudness.longest_silence_seconds, 45.2)
        self.assertIn(
            "silencedetect=noise=-50dB:d=3", " ".join(run.call_args.args[0]),
        )

    def test_silent_audio_repair_rerenders_only_failed_output(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp))
            workspace.initialize()
            source = workspace.root / "source.mp4"
            source.write_bytes(b"source")
            output = workspace.root / "renders" / "collection" / "lesson.mp4"
            output.parent.mkdir(parents=True)
            output.write_bytes(b"silent")
            source_subtitle = output.with_suffix(".en.srt")
            translation_subtitle = output.with_suffix(".zh-Hans.srt")
            source_subtitle.write_text("", encoding="utf-8")
            translation_subtitle.write_text("", encoding="utf-8")
            render = PlatformRender(
                RenderProfile.WECHAT_VERTICAL, 1080, 1920,
                video_path=str(output.relative_to(workspace.root)),
                source_subtitle_path=str(source_subtitle.relative_to(workspace.root)),
                translation_subtitle_path=str(translation_subtitle.relative_to(workspace.root)),
            )
            item = CollectionItem(
                "lesson", CollectionItemKind.WECHAT_SHORT, 1,
                "技术精讲", "技术观点", [SourceRange(0, 180)], [render],
            )
            manifest = VideoCollectionManifest(
                "collection", "candidate", "https://youtube.com/watch?v=demo", "demo",
                "Demo", "Teacher", "Collection", [], [], [item],
                source_media_path="source.mp4", source_duration=180,
            )
            commands = []

            def runner(command, **kwargs):
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch(
                "video_factory.youtube.probe_audio_loudness",
                side_effect=[AudioLoudness(-91, -91), AudioLoudness(-13, -2)],
            ), patch(
                "video_factory.youtube.probe_video",
                return_value=VideoProbe(
                    output, 180, 1080, 1920, "h264", "yuv420p", "aac",
                    audio_duration=180, audio_bitrate=192000,
                ),
            ):
                repaired = YouTubeCollectionRenderer(workspace, runner=runner).repair_silent_audio(manifest)

        self.assertEqual(repaired, [render.video_path])
        self.assertEqual(len(commands), 1)

    def test_slide_translations_follow_reordered_hook_timeline(self) -> None:
        rows = _slide_translation_rows(
            [SlideTranslation(10, 20, "Deep Modules", "deep modules（深模块）")],
            [SourceRange(15, 20), SourceRange(0, 15), SourceRange(20, 30)],
        )

        self.assertEqual(rows, [
            (0, 5, "deep modules（深模块）", None, None),
            (15, 20, "deep modules（深模块）", None, None),
        ])

    def test_slide_translation_uses_chinese_punctuation_font(self) -> None:
        from PIL import ImageFont

        with TemporaryDirectory() as temp, patch.object(
            ImageFont, "truetype", wraps=ImageFont.truetype,
        ) as truetype:
            _write_slide_translation_overlay_concat(
                [SlideTranslation(0, 4, "Model value", "模型价值，正在变化。")],
                [SourceRange(0, 4)], RenderProfile.WECHAT_VERTICAL, 4,
                Path(temp) / "slide.mp4",
            )

        self.assertEqual(
            Path(truetype.call_args_list[0].args[0]),
            _resolve_chinese_subtitle_font_path(),
        )

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

    def test_hook_candidate_rejects_dangling_source_thought(self) -> None:
        cues = [
            TranscriptCue("dangling", 0, 7, "Open source and inference cloud win the"),
            TranscriptCue("complete", 12, 19, "Open source and inference clouds can all win."),
        ]

        hooks = build_hook_candidates(
            "开源、推理云和应用都能赢", "这不是零和游戏。",
            SourceRange(0, 20), cues, "episode", [
                "开源、推理云和应用都能赢",
                "这不是零和游戏", "开源生态仍在扩大",
            ],
        )

        self.assertTrue(all(hook.source_cue_ids == ["complete"] for hook in hooks))


if __name__ == "__main__":
    unittest.main()
