import json
import os
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
from video_factory.youtube_alignment import ALIGNMENT_POLICY_VERSION, source_words_from_cues
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
    _semantic_card_translation_errors,
    _caption_entity_alignment_errors,
    _caption_numeric_alignment_errors,
    _interview_caption_content_fingerprint,
    _numeric_audio_conflict_pairs,
    _compact_translation_trace,
    _write_translation_audit,
    _translation_trace_from_plan,
    _snapshot_plan_hooks,
    _remap_hook_snapshot,
    _terminology_decision_fingerprint,
    _strict_interview_source_fingerprint,
    cached_interview_caption_pipeline_complete,
    cached_joint_caption_pipeline_complete,
    interview_caption_duration_errors,
    INTERVIEW_CAPTION_POLICY_VERSION,
    INTERVIEW_CAPTION_POLICY_FINGERPRINT,
    INTERVIEW_CAPTION_HARD_MAX_SECONDS,
    INTERVIEW_CAPTION_MAX_ENGLISH_WORDS,
    INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND,
    interview_chinese_style_errors,
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
            ],
        )

    def test_preserve_row_with_chinese_target_normalizes_to_translate(self) -> None:
        cues = [TranscriptCue(
            "cue-1", 0, 4, "The request includes an IP address.",
        )]

        entries = NaturalSubtitleTranslator._parse_terminology([{
            "source": "IP address", "strategy": "preserve",
            "target": "IP 地址",
            "rationale": "The acronym stays in English while the compound has a settled Chinese form.",
        }], cues)

        entry = next(item for item in entries if item.source == "IP address")
        self.assertEqual(entry.strategy, TerminologyStrategy.TRANSLATE)
        self.assertEqual(entry.target, "IP 地址")

    def test_contextual_terminology_is_video_scoped_and_survives_cache_parse(self) -> None:
        cues = [TranscriptCue(
            "cue-1", 0, 6,
            "The recommendation agent ranks products from customer signals.",
            "推荐智能体会根据客户信号给产品排序。",
        )]

        entries = NaturalSubtitleTranslator._parse_terminology([{
            "source": "recommendation agent", "strategy": "translate",
            "target": "推荐智能体", "alternatives": [
                "推荐代理", "推荐智能体", "recommendation agent", "第三个备选",
            ],
            "rationale": (
                "It ranks products from customer signals, so agent denotes an "
                "AI task performer rather than a proxy."
            ),
        }, {
            "source": "absent agent", "strategy": "translate",
            "target": "不存在的智能体", "rationale": "The source does not contain it.",
        }], cues)

        contextual = next(
            entry for entry in entries if entry.source == "recommendation agent"
        )
        self.assertEqual(contextual.alternatives, ["推荐代理", "第三个备选"])
        self.assertIn("ranks products", contextual.rationale)
        self.assertNotIn("absent agent", {entry.source for entry in entries})

        restored = NaturalSubtitleTranslator._parse_terminology(
            [asdict(contextual)], cues,
        )
        cached = next(
            entry for entry in restored if entry.source == "recommendation agent"
        )
        self.assertEqual(cached.target, "推荐智能体")
        self.assertEqual(cached.alternatives, ["推荐代理", "第三个备选"])
        self.assertEqual(cached.rationale, contextual.rationale)

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
            downloader_args = command[command.index("--downloader-args") + 1]
            self.assertIn("-reconnect 1", downloader_args)
            self.assertIn("-reconnect_on_network_error 1", downloader_args)

    def test_remote_interval_retries_when_audio_ends_before_video(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = Workspace(root / "workspace")
            workspace.initialize()
            job = root / "job"
            job.mkdir()
            media = job / "clip123.mkv"
            attempts = [0]
            download_windows = []

            def download(*args, **kwargs):
                attempts[0] += 1
                download_windows.append(kwargs.get("download_window"))
                media.write_bytes(f"attempt-{attempts[0]}".encode())
                return media

            incomplete = VideoProbe(
                media, 180, 1920, 1080, "h264", "yuv420p", "aac",
                audio_duration=85,
            )
            complete = VideoProbe(
                media, 600, 1920, 1080, "h264", "yuv420p", "aac",
                audio_duration=600,
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
                _, media_info, _, download_window = acquirer.acquire_remote_media(
                    candidate, {"id": "clip123", "duration": 600},
                    candidate.source_url, job, source_range=SourceRange(100, 276),
                )

            self.assertEqual(attempts[0], 2)
            self.assertEqual(download_windows, [(98.0, 278.0), None])
            self.assertIsNone(download_window)
            self.assertEqual(media_info.duration, 600)

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

        planned_with_download = {
            "original_start": 1142.559, "original_end": 1246.32,
            "download_start": 1141.0, "download_end": 1248.0,
            "media_duration": 107.007, "rebased": False,
        }
        explicit_window, explicit_is_bounded = _local_interview_media_window(
            planned_with_download, 107.007, 3864.0,
        )
        self.assertTrue(explicit_is_bounded)
        self.assertEqual(explicit_window["download_start"], 1141.0)
        self.assertEqual(explicit_window["download_end"], 1248.0)

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

        def select(self, acquired_metadata, acquired_cues, editorial_mode, **kwargs):
            events.append("highlight_selected")
            acquired_cues[:] = [cue for cue in acquired_cues if cue.end > 100 and cue.start < 190]
            return [], dict(plan), []

        def audit(self, editorial_plan, acquired_cues, duration):
            events.append("directing_audited")
            return {"step": "interview_directing_audit", "provenance": {"provider": "test"}}

        def discover(self, acquired_cues, terminology):
            events.append("terminology_discovered")
            return {"step": "selected_subtitle_terminology_discovery", "added_sources": []}

        def joint_translate(
            self, acquired_cues, terminology, source_words,
            alignment_fingerprint="", audio_hypothesis="",
        ):
            events.append("joint_translated")
            for cue in acquired_cues:
                cue.translation = "工程系统改变团队工作流。"
            return {
                "step": "interview_joint_boundary_translation",
                "policy_version": INTERVIEW_CAPTION_POLICY_VERSION,
                "policy_fingerprint": INTERVIEW_CAPTION_POLICY_FINGERPRINT,
                "reviewed_cue_ids": [cue.id for cue in acquired_cues],
            }

        with TemporaryDirectory() as temp, patch(
            "video_factory.youtube.YouTubeAcquirer", FakeAcquirer,
        ), patch.object(
            NaturalSubtitleTranslator, "translate", select,
        ), patch.object(
            NaturalSubtitleTranslator, "audit_interview_directing", audit,
        ), patch.object(
            NaturalSubtitleTranslator, "discover_missing_terminology", discover,
        ), patch.object(
            NaturalSubtitleTranslator, "translate_interview_clip_once", joint_translate,
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
            "metadata_transcript", "highlight_selected", "terminology_discovered", "partial_download",
            "joint_translated", "directing_audited",
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

    def test_bounded_local_interview_media_requires_source_clip_provenance(self) -> None:
        metadata = {
            "id": "interview123", "title": "Known engineer interview",
            "channel": "AI Engineer", "description": "AI engineering interview",
            "duration": 600, "creators": ["Engineer", "Host"],
        }
        cues = [TranscriptCue(
            "c1", 100, 190,
            "A concrete engineering system changes the team workflow.",
        )]
        plan = {
            "editorial_mode": "known_tech_interview_clip", "collection_title": "人物高光",
            "story_start": 100, "story_end": 190, "bilibili_chapters": [],
            "wechat_lessons": [{
                "speaker_label": "Engineer", "title": "工程系统改变协作",
                "thesis": "一个完整工程判断。", "start": 100, "end": 190,
                "framing": "speaker", "hook_headlines": [
                    "工程系统改变协作", "完整判断来自上下文", "团队工作流发生变化",
                ],
            }],
        }

        class FakeAcquirer:
            def __init__(self, workspace):
                self.workspace = workspace

            def acquire(self, url, job, **kwargs):
                candidate = Candidate(
                    "youtube-interview123", SourceType.YOUTUBE, url, metadata["title"],
                    author=metadata["channel"], metadata={"video_id": metadata["id"]},
                )
                info = SourceMediaInfo(1920, 1080, 94, "h264", "aac")
                return candidate, [], dict(metadata), list(cues), "clip.mkv", "subtitles.json3", info

        def select(self, acquired_metadata, acquired_cues, editorial_mode, **kwargs):
            return [], dict(plan), []

        with TemporaryDirectory() as temp, patch(
            "video_factory.youtube.YouTubeAcquirer", FakeAcquirer,
        ), patch.object(
            NaturalSubtitleTranslator, "translate", select,
        ), patch.object(
            NaturalSubtitleTranslator, "discover_missing_terminology",
            return_value={"step": "selected_subtitle_terminology_discovery", "added_sources": []},
        ):
            workspace = Workspace(Path(temp))
            workspace.initialize()
            job = Path(temp) / "job"
            job.mkdir()
            with self.assertRaisesRegex(ValueError, "source_clip provenance"):
                YouTubeCollectionFactory(workspace, object()).generate(
                    "https://youtube.com/watch?v=interview123", job,
                    render=True, editorial_mode="known_tech_interview_clip",
                )

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

        def select(self, acquired_metadata, acquired_cues, editorial_mode, **kwargs):
            return [], dict(plan), []

        with TemporaryDirectory() as temp, patch(
            "video_factory.youtube.YouTubeAcquirer", FakeAcquirer,
        ), patch.object(
            NaturalSubtitleTranslator, "translate", select,
        ), patch.object(
            NaturalSubtitleTranslator, "discover_missing_terminology",
            return_value={"step": "terminology_discovered", "additions": []},
        ), patch.object(
            NaturalSubtitleTranslator, "review_terminology_decisions", return_value=None,
        ), patch.object(
            NaturalSubtitleTranslator, "translate_caption_scopes",
            return_value={"step": "joint_caption_scopes_translation"},
        ), patch(
            "video_factory.youtube.interview_caption_duration_errors", return_value=[],
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





    def test_business_model_uses_contextual_chinese_term(self) -> None:
        cues = [TranscriptCue(
            "cue", 0, 4,
            "As CEO, what is the right business model?",
            "作为 CEO，什么才是正确的商业模式？",
        )]

        errors = terminology_contract_errors(cues, [
            TerminologyEntry("model", TerminologyStrategy.TRANSLATE, target="模型"),
            TerminologyEntry(
                "business model", TerminologyStrategy.TRANSLATE,
                target="商业模式", rationale="CEO asks how the company makes money.",
            ),
        ])

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


    def test_open_source_check_accepts_natural_contextual_chinese(self) -> None:
        cues = [TranscriptCue(
            "cue", 0, 5,
            "There was an open-source check on closed source.",
            "这体现了开源对闭源的制衡。",
        )]

        errors = terminology_contract_errors(cues, [TerminologyEntry(
            "open-source check", TerminologyStrategy.TRANSLATE,
            target="开源对闭源的制衡",
            rationale="The source explicitly contrasts open and closed source.",
        )])

        self.assertEqual(errors, [])












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


    def test_exact_editorial_range_preserves_millisecond_boundaries(self) -> None:
        exact = _requested_exact_range_from_editorial_guidance(
            "Select exactly one answer from 19:02.559 to 20:46.320.",
        )

        self.assertEqual(exact, (1142.559, 1246.32))

    def test_caption_policy_rejects_subsecond_and_dense_cards(self) -> None:
        cue = TranscriptCue(
            "dense", 0, 0.9,
            "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four twenty-five twenty-six twenty-seven twenty-eight twenty-nine.",
            "这是一条明显过于密集而且来不及阅读的中文字幕，任何观众都无法在不到一秒内读完。",
        )

        errors = interview_caption_duration_errors([cue])

        self.assertTrue(any("minimum" in error for error in errors), errors)
        self.assertTrue(any("English words" in error for error in errors), errors)
        self.assertTrue(any("reading units per second" in error for error in errors), errors)

    def test_caption_reading_speed_counts_latin_names_as_words(self) -> None:
        cue = TranscriptCue(
            "named-entity", 84.44, 86.64,
            "And so I think you know companies like PaloAlto Networks",
            "所以我认为像 Palo Alto Networks 这样的公司",
        )

        self.assertEqual(interview_caption_duration_errors([cue]), [])

    def test_caption_policy_allows_brief_acknowledgement_under_standard_minimum(self) -> None:
        cue = TranscriptCue("ack", 0, 1.04, "Yeah.", "对。")

        self.assertEqual(interview_caption_duration_errors([cue]), [])

    def test_cached_semantic_cards_are_immutable_during_rerender(self) -> None:
        cues = [TranscriptCue(
            "cue-1-card-1", 0, 4.5, "The application layer matters.", "应用层很重要。",
        )]
        trace = [{
            "step": "interview_source_word_alignment",
            "fingerprint": "aligned-media-and-transcript",
            "policy_version": ALIGNMENT_POLICY_VERSION,
            "source_ledger_fingerprint": "source-ledger",
            "media_sha256": "media-sha",
            "status": "aligned",
        }, {
            "step": "interview_joint_boundary_translation",
            "cue_ids": [cues[0].id],
            "reviewed_cue_ids": [cues[0].id],
            "policy_version": INTERVIEW_CAPTION_POLICY_VERSION,
            "policy_fingerprint": INTERVIEW_CAPTION_POLICY_FINGERPRINT,
            "caption_content_fingerprint": _interview_caption_content_fingerprint(cues),
            "terminology_decision_fingerprint": _terminology_decision_fingerprint([]),
            "strict_source_fingerprint": _strict_interview_source_fingerprint(cues),
            "alignment_fingerprint": "aligned-media-and-transcript",
            "source_ledger_fingerprint": "source-ledger",
        }]

        self.assertTrue(cached_interview_caption_pipeline_complete(
            cues, trace, "media-sha",
        ))
        self.assertFalse(cached_interview_caption_pipeline_complete(cues, []))

        stale = [trace[0], {**trace[1], "policy_version": "legacy"}]
        self.assertFalse(cached_interview_caption_pipeline_complete(cues, stale))

        edited = [TranscriptCue(
            cues[0].id, cues[0].start, cues[0].end,
            cues[0].source_text, "应用层很关键。",
        )]
        self.assertFalse(cached_interview_caption_pipeline_complete(
            edited, trace, "media-sha",
        ))
        self.assertFalse(cached_interview_caption_pipeline_complete(
            cues, trace, "different-media",
        ))

        mixed = [*cues, TranscriptCue(
            "cue-2", 4.5, 14.5,
            "This stale parent remains much too long to publish safely.",
            "这个过期的父级字幕仍然太长，不能安全发布。",
        )]
        self.assertFalse(cached_interview_caption_pipeline_complete(mixed, trace))

    def test_cached_semantic_cards_are_bound_to_reviewed_terminology_decisions(self) -> None:
        cues = [TranscriptCue(
            "cue-1-card-1", 0, 4.5,
            "Open weights improve auditability.", "开放权重提高了可审计性。",
        )]
        reviewed_terms = [TerminologyEntry(
            "open weights", TerminologyStrategy.TRANSLATE,
            target="开放权重", alternatives=["公开权重"],
            rationale="The speaker means downloadable model weights.",
        )]
        alignment = {
            "step": "interview_source_word_alignment",
            "fingerprint": "aligned-media-and-transcript",
            "policy_version": ALIGNMENT_POLICY_VERSION,
            "source_ledger_fingerprint": "source-ledger",
            "media_sha256": "media-sha",
            "status": "aligned",
        }
        review = {
            "step": "interview_joint_boundary_translation",
            "reviewed_cue_ids": [cues[0].id],
            "policy_version": INTERVIEW_CAPTION_POLICY_VERSION,
            "policy_fingerprint": INTERVIEW_CAPTION_POLICY_FINGERPRINT,
            "caption_content_fingerprint": _interview_caption_content_fingerprint(cues),
            "terminology_decision_fingerprint": _terminology_decision_fingerprint(
                reviewed_terms,
            ),
            "strict_source_fingerprint": _strict_interview_source_fingerprint(cues),
            "alignment_fingerprint": "aligned-media-and-transcript",
            "source_ledger_fingerprint": "source-ledger",
        }

        self.assertTrue(cached_interview_caption_pipeline_complete(
            cues, [alignment, review], "media-sha", reviewed_terms,
        ))

        changed_terms = [TerminologyEntry(
            "open weights", TerminologyStrategy.TRANSLATE,
            target="开放权重", alternatives=["公开参数"],
            rationale="A different decision the reviewer did not inspect.",
        )]
        self.assertFalse(cached_interview_caption_pipeline_complete(
            cues, [alignment, review], "media-sha", changed_terms,
        ))

    def test_non_interview_joint_cache_is_bound_to_cards_and_terminology(self) -> None:
        cues = [TranscriptCue(
            "caption-scope-001-card-0001", 0, 4,
            "Open weights improve audits.", "开放权重让审计更容易。",
        )]
        terms = [TerminologyEntry(
            "Open weights", TerminologyStrategy.TRANSLATE,
            target="开放权重", rationale="The source discusses downloadable weights.",
        )]
        trace = [{
            "step": "joint_caption_scopes_translation",
            "policy_version": INTERVIEW_CAPTION_POLICY_VERSION,
            "policy_fingerprint": INTERVIEW_CAPTION_POLICY_FINGERPRINT,
            "reviewed_cue_ids": [cues[0].id],
            "caption_content_fingerprint": _interview_caption_content_fingerprint(cues),
            "terminology_decision_fingerprint": _terminology_decision_fingerprint(terms),
            "strict_source_fingerprint": _strict_interview_source_fingerprint(cues),
        }]

        self.assertTrue(cached_joint_caption_pipeline_complete(cues, trace, terms))
        changed = [TranscriptCue(
            cues[0].id, 0, 4, cues[0].source_text, "开放参数让审计更容易。",
        )]
        self.assertFalse(cached_joint_caption_pipeline_complete(changed, trace, terms))







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








    def test_modality_wording_is_left_to_independent_semantic_review(self) -> None:
        errors = _semantic_card_translation_errors({
            "id": "card",
            "source": "maybe three or four leading companies were doing it.",
            "duration_seconds": 4.0,
        }, "三四家头部公司都在这样做。", [])

        self.assertNotIn("modality", errors)

    def test_semantic_card_reports_which_translation_term_rule_failed(self) -> None:
        term = TerminologyEntry(
            "retrieval agent", TerminologyStrategy.TRANSLATE, target="检索代理",
        )
        row = {
            "id": "card", "source": "A retrieval agent answers the request.",
            "duration_seconds": 5.0,
        }

        self.assertIn(
            "term:retrieval agent:missing_target:检索代理",
            _semantic_card_translation_errors(row, "它会回答请求。", [term]),
        )
        self.assertIn(
            "term:retrieval agent:remove_english:retrieval agent",
            _semantic_card_translation_errors(
                row, "检索代理 retrieval agent 会回答请求。", [term],
            ),
        )

    def test_semantic_card_accepts_equivalent_term_spacing_and_identity(self) -> None:
        row = {
            "id": "card", "source": "Token usage includes dark tokens.",
            "duration_seconds": 5.0,
        }
        terms = [
            TerminologyEntry(
                "token", TerminologyStrategy.TRANSLATE, target="token",
            ),
            TerminologyEntry(
                "dark tokens", TerminologyStrategy.TRANSLATE, target="暗token",
            ),
        ]

        errors = _semantic_card_translation_errors(
            row, "token 用量包含暗 token。", terms,
        )

        self.assertFalse(any(error.startswith("term:") for error in errors))





    def test_card_term_does_not_enforce_agent_inside_retrieval_agent(self) -> None:
        terms = [
            TerminologyEntry("agent", TerminologyStrategy.TRANSLATE, target="智能体"),
            TerminologyEntry(
                "retrieval agent", TerminologyStrategy.TRANSLATE, target="检索代理",
                alternatives=["检索智能体"],
                rationale="It retrieves information as an AI workflow component.",
            ),
        ]
        row = {
            "id": "card", "source": "The retrieval agent reads the request.",
            "duration_seconds": 5,
        }
        self.assertEqual(
            _semantic_card_translation_errors(row, "检索代理会读取请求。", terms),
            [],
        )
        self.assertEqual(
            _semantic_card_translation_errors(row, "检索智能体会读取请求。", terms),
            [],
        )



    def test_contextual_candidates_are_not_a_deterministic_publication_whitelist(self) -> None:
        cues = [
            TranscriptCue(
                "c1", 0, 4, "A recommendation agent ranks products.",
                "推荐智能体会给产品排序。",
            ),
            TranscriptCue(
                "c2", 4, 8, "The recommendation agent updates the list.",
                "推荐代理会更新列表。",
            ),
        ]
        terminology = [TerminologyEntry(
            "recommendation agent", TerminologyStrategy.TRANSLATE,
            target="推荐智能体", alternatives=["推荐代理"],
            rationale="Both sentences describe one ranking component.",
        )]

        before = [cue.translation for cue in cues]
        enforced = NaturalSubtitleTranslator._enforce_terminology_contract(
            cues, terminology,
        )
        errors = terminology_contract_errors(cues, terminology)

        self.assertEqual(enforced, [])
        self.assertEqual([cue.translation for cue in cues], before)
        self.assertTrue(any("recommendation agent:c2" in error for error in errors))


    def test_preserve_decision_is_reviewed_and_revised_before_translation(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.calls = 0

            def _request_json(self, messages, max_tokens):
                self.calls += 1
                prompt = messages[-1]["content"]
                self.assert_prompt(prompt)
                return ({"terminology": [{
                    "source": "openweight",
                    "strategy": "translate",
                    "target": "开放权重",
                    "alternatives": [],
                    "rationale": "此处与开源并列，特指权重公开。",
                }]}, {"model": "translation-writer"})

            @staticmethod
            def assert_prompt(prompt):
                if "Recent or emerging terminology does not automatically stay in English" not in prompt:
                    raise AssertionError("revision prompt retained the planner's preserve bias")

        class Reviewer:
            def __init__(self) -> None:
                self.calls = 0

            def _request_json(self, messages, max_tokens):
                self.calls += 1
                prompt = messages[-1]["content"]
                terms, _ = json.JSONDecoder().raw_decode(
                    prompt.split("Terms: ", 1)[1]
                )
                term = terms[0]
                preserved = term["strategy"] == "preserve"
                return ({"reviews": [{
                    "source": "openweight",
                    "pass": not preserved,
                    "fidelity_score": 2 if preserved else 5,
                    "naturalness_score": 2 if preserved else 5,
                    "errors": (
                        ["开放权重 is clear here; emerging is insufficient to preserve English"]
                        if preserved else []
                    ),
                }]}, {"model": "independent-reviewer", "call": self.calls})

        cues = [TranscriptCue(
            "cue-1", 0, 5,
            "Open-source openweight models can run locally.", "",
        )]
        terminology = [TerminologyEntry(
            "openweight", TerminologyStrategy.PRESERVE,
            target="openweight",
            rationale="An emerging term without stable Chinese wording.",
        )]
        writer = Writer()
        reviewer = Reviewer()

        trace = NaturalSubtitleTranslator(
            writer, None, reviewer,
        ).review_terminology_decisions(cues, terminology)

        self.assertEqual(writer.calls, 1)
        self.assertEqual(reviewer.calls, 2)
        self.assertEqual(trace["attempt"], 2)
        self.assertEqual(terminology[0].strategy, TerminologyStrategy.TRANSLATE)
        self.assertEqual(terminology[0].target, "开放权重")
        self.assertEqual(trace["decisions"][0]["target"], "开放权重")

    def test_selected_subtitle_discovery_adds_unseen_contextual_term_only(self) -> None:
        class Writer:
            def _request_json(self, messages, max_tokens):
                prompt = messages[-1]["content"]
                self.prompt = prompt
                return {"terminology": [{
                    "source": "recommendation agent",
                    "strategy": "translate",
                    "target": "推荐智能体",
                    "alternatives": ["推荐代理"],
                    "rationale": "It ranks products from user signals as an AI task performer.",
                }]}, {"model": "writer"}

        cues = [TranscriptCue(
            "cue-1", 0, 5,
            "The recommendation agent ranks products from user signals.", "",
        )]
        terminology = [TerminologyEntry(
            "agent", TerminologyStrategy.TRANSLATE, target="智能体",
        )]
        writer = Writer()

        trace = NaturalSubtitleTranslator(writer).discover_missing_terminology(
            cues, terminology,
        )

        self.assertEqual(trace["added_sources"], ["recommendation agent"])
        self.assertEqual(terminology[-1].target, "推荐智能体")
        self.assertIn("Do not select a clip", writer.prompt)

    def test_selected_subtitle_discovery_retries_sources_outside_fixed_passage(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.calls = 0

            def _request_json(self, messages, max_tokens):
                self.calls += 1
                if self.calls == 1:
                    return {"terminology": [{
                        "source": "missing phrase", "strategy": "preserve",
                        "target": "", "alternatives": [], "rationale": "not present",
                    }]}, {"call": 1}
                return {"terminology": []}, {"call": 2}

        writer = Writer()
        terminology: list[TerminologyEntry] = []
        trace = NaturalSubtitleTranslator(writer).discover_missing_terminology(
            [TranscriptCue("cue-1", 0, 4, "The model runs locally.", "")],
            terminology,
        )

        self.assertEqual(writer.calls, 2)
        self.assertEqual(trace["added_sources"], [])
        self.assertEqual(terminology, [])

    def test_selected_subtitle_discovery_discards_repeated_invalid_optional_proposals(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.calls = 0

            def _request_json(self, messages, max_tokens):
                self.calls += 1
                return {"terminology": [{
                    "source": "invented normalized phrase",
                    "strategy": "translate", "target": "虚构短语",
                    "alternatives": [], "rationale": "not in the fixed passage",
                }]}, {"call": self.calls}

        writer = Writer()
        terminology: list[TerminologyEntry] = []
        trace = NaturalSubtitleTranslator(writer).discover_missing_terminology(
            [TranscriptCue("cue-1", 0, 4, "The model runs locally.", "")],
            terminology,
        )

        self.assertEqual(writer.calls, 2)
        self.assertTrue(trace["discarded_invalid_proposals"])
        self.assertEqual(trace["added_sources"], [])
        self.assertEqual(len(trace["attempts"]), 2)
        self.assertEqual(terminology, [])

    def test_independent_terminology_review_can_drop_asr_corruption(self) -> None:
        class Writer:
            def _request_json(self, *args, **kwargs):
                raise AssertionError("dropped ASR corruption must not be revised")

        class Reviewer:
            def _request_json(self, messages, max_tokens):
                return {"reviews": [{
                    "source": "deconlict", "pass": False,
                    "fidelity_score": 1, "naturalness_score": 1,
                    "errors": ["drop:not_a_stable_term"],
                }]}, {"model": "independent-reviewer"}

        cues = [TranscriptCue(
            "cue-1", 0, 5, "These controls deconlict agent permissions.", "",
        )]
        terminology = [TerminologyEntry(
            "deconlict", TerminologyStrategy.TRANSLATE, target="消除冲突",
            rationale="The transcript uses it as an action.",
        )]

        trace = NaturalSubtitleTranslator(
            Writer(), subtitle_reviewer=Reviewer(),
        ).review_terminology_decisions(cues, terminology)

        self.assertEqual(trace["dropped_sources"], ["deconlict"])
        self.assertEqual(terminology, [])

    def test_repeatedly_rejected_optional_terminology_is_discarded(self) -> None:
        class Writer:
            def _request_json(self, messages, max_tokens):
                return {"terminology": [{
                    "source": "dark tokens", "strategy": "translate",
                    "target": "暗黑代币", "alternatives": [],
                    "rationale": "The speakers use it as a market label.",
                }]}, {"model": "writer"}

        class Reviewer:
            def _request_json(self, messages, max_tokens):
                return {"reviews": [{
                    "source": "dark tokens", "pass": False,
                    "fidelity_score": 2, "naturalness_score": 2,
                    "errors": ["target is ambiguous in this context"],
                }]}, {"model": "independent-reviewer"}

        cues = [TranscriptCue(
            "cue-1", 0, 5, "The market calls these dark tokens.", "",
        )]
        terminology = [TerminologyEntry(
            "dark tokens", TerminologyStrategy.TRANSLATE, target="暗色代币",
            rationale="The speakers use it as a market label.",
        )]

        trace = NaturalSubtitleTranslator(
            Writer(), subtitle_reviewer=Reviewer(),
        ).review_terminology_decisions(cues, terminology)

        self.assertEqual(trace["discarded_after_rejection"], ["dark tokens"])
        self.assertIn("dark tokens", trace["final_rejections"])
        self.assertEqual(terminology, [])

    def test_malformed_terminology_review_retries_without_revising_decision(self) -> None:
        class MustNotRevise:
            def _request_json(self, *args, **kwargs):
                raise AssertionError("review schema failure must not revise terminology")

        class Reviewer:
            def __init__(self) -> None:
                self.calls = 0
                self.prompts: list[str] = []

            def _request_json(self, messages, max_tokens):
                self.calls += 1
                prompt = messages[-1]["content"]
                self.prompts.append(prompt)
                if self.calls == 1:
                    return {"reviews": []}, {"model": "reviewer", "call": 1}
                return {"reviews": [{
                    "source": "recommendation agent", "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                }]}, {"model": "reviewer", "call": 2}

        cues = [TranscriptCue(
            "cue-1", 0, 5,
            "The recommendation agent ranks products from user signals.", "",
        )]
        terminology = [TerminologyEntry(
            "recommendation agent", TerminologyStrategy.TRANSLATE,
            target="推荐智能体",
            rationale="It ranks products from signals as an AI task performer.",
        )]
        reviewer = Reviewer()

        trace = NaturalSubtitleTranslator(
            MustNotRevise(), None, reviewer,
        ).review_terminology_decisions(cues, terminology)

        self.assertEqual(reviewer.calls, 2)
        self.assertEqual(terminology[0].target, "推荐智能体")
        self.assertEqual(len(trace["review_structure_failures"]), 1)
        self.assertIn(
            "previous reviewer response failed deterministic structure validation",
            reviewer.prompts[1],
        )







    def test_joint_interview_translation_chooses_boundaries_and_translates_once(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.calls = 0

            def _request_json(self, *args, **kwargs):
                self.calls += 1
                return {"cards": [
                    {"end_word": 10, "text": "这条中文字幕虽然已经超过旧的三十二字限制但在当前显示时间里仍然完全可以读完。"},
                    {"end_word": 20, "text": "第二张卡继续表达后半段意思。"},
                ]}, {"model": "writer"}

        class Reviewer:
            def __init__(self) -> None:
                self.calls = 0

            def _request_json(self, messages, **kwargs):
                self.calls += 1
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                } for row in rows]}, {"model": "reviewer"}

        source = (
            "one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
        )
        cues = [TranscriptCue("cue-1", 0, 12, source)]
        words = source_words_from_cues(cues)
        writer, reviewer = Writer(), Reviewer()

        trace = NaturalSubtitleTranslator(
            writer, subtitle_reviewer=reviewer,
        ).translate_interview_clip_once(cues, [], words)

        self.assertEqual(trace["translation_passes"], 1)
        self.assertEqual(trace["repair_rounds"], 0)
        self.assertEqual(writer.calls, 1)
        self.assertEqual(reviewer.calls, 1)
        self.assertEqual(" ".join(cue.source_text for cue in cues), source)
        self.assertEqual(sum(len(cue.source_tokens) for cue in cues), 20)
        self.assertGreater(len(re.sub(r"\s+", "", cues[0].translation)), 32)

    def test_joint_interview_translation_repairs_only_failed_window(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.calls = 0

            def _request_json(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    cards = [
                        {"end_word": 10, "text": "这" * 80 + "。"},
                        {"end_word": 20, "text": "后半段保持事实并顺畅收尾。"},
                    ]
                else:
                    cards = [
                        {"end_word": 7, "text": "前段经过局部修复。"},
                        {"end_word": 14, "text": "中段可以正常阅读。"},
                        {"end_word": 20, "text": "后段顺畅收尾。"},
                    ]
                return {"cards": cards}, {"model": "writer", "call": self.calls}

        class Reviewer:
            def __init__(self) -> None:
                self.calls = 0

            def _request_json(self, messages, **kwargs):
                self.calls += 1
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                } for row in rows]}, {"model": "reviewer"}

        source = " ".join(f"word{index}" for index in range(1, 21))
        cues = [TranscriptCue("cue-1", 0, 12, source)]
        words = source_words_from_cues(cues)
        writer = Writer()
        reviewer = Reviewer()

        trace = NaturalSubtitleTranslator(
            writer, subtitle_reviewer=reviewer,
        ).translate_interview_clip_once(cues, [], words)

        self.assertEqual(writer.calls, 2)
        self.assertEqual(reviewer.calls, 1)
        self.assertEqual(trace["repair_rounds"], 1)
        self.assertEqual(len(cues), 3)
        self.assertNotIn("这" * 40, cues[0].translation)

    def test_joint_translation_retries_malformed_review_without_retranslating(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.calls = 0

            def _request_json(self, *args, **kwargs):
                self.calls += 1
                return {"cards": [
                    {"end_word": 10, "text": "前半段忠实表达原意。"},
                    {"end_word": 20, "text": "后半段继续完整表达。"},
                ]}, {"model": "writer"}

        class Reviewer:
            def __init__(self) -> None:
                self.calls = 0
                self.prompts: list[str] = []

            def _request_json(self, messages, **kwargs):
                self.calls += 1
                prompt = messages[-1]["content"]
                self.prompts.append(prompt)
                rows = json.loads(prompt.split("Rows: ", 1)[1])
                if self.calls == 1:
                    rows = rows[:1]
                return {"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                } for row in rows]}, {"model": "reviewer", "call": self.calls}

        source = " ".join(f"word{index}" for index in range(1, 21))
        cues = [TranscriptCue("cue-1", 0, 12, source)]
        writer, reviewer = Writer(), Reviewer()

        trace = NaturalSubtitleTranslator(
            writer, subtitle_reviewer=reviewer,
        ).translate_interview_clip_once(
            cues, [], source_words_from_cues(cues),
        )

        self.assertEqual(writer.calls, 1)
        self.assertEqual(reviewer.calls, 2)
        self.assertEqual(trace["repair_rounds"], 0)
        self.assertIn(
            "previous reviewer response failed deterministic structure validation",
            reviewer.prompts[1],
        )

    def test_joint_repair_requires_an_extra_card_after_hard_duration_failure(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                self.prompts.append(prompt)
                ends = (
                    [8, 25, 33] if len(self.prompts) == 1
                    else [8, 25, 29, 33] if len(self.prompts) == 2
                    else [8, 16, 24, 33]
                )
                return {"cards": [
                    {"end_word": end, "text": "这段内容。"}
                    for index, end in enumerate(ends, start=1)
                ]}, {"model": "writer"}

        class Reviewer:
            def __init__(self) -> None:
                self.local_ids: list[str] = []
                self.global_calls = 0

            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                if "Sequence: " in prompt:
                    self.global_calls += 1
                    return {"pass": True, "issues": []}, {"model": "reviewer-global"}
                rows = json.loads(prompt.split("Rows: ", 1)[1])
                self.local_ids.extend(row["id"] for row in rows)
                return {"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                } for row in rows]}, {"model": "reviewer"}

        source = " ".join(f"word{index}" for index in range(1, 34))
        cues = [TranscriptCue("cue-1", 0, 18, source)]
        writer = Writer()

        trace = NaturalSubtitleTranslator(
            writer, subtitle_reviewer=Reviewer(),
        ).translate_interview_clip_once(cues, [], source_words_from_cues(cues))

        self.assertEqual(len(writer.prompts), 3)
        self.assertIn("Return 4–", writer.prompts[1])
        self.assertIn("Deterministic boundary failure", writer.prompts[2])
        self.assertEqual(len(cues), 4)
        self.assertEqual(trace["repair_rounds"], 1)
        self.assertFalse(interview_caption_duration_errors(cues))

    def test_joint_repair_fixes_boundary_after_repeated_unsplit_responses(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.prompts: list[str] = []
                self.fixed_attempts = 0

            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                self.prompts.append(prompt)
                fixed = json.loads(
                    prompt.split("Fixed required end_word boundaries: ", 1)[1]
                    .split(". Include every listed value", 1)[0]
                )
                ends = [8, 25, 33] if len(self.prompts) == 1 else [8, 25, 29, 33]
                if fixed:
                    self.fixed_attempts += 1
                    if self.fixed_attempts > 1:
                        ends = sorted(set([8, *fixed, 25, 29, 33]))
                return {"cards": [
                    {"end_word": end, "text": "这段内容完整表达。"}
                    for index, end in enumerate(ends, start=1)
                ]}, {"model": "writer", "call": len(self.prompts)}

        class Reviewer:
            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                if "Sequence: " in prompt:
                    return {"pass": True, "issues": []}, {"model": "reviewer-global"}
                rows = json.loads(prompt.split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                } for row in rows]}, {"model": "reviewer"}

        source = " ".join(f"word{index}" for index in range(1, 34))
        cues = [TranscriptCue("cue-1", 0, 18, source)]
        writer = Writer()

        trace = NaturalSubtitleTranslator(
            writer, subtitle_reviewer=Reviewer(),
        ).translate_interview_clip_once(cues, [], source_words_from_cues(cues))

        self.assertEqual(len(writer.prompts), 6)
        self.assertIn("Deterministic self-healing selected fixed internal boundaries", writer.prompts[-1])
        self.assertGreaterEqual(len(cues), 5)
        self.assertEqual(trace["repair_rounds"], 1)
        self.assertFalse(interview_caption_duration_errors(cues))

    def test_initial_joint_translation_recovers_only_an_omitted_tail(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.window_sizes: list[int] = []

            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                size = int(re.search(
                    r"exact source window contains (\d+) words", prompt,
                ).group(1))
                self.window_sizes.append(size)
                return {"cards": [{
                    "end_word": 4,
                    "text": "前半句。" if size == 8 else "后半句。",
                }]}, {"model": "writer", "window_size": size}

        class Reviewer:
            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                if "Sequence: " in prompt:
                    return {"pass": True, "issues": []}, {"model": "global"}
                rows = json.loads(prompt.split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                } for row in rows]}, {"model": "reviewer"}

        source = "one two three four five six seven eight"
        cues = [TranscriptCue("cue-1", 0, 6, source)]
        writer = Writer()

        trace = NaturalSubtitleTranslator(
            writer, subtitle_reviewer=Reviewer(),
        ).translate_interview_clip_once(cues, [], source_words_from_cues(cues))

        self.assertEqual(writer.window_sizes, [8, 4])
        self.assertEqual(" ".join(cue.source_text for cue in cues), source)
        recovery = trace["attempts"][0]["windows"][0]["provenance"]
        self.assertEqual(recovery["partial_coverage_recovery"]["covered_words"], 4)

    def test_joint_repair_freezes_boundaries_after_repeated_semantic_drift(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.calls = 0
                self.fixed = False
                self.prompts: list[str] = []

            def _request_json(self, messages, **kwargs):
                self.calls += 1
                prompt = messages[-1]["content"]
                self.prompts.append(prompt)
                fixed = json.loads(
                    prompt.split("Fixed required end_word boundaries: ", 1)[1]
                    .split(". Include every listed value", 1)[0]
                )
                self.fixed = bool(fixed)
                return {"cards": [
                    {"end_word": 10, "text": "第一部分忠实表达。"},
                    {"end_word": 20, "text": "第二部分忠实表达。"},
                    {"end_word": 30, "text": "第三部分忠实表达。"},
                ]}, {"model": "writer", "call": self.calls}

        class Reviewer:
            def __init__(self, writer) -> None:
                self.writer = writer

            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                if "Sequence: " in prompt:
                    return {"pass": True, "issues": []}, {"model": "reviewer-global"}
                rows = json.loads(prompt.split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": row["id"], "pass": self.writer.fixed or index != 1,
                    "fidelity_score": 5 if self.writer.fixed or index != 1 else 2,
                    "naturalness_score": 5 if self.writer.fixed or index != 1 else 3,
                    "errors": [] if self.writer.fixed or index != 1 else [
                        "跨卡边界语义错位：中文提前使用了下一卡的含义。",
                    ],
                } for index, row in enumerate(rows)]}, {"model": "reviewer"}

        source = " ".join(f"word{index}" for index in range(1, 31))
        cues = [TranscriptCue("cue-1", 0, 18, source)]
        writer = Writer()

        trace = NaturalSubtitleTranslator(
            writer, subtitle_reviewer=Reviewer(writer),
        ).translate_interview_clip_once(cues, [], source_words_from_cues(cues))

        self.assertTrue(writer.fixed)
        self.assertEqual(writer.calls, 5)
        self.assertIn("Fixed card translation rows: [", writer.prompts[-1])
        self.assertIn(
            "translate each row's source only into its matching Chinese text",
            writer.prompts[-1],
        )
        self.assertEqual(trace["repair_rounds"], 3)
        self.assertEqual(trace["attempts"][-1]["kind"], "validation_and_review")
        self.assertTrue(any(
            item["kind"] == "fixed_boundary_semantic_repair"
            for item in trace["attempts"]
        ))

    def test_final_independent_adjudication_can_dismiss_nonmaterial_rejection(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.calls = 0

            def _request_json(self, messages, **kwargs):
                self.calls += 1
                return {"cards": [
                    {"end_word": 10, "text": "第一部分忠实表达。"},
                    {"end_word": 20, "text": "第二部分继续说明。"},
                    {"end_word": 30, "text": "第三部分完成论点。"},
                ]}, {"model": "writer", "call": self.calls}

        class Reviewer:
            def __init__(self) -> None:
                self.adjudicated = False
                self.adjudication_prompt = ""

            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                if "Prior rejected rows: " in prompt:
                    self.adjudicated = True
                    self.adjudication_prompt = prompt
                    rows = json.loads(prompt.split("Prior rejected rows: ", 1)[1])
                    return {"reviews": [{
                        "id": row["id"], "pass": True,
                        "fidelity_score": 5, "naturalness_score": 5,
                        "errors": [],
                    } for row in rows]}, {"model": "adjudicator"}
                if "Sequence: " in prompt:
                    return {"pass": True, "issues": []}, {"model": "reviewer-global"}
                rows = json.loads(prompt.split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": row["id"], "pass": index != 1,
                    "fidelity_score": 5 if index != 1 else 3,
                    "naturalness_score": 5 if index != 1 else 3,
                    "errors": [] if index != 1 else [
                        "可能需要增加一个可选连接词。",
                    ],
                } for index, row in enumerate(rows)]}, {"model": "reviewer"}

        source = " ".join(f"word{index}" for index in range(1, 31))
        cues = [TranscriptCue("cue-1", 0, 18, source)]
        writer = Writer()
        reviewer = Reviewer()

        trace = NaturalSubtitleTranslator(
            writer, subtitle_reviewer=reviewer,
        ).translate_interview_clip_once(cues, [], source_words_from_cues(cues))

        self.assertTrue(reviewer.adjudicated)
        self.assertIn("nearest noun", reviewer.adjudication_prompt)
        self.assertIn("earns revenue", reviewer.adjudication_prompt)
        self.assertEqual(writer.calls, 5)
        final_review = trace["attempts"][-1]["review_provenance"]
        self.assertEqual(
            final_review["final_rejection_adjudication"]["remaining_card_ids"],
            [],
        )

    def test_joint_interview_translation_retries_invalid_initial_structure(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.calls = 0
                self.prompts: list[str] = []

            def _request_json(self, messages, **kwargs):
                self.calls += 1
                self.prompts.append(messages[-1]["content"])
                if self.calls == 1:
                    return {"cards": []}, {"model": "writer", "call": 1}
                return {"cards": [
                    {"end_word": 10, "text": "前半段保留原文事实。"},
                    {"end_word": 20, "text": "后半段完成这个说明。"},
                ]}, {"model": "writer", "call": 2}

        class Reviewer:
            def _request_json(self, messages, **kwargs):
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                } for row in rows]}, {"model": "reviewer"}

        source = " ".join(f"word{index}" for index in range(1, 21))
        cues = [TranscriptCue("cue-1", 0, 12, source)]
        writer = Writer()

        trace = NaturalSubtitleTranslator(
            writer, subtitle_reviewer=Reviewer(),
        ).translate_interview_clip_once(cues, [], source_words_from_cues(cues))

        self.assertEqual(writer.calls, 2)
        self.assertEqual(trace["translation_passes"], 1)
        self.assertEqual(trace["attempts"][0]["window_count"], 1)
        self.assertEqual(
            trace["attempts"][0]["windows"][0]["request_attempts"], 2,
        )
        self.assertEqual(len(trace["attempts"][0]["earlier_structure_failures"]), 1)
        self.assertIn("Previous rejection (mandatory:", writer.prompts[1])
        self.assertEqual(" ".join(cue.source_text for cue in cues), source)

    def test_joint_interview_translation_partitions_large_initial_request(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.window_sizes: list[int] = []

            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                rows = json.loads(
                    prompt.split("Source words: ", 1)[1].split("\nNext source:", 1)[0]
                )
                self.window_sizes.append(len(rows))
                ends = list(range(10, len(rows), 10)) + [len(rows)]
                return {"cards": [
                    {"end_word": end, "text": "这段内容。"}
                    for index, end in enumerate(ends, start=1)
                ]}, {"model": "writer"}

        class Reviewer:
            def __init__(self) -> None:
                self.local_ids: list[str] = []
                self.global_calls = 0

            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                if "Sequence: " in prompt:
                    self.global_calls += 1
                    return {"pass": True, "issues": []}, {"model": "reviewer-global"}
                rows = json.loads(prompt.split("Rows: ", 1)[1])
                self.local_ids.extend(row["id"] for row in rows)
                return {"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                } for row in rows]}, {"model": "reviewer"}

        source = " ".join(f"word{index}" for index in range(1, 302))
        cues = [TranscriptCue("cue-1", 0, 181, source)]
        writer = Writer()
        reviewer = Reviewer()

        trace = NaturalSubtitleTranslator(
            writer, subtitle_reviewer=reviewer,
        ).translate_interview_clip_once(cues, [], source_words_from_cues(cues))

        self.assertEqual(len(writer.window_sizes), 3)
        self.assertTrue(all(size <= 150 for size in writer.window_sizes))
        self.assertEqual(trace["translation_passes"], 1)
        self.assertEqual(" ".join(cue.source_text for cue in cues), source)
        self.assertEqual(sorted(reviewer.local_ids), sorted(cue.id for cue in cues))
        self.assertEqual(len(reviewer.local_ids), len(set(reviewer.local_ids)))
        self.assertEqual(reviewer.global_calls, 1)

    def test_joint_caption_scopes_preserve_editorial_plan_and_cut_boundaries(self) -> None:
        class Writer:
            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                rows = json.loads(
                    prompt.split("Source words: ", 1)[1].split("\nNext source:", 1)[0]
                )
                return {"cards": [
                    {"end_word": len(rows) // 2, "text": "这一段先说明系统变化。"},
                    {"end_word": len(rows), "text": "随后说明团队如何响应。"},
                ]}, {"model": "writer"}

        class Reviewer:
            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                if "Sequence: " in prompt:
                    return {"pass": True, "issues": []}, {"model": "reviewer-global"}
                rows = json.loads(prompt.split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                } for row in rows]}, {"model": "reviewer"}

        cues = [
            TranscriptCue("a", 0, 10, "one two [laughter] three four five six seven eight nine ten"),
            TranscriptCue("b", 10, 20, "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"),
        ]
        plan = {
            "editorial_mode": "technical_coverage",
            "collection_title": "固定策划",
            "bilibili_chapters": [],
            "wechat_lessons": [
                {"title": "第一段", "start": 0, "end": 10,
                 "hook_headlines": ["甲", "乙", "丙"]},
                {"title": "第二段", "start": 10, "end": 20,
                 "hook_headlines": ["丁", "戊", "己"]},
            ],
        }
        original_plan = json.loads(json.dumps(plan, ensure_ascii=False))

        trace = NaturalSubtitleTranslator(
            Writer(), subtitle_reviewer=Reviewer(),
        ).translate_caption_scopes(
            cues, [], plan, 20, "technical_coverage",
        )

        self.assertEqual(plan, original_plan)
        self.assertEqual(trace["scope_count"], 2)
        self.assertTrue(trace["cross_scope_consistency_review"]["pass"])
        self.assertTrue(all(cue.end <= 10 or cue.start >= 10 for cue in cues))
        self.assertTrue(all(cue.id.startswith("caption-scope-") for cue in cues))
        self.assertNotIn("laughter", " ".join(cue.source_text for cue in cues))

    def test_subtitle_resegmentation_cannot_reselect_frozen_hook(self) -> None:
        plan = {
            "editorial_mode": "technical_coverage",
            "wechat_lessons": [{
                "title": "系统设计决定交付速度", "thesis": "团队必须自动验证改动。",
                "start": 0, "end": 20,
                "hook_headlines": [
                    "系统瓶颈不在模型", "自动验证决定交付速度", "工具链减少团队返工",
                ],
            }],
        }
        original = [
            TranscriptCue("raw-a", 0, 8, "The system must validate every code change."),
            TranscriptCue("raw-b", 10, 18, "Automation reduces expensive team rework."),
        ]
        snapshot = _snapshot_plan_hooks(plan, 20, original)
        before = snapshot["wechat:1"][0]
        final_cards = [
            TranscriptCue("card-1", 0, 4, "The system must validate", "系统必须验证。"),
            TranscriptCue("card-2", 4, 8, "every code change.", "每次代码改动。"),
            TranscriptCue("card-3", 10, 18, "Automation reduces expensive team rework.", "自动化减少返工。"),
        ]

        remapped = _remap_hook_snapshot(
            snapshot["wechat:1"], final_cards, "final-item",
        )

        self.assertEqual(remapped[0].headline_zh, before.headline_zh)
        self.assertEqual(remapped[0].source_range, before.source_range)
        self.assertNotEqual(remapped[0].source_cue_ids, before.source_cue_ids)
        self.assertTrue(set(remapped[0].source_cue_ids) <= {cue.id for cue in final_cards})

    def test_joint_interview_translation_omits_fillers_before_review(self) -> None:
        class Writer:
            def _request_json(self, *args, **kwargs):
                return {"cards": [{
                    "end_word": 8, "text": "呃，你知道，我是说，我们现在发布。",
                }]}, {"model": "writer"}

        class Reviewer:
            def __init__(self) -> None:
                self.chinese = ""

            def _request_json(self, messages, **kwargs):
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                self.chinese = rows[0]["chinese"]
                return {"reviews": [{
                    "id": rows[0]["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                }]}, {"model": "reviewer"}

        cues = [TranscriptCue(
            "cue-1", 0, 4, "Um, you know, I mean, we ship now.",
        )]
        reviewer = Reviewer()

        NaturalSubtitleTranslator(
            Writer(), subtitle_reviewer=reviewer,
        ).translate_interview_clip_once(cues, [], source_words_from_cues(cues))

        self.assertEqual(cues[0].translation, "我们现在发布。")
        self.assertEqual(reviewer.chinese, "我们现在发布。")

    def test_joint_translation_uses_audio_hypothesis_only_as_conflict_evidence(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.prompt = ""

            def _request_json(self, messages, **kwargs):
                self.prompt = messages[-1]["content"]
                return {"cards": [{
                    "end_word": 8, "text": "这些非常信奉 AI 的创始人会运行编码智能体。",
                }]}, {"model": "writer"}

        class Reviewer:
            def _request_json(self, messages, **kwargs):
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": rows[0]["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                }]}, {"model": "reviewer"}

        source = "These founders are super AID run coding agents."
        cues = [TranscriptCue("cue-1", 0, 5, source)]
        writer = Writer()

        NaturalSubtitleTranslator(
            writer, subtitle_reviewer=Reviewer(),
        ).translate_interview_clip_once(
            cues, [], source_words_from_cues(cues), "alignment",
            "These founders are super AI-pilled, running coding agents.",
        )

        self.assertIn("Nearby audio ASR hypothesis", writer.prompt)
        self.assertIn("AI-pilled", writer.prompt)
        self.assertEqual(cues[0].source_text, source)

    def test_joint_translation_surfaces_numeric_audio_conflict_before_review(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.prompt = ""

            def _request_json(self, messages, **kwargs):
                self.prompt = messages[-1]["content"]
                return {"cards": [{
                    "end_word": 9, "text": "排名前 1% 的用户每月支出 903 美元。",
                }]}, {"model": "writer"}

        class Reviewer:
            def _request_json(self, messages, **kwargs):
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": rows[0]["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                }]}, {"model": "reviewer"}

        source = "The top 1% user is spending $93 per month."
        cues = [TranscriptCue("cue-1", 0, 5, source)]
        writer = Writer()

        trace = NaturalSubtitleTranslator(
            writer, subtitle_reviewer=Reviewer(),
        ).translate_interview_clip_once(
            cues, [], source_words_from_cues(cues), "alignment",
            "The top 1% user is spending $903 per month.",
        )

        self.assertIn('Numeric conflict evidence', writer.prompt)
        self.assertIn('"93"', writer.prompt)
        self.assertIn('"903"', writer.prompt)
        ownership = json.loads(
            writer.prompt.split("Entity ownership: ", 1)[1].split("\nMandatory split ranges:", 1)[0]
        )
        self.assertNotIn("93", {row["entity"] for row in ownership})
        self.assertEqual(trace["audio_conflict_numbers"], ["93"])
        self.assertEqual(trace["audio_only_numbers"], ["903"])
        self.assertEqual(
            trace["numeric_conflict_pairs"][0]["source_value"], "93",
        )
        self.assertEqual(
            trace["numeric_conflict_pairs"][0]["audio_candidate"], "903",
        )
        self.assertEqual(cues[0].source_text, source)
        self.assertIn("903", cues[0].translation)

    def test_numeric_audio_conflicts_pair_only_unambiguous_local_replacements(self) -> None:
        source = "Price moved from 93 dollars to 15 dollars today."
        cues = [TranscriptCue("cue-1", 0, 6, source)]

        pairs = _numeric_audio_conflict_pairs(
            source_words_from_cues(cues),
            "Price moved from 903 dollars to 50 dollars today.",
        )

        self.assertEqual(
            [(row["source_value"], row["audio_candidate"]) for row in pairs],
            [("93", "903"), ("15", "50")],
        )

        ambiguous = _numeric_audio_conflict_pairs(
            source_words_from_cues([TranscriptCue(
                "cue-2", 0, 5, "The values are 10 and 20 today.",
            )]),
            "The values are 100 200 and 300 today.",
        )
        self.assertEqual(ambiguous, [])

    def test_numeric_audio_conflict_accepts_proven_joined_ratio(self) -> None:
        words = source_words_from_cues([TranscriptCue(
            "cue-1", 0, 6,
            "Token use flipped from 8020 closed versus open today.",
        )])
        pairs = _numeric_audio_conflict_pairs(
            words,
            "Token use flipped from 80 20 closed versus open today.",
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["source_value"], "8020")
        self.assertEqual(pairs[0]["audio_candidates"], ["80", "20"])
        self.assertEqual(_caption_numeric_alignment_errors(
            words, 0, len(words), "词元使用量从 80/20 闭源对开源翻转。", pairs,
        ), [])

        repeated_words = source_words_from_cues([TranscriptCue(
            "cue-2", 0, 4, "The ratio changed from 11 today.",
        )])
        repeated_pairs = [{
            "source_word_index": 5, "source_value": "11",
            "audio_candidate": "1/1", "audio_candidates": ["1", "1"],
        }]
        self.assertIn("missing_number:11|1/1@5", _caption_numeric_alignment_errors(
            repeated_words, 0, len(repeated_words), "比例变成 1。", repeated_pairs,
        ))

    def test_numeric_ownership_rejects_missing_and_moved_values(self) -> None:
        words = source_words_from_cues([
            TranscriptCue("cue-1", 0, 3, "The cost is 93 dollars."),
        ])

        errors = _caption_numeric_alignment_errors(
            words, 0, len(words), "价格是 94 美元。", [],
        )

        self.assertIn("missing_number:93@4", errors)
        self.assertIn("moved_number:94", errors)

    def test_numeric_ownership_ignores_corrupt_alphanumeric_fragments(self) -> None:
        words = source_words_from_cues([
            TranscriptCue(
                "cue-1", 0, 4,
                "He worked there when it was a 10erson startup.",
            ),
        ])

        errors = _caption_numeric_alignment_errors(
            words, 0, len(words), "他在那里工作时，那还是家十人初创公司。", [],
        )

        self.assertNotIn("missing_number:10@8", errors)
        self.assertFalse(any(error.startswith("missing_number:10") for error in errors))

        semantic_errors = _semantic_card_translation_errors({
            "id": "cue-1", "source": "It was a 10erson startup.",
            "duration_seconds": 3,
        }, "那是一家 10erson 初创公司。", [], require_punctuation=False)
        self.assertIn("copied_asr_fragment:10erson", semantic_errors)

    def test_numeric_ownership_accepts_english_month_as_numeric_chinese_month(self) -> None:
        words = source_words_from_cues([
            TranscriptCue(
                "cue-1", 0, 4,
                "The conference is September 23rd and 24th.",
            ),
        ])

        errors = _caption_numeric_alignment_errors(
            words, 0, len(words), "大会在 9 月 23 日和 24 日举行。", [],
        )

        self.assertEqual(errors, [])

    def test_numeric_ownership_accepts_percent_inherited_by_a_range(self) -> None:
        adjacent = source_words_from_cues([
            TranscriptCue("cue-1", 0, 3, "70 80% of token usage.")
        ])
        connected = source_words_from_cues([
            TranscriptCue("cue-2", 0, 3, "more than 60 or 70%.")
        ])

        self.assertEqual(_caption_numeric_alignment_errors(
            adjacent, 0, len(adjacent), "词元使用量为 70% 到 80%。", [],
        ), [])
        self.assertEqual(_caption_numeric_alignment_errors(
            connected, 0, len(connected), "超过 60% 或 70%。", [],
        ), [])

    def test_numeric_ownership_still_rejects_unrelated_numeric_month(self) -> None:
        words = source_words_from_cues([
            TranscriptCue("cue-1", 0, 3, "The conference is in autumn.")
        ])

        errors = _caption_numeric_alignment_errors(
            words, 0, len(words), "大会在 9 月举行。", [],
        )

        self.assertIn("moved_number:9", errors)

    def test_numeric_audio_exception_is_bound_to_one_source_occurrence(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.prompt = ""

            def _request_json(self, messages, **kwargs):
                self.prompt = messages[-1]["content"]
                return {"cards": [{
                    "end_word": 12, "text": "标价是 903 美元，后面又提到 93 美元。",
                }]}, {"model": "writer"}

        class Reviewer:
            def _request_json(self, messages, **kwargs):
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": rows[0]["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                }]}, {"model": "reviewer"}

        source = "The listed price 93 dollars today and repeated price 93 dollars later."
        cues = [TranscriptCue("cue-1", 0, 7, source)]
        writer = Writer()
        NaturalSubtitleTranslator(
            writer, subtitle_reviewer=Reviewer(),
        ).translate_interview_clip_once(
            cues, [], source_words_from_cues(cues), "alignment",
            "The listed price 903 dollars today and repeated price 93 dollars later.",
        )

        ownership = json.loads(
            writer.prompt.split("Entity ownership: ", 1)[1].split(
                "\nMandatory split ranges:", 1,
            )[0]
        )
        numeric = next(row for row in ownership if row["entity"] == "93")
        self.assertEqual(numeric["source_word_start_indices"], [10])

    def test_numeric_audio_pair_is_rebased_inside_later_writer_window(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.pairs: list[list[dict]] = []

            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                evidence = json.loads(
                    prompt.split("Numeric conflict evidence: ", 1)[1].split(
                        ". Authoritative English", 1,
                    )[0]
                )
                self.pairs.append(evidence["locally_aligned_pairs"])
                rows = json.loads(
                    prompt.split("Source words: ", 1)[1].split("\nNext source:", 1)[0]
                )
                ends = list(range(10, len(rows), 10)) + [len(rows)]
                cards = []
                left = 0
                for end in ends:
                    contains_conflict = any(
                        row["word"] == "93" for row in rows[left:end]
                    )
                    cards.append({
                        "end_word": end,
                        "text": "音频值是 903。" if contains_conflict else "这段内容。",
                    })
                    left = end
                return {"cards": cards}, {"model": "writer"}

        class Reviewer:
            def _request_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                if "Sequence: " in prompt:
                    return {"pass": True, "issues": []}, {"model": "reviewer-global"}
                rows = json.loads(prompt.split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                } for row in rows]}, {"model": "reviewer"}

        tokens = [f"word{index}" for index in range(1, 181)]
        tokens[159] = "93"
        audio_tokens = list(tokens)
        audio_tokens[159] = "903"
        source = " ".join(tokens)
        cues = [TranscriptCue("cue-1", 0, 108, source)]
        writer = Writer()

        NaturalSubtitleTranslator(
            writer, subtitle_reviewer=Reviewer(),
        ).translate_interview_clip_once(
            cues, [], source_words_from_cues(cues), "alignment",
            " ".join(audio_tokens),
        )

        nonempty = [pairs for pairs in writer.pairs if pairs]
        self.assertEqual(len(nonempty), 1)
        self.assertEqual(nonempty[0][0]["source_word_index"], 40)

    def test_translation_trace_archive_keeps_inline_trace_compact(self) -> None:
        trace = [{
            "step": "interview_joint_boundary_translation",
            "alignment_fingerprint": "abc",
            "attempts": [{"failed_cards": ["large payload"]}],
        }]

        compact = _compact_translation_trace(trace)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            audit = _write_translation_audit(root, trace)
            payload = json.loads((root / audit["asset"]).read_text())
            plan_path = root / "translation-plan.json"
            restored = _translation_trace_from_plan(plan_path, {
                "trace": compact, "translation_audit": audit,
            })

        self.assertNotIn("attempts", compact[0])
        self.assertEqual(compact[0]["alignment_fingerprint"], "abc")
        self.assertEqual(payload["trace"], trace)
        self.assertEqual(restored, trace)
        self.assertEqual(audit["bytes"], len(
            (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        ))

    def test_non_speech_laughter_and_cough_are_removed_from_publishable_source(self) -> None:
        self.assertEqual(
            omit_non_speech_directions("[laughter] Hello. 【咳嗽】 We can begin."),
            "Hello. We can begin.",
        )

    def test_joint_interview_translation_deterministically_coalesces_short_entity_card(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.calls = 0

            def _request_json(self, *args, **kwargs):
                self.calls += 1
                return {"cards": [
                    {"end_word": 9, "text": "系统随后调用 API。"},
                    {"end_word": 10, "text": "接口。"},
                    {"end_word": 20, "text": "其余请求继续按原路径处理。"},
                ]}, {"model": "writer"}

        class Reviewer:
            def _request_json(self, messages, **kwargs):
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                } for row in rows]}, {"model": "reviewer"}

        tokens = [f"word{index}" for index in range(1, 21)]
        tokens[9] = "API"
        source = " ".join(tokens)
        cues = [TranscriptCue("cue-1", 0, 12, source)]
        words = source_words_from_cues(cues)
        writer = Writer()

        trace = NaturalSubtitleTranslator(
            writer, subtitle_reviewer=Reviewer(),
        ).translate_interview_clip_once(cues, [], words)

        self.assertEqual(writer.calls, 1)
        self.assertEqual(len(cues), 2)
        self.assertIn("API", cues[0].source_text)
        self.assertTrue(any(
            item["kind"] == "deterministic_boundary_coalesce"
            for item in trace["attempts"]
        ))

    def test_joint_interview_translation_coalesces_toward_nonadjacent_entity_owner(self) -> None:
        class Writer:
            def _request_json(self, *args, **kwargs):
                return {"cards": [
                    {"end_word": 4, "text": "系统会调用 API。"},
                    {"end_word": 8, "text": "随后检查权限。"},
                    {"end_word": 12, "text": "最后执行接口请求。"},
                ]}, {"model": "writer"}

        class Reviewer:
            def _request_json(self, messages, **kwargs):
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                } for row in rows]}, {"model": "reviewer"}

        tokens = [f"word{index}" for index in range(1, 13)]
        tokens[8] = "API"
        source = " ".join(tokens)
        cues = [TranscriptCue("cue-1", 0, 7.2, source)]

        NaturalSubtitleTranslator(
            Writer(), subtitle_reviewer=Reviewer(),
        ).translate_interview_clip_once(cues, [], source_words_from_cues(cues))

        self.assertEqual(len(cues), 1)
        self.assertIn("API", cues[0].source_text)

    def test_joint_repair_window_expands_to_distant_entity_owner(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.calls = 0

            def _request_json(self, *args, **kwargs):
                self.calls += 1
                first = "系统调用 API。" if self.calls == 1 else "系统先检查权限。"
                return {"cards": [
                    {"end_word": 10, "text": first},
                    {"end_word": 20, "text": "随后准备请求。"},
                    {"end_word": 30, "text": "最后调用 API。"},
                ]}, {"model": "writer", "call": self.calls}

        class Reviewer:
            def _request_json(self, messages, **kwargs):
                rows = json.loads(messages[-1]["content"].split("Rows: ", 1)[1])
                return {"reviews": [{
                    "id": row["id"], "pass": True,
                    "fidelity_score": 5, "naturalness_score": 5, "errors": [],
                } for row in rows]}, {"model": "reviewer"}

        tokens = [f"word{index}" for index in range(1, 31)]
        tokens[25] = "API"
        source = " ".join(tokens)
        cues = [TranscriptCue("cue-1", 0, 18, source)]
        writer = Writer()

        trace = NaturalSubtitleTranslator(
            writer, subtitle_reviewer=Reviewer(),
        ).translate_interview_clip_once(cues, [], source_words_from_cues(cues))

        self.assertEqual(writer.calls, 2)
        self.assertEqual(trace["repair_rounds"], 1)
        self.assertNotIn("API", cues[0].translation)
        self.assertIn("API", cues[-1].source_text)

    def test_acronym_plural_owns_same_singular_chinese_entity(self) -> None:
        self.assertEqual(
            _caption_entity_alignment_errors(
                "We inspect internal APIs and services.",
                "我们会检查内部 API 和服务。",
            ),
            [],
        )

    def test_compact_preserved_term_matches_spaced_source_variant(self) -> None:
        terminology = [TerminologyEntry(
            "openweight", TerminologyStrategy.PRESERVE, target="openweight",
            rationale="The video uses both compact and spaced spellings.",
        )]
        cues = [TranscriptCue(
            "card", 0, 4, "The open weight models are spreading.",
            "openweight 模型正在扩散。",
        )]

        self.assertEqual(terminology_contract_errors(cues, terminology), [])

    def test_contextual_translation_satisfies_acronym_entity_ownership(self) -> None:
        terminology = [TerminologyEntry(
            "RL", TerminologyStrategy.TRANSLATE, target="强化学习",
            rationale="The speaker contrasts training methods in this passage.",
        )]

        self.assertEqual(
            _caption_entity_alignment_errors(
                "RL is useful here.", "这里适合强化学习。", terminology,
            ),
            [],
        )
        self.assertEqual(
            interview_caption_duration_errors([
                TranscriptCue(
                    "rl-card", 0, 3, "RL is useful here.",
                    "这里适合强化学习。",
                ),
            ], terminology),
            [],
        )
        phrase_terminology = [TerminologyEntry(
            "RL environments", TerminologyStrategy.TRANSLATE,
            target="强化学习环境",
        )]
        self.assertEqual(
            _caption_entity_alignment_errors(
                "We test RL environments.", "我们测试强化学习环境。",
                phrase_terminology,
            ),
            [],
        )

    def test_preserved_term_must_remain_on_each_owning_card(self) -> None:
        terminology = [TerminologyEntry(
            "Postgres", TerminologyStrategy.PRESERVE, target="Postgres",
        )]
        cues = [
            TranscriptCue("one", 0, 3, "Postgres stores the rows.", "它会存储这些行。"),
            TranscriptCue("two", 3, 6, "The cache serves reads.", "Postgres 会提供读取。"),
        ]

        errors = terminology_contract_errors(cues, terminology)

        self.assertTrue(any("one" in error and "missing" in error for error in errors))
        self.assertTrue(any("two" in error and "moved" in error for error in errors))

    def test_preserved_term_uses_reviewed_publication_spelling(self) -> None:
        terminology = [TerminologyEntry(
            "crowd strike", TerminologyStrategy.PRESERVE, target="CrowdStrike",
            rationale="The source captions split the company name.",
        )]

        self.assertEqual(terminology_contract_errors([
            TranscriptCue(
                "owner", 0, 4, "The callout is for crowd strike.",
                "这里标注的是 CrowdStrike。",
            ),
        ], terminology), [])
        self.assertTrue(any(
            "missing" in error for error in terminology_contract_errors([
                TranscriptCue(
                    "owner", 0, 4, "The callout is for crowd strike.",
                    "这里标注的是 crowd strike。",
                ),
            ], terminology)
        ))

    def test_longer_preserved_phrase_owns_nested_short_term(self) -> None:
        terminology = [
            TerminologyEntry("runtime", TerminologyStrategy.PRESERVE, target="runtime"),
            TerminologyEntry(
                "agent runtime", TerminologyStrategy.TRANSLATE,
                target="智能体运行时",
            ),
        ]
        cues = [TranscriptCue(
            "one", 0, 4, "The agent runtime gathers context.",
            "智能体运行时会收集上下文。",
        )]

        self.assertEqual(terminology_contract_errors(cues, terminology), [])


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






    def test_spoken_fillers_are_omitted_without_dropping_meaning(self) -> None:
        source = "Um, open source wins, you know, inference cloud, uh."
        translation = "嗯，开源赢了，你知道，推理云，呃。"

        self.assertEqual(
            omit_spoken_fillers_from_translation(source, translation),
            "开源赢了，推理云。",
        )
        self.assertTrue(source_is_spoken_filler_only("Um, uh, you know."))
        self.assertTrue(source_is_spoken_filler_only("Well, so."))
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
        self.assertTrue(source_is_non_speech_only("[laughter】"))
        self.assertEqual(omit_spoken_fillers_from_translation("[coughs]", "[咳嗽]"), "")


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

    def test_common_ai_safety_terms_normalize_to_concise_chinese(self) -> None:
        cues = [TranscriptCue(
            "c1", 0, 5,
            "Use a test harness to detect deceptive behavior.",
            "用测试框架识别刻意欺骗行为。",
        )]
        terminology = NaturalSubtitleTranslator._parse_terminology([
            {"source": "test harness", "strategy": "bilingual_once", "target": "测试框架"},
            {"source": "deceptive", "strategy": "preserve"},
        ], cues)

        self.assertEqual(
            [(term.source, term.strategy, term.target) for term in terminology],
            [
                ("test harness", TerminologyStrategy.TRANSLATE, "测试框架"),
                ("deceptive", TerminologyStrategy.TRANSLATE, "刻意欺骗"),
            ],
        )
        self.assertEqual(terminology_contract_errors(cues, terminology), [])

    def test_terminology_mode_alias_and_chinese_target_do_not_become_preserve(self) -> None:
        cues = [TranscriptCue(
            "c1", 0, 5,
            "Permits and licensing are bureaucratic.",
            "许可、审批和官僚主义流程令人窒息。",
        )]
        terminology = NaturalSubtitleTranslator._parse_terminology([
            {"term": "permit", "mode": "translate", "translation": "许可"},
            {"term": "licensing", "mode": "translate", "translation": "审批"},
            {"source": "bureaucratic", "strategy": "preserve", "target": "官僚主义"},
        ], cues)

        self.assertTrue(all(
            term.strategy == TerminologyStrategy.TRANSLATE
            for term in terminology
        ))
        self.assertEqual(terminology_contract_errors(cues, terminology), [])

    def test_real_estate_uses_natural_contextual_chinese(self) -> None:
        cues = [
            TranscriptCue("buy", 0, 2, "You have to buy real estate.", "你得先买地。"),
            TranscriptCue(
                "price", 2, 5, "Real estate goes from 3,000 to 180,000 an acre.",
                "地价从每英亩三千涨到十八万。",
            ),
            TranscriptCue(
                "space", 5, 8, "The real estate in space is infinite.",
                "太空中的可用空间近乎无限。",
            ),
        ]
        terminology = [
            TerminologyEntry(
                "buy real estate", TerminologyStrategy.TRANSLATE, target="买地",
                rationale="The speaker describes purchasing land.",
            ),
            TerminologyEntry(
                "real estate goes", TerminologyStrategy.TRANSLATE, target="地价",
                rationale="The following numbers are per-acre prices.",
            ),
            TerminologyEntry(
                "real estate in space", TerminologyStrategy.TRANSLATE,
                target="可用空间", rationale="The passage discusses available space.",
            ),
        ]

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
        self.assertEqual(
            [(row["raw"], row["start"], row["end"]) for row in cues[1].source_tokens],
            [
                ("Stop", 1.0, 1.4),
                ("fixing", 1.4, 1.8),
                ("the", 1.8, 2.1),
                ("code.", 2.1, 2.5),
            ],
        )

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
        self.assertEqual(editorial_plan_contract_errors(repaired, 400, cues), [])

    def test_short_interview_highlight_has_no_minimum_duration(self) -> None:
        cues = [TranscriptCue("cue-1", 10, 22, "A complete and useful answer.")]
        plan = {
            "editorial_mode": "known_tech_interview_clip",
            "bilibili_chapters": [],
            "wechat_lessons": [{
                "speaker_label": "Speaker", "title": "短回答也有完整结论",
                "thesis": "一个完整且有价值的回答。", "start": 10, "end": 22,
                "framing": "speaker", "hook_headlines": [
                    "短回答直接给出结论", "关键取舍没有废话", "完整观点不需要凑时长",
                ],
            }],
        }
        self.assertEqual(editorial_plan_contract_errors(plan, 120, cues), [])

    def test_strong_interview_highlight_up_to_300_seconds_needs_no_magic_field(self) -> None:
        cues = [TranscriptCue("cue-1", 10, 220, "A complete and useful technical answer.")]
        row = {
            "speaker_label": "Speaker", "title": "上下文决定工程结论",
            "thesis": "完整保留机制和结果。", "start": 10, "end": 220,
            "framing": "speaker", "hook_headlines": [
                "上下文改变工程判断", "机制解释需要完整保留", "结论来自前因后果",
            ],
        }
        plan = {
            "editorial_mode": "known_tech_interview_clip",
            "bilibili_chapters": [], "wechat_lessons": [row],
        }
        self.assertEqual(editorial_plan_contract_errors(plan, 400, cues), [])
        row["essential_context_justification"] = "Removing the setup would make the mechanism misleading."
        self.assertEqual(editorial_plan_contract_errors(plan, 400, cues), [])

    def test_conference_highlights_are_independent_and_bounded(self) -> None:
        plan = {
            "editorial_mode": "conference_highlights", "bilibili_chapters": [],
            "wechat_lessons": [
                {"title": "检索系统的延迟取舍", "thesis": "解释延迟与召回率。", "start": 100, "end": 180,
                 "hook_headlines": ["延迟决定检索体验", "召回率不是越高越好", "系统取舍改变结果"]},
                {"title": "智能体评测的真实难点", "thesis": "解释生产评测方法。", "start": 500, "end": 620,
                 "hook_headlines": ["离线分数会误导团队", "生产评测需要真实任务", "反馈闭环决定可靠性"]},
            ],
        }
        self.assertEqual(editorial_plan_contract_errors(plan, 30_000), [])
        plan["wechat_lessons"][1]["start"] = 150
        plan["wechat_lessons"][1]["end"] = 250
        self.assertTrue(any(
            "must not overlap" in error
            for error in editorial_plan_contract_errors(plan, 30_000)
        ))

    def test_trusted_multi_hour_stream_uses_conference_highlights(self) -> None:
        item = YouTubeCandidate(
            video_id="conference", url="https://youtube.com/watch?v=conference",
            title="AI Engineer conference: production agent systems",
            channel="AI Engineer", description="Engineering production agent architecture and eval workflow.",
            published_at="20261001", duration_seconds=31_455,
            transcript_available=True, source_width=1920, source_height=1080,
            source_quality_verified=True,
        )
        YouTubeDiscoveryService._score(
            item, DiscoveryConfig(), datetime(2026, 10, 5, tzinfo=UTC),
        )
        self.assertEqual(item.editorial_mode, "conference_highlights")
        self.assertNotIn("duration_out_of_range", item.rejection_reasons)

    def test_legacy_bilingual_once_is_normalized_before_validation(self) -> None:
        cues = [TranscriptCue("cue-1", 0, 5, "This is the system of record.")]
        terms = NaturalSubtitleTranslator._parse_terminology([{
            "source": "system of record", "strategy": "bilingual_once",
            "target": "权威数据源", "first_use_explanation": "权威数据源",
        }, {
            "source": "Harness", "strategy": "bilingual_once",
            "first_use_explanation": "Agent 的执行与反馈框架",
        }], cues)
        by_source = {term.source: term for term in terms}
        self.assertEqual(by_source["system of record"].strategy, TerminologyStrategy.TRANSLATE)
        self.assertNotIn("Harness", by_source)

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

    def test_spoken_boom_interjection_is_not_published_in_chinese(self) -> None:
        self.assertEqual(
            omit_spoken_fillers_from_translation(
                "Boom, here it is right here.", "boom，就在这里。",
            ),
            "就在这里。",
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
