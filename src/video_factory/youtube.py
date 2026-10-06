from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from .llm import OpenAICompatibleStoryWriter
from .compositor import resolve_font_path
from .media import probe_audio_loudness, probe_video
from .models import (
    Candidate, CollectionItem, CollectionItemKind, Evidence, FramingMode, PlatformRender,
    HookSpec, HookStrategy, RenderProfile, RightsReview, SlideTranslation, SourceMediaInfo, SourceRange,
    SourceType, SourceWord, SubtitleMode, TerminologyEntry, TerminologyStrategy, TranscriptCue,
    VideoCollectionManifest, now_iso,
)
from .quality import CheckResult
from .storage import Workspace
from .youtube_runtime import ManagedYouTubeRuntime
from .youtube_alignment import (
    ALIGNMENT_POLICY_VERSION, AudioWordAligner, card_times_from_words,
    source_ledger_fingerprint, source_words_from_cues,
)


DEFAULT_QUERY_POOLS: dict[str, list[str]] = {
    "perplexity": [
        "Aravind Srinivas Perplexity interview",
        "Perplexity CEO AI engineering",
    ],
    "karpathy": [
        "Andrej Karpathy AI talk",
        "Andrej Karpathy agentic engineering",
    ],
    "yc": [
        "Y Combinator AI startup",
        "Y Combinator AI engineering",
    ],
    "all_in": [
        '"All-In Podcast" AI',
        '"All-In Podcast" SaaS',
    ],
    "sequoia": [
        "Sequoia Capital AI startup engineering",
        "Sequoia Capital enterprise software AI",
    ],
    "a16z": [
        "a16z AI enterprise software",
        "a16z AI startup founders",
    ],
    "lightspeed": [
        "Lightspeed Venture Partners AI startup engineering",
        "Lightspeed Venture Partners enterprise AI",
    ],
    "mad_podcast": [
        '"The MAD Podcast" enterprise AI',
        '"The MAD Podcast" future of software',
    ],
    "saastr": [
        '"SaaStr AI" SaaS',
        '"SaaStr AI" enterprise software',
    ],
    "popular_ai": [
        "AI engineering agents talk",
        "coding agents platform engineering",
        "AI developer productivity conference",
    ],
    "startup_builders": [
        "AI startup team technical demo",
        "new AI project engineering launch",
        "startup founders building AI product technical talk",
    ],
    "robotics_physical_ai": [
        "robotics startup engineering talk humanoid demo",
        "physical AI robot learning technical presentation",
        "humanoid robot behind the scenes engineering",
    ],
    "autonomous_driving": [
        "autonomous driving engineering talk field test",
        "self-driving robotaxi technical presentation",
        "autonomous vehicle perception planning engineering",
    ],
}

AUDIENCE_TOPIC_GROUPS: dict[str, tuple[str, ...]] = {
    "ai_models": (
        "ai", "artificial intelligence", "agent", "agentic", "llm", "large language model",
        "foundation model", "reasoning model", "inference", "model training", "perplexity",
    ),
    "software_development": (
        "api", "sdk", "coding", "developer", "engineering", "open source", "github",
        "platform", "software", "database", "cloud", "cybersecurity",
    ),
    "hardware_compute": (
        "gpu", "npu", "tpu", "chip", "semiconductor", "accelerator", "server",
        "data center", "datacenter", "compute", "edge ai", "hbm",
    ),
    "physical_autonomy": (
        "robot", "robotics", "humanoid", "physical ai", "embodied ai",
        "autonomous driving", "self-driving", "robotaxi", "autonomous vehicle",
        "drone", "computer vision", "electric vehicle",
    ),
    "technology_business": (
        "startup", "saas", "crm", "enterprise software", "venture capital",
        "technology company", "tech company",
    ),
}
AUDIENCE_CONSEQUENCE_GROUPS: dict[str, tuple[str, ...]] = {
    "build_or_adopt": (
        "build", "building", "built", "deploy", "deploying", "deployed", "deployment",
        "production", "workflow", "automate", "automation", "adopt", "adoption",
        "available", "launch", "launched", "release", "released",
    ),
    "economics": (
        "cost", "price", "pricing", "cheaper", "revenue", "funding", "acquisition",
        "investment", "earnings", "roi", "productivity",
    ),
    "performance": (
        "latency", "throughput", "benchmark", "faster", "speed", "scale", "scaling",
        "memory", "accuracy", "efficiency",
    ),
    "risk_or_safety": (
        "security", "safety", "risk", "privacy", "regulation", "outage", "failure",
        "ban", "blocked", "vulnerability",
    ),
}
AUDIENCE_TECHNICAL_MARKERS = (
    "api", "sdk", "architecture", "benchmark", "database", "eval", "evaluation",
    "framework", "inference", "infrastructure", "latency", "model training", "production",
    "rag", "security", "system design", "testing", "tool calling", "throughput",
    "system", "systems",
    "computer vision", "field test", "lidar", "manipulation", "motion planning",
    "perception", "sim-to-real", "slam", "world model", "gpu", "semiconductor",
)
AUDIENCE_ACTOR_MARKERS = (
    "openai", "anthropic", "google", "deepmind", "microsoft", "meta", "nvidia",
    "apple", "tesla", "waymo", "amazon", "aws", "deepseek", "qwen", "mistral",
    "perplexity", "karpathy", "y combinator",
)
AUDIENCE_DAILY_LIFE_MARKERS = (
    "home", "transport", "healthcare", "hospital", "education", "school", "factory",
    "manufacturing", "warehouse", "public road", "delivery", "mobility", "energy grid",
)
AUDIENCE_MARKERS = tuple(dict.fromkeys(
    marker
    for group in AUDIENCE_TOPIC_GROUPS.values()
    for marker in group
))
TECHNICAL_SHARE_MARKERS = (
    "agent", "agentic", "api", "architecture", "benchmark", "build", "coding",
    "database", "developer", "engineering", "eval", "evaluation", "framework",
    "inference", "infrastructure", "llm", "model training", "platform engineering",
    "production", "programming", "rag", "sdk", "security", "software", "system design",
    "technical", "testing", "tool calling", "workflow", "saas", "crm", "enterprise software",
    "autonomy", "autonomous driving", "computer vision", "embodied ai", "humanoid",
    "lidar", "manipulation", "motion planning", "perception", "physical ai", "robot",
    "robotics", "robotaxi", "self-driving", "sim-to-real", "slam", "world model",
)
IT_SOFTWARE_AI_SCOPE_MARKERS = (
    # AI and machine intelligence. Avoid the bare word ``model`` because
    # business-model conversations are common on investor channels.
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "generative ai", "agent", "agentic", "llm", "large language model",
    "foundation model", "reasoning model", "model training", "inference",
    "neural network", "computer vision", "natural language processing",
    # Software, developer tooling, and information technology.
    "software", "developer", "coding", "programming", "source code", "api", "sdk",
    "database", "cloud computing", "cloud infrastructure", "cybersecurity",
    "information security", "open source", "github", "saas", "enterprise software",
    "operating system", "ios", "macos", "android", "windows", "linux",
    "computer", "server", "networking", "distributed systems", "data center",
    # Computing hardware and AI-adjacent autonomy.
    "gpu", "npu", "tpu", "semiconductor", "chip", "compute", "edge ai",
    "robot", "robotics", "humanoid", "physical ai", "embodied ai", "robot learning",
    "autonomous driving", "self-driving", "robotaxi", "autonomous vehicle",
    # Chinese equivalents used by discovery sources.
    "人工智能", "机器学习", "深度学习", "生成式ai", "大模型", "智能体", "推理",
    "模型训练", "计算机视觉", "自然语言处理", "软件", "开发者", "编程", "代码",
    "数据库", "云计算", "云基础设施", "网络安全", "信息安全", "开源", "操作系统",
    "服务器", "数据中心", "半导体", "芯片", "算力", "机器人", "具身智能",
    "自动驾驶", "无人驾驶",
)
SPONSOR_SECTION_MARKERS = (
    "thanks to our partners", "thanks to our sponsors", "thanks to the sponsors",
    "this episode is brought to you by", "sponsored by", "our sponsors",
)
INSIGHT_MARKERS = (
    "how", "why", "engineering", "build", "system", "architecture", "workflow", "lessons",
    "team", "scale", "agentic", "technical", "from", "future", "inside",
    "autonomy", "deployment", "field test", "perception", "robotics", "safety",
)
FEATURED_IDENTITIES = (
    "aravind srinivas", "perplexity", "andrej karpathy", "karpathy", "y combinator",
)
DEFAULT_KNOWN_TECH_PEOPLE = (
    "andrej karpathy", "andrew ng", "aravind srinivas", "dario amodei",
    "bjarne stroustrup", "demis hassabis", "fei-fei li", "geoffrey hinton",
    "guido van rossum", "ilya sutskever", "james gosling", "jeff dean",
    "jensen huang", "lex fridman", "linus torvalds", "naval ravikant", "naval",
    "sam altman", "satya nadella", "tim berners-lee", "vint cerf", "yann lecun",
)
INTERVIEW_MARKERS = (
    "interview", "podcast", "conversation", "fireside chat", "q&a", "ask me anything",
    " with ", "panel", "roundtable",
)
CURATED_HIGHLIGHT_CHANNEL_MARKERS = (
    "all-in podcast", "sequoia capital", "a16z", "lightspeed venture partners",
    "the mad podcast", "saastr ai",
)
POLITICAL_PATTERNS = (
    r"\bpolitic(?:s|al)?\b", r"\belections?\b", r"\bpresident(?:ial)?\b",
    r"\bcongress\b", r"\bsenate\b", r"\bgovernment\b", r"\bgeopolit(?:ics|ical)?\b",
    r"\bwar\b", r"\bmilitary\b", r"\btaiwan\b", r"\btrump\b", r"\bbiden\b",
    r"\bdemocrats?\b", r"\brepublicans?\b", r"\bwho funded covid\b",
    r"政治", r"选举", r"总统", r"国会", r"政府", r"地缘政治", r"战争", r"军事",
    r"台湾", r"新冠起源", r"疫情起源",
)
TRUSTED_CHANNEL_MARKERS = (
    "andrej karpathy", "perplexity", "y combinator", "sequoia capital", "all-in podcast",
    "a16z", "lightspeed venture partners", "the mad podcast", "saastr ai",
    "stanford online", "stanford graduate school of business", "ai engineer",
    "lex fridman", "ted", "20vc", "founders forum", "dwarkesh patel",
    "lenny's podcast", "no priors", "cnbc", "bloomberg technology",
    "agility robotics", "apptronik", "boston dynamics", "figure", "ieee spectrum",
    "nvidia", "physical intelligence", "the robot report", "unitree", "waymo",
)
REPOST_DISCLOSURE = re.compile(
    r"(?:source|original video|video credits?)\s*[:：]\s*(?:https?://|@)|"
    r"\b(?:re-?upload(?:ed)?|repost(?:ed)?|originally published by)\b",
    re.IGNORECASE,
)
PROTECTED_TERMS = (
    "AI", "Agent", "API", "SDK", "LLM", "RAG", "MCP", "Skill", "Harness",
    "Claude Code", "GitHub", "Perplexity", "Y Combinator",
)
ESTABLISHED_CHINESE_TERMS: dict[str, str] = {
    "analytics": "数据分析",
    "dashboard": "仪表盘",
    "line charts": "折线图",
    "approval": "审批",
    "first draft": "初稿",
    "high frequency trading": "高频交易",
    "ai driven": "AI 驱动",
    "human supervised": "人工监督",
    "competitor": "竞争对手",
    "opportunities": "机会",
    "model": "模型",
    "chips": "芯片",
    "applications": "应用",
    "infrastructure": "基础设施",
    "data centers": "数据中心",
    "power generation": "发电",
    "application layer": "应用层",
    "earnings": "利润",
    "supply chain": "供应链",
    "bottlenecks": "瓶颈",
    "frontier model": "前沿模型",
    "closed source": "闭源",
    "open-source": "开源",
    "open source": "开源",
    "model layer": "模型层",
    "app tier": "应用层",
    "middleware": "中间件",
    "orchestration layer": "编排层",
    # Common interview/business vocabulary is not product nomenclature.  A
    # planning model may still label these rows preserve/bilingual_once, so
    # normalize them here before the deterministic terminology contract runs.
    "coding agent": "编码智能体",
    "agentic system": "智能体系统",
    "mcp server": "MCP 服务器",
    "evals": "评测",
    "eval": "评测",
    "benchmark": "基准测试",
    "test harness": "测试框架",
    "deceptive": "刻意欺骗",
    "knowledge work": "知识工作",
    "vertical": "垂直行业",
    "productivity gain": "生产力提升",
    "diffusion": "普及",
    "agent": "智能体",
    "bug": "故障",
    "royalty": "版税",
    "token pricing": "token 定价",
    "memory system": "记忆系统",
    "model family": "模型系列",
    "windows interrupt": "Windows 中断",
    "slop": "低质内容",
}
CAPTION_ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "TSMC": ("TSMC", "台积电"),
    "NVIDIA": ("NVIDIA", "英伟达"),
    "AMD": ("AMD",),
}
FILLER_ONLY = re.compile(r"^(?:um+|uh+|you know|like|well|so)[,.!? ]*$", re.IGNORECASE)
NON_SPEECH_DIRECTION = re.compile(
    r"[\[（(【]\s*(?:music|applause|laughter|laughs?|clears? (?:his |her |their )?throat|"
    r"throat clearing|coughs?|sighs?|breathes?|breathing|silence|background noise|"
    r"inaudible|音乐|掌声|笑声|大笑|清(?:了清)?嗓(?:子)?|清喉咙|咳嗽|叹气|呼吸声|"
    r"无声|背景噪音|听不清)\s*[\]）)】]",
    re.IGNORECASE,
)

# Naturalness is judged by the translation/editor models.  Deterministic code
# validates measurable fidelity and layout invariants; it does not maintain a
# growing blacklist of Chinese phrases produced by earlier model runs.
INTERVIEW_CHINESE_STYLE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = ()
INTERVIEW_BOUNDARY_PADDING_SECONDS = 2.0
INTERVIEW_PREFERRED_MAX_SECONDS = 180.0
INTERVIEW_MAX_SECONDS = 300.0
CONFERENCE_HIGHLIGHT_MIN_SECONDS = 45.0
CONFERENCE_HIGHLIGHT_MAX_SECONDS = 300.0
CONFERENCE_HIGHLIGHT_MAX_TOTAL_SECONDS = 900.0
INTERVIEW_MAX_INTERNAL_SILENCE_SECONDS = 3.0
INTERVIEW_CAPTION_POLICY_VERSION = "2026-10-06-v10-subtitle-hardening"
INTERVIEW_CAPTION_TARGET_MAX_SECONDS = 5.0
INTERVIEW_CAPTION_HARD_MAX_SECONDS = 7.5
INTERVIEW_CAPTION_MIN_SECONDS = 1.2
INTERVIEW_CAPTION_BRIEF_ACK_MIN_SECONDS = 0.75
INTERVIEW_CAPTION_TARGET_MAX_ENGLISH_WORDS = 14
INTERVIEW_CAPTION_MAX_ENGLISH_WORDS = 28
INTERVIEW_CAPTION_TARGET_MAX_CHINESE_CHARACTERS = 22
INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND = 12.0
INTERVIEW_CAPTION_MAX_RENDERED_LINES = 3
INTERVIEW_HOOK_CONTEXT_MIN_VISIBLE_CHARACTERS = 22
INTERVIEW_HOOK_CONTEXT_DRAFT_TARGET_MAX_VISIBLE_CHARACTERS = 105
INTERVIEW_DIRECTING_POLICY_VERSION = "2026-09-21-v3-atomic-hook-pair"


def _subtitle_reading_units(value: str) -> int:
    """Count CJK glyphs and Latin tokens as comparable subtitle reading units."""
    return len(re.findall(
        r"[A-Za-z0-9]+(?:[._/+:'’-][A-Za-z0-9]+)*"
        r"|[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"
        r"|[^\s]",
        value,
    ))


def _interview_caption_policy_fingerprint() -> str:
    policy = {
        "version": INTERVIEW_CAPTION_POLICY_VERSION,
        "target_max_seconds": INTERVIEW_CAPTION_TARGET_MAX_SECONDS,
        "hard_max_seconds": INTERVIEW_CAPTION_HARD_MAX_SECONDS,
        "minimum_seconds": INTERVIEW_CAPTION_MIN_SECONDS,
        "brief_acknowledgement_minimum_seconds": (
            INTERVIEW_CAPTION_BRIEF_ACK_MIN_SECONDS
        ),
        "target_maximum_english_words": INTERVIEW_CAPTION_TARGET_MAX_ENGLISH_WORDS,
        "maximum_english_words": INTERVIEW_CAPTION_MAX_ENGLISH_WORDS,
        "target_maximum_chinese_characters": (
            INTERVIEW_CAPTION_TARGET_MAX_CHINESE_CHARACTERS
        ),
        "maximum_chinese_characters_per_second": (
            INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND
        ),
        "maximum_rendered_lines": INTERVIEW_CAPTION_MAX_RENDERED_LINES,
    }
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


INTERVIEW_CAPTION_POLICY_FINGERPRINT = _interview_caption_policy_fingerprint()


class YouTubeAcquisitionError(RuntimeError):
    pass


class YouTubeWebAuthRequired(YouTubeAcquisitionError):
    pass


class SourceBelow1080Error(YouTubeAcquisitionError):
    pass


class InterviewJointTranslationError(ValueError):
    """A failed self-healing run with a serializable diagnostic trace."""

    def __init__(self, message: str, trace: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.trace = trace


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    cadence_hours: int = 2
    max_source_selections: int = 1
    minimum_score: float = 70.0
    minimum_duration_seconds: int = 900
    maximum_duration_seconds: int = 7200
    lookback_days: int = 30
    results_per_query: int = 8
    metadata_probe_limit: int = 10
    minimum_audience_score: float = 12.0
    timezone: str = "Asia/Tokyo"
    query_pools: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_QUERY_POOLS))
    channel_sources: dict[str, list[str]] = field(default_factory=dict)
    known_tech_people: list[str] = field(default_factory=lambda: list(DEFAULT_KNOWN_TECH_PEOPLE))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoveryConfig":
        aliases = {
            "source_duration_minutes": None,
        }
        unknown = set(data) - set(cls.__dataclass_fields__) - set(aliases)
        if unknown:
            raise ValueError("unsupported YouTube discovery config fields: " + ", ".join(sorted(unknown)))
        values = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        duration = data.get("source_duration_minutes")
        if isinstance(duration, (list, tuple)) and len(duration) == 2:
            values["minimum_duration_seconds"] = int(float(duration[0]) * 60)
            values["maximum_duration_seconds"] = int(float(duration[1]) * 60)
        return cls(**values)

    @classmethod
    def from_path(cls, path: Path | None) -> "DiscoveryConfig":
        if path is None:
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(slots=True)
class YouTubeCandidate:
    video_id: str
    url: str
    title: str
    channel: str = ""
    description: str = ""
    published_at: str = ""
    duration_seconds: float = 0.0
    view_count: int = 0
    chapters: list[dict[str, Any]] = field(default_factory=list)
    creators: list[str] = field(default_factory=list)
    transcript_available: bool = False
    source_width: int = 0
    source_height: int = 0
    source_quality_verified: bool = False
    matched_pools: list[str] = field(default_factory=list)
    discovery_routes: list[str] = field(default_factory=list)
    source_channel_url: str = ""
    channel_recency_rank: int = 1_000_000
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    audience_breakdown: dict[str, float] = field(default_factory=dict)
    audience_matches: dict[str, list[str]] = field(default_factory=dict)
    scope_markers: list[str] = field(default_factory=list)
    eligible: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    editorial_mode: str = ""
    matched_known_people: list[str] = field(default_factory=list)
    political_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DiscoveryRun:
    id: str
    status: str
    started_at: str
    completed_at: str = ""
    next_run_at: str = ""
    candidates: list[YouTubeCandidate] = field(default_factory=list)
    selected: YouTubeCandidate | None = None
    generation_result: dict[str, Any] | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        if re.fullmatch(r"\d{8}", value):
            return datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _normalized_title(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value.casefold()).strip()


def _title_similarity(left: str, right: str) -> float:
    a, b = set(_normalized_title(left).split()), set(_normalized_title(right).split())
    return len(a & b) / len(a | b) if a and b else 0.0


def _contains_editorial_marker(text: str, marker: str) -> bool:
    """Match English terms as tokens/phrases while retaining CJK substring matching."""
    normalized = text.casefold()
    marker_normalized = marker.casefold().strip()
    if not marker_normalized:
        return False
    if re.search(r"[\u3400-\u9fff]", marker_normalized):
        return marker_normalized in normalized
    phrase = re.escape(marker_normalized).replace(r"\ ", r"[\s_-]+")
    return bool(re.search(
        rf"(?<![a-z0-9]){phrase}(?![a-z0-9])", normalized,
    ))


def _matching_markers(text: str, markers: tuple[str, ...]) -> list[str]:
    return [marker for marker in markers if _contains_editorial_marker(text, marker)]


def _description_before_sponsors(description: str) -> str:
    """Exclude recurring ad copy from editorial classification and scoring."""
    lowered = description.casefold()
    boundaries = [
        index for marker in SPONSOR_SECTION_MARKERS
        if (index := lowered.find(marker)) >= 0
    ]
    return description[:min(boundaries)].strip() if boundaries else description.strip()


def youtube_editorial_subject_text(
    title: str, description: str, chapters: list[dict[str, Any]],
) -> str:
    """Return only episode-level subject matter, without channel or sponsor identity."""
    chapter_text = " ".join(str(item.get("title") or "") for item in chapters)
    return " ".join(filter(None, [
        title.strip(), _description_before_sponsors(description), chapter_text.strip(),
    ])).strip()


def it_software_ai_markers(text: str) -> list[str]:
    """Find explicit IT/software/AI scope signals; generic business/tech words do not count."""
    return _matching_markers(text, IT_SOFTWARE_AI_SCOPE_MARKERS)


def audience_relevance(text: str) -> tuple[float, dict[str, float], dict[str, list[str]]]:
    """Score audience fit by topic, consequence, specificity, and recognizable stakes."""
    topic_matches = {
        name: _matching_markers(text, markers)
        for name, markers in AUDIENCE_TOPIC_GROUPS.items()
    }
    consequence_matches = {
        name: _matching_markers(text, markers)
        for name, markers in AUDIENCE_CONSEQUENCE_GROUPS.items()
    }
    matched_topic_groups = [name for name, values in topic_matches.items() if values]
    matched_consequence_groups = [name for name, values in consequence_matches.items() if values]
    technical = _matching_markers(text, AUDIENCE_TECHNICAL_MARKERS)
    actors = _matching_markers(text, AUDIENCE_ACTOR_MARKERS)
    daily_life = _matching_markers(text, AUDIENCE_DAILY_LIFE_MARKERS)

    topic_fit = min(8.0, 4.0 + 2.0 * len(matched_topic_groups)) if matched_topic_groups else 0.0
    practical_consequence = min(8.0, 2.0 * len(matched_consequence_groups))
    technical_specificity = min(5.0, float(len(technical)))
    recognizable_stakes = (2.0 if actors else 0.0) + (2.0 if daily_life else 0.0)
    breakdown = {
        "topic_fit": topic_fit,
        "practical_consequence": practical_consequence,
        "technical_specificity": technical_specificity,
        "recognizable_or_daily_stakes": recognizable_stakes,
    }
    matches = {
        "topic_groups": matched_topic_groups,
        "topic_markers": sorted({value for values in topic_matches.values() for value in values}),
        "consequence_groups": matched_consequence_groups,
        "consequence_markers": sorted({
            value for values in consequence_matches.values() for value in values
        }),
        "technical_markers": technical,
        "actor_markers": actors,
        "daily_life_markers": daily_life,
    }
    return round(sum(breakdown.values()), 2), breakdown, matches


def technical_share_markers(text: str) -> list[str]:
    haystack = text.casefold()
    return [marker for marker in TECHNICAL_SHARE_MARKERS if marker in haystack]


def political_markers(text: str) -> list[str]:
    return [pattern for pattern in POLITICAL_PATTERNS if re.search(pattern, text, re.IGNORECASE)]


def classify_youtube_editorial(
    title: str, channel: str, description: str, chapters: list[dict[str, Any]],
    creators: list[str], known_tech_people: list[str] | tuple[str, ...] = DEFAULT_KNOWN_TECH_PEOPLE,
) -> tuple[str, list[str], list[str]]:
    subject_text = youtube_editorial_subject_text(title, description, chapters).casefold()
    identity_text = " ".join([channel, *creators]).casefold()
    text = f"{subject_text} {identity_text}"
    political = political_markers(subject_text)
    known = [name for name in known_tech_people if name.casefold() in text]
    curated_highlight_source = any(
        marker in channel.casefold() for marker in CURATED_HIGHLIGHT_CHANNEL_MARKERS
    )
    interview = (
        any(marker in text for marker in INTERVIEW_MARKERS)
        or len([item for item in creators if item.strip()]) >= 2
    )
    technical = bool(technical_share_markers(subject_text))
    scope_markers = it_software_ai_markers(subject_text)
    curated_highlight_relevant = bool(scope_markers)
    builder_story = any(marker in text for marker in (
        "startup", "we built", "our team", "project launch", "product demo",
        "robot", "robotics", "humanoid", "physical ai", "embodied ai",
        "autonomous driving", "self-driving", "robotaxi",
    ))
    if not scope_markers:
        return ("political_rejected" if political else "rejected"), known, political
    if curated_highlight_source and curated_highlight_relevant:
        return "known_tech_interview_clip", known, political
    if interview and known:
        return "known_tech_interview_clip", known, political
    if political:
        return "political_rejected", known, political
    if technical and known:
        return "known_tech_interview_clip", known, []
    if curated_highlight_source:
        return "rejected", known, political
    if technical and (not interview or builder_story):
        return "technical_coverage", known, []
    return "rejected", known, []


class YouTubeDiscoveryService:
    def __init__(
        self, workspace: Workspace, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        clock: Callable[[], datetime] | None = None,
        runtime: ManagedYouTubeRuntime | None = None,
    ) -> None:
        self.workspace = workspace
        self.runner = runner or subprocess.run
        self.custom_runner = runner is not None
        self.runtime = runtime or ManagedYouTubeRuntime()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.state_path = workspace.root / "youtube-discovery-state.json"
        self.last_trace: dict[str, Any] = {}

    def status(self) -> dict[str, Any]:
        return self._load_state()

    def run(
        self, config: DiscoveryConfig, scheduled: bool = True,
        on_selected: Callable[[YouTubeCandidate], dict[str, Any]] | None = None,
    ) -> DiscoveryRun:
        now = self.clock().astimezone(UTC)
        state = self._load_state()
        next_run = _parse_iso(str(state.get("next_run_at") or ""))
        if scheduled and next_run and now < next_run:
            return DiscoveryRun(
                id=f"discovery-{uuid.uuid4().hex[:10]}", status="not_due",
                started_at=now.isoformat().replace("+00:00", "Z"),
                completed_at=now.isoformat().replace("+00:00", "Z"),
                next_run_at=next_run.isoformat().replace("+00:00", "Z"),
            )

        run = DiscoveryRun(
            id=f"discovery-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}",
            status="running", started_at=now.isoformat().replace("+00:00", "Z"),
        )
        try:
            rough = self._search(config)
            seen_ids = set(str(item) for item in state.get("selected_ids", []))
            seen_titles = [str(item) for item in state.get("selected_titles", [])]
            unique = [item for item in rough if item.video_id not in seen_ids]
            unique = [
                item for item in unique
                if not any(_title_similarity(item.title, title) >= 0.78 for title in seen_titles)
            ]
            probe_candidates = self._choose_probe_candidates(unique, config)
            detailed = [self._hydrate(item) for item in probe_candidates]
            for item in detailed:
                self._score(item, config, now)
            detailed.sort(key=lambda item: (-item.score, -item.view_count, item.video_id))
            run.candidates = detailed
            eligible = [item for item in detailed if item.eligible and item.score >= config.minimum_score]
            if eligible and config.max_source_selections > 0:
                run.selected = eligible[0]
                run.status = "selected"
                consume_selection = True
                if on_selected:
                    try:
                        run.generation_result = on_selected(run.selected)
                    except Exception as error:
                        run.status = "generation_failed"
                        run.error = f"{type(error).__name__}: {error}"
                        if isinstance(error, SourceBelow1080Error) or "YouTube runtime" in str(error):
                            consume_selection = False
                if consume_selection:
                    state.setdefault("selected_ids", []).append(run.selected.video_id)
                    state.setdefault("selected_titles", []).append(run.selected.title)
            else:
                run.status = "no_selection"
        except Exception as error:
            run.status = "configuration_blocked" if "YouTube runtime" in str(error) else "search_failed"
            run.error = f"{type(error).__name__}: {error}"
            run.completed_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
            self._append_history(state, run)
            self._save_state(state)
            return run

        completed = self.clock().astimezone(UTC)
        run.completed_at = completed.isoformat().replace("+00:00", "Z")
        run.next_run_at = (completed + timedelta(hours=config.cadence_hours)).isoformat().replace("+00:00", "Z")
        state["last_completed_at"] = run.completed_at
        state["next_run_at"] = run.next_run_at
        self._append_history(state, run)
        self._save_state(state)
        return run

    def _search(self, config: DiscoveryConfig) -> list[YouTubeCandidate]:
        results: dict[str, YouTubeCandidate] = {}
        trace_sources: list[dict[str, Any]] = []
        failed_sources = 0

        def ingest(
            payload: dict[str, Any], pool: str, route: str,
            source_channel_url: str = "",
        ) -> int:
            added = 0
            for position, raw in enumerate(payload.get("entries", [])):
                if not isinstance(raw, dict):
                    continue
                video_id = str(raw.get("id") or "").strip()
                if not video_id:
                    continue
                raw_url = str(raw.get("webpage_url") or raw.get("url") or "")
                canonical_url = (
                    raw_url if raw_url.startswith(("http://", "https://"))
                    else f"https://www.youtube.com/watch?v={video_id}"
                )
                item = results.setdefault(video_id, YouTubeCandidate(
                    video_id=video_id, url=canonical_url,
                    title=str(raw.get("title") or ""),
                    channel=str(raw.get("channel") or raw.get("uploader") or ""),
                    duration_seconds=float(raw.get("duration") or 0),
                    view_count=int(raw.get("view_count") or 0),
                ))
                if pool not in item.matched_pools:
                    item.matched_pools.append(pool)
                route_label = f"{route}:{pool}"
                if route_label not in item.discovery_routes:
                    item.discovery_routes.append(route_label)
                if route == "channel":
                    item.source_channel_url = source_channel_url
                    item.channel_recency_rank = min(item.channel_recency_rank, position)
                added += 1
            return added

        for pool, channel_urls in config.channel_sources.items():
            for channel_url in channel_urls:
                videos_url = channel_url.rstrip("/")
                if not videos_url.endswith("/videos"):
                    videos_url += "/videos"
                command = [
                    self._executable(), "--flat-playlist", "--dump-single-json", "--skip-download",
                    "--playlist-end", str(config.results_per_query),
                    *self.runtime.extractor_arguments("gvs"), videos_url,
                ]
                completed = self.runner(command, check=False, capture_output=True, text=True)
                if completed.returncode != 0:
                    failed_sources += 1
                    trace_sources.append({
                        "pool": pool, "route": "channel", "source": channel_url,
                        "status": "failed", "rows": 0,
                        "error": (completed.stderr or completed.stdout).strip()[:500],
                    })
                    continue
                try:
                    payload = json.loads(completed.stdout)
                except json.JSONDecodeError as error:
                    failed_sources += 1
                    trace_sources.append({
                        "pool": pool, "route": "channel", "source": channel_url,
                        "status": "failed", "rows": 0,
                        "error": f"JSONDecodeError: {error}",
                    })
                    continue
                rows = ingest(payload, pool, "channel", channel_url)
                trace_sources.append({
                    "pool": pool, "route": "channel", "source": channel_url,
                    "status": "ok", "rows": rows,
                })

        cutoff = (
            self.clock().astimezone(UTC) - timedelta(days=config.lookback_days)
        ).date().isoformat()
        for pool, queries in config.query_pools.items():
            for query in queries:
                dated_query = f"{query} after:{cutoff}"
                command = [
                    self._executable(), "--flat-playlist", "--dump-single-json", "--skip-download",
                    *self.runtime.extractor_arguments("gvs"),
                    f"ytsearch{config.results_per_query}:{dated_query}",
                ]
                completed = self.runner(command, check=False, capture_output=True, text=True)
                if completed.returncode != 0:
                    failed_sources += 1
                    trace_sources.append({
                        "pool": pool, "route": "search", "source": query,
                        "effective_query": dated_query, "status": "failed", "rows": 0,
                        "error": (completed.stderr or completed.stdout).strip()[:500],
                    })
                    continue
                try:
                    payload = json.loads(completed.stdout)
                except json.JSONDecodeError as error:
                    failed_sources += 1
                    trace_sources.append({
                        "pool": pool, "route": "search", "source": query,
                        "effective_query": dated_query, "status": "failed", "rows": 0,
                        "error": f"JSONDecodeError: {error}",
                    })
                    continue
                rows = ingest(payload, pool, "search")
                trace_sources.append({
                    "pool": pool, "route": "search", "source": query,
                    "effective_query": dated_query, "status": "ok", "rows": rows,
                })
        self.last_trace = {
            "sources": trace_sources,
            "funnel": {
                "channel_sources": sum(len(values) for values in config.channel_sources.values()),
                "query_count": sum(len(values) for values in config.query_pools.values()),
                "sources_failed": failed_sources,
                "rough_candidates": len(results),
            },
        }
        if not results and failed_sources:
            raise RuntimeError("all YouTube discovery routes failed or returned no candidates")
        return list(results.values())

    def _hydrate(self, item: YouTubeCandidate) -> YouTubeCandidate:
        command = [
            self._executable(), "--dump-single-json", "--skip-download", "--ignore-no-formats-error",
            *self.runtime.extractor_arguments("gvs"), item.url,
        ]
        completed = self.runner(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0 and not completed.stdout.strip():
            item.rejection_reasons.append("metadata_unavailable")
            return item
        try:
            raw = json.loads(completed.stdout)
        except json.JSONDecodeError:
            item.rejection_reasons.append("metadata_unavailable")
            return item
        item.title = str(raw.get("title") or item.title)
        item.channel = str(raw.get("channel") or raw.get("uploader") or item.channel)
        canonical_channel = str(
            raw.get("channel_url") or raw.get("uploader_url") or ""
        ).strip()
        channel_id = str(raw.get("channel_id") or "").strip()
        if not canonical_channel and channel_id:
            canonical_channel = f"https://www.youtube.com/channel/{channel_id}"
        if canonical_channel:
            item.source_channel_url = canonical_channel
        # Search output should remain auditable without dumping an entire promotional
        # description into every CLI run. Acquisition records the full metadata later.
        item.description = str(raw.get("description") or "")[:2000]
        item.published_at = str(raw.get("upload_date") or raw.get("release_date") or "")
        item.duration_seconds = float(raw.get("duration") or item.duration_seconds or 0)
        item.view_count = int(raw.get("view_count") or item.view_count or 0)
        item.chapters = [dict(chapter) for chapter in (raw.get("chapters") or []) if isinstance(chapter, dict)]
        item.creators = [str(value) for value in (raw.get("creators") or []) if str(value).strip()]
        if not item.creators and str(raw.get("creator") or "").strip():
            item.creators = [value.strip() for value in str(raw["creator"]).split(",") if value.strip()]
        item.transcript_available = bool(raw.get("subtitles") or raw.get("automatic_captions"))
        formats = [
            row for row in (raw.get("formats") or [])
            if isinstance(row, dict) and int(row.get("height") or 0) > 0
        ]
        if formats:
            best = max(formats, key=lambda row: (int(row.get("height") or 0), int(row.get("width") or 0)))
            item.source_width = int(best.get("width") or 0)
            item.source_height = int(best.get("height") or 0)
            item.source_quality_verified = True
            if item.source_width < 1920 or item.source_height < 1080:
                item.rejection_reasons.append("source_below_1080")
        else:
            item.rejection_reasons.append("source_quality_unavailable")
        return item

    def _executable(self) -> str:
        return "yt-dlp" if self.custom_runner else self.runtime.require_executable()

    @staticmethod
    def _choose_probe_candidates(
        candidates: list[YouTubeCandidate], config: DiscoveryConfig,
    ) -> list[YouTubeCandidate]:
        """Spend metadata probes across every editorial pool, with originals first."""
        configured_pools = list(dict.fromkeys([
            *config.query_pools.keys(), *config.channel_sources.keys(),
        ]))
        pools = [
            pool for pool in configured_pools
            if any(pool in item.matched_pools for item in candidates)
        ]
        direct_pools = [
            pool for pool in pools
            if any(
                pool in item.matched_pools
                and any(route == f"channel:{pool}" for route in item.discovery_routes)
                for item in candidates
            )
        ]
        limit = max(0, config.metadata_probe_limit, len(pools) + len(direct_pools))
        if not limit:
            return []

        def priority(item: YouTubeCandidate) -> tuple[int, int, int, int, int, int, str]:
            channel = item.channel.casefold()
            identity = f"{item.title} {item.channel}".casefold()
            return (
                int(any(route.startswith("channel:") for route in item.discovery_routes)),
                -item.channel_recency_rank,
                int(any(marker in channel for marker in TRUSTED_CHANNEL_MARKERS)),
                int(any(marker in identity for marker in FEATURED_IDENTITIES)),
                len(item.matched_pools), item.view_count, item.video_id,
            )

        ranked = sorted(candidates, key=priority, reverse=True)
        per_pool = {
            pool: [item for item in ranked if pool in item.matched_pools]
            for pool in pools
        }
        selected: list[YouTubeCandidate] = []
        selected_ids: set[str] = set()
        while len(selected) < limit:
            added = False
            for pool in pools:
                queue = per_pool[pool]
                while queue and queue[0].video_id in selected_ids:
                    queue.pop(0)
                if not queue:
                    continue
                item = queue.pop(0)
                selected.append(item)
                selected_ids.add(item.video_id)
                added = True
                if len(selected) >= limit:
                    break
            if not added:
                break
        for item in ranked:
            if len(selected) >= limit:
                break
            if item.video_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.video_id)
        return selected

    @staticmethod
    def _score(item: YouTubeCandidate, config: DiscoveryConfig, now: datetime) -> None:
        subject_text = youtube_editorial_subject_text(
            item.title, item.description, item.chapters,
        )
        haystack = subject_text.casefold()
        channel = item.channel.casefold()
        reasons = list(item.rejection_reasons)
        item.scope_markers = it_software_ai_markers(subject_text)
        mode, known_people, political = classify_youtube_editorial(
            item.title, item.channel, item.description, item.chapters, item.creators,
            config.known_tech_people,
        )
        item.editorial_mode = mode
        item.matched_known_people = known_people
        item.political_signals = political
        trusted_channel = any(marker in channel for marker in TRUSTED_CHANNEL_MARKERS)
        trusted_conference = bool(
            trusted_channel
            and item.duration_seconds > config.maximum_duration_seconds
            and not item.chapters
            and mode in {"technical_coverage", "known_tech_interview_clip"}
        )
        if trusted_conference:
            item.editorial_mode = mode = "conference_highlights"
        if not item.scope_markers:
            reasons.append("outside_it_software_ai_scope")
        audience, audience_breakdown, audience_matches = audience_relevance(haystack)
        item.audience_breakdown = audience_breakdown
        item.audience_matches = audience_matches
        if audience < config.minimum_audience_score:
            reasons.append("audience_mismatch")
        if mode == "political_rejected":
            reasons.append("political_content_forbidden")
        elif mode == "rejected":
            reasons.append("not_technical_share_or_known_tech_interview")
        if not trusted_conference and not (
            config.minimum_duration_seconds
            <= item.duration_seconds
            <= config.maximum_duration_seconds
        ):
            reasons.append("duration_out_of_range")
        if item.duration_seconds < 900:
            reasons.append("insufficient_material_for_main_and_three_episodes")
        if not item.transcript_available:
            reasons.append("english_transcript_unavailable")
        if not item.title.strip() or not item.channel.strip():
            reasons.append("missing_identity")
        obvious_repost = not trusted_channel and bool(REPOST_DISCLOSURE.search(item.description))
        if obvious_repost:
            reasons.append("secondary_repost_source")

        if trusted_channel:
            authority = 20.0
        elif known_people or any(marker in haystack for marker in FEATURED_IDENTITIES):
            authority = 9.0
        else:
            authority = 6.0
        insight_hits = len(_matching_markers(haystack, INSIGHT_MARKERS))
        insight = min(20.0, 8.0 + insight_hits * 2.0 + min(len(item.chapters), 4))
        published = _parse_iso(item.published_at)
        age_days = max(1.0, (now - published).total_seconds() / 86400) if published else float(config.lookback_days)
        if published and age_days > config.lookback_days:
            reasons.append("outside_lookback")
        freshness = max(0.0, 10.0 * (1.0 - max(0.0, age_days - 1) / max(config.lookback_days, 1)))
        velocity = item.view_count / age_days
        heat = 15.0 if velocity >= 100_000 else 12.0 if velocity >= 20_000 else 8.0 if velocity >= 3_000 else 4.0
        chinese_gap = 2.0 if re.search(r"[\u4e00-\u9fff]", item.title) else 10.0
        item.score_breakdown = {
            "audience_value": audience, "source_authority": authority,
            "insight_density": insight, "heat_velocity": heat,
            "freshness": round(freshness, 2), "chinese_coverage_gap": chinese_gap,
        }
        item.score = round(sum(item.score_breakdown.values()), 2)
        item.rejection_reasons = list(dict.fromkeys(reasons))
        item.eligible = not item.rejection_reasons

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"selected_ids": [], "selected_titles": [], "history": []}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _append_history(state: dict[str, Any], run: DiscoveryRun) -> None:
        summary = {
            "id": run.id, "status": run.status, "started_at": run.started_at,
            "completed_at": run.completed_at, "next_run_at": run.next_run_at,
            "selected_video_id": run.selected.video_id if run.selected else None,
            "selected_score": run.selected.score if run.selected else None,
            "error": run.error,
        }
        state.setdefault("history", []).append(summary)
        state["history"] = state["history"][-100:]


def _join_caption_fragments(fragments: list[str]) -> str:
    """Join JSON3 word fragments without losing spaces at caption-line boundaries."""
    result = ""
    closing_punctuation = set(",.!?;:%)]}，。！？；：、")
    opening_punctuation = set("([{“‘")
    contractions = ("'s", "'re", "'ve", "'ll", "'d", "'m", "n't")
    for raw in fragments:
        fragment = re.sub(r"\s+", " ", raw).strip()
        if not fragment:
            continue
        if not result:
            result = fragment
            continue
        needs_space = (
            fragment[0] not in closing_punctuation
            and not fragment.casefold().startswith(contractions)
            and result[-1] not in opening_punctuation
            and result[-1] not in "-/—–"
        )
        result += (" " if needs_space else "") + fragment
    return result.strip()


def _join_timed_caption_fragments(
    fragments: list[tuple[float, float, str]],
) -> tuple[str, list[dict[str, Any]]]:
    """Join JSON3 fragments and retain timing for the resulting whitespace words."""
    result = ""
    spans: list[tuple[int, int, float, float]] = []
    closing_punctuation = set(",.!?;:%)]}，。！？；：、")
    opening_punctuation = set("([{“‘")
    contractions = ("'s", "'re", "'ve", "'ll", "'d", "'m", "n't")
    for start, end, raw in fragments:
        fragment = re.sub(r"\s+", " ", raw).strip()
        if not fragment:
            continue
        needs_space = bool(result) and (
            fragment[0] not in closing_punctuation
            and not fragment.casefold().startswith(contractions)
            and result[-1] not in opening_punctuation
            and result[-1] not in "-/—–"
        )
        if needs_space:
            result += " "
        left = len(result)
        result += fragment
        spans.append((left, len(result), start, end))
    timed_words: list[dict[str, Any]] = []
    for match in re.finditer(r"\S+", result):
        owners = [
            row for row in spans
            if row[0] < match.end() and match.start() < row[1]
        ]
        if not owners:
            continue
        timed_words.append({
            "raw": match.group(0),
            "start": round(min(row[2] for row in owners), 3),
            "end": round(max(row[3] for row in owners), 3),
        })
    return result.strip(), timed_words


def parse_youtube_json3(path: Path) -> list[TranscriptCue]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokens: list[tuple[float, float, str]] = []
    seen: set[tuple[int, str]] = set()
    for event in payload.get("events", []):
        if not event.get("segs"):
            continue
        event_start = float(event.get("tStartMs") or 0) / 1000
        event_end = event_start + float(event.get("dDurationMs") or 0) / 1000
        segs = event.get("segs", [])
        for index, seg in enumerate(segs):
            text = re.sub(r"^>>\s*", "", str(seg.get("utf8") or ""))
            if not text.strip():
                continue
            start = event_start + float(seg.get("tOffsetMs") or 0) / 1000
            if index + 1 < len(segs):
                end = event_start + float(segs[index + 1].get("tOffsetMs") or 0) / 1000
            else:
                end = min(event_end, start + max(0.3, min(1.2, len(text.strip()) * 0.08)))
            key = (round(start * 100), re.sub(r"\s+", " ", text).strip().casefold())
            if key in seen:
                continue
            seen.add(key)
            tokens.append((start, max(end, start + 0.08), text))
    tokens.sort(key=lambda item: (item[0], item[1]))

    cues: list[TranscriptCue] = []
    buffer: list[tuple[float, float, str]] = []
    cue_start = cue_end = 0.0

    def flush() -> None:
        nonlocal buffer, cue_start, cue_end
        text, source_tokens = _join_timed_caption_fragments(buffer)
        if text:
            cues.append(TranscriptCue(
                id=f"cue-{len(cues) + 1:04d}", start=round(cue_start, 3),
                end=round(max(cue_end, cue_start + 0.8), 3), source_text=text,
                source_tokens=source_tokens,
            ))
        buffer = []

    for start, end, text in tokens:
        if not buffer:
            cue_start = start
        prospective = _join_caption_fragments([
            *[item[2] for item in buffer], text,
        ])
        if buffer and (start - cue_end > 0.6 or start - cue_start > 6.0 or len(prospective) > 105):
            flush()
            cue_start = start
        buffer.append((start, end, text))
        cue_end = end
        if re.search(r"[.!?][\"']?$", text.strip()) and cue_end - cue_start >= 1.0:
            flush()
    if buffer:
        flush()
    for current, following in zip(cues, cues[1:]):
        if 0 < following.start - current.end <= 2.0:
            # Fill a visible gap without shortening the cue below its final
            # authoritative fragment or invalidating retained token timing.
            current.end = round(max(
                current.end,
                min(following.start - 0.05, current.start + 6.0),
            ), 3)
    return cues


def terminology_contract_errors(
    cues: list[TranscriptCue], terminology: list[TerminologyEntry],
) -> list[str]:
    errors: list[str] = []
    combined_source = "\n".join(item.source_text for item in cues)
    combined_target = "\n".join(item.translation for item in cues)
    for entry in terminology:
        if not _contains_term(combined_source, entry.source):
            continue
        established_target = ESTABLISHED_CHINESE_TERMS.get(entry.source.casefold())
        if established_target and (
            entry.strategy != TerminologyStrategy.TRANSLATE
            or entry.target.strip() != established_target
            or bool(entry.alternatives)
        ):
            errors.append(
                f"term:{entry.source}: established Chinese term must translate as "
                f"{established_target!r}, not {entry.strategy.value}"
            )
            continue
        if entry.strategy == TerminologyStrategy.TRANSLATE and not entry.target.strip():
            errors.append(f"term:{entry.source}: translated terms require target")
        if entry.strategy == TerminologyStrategy.TRANSLATE and entry.target.strip():
            # A shorter glossary term must never be enforced inside a longer
            # declared phrase. The longer phrase owns that source span,
            # regardless of whether it is translated or preserved.
            protected_terms = [
                item.source for item in terminology
                if item is not entry
                and len(item.source) > len(entry.source)
                and entry.source.casefold() in item.source.casefold()
            ]
            relevant = [
                item for item in cues
                if _contains_unprotected_term(
                    item.source_text, entry.source, protected_terms,
                )
            ]
            for item in relevant:
                if not _translated_term_present(entry, item):
                    errors.append(
                        f"term:{entry.source}:{item.id}: translated target "
                        f"{entry.target!r} is missing"
                    )
                if (
                    entry.source.casefold() != entry.target.casefold()
                    and _contains_unprotected_term(
                        item.translation, entry.source, protected_terms,
                    )
                ):
                    errors.append(
                        f"term:{entry.source}:{item.id}: ordinary English term remains "
                        "in Chinese translation"
                    )
        if entry.strategy == TerminologyStrategy.PRESERVE:
            protected_terms = [
                item.source for item in terminology
                if item is not entry
                and len(item.source) > len(entry.source)
                and entry.source.casefold() in item.source.casefold()
            ]
            for item in cues:
                source_owns_term = _contains_unprotected_term(
                    item.source_text, entry.source, protected_terms,
                )
                target_has_term = _contains_unprotected_term(
                    item.translation, entry.source, protected_terms,
                )
                if source_owns_term and not target_has_term:
                    errors.append(
                        f"term:{entry.source}:{item.id}: preserved English term "
                        "is missing from its source card"
                    )
                elif target_has_term and not source_owns_term:
                    errors.append(
                        f"term:{entry.source}:{item.id}: preserved English term "
                        "moved from another source card"
                    )
        if entry.strategy == TerminologyStrategy.BILINGUAL_ONCE:
            # Kept only so already-materialized legacy manifests remain
            # renderable. New/cached planning rows are normalized by
            # _parse_terminology before reaching this validator.
            if not _contains_term(combined_target, entry.source) or not entry.first_use_explanation.strip():
                errors.append(f"term:{entry.source}: bilingual_once needs English and first-use explanation")
                continue
            first_source_cue = next(
                (item for item in cues if _contains_term(item.source_text, entry.source)), None,
            )
            if first_source_cue and (
                not _contains_term(first_source_cue.translation, entry.source)
                or entry.first_use_explanation not in first_source_cue.translation
            ):
                errors.append(
                    f"term:{entry.source}: first source use must include English and the declared Chinese explanation"
                )
            if combined_target.count(entry.first_use_explanation) != 1:
                errors.append(f"term:{entry.source}: first-use explanation must appear exactly once")
    literal_false_friends = ("挽具", "铺好的道路", "人类触摸", "技能登记处")
    for phrase in literal_false_friends:
        if phrase in combined_target:
            errors.append(f"translation: unnatural literal term is forbidden: {phrase}")
    return errors


def _translated_term_present(
    entry: TerminologyEntry, cue: TranscriptCue, *, allow_alternatives: bool = False,
) -> bool:
    """Check the selected target, optionally admitting planner candidates pre-review."""
    targets = [entry.target]
    if allow_alternatives:
        targets.extend(entry.alternatives)
    return any(target and target in cue.translation for target in targets)


def _accepted_terminology_targets(
    entry: TerminologyEntry, source: str,
) -> list[str]:
    """Expose only the per-video choices proposed from this video's context."""
    if not _contains_term(source, entry.source):
        return []
    targets = [entry.target, *entry.alternatives]
    return list(dict.fromkeys(targets))


def _terminology_prompt_row(
    entry: TerminologyEntry, source: str,
) -> dict[str, Any]:
    row = asdict(entry)
    row["accepted_targets"] = _accepted_terminology_targets(entry, source)
    return row


def _relevant_terminology_prompt_rows(
    terminology: list[TerminologyEntry], source: str,
) -> list[dict[str, Any]]:
    """Return glossary rows whose complete source phrase occurs in this context."""
    rows: list[dict[str, Any]] = []
    for entry in terminology:
        protected_terms = [
            other.source for other in terminology
            if other is not entry
            and len(other.source) > len(entry.source)
            and entry.source.casefold() in other.source.casefold()
        ]
        if _contains_unprotected_term(source, entry.source, protected_terms):
            rows.append(_terminology_prompt_row(entry, source))
    return rows


def _adopt_independently_reviewed_terminology_targets(
    cues: list[TranscriptCue], terminology: list[TerminologyEntry],
) -> list[dict[str, str]]:
    """Commit one reviewer-approved contextual rendering as the video target.

    Alternatives are candidates for semantic review, not deterministic aliases.
    Adoption is allowed only when every occurrence uses the same declared
    rendering. The final terminology contract then enforces that one target.
    """
    adopted: list[dict[str, str]] = []
    for entry in terminology:
        if entry.strategy != TerminologyStrategy.TRANSLATE or not entry.alternatives:
            continue
        protected_terms = [
            other.source for other in terminology
            if other is not entry
            and len(other.source) > len(entry.source)
            and entry.source.casefold() in other.source.casefold()
        ]
        relevant = [
            cue for cue in cues
            if _contains_unprotected_term(cue.source_text, entry.source, protected_terms)
        ]
        if not relevant:
            continue
        candidates = list(dict.fromkeys([entry.target, *entry.alternatives]))
        selected: list[str] = []
        for cue in relevant:
            matches = sorted(
                (target for target in candidates if target and target in cue.translation),
                key=len, reverse=True,
            )
            if not matches:
                selected = []
                break
            selected.append(matches[0])
        if len(set(selected)) != 1 or not selected or selected[0] == entry.target:
            continue
        previous = entry.target
        entry.target = selected[0]
        entry.alternatives = list(dict.fromkeys([
            previous, *[item for item in entry.alternatives if item != entry.target],
        ]))[:2]
        adopted.append({
            "source": entry.source,
            "previous_target": previous,
            "target": entry.target,
        })
    return adopted


def enforce_cached_terminology_contract(
    cues: list[TranscriptCue], terminology: list[TerminologyEntry],
) -> dict[str, Any] | None:
    """Re-audit reviewed/cached subtitles before rendering them.

    A human may improve the Chinese after the original model pass. The cached
    path must therefore run the same deterministic terminology enforcement as
    a fresh translation instead of rendering first and reporting the failure
    only at final validation.
    """
    original = [asdict(item) for item in terminology]
    combined_source = "\n".join(item.source_text for item in cues)
    stale_annotations: list[str] = []
    for entry in terminology:
        if _contains_term(combined_source, entry.source):
            continue
        annotation = re.compile(
            rf"\s*[（(]\s*{re.escape(entry.source)}\s*[）)]",
            re.IGNORECASE,
        )
        for cue in cues:
            cleaned = annotation.sub("", cue.translation)
            if cleaned != cue.translation:
                cue.translation = cleaned
                stale_annotations.append(entry.source)
    terminology[:] = NaturalSubtitleTranslator._parse_terminology(original, cues)
    normalized = original != [asdict(item) for item in terminology]
    errors_before = terminology_contract_errors(cues, terminology)
    if not errors_before:
        return {
            "step": "cached_terminology_enforcement",
            "errors_before": [],
            "terms": [],
            "normalized_policy": normalized,
            "removed_stale_annotations": list(dict.fromkeys(stale_annotations)),
        } if normalized or stale_annotations else None
    enforced = NaturalSubtitleTranslator._enforce_terminology_contract(cues, terminology)
    errors_after = terminology_contract_errors(cues, terminology)
    if errors_after:
        raise ValueError(
            "cached translation terminology contract failed after deterministic repair: "
            + "; ".join(errors_after)
        )
    return {
        "step": "cached_terminology_enforcement",
        "errors_before": errors_before,
        "terms": enforced,
        "normalized_policy": normalized,
        "removed_stale_annotations": list(dict.fromkeys(stale_annotations)),
    }


def _contains_term(value: str, term: str) -> bool:
    """Match Latin terms as tokens, never as pieces of a larger identifier."""
    if re.search(r"[A-Za-z0-9]", term):
        plural = ""
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9 .+-]*[A-Za-z]", term) and not term.casefold().endswith("s"):
            plural = r"(?:s|es)?"
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(term)}{plural}(?![A-Za-z0-9])", value,
            re.IGNORECASE,
        ):
            return True
        compact = re.sub(r"[^A-Za-z0-9]+", "", term).casefold()
        if len(compact) >= 8 and re.fullmatch(r"[A-Za-z0-9]+", term):
            tokens = re.findall(r"[A-Za-z0-9]+", value)
            return any(
                "".join(tokens[start:end]).casefold() in {compact, compact + "s", compact + "es"}
                for start in range(len(tokens))
                for end in range(start + 2, min(len(tokens), start + 3) + 1)
            )
        return False
    return term.casefold() in value.casefold()


def _contains_unprotected_term(
    value: str, term: str, protected_terms: list[str],
) -> bool:
    masked = value
    for protected in protected_terms:
        masked = re.sub(re.escape(protected), " ", masked, flags=re.IGNORECASE)
    return _contains_term(masked, term)


def rebalance_translated_cues(cues: list[TranscriptCue], max_chars_per_second: float = 12.0) -> list[TranscriptCue]:
    """Merge adjacent subtitle thoughts when a faithful translation cannot be read in time."""
    balanced: list[TranscriptCue] = []
    index = 0
    while index < len(cues):
        current = cues[index]
        needed = _subtitle_reading_units(current.translation) / max_chars_per_second if current.translation else 0
        if needed > current.duration and index + 1 < len(cues):
            following = cues[index + 1]
            combined_duration = following.end - current.start
            if following.start - current.end <= 0.8 and combined_duration <= 6.5:
                separator = "" if re.search(r"[，。；：！？,.!?]$", current.translation) else "，"
                balanced.append(TranscriptCue(
                    id=current.id, start=current.start, end=following.end,
                    source_text=f"{current.source_text} {following.source_text}".strip(),
                    translation=f"{current.translation}{separator}{following.translation}".strip(),
                    speaker=current.speaker, confidence=current.confidence,
                    source_tokens=[*current.source_tokens, *following.source_tokens],
                ))
                index += 2
                continue
        balanced.append(current)
        index += 1
    return balanced


def fast_translation_cues(
    cues: list[TranscriptCue], max_chars_per_second: float = 12.0,
) -> list[TranscriptCue]:
    return [
        cue for cue in cues
        if cue.translation.strip()
        and _subtitle_reading_units(cue.translation)
        / max(cue.duration, 0.1) > max_chars_per_second
    ]


def interview_chinese_style_errors(cues: list[TranscriptCue]) -> dict[str, list[str]]:
    errors: dict[str, list[str]] = {}
    for cue in cues:
        found = [
            reason for pattern, reason in INTERVIEW_CHINESE_STYLE_PATTERNS
            if pattern.search(cue.translation)
        ]
        visible = _subtitle_reading_units(cue.translation)
        if (
            cue.translation.strip()
            and visible / max(cue.duration, 0.1)
            > INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND
        ):
            found.append(
                "exceeds "
                f"{INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND:g} visible "
                "characters per second"
            )
        if found:
            errors[cue.id] = found
    return errors


SOURCE_DANGLING_END = re.compile(
    r"(?:\b(?:a|about|an|and|are|as|at|because|but|by|can|could|for|from|if|in|is|my|"
    r"maybe|of|on|or|our|so|that|the|their|then|this|to|uh|um|was|were|when|where|which|"
    r"though|while|will|with|would|your|can't|couldn't|didn't|doesn't|don't|won't|wouldn't)|"
    r"[-,:;])\s*[\"'”’]?$",
    re.IGNORECASE,
)


def rebalance_source_cues(cues: list[TranscriptCue], maximum_duration: float = 12.0) -> list[TranscriptCue]:
    """Merge caption fragments whose visible English cannot stand on its own."""
    balanced: list[TranscriptCue] = []
    index = 0
    while index < len(cues):
        seed = cues[index]
        current = TranscriptCue(
            id=seed.id, start=seed.start, end=seed.end,
            source_text=seed.source_text, translation=seed.translation,
            speaker=seed.speaker, confidence=seed.confidence,
            original_start=seed.original_start, original_end=seed.original_end,
            source_tokens=list(seed.source_tokens),
        )
        index += 1
        while index < len(cues):
            following = cues[index]
            combined_duration = following.end - current.start
            dangling = bool(SOURCE_DANGLING_END.search(current.source_text.strip()))
            connector_only = bool(re.fullmatch(
                r"(?:and|but|or|so|then|because|uh|um)[,\s]*",
                current.source_text.strip(), re.IGNORECASE,
            ))
            next_word = re.search(r"[A-Za-z]", following.source_text)
            lowercase_continuation = bool(
                current.source_text.rstrip()
                and current.source_text.rstrip()[-1] not in ".?!"
                and next_word and next_word.group(0).islower()
            )
            if (
                (dangling or connector_only or lowercase_continuation)
                and following.start - current.end <= 0.8
                and combined_duration <= maximum_duration
            ):
                current.end = following.end
                current.source_text = f"{current.source_text} {following.source_text}".strip()
                current.translation = ""
                current.speaker = current.speaker or following.speaker
                current.original_end = following.original_end
                current.source_tokens.extend(following.source_tokens)
                index += 1
                continue
            break
        balanced.append(current)
    return balanced


def _required_short_source_ranges(
    cues: list[TranscriptCue], story_start: float, story_end: float,
    maximum_count: int = 8,
) -> list[dict[str, float]]:
    """Choose complete short-source lesson boundaries; the model only names them."""
    span = story_end - story_start
    count = min(maximum_count, max(3, math.ceil(span / 330.0)))
    if not 180 <= span / count <= 360:
        raise ValueError(
            f"source cannot be divided into 3–{maximum_count} lessons of 180–360 seconds: {span:.1f}s"
        )
    cue_ends = sorted({
        cue.end for cue in cues if story_start < cue.end < story_end
    })
    boundaries = [story_start]
    for index in range(1, count):
        remaining = count - index
        low = max(boundaries[-1] + 180.0, story_end - remaining * 360.0)
        high = min(boundaries[-1] + 360.0, story_end - remaining * 180.0)
        target = story_start + span * index / count
        candidates = [value for value in cue_ends if low <= value <= high]
        boundary = min(candidates, key=lambda value: abs(value - target)) if candidates \
            else min(high, max(low, target))
        boundaries.append(round(boundary, 3))
    boundaries.append(story_end)
    return [
        {"start": boundaries[index], "end": boundaries[index + 1]}
        for index in range(count)
    ]


def _normalized_plan_title(value: Any, index: int, prefix: str, used: set[str]) -> str:
    title = _headline_fragment(str(value or "").strip(), 26).rstrip("，、：；,;: ")
    if title in used:
        continued = _headline_fragment(f"{title}（续）", 30).rstrip("，、：；,;: ")
        title = continued if continued and continued not in used else ""
    if len(re.sub(r"\s+", "", title)) < 4:
        title = f"{prefix}第{index}部分"
    used.add(title)
    return title


def _normalized_plan_hooks(value: Any, title: str) -> list[str]:
    supplied = [str(item).strip() for item in value] if isinstance(value, list) else []
    valid = [
        item for item in supplied
        if 6 <= len(re.sub(r"\s+", "", item)) <= 30
        and not item.endswith(("，", "、", "：", "；", ",", ":", ";"))
    ]
    hooks = list(dict.fromkeys(valid))
    subject = _headline_fragment(title.split("：", 1)[0], 16) or "这段课程"
    for fallback in (
        f"{subject}的关键判断",
        f"{subject}的工程取舍",
        f"{subject}带来的系统变化",
    ):
        if fallback not in hooks:
            hooks.append(fallback)
        if len(hooks) == 3:
            break
    return hooks[:3]


def _even_cue_ranges(
    cues: list[TranscriptCue], start: float, end: float, count: int,
    minimum: float, maximum: float,
) -> list[dict[str, float]]:
    cue_ends = sorted({cue.end for cue in cues if start < cue.end < end})
    boundaries = [start]
    for index in range(1, count):
        remaining = count - index
        low = max(boundaries[-1] + minimum, end - remaining * maximum)
        high = min(boundaries[-1] + maximum, end - remaining * minimum)
        target = start + (end - start) * index / count
        candidates = [value for value in cue_ends if low <= value <= high]
        boundary = min(candidates, key=lambda value: abs(value - target)) if candidates else min(high, max(low, target))
        boundaries.append(round(boundary, 3))
    boundaries.append(end)
    return [
        {"start": boundaries[index], "end": boundaries[index + 1]}
        for index in range(count)
    ]


def normalize_editorial_plan_structure(
    plan: dict[str, Any], cues: list[TranscriptCue], duration: float,
    editorial_mode: str = "study",
) -> dict[str, Any]:
    """Repair only structural fields after semantic LLM repair is exhausted."""
    normalized = dict(plan)
    mode = str(plan.get("editorial_mode") or editorial_mode or "study")
    normalized["editorial_mode"] = mode
    if mode == "known_tech_interview_clip":
        rows = [
            row for row in plan.get("wechat_lessons", []) if isinstance(row, dict)
        ] if isinstance(plan.get("wechat_lessons"), list) else []
        raw = rows[0] if rows else {}
        proposed = _coerce_range(raw, duration)
        if proposed is None:
            # Do not invent a mathematical clip boundary. Keeping the invalid
            # shape forces the semantic repair pass to select a real thought.
            start, end = 0.0, 0.0
        else:
            # Oversized interview passages must be semantically shortened by
            # the editor, never center-cropped to the duration ceiling.
            start, end = proposed.start, proposed.end
        title = _normalized_plan_title(raw.get("title"), 1, "高光", set())
        normalized.update({
            "story_start": round(start, 3), "story_end": round(end, 3),
            "bilibili_chapters": [],
            "wechat_lessons": [{
                **raw, "start": round(start, 3), "end": round(end, 3),
                "title": title,
                "thesis": str(raw.get("thesis") or "提炼一段可独立理解、值得转发的技术洞见。").strip(),
                "framing": str(raw.get("framing") or "speaker"),
                "hook_headlines": _normalized_plan_hooks(raw.get("hook_headlines"), title),
            }],
        })
        return normalized
    if mode == "conference_highlights":
        rows = [
            row for row in plan.get("wechat_lessons", []) if isinstance(row, dict)
        ] if isinstance(plan.get("wechat_lessons"), list) else []
        normalized_rows: list[dict[str, Any]] = []
        used_titles: set[str] = set()
        for index, raw in enumerate(rows[:5], start=1):
            proposed = _coerce_range(raw, duration)
            if proposed is None:
                continue
            title = _normalized_plan_title(raw.get("title"), index, "大会高光", used_titles)
            normalized_rows.append({
                **raw,
                "start": round(proposed.start, 3),
                "end": round(proposed.end, 3),
                "title": title,
                "thesis": str(raw.get("thesis") or "提炼一个可独立理解的技术观点。").strip(),
                "framing": str(raw.get("framing") or "speaker"),
                "hook_headlines": _normalized_plan_hooks(raw.get("hook_headlines"), title),
            })
        normalized.update({
            "story_start": 0.0, "story_end": duration,
            "bilibili_chapters": [], "wechat_lessons": normalized_rows,
        })
        return normalized
    try:
        story_start = float(plan.get("story_start", 0))
        story_end = float(plan.get("story_end", duration))
    except (TypeError, ValueError):
        story_start, story_end = 0.0, duration
    if not 0 <= story_start < story_end <= duration + 0.5 or story_end - story_start < duration * 0.9:
        story_start, story_end = 0.0, duration
    story_end = min(story_end, duration)
    normalized["story_start"] = story_start
    normalized["story_end"] = story_end
    span = story_end - story_start

    if mode == "technical_coverage":
        normalized["bilibili_chapters"] = []
    else:
        raw_chapters = [
            row for row in plan.get("bilibili_chapters", []) if isinstance(row, dict)
        ] if isinstance(plan.get("bilibili_chapters"), list) else []
        minimum_chapters = max(1, math.ceil(span / 1800.0))
        maximum_chapters = min(8, max(minimum_chapters, math.floor(span / 480.0)))
        chapter_count = min(max(len(raw_chapters), minimum_chapters), maximum_chapters)
        chapter_ranges = _even_cue_ranges(cues, story_start, story_end, chapter_count, 480.0, 1800.0)
        used_titles: set[str] = set()
        chapters: list[dict[str, Any]] = []
        for index, source_range in enumerate(chapter_ranges, start=1):
            raw = raw_chapters[min(index - 1, len(raw_chapters) - 1)] if raw_chapters else {}
            title = _normalized_plan_title(raw.get("title"), index, "课程", used_titles)
            chapters.append({
                **raw, **source_range,
                "title": title,
                "thesis": str(raw.get("thesis") or f"完整保留课程第{index}部分的核心论证。").strip(),
                "framing": str(raw.get("framing") or "auto")
                if str(raw.get("framing") or "auto") in {item.value for item in FramingMode} else "auto",
                "hook_headlines": _normalized_plan_hooks(raw.get("hook_headlines"), title),
            })
        normalized["bilibili_chapters"] = chapters

    raw_lessons = [
        row for row in plan.get("wechat_lessons", []) if isinstance(row, dict)
    ] if isinstance(plan.get("wechat_lessons"), list) else []
    minimum_lessons = max(1, math.ceil(span / 360.0)) if mode == "technical_coverage" else (
        3 if duration <= 2700 else 4
    )
    maximum_lessons = min(24, max(minimum_lessons, math.floor(span / 180.0))) \
        if mode == "technical_coverage" else 8
    lesson_count = min(max(len(raw_lessons), minimum_lessons), maximum_lessons)
    if mode == "technical_coverage" or duration <= 2700:
        lesson_ranges = _required_short_source_ranges(
            cues, story_start, story_end, 24 if mode == "technical_coverage" else 8,
        )
        lesson_count = len(lesson_ranges)
    else:
        lesson_ranges = []
        for index in range(lesson_count):
            raw = raw_lessons[min(index, len(raw_lessons) - 1)] if raw_lessons else {}
            proposed = _coerce_range(raw, duration)
            if proposed:
                length = min(360.0, max(180.0, proposed.duration))
                center = (proposed.start + proposed.end) / 2
            else:
                length = min(360.0, max(180.0, span / max(lesson_count, 1) * 0.65))
                center = story_start + span * (index + 0.5) / lesson_count
            start = min(max(story_start, center - length / 2), story_end - length)
            lesson_ranges.append({"start": round(start, 3), "end": round(start + length, 3)})
    used_titles = set()
    lessons: list[dict[str, Any]] = []
    for index, source_range in enumerate(lesson_ranges, start=1):
        raw = raw_lessons[min(index - 1, len(raw_lessons) - 1)] if raw_lessons else {}
        title = _normalized_plan_title(raw.get("title"), index, "精讲", used_titles)
        lessons.append({
            **raw, **source_range,
            "title": title,
            "thesis": str(raw.get("thesis") or f"提炼课程第{index}个可独立理解的技术观点。").strip(),
            "framing": str(raw.get("framing") or "auto")
            if str(raw.get("framing") or "auto") in {item.value for item in FramingMode} else "auto",
            "hook_headlines": _normalized_plan_hooks(raw.get("hook_headlines"), title),
        })
    normalized["wechat_lessons"] = lessons
    return normalized


def _metadata_speaker_label(metadata: dict[str, Any]) -> str:
    """Return a short speaker identity grounded only in publisher metadata."""
    known_people = [
        str(value).strip().title()
        for value in metadata.get("known_tech_people", [])
        if str(value).strip()
    ]
    if known_people:
        return known_people[0]

    credited: list[str] = []
    for key in ("creators", "creator", "cast"):
        raw = metadata.get(key)
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if isinstance(value, dict):
                value = value.get("name")
            name = str(value or "").strip()
            if name and name not in credited:
                credited.append(name)
    if credited:
        pair = " × ".join(credited[:2])
        return pair if len(re.sub(r"\s+", "", pair)) <= 24 else credited[0]

    description = str(metadata.get("description") or "")
    name = r"[A-Z][A-Za-z.'’-]+(?:\s+[A-Z][A-Za-z.'’-]+){1,3}"
    dialogue_patterns = (
        rf"({name})\s+sits down with\s+({name})",
        rf"({name})\s+(?:talks|speaks|chats) with\s+({name})",
        rf"({name})\s+interviews\s+({name})",
    )
    for pattern in dialogue_patterns:
        match = re.search(pattern, description)
        if not match:
            continue
        pair = f"{match.group(1)} × {match.group(2)}"
        if len(re.sub(r"\s+", "", pair)) <= 24:
            return pair
        return match.group(2)

    joins = re.search(rf"({name}),?\s+joins\b", description)
    if joins:
        return joins.group(1)

    channel = str(metadata.get("channel") or metadata.get("uploader") or "").strip()
    if channel:
        label = f"{channel} 对谈"
        if 2 <= len(re.sub(r"\s+", "", label)) <= 24:
            return label
    return ""


def _apply_metadata_speaker_fallback(row: dict[str, Any], fallback: str) -> None:
    if not fallback:
        return
    visible = len(re.sub(r"\s+", "", str(row.get("speaker_label") or "")))
    if not 2 <= visible <= 24:
        row["speaker_label"] = fallback


def _editorial_planning_transcript(
    metadata: dict[str, Any], cues: list[TranscriptCue], editorial_mode: str,
    maximum_characters: int = 32_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bound long-form planning input while retaining high-value exact source windows."""
    complete = [
        {"id": item.id, "start": item.start, "end": item.end, "text": item.source_text}
        for item in cues
    ]
    original_characters = sum(len(item.source_text) for item in cues)
    if editorial_mode not in {
        "known_tech_interview_clip", "conference_highlights",
    } or original_characters <= maximum_characters:
        return complete, {
            "step": "planning_input", "mode": "complete_transcript",
            "original_cues": len(cues), "planning_rows": len(complete),
            "original_characters": original_characters,
        }

    chapters = [
        dict(row) for row in metadata.get("chapters") or [] if isinstance(row, dict)
    ]
    windows: list[dict[str, Any]] = []
    window_seconds = 90.0
    duration = float(metadata.get("duration") or (cues[-1].end if cues else 0))
    start = 0.0
    while start < duration:
        end = min(duration, start + window_seconds)
        local = [cue for cue in cues if cue.end > start and cue.start < end]
        if local:
            text = " ".join(cue.source_text for cue in local).strip()
            chapter = next(
                (
                    str(row.get("title") or "")
                    for row in chapters
                    if float(row.get("start_time") or 0) < end
                    and float(row.get("end_time") or duration) > start
                ),
                "",
            )
            audience, _, _ = audience_relevance(f"{chapter} {text}")
            insight = len(_matching_markers(text, INSIGHT_MARKERS))
            specificity = min(4, len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text)))
            tension = min(3, len(re.findall(
                r"\b(?:but|however|instead|versus|vs\.?|surpris|wrong|revers|"
                r"why|how|risk|breakthrough)\w*\b",
                text, re.IGNORECASE,
            )))
            political = political_markers(f"{chapter} {text}")
            score = audience * 2.0 + insight * 1.5 + specificity + tension * 2.0
            if political:
                score -= 100.0
            windows.append({
                "id": f"window-{len(windows) + 1}", "start": round(local[0].start, 3),
                "end": round(local[-1].end, 3), "chapter": chapter, "text": text,
                "score": score, "audience_score": audience, "political": political,
            })
        start = end

    nonpolitical = [row for row in windows if not row["political"]]
    audience_qualified = [
        row for row in nonpolitical if float(row["audience_score"]) >= 12.0
    ]
    ranked = sorted(
        audience_qualified or nonpolitical or windows,
        key=lambda row: (-float(row["score"]), row["start"]),
    )
    selected: list[dict[str, Any]] = []
    selected_characters = 0
    for row in ranked:
        row_characters = len(str(row["text"]))
        if selected and selected_characters + row_characters > maximum_characters:
            continue
        selected.append(row)
        selected_characters += row_characters
        if selected_characters >= maximum_characters * 0.82 or len(selected) >= 18:
            break
    selected.sort(key=lambda row: float(row["start"]))
    planning_rows = [{
        "id": row["id"], "start": row["start"], "end": row["end"],
        "chapter": row["chapter"], "text": row["text"],
    } for row in selected]
    return planning_rows, {
        "step": "planning_input", "mode": "ranked_nonpolitical_windows",
        "original_cues": len(cues), "planning_rows": len(planning_rows),
        "original_characters": original_characters,
        "planning_characters": selected_characters,
        "window_seconds": window_seconds,
        "selected_ranges": [
            {"start": row["start"], "end": row["end"], "chapter": row["chapter"]}
            for row in planning_rows
        ],
    }


_GUIDANCE_TIME_RANGE = re.compile(
    r"(?<!\d)(?P<start>(?:\d{1,2}:)?\d{1,2}:[0-5]\d(?:\.\d{1,3})?)\s*"
    r"(?:-|–|—|to|through)\s*"
    r"(?P<end>(?:\d{1,2}:)?\d{1,2}:[0-5]\d(?:\.\d{1,3})?)(?!\d)",
    re.IGNORECASE,
)


def _guidance_clock_seconds(value: str) -> float:
    raw_parts = value.split(":")
    parts = [int(part) for part in raw_parts[:-1]]
    seconds = float(raw_parts[-1])
    if len(parts) == 2:
        return float(parts[0] * 3600 + parts[1] * 60 + seconds)
    return float(parts[0] * 60 + seconds)


def _add_guidance_ranges_to_planning_transcript(
    planning_rows: list[dict[str, Any]], cues: list[TranscriptCue], guidance: str,
    padding_seconds: float = 12.0,
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    """Expose human/YouTube-Ask timestamp hints to both planning and repair.

    Long interviews normally use ranked windows to bound the prompt. Explicit
    timestamp advice can otherwise point at source words that the planner never
    receives, making a valid title impossible to retrieve or repair.
    """
    ranges: list[dict[str, float]] = []
    for match in _GUIDANCE_TIME_RANGE.finditer(guidance):
        start = _guidance_clock_seconds(match.group("start"))
        end = _guidance_clock_seconds(match.group("end"))
        if end <= start:
            continue
        ranges.append({"start": start, "end": end})
    if not ranges:
        return planning_rows, ranges
    rows_by_id = {str(row.get("id") or ""): dict(row) for row in planning_rows}
    for cue in cues:
        if not any(
            cue.end > max(0.0, row["start"] - padding_seconds)
            and cue.start < row["end"] + padding_seconds
            for row in ranges
        ):
            continue
        rows_by_id[cue.id] = {
            "id": cue.id, "start": cue.start, "end": cue.end,
            "chapter": "editorial guidance", "text": cue.source_text,
        }
    rows = sorted(rows_by_id.values(), key=lambda row: float(row.get("start") or 0))
    return rows, ranges


def _requested_title_from_editorial_guidance(guidance: str | None) -> str:
    if not guidance:
        return ""
    match = re.search(
        r"(?:angle/title|title|标题)\s*[:：]\s*(.+?)"
        r"(?=\s+(?:Select|Use|Keep)\b|\n|$)",
        guidance, re.IGNORECASE,
    )
    return match.group(1).strip().rstrip(".。") if match else ""


def _requested_exact_range_from_editorial_guidance(
    guidance: str | None,
) -> tuple[float, float] | None:
    if not guidance or not re.search(r"\bexact(?:ly)?\b", guidance, re.IGNORECASE):
        return None
    match = _GUIDANCE_TIME_RANGE.search(guidance)
    if not match:
        return None
    start = _guidance_clock_seconds(match.group("start"))
    end = _guidance_clock_seconds(match.group("end"))
    if not 0 < end - start <= INTERVIEW_MAX_SECONDS:
        return None
    return start, end


def _apply_supported_requested_title(
    plan: dict[str, Any], cues: list[TranscriptCue], duration: float,
    guidance: str | None,
) -> dict[str, Any] | None:
    requested = _requested_title_from_editorial_guidance(guidance)
    if not requested:
        return None
    rows = plan.get("wechat_lessons")
    row = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None
    source_range = _coerce_range(row, duration) if row else None
    if source_range is None:
        return {"step": "requested_title", "applied": False, "reason": "no valid selected range"}
    selected_text = " ".join(
        cue.source_text for cue in cues
        if cue.end > source_range.start and cue.start < source_range.end
    )
    errors = _interview_title_entailment_errors(requested, selected_text)
    visible = len(re.sub(r"\s+", "", requested))
    if errors or not 4 <= visible <= 36:
        return {
            "step": "requested_title", "applied": False,
            "reason": "; ".join(errors) if errors else f"title length {visible} is outside 4–36",
        }
    row["title"] = requested
    plan["collection_title"] = requested
    hooks = [
        str(value).strip() for value in row.get("hook_headlines", [])
        if str(value).strip()
    ] if isinstance(row.get("hook_headlines"), list) else []
    if len(hooks) >= 3:
        # A human-requested, source-supported title is the editorial winner.
        # Keep metadata and the first visible frame from drifting apart during
        # a later directing pass.
        row["hook_headlines"] = [
            requested,
            *[value for value in hooks if value != requested],
        ][:3]
    return {"step": "requested_title", "applied": True, "title": requested}


def _matching_completed_directing_audit(
    trace: list[dict[str, Any]], hooks: list[str], context: str,
) -> dict[str, Any] | None:
    """Find an accepted first-screen pair without invalidating it on policy upgrades."""
    if not hooks or not context or not _interview_hook_context_fits_overlay(context):
        return None
    return next((
        item for item in reversed(trace)
        if item.get("step") == "interview_directing_audit"
        and list(item.get("hook_headlines") or []) == hooks
        and (
            not item.get("winning_context")
            or str(item.get("winning_context") or "").strip() == context
        )
    ), None)




class NaturalSubtitleTranslator:
    def __init__(
        self, writer: OpenAICompatibleStoryWriter,
        directing_writer: OpenAICompatibleStoryWriter | None = None,
        subtitle_reviewer: OpenAICompatibleStoryWriter | None = None,
        runtime_guidance: str = "",
    ) -> None:
        self.writer = writer
        self.directing_writer = directing_writer
        self.subtitle_reviewer = subtitle_reviewer
        self.runtime_guidance = runtime_guidance.strip()

    def translate(
        self, metadata: dict[str, Any], cues: list[TranscriptCue], editorial_mode: str = "study",
        editorial_guidance: str | None = None,
        *, plan_only: bool = False,
    ) -> tuple[list[TerminologyEntry], dict[str, Any], list[dict[str, Any]]]:
        """Create the immutable editorial plan; subtitle translation is always deferred.

        ``plan_only`` remains accepted for adapters that predate the shared joint
        subtitle stage. It no longer enables the retired cue translation path.
        """
        cues[:] = rebalance_source_cues(cues)
        transcript, planning_input_trace = _editorial_planning_transcript(
            metadata, cues, editorial_mode,
        )
        if editorial_mode == "known_tech_interview_clip" and editorial_guidance:
            transcript, guidance_ranges = _add_guidance_ranges_to_planning_transcript(
                transcript, cues, editorial_guidance,
            )
            planning_input_trace["guidance_ranges"] = guidance_ranges
            planning_input_trace["planning_rows"] = len(transcript)
            planning_input_trace["planning_characters"] = sum(
                len(str(row.get("text") or "")) for row in transcript
            )
        duration = float(metadata.get("duration") or (cues[-1].end if cues else 0))
        mandatory_layout = ""
        if editorial_mode == "technical_coverage" or (
            editorial_mode == "study" and duration <= 2700
        ):
            lesson_ranges = _required_short_source_ranges(
                cues, 0.0, duration, 24 if editorial_mode == "technical_coverage" else 8,
            )
            mandatory_layout = (
                "Mandatory short-source layout: set story_start=0 and story_end="
                f"{duration:.3f}; "
                + ("return exactly one Bilibili chapter with that same range; " if editorial_mode == "study" else "return no Bilibili chapters; ")
                + "return WeChat lessons using these exact ranges in this exact order, changing "
                "only title, thesis, framing, and hook_headlines: "
                + json.dumps(lesson_ranges, ensure_ascii=False)
            )
        if editorial_mode == "known_tech_interview_clip":
            edition_contract = (
                "Create exactly one exceptionally compelling WeChat highlight from this technology video, interview, or operator conversation. "
                "Select one continuous, self-contained source range with the strongest surprising, "
                "counterintuitive, useful, or emotionally resonant insight about technology, AI, companies, founders, "
                "products, business strategy, markets, investment, infrastructure, or engineering. "
                "The clip itself, its title, thesis, and all hooks must contain no politics, elections, government, "
                "geopolitics, war, military, politicians, or political advocacy. Comparisons between countries' "
                "technology, engineering practice, research, schools, universities, talent, and education are allowed "
                "when they remain non-political. Return editorial_mode=known_tech_interview_clip, no Bilibili chapters, "
                "and exactly one wechat_lessons row. Add speaker_label: a short, recognizable, evidence-backed identity "
                "such as 'C++之父' or the person's name; never invent an honorific. Its three hooks must be unusually attractive and specific: open "
                "a curiosity gap, reveal the speaker's concrete claim, and promise a useful payoff—without clickbait or invention."
                " Choose the strongest audience-relevant title angle first, then retrieve one continuous passage that directly delivers every concrete promise in that title. "
                "Do not attach the title to a merely related high-scoring passage. The passage must include the speaker's answer or payoff, not stop on the host's question."
            )
        elif editorial_mode == "conference_highlights":
            edition_contract = (
                "This is a trusted multi-hour conference stream without useful chapters. Return 1–5 distinct, non-overlapping, standalone WeChat highlights and no Bilibili chapters. Each highlight must be 45–300 seconds; all highlights together must total no more than 900 seconds. Select only self-contained technical ideas with their own setup and payoff. Do not create a numbered series or make one clip depend on another. If no qualifying highlight exists, return an empty wechat_lessons list."
            )
        elif editorial_mode == "technical_coverage":
            edition_contract = (
                "Bilibili production is paused. Create only chronological WeChat 3–6 minute technical mini-lessons. "
                "Together the lessons must cover the complete non-political technical story, including sources longer "
                "than 45 minutes; split into as many lessons as needed (up to 24) instead of silently dropping sections. Return "
                "editorial_mode=technical_coverage and an empty bilibili_chapters list."
            )
        else:
            edition_contract = (
                "Plan two independent editions from the same source. Bilibili is a complete study collection: "
                "chronological, non-overlapping chapters cover the substantive story and never exceed 30 minutes. "
                "WeChat uses 3–6 minute mini-lessons."
            )
        planning_prompt = "\n".join([
            "You are the senior Chinese editor for an evidence-bound AI engineering video channel.",
            "Audience: Chinese developers, AI builders, tech leads, and platform leaders.",
            "Create a curated but faithful edition. Natural Chinese matters more than mirroring English word order.",
            "Never invent a Chinese term merely to make the subtitle fully Chinese. Product names, code, APIs, and emerging terms without a settled Chinese equivalent stay in English.",
            "Preserve English only for product/company names, acronyms, code/API identifiers, or genuinely unsettled technical terms. Ordinary business and technical phrases with established Chinese—including vertical, token pricing, benchmark, eval, agentic system, knowledge work, productivity gain, royalty, and memory system—must use translate, never preserve or bilingual_once.",
            "Every terminology item must choose only translate or preserve. The original English captions are already visible, so never use bilingual_once. Use translate for ordinary established vocabulary and preserve only names, code, APIs, acronyms, and genuinely unsettled terms.",
            "For each translate row return source, strategy, target, alternatives, rationale. target is the Chinese rendering you select for this video after reading nearby sentences, the actor's role, and its actions. alternatives is an empty list unless one or two other Chinese renderings are genuinely defensible in this same video context; never manufacture synonyms. rationale briefly cites the contextual evidence for the selected meaning. Handle previously unseen compounds such as retrieval agent or recommendation agent by reasoning from this transcript, not by applying a word-specific rule. For preserve rows return alternatives as an empty list and briefly explain why English should remain.",
            edition_contract,
            f"Return JSON with editorial_mode, collection_title, story_start, story_end, terminology, bilibili_chapters, and wechat_lessons. Every returned lesson has title, thesis, numeric start/end source seconds, framing (auto|speaker|slide|split), and hook_headlines. For an interview highlight also return hook_context: one natural {INTERVIEW_HOOK_CONTEXT_MIN_VISIBLE_CHARACTERS}–{INTERVIEW_HOOK_CONTEXT_DRAFT_TARGET_MAX_VISIBLE_CHARACTERS} visible-character fixed explanatory subtitle placed persistently below the hook for the entire clip. This range is a drafting target; deterministic validation uses the real rendered three-line panel rather than a raw character cutoff. Together, the title and hook_context must let the audience grasp the full conflict, mechanism, and resolution; do not split essential meaning across rotating headlines or spoken captions. Write it like a sharp Chinese tech editor explaining the point to a colleague: short subject-verb clauses, spoken cause-and-effect, and concrete actors/actions. The hook states the tension; hook_context must advance the story rather than repeat it. When the selected source gives an answer, remedy, alternative, or decision, hook_context must include that payoff. It should be understandable at first glance, not read like a translated report or academic abstract. For example, prefer 'SaaS 想靠 API 涨价补收入；客户只愿为结果付费，否则就把数据搬走' over '当厂商因席位减少而提高价格时，反而会激励客户将数据移出系统'. Use split when the source simultaneously shows a speaker pane and a slide pane; this preserves the complete left speaker instead of treating the slide crop as the whole frame. Use slide only for a true slide-only shot or when the crop retains all meaningful content.",
            "Return at most 12 terminology rows, using only terms that occur inside the selected source range. Do not translate or reproduce the transcript in this planning response.",
            (
                "Validated reusable guidance from self-audit: " + self.runtime_guidance
            ) if self.runtime_guidance else "",
            (
                "Advisory from YouTube Ask or a human editor (use it only to locate a stronger source-backed angle; "
                "verify every visible claim against the supplied transcript and omit unsupported details): "
                + editorial_guidance.strip()
            ) if editorial_guidance and editorial_guidance.strip() else "",
            "For interview highlights there is no minimum duration: a short, strong, self-contained answer is best. Prefer passages at or below 180 seconds. A strong passage may extend to 300 seconds when trimming would weaken or break the complete idea; essential_context_justification is useful audit metadata but is not required. Never pad a clip to reach a target length.",
            "hook_headlines contains exactly three distinct, evidence-faithful Simplified Chinese headlines of 8–36 visible characters for an interview highlight, otherwise 8–30: one tension or contrarian claim, one concrete technical choice with stakes, and one consequence or outcome claim. Make each one specific enough to earn the next 8 seconds, not merely label the topic. Every headline must be a complete natural Chinese clause and keep every claim source-backed.",
            "Titles and thesis must sound originally written in natural Simplified Chinese, with direct spoken syntax and concrete cause-and-effect. Use short clauses a Chinese technology practitioner would actually say; do not mirror English subordinate-clause order. An interview title may use up to 36 visible characters when the trigger and consequence both matter; other theme titles use at most 30.",
            mandatory_layout,
            "Metadata: " + json.dumps({
                key: metadata.get(key) for key in (
                    "id", "title", "channel", "uploader", "duration", "description", "chapters",
                )
            }, ensure_ascii=False),
            "Transcript: " + json.dumps(transcript, ensure_ascii=False),
        ])
        plan, provenance = self.writer._request_json([
            {"role": "system", "content": "Return one valid JSON object only."},
            {"role": "user", "content": planning_prompt},
        ], max_tokens=12000)
        plan["editorial_mode"] = editorial_mode
        plan, plan_repairs = self.ensure_editorial_plan(
            metadata, cues, plan, editorial_mode, planning_transcript=transcript,
            editorial_guidance=editorial_guidance,
        )
        if editorial_mode in {"known_tech_interview_clip", "conference_highlights"}:
            selected_ranges = [
                source_range for raw in plan.get("wechat_lessons", [])
                if isinstance(raw, dict)
                and (source_range := _coerce_range(raw, duration)) is not None
            ]
            if editorial_mode == "conference_highlights" and not selected_ranges:
                raise ValueError("no_self_contained_conference_highlight")
            if editorial_mode == "known_tech_interview_clip":
                selected_ranges = selected_ranges[:1]
            cues[:] = [
                cue for cue in cues
                if any(
                    min(cue.end, selected.end) - max(cue.start, selected.start) >= 0.5
                    for selected in selected_ranges
                )
            ]
        if editorial_mode == "known_tech_interview_clip":
            selected = _coerce_range(plan["wechat_lessons"][0], duration)
            if selected is None:
                raise ValueError("interview highlight has no valid source range")
        terminology = self._parse_terminology(plan.get("terminology", []), cues)
        traces: list[dict[str, Any]] = [
            planning_input_trace,
            {"step": "translation_plan", "provenance": provenance}, *plan_repairs,
        ]
        for cue in cues:
            cue.translation = ""
        traces.append({
            "step": (
                "interview_translation_deferred_until_audio_alignment"
                if editorial_mode == "known_tech_interview_clip"
                else "subtitle_translation_deferred_until_joint_boundaries"
            ),
            "cue_ids": [cue.id for cue in cues],
        })
        return terminology, plan, traces


    def audit_interview_directing(
        self, plan: dict[str, Any], cues: list[TranscriptCue], duration: float,
    ) -> dict[str, Any]:
        """Select one complete headline/context package without mixing candidates."""
        rows = plan.get("wechat_lessons")
        row = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None
        source_range = _coerce_range(row, duration) if row else None
        if row is None or source_range is None:
            raise ValueError("interview directing audit requires one valid selected clip")
        selected_cues = [
            cue for cue in cues
            if cue.end > source_range.start and cue.start < source_range.end
        ]
        selected_text = " ".join(cue.source_text for cue in selected_cues)
        current_hooks = [
            str(value).strip() for value in row.get("hook_headlines", [])
            if str(value).strip()
        ] if isinstance(row.get("hook_headlines"), list) else []
        existing_context = str(row.get("hook_context") or "").strip()
        incumbent_headline = current_hooks[0] if current_hooks else str(row.get("title") or "").strip()
        incumbent_valid = (
            8 <= len(re.sub(r"\s+", "", incumbent_headline)) <= 36
            and len(re.sub(r"\s+", "", existing_context))
            >= INTERVIEW_HOOK_CONTEXT_MIN_VISIBLE_CHARACTERS
            and _interview_hook_context_fits_overlay(existing_context)
            and not _interview_title_entailment_errors(incumbent_headline, selected_text)
            and not political_markers(incumbent_headline)
        )
        prompt = "\n".join([
            "You are the final directing critic for a BGM-only Chinese technical-intelligence short video.",
            "Positioning: this is a fast radar for Chinese developers, AI builders, tech leads, and product leaders—not a lecture or a corporate news brief.",
            "The first screen is one indivisible editorial unit: a short headline plus a fixed explanatory subtitle. Never optimize either field in isolation or combine a headline from one idea with the subtitle from another.",
            "Return one JSON object with critique and hook_options. hook_options must contain exactly three objects, each with headline and context. Every pair must independently let a viewer grasp the full conflict, mechanism, and resolution without relying on rotating headlines or spoken captions.",
            "Each headline must be a distinct, source-supported, natural Chinese line of 8–36 visible characters. Optimize it for an immediate scroll-stop: one memorable tension, recognizable actor, number, reversal, or consequence, readable in under two seconds.",
            f"Each context must have at least {INTERVIEW_HOOK_CONTEXT_MIN_VISIBLE_CHARACTERS} visible characters, fit the fixed three-line panel, and aim for no more than {INTERVIEW_HOOK_CONTEXT_DRAFT_TARGET_MAX_VISIBLE_CHARACTERS} visible characters. It must explain the answer or payoff rather than repeat its own headline.",
            "Do not change the title, clip boundaries, speaker identity, facts, or story angle.",
            "Title: " + str(row.get("title") or ""),
            "Thesis: " + str(row.get("thesis") or ""),
            "Current first-screen pair: " + json.dumps({
                "headline": incumbent_headline, "context": existing_context,
            }, ensure_ascii=False),
            "Selected source transcript: " + json.dumps([
                {"id": cue.id, "start": cue.start, "end": cue.end, "text": cue.source_text}
                for cue in selected_cues
            ], ensure_ascii=False),
        ])
        # Translation recovery and directing taste are separate jobs. The
        # factory supplies an independent directing model when configured;
        # otherwise retain the primary editorial model as a compatibility
        # fallback instead of the sticky translation transport fallback.
        critic = self.directing_writer or getattr(self.writer, "primary", self.writer)
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                draft, provenance = critic._request_json([
                    {"role": "system", "content": "Return one valid JSON object only."},
                    {"role": "user", "content": prompt},
                ], max_tokens=6000)
                raw_options = draft.get("hook_options")
                if not isinstance(raw_options, list) or len(raw_options) != 3:
                    raise ValueError("directing critic must return exactly three hook packages")
                candidates: list[dict[str, str]] = []
                if incumbent_valid:
                    candidates.append({
                        "id": "incumbent", "headline": incumbent_headline,
                        "context": existing_context,
                    })
                rejected_proposals: list[dict[str, str]] = []
                for index, option in enumerate(raw_options, start=1):
                    headline = str(option.get("headline") or "").strip() \
                        if isinstance(option, dict) else ""
                    context = str(option.get("context") or "").strip() \
                        if isinstance(option, dict) else ""
                    valid = (
                        8 <= len(re.sub(r"\s+", "", headline)) <= 36
                        and len(re.sub(r"\s+", "", context))
                        >= INTERVIEW_HOOK_CONTEXT_MIN_VISIBLE_CHARACTERS
                        and _interview_hook_context_fits_overlay(context)
                        and not _interview_title_entailment_errors(headline, selected_text)
                        and not political_markers(headline)
                    )
                    if not valid:
                        rejected_proposals.append({"headline": headline, "context": context})
                        continue
                    if any(
                        item["headline"] == headline and item["context"] == context
                        for item in candidates
                    ):
                        continue
                    candidates.append({
                        "id": f"proposal-{index}", "headline": headline,
                        "context": context,
                    })
                if len(candidates) < 3:
                    raise ValueError("directing comparison has fewer than three valid hook packages")
                judge = getattr(self.writer, "fallback", getattr(self.writer, "primary", self.writer))
                verdict, judge_provenance = judge._request_json([
                    {"role": "system", "content": "Return one valid JSON object only."},
                    {"role": "user", "content": "\n".join([
                        "You are the final Chinese first-screen editor for a BGM-only technical-intelligence short.",
                        "Audience: Chinese developers, AI builders, tech leads, and product leaders.",
                        "Compare complete headline/context packages. A package succeeds only when the headline attracts attention and its own context delivers the full meaning. Never mix fields between packages. Preserve the incumbent unless a challenger is clearly better as a complete unit; novelty alone is not improvement.",
                        "Return JSON with rationale and ranked_ids. ranked_ids must be exactly three distinct ids copied from Candidates, strongest first. Do not rewrite any headline or context.",
                        "Candidates: " + json.dumps(candidates, ensure_ascii=False),
                        "Selected source transcript: " + selected_text,
                    ])},
                ], max_tokens=1800)
                ranked_ids = [
                    str(value).strip() for value in verdict.get("ranked_ids", [])
                    if str(value).strip()
                ] if isinstance(verdict.get("ranked_ids"), list) else []
                by_id = {item["id"]: item for item in candidates}
                if (
                    len(ranked_ids) != 3 or len(set(ranked_ids)) != 3
                    or any(value not in by_id for value in ranked_ids)
                ):
                    raise ValueError("directing judge did not rank three supplied hook packages")
                ranked = [by_id[value] for value in ranked_ids]
                winner = ranked[0]
                row["hook_headlines"] = [item["headline"] for item in ranked]
                row["hook_context"] = winner["context"]
                row["title"] = winner["headline"]
                plan["collection_title"] = winner["headline"]
                return {
                    "step": "interview_directing_audit",
                    "policy_version": INTERVIEW_DIRECTING_POLICY_VERSION,
                    "critique": str(draft.get("critique") or "").strip(),
                    "candidate_pairs": candidates,
                    "rejected_proposals": rejected_proposals,
                    "hook_headlines": [item["headline"] for item in ranked],
                    "winning_title": winner["headline"],
                    "winning_context": winner["context"],
                    "winning_pair_id": winner["id"],
                    "judge_rationale": str(verdict.get("rationale") or "").strip(),
                    "attempt": attempt + 1,
                    "provenance": provenance,
                    "judge_provenance": judge_provenance,
                }
            except Exception as error:
                last_error = error
                prompt += (
                    "\nPrevious directing attempt failed validation: "
                    f"{type(error).__name__}: {error}. "
                    "Return three complete headline/context packages using only "
                    "concepts and stakes stated explicitly in the selected transcript."
                )
        raise RuntimeError(
            "primary interview directing audit failed; refusing fallback-quality hooks: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error



    def translate_interview_clip_once(
        self, cues: list[TranscriptCue], terminology: list[TerminologyEntry],
        source_words: list[SourceWord], alignment_fingerprint: str = "",
        audio_hypothesis: str = "",
    ) -> dict[str, Any]:
        """Choose word boundaries and Chinese together, then repair local windows."""
        writer = self.writer if hasattr(self.writer, "_request_json") else None
        reviewer = self.subtitle_reviewer or self.directing_writer
        if writer is None or reviewer is None or reviewer is writer:
            raise RuntimeError(
                "interview joint translation requires distinct writer and reviewer calls"
            )
        words = [word for word in source_words if word.raw]
        if not words or not cues:
            raise ValueError("interview joint translation has no source words")
        expected_source = re.sub(r"\s+", " ", " ".join(
            cue.source_text for cue in cues
        )).strip()
        actual_source = re.sub(r"\s+", " ", " ".join(
            word.raw for word in words
        )).strip()
        if expected_source != actual_source:
            raise ValueError("source-word ledger does not exactly reconstruct interview text")

        clip_start = cues[0].start
        clip_end = cues[-1].end
        glossary = _relevant_terminology_prompt_rows(terminology, expected_source)
        attempts: list[dict[str, Any]] = []
        hypothesis_words = re.findall(r"\S+", audio_hypothesis)
        hypothesis_norm = [
            re.sub(r"[^a-z0-9']+", "", value.casefold().replace("’", "'"))
            for value in hypothesis_words
        ]
        source_norm = [word.normalized for word in words]
        audio_conflict_entities = {
            entity for entity in _uppercase_source_entities(expected_source)
            if audio_hypothesis and not _contains_term(audio_hypothesis, entity)
        }
        source_numbers = set(re.findall(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)*(?:%|x)?", expected_source))
        hypothesis_numbers = set(re.findall(
            r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)*(?:%|x)?", audio_hypothesis,
        ))
        audio_conflict_numbers = (
            source_numbers - hypothesis_numbers if audio_hypothesis else set()
        )
        audio_only_numbers = (
            hypothesis_numbers - source_numbers if audio_hypothesis else set()
        )
        numeric_conflict_pairs = _numeric_audio_conflict_pairs(words, audio_hypothesis)
        paired_source_number_indices = {
            int(row["source_word_index"]) - 1 for row in numeric_conflict_pairs
        }
        hypothesis_blocks = SequenceMatcher(
            None, source_norm, hypothesis_norm, autojunk=False,
        ).get_matching_blocks() if hypothesis_norm else []

        def audio_evidence_for_window(start_word: int, end_word: int) -> str:
            matched_hypothesis_indices = [
                block.b + offset
                for block in hypothesis_blocks if block.size
                for offset in range(block.size)
                if start_word <= block.a + offset < end_word
            ]
            if not matched_hypothesis_indices:
                return ""
            start = max(0, min(matched_hypothesis_indices) - 16)
            end = min(len(hypothesis_words), max(matched_hypothesis_indices) + 17)
            return " ".join(hypothesis_words[start:end])

        def request_window(
            start_word: int, end_word: int, previous_source: str,
            next_source: str, rejection: str, repair_round: int,
            internal_attempt: int = 0, minimum_card_count: int = 0,
            mandatory_split_ranges: tuple[tuple[int, int], ...] = (),
            required_boundaries: tuple[int, ...] = (),
            freeze_boundaries: bool = False,
        ) -> tuple[list[tuple[int, int, str]], dict[str, Any]]:
            window = words[start_word:end_word]
            duration = max(0.1, window[-1].end - window[0].start)
            maximum_count = max(1, min(
                len(window), math.floor(duration / INTERVIEW_CAPTION_MIN_SECONDS),
            ))
            minimum_count = max(
                1, minimum_card_count, len(required_boundaries) + 1,
                math.ceil(duration / INTERVIEW_CAPTION_HARD_MAX_SECONDS),
            )
            if freeze_boundaries:
                maximum_count = minimum_count = len(required_boundaries) + 1
            if minimum_count > maximum_count:
                raise ValueError(
                    "joint translation window cannot satisfy both minimum duration "
                    f"and required split count: {minimum_count}>{maximum_count}"
                )
            target_count = max(
                minimum_count,
                math.ceil(duration / INTERVIEW_CAPTION_TARGET_MAX_SECONDS),
                math.ceil(len(window) / INTERVIEW_CAPTION_TARGET_MAX_ENGLISH_WORDS),
            )
            target_count = min(maximum_count, target_count)
            window_numeric_conflict_pairs = [{
                **row,
                "source_word_index": int(row["source_word_index"]) - start_word,
            } for row in numeric_conflict_pairs
                if start_word < int(row["source_word_index"]) <= end_word]
            ownership_terms = set(_uppercase_source_entities(
                " ".join(word.raw for word in window)
            ))
            ownership_terms.update(
                entry.source for entry in terminology
                if entry.strategy == TerminologyStrategy.PRESERVE
                and _contains_term(" ".join(word.raw for word in window), entry.source)
            )
            ownership: list[dict[str, Any]] = []
            for entity in sorted(ownership_terms, key=lambda value: (-len(value), value)):
                width = max(1, len(re.findall(r"\S+", entity)))
                indices = [
                    index + 1
                    for index in range(max(0, len(window) - width + 1))
                    if _contains_term(
                        " ".join(word.raw for word in window[index:index + width]),
                        entity,
                    )
                ]
                ownership.append({
                    "entity": entity,
                    "source_word_start_indices": indices,
                    "word_count": width,
                })
            # Numeric ASR exceptions belong to one source occurrence.  A value may
            # appear more than once in the passage while only one occurrence has
            # strong local anchors, so never exempt every matching value by text.
            numeric_occurrences: dict[str, list[int]] = {}
            for absolute_index in range(start_word, end_word):
                match = re.search(
                    r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)*"
                    r"(?:st|nd|rd|th)?(?:%|x)?(?![A-Za-z0-9])",
                    words[absolute_index].raw,
                    re.IGNORECASE,
                )
                if match and absolute_index not in paired_source_number_indices:
                    numeric_occurrences.setdefault(match.group(0), []).append(
                        absolute_index - start_word + 1
                    )
            ownership.extend({
                "entity": entity,
                "source_word_start_indices": indices,
                "word_count": 1,
            } for entity, indices in sorted(numeric_occurrences.items()))
            translated_term_ownership: list[dict[str, Any]] = []
            for entry in terminology:
                if entry.strategy != TerminologyStrategy.TRANSLATE or not entry.target:
                    continue
                width = max(1, len(re.findall(r"\S+", entry.source)))
                indices = [
                    index + 1
                    for index in range(max(0, len(window) - width + 1))
                    if _contains_term(
                        " ".join(word.raw for word in window[index:index + width]),
                        entry.source,
                    )
                ]
                if indices:
                    translated_term_ownership.append({
                        "source": entry.source, "required_target": entry.target,
                        "source_word_start_indices": indices, "word_count": width,
                    })
            fixed_card_rows: list[dict[str, Any]] = []
            if freeze_boundaries:
                left = 0
                for relative_end in [
                    *(boundary - start_word for boundary in required_boundaries),
                    len(window),
                ]:
                    fixed_card_rows.append({
                        "end_word": relative_end,
                        "source": " ".join(
                            word.raw for word in window[left:relative_end]
                        ),
                    })
                    left = relative_end
            prompt = "\n".join([
                "Create publication-ready bilingual cards for this ordered technology-video speech passage. Choose English word boundaries and write the matching natural Simplified Chinese in the same response. This is the only normal-path translation pass.",
                "Return {cards:[{end_word,text}]}. end_word is an exclusive cumulative index into Source words beginning at 1. It must increase strictly and the final value must equal the supplied word count. Never rewrite, omit, duplicate, or reorder an English word.",
                f"This exact source window contains {len(window)} words. The final card's end_word MUST be exactly {len(window)}; {len(window) - 1} or any smaller value is invalid and omits source evidence.",
                "A sentence may continue across adjacent cards. Cut at a natural speech or semantic boundary without stranding a preposition, noun phrase, condition, or entity. Chinese must preserve actors, actions, negation, uncertainty, entities, cause-effect, scope, and numbers except where the supplied audio-conflict evidence strongly supports a different audible number. Remove hesitation and duplicated speech noise without inventing facts.",
                "Do not translate discourse fillers, emphatic sound interjections, or sound descriptions. Omit um/uh/erm/hmm, filler uses of like/well/so, you know, I mean, and an emphatic 'boom' from Chinese while preserving the substantive clause. Never render [laughter], coughing, throat clearing, music, applause, breathing, or similar caption metadata in Chinese.",
                "Entity ownership is strict: a product, acronym, company, API, or number may appear in Chinese only when that same card's English word range contains it. Numeric tokens explicitly listed as audio conflicts below are the exception: use the strongly supported nearby audible value on that same semantic card. If a rejected Chinese card moved an entity from a neighbor, either move the English boundary to include the entity or remove it from that Chinese card; never repeat it on both cards.",
                "The Entity ownership table below is deterministic evidence. Each listed entity may appear only in a Chinese card whose inclusive source range contains one of its word indices. If feedback says moved:X, remove X from the wrong Chinese card or move the boundary over X. If feedback says missing:X, preserve X in its owner card.",
                "Use the terminology decision and its evidence. alternatives are candidates for the independent reviewer, not a whitelist. A longer phrase owns nested shorter terms. Keep one rendering consistent across the passage.",
                "Translated terminology ownership: " + json.dumps(
                    translated_term_ownership, ensure_ascii=False,
                ) + ". For every listed occurrence, put required_target verbatim in the Chinese card that owns the source phrase; never shift it to a neighbor.",
                f"Return {minimum_count}–{maximum_count} cards; aim for {target_count}. Every card must last {INTERVIEW_CAPTION_MIN_SECONDS:g}–{INTERVIEW_CAPTION_HARD_MAX_SECONDS:g} seconds, contain no more than {INTERVIEW_CAPTION_MAX_ENGLISH_WORDS} English words, remain below {INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND:g} reading units per second, and fit the measured three-line panel. A CJK glyph, a Latin word, or punctuation is one reading unit, so do not count every letter of an English product name separately. For each end_word, the display cut occurs at the midpoint between the previous word's end and the next word's start; calculate durations from the supplied times and avoid sub-{INTERVIEW_CAPTION_MIN_SECONDS:g}-second cards. Aim for {INTERVIEW_CAPTION_TARGET_MAX_CHINESE_CHARACTERS} reading units, but use rendered fit and reading speed rather than a raw hard ceiling.",
                "Previous and next source are context only. Never import their claims into this window.",
                "Audio ASR is conflict evidence only and never replaces the authoritative Source words. When Source contains an obvious transcription ambiguity, use the nearby audio hypothesis plus surrounding actions to translate the most supported audible meaning while leaving the returned English boundaries unchanged. Do not invent a correction when the audio evidence is also uncertain.",
                "Terminology: " + json.dumps(glossary, ensure_ascii=False),
                "Entity ownership: " + json.dumps(ownership, ensure_ascii=False),
                "Mandatory split ranges: " + json.dumps([{
                    "source_word_range": [left - start_word, right - start_word],
                    "required_end_word_between": [
                        left - start_word + 1, right - start_word - 1,
                    ],
                } for left, right in mandatory_split_ranges], ensure_ascii=False)
                + ". Every listed rejected range must contain at least one new end_word strictly inside it; adding a boundary only in a neighboring card does not repair the failure.",
                "Fixed required end_word boundaries: " + json.dumps([
                    boundary - start_word for boundary in required_boundaries
                ]) + ". Include every listed value exactly as an end_word. These deterministic boundaries are supplied only after repeated failure to split a rejected overlong card.",
                "Fixed card translation rows: " + json.dumps(
                    fixed_card_rows, ensure_ascii=False,
                ) + ". When this list is non-empty, return exactly these end_word values and translate each row's source only into its matching Chinese text. Do not move a clause, entity, number, or terminology target into an earlier or later row.",
                "Source entities absent or contradicted in audio ASR (the semantic reviewer decides their meaning; do not mechanically copy them into Chinese): "
                + json.dumps(sorted(audio_conflict_entities), ensure_ascii=False),
                "Numeric conflict evidence: " + json.dumps({
                    "source_only_numbers": sorted(audio_conflict_numbers),
                    "audio_only_candidate_numbers": sorted(audio_only_numbers),
                    "locally_aligned_pairs": window_numeric_conflict_pairs,
                }, ensure_ascii=False)
                + ". Authoritative English stays unchanged. Use an audio candidate only from locally_aligned_pairs on the matching semantic card. Global unpaired values are diagnostic only; retain the source value when no local pair exists.",
                "Nearby audio ASR hypothesis: " + audio_evidence_for_window(
                    start_word, end_word,
                ),
                ("Validated reusable guidance: " + self.runtime_guidance) if self.runtime_guidance else "",
                "Previous source: " + previous_source,
                "Source words: " + json.dumps([{
                    "index": index, "word": word.raw,
                    "start": round(word.start, 3), "end": round(word.end, 3),
                    "alignment_status": word.alignment_status,
                    "confidence": word.confidence,
                } for index, word in enumerate(window, start=1)], ensure_ascii=False),
                "Next source: " + next_source,
                (
                    "Previous rejection (mandatory: change every rejected wording or "
                    "boundary; never return the rejected Chinese unchanged): " + rejection
                ) if rejection else "",
            ])
            draft, provenance = writer._request_json([
                {"role": "system", "content": "Return one valid JSON object only."},
                {"role": "user", "content": prompt},
            ], max_tokens=9000 if repair_round == 0 else 3500)
            rows = draft.get("cards")
            if not isinstance(rows, list) or not minimum_count <= len(rows) <= maximum_count:
                if repair_round > 0 and internal_attempt < 2:
                    return request_window(
                        start_word, end_word, previous_source, next_source,
                        rejection + (
                            "\nDeterministic structure failure: return between "
                            f"{minimum_count} and {maximum_count} cards; got "
                            f"{len(rows) if isinstance(rows, list) else type(rows).__name__}."
                        ),
                        repair_round, internal_attempt=internal_attempt + 1,
                        minimum_card_count=minimum_count,
                        mandatory_split_ranges=mandatory_split_ranges,
                        required_boundaries=required_boundaries,
                        freeze_boundaries=freeze_boundaries,
                    )
                raise ValueError(
                    f"joint translation returned invalid card count for words "
                    f"{start_word}:{end_word}"
                )
            ends = [row.get("end_word") if isinstance(row, dict) else None for row in rows]
            if (
                any(type(value) is not int for value in ends)
                or ends[-1] != len(window)
                or any(left >= right for left, right in zip((0, *ends[:-1]), ends))
            ):
                if repair_round > 0 and internal_attempt < 2:
                    return request_window(
                        start_word, end_word, previous_source, next_source,
                        rejection + (
                            "\nDeterministic structure failure: end_word must increase "
                            f"strictly and finish at {len(window)}; got {ends}."
                        ),
                        repair_round, internal_attempt=internal_attempt + 1,
                        minimum_card_count=minimum_count,
                        mandatory_split_ranges=mandatory_split_ranges,
                        required_boundaries=required_boundaries,
                        freeze_boundaries=freeze_boundaries,
                    )
                raise ValueError(
                    f"joint translation returned invalid word boundaries: {ends}"
                )
            absolute_boundaries = {
                start_word + value for value in ends[:-1]
            }
            expected_frozen_ends = [
                *(boundary - start_word for boundary in required_boundaries),
                len(window),
            ]
            if freeze_boundaries and ends != expected_frozen_ends:
                if internal_attempt < 3:
                    return request_window(
                        start_word, end_word, previous_source, next_source,
                        rejection + (
                            "\nDeterministic frozen-boundary failure: return exactly "
                            + json.dumps(expected_frozen_ends)
                        ),
                        repair_round, internal_attempt=internal_attempt + 1,
                        minimum_card_count=minimum_count,
                        mandatory_split_ranges=mandatory_split_ranges,
                        required_boundaries=required_boundaries,
                        freeze_boundaries=True,
                    )
                raise ValueError(
                    "joint translation changed frozen source boundaries: "
                    + json.dumps(ends)
                )
            missing_required = sorted(
                boundary for boundary in required_boundaries
                if boundary not in absolute_boundaries
            )
            if missing_required:
                if repair_round > 0 and internal_attempt < 3:
                    return request_window(
                        start_word, end_word, previous_source, next_source,
                        rejection + (
                            "\nDeterministic boundary failure: include these exact "
                            "absolute source boundaries: "
                            + json.dumps(missing_required)
                        ),
                        repair_round, internal_attempt=internal_attempt + 1,
                        minimum_card_count=minimum_count,
                        mandatory_split_ranges=mandatory_split_ranges,
                        required_boundaries=required_boundaries,
                        freeze_boundaries=freeze_boundaries,
                    )
                raise ValueError(
                    "joint translation omitted fixed source boundaries: "
                    + json.dumps(missing_required)
                )
            unsplit = [
                (left, right) for left, right in mandatory_split_ranges
                if not any(left < boundary < right for boundary in absolute_boundaries)
            ]
            if unsplit:
                if repair_round > 0 and internal_attempt < 2:
                    return request_window(
                        start_word, end_word, previous_source, next_source,
                        rejection + (
                            "\nDeterministic boundary failure: add an end_word "
                            "strictly inside each rejected source range: "
                            + json.dumps(unsplit)
                        ),
                        repair_round, internal_attempt=internal_attempt + 1,
                        minimum_card_count=minimum_count,
                        mandatory_split_ranges=mandatory_split_ranges,
                        required_boundaries=required_boundaries,
                        freeze_boundaries=freeze_boundaries,
                    )
                if repair_round > 0 and not required_boundaries:
                    forced: list[int] = []
                    for left, right in unsplit:
                        candidates = list(range(left + 1, right))
                        if not candidates:
                            continue
                        midpoint = (words[left].start + words[right - 1].end) / 2
                        viable = [
                            boundary for boundary in candidates
                            if (
                                (words[boundary - 1].end + words[boundary].start) / 2
                                - words[left].start
                            ) >= INTERVIEW_CAPTION_MIN_SECONDS
                            and (
                                words[right - 1].end
                                - (words[boundary - 1].end + words[boundary].start) / 2
                            ) >= INTERVIEW_CAPTION_MIN_SECONDS
                        ]
                        pool = viable or candidates
                        forced.append(min(pool, key=lambda boundary: abs(
                            (words[boundary - 1].end + words[boundary].start) / 2
                            - midpoint
                        )))
                    if forced:
                        return request_window(
                            start_word, end_word, previous_source, next_source,
                            rejection + (
                                "\nDeterministic self-healing selected fixed internal "
                                "boundaries after repeated noncompliance: "
                                + json.dumps(forced)
                            ),
                            repair_round, internal_attempt=internal_attempt + 1,
                            minimum_card_count=max(minimum_count, len(forced) + 1),
                            mandatory_split_ranges=mandatory_split_ranges,
                            required_boundaries=tuple(forced),
                            freeze_boundaries=False,
                        )
                raise ValueError(
                    "joint translation did not split rejected source ranges: "
                    + json.dumps(unsplit)
                )
            spans: list[tuple[int, int, str]] = []
            left = 0
            for row, right in zip(rows, ends):
                text_value = str(row.get("text") or "").strip()
                if not text_value:
                    if repair_round > 0 and internal_attempt < 2:
                        return request_window(
                            start_word, end_word, previous_source, next_source,
                            rejection + "\nDeterministic structure failure: no Chinese card may be empty.",
                            repair_round, internal_attempt=internal_attempt + 1,
                            minimum_card_count=minimum_count,
                            mandatory_split_ranges=mandatory_split_ranges,
                            required_boundaries=required_boundaries,
                            freeze_boundaries=freeze_boundaries,
                        )
                    raise ValueError("joint translation returned an empty Chinese card")
                spans.append((start_word + left, start_word + right, text_value))
                left = right
            if repair_round > 0:
                proposed_cards = materialize(spans)
                local_errors = interview_caption_duration_errors(
                    proposed_cards, terminology, audio_conflict_entities,
                )
                for card in proposed_cards:
                    local_errors.extend(
                        f"{card.id}:{error}"
                        for error in _semantic_card_translation_errors({
                            "id": card.id, "source": card.source_text,
                            "duration_seconds": card.duration,
                        }, card.translation, terminology, require_punctuation=False,
                           audio_conflict_entities=audio_conflict_entities)
                    )
                if local_errors and internal_attempt == 0:
                    split_failure = any(
                        "hard maximum" in error
                        or "English words" in error
                        or "does not fit the three-line subtitle panel" in error
                        for error in local_errors
                    )
                    stricter_minimum = (
                        min(maximum_count, len(spans) + 1)
                        if split_failure and len(spans) < maximum_count
                        else minimum_count
                    )
                    retry_feedback = json.dumps({
                        "errors": local_errors,
                        "rejected_cards": [{
                            "source": card.source_text,
                            "chinese": card.translation,
                            "duration": round(card.duration, 3),
                        } for card in proposed_cards],
                    }, ensure_ascii=False)
                    return request_window(
                        start_word, end_word, previous_source, next_source,
                        retry_feedback, repair_round, internal_attempt=1,
                        minimum_card_count=stricter_minimum,
                        mandatory_split_ranges=mandatory_split_ranges,
                        required_boundaries=required_boundaries,
                        freeze_boundaries=freeze_boundaries,
                    )
            return spans, provenance

        def materialize(
            spans: list[tuple[int, int, str]],
        ) -> list[TranscriptCue]:
            cards: list[TranscriptCue] = []
            for index, (start_word, end_word, chinese) in enumerate(spans, start=1):
                if start_word == 0:
                    start = clip_start
                else:
                    start = (words[start_word - 1].end + words[start_word].start) / 2
                if end_word == len(words):
                    end = clip_end
                else:
                    end = (words[end_word - 1].end + words[end_word].start) / 2
                source_text = " ".join(
                    word.raw for word in words[start_word:end_word]
                )
                translation = omit_spoken_fillers_from_translation(
                    source_text, normalize_chinese_subtitle(chinese),
                )
                cards.append(TranscriptCue(
                    id=f"interview-card-{index}", start=round(start, 3),
                    end=round(end, 3),
                    source_text=source_text, translation=translation,
                    source_tokens=[{
                        "raw": word.raw,
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                    } for word in words[start_word:end_word]],
                    confidence=min((
                        word.confidence for word in words[start_word:end_word]
                        if word.confidence is not None
                    ), default=None),
                ))
            return cards

        def coalesce_mechanical_boundary_failures(
            spans: list[tuple[int, int, str]],
        ) -> tuple[list[tuple[int, int, str]], list[dict[str, Any]]]:
            """Merge only failures whose correct neighbor is mechanically knowable."""
            normalized = list(spans)
            changes: list[dict[str, Any]] = []
            while len(normalized) > 1:
                cards = materialize(normalized)
                target: int | None = next((
                    index for index, card in enumerate(cards)
                    if card.duration < INTERVIEW_CAPTION_MIN_SECONDS - 1e-6
                    and not (
                        card.duration >= INTERVIEW_CAPTION_BRIEF_ACK_MIN_SECONDS - 1e-6
                        and len(card.source_text.split()) <= 2
                        and len(re.sub(r"\s+", "", card.translation)) <= 4
                    )
                ), None)
                reason = "minimum_duration"
                preferred_neighbors: list[int] = []
                if target is None:
                    for index, card in enumerate(cards):
                        moved = [
                            value.split(":", 1)[1]
                            for value in _caption_entity_alignment_errors(
                                card.source_text, card.translation, terminology,
                                audio_conflict_entities,
                            ) if value.startswith("moved:")
                        ]
                        for entity in moved:
                            owners = [
                                owner for owner, candidate in enumerate(cards)
                                if owner != index
                                and _contains_term(candidate.source_text, entity)
                            ]
                            if owners:
                                owner = min(owners, key=lambda value: abs(value - index))
                                target = index
                                preferred_neighbors = [
                                    index - 1 if owner < index else index + 1
                                ]
                                reason = f"entity_owner:{entity}"
                            if target is not None:
                                break
                        if target is not None:
                            break
                if target is None:
                    break
                candidates = preferred_neighbors or [
                    index for index in (target - 1, target + 1)
                    if 0 <= index < len(normalized)
                ]
                accepted: tuple[int, list[tuple[int, int, str]], TranscriptCue] | None = None
                for neighbor in sorted(
                    candidates, key=lambda index: cards[index].duration,
                ):
                    left = min(target, neighbor)
                    right = max(target, neighbor)
                    merged_text = normalize_chinese_subtitle(
                        normalized[left][2].rstrip("，；,; ")
                        + normalized[right][2].lstrip()
                    )
                    proposed = [
                        *normalized[:left],
                        (normalized[left][0], normalized[right][1], merged_text),
                        *normalized[right + 1:],
                    ]
                    merged_card = materialize(proposed)[left]
                    remaining_entity_errors = _caption_entity_alignment_errors(
                        merged_card.source_text, merged_card.translation, terminology,
                        audio_conflict_entities,
                    )
                    allow_step_toward_entity_owner = (
                        reason.startswith("entity_owner:")
                        and all(value.startswith("moved:") for value in remaining_entity_errors)
                    )
                    if (
                        merged_card.duration <= INTERVIEW_CAPTION_HARD_MAX_SECONDS + 1e-6
                        and len(merged_card.source_text.split())
                        <= INTERVIEW_CAPTION_MAX_ENGLISH_WORDS
                        and _subtitle_reading_units(merged_card.translation)
                        / max(merged_card.duration, 0.1)
                        <= INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND
                        and not interview_caption_layout_errors([merged_card])
                        and (
                            not remaining_entity_errors
                            or allow_step_toward_entity_owner
                        )
                    ):
                        accepted = left, proposed, merged_card
                        break
                if accepted is None:
                    break
                left, normalized, merged_card = accepted
                changes.append({
                    "reason": reason,
                    "merged_at_card": left + 1,
                    "resulting_source": merged_card.source_text,
                    "resulting_duration": round(merged_card.duration, 3),
                })
            return normalized, changes

        def validate_and_review(
            spans: list[tuple[int, int, str]], repair_round: int,
        ) -> tuple[list[TranscriptCue], dict[int, list[str]], dict[str, Any] | None]:
            cards = materialize(spans)
            problems: dict[int, list[str]] = {}
            for error in interview_caption_duration_errors(
                cards, terminology, audio_conflict_entities,
            ):
                match = re.match(r"interview-card-(\d+)", error)
                if match:
                    problems.setdefault(int(match.group(1)) - 1, []).append(error)
            for index, card in enumerate(cards):
                semantic_errors = _semantic_card_translation_errors({
                    "id": card.id, "source": card.source_text,
                    "duration_seconds": card.duration,
                }, card.translation, terminology, require_punctuation=False,
                   audio_conflict_entities=audio_conflict_entities)
                if semantic_errors:
                    problems.setdefault(index, []).extend(semantic_errors)
                left, right, _ = spans[index]
                numeric_errors = _caption_numeric_alignment_errors(
                    words, left, right, card.translation,
                    numeric_conflict_pairs,
                )
                if numeric_errors:
                    problems.setdefault(index, []).extend(numeric_errors)
            if problems:
                return cards, problems, None
            expected = {card.id for card in cards}

            def partition_indices() -> list[tuple[int, int]]:
                partitions: list[tuple[int, int]] = []
                start = 0
                while start < len(cards):
                    end = min(len(cards), start + 24)
                    while end > start + 1 and len(json.dumps([
                        {"id": card.id, "source": card.source_text,
                         "chinese": card.translation}
                        for card in cards[start:end]
                    ], ensure_ascii=False)) > 10000:
                        end -= 1
                    if end < len(cards) and end - start > 12:
                        sentence_ends = [
                            index for index in range(start + 12, end)
                            if re.search(r"[.!?][\"')\]]?$", cards[index - 1].source_text)
                        ]
                        if sentence_ends:
                            end = sentence_ends[-1]
                    partitions.append((start, end))
                    start = end
                return partitions

            partitions = partition_indices()
            by_id: dict[str, dict[str, Any]] = {}
            local_review_traces: list[dict[str, Any]] = []
            for partition_number, (core_start, core_end) in enumerate(partitions, start=1):
                core = cards[core_start:core_end]
                context = cards[max(0, core_start - 2):core_start] + cards[
                    core_end:min(len(cards), core_end + 2)
                ]
                core_expected = {card.id for card in core}
                attempt_provenances: list[dict[str, Any]] = []
                structure_failures: list[list[str]] = []
                for review_attempt in range(2):
                    reviewed, attempt_provenance = reviewer._request_json([
                        {"role": "system", "content": "Return one valid JSON object only."},
                        {"role": "user", "content": "\n".join([
                            "Independently review this ordered bilingual subtitle partition. Do not rewrite it. Context rows are read-only and must not appear in the response. A sentence may continue naturally across cards; do not require every card to be a standalone proposition.",
                            "Reject only material omissions, inventions, changed modality or scope, misplaced entities, misleading cuts, incomprehensible Chinese, or a contextual term unsupported by the actual actor and actions. Accept faithful natural paraphrases. A preference, optional nuance, or stylistic suggestion is not enough to fail a card. alternatives are candidates, not automatic approvals.",
                            "The supplied English is the authoritative publishable wording and may contain ASR spelling or spacing artifacts. Never demand that Chinese preserve a broken fragment literally or that English be corrected. Use nearby audio and context to judge the spoken meaning. Natural Chinese may make an implicit head noun or relation explicit when it does not add a new factual claim.",
                            "Reject Chinese that translates discourse fillers or sound metadata. Their omission is intentional and is not a semantic omission.",
                            "Return every core id exactly once as {reviews:[{id,pass,fidelity_score,naturalness_score,errors}]}. Pass only when both scores are at least 4.",
                            "Terminology: " + json.dumps(glossary, ensure_ascii=False),
                            "Local audio ASR hypothesis (conflict evidence only): "
                            + audio_evidence_for_window(
                                spans[core_start][0], spans[core_end - 1][1],
                            ),
                            "Local numeric conflict evidence: " + json.dumps([
                                row for row in numeric_conflict_pairs
                                if spans[core_start][0] < int(row["source_word_index"])
                                <= spans[core_end - 1][1]
                            ], ensure_ascii=False),
                            (
                                "The previous reviewer response failed deterministic structure validation: "
                                + json.dumps(structure_failures[-1], ensure_ascii=False)
                                + ". Review the unchanged core rows again and return the exact schema."
                            ) if structure_failures else "",
                            "Context rows: " + json.dumps([{
                                "id": card.id, "source": card.source_text,
                                "chinese": card.translation,
                            } for card in context], ensure_ascii=False),
                            "Rows: " + json.dumps([{
                                "id": card.id, "source": card.source_text,
                                "chinese": card.translation,
                            } for card in core], ensure_ascii=False),
                        ])},
                    ], max_tokens=min(8000, 500 + 220 * len(core)))
                    attempt_provenances.append(attempt_provenance)
                    rows = reviewed.get("reviews")
                    structure_errors: list[str] = []
                    if not isinstance(rows, list):
                        structure_errors.append("reviews must be an array")
                        candidate_rows: list[dict[str, Any]] = []
                    else:
                        candidate_rows = [row for row in rows if isinstance(row, dict)]
                        if len(candidate_rows) != len(rows):
                            structure_errors.append("every review must be an object")
                    ids = [str(row.get("id") or "") for row in candidate_rows]
                    if (
                        len(ids) != len(core_expected)
                        or len(set(ids)) != len(ids)
                        or set(ids) != core_expected
                    ):
                        structure_errors.append(
                            "review ids must contain every core id exactly once"
                        )
                    for row in candidate_rows:
                        row_id = str(row.get("id") or "")
                        if type(row.get("pass")) is not bool:
                            structure_errors.append(f"{row_id}: pass must be boolean")
                        for score_name in ("fidelity_score", "naturalness_score"):
                            if not 1 <= _review_score_out_of_five(row.get(score_name)) <= 5:
                                structure_errors.append(
                                    f"{row_id}: {score_name} must be between 1 and 5"
                                )
                        if not isinstance(row.get("errors"), list):
                            structure_errors.append(f"{row_id}: errors must be an array")
                    if not structure_errors:
                        by_id.update({str(row["id"]): row for row in candidate_rows})
                        break
                    structure_failures.append(structure_errors)
                else:
                    raise ValueError(
                        "independent subtitle reviewer returned invalid partition structure twice: "
                        + json.dumps(structure_failures, ensure_ascii=False)
                    )
                local_review_traces.append({
                    "partition": partition_number,
                    "core_card_ids": [card.id for card in core],
                    "context_card_ids": [card.id for card in context],
                    "attempts": attempt_provenances,
                    "structure_failures": structure_failures,
                })

            global_trace: dict[str, Any] | None = None
            if len(partitions) > 1:
                global_failures: list[list[str]] = []
                global_provenances: list[dict[str, Any]] = []
                for global_attempt in range(2):
                    verdict, global_provenance = reviewer._request_json([
                        {"role": "system", "content": "Return one valid JSON object only."},
                        {"role": "user", "content": "\n".join([
                            "Check only cross-partition consistency in this already locally reviewed bilingual subtitle sequence. Do not rewrite it.",
                            "Check consistent contextual terminology, entity ownership, adjacent boundary meaning, and cross-card duplication or omission. Return {pass,issues:[{ids,errors}]}. pass=true requires an empty issues array. Every issue id must come from Sequence.",
                            "The English sequence is authoritative and may retain ASR spelling/spacing artifacts. Do not reject faithful Chinese for resolving an obvious spoken form such as 'verse' meaning 'versus', for preserving the exact English name spelling, or for adding a grammatically implicit head noun without a new factual claim. Audio ASR is read-only meaning evidence and never changes the English rows.",
                            "Terminology: " + json.dumps(glossary, ensure_ascii=False),
                            "Audio ASR hypothesis: " + audio_evidence_for_window(
                                spans[0][0], spans[-1][1],
                            ),
                            (
                                "The previous global response failed deterministic structure validation: "
                                + json.dumps(global_failures[-1], ensure_ascii=False)
                                + ". Check the unchanged sequence again and return the exact schema."
                            ) if global_failures else "",
                            "Sequence: " + json.dumps([{
                                "id": card.id, "source": card.source_text,
                                "chinese": card.translation,
                            } for card in cards], ensure_ascii=False),
                        ])},
                    ], max_tokens=3500)
                    global_provenances.append(global_provenance)
                    structure_errors: list[str] = []
                    passed = verdict.get("pass")
                    issues = verdict.get("issues")
                    if type(passed) is not bool:
                        structure_errors.append("pass must be boolean")
                    if not isinstance(issues, list):
                        structure_errors.append("issues must be an array")
                        issue_rows: list[dict[str, Any]] = []
                    else:
                        issue_rows = [row for row in issues if isinstance(row, dict)]
                        if len(issue_rows) != len(issues):
                            structure_errors.append("every issue must be an object")
                    for issue in issue_rows:
                        ids = issue.get("ids")
                        errors = issue.get("errors")
                        if (
                            not isinstance(ids, list) or not ids
                            or not set(map(str, ids)) <= expected
                        ):
                            structure_errors.append("issue ids must be non-empty sequence ids")
                        if not isinstance(errors, list) or not errors:
                            structure_errors.append("issue errors must be a non-empty array")
                    if passed is True and issue_rows:
                        structure_errors.append("pass=true requires no issues")
                    if passed is False and not issue_rows:
                        structure_errors.append("pass=false requires at least one issue")
                    if not structure_errors:
                        if passed is False:
                            index_by_id = {card.id: index for index, card in enumerate(cards)}
                            for issue in issue_rows:
                                for card_id in map(str, issue["ids"]):
                                    problems.setdefault(index_by_id[card_id], []).extend(
                                        str(error) for error in issue["errors"]
                                    )
                        break
                    global_failures.append(structure_errors)
                else:
                    raise ValueError(
                        "global subtitle consistency reviewer returned invalid structure twice: "
                        + json.dumps(global_failures, ensure_ascii=False)
                    )
                global_trace = {
                    "attempts": global_provenances,
                    "structure_failures": global_failures,
                    "pass": passed,
                    "issues": issue_rows,
                }
            provenance = {
                "local_review_partitions": local_review_traces,
                "global_consistency_review": global_trace,
            }
            for index, card in enumerate(cards):
                row = by_id.get(card.id, {})
                if (
                    row.get("pass") is not True
                    or _review_score_out_of_five(row.get("fidelity_score")) < 4
                    or _review_score_out_of_five(row.get("naturalness_score")) < 4
                ):
                    raw_errors = row.get("errors", ["missing independent review"])
                    errors = [str(value) for value in raw_errors] \
                        if isinstance(raw_errors, list) else [str(raw_errors)]
                    problems.setdefault(index, []).extend(errors)
            if repair_round >= 4 and problems:
                # The translator has already had three movable-boundary repairs
                # plus one exact-card retranslation. Ask the independent reviewer
                # to adjudicate its remaining rejections once, so tentative or
                # self-contradictory review prose cannot block a valid sequence.
                failed_rows = [{
                    "id": cards[index].id,
                    "source": cards[index].source_text,
                    "chinese": cards[index].translation,
                    "prior_errors": errors,
                    "previous": ({
                        "source": cards[index - 1].source_text,
                        "chinese": cards[index - 1].translation,
                    } if index else None),
                    "next": ({
                        "source": cards[index + 1].source_text,
                        "chinese": cards[index + 1].translation,
                    } if index + 1 < len(cards) else None),
                    "audio_hypothesis": audio_evidence_for_window(
                        spans[index][0], spans[index][1],
                    ),
                } for index, errors in sorted(problems.items())]
                expected_ids = {row["id"] for row in failed_rows}
                adjudication_attempts: list[dict[str, Any]] = []
                adjudication_structure_failures: list[list[str]] = []
                for adjudication_attempt in range(2):
                    adjudication, adjudication_provenance = reviewer._request_json([
                        {"role": "system", "content": "Return one valid JSON object only."},
                        {"role": "user", "content": "\n".join([
                            "Independently adjudicate the remaining subtitle review rejections after bounded repair. Do not rewrite any text.",
                            "Judge only whether each prior error identifies a material omission, invention, changed actor/modality/scope, misplaced entity, misleading cut, or incomprehensible Chinese. A sentence may continue across cards. Dismiss preferences, optional connective wording, tentative 'may affect' concerns, and an error that says the translation has no problem.",
                            "English is authoritative but may contain ASR spelling or spacing artifacts. Use each row's audio_hypothesis and context to judge spoken meaning; do not demand literal Chinese for a broken fragment, an English spelling correction, or removal of a natural implicit Chinese head noun when no factual claim was added.",
                            "Return every id exactly once as {reviews:[{id,pass,fidelity_score,naturalness_score,errors}]}. pass=true means the unchanged card and its boundary are publication-ready and errors must be empty. pass=false requires concise material errors. Both scores must be 1–5.",
                            (
                                "The previous adjudication response failed deterministic structure validation: "
                                + json.dumps(
                                    adjudication_structure_failures[-1],
                                    ensure_ascii=False,
                                )
                            ) if adjudication_structure_failures else "",
                            "Terminology: " + json.dumps(glossary, ensure_ascii=False),
                            "Prior rejected rows: " + json.dumps(
                                failed_rows, ensure_ascii=False,
                            ),
                        ])},
                    ], max_tokens=min(5000, 600 + 260 * len(failed_rows)))
                    adjudication_attempts.append(adjudication_provenance)
                    raw_rows = adjudication.get("reviews")
                    candidate_rows = (
                        [row for row in raw_rows if isinstance(row, dict)]
                        if isinstance(raw_rows, list) else []
                    )
                    structure_errors: list[str] = []
                    ids = [str(row.get("id") or "") for row in candidate_rows]
                    if (
                        not isinstance(raw_rows, list)
                        or len(candidate_rows) != len(raw_rows)
                        or len(ids) != len(expected_ids)
                        or len(set(ids)) != len(ids)
                        or set(ids) != expected_ids
                    ):
                        structure_errors.append(
                            "adjudication must review every failed id exactly once"
                        )
                    for row in candidate_rows:
                        if type(row.get("pass")) is not bool:
                            structure_errors.append("adjudication pass must be boolean")
                        if not isinstance(row.get("errors"), list):
                            structure_errors.append("adjudication errors must be an array")
                        for score_name in ("fidelity_score", "naturalness_score"):
                            if not 1 <= _review_score_out_of_five(row.get(score_name)) <= 5:
                                structure_errors.append(
                                    f"adjudication {score_name} must be between 1 and 5"
                                )
                    if not structure_errors:
                        by_adjudicated_id = {
                            str(row["id"]): row for row in candidate_rows
                        }
                        retained: dict[int, list[str]] = {}
                        for index, prior_errors in problems.items():
                            row = by_adjudicated_id[cards[index].id]
                            if (
                                row.get("pass") is True
                                and _review_score_out_of_five(row.get("fidelity_score")) >= 4
                                and _review_score_out_of_five(row.get("naturalness_score")) >= 4
                                and not row.get("errors")
                            ):
                                continue
                            raw_errors = row.get("errors")
                            retained[index] = (
                                [str(value) for value in raw_errors]
                                if isinstance(raw_errors, list) and raw_errors
                                else prior_errors
                            )
                        problems = retained
                        break
                    adjudication_structure_failures.append(structure_errors)
                else:
                    raise ValueError(
                        "final subtitle adjudication returned invalid structure twice: "
                        + json.dumps(
                            adjudication_structure_failures, ensure_ascii=False,
                        )
                    )
                provenance["final_rejection_adjudication"] = {
                    "attempts": adjudication_attempts,
                    "structure_failures": adjudication_structure_failures,
                    "remaining_card_ids": [
                        cards[index].id for index in sorted(problems)
                    ],
                }
            return cards, problems, provenance

        def initial_windows() -> list[tuple[int, int]]:
            """Bound response size while assigning every source word exactly once."""
            windows: list[tuple[int, int]] = []
            start = 0
            while len(words) - start > 150:
                lower = start + 90
                upper = min(len(words), start + 150)
                candidates = [
                    index for index in range(lower, upper + 1)
                    if re.search(r"[.!?][\"')\]]?$", words[index - 1].raw)
                ]
                end = max(candidates) if candidates else start + 120
                windows.append((start, end))
                start = end
            windows.append((start, len(words)))
            return windows

        spans: list[tuple[int, int, str]] = []
        initial_window_traces: list[dict[str, Any]] = []
        initial_structure_failures: list[str] = []
        for window_start, window_end in initial_windows():
            initial_rejection = ""
            window_failures: list[str] = []
            for initial_attempt in range(2):
                try:
                    replacement, provenance = request_window(
                        window_start, window_end,
                        " ".join(word.raw for word in words[max(0, window_start - 16):window_start]),
                        " ".join(word.raw for word in words[window_end:min(len(words), window_end + 16)]),
                        initial_rejection, 0, internal_attempt=initial_attempt,
                    )
                    spans.extend(replacement)
                    initial_window_traces.append({
                        "source_word_window": [window_start, window_end],
                        "request_attempts": initial_attempt + 1,
                        "earlier_structure_failures": window_failures,
                        "provenance": provenance,
                    })
                    break
                except ValueError as exc:
                    failure = str(exc)
                    window_failures.append(failure)
                    initial_structure_failures.append(failure)
                    initial_rejection = (
                        "The previous response for this exact source window failed "
                        f"deterministic output validation: {exc}. Return the complete "
                        "cards array again with strictly increasing end_word values "
                        "and exact final coverage."
                    )
            else:
                diagnostic = [{
                    "round": 0, "kind": "initial_joint_translation_failed",
                    "source_word_window": [window_start, window_end],
                    "errors": window_failures,
                }]
                raise InterviewJointTranslationError(
                    "interview initial joint translation failed after one structural "
                    "self-repair: " + "; ".join(window_failures), diagnostic,
                )
        attempts.append({
            "round": 0, "kind": "initial_joint_translation",
            "window_count": len(initial_window_traces),
            "windows": initial_window_traces,
            "earlier_structure_failures": initial_structure_failures,
        })
        last_problems: dict[int, list[str]] = {}
        final_cards: list[TranscriptCue] = []
        review_provenance: dict[str, Any] | None = None
        for repair_round in range(6):
            spans, deterministic_merges = coalesce_mechanical_boundary_failures(spans)
            if deterministic_merges:
                attempts.append({
                    "round": repair_round,
                    "kind": "deterministic_boundary_coalesce",
                    "changes": deterministic_merges,
                })
            final_cards, last_problems, review_provenance = validate_and_review(
                spans, repair_round,
            )
            attempts.append({
                "round": repair_round, "kind": "validation_and_review",
                "failed_card_ids": [final_cards[index].id for index in last_problems],
                "errors": last_problems,
                "failed_cards": [{
                    "id": final_cards[index].id,
                    "source": final_cards[index].source_text,
                    "chinese": final_cards[index].translation,
                    "start": final_cards[index].start,
                    "end": final_cards[index].end,
                } for index in last_problems],
                "review_provenance": review_provenance,
            })
            if not last_problems:
                break
            if repair_round == 5:
                break
            failed = sorted(last_problems)
            windows: list[tuple[int, int]] = []
            for index in failed:
                start = max(0, index - 1)
                end = min(len(spans), index + 2)
                moved_entities = {
                    match.group(1)
                    for error in last_problems[index]
                    for match in [re.search(r"moved:([A-Za-z0-9._+-]+)", error)]
                    if match
                }
                for entity in moved_entities:
                    owners = [
                        owner for owner, card in enumerate(final_cards)
                        if _contains_term(card.source_text, entity)
                    ]
                    if owners:
                        owner = min(owners, key=lambda value: abs(value - index))
                        start = max(0, min(start, owner - 1))
                        end = min(len(spans), max(end, owner + 2))
                if windows and start <= windows[-1][1]:
                    windows[-1] = (windows[-1][0], max(windows[-1][1], end))
                else:
                    windows.append((start, end))
            repaired: list[tuple[int, int, str]] = []
            cursor = 0
            repair_traces: list[dict[str, Any]] = []
            for start, end in windows:
                repaired.extend(spans[cursor:start])
                word_start = spans[start][0]
                word_end = spans[end - 1][1]
                previous = final_cards[start - 1].source_text if start else ""
                following = final_cards[end].source_text if end < len(final_cards) else ""
                rejection = json.dumps({
                    final_cards[index].id: {
                        "source": final_cards[index].source_text,
                        "rejected_chinese": final_cards[index].translation,
                        "errors": last_problems[index],
                    }
                    for index in range(start, end) if index in last_problems
                }, ensure_ascii=False)
                requires_extra_card = any(
                    "hard maximum" in error
                    or "English words" in error
                    or "does not fit the three-line subtitle panel" in error
                    for index in range(start, end)
                    for error in last_problems.get(index, [])
                )
                mandatory_split_ranges = tuple(
                    (spans[index][0], spans[index][1])
                    for index in range(start, end)
                    if index in last_problems
                    and spans[index][1] - spans[index][0] > 1
                    and any(
                        "hard maximum" in error
                        or "English words" in error
                        or "does not fit the three-line subtitle panel" in error
                        for error in last_problems[index]
                    )
                )
                freeze_semantic_boundaries = (
                    repair_round >= 3 and not requires_extra_card
                )
                replacement, provenance = request_window(
                    word_start, word_end, previous, following, rejection,
                    repair_round + 1,
                    minimum_card_count=(end - start + 1 if requires_extra_card else 0),
                    mandatory_split_ranges=mandatory_split_ranges,
                    required_boundaries=(
                        tuple(spans[index][1] for index in range(start, end - 1))
                        if freeze_semantic_boundaries else ()
                    ),
                    freeze_boundaries=freeze_semantic_boundaries,
                )
                repaired.extend(replacement)
                repair_traces.append({
                    "old_card_window": [start + 1, end],
                    "source_word_window": [word_start, word_end],
                    "new_card_count": len(replacement),
                    "replacement_cards": [{
                        "source_word_window": [left, right],
                        "source": " ".join(
                            word.raw for word in words[left:right]
                        ),
                        "chinese": chinese,
                    } for left, right, chinese in replacement],
                    "provenance": provenance,
                })
                cursor = end
            repaired.extend(spans[cursor:])
            spans = repaired
            attempts.append({
                "round": repair_round + 1,
                "kind": (
                    "fixed_boundary_semantic_repair"
                    if repair_round >= 3 else "local_joint_repair"
                ),
                "windows": repair_traces,
            })
        if last_problems:
            raise InterviewJointTranslationError(
                "interview joint translation exhausted three movable-boundary repairs "
                "and two fixed-boundary semantic repairs: "
                + json.dumps(last_problems, ensure_ascii=False), attempts,
            )
        if " ".join(card.source_text for card in final_cards) != actual_source:
            raise ValueError("joint subtitle cards lost or reordered source words")
        cues[:] = final_cards
        adopted_targets = _adopt_independently_reviewed_terminology_targets(
            cues, terminology,
        )
        if term_errors := terminology_contract_errors(cues, terminology):
            raise ValueError(
                "joint subtitle cards violate terminology contract: "
                + "; ".join(term_errors)
            )
        return {
            "step": "interview_joint_boundary_translation",
            "policy_version": INTERVIEW_CAPTION_POLICY_VERSION,
            "policy_fingerprint": INTERVIEW_CAPTION_POLICY_FINGERPRINT,
            "reviewed_cue_ids": [cue.id for cue in cues],
            "source_word_ids": [word.id for word in words],
            "source_ledger_fingerprint": source_ledger_fingerprint(words),
            "strict_source_fingerprint": _strict_interview_source_fingerprint(cues),
            "caption_content_fingerprint": _interview_caption_content_fingerprint(cues),
            "terminology_decision_fingerprint": _terminology_decision_fingerprint(
                terminology,
            ),
            "alignment_fingerprint": alignment_fingerprint,
            "audio_hypothesis_sha256": (
                hashlib.sha256(audio_hypothesis.encode("utf-8")).hexdigest()
                if audio_hypothesis else ""
            ),
            "audio_conflict_entities": sorted(audio_conflict_entities),
            "audio_conflict_numbers": sorted(audio_conflict_numbers),
            "audio_only_numbers": sorted(audio_only_numbers),
            "numeric_conflict_pairs": numeric_conflict_pairs,
            "translation_passes": 1,
            "repair_rounds": sum(
                item["kind"] == "local_joint_repair" for item in attempts
            ),
            "attempts": attempts,
            "adopted_terminology_targets": adopted_targets,
        }

    def translate_caption_scopes(
        self, cues: list[TranscriptCue], terminology: list[TerminologyEntry],
        plan: dict[str, Any], duration: float, editorial_mode: str,
    ) -> dict[str, Any]:
        """Run the joint subtitle engine inside immutable editorial cut ranges."""
        publishable_source_cues: list[TranscriptCue] = []
        for cue in cues:
            source_text = omit_non_speech_directions(cue.source_text)
            if not source_text:
                continue
            publishable_source_cues.append(TranscriptCue(
                **{
                    **asdict(cue),
                    "source_text": source_text,
                    "translation": "",
                    "source_tokens": [
                        row for row in cue.source_tokens
                        if omit_non_speech_directions(
                            str(row.get("raw") or "")
                        ).strip()
                    ],
                }
            ))
        source_words = source_words_from_cues(publishable_source_cues)
        ranges = _caption_plan_ranges(plan, duration)
        scope_traces: list[dict[str, Any]] = []
        translated: list[TranscriptCue] = []
        assigned_word_ids: set[str] = set()
        for scope_number, source_range in enumerate(ranges, start=1):
            scope_words = [
                word for word in source_words
                if source_range.start <= (word.start + word.end) / 2 < source_range.end
                and word.id not in assigned_word_ids
            ]
            if not scope_words:
                continue
            assigned_word_ids.update(word.id for word in scope_words)
            scope_cues = [TranscriptCue(
                id=f"caption-scope-{scope_number:03d}",
                start=scope_words[0].start,
                end=scope_words[-1].end,
                source_text=" ".join(word.raw for word in scope_words),
                source_tokens=[{
                    "raw": word.raw, "start": word.start, "end": word.end,
                } for word in scope_words],
            )]
            scope_trace = self.translate_interview_clip_once(
                scope_cues, terminology, scope_words,
                "json3:" + source_ledger_fingerprint(scope_words), "",
            )
            for card_number, card in enumerate(scope_cues, start=1):
                card.id = (
                    f"caption-scope-{scope_number:03d}-card-{card_number:04d}"
                )
            translated.extend(scope_cues)
            scope_traces.append({
                "scope": scope_number,
                "source_range": [source_range.start, source_range.end],
                "reviewed_cue_ids": [card.id for card in scope_cues],
                "trace": scope_trace,
            })
        selected_word_ids = {
            word.id for word in source_words
            if any(
                source_range.start <= (word.start + word.end) / 2 < source_range.end
                for source_range in ranges
            )
        }
        if assigned_word_ids != selected_word_ids:
            missing = sorted(selected_word_ids - assigned_word_ids)
            duplicated_or_extra = sorted(assigned_word_ids - selected_word_ids)
            raise ValueError(
                "joint caption scopes did not cover selected source words exactly once: "
                f"missing={missing[:10]}, extra={duplicated_or_extra[:10]}"
            )
        if not translated:
            raise ValueError("selected editorial ranges contain no subtitle source words")
        translated.sort(key=lambda cue: (cue.start, cue.end, cue.id))
        if term_errors := terminology_contract_errors(translated, terminology):
            raise ValueError(
                "joint caption scopes violate terminology contract: "
                + "; ".join(term_errors)
            )
        cross_scope_trace: dict[str, Any] | None = None
        if len(scope_traces) > 1:
            boundary_ids = {
                cue_id
                for scope in scope_traces
                for cue_id in (
                    scope["reviewed_cue_ids"][0],
                    scope["reviewed_cue_ids"][-1],
                )
            }
            consistency_rows = [
                {
                    "id": cue.id, "source": cue.source_text,
                    "chinese": cue.translation,
                }
                for cue in translated
                if cue.id in boundary_ids or any(
                    _contains_term(cue.source_text, entry.source)
                    or entry.target in cue.translation
                    or any(value in cue.translation for value in entry.alternatives)
                    for entry in terminology
                )
            ]
            expected_ids = {row["id"] for row in consistency_rows}
            structure_failures: list[list[str]] = []
            provenances: list[dict[str, Any]] = []
            for attempt in range(2):
                verdict, provenance = (
                    self.subtitle_reviewer or self.directing_writer
                )._request_json([
                    {"role": "system", "content": "Return one valid JSON object only."},
                    {"role": "user", "content": "\n".join([
                        "Independently check only cross-scope consistency in these already reviewed bilingual subtitle boundary and terminology rows. Do not rewrite them.",
                        "Reject a contextual term that changes meaning without source support, an entity or number moved across a scope, or meaning duplicated or lost at a scope boundary. Return {pass,issues:[{ids,errors}]}. pass=true requires an empty issues array.",
                        "Terminology: " + json.dumps(
                            _relevant_terminology_prompt_rows(
                                terminology,
                                " ".join(cue.source_text for cue in translated),
                            ), ensure_ascii=False,
                        ),
                        (
                            "The previous response failed deterministic structure validation: "
                            + json.dumps(structure_failures[-1], ensure_ascii=False)
                        ) if structure_failures else "",
                        "Sequence: " + json.dumps(consistency_rows, ensure_ascii=False),
                    ])},
                ], max_tokens=3500)
                provenances.append(provenance)
                passed = verdict.get("pass")
                issues = verdict.get("issues")
                errors: list[str] = []
                if type(passed) is not bool:
                    errors.append("pass must be boolean")
                if not isinstance(issues, list):
                    errors.append("issues must be an array")
                    issue_rows: list[dict[str, Any]] = []
                else:
                    issue_rows = [row for row in issues if isinstance(row, dict)]
                    if len(issue_rows) != len(issues):
                        errors.append("every issue must be an object")
                for issue in issue_rows:
                    ids = issue.get("ids")
                    issue_errors = issue.get("errors")
                    if (
                        not isinstance(ids, list) or not ids
                        or not set(map(str, ids)) <= expected_ids
                    ):
                        errors.append("issue ids must be non-empty sequence ids")
                    if not isinstance(issue_errors, list) or not issue_errors:
                        errors.append("issue errors must be a non-empty array")
                if passed is True and issue_rows:
                    errors.append("pass=true requires no issues")
                if passed is False and not issue_rows:
                    errors.append("pass=false requires at least one issue")
                if errors:
                    structure_failures.append(errors)
                    continue
                cross_scope_trace = {
                    "pass": passed, "issues": issue_rows,
                    "attempts": provenances,
                    "structure_failures": structure_failures,
                    "reviewed_cue_ids": sorted(expected_ids),
                }
                if passed is False:
                    raise InterviewJointTranslationError(
                        "cross-scope consistency review rejected joint captions",
                        [{
                            "kind": "cross_scope_consistency_review",
                            "issues": issue_rows,
                            "provenance": provenance,
                        }],
                    )
                break
            else:
                raise ValueError(
                    "cross-scope consistency reviewer returned invalid structure twice: "
                    + json.dumps(structure_failures, ensure_ascii=False)
                )
        cues[:] = translated
        return {
            "step": "joint_caption_scopes_translation",
            "editorial_mode": editorial_mode,
            "policy_version": INTERVIEW_CAPTION_POLICY_VERSION,
            "policy_fingerprint": INTERVIEW_CAPTION_POLICY_FINGERPRINT,
            "scope_count": len(scope_traces),
            "scope_traces": scope_traces,
            "cross_scope_consistency_review": cross_scope_trace,
            "reviewed_cue_ids": [cue.id for cue in cues],
            "strict_source_fingerprint": _strict_interview_source_fingerprint(cues),
            "caption_content_fingerprint": _interview_caption_content_fingerprint(cues),
            "terminology_decision_fingerprint": _terminology_decision_fingerprint(
                terminology,
            ),
        }



    def discover_missing_terminology(
        self, cues: list[TranscriptCue], terminology: list[TerminologyEntry],
    ) -> dict[str, Any]:
        """Propose missing terms from the fixed subtitle source without editing the plan."""
        existing = {entry.source.casefold() for entry in terminology}
        transcript = [{"id": cue.id, "text": cue.source_text} for cue in cues]
        failures: list[str] = []
        discovery_attempts: list[dict[str, Any]] = []
        for attempt in range(2):
            response, provenance = self.writer._request_json([
                {"role": "system", "content": "Return one valid JSON object only."},
                {"role": "user", "content": "\n".join([
                    "Inspect only this already-selected English subtitle passage for missing terminology needed by a Simplified Chinese translation. Do not select a clip, write a title, propose a hook, summarize, or translate captions.",
                    "Return at most 8 source phrases that occur verbatim in the passage, are absent from Existing terminology, and need one video-level decision because they are a technical compound, an emerging term, or context-sensitive vocabulary whose inconsistent translation could change meaning. Do not return ordinary words merely to fill the quota.",
                    "Do not propose apparent ASR corruption, misspellings, broken fragments, or invented normalized phrases. If a phrase is not a stable term exactly present in the selected passage, omit it; the independent subtitle reviewer will handle its sentence meaning later.",
                    "For each row choose translate or preserve. translate requires a concise natural Chinese target; preserve is only for a product/company name, acronym, code/API identifier, or a genuinely unsettled term without a clear Chinese rendering. Include at most two defensible Chinese alternatives and a concise rationale citing the actor/action/context.",
                    "Return {terminology:[{source,strategy,target,alternatives,rationale}]}. Return an empty array when nothing is missing.",
                    "Existing terminology: " + json.dumps(
                        sorted(entry.source for entry in terminology), ensure_ascii=False,
                    ),
                    (
                        "The previous response failed deterministic validation: "
                        + failures[-1] + ". Return the complete corrected object."
                    ) if failures else "",
                    "Selected subtitle passage: " + json.dumps(transcript, ensure_ascii=False),
                ])},
            ], max_tokens=3000)
            raw = response.get("terminology")
            if not isinstance(raw, list) or len(raw) > 8:
                failures.append("terminology must be an array with at most 8 rows")
                discovery_attempts.append({
                    "attempt": attempt + 1, "validation_error": failures[-1],
                    "provenance": provenance,
                })
                continue
            raw_sources = [
                str(row.get("source") or "").strip()
                for row in raw if isinstance(row, dict)
            ]
            if len(raw_sources) != len(raw) or len({value.casefold() for value in raw_sources}) != len(raw):
                failures.append("every row needs one unique source")
                discovery_attempts.append({
                    "attempt": attempt + 1, "validation_error": failures[-1],
                    "provenance": provenance,
                })
                continue
            invalid_sources = [
                source for source in raw_sources
                if not source or source.casefold() in existing
                or not any(_contains_term(cue.source_text, source) for cue in cues)
            ]
            if invalid_sources:
                failures.append(
                    "sources must occur in the passage and must not duplicate existing terms: "
                    + json.dumps(invalid_sources, ensure_ascii=False)
                )
                discovery_attempts.append({
                    "attempt": attempt + 1, "validation_error": failures[-1],
                    "rejected_sources": invalid_sources, "provenance": provenance,
                })
                continue
            parsed = self._parse_terminology(raw, cues)
            returned = {source.casefold() for source in raw_sources}
            additions = [
                entry for entry in parsed
                if entry.source.casefold() in returned
                and entry.source.casefold() not in existing
                and entry.rationale
                and (
                    entry.strategy == TerminologyStrategy.PRESERVE
                    or bool(re.search(r"[\u3400-\u9fff]", entry.target))
                )
            ]
            if len(additions) != len(raw):
                failures.append(
                    "every row needs a valid strategy, target, and contextual rationale"
                )
                discovery_attempts.append({
                    "attempt": attempt + 1, "validation_error": failures[-1],
                    "provenance": provenance,
                })
                continue
            terminology.extend(additions)
            discovery_attempts.append({
                "attempt": attempt + 1, "added_sources": [
                    entry.source for entry in additions
                ], "provenance": provenance,
            })
            return {
                "step": "selected_subtitle_terminology_discovery",
                "attempt": attempt + 1,
                "existing_sources": sorted(existing),
                "added_sources": [entry.source for entry in additions],
                "decisions": [asdict(entry) for entry in additions],
                "earlier_failures": failures,
                "attempts": discovery_attempts,
                "provenance": provenance,
            }
        # This discovery pass is optional enrichment. Invalid model proposals
        # must never become terminology, but they also must not block the fixed
        # clip: card-level semantic review still judges the actual translation.
        return {
            "step": "selected_subtitle_terminology_discovery",
            "attempt": len(discovery_attempts),
            "existing_sources": sorted(existing),
            "added_sources": [],
            "decisions": [],
            "earlier_failures": failures,
            "discarded_invalid_proposals": True,
            "attempts": discovery_attempts,
            "provenance": discovery_attempts[-1].get("provenance")
            if discovery_attempts else None,
        }

    def review_terminology_decisions(
        self, cues: list[TranscriptCue], terminology: list[TerminologyEntry],
    ) -> dict[str, Any] | None:
        """Review planner terminology choices before interview translation."""
        contextual = [
            entry for entry in terminology
            if entry.rationale
            and entry.source.casefold() not in ESTABLISHED_CHINESE_TERMS
            and any(_contains_term(cue.source_text, entry.source) for cue in cues)
        ]
        if not contextual:
            return None
        reviewer = self.subtitle_reviewer or self.directing_writer
        if reviewer is None or reviewer is self.writer:
            raise RuntimeError(
                "terminology decisions require an independent subtitle reviewer"
            )

        review_attempts: list[dict[str, Any]] = []
        review_structure_failures: list[dict[str, Any]] = []
        revision_attempts: list[dict[str, Any]] = []
        last_rejected: dict[str, list[str]] = {}
        dropped_sources: list[str] = []
        for attempt in range(2):
            rows: list[dict[str, Any]] = []
            for entry in contextual:
                matching = [
                    index for index, cue in enumerate(cues)
                    if _contains_term(cue.source_text, entry.source)
                ]
                context_indices = sorted({
                    neighbor
                    for index in matching
                    for neighbor in (index - 1, index, index + 1)
                    if 0 <= neighbor < len(cues)
                })
                rows.append({
                    **_terminology_prompt_row(
                        entry, " ".join(cues[index].source_text for index in matching),
                    ),
                    "context": [cues[index].source_text for index in context_indices],
                })
            expected = {entry.source.casefold(): entry for entry in contextual}
            reviews: dict[str, dict[str, Any]] = {}
            phase_structure_failures: list[list[str]] = []
            for structure_attempt in range(2):
                verdict, provenance = reviewer._request_json([
                    {"role": "system", "content": "Return one valid JSON object only."},
                    {"role": "user", "content": "\n".join([
                        "Independently review each proposed terminology decision for a Simplified Chinese technology interview before subtitles are translated. Do not rewrite captions.",
                        "Treat both translate and preserve as proposals, not instructions. Preserve English only for a product or company name, acronym, code/API identifier, or a genuinely unsettled term that lacks a clear natural Chinese rendering in this context. A term being recent or emerging is not by itself evidence that Chinese readers should see the English form. Reject preserve when a concise, established or compositionally clear Chinese rendering accurately conveys the source distinction.",
                        "For translate, reject a target that changes the actor, action, technical distinction, or contextual sense. Judge the supplied rationale against the source context rather than trusting it. Alternatives are candidates, not automatic approvals.",
                        "If the exact source is an apparent ASR corruption, misspelling, or broken fragment rather than a stable term, reject it and make the first error exactly drop:not_a_stable_term. Use that marker only when the source itself should not be a video-level terminology decision; sentence-level meaning remains for the subtitle reviewer.",
                        "Return every source exactly once as {reviews:[{source,pass,fidelity_score,naturalness_score,errors}]}. Pass only when both scores are at least 4 and the exact strategy and target are publication-ready. Do not return replacement wording.",
                        (
                            "The previous reviewer response failed deterministic structure validation: "
                            + json.dumps(phase_structure_failures[-1], ensure_ascii=False)
                            + ". Review the unchanged term decisions again and return the exact schema."
                        ) if phase_structure_failures else "",
                        "Terms: " + json.dumps(rows, ensure_ascii=False),
                    ])},
                ], max_tokens=4000)
                review_attempts.append(provenance)
                raw_reviews = verdict.get("reviews")
                structure_errors: list[str] = []
                if not isinstance(raw_reviews, list):
                    structure_errors.append("reviews must be an array")
                    candidate_reviews: list[dict[str, Any]] = []
                else:
                    candidate_reviews = [
                        row for row in raw_reviews if isinstance(row, dict)
                    ]
                    if len(candidate_reviews) != len(raw_reviews):
                        structure_errors.append("every review must be an object")
                sources = [
                    str(row.get("source") or "").casefold()
                    for row in candidate_reviews
                ]
                if (
                    len(sources) != len(expected)
                    or len(set(sources)) != len(sources)
                    or set(sources) != set(expected)
                ):
                    structure_errors.append(
                        "review sources must contain every supplied source exactly once"
                    )
                for row in candidate_reviews:
                    source = str(row.get("source") or "")
                    if type(row.get("pass")) is not bool:
                        structure_errors.append(f"{source}: pass must be boolean")
                    for score_name in ("fidelity_score", "naturalness_score"):
                        if not 1 <= _review_score_out_of_five(row.get(score_name)) <= 5:
                            structure_errors.append(
                                f"{source}: {score_name} must be between 1 and 5"
                            )
                    if not isinstance(row.get("errors"), list):
                        structure_errors.append(f"{source}: errors must be an array")
                if not structure_errors:
                    reviews = {
                        str(row["source"]).casefold(): row
                        for row in candidate_reviews
                    }
                    break
                phase_structure_failures.append(structure_errors)
                review_structure_failures.append({
                    "decision_attempt": attempt + 1,
                    "structure_attempt": structure_attempt + 1,
                    "errors": structure_errors,
                })
            else:
                raise ValueError(
                    "terminology reviewer returned invalid structure twice: "
                    + json.dumps(phase_structure_failures, ensure_ascii=False)
                )
            rejected: dict[str, list[str]] = {}
            dropped_keys: set[str] = set()
            for key, entry in expected.items():
                row = reviews.get(key, {})
                if (
                    set(reviews) != set(expected)
                    or row.get("pass") is not True
                    or _review_score_out_of_five(row.get("fidelity_score")) < 4
                    or _review_score_out_of_five(row.get("naturalness_score")) < 4
                ):
                    raw_errors = row.get("errors", ["missing independent review"])
                    rejected[entry.source] = (
                        [str(value) for value in raw_errors]
                        if isinstance(raw_errors, list) else [str(raw_errors)]
                    )
                    if rejected[entry.source] and rejected[entry.source][0] == "drop:not_a_stable_term":
                        dropped_keys.add(key)
            if dropped_keys:
                dropped_sources.extend(
                    expected[key].source for key in sorted(dropped_keys)
                )
                terminology[:] = [
                    entry for entry in terminology
                    if entry.source.casefold() not in dropped_keys
                ]
                contextual = [
                    entry for entry in contextual
                    if entry.source.casefold() not in dropped_keys
                ]
                rejected = {
                    source: errors for source, errors in rejected.items()
                    if source.casefold() not in dropped_keys
                }
            if not rejected:
                return {
                    "step": "terminology_decision_review",
                    "attempt": attempt + 1,
                    "reviewed_sources": sorted(expected),
                    "review_attempts": review_attempts,
                    "review_structure_failures": review_structure_failures,
                    "revision_attempts": revision_attempts,
                    "dropped_sources": dropped_sources,
                    "decisions": [asdict(entry) for entry in contextual],
                }
            last_rejected = rejected
            if attempt == 1:
                break

            rejected_entries = [
                entry for entry in contextual if entry.source in rejected
            ]
            revision, revision_provenance = self.writer._request_json([
                {"role": "system", "content": "Return one valid JSON object only."},
                {"role": "user", "content": "\n".join([
                    "Revise only the rejected terminology decisions for Simplified Chinese technology subtitles. Return a decision, not a caption translation.",
                    "Use translate with a concise natural Chinese target whenever the contextual meaning has an established or compositionally clear Chinese rendering. Preserve only product/company names, acronyms, code/API identifiers, or genuinely unsettled terms without a clear Chinese rendering. Recent or emerging terminology does not automatically stay in English.",
                    "Return every rejected source exactly once as {terminology:[{source,strategy,target,alternatives,rationale}]}. strategy is translate or preserve. A translate target must contain Chinese; preserve must keep the source English and use an empty alternatives list. Give context evidence in rationale and at most two defensible Chinese alternatives.",
                    "Rejected decisions: " + json.dumps([
                        {
                            **asdict(entry),
                            "review_errors": rejected[entry.source],
                            "context": [
                                cue.source_text for cue in cues
                                if _contains_term(cue.source_text, entry.source)
                            ],
                        } for entry in rejected_entries
                    ], ensure_ascii=False),
                ])},
            ], max_tokens=3000)
            revision_attempts.append(revision_provenance)
            expected_revised = {entry.source.casefold() for entry in rejected_entries}
            raw_revised = revision.get("terminology")
            returned_sources = {
                str(row.get("source") or "").casefold()
                for row in raw_revised if isinstance(row, dict)
            } if isinstance(raw_revised, list) else set()
            if returned_sources != expected_revised:
                raise ValueError(
                    "terminology decision revision omitted or added sources"
                )
            revised = self._parse_terminology(raw_revised, cues)
            revised_by_source = {
                entry.source.casefold(): entry for entry in revised
                if entry.source.casefold() in expected_revised
            }
            if set(revised_by_source) != expected_revised:
                raise ValueError(
                    "terminology decision revision omitted or added sources"
                )
            for entry in rejected_entries:
                replacement = revised_by_source[entry.source.casefold()]
                if (
                    replacement.strategy == TerminologyStrategy.TRANSLATE
                    and not re.search(r"[\u3400-\u9fff]", replacement.target)
                ):
                    raise ValueError(
                        f"revised translate terminology lacks Chinese target: {entry.source}"
                    )
                entry.strategy = replacement.strategy
                entry.target = replacement.target
                entry.alternatives = replacement.alternatives
                entry.rationale = replacement.rationale
                entry.first_use_explanation = ""
                entry.notes = replacement.notes
        rejected_keys = {source.casefold() for source in last_rejected}
        discarded_sources = [
            entry.source for entry in contextual
            if entry.source.casefold() in rejected_keys
        ]
        terminology[:] = [
            entry for entry in terminology
            if entry.source.casefold() not in rejected_keys
        ]
        return {
            "step": "terminology_decision_review",
            "attempt": 2,
            "reviewed_sources": sorted(
                entry.source.casefold() for entry in contextual
            ),
            "review_attempts": review_attempts,
            "review_structure_failures": review_structure_failures,
            "revision_attempts": revision_attempts,
            "dropped_sources": dropped_sources,
            "discarded_after_rejection": discarded_sources,
            "final_rejections": last_rejected,
            "decisions": [
                asdict(entry) for entry in contextual
                if entry.source.casefold() not in rejected_keys
            ],
        }


    def ensure_editorial_plan(
        self, metadata: dict[str, Any], cues: list[TranscriptCue], plan: dict[str, Any],
        editorial_mode: str = "study", planning_transcript: list[dict[str, Any]] | None = None,
        editorial_guidance: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        duration = float(metadata.get("duration") or (cues[-1].end if cues else 0))
        plan["editorial_mode"] = editorial_mode
        fallback_speaker = _metadata_speaker_label(metadata)
        traces: list[dict[str, Any]] = []
        if editorial_mode == "known_tech_interview_clip":
            exact_range = _requested_exact_range_from_editorial_guidance(
                editorial_guidance,
            )
            rows = plan.get("wechat_lessons")
            if (
                exact_range is not None and isinstance(rows, list) and rows
                and isinstance(rows[0], dict)
            ):
                rows[0]["start"], rows[0]["end"] = exact_range
                plan["story_start"], plan["story_end"] = exact_range
                traces.append({
                    "step": "requested_exact_range",
                    "start": exact_range[0], "end": exact_range[1],
                })
            requested_title_trace = _apply_supported_requested_title(
                plan, cues, duration, editorial_guidance,
            )
            if requested_title_trace and requested_title_trace.get("applied"):
                # Apply an explicitly requested, source-entailed title before
                # validating the model draft. Otherwise a model-invented title
                # can fail the contract and prevent the supported replacement
                # from being applied later in the pipeline.
                traces.append(requested_title_trace)
        errors = editorial_plan_contract_errors(plan, duration, cues)
        if not errors:
            return plan, traces
        if editorial_mode == "known_tech_interview_clip" and fallback_speaker:
            rows = plan.get("wechat_lessons")
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                _apply_metadata_speaker_fallback(rows[0], fallback_speaker)
        normalized = normalize_editorial_plan_structure(plan, cues, duration, editorial_mode)
        normalized_errors = editorial_plan_contract_errors(normalized, duration, cues)
        traces.append({
            "step": "deterministic_editorial_structure_pre_repair",
            "errors_before": errors, "errors_after": normalized_errors,
        })
        if not normalized_errors:
            return normalized, traces
        if len(normalized_errors) < len(errors):
            plan, errors = normalized, normalized_errors
        transcript = planning_transcript or [
            {"id": item.id, "start": item.start, "end": item.end, "text": item.source_text}
            for item in cues
        ]
        original_terminology = plan.get("terminology", [])
        for attempt in range(2):
            mandatory_layout = ""
            if editorial_mode == "technical_coverage" or (
                editorial_mode == "study" and duration <= 2700
            ):
                lesson_ranges = _required_short_source_ranges(
                    cues, 0.0, duration, 24 if editorial_mode == "technical_coverage" else 8,
                )
                mandatory_layout = (
                    "This is a mandatory deterministic boundary contract: story_start=0, "
                    f"story_end={duration:.3f}, "
                    + (
                        f"one Bilibili chapter from 0 to {duration:.3f}, "
                        if editorial_mode == "study" else "no Bilibili chapters, "
                    )
                    + "and WeChat lessons with exactly these ordered start/end pairs: "
                    + json.dumps(lesson_ranges, ensure_ascii=False)
                    + ". Copy every number exactly; write only the titles, theses, framing, and hooks."
                )
            if editorial_mode == "known_tech_interview_clip":
                repair_contract = (
                    "Return exactly one continuous WeChat clip and no Bilibili chapters. There is no minimum duration: prefer the shortest strong, complete passage at or below 180 seconds. A strong complete passage may extend to 300 seconds; essential_context_justification is optional audit metadata. "
                    "Choose the strongest self-contained technology, AI, engineering, learning, or education insight. "
                    "The selected source words and all visible copy must contain no politics, government, elections, "
                    "geopolitics, war, military, politicians, or advocacy. Non-political comparisons between countries' "
                    "technology, research, engineering, talent, schools, universities, or education are allowed. "
                    "Make the three hooks exceptionally attractive through a concrete surprise, tension, or payoff, "
                    "while remaining fully supported by the selected words. Include a short evidence-backed speaker_label."
                    " If the current title is strong and the complete transcript contains a continuous passage that supports it, preserve the title and move start/end to that passage. "
                    "Do not weaken a good title merely to salvage a mismatched clip. Rewrite the title only when no continuous source passage can deliver its promise, and never end before the answer."
                    + (
                        " Treat this advisory as retrieval guidance, then verify it against the transcript: "
                        + editorial_guidance.strip()
                        if editorial_guidance and editorial_guidance.strip() else ""
                    )
                )
            elif editorial_mode == "conference_highlights":
                repair_contract = (
                    "Return 1–5 distinct, non-overlapping, standalone WeChat highlights and no Bilibili chapters. "
                    "Each clip must be 45–300 seconds and all clips together no more than 900 seconds. "
                    "Every clip needs its own complete setup, technical idea, and payoff; clips must not form a dependent series. "
                    "If the transcript contains no qualifying clip, return an empty wechat_lessons list."
                )
            elif editorial_mode == "technical_coverage":
                repair_contract = (
                    "Bilibili is paused: return no Bilibili chapters. Return chronological 3–6 minute WeChat technical "
                    "lessons covering the complete non-political technical story; use as many as required, up to 24."
                )
            else:
                repair_contract = (
                    "Bilibili chapters are chronological 600–1800 second study chapters covering the substantive story. "
                    "WeChat lessons are 180–360 seconds."
                )
            prompt = "\n".join([
                "Repair this editorial plan for a Chinese AI engineering video collection. Do not translate subtitles.",
                "Return the complete plan with editorial_mode, collection_title, story_start, story_end, terminology, bilibili_chapters, and wechat_lessons.",
                repair_contract,
                "Every Bilibili chapter and WeChat lesson needs exactly three distinct hook_headlines of 8–26 visible characters: complete, evidence-faithful clauses representing a tension or contrarian claim, a concrete technical choice with stakes, and a consequence or outcome claim. Each must earn the next 8 seconds rather than merely label the topic. Never bisect an English term, append a generic 为什么, ask an empty question, or use sensational clickbait. Use framing=split when speaker and slide are visible together; never crop away the speaker pane merely to enlarge the slide.",
                mandatory_layout,
                "Keep the existing terminology unless it must be normalized. Terminology fields are source, strategy (translate|preserve), target, alternatives, rationale, notes. For translate, target is the rendering selected from this video's nearby actors/actions; alternatives has at most two genuinely context-supported candidates and rationale states the evidence. Preserve rows use an empty alternatives list. Never emit bilingual_once.",
                "Contract errors: " + json.dumps(errors, ensure_ascii=False),
                "Current plan: " + json.dumps(plan, ensure_ascii=False),
                "Transcript: " + json.dumps(transcript, ensure_ascii=False),
            ])
            repaired, provenance = self.writer._request_json([
                {"role": "system", "content": "Return one valid JSON object only."},
                {"role": "user", "content": prompt},
            ], max_tokens=(12000 if editorial_mode == "known_tech_interview_clip" else 7000))
            if not repaired.get("terminology") and original_terminology:
                repaired["terminology"] = original_terminology
            repaired["editorial_mode"] = editorial_mode
            if editorial_mode == "known_tech_interview_clip" and fallback_speaker:
                repaired_rows = repaired.get("wechat_lessons")
                if isinstance(repaired_rows, list) and repaired_rows and isinstance(repaired_rows[0], dict):
                    _apply_metadata_speaker_fallback(repaired_rows[0], fallback_speaker)
            for key in ("collection_title",):
                if not str(repaired.get(key) or "").strip():
                    repaired[key] = plan.get(key, "")
            plan = repaired
            errors = editorial_plan_contract_errors(plan, duration, cues)
            traces.append({
                "step": "editorial_plan_repair", "attempt": attempt + 1,
                "errors_after": errors, "provenance": provenance,
            })
            if not errors:
                return plan, traces
        if editorial_mode == "known_tech_interview_clip" and fallback_speaker:
            plan_rows = plan.get("wechat_lessons")
            if isinstance(plan_rows, list) and plan_rows and isinstance(plan_rows[0], dict):
                _apply_metadata_speaker_fallback(plan_rows[0], fallback_speaker)
        plan = normalize_editorial_plan_structure(plan, cues, duration, editorial_mode)
        errors = editorial_plan_contract_errors(plan, duration, cues)
        traces.append({
            "step": "deterministic_editorial_structure_repair",
            "errors_after": errors,
        })
        if not errors:
            return plan, traces
        raise ValueError("editorial plan contract failed after repair: " + "; ".join(errors))

    @staticmethod
    def _enforce_terminology_contract(
        cues: list[TranscriptCue], terminology: list[TerminologyEntry],
    ) -> list[str]:
        """Apply a minimal, auditable fallback when a model drops an exact term."""
        enforced: list[str] = []
        combined_target = "\n".join(item.translation for item in cues)
        replacements = {
            "挽具": "Harness", "铺好的道路": "成熟路径",
            "人类触摸": "人工参与", "技能登记处": "Skill 注册中心",
        }
        for cue in cues:
            for literal, natural in replacements.items():
                if literal in cue.translation:
                    cue.translation = cue.translation.replace(literal, natural)
                    enforced.append(literal)
        for entry in terminology:
            first = next(
                (cue for cue in cues if _contains_term(cue.source_text, entry.source)), None,
            )
            if first is None:
                continue
            if entry.strategy == TerminologyStrategy.PRESERVE:
                if not _contains_term(combined_target, entry.source):
                    if entry.source == "Skill" and "技能" in first.translation:
                        first.translation = first.translation.replace("技能", "Skill", 1)
                    else:
                        first.translation = f"{first.translation.rstrip('。')}（{entry.source}）。"
                    combined_target += "\n" + entry.source
                    enforced.append(entry.source)
            elif entry.strategy == TerminologyStrategy.TRANSLATE and entry.target:
                protected_terms = [
                    item.source for item in terminology
                    if item is not entry
                    and len(item.source) > len(entry.source)
                    and entry.source.casefold() in item.source.casefold()
                ]
                for cue in cues:
                    if not _contains_unprotected_term(
                        cue.source_text, entry.source, protected_terms,
                    ):
                        continue
                    protected: dict[str, str] = {}
                    protected_translation = cue.translation
                    for preserve_index, preserve_entry in enumerate(terminology):
                        if (
                            preserve_entry.strategy != TerminologyStrategy.PRESERVE
                            or entry.source.casefold() not in preserve_entry.source.casefold()
                            or not _contains_term(protected_translation, preserve_entry.source)
                        ):
                            continue
                        marker = f"__VF_PRESERVE_{preserve_index}__"
                        protected_translation = re.sub(
                            re.escape(preserve_entry.source), marker,
                            protected_translation, flags=re.IGNORECASE,
                        )
                        protected[marker] = preserve_entry.source
                    replaced = re.sub(
                        (
                            rf"(?<![A-Za-z0-9]){re.escape(entry.source)}(?:s|es)?(?![A-Za-z0-9])"
                            if re.fullmatch(
                                r"[A-Za-z][A-Za-z0-9 .+-]*[A-Za-z]", entry.source,
                            ) and not entry.source.casefold().endswith("s")
                            else re.escape(entry.source)
                        ),
                        entry.target, protected_translation,
                        flags=re.IGNORECASE,
                    )
                    for marker, preserved_term in protected.items():
                        replaced = replaced.replace(marker, preserved_term)
                    # Old cached plans sometimes used ``bilingual_once`` for
                    # ordinary vocabulary, producing text such as
                    # ``vertical：垂直行业``.  Once the policy is normalized
                    # to translate, collapse the now-duplicated Chinese label.
                    replaced = re.sub(
                        rf"{re.escape(entry.target)}\s*[：:]\s*{re.escape(entry.target)}",
                        entry.target,
                        replaced,
                    )
                    if entry.source.casefold() == "vertical":
                        replaced = re.sub(r"高薪\s*垂直行业", "高薪的垂直行业", replaced)
                    if replaced != cue.translation:
                        cue.translation = normalize_chinese_subtitle(replaced)
                        enforced.append(entry.source)
            elif entry.strategy == TerminologyStrategy.BILINGUAL_ONCE:
                explanation = entry.first_use_explanation
                if explanation:
                    for cue in cues:
                        if cue is first:
                            before, marker, after = cue.translation.partition(explanation)
                            if marker and explanation in after:
                                cue.translation = before + marker + after.replace(explanation, "")
                                enforced.append(entry.source)
                            continue
                        if explanation not in cue.translation:
                            continue
                        for separator in ("：", ": ", ":"):
                            cue.translation = cue.translation.replace(
                                f"{entry.source}{separator}{explanation}", entry.source,
                            )
                        cue.translation = cue.translation.replace(explanation, "")
                        cue.translation = re.sub(r"（\s*[：:，,；;]?\s*）", "", cue.translation)
                        cue.translation = re.sub(r"[：:]\s*([，。；;])", r"\1", cue.translation)
                        enforced.append(entry.source)
                needs_source = not _contains_term(first.translation, entry.source)
                needs_explanation = entry.first_use_explanation not in first.translation
                if needs_source and not needs_explanation:
                    first.translation = first.translation.replace(
                        entry.first_use_explanation,
                        f"{entry.source}：{entry.first_use_explanation}",
                        1,
                    )
                    enforced.append(entry.source)
                elif needs_source or needs_explanation:
                    label = entry.source
                    if entry.first_use_explanation:
                        label += f"：{entry.first_use_explanation}"
                    first.translation = f"{first.translation.rstrip('。')}（{label}）。"
                    enforced.append(entry.source)
        return list(dict.fromkeys(enforced))


    @staticmethod
    def _parse_terminology(raw: Any, cues: list[TranscriptCue]) -> list[TerminologyEntry]:
        entries: list[TerminologyEntry] = []
        source_text = "\n".join(item.source_text for item in cues)
        for item in raw if isinstance(raw, list) else []:
            raw_source = item.get("source") or item.get("term") if isinstance(item, dict) else ""
            source = raw_source.strip() if isinstance(raw_source, str) else ""
            if (
                not isinstance(item, dict)
                or not source
                or len(source) > 120
                or not _contains_term(source_text, source)
            ):
                continue
            try:
                strategy = TerminologyStrategy(str(
                    item.get("strategy") or item.get("choice")
                    or item.get("mode") or "preserve"
                ))
            except ValueError:
                strategy = TerminologyStrategy.PRESERVE
            raw_target = item.get("target") or item.get("translation")
            target = raw_target.strip() if isinstance(raw_target, str) else ""
            if len(target) > 48:
                target = ""
            raw_explanation = item.get("first_use_explanation")
            explanation = (
                raw_explanation.strip()[:120]
                if isinstance(raw_explanation, str) else ""
            )
            raw_alternatives = item.get("alternatives", item.get("accepted_targets", []))
            alternatives: list[str] = []
            if isinstance(raw_alternatives, list):
                for raw_alternative in raw_alternatives:
                    if not isinstance(raw_alternative, str):
                        continue
                    alternative = re.sub(r"\s+", " ", raw_alternative).strip()
                    if (
                        alternative
                        and alternative != target
                        and len(alternative) <= 48
                        and re.search(r"[\u3400-\u9fff]", alternative)
                        and alternative not in alternatives
                    ):
                        alternatives.append(alternative)
                    if len(alternatives) == 2:
                        break
            raw_rationale = item.get("rationale")
            rationale = (
                re.sub(r"\s+", " ", raw_rationale).strip()[:240]
                if isinstance(raw_rationale, str) else ""
            )
            if alternatives and not rationale:
                alternatives = []
            established_target = ESTABLISHED_CHINESE_TERMS.get(source.casefold())
            if established_target:
                strategy = TerminologyStrategy.TRANSLATE
                target = established_target
                alternatives = []
                explanation = ""
            elif strategy == TerminologyStrategy.BILINGUAL_ONCE:
                # Backward-compatible cache migration. The English source is
                # already visible, so a stable Chinese explanation becomes a
                # normal translation; names without one remain in English.
                chinese_target = target or explanation
                parenthetical = re.search(r"[（(]([^（）()]*[\u3400-\u9fff][^（）()]*)[）)]", chinese_target)
                if parenthetical:
                    chinese_target = parenthetical.group(1).strip()
                if source in PROTECTED_TERMS:
                    strategy = TerminologyStrategy.PRESERVE
                    target = ""
                elif chinese_target and re.search(r"[\u3400-\u9fff]", chinese_target):
                    strategy = TerminologyStrategy.TRANSLATE
                    target = chinese_target
                else:
                    strategy = TerminologyStrategy.PRESERVE
                    target = ""
                explanation = ""
            elif (
                strategy == TerminologyStrategy.PRESERVE
                and source not in PROTECTED_TERMS
                and target
                and target.casefold() != source.casefold()
                and re.search(r"[\u3400-\u9fff]", target)
            ):
                # A row cannot simultaneously preserve an English form and
                # require a different Chinese target. Except for explicitly
                # protected product/code names, the supplied Chinese target
                # makes this a translate decision regardless of capitalization.
                strategy = TerminologyStrategy.TRANSLATE
                explanation = ""
            # An incomplete strategy row cannot be enforced downstream. Keep
            # the exact source term instead of failing the whole video or
            # inventing a translation/explanation the model did not supply.
            if strategy == TerminologyStrategy.TRANSLATE and not target:
                strategy = TerminologyStrategy.PRESERVE
            if strategy != TerminologyStrategy.TRANSLATE:
                alternatives = []
            entries.append(TerminologyEntry(
                source=source, strategy=strategy, target=target,
                alternatives=alternatives, rationale=rationale,
                first_use_explanation=explanation,
                notes=(
                    str(item.get("notes") or "").strip()[:240]
                    if isinstance(item.get("notes"), str) else ""
                ),
            ))
        known = {item.source.casefold() for item in entries}
        for term, target in ESTABLISHED_CHINESE_TERMS.items():
            if _contains_term(source_text, term) and term.casefold() not in known:
                entries.append(TerminologyEntry(
                    source=term, strategy=TerminologyStrategy.TRANSLATE,
                    target=target,
                ))
                known.add(term.casefold())
        for term in PROTECTED_TERMS:
            if (
                _contains_term(source_text, term)
                and term.casefold() not in known
                and not any(_contains_term(entry.source, term) for entry in entries)
            ):
                entries.append(TerminologyEntry(
                    term, TerminologyStrategy.PRESERVE,
                ))
        return entries


def _parse_timestamp_seconds(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamp is empty")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    if len(parts) not in {2, 3} or any(not re.fullmatch(r"\d+(?:\.\d+)?", part) for part in parts):
        raise ValueError(f"invalid timestamp: {text}")
    values = [float(part) for part in parts]
    if len(values) == 2:
        minutes, seconds = values
        return minutes * 60 + seconds
    hours, minutes, seconds = values
    return hours * 3600 + minutes * 60 + seconds


def _coerce_range(raw: Any, duration: float) -> SourceRange | None:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        raw = {"start": raw[0], "end": raw[1]}
    if not isinstance(raw, dict):
        return None
    try:
        start = _parse_timestamp_seconds(raw.get("start", raw.get("start_time")))
        end = _parse_timestamp_seconds(raw.get("end", raw.get("end_time")))
    except (TypeError, ValueError):
        return None
    if start < 0 or end <= start or end > duration + 0.5:
        return None
    try:
        framing = FramingMode(str(raw.get("framing") or "auto"))
    except ValueError:
        framing = FramingMode.AUTO
    crop = raw.get("crop") if isinstance(raw.get("crop"), dict) else {}
    try:
        crop_values = {
            "crop_x": int(crop["x"]), "crop_y": int(crop["y"]),
            "crop_width": int(crop["width"]), "crop_height": int(crop["height"]),
        } if crop else {}
    except (KeyError, TypeError, ValueError):
        crop_values = {}
    if crop_values and not (
        crop_values["crop_x"] >= 0
        and crop_values["crop_y"] >= 0
        and crop_values["crop_width"] > 0
        and crop_values["crop_height"] > 0
        and crop_values["crop_x"] + crop_values["crop_width"] <= 1920
        and crop_values["crop_y"] + crop_values["crop_height"] <= 1080
    ):
        return None
    return SourceRange(
        start, min(end, duration), framing, str(raw.get("reason") or ""), **crop_values,
        original_start=float(raw["original_start"]) if raw.get("original_start") is not None else None,
        original_end=float(raw["original_end"]) if raw.get("original_end") is not None else None,
    )


def _coverage_contract_errors(
    ranges: list[SourceRange], start: float, end: float, minimum_ratio: float,
    maximum_gap: float, label: str,
) -> list[str]:
    if not ranges or end <= start:
        return [f"{label} has no valid coverage ranges"]
    errors: list[str] = []
    if ranges != sorted(ranges, key=lambda item: item.start):
        errors.append(f"{label} must be chronological")
    ordered = sorted(ranges, key=lambda item: item.start)
    cursor = start
    covered = 0.0
    for source_range in ordered:
        if source_range.start < cursor - 0.05:
            errors.append(f"{label} ranges must not overlap")
        gap = max(0.0, source_range.start - cursor)
        if gap > maximum_gap:
            errors.append(f"{label} gap exceeds {maximum_gap:.0f} seconds")
        clipped_start = max(start, source_range.start)
        clipped_end = min(end, source_range.end)
        if clipped_end > clipped_start:
            covered += clipped_end - clipped_start
        cursor = max(cursor, source_range.end)
    if end - cursor > maximum_gap:
        errors.append(f"{label} ending gap exceeds {maximum_gap:.0f} seconds")
    ratio = covered / (end - start)
    if ratio < minimum_ratio:
        errors.append(f"{label} must cover at least {minimum_ratio:.0%} of the story; got {ratio:.1%}")
    return list(dict.fromkeys(errors))


def _study_plan_contract_errors(plan: dict[str, Any], duration: float) -> list[str]:
    errors: list[str] = []
    mode = str(plan.get("editorial_mode") or "study")
    try:
        story_start = float(plan.get("story_start", 0))
        story_end = float(plan.get("story_end", duration))
    except (TypeError, ValueError):
        return ["story_start and story_end must be numeric source seconds"]
    if not 0 <= story_start < story_end <= duration + 0.5:
        errors.append("story_start/story_end must stay inside the source duration")
        story_start, story_end = 0.0, duration
    if story_end - story_start < duration * 0.9:
        errors.append("the substantive story must retain at least 90% of the source")

    raw_chapters = plan.get("bilibili_chapters") if isinstance(plan.get("bilibili_chapters"), list) else []
    chapters: list[SourceRange] = []
    chapter_titles: list[str] = []
    for raw in raw_chapters:
        if not isinstance(raw, dict):
            continue
        source_range = _coerce_range(raw, duration)
        title = str(raw.get("title") or "").strip()
        thesis = str(raw.get("thesis") or "").strip()
        hooks = [str(item).strip() for item in raw.get("hook_headlines", [])] \
            if isinstance(raw.get("hook_headlines"), list) else []
        valid_hooks = [
            item for item in hooks
            if 6 <= len(re.sub(r"\s+", "", item)) <= 30
            and not item.endswith(("，", "、", "：", "；", ",", ":", ";"))
        ]
        visible = len(re.sub(r"\s+", "", title))
        if (
            source_range and 480 <= source_range.duration <= 1800
            and 4 <= visible <= 30 and thesis
            and len(hooks) == 3 and len(valid_hooks) == 3 and len(set(hooks)) == 3
            and not title.endswith(("，", "、", "：", "；", ",", ":", ";"))
        ):
            chapters.append(source_range)
            chapter_titles.append(_normalized_title(title))
    if mode == "technical_coverage":
        if raw_chapters:
            errors.append("Bilibili is paused; technical_coverage must return no bilibili_chapters")
    else:
        if len(raw_chapters) != len(chapters) or not 1 <= len(chapters) <= 8:
            errors.append(
                "bilibili_chapters must contain 1–8 complete study chapters of 480–1800 seconds"
            )
        if story_end - story_start <= 1800 and len(chapters) != 1:
            errors.append("a story of 30 minutes or less must use exactly one complete Bilibili chapter")
        if len(set(chapter_titles)) != len(chapter_titles):
            errors.append("Bilibili chapter titles must be distinct")
        if chapters:
            errors.extend(_coverage_contract_errors(
                chapters, story_start, story_end, 0.95, 60.0, "Bilibili chapters",
            ))

    raw_lessons = plan.get("wechat_lessons") if isinstance(plan.get("wechat_lessons"), list) else []
    lessons: list[SourceRange] = []
    lesson_titles: list[str] = []
    for raw in raw_lessons:
        if not isinstance(raw, dict):
            continue
        source_range = _coerce_range(raw, duration)
        title = str(raw.get("title") or "").strip()
        thesis = str(raw.get("thesis") or "").strip()
        hooks = [str(item).strip() for item in raw.get("hook_headlines", [])] \
            if isinstance(raw.get("hook_headlines"), list) else []
        valid_hooks = [
            item for item in hooks
            if 6 <= len(re.sub(r"\s+", "", item)) <= 30
            and not item.endswith(("，", "、", "：", "；", ",", ":", ";"))
        ]
        visible = len(re.sub(r"\s+", "", title))
        if (
            source_range and 180 <= source_range.duration <= 360
            and 4 <= visible <= 30 and thesis
            and len(hooks) == 3 and len(valid_hooks) == 3 and len(set(hooks)) == 3
            and not title.endswith(("，", "、", "：", "；", ",", ":", ";"))
        ):
            lessons.append(source_range)
            lesson_titles.append(_normalized_title(title))
    span = story_end - story_start
    minimum_lessons = max(1, math.ceil(span / 360.0)) if mode == "technical_coverage" else (
        3 if duration <= 2700 else 4
    )
    maximum_lessons = min(24, max(minimum_lessons, math.floor(span / 180.0))) \
        if mode == "technical_coverage" else 8
    if len(raw_lessons) != len(lessons) or not minimum_lessons <= len(lessons) <= maximum_lessons:
        errors.append(
            f"wechat_lessons must contain {minimum_lessons}–{maximum_lessons} complete lessons of 180–360 seconds"
        )
    if len(set(lesson_titles)) != len(lesson_titles):
        errors.append("WeChat lesson titles must be distinct")
    if (mode == "technical_coverage" or duration <= 2700) and lessons:
        errors.extend(_coverage_contract_errors(
            lessons, story_start, story_end, 0.9, 60.0, "WeChat lessons",
        ))
    return errors


def _interview_clip_contract_errors(
    plan: dict[str, Any], duration: float, cues: list[TranscriptCue] | None = None,
) -> list[str]:
    errors: list[str] = []
    if plan.get("bilibili_chapters"):
        errors.append("Bilibili is paused; interview clips must not include bilibili_chapters")
    rows = [
        row for row in plan.get("wechat_lessons", []) if isinstance(row, dict)
    ] if isinstance(plan.get("wechat_lessons"), list) else []
    if len(rows) != 1:
        return [*errors, "known-tech interview must contain exactly one WeChat highlight clip"]
    raw = rows[0]
    source_range = _coerce_range(raw, duration)
    if source_range is None or not 0 < source_range.duration <= INTERVIEW_MAX_SECONDS:
        proposed = (
            "invalid"
            if source_range is None else
            f"start={source_range.start:.3f}, end={source_range.end:.3f}, duration={source_range.duration:.3f}s"
        )
        errors.append(
            "interview highlight must be a positive, complete source range no longer than "
            f"{INTERVIEW_MAX_SECONDS:.0f}s ({proposed})"
        )
    title = str(raw.get("title") or "").strip()
    thesis = str(raw.get("thesis") or "").strip()
    speaker_label = str(raw.get("speaker_label") or "").strip()
    hooks = [str(item).strip() for item in raw.get("hook_headlines", [])] \
        if isinstance(raw.get("hook_headlines"), list) else []
    valid_hooks = [
        item for item in hooks
        if 6 <= len(re.sub(r"\s+", "", item)) <= 36
        and not item.endswith(("，", "、", "：", "；", ",", ":", ";"))
        and not any(generic in item for generic in ("你知道吗", "震惊", "一定要看", "看完就懂"))
    ]
    if not 4 <= len(re.sub(r"\s+", "", title)) <= 36 or not thesis:
        errors.append("interview highlight requires a concrete title and thesis")
    if not 2 <= len(re.sub(r"\s+", "", speaker_label)) <= 24:
        errors.append("interview highlight requires a concise, evidence-backed speaker_label")
    if len(hooks) != 3 or len(valid_hooks) != 3 or len(set(hooks)) != 3:
        errors.append("interview highlight requires three distinct, specific, high-retention hooks")
    if source_range is not None and cues:
        selected_text = " ".join(
            cue.source_text for cue in cues
            if cue.end > source_range.start and cue.start < source_range.end
        )
        errors.extend(_interview_title_entailment_errors(title, selected_text))
        political = political_markers(" ".join([selected_text, speaker_label, title, thesis, *hooks]))
        if political:
            errors.append(
                "selected interview clip contains forbidden political content: "
                + ", ".join(political[:8])
            )
    return errors


def _conference_highlight_contract_errors(
    plan: dict[str, Any], duration: float, cues: list[TranscriptCue] | None = None,
) -> list[str]:
    errors: list[str] = []
    if plan.get("bilibili_chapters"):
        errors.append("Bilibili is paused; conference highlights must not include chapters")
    rows = [
        row for row in plan.get("wechat_lessons", []) if isinstance(row, dict)
    ] if isinstance(plan.get("wechat_lessons"), list) else []
    if not rows:
        return [*errors, "no_self_contained_conference_highlight"]
    if not 1 <= len(rows) <= 5:
        errors.append(f"conference highlights must contain 1–5 clips (received {len(rows)})")
    ranges: list[SourceRange] = []
    titles: list[str] = []
    for index, row in enumerate(rows, start=1):
        source_range = _coerce_range(row, duration)
        if source_range is None or not (
            CONFERENCE_HIGHLIGHT_MIN_SECONDS
            <= source_range.duration
            <= CONFERENCE_HIGHLIGHT_MAX_SECONDS
        ):
            proposed = "invalid" if source_range is None else (
                f"start={source_range.start:.3f}, end={source_range.end:.3f}, "
                f"duration={source_range.duration:.3f}s"
            )
            errors.append(
                f"conference highlight {index} must be 45–300s ({proposed})"
            )
            continue
        ranges.append(source_range)
        title = str(row.get("title") or "").strip()
        thesis = str(row.get("thesis") or "").strip()
        hooks = [str(value).strip() for value in row.get("hook_headlines", [])] \
            if isinstance(row.get("hook_headlines"), list) else []
        if not title or not thesis or len(hooks) != 3 or len(set(hooks)) != 3:
            errors.append(
                f"conference highlight {index} requires a distinct title, thesis, and three hooks"
            )
        titles.append(_normalized_title(title))
    ordered = sorted(ranges, key=lambda value: value.start)
    for previous, current in zip(ordered, ordered[1:]):
        if current.start < previous.end:
            errors.append(
                "conference highlights must not overlap "
                f"({previous.start:.3f}–{previous.end:.3f} and "
                f"{current.start:.3f}–{current.end:.3f})"
            )
    total = sum(value.duration for value in ranges)
    if total > CONFERENCE_HIGHLIGHT_MAX_TOTAL_SECONDS:
        errors.append(
            f"conference highlights total {total:.3f}s; maximum is "
            f"{CONFERENCE_HIGHLIGHT_MAX_TOTAL_SECONDS:.0f}s"
        )
    if len(set(titles)) != len(titles):
        errors.append("conference highlights must cover distinct ideas with distinct titles")
    return errors


def editorial_plan_contract_errors(
    plan: dict[str, Any], duration: float, cues: list[TranscriptCue] | None = None,
) -> list[str]:
    if str(plan.get("editorial_mode") or "") == "known_tech_interview_clip":
        return _interview_clip_contract_errors(plan, duration, cues)
    if str(plan.get("editorial_mode") or "") == "conference_highlights":
        return _conference_highlight_contract_errors(plan, duration, cues)
    if "bilibili_chapters" in plan or "wechat_lessons" in plan:
        return _study_plan_contract_errors(plan, duration)
    errors: list[str] = []
    raw_main = plan.get("main_ranges") if isinstance(plan.get("main_ranges"), list) else []
    main_ranges = [item for raw in raw_main if (item := _coerce_range(raw, duration))]
    main_duration = sum(item.duration for item in main_ranges)
    if not 900 <= main_duration <= 1320:
        errors.append(f"main edit must total 900–1320 seconds; got {main_duration:.1f}")
    raw_themes = plan.get("themes") if isinstance(plan.get("themes"), list) else []
    valid_themes = []
    for raw in raw_themes:
        if not isinstance(raw, dict):
            continue
        source_range = _coerce_range(raw, duration)
        title = str(raw.get("title") or "").strip()
        visible_title_length = len(re.sub(r"\s+", "", title))
        if (
            source_range and 270 <= source_range.duration <= 330
            and title and 4 <= visible_title_length <= 30
            and not title.endswith(("，", "、", "：", "；", ",", ":", ";"))
            and str(raw.get("thesis") or "").strip()
        ):
            valid_themes.append(raw)
            proposed = raw.get("hook_headlines")
            if proposed is not None:
                hooks = [str(item).strip() for item in proposed] if isinstance(proposed, list) else []
                valid_hooks = [
                    item for item in hooks
                    if 6 <= len(re.sub(r"\s+", "", item)) <= 30
                    and not item.endswith(("，", "、", "：", "；", ",", ":", ";"))
                ]
                if len(hooks) != 3 or len(valid_hooks) != 3 or len(set(hooks)) != 3:
                    errors.append(f"theme {title!r} must provide three distinct complete hook_headlines")
    if len(raw_themes) != len(valid_themes) or not 3 <= len(valid_themes) <= 5:
        errors.append(
            "themes must contain exactly 3–5 complete episodes, each 270–330 seconds "
            f"with title/thesis; got {len(valid_themes)} valid of {len(raw_themes)}"
        )
    titles = [_normalized_title(str(raw.get("title") or "")) for raw in valid_themes]
    if len(set(titles)) != len(titles):
        errors.append("episode titles must be distinct")
    return errors


HOOK_GREETING = re.compile(r"\b(?:welcome|hello|hi everyone|good morning|thank you)\b", re.IGNORECASE)
HOOK_SIGNAL_MARKERS = (
    "not", "never", "stop", "instead", "problem", "wrong", "can't", "won't",
    "why", "how", "only", "must", "scale", "cost", "risk", "future",
)
HOOK_CONCRETE_MARKERS = (
    "agent", "automation", "authentication", "coding", "engineering", "evaluation",
    "guardrail", "harness", "organization", "platform", "skill", "system", "team",
)
HOOK_WEAK_MARKERS = (
    "i think", "i guess", "i'm not the only one", "in this event", "kind of",
    "that's what", "that is the difference",
)
HOOK_CONCEPT_MARKERS: dict[str, tuple[str, ...]] = {
    "开源": ("open source", "open-source"), "云": ("cloud",),
    "应用": ("application", "app"), "零和": ("zero sum", "zero-sum"),
    "赢": ("win", "succeed", "success"), "需求": ("demand",),
    "算力": ("compute", "gpu"), "价格": ("price", "cost"),
    "数据": ("data",), "模型": ("model",), "代理": ("agent",),
    "收入": ("revenue",), "财报": ("earnings",),
}

# These concepts are deliberately narrower than general keyword relevance.  They
# are used as an entailment tripwire for interview titles: if a title promises a
# concrete subject, the selected source passage must actually say it.  This
# prevents a fluent model from naming a stronger idea found elsewhere in a long
# interview while selecting an unrelated passage.
INTERVIEW_TITLE_CONCEPTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "coding": (("代码", "编程", "ai coding"), ("code", "coding", "developer")),
    "documents": (("文档", "写作"), ("document", "writing", "write")),
    "competition": (("竞赛", "竞争"), ("race", "competition", "compete", "competitive cycle", "compet cycle")),
    "applications": (("应用",), ("application", "applications", "app layer")),
    "models": (("模型",), ("model", "models", "model layer")),
    "profit": (("利润", "赚钱", "收入"), ("profit", "margin", "revenue", "economic rent")),
    "memory": (("记忆", "缓存", "kv cache"), ("memory", "cache", "kv cache")),
    # In a tightly selected marketing passage the speaker may say "dashboard"
    # and "high frequency trading" without repeating the domain noun.
    "marketing": (("营销",), ("marketing", "dashboard", "high frequency trading")),
    "dashboards": (("仪表盘", "看板"), ("dashboard", "dashboards")),
    "high_frequency_trading": (("高频交易",), ("high frequency trading",)),
    "electricity": (("电力", "发电", "能源"), ("electricity", "power", "generation", "energy")),
    "open_source": (("开源", "开放权重"), ("open source", "open-source", "open weights")),
    "price": (("价格", "成本", "涨价"), ("price", "pricing", "cost", "$")),
}


def _interview_title_entailment_errors(title: str, source_text: str) -> list[str]:
    target = title.casefold()
    evidence = source_text.casefold()
    promised = {
        name for name, (title_markers, _) in INTERVIEW_TITLE_CONCEPTS.items()
        if any(marker.casefold() in target for marker in title_markers)
    }
    # A single broad concept is not discriminative enough to gate publication.
    # Two or more concrete promises are: require every one of them in the clip.
    if len(promised) < 2:
        return []
    def supported(name: str) -> bool:
        if any(
            marker.casefold() in evidence
            for marker in INTERVIEW_TITLE_CONCEPTS[name][1]
        ):
            return True
        if name == "competition":
            # A speaker can frame the economic cycle as the industry contest
            # without uttering the noun "competition".  Require the complete
            # contrast and payoff chain; a generic mention of "cycle" alone is
            # never enough.
            return (
                "cycle" in evidence
                and "industry" in evidence
                and "model" in evidence
                and "earnings" in evidence
                and any(term in evidence for term in ("application layer", "app layer"))
            )
        return False

    missing = sorted(name for name in promised if not supported(name))
    return [
        "interview title promises concepts absent from the selected source: "
        + ", ".join(missing)
    ] if missing else []


def _normalize_hook_copy(value: str) -> str:
    # SAS is a separate analytics company, but "SAS 末日论/软件模式" in an AI
    # market hook is an unambiguous misspelling of SaaS.
    return re.sub(r"\bSAS(?=\s*(?:末日|软件|模式|公司|行业))", "SaaS", value)


def _hook_semantic_hits(headline: str, thesis: str, group: list[TranscriptCue]) -> int:
    target = f"{headline} {thesis}".casefold()
    evidence = " ".join(
        f"{cue.source_text} {cue.translation}" for cue in group
    ).casefold()
    hits = 0
    for chinese, alternatives in HOOK_CONCEPT_MARKERS.items():
        if chinese in target and any(item.casefold() in evidence for item in alternatives):
            hits += 1
    named_markers = {
        item.casefold() for item in re.findall(r"[A-Za-z][A-Za-z0-9.+-]{1,}", target)
        if item.casefold() not in {"the", "and", "why", "how"}
    }
    return hits + sum(1 for marker in named_markers if marker in evidence)


def _headline_fragment(value: str, limit: int = 26) -> str:
    compact = _normalize_hook_copy(re.sub(r"\s+", " ", value).strip().rstrip("。"))
    if len(re.sub(r"\s+", "", compact)) <= limit:
        return compact
    clauses = [
        item.strip().rstrip("，、：；,;: ")
        for item in re.split(r"[，、：；。！？,;:!?]+", compact)
        if item.strip()
    ]
    complete = [
        item for item in clauses
        if 6 <= len(re.sub(r"\s+", "", item)) <= limit
    ]
    if complete:
        return complete[0]
    # Space-delimited fallback is safe for English terms because it never slices a token.
    words = compact.split()
    if len(words) > 1:
        selected: list[str] = []
        for word in words:
            candidate = " ".join([*selected, word])
            if len(re.sub(r"\s+", "", candidate)) > limit:
                break
            selected.append(word)
        candidate = " ".join(selected).rstrip("，、：；,;: ")
        if len(re.sub(r"\s+", "", candidate)) >= 6:
            return candidate
    return ""


def _wrap_visual_lines(value: str, maximum_width: int, maximum_lines: int = 2) -> str:
    """Wrap mixed Chinese/English copy without splitting English identifiers."""
    tokens = re.findall(r"[A-Za-z0-9.+-]+|\s+|.", normalize_chinese_subtitle(value))
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = current + token
        if current.strip() and _visual_width(candidate.strip()) > maximum_width:
            lines.append(current.strip())
            current = token.lstrip()
            if len(lines) == maximum_lines:
                break
        else:
            current = candidate
    if len(lines) < maximum_lines and current.strip():
        lines.append(current.strip())
    return "\n".join(lines[:maximum_lines])


def _fit_text_by_pixels(
    value: str, font_path: Path, maximum_width: int, start_size: int,
    minimum_size: int = 46, maximum_lines: int = 2,
) -> tuple[Any, str]:
    """Fit mixed Chinese/English text without clipping or splitting identifiers."""
    from PIL import ImageFont

    normalized = normalize_chinese_subtitle(value)
    tokens = re.findall(r"[A-Za-z0-9.+-]+|\s+|.", normalized)
    # Prefer a real clause boundary even when that needs a slightly smaller
    # font. Otherwise the first larger size that fits can split one Chinese
    # phrase (for example “模型竞赛”) across two lines.
    if maximum_lines == 2:
        for size in range(start_size, minimum_size - 1, -2):
            font = ImageFont.truetype(str(font_path), size)
            if font.getlength(normalized) <= maximum_width:
                return font, normalized
            semantic: list[tuple[float, str]] = []
            for index in range(1, len(tokens)):
                first = "".join(tokens[:index]).strip()
                second = "".join(tokens[index:]).strip()
                if (
                    not first or not second
                    or first[-1] not in "：，。；！？:,.;!?"
                    or second[0] in "、，。；：！？》）】〕〉］},.;:!?)]"
                ):
                    continue
                first_width = font.getlength(first)
                second_width = font.getlength(second)
                if max(first_width, second_width) > maximum_width:
                    continue
                if first_width < maximum_width * 0.58:
                    continue
                semantic.append((
                    abs(first_width - maximum_width * 0.88)
                    + abs(first_width - second_width) * 0.18
                    - maximum_width * 0.05,
                    f"{first}\n{second}",
                ))
            if semantic:
                return font, min(semantic, key=lambda item: item[0])[1]
    for size in range(start_size, minimum_size - 1, -2):
        font = ImageFont.truetype(str(font_path), size)
        if font.getlength(normalized) <= maximum_width:
            return font, normalized
        if maximum_lines > 2:
            lines: list[str] = []
            current = ""
            for token in tokens:
                candidate = current + token
                if current.strip() and font.getlength(candidate.strip()) > maximum_width:
                    lines.append(current.strip())
                    current = token.lstrip()
                else:
                    current = candidate
            if current.strip():
                lines.append(current.strip())
            if len(lines) <= maximum_lines:
                return font, "\n".join(lines)
        if maximum_lines < 2:
            continue
        candidates: list[tuple[float, str]] = []
        for index in range(1, len(tokens)):
            first = "".join(tokens[:index]).strip()
            second = "".join(tokens[index:]).strip()
            if not first or not second:
                continue
            # Chinese kinsoku rules: closing/list punctuation must not begin a
            # line, while opening punctuation must not be left at line end.
            if second[0] in "、，。；：！？）》】〕〉］},.;:!?)]":
                continue
            if first[-1] in "（《【〔〈［{([":
                continue
            first_width = font.getlength(first)
            second_width = font.getlength(second)
            if max(first_width, second_width) > maximum_width:
                continue
            # Left-aligned mobile headlines should use the first line before
            # wrapping. A former large colon bonus produced a short premise on
            # line one and an unnecessarily crowded reveal on line two.
            target_first_width = maximum_width * 0.88
            first_fill_penalty = abs(first_width - target_first_width)
            balance_penalty = abs(first_width - second_width) * 0.18
            punctuation_bonus = (
                maximum_width * 0.05
                if first[-1] in "：，。；！？:,.;!?" else 0.0
            )
            orphan_penalty = (
                maximum_width * 0.4 if first_width < maximum_width * 0.58 else 0.0
            )
            candidates.append((
                first_fill_penalty + balance_penalty + orphan_penalty - punctuation_bonus,
                f"{first}\n{second}",
            ))
        if candidates:
            return font, min(candidates, key=lambda item: item[0])[1]
    font = ImageFont.truetype(str(font_path), minimum_size)
    return font, _wrap_visual_lines(
        normalized, max(1, maximum_width * 2 // minimum_size), maximum_lines,
    )


def _interview_hook_context_fits_overlay(value: str) -> bool:
    """Validate fixed copy against the real WeChat hook panel geometry.

    Character counts are a poor proxy for mixed Chinese/English copy.  This
    uses the same font, width, sizes, line count, and spacing as the renderer
    and also verifies that its fallback wrapper did not truncate any text.
    """
    from PIL import Image, ImageDraw

    normalized = normalize_chinese_subtitle(value)
    if not normalized:
        return False
    maximum_width = 1080 - 156
    font, wrapped = _fit_text_by_pixels(
        normalized, _resolve_chinese_subtitle_font_path(), maximum_width, 30,
        minimum_size=24, maximum_lines=3,
    )
    if re.sub(r"\s+", "", wrapped) != re.sub(r"\s+", "", normalized):
        return False
    lines = wrapped.splitlines()
    if not 1 <= len(lines) <= 3 or any(
        font.getlength(line) > maximum_width + 1 for line in lines
    ):
        return False
    draw = ImageDraw.Draw(Image.new("RGBA", (1080, 430), (0, 0, 0, 0)))
    bbox = draw.multiline_textbbox(
        (0, 0), wrapped, font=font, anchor="la", align="left", spacing=7,
    )
    # The renderer places this block at y=306 inside a panel ending at y=410.
    return bbox[3] <= 104


def _fit_interview_hook_headline(value: str, font_path: Path) -> tuple[Any, str]:
    """Keep the prominent first-version wrap unless it exceeds safe width."""
    from PIL import ImageFont

    maximum_width = 1080 - 156
    preferred = wrap_subtitle(value, 18)
    preferred_font = ImageFont.truetype(str(font_path), 64)
    if (
        1 <= len(preferred.splitlines()) <= 2
        and all(
            preferred_font.getlength(line) <= maximum_width + 1
            for line in preferred.splitlines()
        )
    ):
        return preferred_font, preferred
    return _fit_text_by_pixels(
        value, font_path, maximum_width, 64, minimum_size=48,
    )


_SPOKEN_FILLER_ONLY = re.compile(
    r"^(?:(?:um+|uh+|erm+|hmm+|mm+|you know|i mean|like|well|so)[\s,.;:!?-]*)+$",
    re.IGNORECASE,
)


def omit_non_speech_directions(value: str) -> str:
    """Remove caption metadata that describes a sound instead of speech."""
    cleaned = NON_SPEECH_DIRECTION.sub(" ", value)
    cleaned = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?，。；：！？])(?:\s*[,.;:!?，。；：！？])+", r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ,，")


def source_is_non_speech_only(value: str) -> bool:
    stripped = omit_non_speech_directions(value)
    return stripped != value.strip() and not stripped.strip(" .,!?:;，。！？：；")


def source_is_spoken_filler_only(value: str) -> bool:
    speech = omit_non_speech_directions(value)
    return bool(_SPOKEN_FILLER_ONLY.fullmatch(re.sub(r"\s+", " ", speech).strip()))


def source_is_omittable_caption_only(value: str) -> bool:
    return source_is_non_speech_only(value) or source_is_spoken_filler_only(value)


_VERSIONED_PRODUCT_REVEAL = re.compile(
    r"^\s*(?:and\s+)?here(?:\s+is|'s)\s+"
    r"(?P<name>[A-Z][A-Za-z0-9._+-]*(?:\s+[A-Za-z0-9][A-Za-z0-9._+-]*){0,3})"
    r"(?=\s*[,!.?:;]|\s*$)",
    re.IGNORECASE,
)


def normalize_product_reveal_translation(source: str, translation: str) -> str:
    """Correct one high-confidence launch idiom without rewriting normal deixis.

    The deterministic fallback is intentionally narrow: the introduced name
    must contain a numeric version and the Chinese must have copied both the
    literal ``这是`` prefix and exact product spelling.  Broader cases stay
    with the contextual translator instead of being guessed in code.
    """
    match = _VERSIONED_PRODUCT_REVEAL.match(source)
    if match is None:
        return translation
    name = match.group("name").strip().rstrip(".!?:;")
    if not re.search(r"\d", name):
        return translation
    literal = re.match(
        rf"^\s*这是\s*{re.escape(name)}(?P<tail>.*)$",
        normalize_chinese_subtitle(translation),
        re.IGNORECASE,
    )
    if literal is None:
        return translation
    tail = literal.group("tail").strip()
    # “这是 Fable 5.1 模型” carries no extra meaning beyond the reveal.
    # Preserve any real following clause, however.
    tail = re.sub(r"^(?:这个|一款|一个)?\s*模型(?=[，,。.!！？?]|$)", "", tail).strip()
    if not tail.strip("，,。.!！？?：:；; "):
        return f"{name} 来了。"
    if tail.startswith(("，", ",")):
        return normalize_chinese_subtitle(f"{name} 来了{tail}")
    return normalize_chinese_subtitle(f"{name} 来了，{tail}")


def omit_spoken_fillers_from_translation(source: str, translation: str) -> str:
    """Remove hesitation and non-speech metadata while retaining speech meaning."""
    if source_is_omittable_caption_only(source):
        return ""
    cleaned = normalize_chinese_subtitle(omit_non_speech_directions(translation))
    if NON_SPEECH_DIRECTION.search(source):
        # A model may paraphrase the stage direction without brackets. Only
        # remove it when the source explicitly contains that direction.
        cleaned = re.sub(
            r"^(?:清(?:了清)?嗓(?:子)?|清喉咙|咳嗽(?:了)?|叹(?:了)?口气|笑(?:了)?|"
            r"掌声|音乐)[，,、。.!！？?\s]*",
            "", cleaned,
        )
    lowered = source.casefold()
    if re.search(r"\b(?:um+|uh+|erm+|hmm+|mm+)\b", lowered):
        cleaned = re.sub(
            r"(^|[，,、。.!！？?\s])(?:嗯+|呃+|额+|啊+|唔+)(?=$|[，,、。.!！？?\s])",
            r"\1", cleaned,
        )
    if re.match(r"^\s*(?:um+|uh+|erm+|hmm+|mm+)\b", lowered):
        cleaned = re.sub(r"^(?:嗯+|呃+|额+|啊+|唔+)[，,、。.!？?\s]*", "", cleaned)
    if "you know" in lowered:
        cleaned = re.sub(r"(?:你知道(?:吗)?|大家知道)[，,、。.!？?\s]*", "", cleaned)
    if re.search(r"\bi mean\b", lowered):
        cleaned = re.sub(r"(?:我是说|我的意思是)[，,、。.!？?\s]*", "", cleaned)
    if re.match(r"^\s*boom\b", lowered):
        cleaned = re.sub(
            r"^(?:boom|砰(?:的?一声|一下)?|嘭(?:的?一声|一下)?)[，,、。.!！？?\s]*",
            "", cleaned, flags=re.IGNORECASE,
        )
    if re.search(r"\b(?:um+|uh+|erm+|hmm+|mm+)\b[\s,.;:!?-]*$", lowered):
        cleaned = re.sub(r"[，,、。.!？?\s]*(?:嗯+|呃+|额+|啊+|唔+|呢)[，,、。.!？?\s]*$", "", cleaned)
    cleaned = re.sub(r"[，,、]+(?=[。.!！？?])", "", cleaned)
    cleaned = re.sub(r"^[，,、。.!？?\s]+|[，,、\s]+$", "", cleaned)
    return normalize_product_reveal_translation(source, normalize_chinese_subtitle(cleaned))


def omit_spoken_fillers(cues: list[TranscriptCue]) -> list[str]:
    changed: list[str] = []
    for cue in cues:
        cleaned = omit_spoken_fillers_from_translation(cue.source_text, cue.translation)
        if cleaned != cue.translation:
            cue.translation = cleaned
            changed.append(cue.id)
    return changed


def build_hook_candidates(
    title: str, thesis: str, source_range: SourceRange, cues: list[TranscriptCue], item_id: str,
    proposed_headlines: Any = None, speaker_label: str = "", hook_context: str = "",
) -> list[HookSpec]:
    headline_limit = 36 if speaker_label.strip() else 30
    eligible = [
        cue for cue in cues
        if cue.start >= source_range.start and cue.end <= source_range.end
        and cue.source_text.strip() and not HOOK_GREETING.search(cue.source_text)
    ]
    windows: list[tuple[float, list[TranscriptCue]]] = []
    for index, cue in enumerate(eligible):
        group = [cue]
        end_index = index + 1
        while (
            group[-1].end - group[0].start < 6.0
            or SOURCE_DANGLING_END.search(group[-1].source_text.strip())
        ) and end_index < len(eligible):
            following = eligible[end_index]
            if following.start - group[-1].end > 0.8 or following.end - group[0].start > 10.0:
                break
            group.append(following)
            end_index += 1
        duration = group[-1].end - group[0].start
        if not 6.0 <= duration <= 10.0 or SOURCE_DANGLING_END.search(
            group[-1].source_text.strip()
        ):
            continue
        combined = " ".join(item.source_text for item in group).casefold()
        signal_hits = sum(1 for marker in HOOK_SIGNAL_MARKERS if marker in combined)
        concrete_hits = sum(1 for marker in HOOK_CONCRETE_MARKERS if marker in combined)
        weak_hits = sum(1 for marker in HOOK_WEAK_MARKERS if marker in combined)
        filler_hits = len(re.findall(r"\b(?:um|uh|yeah|yes)\b|\byou know\b", combined))
        specificity = min(4, len(re.findall(r"\b[A-Z][A-Za-z0-9.+-]*\b", " ".join(item.source_text for item in group))))
        score = (
            signal_hits * 3 + concrete_hits * 2 + specificity
            + min(5, len(combined) / 35) - weak_hits * 4 - filler_hits * 2
        )
        windows.append((score, group))
    windows.sort(key=lambda row: (-row[0], row[1][0].start))
    distinct_windows: list[tuple[float, list[TranscriptCue]]] = []
    for score, group in windows:
        if any(abs(group[0].start - existing[1][0].start) < 2.0 for existing in distinct_windows):
            continue
        distinct_windows.append((score, group))
    if not distinct_windows:
        raise ValueError(f"{item_id}: no evidence-backed 6–10 second hook window")

    title_headline = _headline_fragment(title, headline_limit)
    if not title_headline:
        raise ValueError(f"{item_id}: title cannot be shortened without breaking a clause or English term")
    subject = _headline_fragment(title.split("：", 1)[0], 18) or title_headline
    fallback_headlines = [
        title_headline,
        _headline_fragment(thesis, headline_limit) or f"{subject}的关键取舍是什么？",
        _headline_fragment(f"{subject}真正改变了什么？", headline_limit) or "真正的工程代价是什么？",
    ]
    supplied = [str(item).strip() for item in proposed_headlines] if isinstance(proposed_headlines, list) else []
    headlines: list[str] = []
    for raw in [*supplied, *fallback_headlines]:
        headline = _headline_fragment(raw, headline_limit)
        if headline and headline not in headlines:
            headlines.append(headline)
        if len(headlines) == 3:
            break
    if len(headlines) != 3:
        raise ValueError(f"{item_id}: three complete, distinct hook headlines are required")
    selected_windows: list[list[TranscriptCue]] = []
    for headline in headlines:
        available = [
            row for row in distinct_windows
            if not any(abs(row[1][0].start - used[0].start) < 2.0 for used in selected_windows)
        ] or distinct_windows
        _, chosen = max(
            available,
            key=lambda row: (
                _hook_semantic_hits(headline, thesis, row[1]) * 8 + row[0],
                -row[1][0].start,
            ),
        )
        selected_windows.append(chosen)
    strategies = [HookStrategy.CONTRARIAN, HookStrategy.QUESTION, HookStrategy.OUTCOME]
    hooks: list[HookSpec] = []
    for index, (group, headline, strategy) in enumerate(zip(selected_windows, headlines, strategies), start=1):
        hooks.append(HookSpec(
            id=f"{item_id}-hook-{index}", strategy=strategy,
            headline_zh=headline, promise=hook_context.strip() or thesis,
            source_range=SourceRange(
                group[0].start, group[-1].end,
                source_range.framing if source_range.framing != FramingMode.AUTO
                else (FramingMode.SPEAKER if group[0].speaker else FramingMode.AUTO),
                "evidence-backed cold open",
                source_range.crop_x, source_range.crop_y,
                source_range.crop_width, source_range.crop_height,
                original_start=(group[0].original_start if group[0].original_start is not None else None),
                original_end=(group[-1].original_end if group[-1].original_end is not None else None),
            ),
            source_cue_ids=[item.id for item in group],
            payoff_cue_ids=[item.id for item in group],
            speaker_label=speaker_label.strip(),
            selected=index == 1,
        ))
    return hooks


def hook_contract_errors(
    hook: HookSpec, item: CollectionItem, transcript: list[TranscriptCue],
    profile: RenderProfile | None = None,
) -> list[str]:
    errors: list[str] = []
    if not 6.0 <= hook.source_range.duration <= 10.0:
        errors.append("hook duration must be 6–10 seconds")
    episode_start = min(item.source_ranges, key=lambda row: row.start).start
    episode_end = max(item.source_ranges, key=lambda row: row.end).end
    if not episode_start <= hook.source_range.start < hook.source_range.end <= episode_end:
        errors.append("hook range must stay inside the episode")
    cue_ids = {cue.id for cue in transcript}
    if not hook.source_cue_ids or not set(hook.source_cue_ids) <= cue_ids:
        errors.append("hook source cue ids are missing or invalid")
    if not hook.payoff_cue_ids or not set(hook.payoff_cue_ids) <= cue_ids:
        errors.append("hook payoff cue ids are missing or invalid")
    if HOOK_GREETING.search(" ".join(
        cue.source_text for cue in transcript if cue.id in set(hook.source_cue_ids)
    )):
        errors.append("hook must not begin with a greeting")
    source_hook_cues = [
        cue for cue in transcript if cue.id in set(hook.source_cue_ids)
    ]
    if source_hook_cues and SOURCE_DANGLING_END.search(source_hook_cues[-1].source_text.strip()):
        errors.append("hook evidence must end on a complete source thought")
    headline_length = len(re.sub(r"\s+", "", hook.headline_zh))
    headline_limit = 36 if hook.speaker_label.strip() else 30
    if not 6 <= headline_length <= headline_limit:
        errors.append(f"hook headline must contain 6–{headline_limit} visible characters")
    if any(phrase in hook.headline_zh for phrase in ("你知道吗", "震惊", "一定要看", "看完就懂")):
        errors.append("generic clickbait hook is forbidden")
    if not hook.promise.strip():
        errors.append("hook promise is required")
    if profile in {None, RenderProfile.WECHAT_VERTICAL} and not hook.persistent_title:
        errors.append("WeChat hook headline must persist through the full video")
    return errors


def _remap_hook_snapshot(
    hooks: list[HookSpec] | None, cues: list[TranscriptCue], item_id: str,
) -> list[HookSpec]:
    """Keep pre-subtitle hook choices while binding them to final card IDs."""
    if not hooks:
        return []
    remapped: list[HookSpec] = []
    for index, hook in enumerate(hooks, start=1):
        cue_ids = [
            cue.id for cue in cues
            if cue.end > hook.source_range.start
            and cue.start < hook.source_range.end
        ]
        if not cue_ids:
            raise ValueError(
                f"{item_id}: frozen hook range has no final subtitle cards"
            )
        remapped.append(HookSpec(
            id=f"{item_id}-hook-{index}", strategy=hook.strategy,
            headline_zh=hook.headline_zh, promise=hook.promise,
            source_range=hook.source_range, source_cue_ids=cue_ids,
            payoff_cue_ids=list(cue_ids), speaker_label=hook.speaker_label,
            motion=hook.motion, persistent_title=hook.persistent_title,
            selected=hook.selected,
        ))
    return remapped


def _snapshot_plan_hooks(
    plan: dict[str, Any], duration: float, cues: list[TranscriptCue],
    known_people: list[str] | None = None,
) -> dict[str, list[HookSpec]]:
    """Freeze hook text and time windows before subtitle cards are regenerated."""
    snapshot: dict[str, list[HookSpec]] = {}
    for key, prefix in (("bilibili_chapters", "bilibili"), ("wechat_lessons", "wechat")):
        rows = plan.get(key)
        if not isinstance(rows, list):
            continue
        for index, raw in enumerate(rows, start=1):
            if not isinstance(raw, dict):
                continue
            source_range = _coerce_range(raw, duration)
            if source_range is None:
                continue
            speaker_label = str(raw.get("speaker_label") or "")
            if (
                not speaker_label and known_people
                and str(plan.get("editorial_mode") or "")
                == "known_tech_interview_clip"
            ):
                speaker_label = known_people[0]
            snapshot[f"{prefix}:{index}"] = build_hook_candidates(
                str(raw.get("title") or "").strip(),
                str(raw.get("thesis") or "").strip(),
                source_range, cues, f"snapshot-{prefix}-{index}",
                raw.get("hook_headlines"), speaker_label,
                str(raw.get("hook_context") or ""),
            )
    return snapshot


def build_collection_manifest(
    candidate: Candidate, metadata: dict[str, Any], cues: list[TranscriptCue],
    terminology: list[TerminologyEntry], plan: dict[str, Any], source_media_path: str,
    source_subtitle_path: str, source_media_info: SourceMediaInfo | None = None,
    hook_snapshot: dict[str, list[HookSpec]] | None = None,
) -> VideoCollectionManifest:
    duration = float(metadata.get("duration") or (cues[-1].end if cues else 0))
    contract_errors = editorial_plan_contract_errors(plan, duration, cues)
    if contract_errors:
        raise ValueError("editorial plan contract failed: " + "; ".join(contract_errors))
    if "bilibili_chapters" in plan or "wechat_lessons" in plan:
        collection_id = f"youtube-{candidate.metadata.get('video_id') or candidate.id}-{uuid.uuid4().hex[:8]}"
        source_line = f"来源：{candidate.author or metadata.get('channel') or 'YouTube'}｜{candidate.source_url}"
        items: list[CollectionItem] = []
        editorial_mode = str(plan.get("editorial_mode") or "study")
        known_people = [
            str(value).strip() for value in metadata.get("known_tech_people", [])
            if str(value).strip()
        ]
        short_tags = (
            ["AI", "科技人物", "对谈高光", "中文字幕"]
            if editorial_mode == "known_tech_interview_clip"
            else ["AI", "开发者", "技术分享", "短课"]
        )
        chapter_rows = [
            raw for raw in plan.get("bilibili_chapters", []) if isinstance(raw, dict)
        ]
        for index, raw in enumerate(chapter_rows, start=1):
            source_range = _coerce_range(raw, duration)
            if source_range is None:
                raise ValueError(f"Bilibili chapter {index} has an invalid source range")
            title = str(raw.get("title") or f"学习章节 {index}").strip()
            thesis = str(raw.get("thesis") or "保留原视频中的完整论证。").strip()
            item_id = f"{collection_id}-chapter-{index}"
            hook_candidates = _remap_hook_snapshot(
                (hook_snapshot or {}).get(f"bilibili:{index}"), cues, item_id,
            ) or build_hook_candidates(
                title, thesis, source_range, cues, item_id, raw.get("hook_headlines"),
            )
            items.append(CollectionItem(
                id=item_id,
                kind=CollectionItemKind.BILIBILI_CHAPTER, order=index,
                title=title, thesis=thesis, source_ranges=[source_range],
                renders=[PlatformRender(
                    RenderProfile.BILIBILI_LANDSCAPE, 1920, 1080, title=title,
                    description=source_line, tags=["AI", "AI工程", "中文字幕", "学习合集"],
                    hook_candidates=hook_candidates, selected_hook=hook_candidates[0],
                )],
            ))
        lesson_rows = [
            raw for raw in plan.get("wechat_lessons", []) if isinstance(raw, dict)
        ]
        for index, raw in enumerate(lesson_rows, start=1):
            source_range = _coerce_range(raw, duration)
            if source_range is None:
                raise ValueError(f"WeChat lesson {index} has an invalid source range")
            title = str(raw.get("title") or f"核心短课 {index}").strip()
            thesis = str(raw.get("thesis") or "解释一个可独立学习的完整观点。").strip()
            item_id = f"{collection_id}-short-{index}"
            hook_candidates = _remap_hook_snapshot(
                (hook_snapshot or {}).get(f"wechat:{index}"), cues, item_id,
            ) or build_hook_candidates(
                title, thesis, source_range, cues, item_id, raw.get("hook_headlines"),
                str(raw.get("speaker_label") or (
                    known_people[0] if editorial_mode == "known_tech_interview_clip" and known_people else ""
                )),
                str(raw.get("hook_context") or ""),
            )
            items.append(CollectionItem(
                id=item_id, kind=CollectionItemKind.WECHAT_SHORT,
                order=len(chapter_rows) + index, title=title, thesis=thesis,
                source_ranges=[source_range],
                renders=[PlatformRender(
                    RenderProfile.WECHAT_VERTICAL, 1080, 1920, title=title,
                    description=source_line, tags=short_tags,
                    hook_candidates=hook_candidates, selected_hook=hook_candidates[0],
                )],
            ))
        return VideoCollectionManifest(
            id=collection_id, candidate_id=candidate.id, source_url=candidate.source_url,
            source_video_id=str(candidate.metadata.get("video_id") or ""),
            source_title=candidate.title, source_channel=candidate.author or "",
            collection_title=str(plan.get("collection_title") or f"{candidate.author or 'AI'} 中文学习合集").strip(),
            transcript=cues, terminology=terminology, items=items,
            editorial_mode=editorial_mode,
            source_media_path=source_media_path, source_subtitle_path=source_subtitle_path,
            source_duration=duration, source_media_info=source_media_info,
            rights_review=RightsReview(),
        )
    raw_main = plan.get("main_ranges") if isinstance(plan.get("main_ranges"), list) else []
    main_ranges = [item for raw in raw_main if (item := _coerce_range(raw, duration))]
    main_duration = sum(item.duration for item in main_ranges)
    if not 900 <= main_duration <= 1320:
        raise ValueError(
            f"editorial plan must provide a coherent 15–22 minute main edit; got {main_duration:.1f}s"
        )

    theme_rows = [item for item in plan.get("themes", []) if isinstance(item, dict)] if isinstance(plan.get("themes"), list) else []
    theme_ranges: list[tuple[dict[str, Any], SourceRange]] = []
    for raw in theme_rows:
        source_range = _coerce_range(raw, duration)
        title = str(raw.get("title") or "").strip()
        thesis = str(raw.get("thesis") or "").strip()
        if source_range and title and thesis and 270 <= source_range.duration <= 330:
            theme_ranges.append((raw, source_range))
    if not 3 <= len(theme_ranges) <= 5:
        raise ValueError(
            "editorial plan must provide 3–5 complete thematic episodes of 270–330 seconds; "
            f"got {len(theme_ranges)} valid episodes"
        )
    normalized_titles = [_normalized_title(str(raw["title"])) for raw, _ in theme_ranges]
    if len(set(normalized_titles)) != len(normalized_titles):
        raise ValueError("editorial plan episode titles must be distinct")

    collection_id = f"youtube-{candidate.metadata.get('video_id') or candidate.id}-{uuid.uuid4().hex[:8]}"
    source_line = f"来源：{candidate.author or metadata.get('channel') or 'YouTube'}｜{candidate.source_url}"
    main_title = str(plan.get("main_title") or candidate.title).strip()
    items = [CollectionItem(
        id=f"{collection_id}-main", kind=CollectionItemKind.MAIN, order=1,
        title=main_title, thesis=str(plan.get("main_thesis") or "保留讲者的完整核心论证。"),
        source_ranges=main_ranges,
        renders=[PlatformRender(
            RenderProfile.BILIBILI_LANDSCAPE, 1920, 1080, title=main_title,
            description=source_line, tags=["AI", "AI工程", "中文字幕"],
        )],
    )]
    for index, (raw, source_range) in enumerate(theme_ranges, start=1):
        episode_id = f"{collection_id}-episode-{index}"
        title = str(raw.get("title") or f"核心观点 {index}").strip()
        thesis = str(raw.get("thesis") or "围绕原始演讲中的一个完整技术观点。").strip()
        hook_candidates = build_hook_candidates(
            title, thesis, source_range, cues, episode_id, raw.get("hook_headlines"),
        )
        items.append(CollectionItem(
            id=episode_id, kind=CollectionItemKind.EPISODE,
            order=index + 1, title=title, thesis=thesis, source_ranges=[source_range],
            renders=[
                PlatformRender(
                    RenderProfile.BILIBILI_LANDSCAPE, 1920, 1080, title=title,
                    description=source_line, tags=["AI", "AI工程", "中文字幕"],
                ),
                PlatformRender(
                    RenderProfile.WECHAT_VERTICAL, 1080, 1920, title=title,
                    description=source_line, tags=["AI", "开发者", "技术团队"],
                    hook_candidates=hook_candidates, selected_hook=hook_candidates[0],
                ),
            ],
        ))
    return VideoCollectionManifest(
        id=collection_id, candidate_id=candidate.id, source_url=candidate.source_url,
        source_video_id=str(candidate.metadata.get("video_id") or ""),
        source_title=candidate.title, source_channel=candidate.author or "",
        collection_title=str(plan.get("collection_title") or f"{candidate.author or 'AI'} 中文精选").strip(),
        transcript=cues, terminology=terminology, items=items,
        editorial_mode=str(plan.get("editorial_mode") or "study"),
        source_media_path=source_media_path, source_subtitle_path=source_subtitle_path,
        source_duration=duration, source_media_info=source_media_info, rights_review=RightsReview(),
    )


class YouTubeAcquirer:
    def __init__(
        self, workspace: Workspace,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        runtime: ManagedYouTubeRuntime | None = None,
    ) -> None:
        self.workspace = workspace
        self.runner = runner or subprocess.run
        self.custom_runner = runner is not None
        self.runtime = runtime or ManagedYouTubeRuntime()

    def acquire(
        self, url: str, job: Path, local_media: Path | None = None,
        local_subtitles: Path | None = None,
        download_media: bool = True,
    ) -> tuple[
        Candidate, list[Evidence], dict[str, Any], list[TranscriptCue], str, str,
        SourceMediaInfo | None,
    ]:
        job.mkdir(parents=True, exist_ok=True)
        try:
            metadata = self._metadata(url)
        except YouTubeAcquisitionError:
            metadata = self._cached_metadata(url)
            if metadata is None:
                raise
        video_id = str(metadata.get("id") or "")
        if not video_id:
            raise YouTubeAcquisitionError("YouTube metadata has no video id")
        metadata_path = job / f"{video_id}.metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if local_subtitles is not None:
            if not local_subtitles.is_file():
                raise FileNotFoundError(local_subtitles)
            if local_subtitles.suffix.casefold() != ".json3":
                raise YouTubeAcquisitionError("local YouTube subtitles must use yt-dlp json3 format")
            subtitle = local_subtitles
        else:
            try:
                subtitle = self._subtitle(url, video_id, job)
            except YouTubeAcquisitionError:
                # Refresh should prefer live captions, but a transient YouTube
                # transport failure must not make an otherwise reproducible
                # generation job dependent on Codex.  The content-addressed
                # archive is the agent's second acquisition path.
                subtitle = self._cached_subtitle(video_id)
                if subtitle is None:
                    raise
        cues = parse_youtube_json3(subtitle)
        if not cues:
            raise YouTubeAcquisitionError("YouTube transcript is empty after normalization")
        candidate = Candidate(
            id=f"youtube-{video_id}", source_type=SourceType.YOUTUBE, source_url=url,
            title=str(metadata.get("title") or video_id),
            author=str(metadata.get("channel") or metadata.get("uploader") or ""),
            published_at=str(metadata.get("upload_date") or ""), dedupe_key=f"youtube:{video_id}",
            metadata={
                "video_id": video_id, "duration": metadata.get("duration"),
                "chapters": metadata.get("chapters", []), "extractor_client": "mweb",
                "creators": metadata.get("creators", []),
            },
        )
        self.workspace.save_candidate(candidate)
        metadata_asset, metadata_hash = self.workspace.archive_asset(metadata_path, "youtube-metadata", metadata_path.name)
        subtitle_asset, subtitle_hash = self.workspace.archive_asset(subtitle, "youtube-subtitles", subtitle.name)
        evidence = [
            Evidence(
                id=f"{candidate.id}-metadata", candidate_id=candidate.id, url=url,
                quote=json.dumps({
                    "title": candidate.title, "channel": candidate.author,
                    "description": metadata.get("description", ""), "chapters": metadata.get("chapters", []),
                }, ensure_ascii=False),
                source_kind="youtube:metadata", captured_asset=metadata_asset, sha256=metadata_hash,
                metadata={"video_id": video_id, "client": "mweb"},
            ),
            Evidence(
                id=f"{candidate.id}-transcript", candidate_id=candidate.id, url=url,
                quote="\n".join(item.source_text for item in cues), source_kind="youtube:transcript",
                captured_asset=subtitle_asset, sha256=subtitle_hash,
            ),
        ]
        media_asset = ""
        media_info: SourceMediaInfo | None = None
        if local_media:
            probe = probe_video(local_media)
            self._require_1080p(probe.width, probe.height, local_media)
            self._require_complete_audio(probe, local_media)
            media_asset, media_hash = self.workspace.archive_asset(local_media, "youtube-video", local_media.name)
            media_info = self._source_media_info(probe, media_hash, "local", "local")
            evidence.append(Evidence(
                id=f"{candidate.id}-video", candidate_id=candidate.id, url=url,
                quote="Original source video supplied locally.", source_kind="youtube:video",
                captured_asset=media_asset, sha256=media_hash,
            ))
        elif download_media:
            downloaded = self._download_media(url, video_id, job)
            probe = probe_video(downloaded)
            self._require_1080p(probe.width, probe.height, downloaded)
            media_asset, media_hash = self.workspace.archive_asset(downloaded, "youtube-video", downloaded.name)
            media_info = self._source_media_info(
                probe, media_hash, str(metadata.get("format_id") or "best-1080+"), "mweb",
            )
            evidence.append(Evidence(
                id=f"{candidate.id}-video", candidate_id=candidate.id, url=url,
                quote="Original YouTube source video.", source_kind="youtube:video",
                captured_asset=media_asset, sha256=media_hash,
            ))
        for item in evidence:
            self.workspace.save_evidence(item)
        return candidate, evidence, metadata, cues, media_asset, subtitle_asset, media_info

    def acquire_remote_media(
        self, candidate: Candidate, metadata: dict[str, Any], url: str, job: Path,
        source_range: SourceRange | None = None,
        boundary_padding: float = INTERVIEW_BOUNDARY_PADDING_SECONDS,
    ) -> tuple[str, SourceMediaInfo, Evidence, dict[str, float] | None]:
        """Download/archive either the complete source or one padded interview interval."""
        video_id = str(metadata.get("id") or candidate.metadata.get("video_id") or "")
        if not video_id:
            raise YouTubeAcquisitionError("YouTube metadata has no video id")
        original_duration = float(metadata.get("duration") or 0)
        download_window: dict[str, float] | None = None
        if source_range is not None:
            start = max(0.0, source_range.start - max(0.0, boundary_padding))
            end = source_range.end + max(0.0, boundary_padding)
            if original_duration > 0:
                end = min(original_duration, end)
            download_window = {
                "original_start": source_range.start,
                "original_end": source_range.end,
                "download_start": start,
                "download_end": end,
            }
        requested_window = (
            (download_window["download_start"], download_window["download_end"])
            if download_window else None
        )
        for attempt in range(2):
            try:
                downloaded = self._download_media(
                    url, video_id, job, download_window=requested_window,
                )
            except YouTubeAcquisitionError:
                if attempt:
                    raise
                continue
            probe = probe_video(downloaded)
            try:
                self._require_complete_audio(probe, downloaded)
            except YouTubeAcquisitionError:
                if attempt:
                    raise
                # A bounded transfer can exit zero after one stream ends early.
                # Remove only this job's incomplete generated file. Direct
                # remote section seeks are not resumable when Google closes
                # one DASH stream early, so retry by downloading the complete
                # source through yt-dlp's resumable native path. The renderer
                # still uses only the exact selected source range.
                downloaded.unlink(missing_ok=True)
                if requested_window is not None:
                    requested_window = None
                    download_window = None
                continue
            break
        self._require_1080p(probe.width, probe.height, downloaded)
        media_asset, media_hash = self.workspace.archive_asset(
            downloaded, "youtube-video", downloaded.name,
        )
        media_info = self._source_media_info(
            probe, media_hash, str(metadata.get("format_id") or "best-1080+"), "mweb",
        )
        evidence = Evidence(
            id=f"{candidate.id}-video", candidate_id=candidate.id, url=url,
            quote=(
                "Original YouTube source video interval."
                if download_window else "Original YouTube source video."
            ),
            source_kind="youtube:video", captured_asset=media_asset, sha256=media_hash,
            metadata={"source_clip": download_window} if download_window else {},
        )
        self.workspace.save_evidence(evidence)
        return media_asset, media_info, evidence, download_window

    def _cached_metadata(self, url: str) -> dict[str, Any] | None:
        match = re.search(r"(?:[?&]v=|youtu\.be/)([A-Za-z0-9_-]{6,})", url)
        if not match:
            return None
        video_id = match.group(1)
        candidates = [
            *self.workspace.root.glob(f"jobs/*/{video_id}.metadata.json"),
            *self.workspace.root.glob(f"assets/youtube-metadata/*/{video_id}.metadata.json"),
        ]
        for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and str(payload.get("id") or "") == video_id:
                return payload
        return None

    def _cached_subtitle(self, video_id: str) -> Path | None:
        candidates = [
            *self.workspace.root.glob(f"jobs/*/{video_id}.en*.json3"),
            *self.workspace.root.glob(
                f"assets/youtube-subtitles/*/{video_id}.en*.json3",
            ),
        ]
        for path in sorted(
            set(candidates), key=lambda item: item.stat().st_mtime, reverse=True,
        ):
            try:
                if parse_youtube_json3(path):
                    return path
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
        return None

    def _executable(self) -> str:
        return "yt-dlp" if self.custom_runner else self.runtime.require_executable()

    def _extractor_args(self, purpose: str = "gvs") -> list[str]:
        return self.runtime.extractor_arguments("subs" if purpose == "subs" else "gvs")

    def _auth_args(self) -> list[str]:
        browser = os.environ.get("VIDEO_FACTORY_YOUTUBE_COOKIES_FROM_BROWSER", "").strip()
        return ["--cookies-from-browser", browser] if browser else []

    def _metadata(self, url: str) -> dict[str, Any]:
        command = [
            self._executable(), "--dump-single-json", "--skip-download", "--ignore-no-formats-error",
            *self._extractor_args(), *self._auth_args(), url,
        ]
        completed = self.runner(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            self._raise_download_error(completed)
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            self._raise_download_error(completed, error)
            raise AssertionError("unreachable")
        if not isinstance(payload, dict):
            raise YouTubeAcquisitionError("YouTube metadata response must be a JSON object")
        return payload

    def _subtitle(self, url: str, video_id: str, job: Path) -> Path:
        output = str(job / "%(id)s")
        command = [
            self._executable(), "--skip-download", "--write-sub", "--write-auto-sub",
            "--sub-langs", "en-orig,en", "--sub-format", "json3", "--no-playlist",
            *self._extractor_args("subs"), *self._auth_args(),
            "-o", output, url,
        ]
        completed = self.runner(command, check=False, capture_output=True, text=True)
        files = sorted(job.glob(f"{video_id}.en*.json3"))
        if completed.returncode != 0 or not files:
            self._raise_download_error(completed)
        return files[0]

    def _download_media(
        self, url: str, video_id: str, job: Path,
        download_window: tuple[float, float] | None = None,
    ) -> Path:
        output = str(job / "%(id)s.%(ext)s")
        command = [
            self._executable(), "--no-playlist",
            "--socket-timeout", "30", "--retries", "2", "--fragment-retries", "2",
            "-f", "bestvideo[height>=1080][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[height>=1080]+bestaudio/best[height>=1080]",
            "--merge-output-format", "mkv",
        ]
        if download_window is not None:
            start, end = download_window
            if start < 0 or end <= start:
                raise ValueError("YouTube download interval must be a positive source range")
            command.extend([
                "--download-sections", f"*{start:.3f}-{end:.3f}",
                # Stream-copy section cuts may begin video on an earlier keyframe
                # while audio retains a positive start PTS.  That offset later
                # becomes a fabricated silent gap when ranges are concatenated.
                # Interview clips are short enough to pay the bounded re-encode
                # cost for frame-accurate, zero-based A/V timelines.
                "--force-keyframes-at-cuts",
                # Google can close one DASH input after a long idle burst.
                # Without reconnect, ffmpeg exits zero with a truncated audio
                # track and yt-dlp cannot resume the sectioned transfer.
                "--downloader-args",
                (
                    "ffmpeg_i:-reconnect 1 -reconnect_streamed 1 "
                    "-reconnect_on_network_error 1 "
                    "-reconnect_on_http_error 4xx,5xx -reconnect_delay_max 5"
                ),
            ])
        command.extend([*self._extractor_args(), *self._auth_args(), "-o", output, url])
        try:
            timeout_seconds = max(
                60.0,
                float(os.environ.get("VIDEO_FACTORY_YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS", "480")),
            )
        except ValueError:
            timeout_seconds = 480.0
        try:
            completed = self.runner(
                command, check=False, capture_output=True, text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            for partial in job.glob(f"{video_id}.*"):
                if partial.suffix in {".mp4", ".mkv", ".webm", ".part"}:
                    partial.unlink(missing_ok=True)
            raise YouTubeAcquisitionError(
                f"YouTube media acquisition timed out after {timeout_seconds:.0f}s"
            ) from error
        files = [item for item in job.glob(f"{video_id}.*") if item.suffix in {".mp4", ".mkv", ".webm"}]
        if completed.returncode != 0 or not files:
            self._raise_download_error(completed)
        return sorted(files)[0]

    def _source_media_info(
        self, probe: Any, media_hash: str, format_id: str, client: str,
    ) -> SourceMediaInfo:
        return SourceMediaInfo(
            width=probe.width, height=probe.height, duration=probe.duration,
            video_codec=probe.video_codec, audio_codec=probe.audio_codec or "",
            format_id=format_id, acquisition_client=client, sha256=media_hash,
            runtime=self.runtime.installation_metadata() if not self.custom_runner else {},
        )

    @staticmethod
    def _require_1080p(width: int, height: int, path: Path) -> None:
        if width < 1920 or height < 1080:
            raise SourceBelow1080Error(
                f"source_below_1080: {path} is {width}x{height}; require at least 1920x1080"
            )

    @staticmethod
    def _require_complete_audio(probe: Any, path: Path) -> None:
        if not probe.audio_codec:
            raise YouTubeAcquisitionError(f"YouTube media has no audio track: {path}")
        if (
            probe.audio_duration is not None
            and probe.audio_duration < probe.duration - 1.0
        ):
            raise YouTubeAcquisitionError(
                "YouTube media audio ended before video: "
                f"{probe.audio_duration:.2f}s / {probe.duration:.2f}s in {path}"
            )

    @staticmethod
    def _raise_download_error(
        completed: subprocess.CompletedProcess[str], cause: Exception | None = None,
    ) -> None:
        detail = (completed.stderr or completed.stdout or "YouTube acquisition failed").strip()
        folded = detail.casefold()
        if "requested format is not available" in folded and not any(
            marker in folded for marker in ("po token", "sabr", "403", "sign in")
        ):
            raise SourceBelow1080Error(
                "source_below_1080: YouTube exposes no downloadable format at or above 1920x1080"
            ) from cause
        if any(marker in folded for marker in ("po token", "sabr", "403", "sign in")):
            raise YouTubeWebAuthRequired(
                "YouTube mweb acquisition requires the managed PO-token provider, an explicit token, or local files; "
                "the factory will not silently fall back to android_vr"
            ) from cause
        raise YouTubeAcquisitionError(detail[-2000:]) from cause


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _visual_width(value: str) -> int:
    return sum(1 if ord(char) < 128 else 2 for char in value)


def normalize_chinese_subtitle(value: str) -> str:
    value = re.sub(r"\s*([，。！？；：、])\s*", r"\1", value.strip())
    value = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", value)
    value = re.sub(r"([\u3400-\u9fff])([A-Za-z0-9])", r"\1 \2", value)
    value = re.sub(r"([A-Za-z0-9])([\u3400-\u9fff])", r"\1 \2", value)
    return re.sub(r"[ \t]{2,}", " ", value)


def wrap_subtitle(value: str, max_chinese_chars: int) -> str:
    value = normalize_chinese_subtitle(value)
    max_width = max_chinese_chars * 2
    if _visual_width(value) <= max_width:
        return value
    split_candidates = [index for index, char in enumerate(value) if char in "，。；：！？、,.;:!? "]
    midpoint = len(value) / 2
    if split_candidates:
        split = min(split_candidates, key=lambda item: abs(item - midpoint)) + 1
    else:
        split = min(len(value), max_chinese_chars)
    return value[:split].strip() + "\n" + value[split:].strip()


def render_source_ranges(item: CollectionItem, render: PlatformRender) -> list[SourceRange]:
    """Keep source speech chronological; the visual headline provides the hook.

    Moving a high-scoring 6–10 second window to the front frequently starts in
    the middle of a sentence and resumes that sentence a minute later.  The
    selected hook remains auditable evidence for the persistent headline, but
    spoken interviews are never remixed out of order.
    """
    return list(item.source_ranges)


def _wrap_english(value: str, max_chars: int) -> str:
    words = re.sub(r"\s+", " ", value).strip().split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _wrap_subtitle_pixels(value: str, font: Any, maximum_width: int) -> str:
    """Reflow complete subtitle copy using the font that will be rendered.

    SRT line breaks are only editorial hints.  The final portrait/landscape
    canvas has different pixel budgets, so trusting a two-line character
    estimate can clip both Latin and CJK text at the right edge.
    """
    normalized = re.sub(r"\s+", " ", value.replace("\n", " ")).strip()
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’._+/%:-]*|\s+|.", normalized)
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = current + token
        if current.strip() and font.getlength(candidate.strip()) > maximum_width:
            lines.append(current.strip())
            current = token.lstrip()
        else:
            current = candidate
    if current.strip():
        lines.append(current.strip())
    closing = "、，。；：！？）》】〕〉］},.;:!?)]"
    for index in range(1, len(lines)):
        if not lines[index] or lines[index][0] not in closing or not lines[index - 1]:
            continue
        previous = lines[index - 1]
        match = re.search(r"([A-Za-z0-9][A-Za-z0-9'’._+/%:-]*|.)$", previous)
        if not match:
            continue
        carry = match.group(1)
        lines[index - 1] = previous[:match.start()].rstrip()
        lines[index] = carry + lines[index]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _fit_subtitle_by_pixels(
    value: str, font_path: Path, maximum_width: int, start_size: int,
    minimum_size: int, maximum_lines: int, stroke_width: int = 0,
) -> tuple[Any, str]:
    """Fit without clipping or deleting words; line count wins over font size."""
    from PIL import ImageFont

    # Pillow's stroke expands the actual bitmap on both sides even when the
    # un-stroked glyph advance fits. Reserve that paint width before wrapping.
    text_width = max(1, maximum_width - stroke_width * 2)
    for size in range(start_size, minimum_size - 1, -2):
        font = ImageFont.truetype(str(font_path), size)
        wrapped = _wrap_subtitle_pixels(value, font, text_width)
        if (
            len(wrapped.splitlines()) <= maximum_lines
            and all(font.getlength(line) <= text_width for line in wrapped.splitlines())
        ):
            return font, wrapped
    # A kinsoku correction can make a line a few pixels wider at the normal
    # minimum. Permit a narrowly bounded emergency reduction rather than
    # clipping or placing closing punctuation on an isolated line.
    for size in range(minimum_size - 2, max(22, minimum_size - 8) - 1, -2):
        font = ImageFont.truetype(str(font_path), size)
        wrapped = _wrap_subtitle_pixels(value, font, text_width)
        if all(font.getlength(line) <= text_width for line in wrapped.splitlines()):
            return font, wrapped
    font = ImageFont.truetype(str(font_path), max(22, minimum_size - 8))
    # Preserve every word even for an abnormally long cue.  The dynamic card
    # height below keeps extra lines visible instead of truncating them.
    return font, _wrap_subtitle_pixels(value, font, text_width)


def interview_caption_layout_errors(cues: list[TranscriptCue]) -> list[str]:
    """Measure the same portrait subtitle panel used by the renderer."""
    maximum_width = 1080 - 200
    english_font_path = resolve_font_path()
    chinese_font_path = _resolve_chinese_subtitle_font_path()
    errors: list[str] = []
    for cue in cues:
        english_font, english = _fit_subtitle_by_pixels(
            cue.source_text, english_font_path, maximum_width, 28, 20,
            INTERVIEW_CAPTION_MAX_RENDERED_LINES, stroke_width=1,
        )
        chinese_font, chinese = _fit_subtitle_by_pixels(
            cue.translation, chinese_font_path, maximum_width, 44, 34,
            INTERVIEW_CAPTION_MAX_RENDERED_LINES, stroke_width=2,
        )
        if (
            len(english.splitlines()) > INTERVIEW_CAPTION_MAX_RENDERED_LINES
            or getattr(english_font, "size", 20) < 20
        ):
            errors.append(f"{cue.id} English does not fit the three-line subtitle panel")
        if (
            len(chinese.splitlines()) > INTERVIEW_CAPTION_MAX_RENDERED_LINES
            or getattr(chinese_font, "size", 34) < 34
        ):
            errors.append(f"{cue.id} Chinese does not fit the three-line subtitle panel")
    return errors






def _semantic_source_part_is_dangling(value: str) -> bool:
    """Reject fixed card cuts that leave a setup without its complement."""
    stripped = value.strip()
    cleaned = re.sub(r"[-,:;]+$", "", stripped).strip()
    if (
        re.match(
            r"^(?:(?:um+|uh+|i\s+mean|you\s+know|and)\W+)*"
            r"(?:if|when)\b",
            cleaned, re.IGNORECASE,
        )
        and not re.search(r"[.!?][\"'”’)]?$", stripped)
    ):
        # A card beginning with a dependent condition is not safe to split
        # again before a sentence boundary. This catches both a comma cut
        # ("If we don't obsolete our products,") and an earlier conjunction
        # cut ("If we don't obsolete our products / and services, ...").
        # Keep the complete conditional together instead of asking the
        # translation model to invent or copy its consequence.
        return True
    if (
        re.search(r"\b(?:uh|um)$", cleaned, re.IGNORECASE)
        and not re.search(r"\bof\s+(?:uh|um)$", cleaned, re.IGNORECASE)
    ):
        return True
    cleaned = re.sub(
        r"(?:\s+(?:and|of))?\s+(?:uh|um)(?:\s+(?:uh|um))*$",
        "", cleaned, flags=re.IGNORECASE,
    ).strip()
    if re.search(r"\bwork\s+of$", cleaned, re.IGNORECASE):
        return False
    if SOURCE_DANGLING_END.search(cleaned):
        return True
    if re.search(
        r"\b(?:upfront|like|allows?|mandates?|I\s+don['’]t\s+think|amount\s+of\b.*\badded)$",
        cleaned, re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:predicts?|forecast(?:s|ed)?|found|divided|figure(?:d)?\s+out)"
        r"(?:\s+(?:that|the|a|an|our|their|his|her|its|\w+)){0,3}$",
        cleaned, re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:that|which|where)\s+(?:you|we|they|he|she|it)$",
        cleaned, re.IGNORECASE,
    ):
        return True
    if re.search(r"\b(?:they're|we're|you're|it's|that's)$", cleaned, re.IGNORECASE):
        return True
    if (
        re.match(r"^(?:you\s+know|i\s+mean)\b", cleaned, re.IGNORECASE)
        and len(re.findall(r"\S+", cleaned)) <= 5
        and not re.search(r"[.!?][\"'”’)]?$", stripped)
    ):
        return True
    # These are explicit complement heads seen in spoken interview clauses;
    # do not reject every noun-ending clause because many are complete.
    return bool(
        re.search(
            r"\bwhat\s+(?:our|their|his|her|its|the)\s+\w+$",
            cleaned, re.IGNORECASE,
        )
    )


def _uppercase_source_entities(value: str) -> set[str]:
    """Return auditable acronym-like entities that must not move across cards."""
    return set(re.findall(
        r"\b([A-Z][A-Z0-9.-]{1,}?)(?:s|es)?\b", value,
    )) - {"AI", "US", "USA"}


def _caption_entity_alignment_errors(
    source: str, translation: str,
    terminology: list[TerminologyEntry] | None = None,
    audio_conflict_entities: set[str] | None = None,
) -> list[str]:
    """Keep named entities in their fixed source card, allowing known Chinese aliases."""
    errors: list[str] = []
    source_entities = _uppercase_source_entities(source)
    target_entities = _uppercase_source_entities(translation)
    for entity in source_entities:
        if entity in (audio_conflict_entities or set()):
            continue
        contextual_targets = tuple(
            entry.target for entry in (terminology or [])
            if entry.strategy == TerminologyStrategy.TRANSLATE
            and entry.target.strip()
            and (
                entry.source.casefold() == entity.casefold()
                or (
                    _contains_term(entry.source, entity)
                    and _contains_term(source, entry.source)
                )
            )
        )
        aliases = (*CAPTION_ENTITY_ALIASES.get(entity, (entity,)), *contextual_targets)
        if not any(alias in translation for alias in aliases):
            errors.append(f"missing:{entity}")
    for entity in target_entities - source_entities:
        errors.append(f"moved:{entity}")
    for entity, aliases in CAPTION_ENTITY_ALIASES.items():
        if entity in source_entities:
            continue
        if any(alias in translation for alias in aliases):
            errors.append(f"moved:{entity}")
    return errors


def _semantic_card_translation_errors(
    row: dict[str, Any], translation: str,
    terminology: list[TerminologyEntry],
    *, require_punctuation: bool = True,
    audio_conflict_entities: set[str] | None = None,
) -> list[str]:
    """Validate one isolated semantic card before it can be locked for rendering."""
    errors: list[str] = []
    source = str(row.get("source") or "")
    duration = max(float(row.get("duration_seconds") or 0.1), 0.1)
    visible = _subtitle_reading_units(translation)
    if not translation:
        errors.append("empty")
    if visible / duration > INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND:
        errors.append("reading_speed")
    if require_punctuation and translation and not re.search(r"[。！？!?]$", translation):
        errors.append("punctuation")
    errors.extend(_caption_entity_alignment_errors(
        source, translation, terminology, audio_conflict_entities,
    ))
    for term in terminology:
        protected_terms = [
            other.source for other in terminology
            if other is not term
            and len(other.source) > len(term.source)
            and term.source.casefold() in other.source.casefold()
        ]
        if (
            term.strategy == TerminologyStrategy.TRANSLATE
            and term.target
            and _contains_unprotected_term(source, term.source, protected_terms)
        ):
            if not _translated_term_present(
                term, TranscriptCue(
                    id=str(row.get("id") or ""), start=0, end=duration,
                    source_text=source, translation=translation,
                ),
                allow_alternatives=True,
            ):
                errors.append(f"term:{term.source}:missing_target:{term.target}")
            if _contains_term(translation, term.source):
                errors.append(f"term:{term.source}:remove_english:{term.source}")
    return errors


def _review_score_out_of_five(value: object) -> float:
    """Accept reviewer scores expressed consistently on either 0–1 or 1–5."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 <= score <= 1.0:
        return score * 5.0
    return score


def _caption_numeric_alignment_errors(
    source_words: list[SourceWord], start_word: int, end_word: int,
    translation: str, numeric_conflict_pairs: list[dict[str, Any]],
) -> list[str]:
    """Enforce Arabic-number ownership while allowing one locally paired ASR value."""
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(\d+(?:[.,]\d+)*)"
        r"(?:st|nd|rd|th)?([%％]|x)?(?![A-Za-z0-9])",
        re.IGNORECASE,
    )

    def values(value: str) -> list[str]:
        return [
            (match.group(1) + (match.group(2) or "")).replace("％", "%")
            for match in pattern.finditer(value)
        ]

    pair_by_index = {
        int(row["source_word_index"]) - 1: row
        for row in numeric_conflict_pairs
        if start_word <= int(row["source_word_index"]) - 1 < end_word
    }
    occurrences = [
        (index, number)
        for index in range(start_word, end_word)
        for number in values(source_words[index].raw)
    ]
    inherited_percent_indices: set[int] = set()
    for occurrence_index, (index, number) in enumerate(occurrences):
        if number.endswith("%"):
            continue
        for later_index, later_number in occurrences[occurrence_index + 1:]:
            if later_index - index > 3:
                break
            if later_number.endswith("%"):
                inherited_percent_indices.add(index)
                break
    target_counts: dict[str, int] = {}
    for number in values(translation):
        target_counts[number] = target_counts.get(number, 0) + 1
    errors: list[str] = []
    for index, number in occurrences:
        if index in pair_by_index:
            continue
        accepted = next((
            candidate for candidate in (
                number,
                *((number + "%",) if index in inherited_percent_indices else ()),
            )
            if target_counts.get(candidate, 0)
        ), "")
        if accepted:
            target_counts[accepted] -= 1
        else:
            errors.append(f"missing_number:{number}@{index + 1}")
    for index, number in occurrences:
        pair = pair_by_index.get(index)
        if pair is None:
            continue
        candidates = [
            str(pair.get("audio_candidate") or "").replace("％", "%"), number,
        ]
        adopted = next((
            candidate for candidate in candidates
            if candidate and target_counts.get(candidate, 0)
        ), "")
        if adopted:
            target_counts[adopted] -= 1
        else:
            errors.append(
                f"missing_number:{number}|{candidates[0]}@{index + 1}"
            )
    # A translated English month is commonly and unambiguously rendered as
    # its numeric Chinese month (September -> 9月). Consume only target values
    # that are attached to 月 and only when that month name occurs in this
    # exact source card, so an unrelated invented number remains an error.
    month_numbers = {
        name: str(index) for index, name in enumerate((
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ), start=1)
    }
    source_text = " ".join(word.raw for word in source_words[start_word:end_word])
    source_months = {
        number for name, number in month_numbers.items()
        if re.search(rf"\b{re.escape(name)}\b", source_text, re.IGNORECASE)
    }
    translated_months = re.findall(r"(?<!\d)(1[0-2]|0?[1-9])\s*月", translation)
    for number in translated_months:
        canonical = str(int(number))
        if canonical in source_months and target_counts.get(canonical, 0):
            target_counts[canonical] -= 1
    for number, count in sorted(target_counts.items()):
        errors.extend([f"moved_number:{number}"] * count)
    return errors


def _numeric_audio_conflict_pairs(
    source_words: list[SourceWord], audio_hypothesis: str,
) -> list[dict[str, Any]]:
    """Pair one changed number on each side only inside the same aligned replace span."""
    hypothesis_words = re.findall(r"\S+", audio_hypothesis)
    source_norm = [word.normalized for word in source_words]
    hypothesis_norm = [
        re.sub(r"[^a-z0-9']+", "", value.casefold().replace("’", "'"))
        for value in hypothesis_words
    ]

    def number(value: str) -> str:
        match = re.search(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)*(?:%|x)?", value)
        return match.group(0) if match else ""

    pairs: list[dict[str, Any]] = []
    matcher = SequenceMatcher(None, source_norm, hypothesis_norm, autojunk=False)
    opcodes = matcher.get_opcodes()
    for opcode_index, (tag, source_start, source_end, audio_start, audio_end) in enumerate(opcodes):
        if tag != "replace":
            continue
        previous_anchor = opcodes[opcode_index - 1] if opcode_index else None
        next_anchor = opcodes[opcode_index + 1] if opcode_index + 1 < len(opcodes) else None
        if (
            previous_anchor is None or previous_anchor[0] != "equal"
            or previous_anchor[2] - previous_anchor[1] < 2
            or next_anchor is None or next_anchor[0] != "equal"
            or next_anchor[2] - next_anchor[1] < 2
        ):
            continue
        source_numbers = [
            (index, number(source_words[index].raw))
            for index in range(source_start, source_end)
            if number(source_words[index].raw)
        ]
        audio_numbers = [
            (index, number(hypothesis_words[index]))
            for index in range(audio_start, audio_end)
            if number(hypothesis_words[index])
        ]
        if (
            len(source_numbers) != 1 or len(audio_numbers) != 1
            or source_numbers[0][1] == audio_numbers[0][1]
        ):
            continue
        source_index, source_value = source_numbers[0]
        audio_index, audio_value = audio_numbers[0]
        pairs.append({
            "source_value": source_value,
            "audio_candidate": audio_value,
            "source_word_index": source_index + 1,
            "audio_word_index": audio_index + 1,
            "source_context": " ".join(
                word.raw for word in source_words[
                    max(0, source_index - 5):min(len(source_words), source_index + 6)
                ]
            ),
            "audio_context": " ".join(
                hypothesis_words[
                    max(0, audio_index - 5):min(len(hypothesis_words), audio_index + 6)
                ]
            ),
            "basis": "single-number aligned replacement",
        })
    return pairs










_ASR_REPEAT_IGNORE = {
    "a", "an", "and", "he's", "i", "i'm", "it's", "or", "she's", "that's",
    "the", "they're", "there's", "uh", "um", "we're", "you", "you're",
}
_KNOWN_CYCLE_MODIFIERS = {
    "business", "competitive", "compute", "development", "economic",
    "hardware", "hype", "inference", "investment", "market", "model",
    "product", "release", "technology", "training",
}
TARGETED_WHISPER_POLICY_VERSION = 6
TARGETED_WHISPER_CONTEXT_SECONDS = 15.0










def interview_caption_duration_errors(
    cues: list[TranscriptCue],
    terminology: list[TerminologyEntry] | None = None,
    audio_conflict_entities: set[str] | None = None,
) -> list[str]:
    """Validate every measurable invariant in the shared caption policy."""
    errors: list[str] = []
    for cue in cues:
        source = re.sub(r"\s+", " ", cue.source_text).strip()
        translation = re.sub(r"\s+", "", cue.translation)
        reading_units = _subtitle_reading_units(cue.translation)
        words = len(re.findall(r"\S+", source))
        if not source or not translation:
            errors.append(f"{cue.id} is missing bilingual caption text")
            continue
        brief_acknowledgement = (
            cue.duration >= INTERVIEW_CAPTION_BRIEF_ACK_MIN_SECONDS - 1e-6
            and words <= 2
            and len(translation) <= 4
        )
        if (
            cue.duration < INTERVIEW_CAPTION_MIN_SECONDS - 1e-6
            and not brief_acknowledgement
        ):
            errors.append(
                f"{cue.id} lasts {cue.duration:.2f}s, below the "
                f"{INTERVIEW_CAPTION_MIN_SECONDS:.2f}s minimum"
            )
        if cue.duration > INTERVIEW_CAPTION_HARD_MAX_SECONDS + 1e-6:
            errors.append(
                f"{cue.id} lasts {cue.duration:.2f}s, above the "
                f"{INTERVIEW_CAPTION_HARD_MAX_SECONDS:.2f}s hard maximum"
            )
        if words > INTERVIEW_CAPTION_MAX_ENGLISH_WORDS:
            errors.append(
                f"{cue.id} has {words} English words, above the "
                f"{INTERVIEW_CAPTION_MAX_ENGLISH_WORDS}-word maximum"
            )
        if (
            reading_units / max(cue.duration, 0.1)
            > INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND
        ):
            errors.append(
                f"{cue.id} exceeds "
                f"{INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND:g} "
                "subtitle reading units per second"
            )
    errors.extend(interview_caption_layout_errors(cues))
    return errors


def cached_interview_caption_pipeline_complete(
    cues: list[TranscriptCue], trace: list[dict[str, Any]],
    media_sha256: str = "",
    terminology: list[TerminologyEntry] | None = None,
) -> bool:
    """Identify an immutable plan reviewed under the exact current policy."""
    substantive_ids = {
        cue.id for cue in cues
        if cue.source_text.strip() or cue.translation.strip()
    }
    if not substantive_ids or interview_caption_duration_errors(cues, terminology):
        return False
    alignments = {
        str(item.get("fingerprint")): item
        for item in trace if isinstance(item, dict)
        and item.get("step") == "interview_source_word_alignment"
        and item.get("policy_version") == ALIGNMENT_POLICY_VERSION
        and item.get("fingerprint")
        and item.get("status") in {"aligned", "audio_alignment_degraded"}
        and item.get("source_ledger_fingerprint")
        and item.get("media_sha256") == media_sha256
    }
    reviewed = any(
        item.get("step") == "interview_joint_boundary_translation"
        and item.get("policy_version") == INTERVIEW_CAPTION_POLICY_VERSION
        and item.get("policy_fingerprint") == INTERVIEW_CAPTION_POLICY_FINGERPRINT
        and item.get("caption_content_fingerprint")
        == _interview_caption_content_fingerprint(cues)
        and item.get("terminology_decision_fingerprint")
        == _terminology_decision_fingerprint(terminology or [])
        and item.get("strict_source_fingerprint")
        == _strict_interview_source_fingerprint(cues)
        and str(item.get("alignment_fingerprint")) in alignments
        and item.get("source_ledger_fingerprint")
        == alignments[str(item.get("alignment_fingerprint"))].get(
            "source_ledger_fingerprint"
        )
        and item.get("audio_hypothesis_sha256", "")
        == alignments[str(item.get("alignment_fingerprint"))].get(
            "audio_hypothesis_sha256", ""
        )
        and substantive_ids <= {
            str(cue_id) for cue_id in item.get("reviewed_cue_ids", [])
        }
        for item in trace if isinstance(item, dict)
    )
    return bool(media_sha256 and alignments and reviewed)


def cached_joint_caption_pipeline_complete(
    cues: list[TranscriptCue], trace: list[dict[str, Any]],
    terminology: list[TerminologyEntry] | None = None,
) -> bool:
    """Validate a non-interview joint-caption cache without audio provenance."""
    substantive_ids = {cue.id for cue in cues if cue.source_text.strip()}
    if not substantive_ids or interview_caption_duration_errors(cues, terminology):
        return False
    return any(
        item.get("step") == "joint_caption_scopes_translation"
        and item.get("policy_version") == INTERVIEW_CAPTION_POLICY_VERSION
        and item.get("policy_fingerprint") == INTERVIEW_CAPTION_POLICY_FINGERPRINT
        and item.get("caption_content_fingerprint")
        == _interview_caption_content_fingerprint(cues)
        and item.get("terminology_decision_fingerprint")
        == _terminology_decision_fingerprint(terminology or [])
        and item.get("strict_source_fingerprint")
        == _strict_interview_source_fingerprint(cues)
        and substantive_ids <= {
            str(cue_id) for cue_id in item.get("reviewed_cue_ids", [])
        }
        for item in trace if isinstance(item, dict)
    )


def _caption_plan_ranges(
    plan: dict[str, Any], duration: float,
) -> list[SourceRange]:
    """Return selected atomic intervals without changing any editorial row."""
    selected: list[SourceRange] = []
    for key in ("bilibili_chapters", "wechat_lessons"):
        rows = plan.get(key)
        if not isinstance(rows, list):
            continue
        selected.extend(
            source_range for raw in rows if isinstance(raw, dict)
            and (source_range := _coerce_range(raw, duration)) is not None
        )
    if not selected:
        return [SourceRange(0.0, duration)] if duration > 0 else []
    endpoints = sorted({
        round(value, 6) for source_range in selected
        for value in (source_range.start, source_range.end)
    })
    atomic: list[SourceRange] = []
    for start, end in zip(endpoints, endpoints[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2
        if any(row.start <= midpoint < row.end for row in selected):
            atomic.append(SourceRange(start, end))
    return atomic


def _source_cues_for_plan(
    cues: list[TranscriptCue], plan: dict[str, Any], duration: float,
) -> list[TranscriptCue]:
    ranges = _caption_plan_ranges(plan, duration)
    return [
        TranscriptCue(**asdict(cue)) for cue in cues
        if any(
            min(cue.end, source_range.end)
            - max(cue.start, source_range.start) >= 0.01
            for source_range in ranges
        )
    ]


def select_caption_incumbent(
    workspace: Workspace, source_video_id: str,
    requested_plan: Path | None, editorial_mode: str,
    editorial_guidance: str | None = None,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Retain a human-accepted caption plan for the same source and clip."""
    if editorial_guidance and editorial_guidance.strip():
        return requested_plan, None
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", source_video_id).strip("-")
    incumbent = (
        workspace.root / "editorial" / "caption-incumbents" / f"{safe_id}.json"
    )
    if not incumbent.is_file():
        return requested_plan, None
    try:
        incumbent_data = json.loads(incumbent.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return requested_plan, None
    if (
        str(incumbent_data.get("source_video_id") or "") != source_video_id
        or str(incumbent_data.get("editorial_mode") or "") != editorial_mode
    ):
        return requested_plan, None
    requested_clip: dict[str, Any] = {}
    requested_data: dict[str, Any] = {}
    if requested_plan is not None and requested_plan.resolve() != incumbent.resolve():
        try:
            requested_data = json.loads(requested_plan.read_text(encoding="utf-8"))
            requested_clip = dict(requested_data.get("source_clip") or {})
        except (OSError, json.JSONDecodeError):
            return requested_plan, None
    incumbent_clip = dict(incumbent_data.get("source_clip") or {})
    if requested_clip and any(
        abs(float(requested_clip.get(key) or 0) - float(incumbent_clip.get(key) or 0)) > 1.0
        for key in ("original_start", "original_end")
    ):
        return requested_plan, None
    requested_trace = [
        row for row in requested_data.get("trace", []) if isinstance(row, dict)
    ]
    inherited_incumbent = any(
        row.get("step") == "caption_incumbent_retained"
        and Path(str(row.get("incumbent_plan") or "")).resolve() == incumbent.resolve()
        for row in requested_trace
    )
    audio_verified_successor = (
        inherited_incumbent
        and any(row.get("step") == "targeted_whisper_caption_audit" for row in requested_trace)
        and any(row.get("step") == "audio_verified_caption_repair" for row in requested_trace)
    )
    if requested_plan is not None and audio_verified_successor:
        return requested_plan, {
            "step": "caption_incumbent_verified_successor_reused",
            "source_video_id": source_video_id,
            "incumbent_plan": str(incumbent),
            "verified_successor_plan": str(requested_plan),
            "reason": "successor changed only audio-flagged cards and passed independent review",
        }
    return incumbent, {
        "step": "caption_incumbent_retained",
        "source_video_id": source_video_id,
        "incumbent_plan": str(incumbent),
        "challenger_plan": str(requested_plan) if requested_plan else "",
        "reason": (
            "human-accepted captions remain immutable for the same source clip; "
            "audio verification may correct evidence but cannot globally rewrite copy"
        ),
    }


def _balanced_subtitle_parts(value: str, count: int, chinese: bool) -> list[str]:
    """Split copy into readable phrase cards without bisecting identifiers."""
    normalized = normalize_chinese_subtitle(value) if chinese else re.sub(
        r"\s+", " ", value,
    ).strip()
    if count <= 1 or not normalized:
        return [normalized]
    if chinese:
        # Chinese has no explicit word boundaries.  Split only on authored
        # clause punctuation; an arbitrary character cut can turn “应用” into
        # “应” / “用” across two cards and is worse than holding a longer card.
        tokens = [
            token.strip() for token in re.findall(
                r"[^，。；：！？]+[，。；：！？]?", normalized,
            ) if token.strip()
        ]
        if len(tokens) <= 1:
            return [normalized]
        count = min(count, len(tokens))
    else:
        tokens = re.findall(r"\S+", normalized)
    if len(tokens) < count:
        return [normalized]

    parts: list[list[str]] = []
    cursor = 0
    remaining_weight = sum(max(1, _visual_width(token)) for token in tokens)
    for part_index in range(count - 1):
        remaining_parts = count - part_index
        target = remaining_weight / remaining_parts
        current: list[str] = []
        current_weight = 0
        maximum_end = len(tokens) - (remaining_parts - 1)
        while cursor < maximum_end:
            token = tokens[cursor]
            current.append(token)
            current_weight += max(1, _visual_width(token))
            cursor += 1
            if current_weight < target:
                continue
            # Keep closing punctuation on the phrase it completes and avoid
            # leaving an opening bracket at the end of a card.
            while cursor < maximum_end and tokens[cursor] in "、，。；：！？）》】〕〉］},.;:!?)]":
                token = tokens[cursor]
                current.append(token)
                current_weight += max(1, _visual_width(token))
                cursor += 1
            if current[-1] in "（《【〔〈［{([" and cursor < maximum_end:
                token = tokens[cursor]
                current.append(token)
                current_weight += max(1, _visual_width(token))
                cursor += 1
            break
        parts.append(current)
        remaining_weight -= current_weight
    parts.append(tokens[cursor:])

    if chinese:
        rendered = [normalize_chinese_subtitle("".join(part)) for part in parts]
    else:
        rendered = [" ".join(part).strip() for part in parts]
    return [part for part in rendered if part]


def split_bilingual_subtitle_display(
    english: str, chinese: str, duration: float, semantic_locked: bool = False,
) -> list[tuple[str, str]]:
    """Return one already-aligned bilingual card without repartitioning it.

    English-only semantic boundaries cannot establish where an existing
    Chinese paragraph carries the corresponding meaning. Segmentation belongs
    in the joint boundary-and-translation engine, where every fixed source span
    gets its own translation and fidelity review.
    """
    if duration <= 0:
        return []
    return [(english, chinese)]


def merge_dependent_subtitle_cues(
    cues: list[TranscriptCue], maximum_duration: float = 30.0,
) -> list[TranscriptCue]:
    """Join adjacent caption fragments until each card carries a complete phrase."""
    merged: list[TranscriptCue] = []
    index = 0
    while index < len(cues):
        source = omit_non_speech_directions(cues[index].source_text)
        translation = cues[index].translation.strip()
        current = TranscriptCue(
            id=cues[index].id, start=cues[index].start, end=cues[index].end,
            source_text=source, translation=translation, speaker=cues[index].speaker,
            confidence=cues[index].confidence,
            original_start=cues[index].original_start,
            original_end=cues[index].original_end,
            source_tokens=list(cues[index].source_tokens),
        )
        index += 1
        while index < len(cues):
            following = cues[index]
            if "-card-" in current.id or "-card-" in following.id:
                break
            following_source = omit_non_speech_directions(following.source_text)
            if not following_source or not following.translation.strip():
                break
            compact = current.source_text.rstrip()
            next_word = re.search(r"[A-Za-z]", following_source)
            dependent = bool(SOURCE_DANGLING_END.search(compact)) or bool(
                compact and compact[-1] not in ".?!。！？"
                and next_word and next_word.group(0).islower()
            )
            if (
                not dependent or following.start - current.end > 0.8
                or (
                    following.end - current.start > maximum_duration
                    and not (
                        SOURCE_DANGLING_END.search(compact)
                        and following.end - current.start <= maximum_duration + 15.0
                    )
                )
            ):
                break
            current.end = following.end
            current.source_text = f"{current.source_text} {following_source}".strip()
            current.translation = normalize_chinese_subtitle(
                f"{current.translation}{following.translation.strip()}"
            )
            current.original_end = following.original_end
            current.source_tokens.extend(following.source_tokens)
            index += 1
        if current.source_text and current.translation:
            merged.append(current)
    return merged


def merge_interview_source_cues(
    cues: list[TranscriptCue], maximum_duration: float = 60.0,
) -> list[TranscriptCue]:
    """Join provisional JSON3 fragments before the single translation pass."""
    merged: list[TranscriptCue] = []
    index = 0
    while index < len(cues):
        first = cues[index]
        current = TranscriptCue(
            id=first.id, start=first.start, end=first.end,
            source_text=omit_non_speech_directions(first.source_text),
            speaker=first.speaker, confidence=first.confidence,
            original_start=first.original_start, original_end=first.original_end,
            source_tokens=[
                row for row in first.source_tokens
                if omit_non_speech_directions(str(row.get("raw") or "")).strip()
            ],
        )
        index += 1
        while index < len(cues):
            following = cues[index]
            following_source = omit_non_speech_directions(following.source_text)
            compact = current.source_text.rstrip()
            next_word = re.search(r"[A-Za-z]", following_source)
            dependent = bool(SOURCE_DANGLING_END.search(compact)) or bool(
                compact and compact[-1] not in ".?!。！？"
                and next_word and next_word.group(0).islower()
            )
            if (
                not following_source
                or not dependent
                or following.start - current.end > 0.8
                or following.end - current.start > maximum_duration
            ):
                break
            current.end = following.end
            current.original_end = following.original_end
            current.source_text = f"{current.source_text} {following_source}".strip()
            current.source_tokens.extend(
                row for row in following.source_tokens
                if omit_non_speech_directions(str(row.get("raw") or "")).strip()
            )
            index += 1
        if current.source_text:
            merged.append(current)
    return merged


def _subtitle_rows(
    manifest: VideoCollectionManifest, ranges: list[SourceRange], profile: RenderProfile,
) -> list[tuple[float, float, str, str]]:
    chinese_limit = 22
    english_limit = 72 if profile == RenderProfile.BILIBILI_LANDSCAPE else 58
    rows: list[tuple[float, float, str, str]] = []
    # Interview plans are merged and semantically segmented during planning.
    # Do not merge again at render time: legacy cached manifests may contain
    # coarse, unsegmented cues, and cross-cue merging makes their bilingual
    # text drift before the density fallback can split it.
    display_cues = (
        list(manifest.transcript)
        if manifest.editorial_mode == "known_tech_interview_clip"
        else merge_dependent_subtitle_cues(manifest.transcript)
    )
    if manifest.editorial_mode == "known_tech_interview_clip":
        if errors := interview_caption_duration_errors(display_cues):
            raise ValueError(
                "interview captions do not satisfy policy "
                f"{INTERVIEW_CAPTION_POLICY_VERSION}; regenerate the cached "
                "translation plan before rendering: " + "; ".join(errors)
            )
    offset = 0.0
    for source_range in ranges:
        for cue in display_cues:
            if cue.end <= source_range.start or cue.start >= source_range.end or not cue.translation.strip():
                continue
            start = offset + max(0.0, cue.start - source_range.start)
            end = offset + min(source_range.duration, cue.end - source_range.start)
            if end > start:
                source_text = omit_non_speech_directions(cue.source_text)
                if not source_text:
                    continue
                display_end = max(end, start + 0.8)
                display_parts = split_bilingual_subtitle_display(
                    source_text, cue.translation, display_end - start,
                    semantic_locked="-card-" in cue.id,
                )
                weights = [
                    max(1, len(re.findall(r"\S+", english_part)))
                    for english_part, _ in display_parts
                ]
                total_weight = sum(weights)
                elapsed = 0.0
                for part_index, (english_part, chinese_part) in enumerate(display_parts):
                    part_start = start + elapsed
                    elapsed += (display_end - start) * weights[part_index] / total_weight
                    part_end = display_end if part_index == len(display_parts) - 1 else start + elapsed
                    rows.append((
                        part_start, part_end,
                        _wrap_english(english_part, english_limit),
                        wrap_subtitle(chinese_part, chinese_limit),
                    ))
        offset += source_range.duration
    if manifest.editorial_mode == "known_tech_interview_clip":
        rendered_cues = [
            TranscriptCue(
                id=f"rendered-row-{index}", start=start, end=end,
                source_text=english, translation=chinese,
            )
            for index, (start, end, english, chinese) in enumerate(rows, start=1)
        ]
        errors = interview_caption_duration_errors(rendered_cues)
        for cue, (_, _, english, chinese) in zip(rendered_cues, rows):
            english_lines = len(english.splitlines())
            chinese_lines = len(chinese.splitlines())
            if english_lines > INTERVIEW_CAPTION_MAX_RENDERED_LINES:
                errors.append(
                    f"{cue.id} wraps to {english_lines} English lines"
                )
            if chinese_lines > INTERVIEW_CAPTION_MAX_RENDERED_LINES:
                errors.append(
                    f"{cue.id} wraps to {chinese_lines} Chinese lines"
                )
        if errors:
            raise ValueError(
                "rendered interview subtitle rows violate caption policy: "
                + "; ".join(errors)
            )
    return rows


def _write_srt(path: Path, rows: list[tuple[float, float, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content: list[str] = []
    for index, (start, end, text) in enumerate(rows, start=1):
        content.extend([str(index), f"{_srt_time(start)} --> {_srt_time(end)}", text, ""])
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def write_item_subtitle_files(
    manifest: VideoCollectionManifest, item: CollectionItem, render: PlatformRender, stem: Path,
) -> tuple[Path, Path, Path]:
    rows = _subtitle_rows(manifest, render_source_ranges(item, render), render.profile)
    source = _write_srt(stem.with_suffix(".en.srt"), [(a, b, en) for a, b, en, _ in rows])
    chinese = _write_srt(stem.with_suffix(".zh-Hans.srt"), [(a, b, zh) for a, b, _, zh in rows])
    combined = _write_srt(
        stem.with_suffix(".bilingual.srt"),
        [(a, b, f"{en}\n{zh}") for a, b, en, zh in rows],
    )
    return source, chinese, combined


def write_item_srt(
    manifest: VideoCollectionManifest, item: CollectionItem, profile: RenderProfile, output: Path,
) -> Path:
    rows = _subtitle_rows(manifest, item.source_ranges, profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    return _write_srt(output, [(a, b, f"{en}\n{zh}") for a, b, en, zh in rows])


def _read_srt_rows(path: Path) -> list[tuple[float, float, str]]:
    def seconds(value: str) -> float:
        hours, minutes, tail = value.split(":")
        secs, millis = tail.split(",")
        return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(millis) / 1000

    rows: list[tuple[float, float, str]] = []
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip())
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3 or " --> " not in lines[1]:
            continue
        start, end = lines[1].split(" --> ", 1)
        rows.append((seconds(start), seconds(end), "\n".join(lines[2:])))
    return rows


def _write_subtitle_overlay_concat(
    source_subtitle: Path, translation_subtitle: Path, profile: RenderProfile,
    duration: float, output: Path,
) -> tuple[Path, int]:
    """Render timed transparent PNG overlays for FFmpeg builds without libass."""
    from PIL import Image, ImageDraw, ImageFont

    width = 1920 if profile == RenderProfile.BILIBILI_LANDSCAPE else 1080
    base_height = 250 if profile == RenderProfile.BILIBILI_LANDSCAPE else 330
    english_size = 24 if profile == RenderProfile.BILIBILI_LANDSCAPE else 28
    chinese_size = 40 if profile == RenderProfile.BILIBILI_LANDSCAPE else 44
    configured_font = resolve_font_path()
    chinese_font_path = _resolve_chinese_subtitle_font_path()
    frame_dir = output.with_suffix(".subtitle-frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    maximum_text_width = width - (180 if profile == RenderProfile.BILIBILI_LANDSCAPE else 200)
    fitted_rows: list[tuple[float, float, str, str, Any, Any, int]] = []
    maximum_height = base_height
    source_rows = _read_srt_rows(source_subtitle)
    translation_rows = _read_srt_rows(translation_subtitle)
    if len(source_rows) != len(translation_rows):
        raise ValueError("English and Chinese subtitle row counts differ")
    for source_row, translation_row in zip(source_rows, translation_rows):
        start, end, english = source_row
        zh_start, zh_end, chinese = translation_row
        if abs(start - zh_start) > 0.02 or abs(end - zh_end) > 0.02:
            raise ValueError("English and Chinese subtitle timing differs")
        english_font, english = _fit_subtitle_by_pixels(
            english, configured_font, maximum_text_width, english_size,
            20 if profile == RenderProfile.WECHAT_VERTICAL else 18, 3,
            stroke_width=1,
        )
        chinese_font, chinese = _fit_subtitle_by_pixels(
            chinese, chinese_font_path, maximum_text_width, chinese_size,
            34 if profile == RenderProfile.WECHAT_VERTICAL else 30, 3,
            stroke_width=2,
        )
        if (
            len(english.splitlines()) > INTERVIEW_CAPTION_MAX_RENDERED_LINES
            or len(chinese.splitlines()) > INTERVIEW_CAPTION_MAX_RENDERED_LINES
        ):
            raise ValueError(
                "subtitle exceeds the hard rendered line-count policy after "
                "font measurement"
            )
        probe = Image.new("RGBA", (width, base_height), (0, 0, 0, 0))
        probe_draw = ImageDraw.Draw(probe)
        english_bbox = probe_draw.multiline_textbbox(
            (0, 0), english, font=english_font, spacing=6, stroke_width=1,
        )
        chinese_bbox = probe_draw.multiline_textbbox(
            (0, 0), chinese, font=chinese_font, spacing=8, stroke_width=2,
        )
        english_width = english_bbox[2] - english_bbox[0]
        chinese_width = chinese_bbox[2] - chinese_bbox[0]
        if english_width > maximum_text_width + 1 or chinese_width > maximum_text_width + 1:
            raise ValueError(
                "subtitle cannot fit inside the mobile safe width after pixel reflow: "
                f"english={english_width}px, chinese={chinese_width}px, "
                f"maximum={maximum_text_width}px"
            )
        required_height = (
            (english_bbox[3] - english_bbox[1])
            + 12 + (chinese_bbox[3] - chinese_bbox[1]) + 60
        )
        maximum_height = max(maximum_height, required_height)
        fitted_rows.append((start, end, english, chinese, english_font, chinese_font, required_height))
    height = maximum_height
    blank = frame_dir / "blank.png"
    Image.new("RGBA", (width, height), (0, 0, 0, 0)).save(blank)

    rendered: dict[tuple[str, str], Path] = {}
    sequence: list[tuple[Path, float]] = []
    cursor = 0.0
    for start, end, english, chinese, english_font, chinese_font, _ in fitted_rows:
        start = max(cursor, start)
        end = min(duration, max(start, end))
        if start - cursor > 0.01:
            sequence.append((blank, start - cursor))
        if end - start <= 0.01:
            continue
        key = (english, chinese)
        frame = rendered.get(key)
        if frame is None:
            frame = frame_dir / f"subtitle-{len(rendered) + 1:04d}.png"
            canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            left_aligned = profile == RenderProfile.WECHAT_VERTICAL
            text_x = 100 if left_aligned else width / 2
            anchor = "la" if left_aligned else "ma"
            align = "left" if left_aligned else "center"
            english_bbox = draw.multiline_textbbox(
                (text_x, 0), english, font=english_font, anchor=anchor,
                align=align, spacing=6, stroke_width=1,
            )
            chinese_bbox = draw.multiline_textbbox(
                (text_x, 0), chinese, font=chinese_font, anchor=anchor,
                align=align, spacing=8, stroke_width=2,
            )
            english_height = english_bbox[3] - english_bbox[1]
            chinese_height = chinese_bbox[3] - chinese_bbox[1]
            gap = 12
            total_height = english_height + gap + chinese_height
            top = max(12, (height - total_height) / 2)
            padding_x, padding_y = 28, 18
            if left_aligned:
                card_left, card_right = 70, width - 70
            else:
                max_text_width = max(
                    english_bbox[2] - english_bbox[0], chinese_bbox[2] - chinese_bbox[0],
                )
                card_left = width / 2 - max_text_width / 2 - padding_x
                card_right = width / 2 + max_text_width / 2 + padding_x
            draw.rounded_rectangle(
                (
                    card_left, top - padding_y, card_right,
                    top + total_height + padding_y,
                ),
                radius=18, fill=(0, 0, 0, 170),
            )
            draw.multiline_text(
                (text_x, top), english, font=english_font, fill=(215, 222, 233, 255),
                anchor=anchor, align=align, spacing=6, stroke_width=1,
                stroke_fill=(0, 0, 0, 235),
            )
            draw.multiline_text(
                (text_x, top + english_height + gap), chinese,
                font=chinese_font, fill=(255, 255, 255, 255),
                anchor=anchor, align=align, spacing=8, stroke_width=2,
                stroke_fill=(0, 0, 0, 235),
            )
            canvas.save(frame)
            rendered[key] = frame
        sequence.append((frame, end - start))
        cursor = end
    if duration - cursor > 0.01:
        sequence.append((blank, duration - cursor))
    if not sequence:
        sequence = [(blank, max(duration, 0.1))]

    concat = output.with_suffix(".subtitles.ffconcat")
    lines = ["ffconcat version 1.0"]
    for frame, segment_duration in sequence:
        escaped = str(frame.resolve()).replace("'", "'\\''")
        lines.extend([f"file '{escaped}'", f"duration {segment_duration:.6f}"])
    escaped_last = str(sequence[-1][0].resolve()).replace("'", "'\\''")
    lines.append(f"file '{escaped_last}'")
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return concat, height


def _slide_translation_rows(
    translations: list[SlideTranslation], ranges: list[SourceRange],
) -> list[tuple[float, float, str, int | None, int | None]]:
    """Map source-timed slide translations onto a possibly reordered edit."""
    rows: list[tuple[float, float, str, int | None, int | None]] = []
    offset = 0.0
    for source_range in ranges:
        for translation in translations:
            overlap_start = max(source_range.start, translation.start)
            overlap_end = min(source_range.end, translation.end)
            if overlap_end <= overlap_start or not translation.translation.strip():
                continue
            rows.append((
                offset + overlap_start - source_range.start,
                offset + overlap_end - source_range.start,
                translation.translation.strip(),
                translation.source_text_bottom,
                translation.source_text_center_x,
            ))
        offset += source_range.duration
    rows.sort(key=lambda row: (row[0], row[1]))
    merged: list[tuple[float, float, str, int | None, int | None]] = []
    for start, end, text, source_text_bottom, source_text_center_x in rows:
        if (
            merged and text == merged[-1][2]
            and source_text_bottom == merged[-1][3]
            and source_text_center_x == merged[-1][4]
            and start - merged[-1][1] <= 0.05
        ):
            merged[-1] = (
                merged[-1][0], max(end, merged[-1][1]), text,
                source_text_bottom, source_text_center_x,
            )
        else:
            merged.append((
                start, end, text, source_text_bottom, source_text_center_x,
            ))
    return merged


def _write_slide_translation_overlay_concat(
    translations: list[SlideTranslation], ranges: list[SourceRange],
    profile: RenderProfile, duration: float, output: Path,
) -> tuple[Path, int]:
    """Render a distinct Chinese layer for text embedded in slide pixels."""
    from PIL import Image, ImageDraw, ImageFont

    is_bilibili = profile == RenderProfile.BILIBILI_LANDSCAPE
    width, height = (1920, 1080) if is_bilibili else (1080, 720)
    font_size = 34
    wrap_limit = 28 if is_bilibili else 20
    chinese_font_path = _resolve_chinese_subtitle_font_path()
    font = ImageFont.truetype(str(chinese_font_path), font_size)
    frame_dir = output.with_suffix(".slide-translation-frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    blank = frame_dir / "blank.png"
    Image.new("RGBA", (width, height), (0, 0, 0, 0)).save(blank)

    rendered: dict[tuple[str, int | None, int | None], Path] = {}
    sequence: list[tuple[Path, float]] = []
    cursor = 0.0
    for (
        start, end, text, source_text_bottom, source_text_center_x,
    ) in _slide_translation_rows(translations, ranges):
        start = max(cursor, start)
        end = min(duration, max(start, end))
        if start - cursor > 0.01:
            sequence.append((blank, start - cursor))
        if end - start <= 0.01:
            continue
        key = (text, source_text_bottom, source_text_center_x)
        frame = rendered.get(key)
        if frame is None:
            frame = frame_dir / f"slide-{len(rendered) + 1:03d}.png"
            canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            wrapped = wrap_subtitle(text, wrap_limit)
            source_center_x = source_text_center_x if source_text_center_x is not None else 1170
            center_x = source_center_x if is_bilibili else source_center_x * (1080 / 1920)
            bbox = draw.multiline_textbbox(
                (center_x, 0), wrapped, font=font, anchor="ma",
                align="center", spacing=7, stroke_width=1,
            )
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            box_width = min(800 if not is_bilibili else 1100, text_width + 48)
            box_height = text_height + 28
            center_x = min(
                width - box_width / 2 - 8,
                max(box_width / 2 + 8, center_x),
            )
            source_bottom = source_text_bottom if source_text_bottom is not None else 620
            if is_bilibili:
                box_top = min(930 - box_height, max(20, source_bottom + 14))
            else:
                # The complete 1920x1080 source is fitted to 1080x608 and
                # vertically padded by 56 pixels inside the 1080x720 stage.
                mapped_bottom = 56 + source_bottom * (608 / 1080)
                box_top = min(700 - box_height, max(12, mapped_bottom + 14))
            box = (
                center_x - box_width / 2, box_top,
                center_x + box_width / 2, box_top + box_height,
            )
            draw.rounded_rectangle(
                box, radius=16, fill=(4, 14, 24, 220),
                outline=(77, 208, 225, 225), width=2,
            )
            draw.multiline_text(
                (center_x, box_top + box_height / 2), wrapped, font=font,
                fill=(255, 255, 255, 255), anchor="mm", align="center",
                spacing=7, stroke_width=1, stroke_fill=(0, 0, 0, 240),
            )
            canvas.save(frame)
            rendered[key] = frame
        sequence.append((frame, end - start))
        cursor = end
    if duration - cursor > 0.01:
        sequence.append((blank, duration - cursor))
    if not sequence:
        sequence = [(blank, max(duration, 0.1))]

    concat = output.with_suffix(".slide-translations.ffconcat")
    lines = ["ffconcat version 1.0"]
    for frame, segment_duration in sequence:
        escaped = str(frame.resolve()).replace("'", "'\\''")
        lines.extend([f"file '{escaped}'", f"duration {segment_duration:.6f}"])
    escaped_last = str(sequence[-1][0].resolve()).replace("'", "'\\''")
    lines.append(f"file '{escaped_last}'")
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return concat, height


def _resolve_headline_font_path() -> Path:
    """Use a CJK display font whose fullwidth punctuation sits on the baseline."""
    configured = os.environ.get("VIDEO_FACTORY_HEADLINE_FONT", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return resolve_font_path()


def _resolve_chinese_subtitle_font_path() -> Path:
    """Use a CJK font with conventionally baseline-aligned fullwidth punctuation."""
    configured = os.environ.get("VIDEO_FACTORY_CHINESE_SUBTITLE_FONT", "").strip()
    if configured and Path(configured).is_file():
        return Path(configured)
    return _resolve_headline_font_path()


def _write_hook_overlay_concat(
    hook: HookSpec, duration: float, output: Path,
    profile: RenderProfile = RenderProfile.WECHAT_VERTICAL,
) -> tuple[Path, int]:
    from PIL import Image, ImageDraw, ImageFont

    is_bilibili = profile == RenderProfile.BILIBILI_LANDSCAPE
    identity_style = bool(hook.speaker_label.strip()) and not is_bilibili
    width, height = (1920, 240) if is_bilibili else (1080, 430 if identity_style else 250)
    headline_font_path = _resolve_headline_font_path()
    chinese_font_path = _resolve_chinese_subtitle_font_path()
    hero_font = ImageFont.truetype(str(headline_font_path), 66 if is_bilibili else (64 if identity_style else 58))
    compact_font = ImageFont.truetype(str(headline_font_path), 32)
    identity_font = ImageFont.truetype(str(chinese_font_path), 30)
    context_font = ImageFont.truetype(str(chinese_font_path), 30)
    frame_dir = output.with_suffix(".hook-frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    hero_frame = frame_dir / "hook.png"
    compact_frame = frame_dir / "hook-compact.png"
    blank_frame = frame_dir / "blank.png"

    hero_canvas = Image.new(
        "RGBA", (width, height),
        (3, 8, 20, 238) if identity_style else (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(hero_canvas)
    headline = wrap_subtitle(hook.headline_zh, 28 if is_bilibili else 18)
    if identity_style:
        hero_font, headline = _fit_interview_hook_headline(
            hook.headline_zh, headline_font_path,
        )
        # Keep identity separate from the editorial statement. The hook and its
        # causal payoff share one left-aligned panel below the speaker line.
        draw.rounded_rectangle(
            (42, 92, width - 42, height - 20), radius=22,
            fill=(5, 14, 29, 232), outline=(91, 112, 139, 180), width=2,
        )
        identity = hook.speaker_label.strip()
        draw.text(
            (78, 42), identity, font=identity_font, fill=(102, 194, 255, 255),
            anchor="la",
        )
        draw.multiline_text(
            (78, 116), headline, font=hero_font, fill=(255, 255, 255, 255),
            anchor="la", align="left", spacing=9, stroke_width=1,
            stroke_fill=(0, 0, 0, 220),
        )
        context_font, context = _fit_text_by_pixels(
            hook.promise, chinese_font_path, width - 156, 30, minimum_size=24,
            maximum_lines=3,
        )
        if context:
            draw.multiline_text(
                (78, 306), context, font=context_font,
                fill=(205, 217, 232, 255), anchor="la", align="left", spacing=7,
            )
    else:
        headline_y = height / 2
        bbox = draw.multiline_textbbox(
            (width / 2, headline_y), headline, font=hero_font, anchor="mm",
            align="center", spacing=10, stroke_width=2,
        )
        draw.rounded_rectangle(
            (bbox[0] - 34, bbox[1] - 22, bbox[2] + 34, bbox[3] + 22),
            radius=22, fill=(3, 8, 20, 218), outline=(255, 224, 99, 220), width=3,
        )
        draw.multiline_text(
            (width / 2, headline_y), headline, font=hero_font, fill=(255, 255, 255, 255),
            anchor="mm", align="center", spacing=10, stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
    hero_canvas.save(hero_frame)

    Image.new("RGBA", (width, height), (0, 0, 0, 0)).save(blank_frame)

    compact_canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    compact_draw = ImageDraw.Draw(compact_canvas)
    compact_headline = wrap_subtitle(hook.headline_zh, 26)
    compact_bbox = compact_draw.multiline_textbbox(
        (width / 2, 48), compact_headline, font=compact_font, anchor="ma",
        align="center", spacing=5, stroke_width=1,
    )
    compact_draw.rounded_rectangle(
        (
            compact_bbox[0] - 24, compact_bbox[1] - 12,
            compact_bbox[2] + 24, compact_bbox[3] + 12,
        ),
        radius=16, fill=(3, 8, 20, 190), outline=(255, 224, 99, 145), width=2,
    )
    compact_draw.multiline_text(
        (width / 2, 48), compact_headline, font=compact_font,
        fill=(245, 247, 250, 255), anchor="ma", align="center", spacing=5,
        stroke_width=1, stroke_fill=(0, 0, 0, 235),
    )
    compact_canvas.save(compact_frame)

    concat = output.with_suffix(".hook.ffconcat")
    hero_duration = min(
        duration,
        hook.source_range.duration if is_bilibili else (duration if identity_style else 7.0),
    )
    rows = ["ffconcat version 1.0"]
    if is_bilibili and duration > hero_duration:
        rows.extend([
            f"file '{hero_frame.resolve()}'", f"duration {hero_duration:.6f}",
            f"file '{blank_frame.resolve()}'", f"duration {duration - hero_duration:.6f}",
            f"file '{blank_frame.resolve()}'",
        ])
    elif hook.persistent_title and duration > hero_duration:
        rows.extend([
            f"file '{hero_frame.resolve()}'", f"duration {hero_duration:.6f}",
            f"file '{compact_frame.resolve()}'", f"duration {duration - hero_duration:.6f}",
            f"file '{compact_frame.resolve()}'",
        ])
    else:
        rows.extend([
            f"file '{hero_frame.resolve()}'", f"duration {hero_duration:.6f}",
            f"file '{hero_frame.resolve()}'",
        ])
    concat.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return concat, height


class YouTubeCollectionRenderer:
    def __init__(self, workspace: Workspace, runner: Callable[..., subprocess.CompletedProcess[str]] | None = None) -> None:
        self.workspace = workspace
        self.runner = runner or subprocess.run

    def render(self, manifest: VideoCollectionManifest) -> VideoCollectionManifest:
        # Legacy non-interview manifests may need deterministic migration.
        # Reviewed v6 interview cards are immutable at render time; changing
        # their text here would invalidate the reviewer/content fingerprint.
        if manifest.editorial_mode != "known_tech_interview_clip":
            omit_spoken_fillers(manifest.transcript)
            enforce_cached_terminology_contract(
                manifest.transcript, manifest.terminology,
            )
        if terminology_errors := terminology_contract_errors(
            manifest.transcript, manifest.terminology,
        ):
            raise ValueError(
                "cached collection failed terminology preflight before render: "
                + "; ".join(terminology_errors)
            )
        if not manifest.source_media_path:
            raise FileNotFoundError("collection has no archived source media")
        source = self.workspace.root / manifest.source_media_path
        if not source.is_file():
            raise FileNotFoundError(source)
        root = self.workspace.renders_dir / manifest.id
        root.mkdir(parents=True, exist_ok=True)
        for item in manifest.items:
            for render in item.renders:
                stem = f"{item.order:02d}-{item.kind.value}-{render.profile.value}"
                source_subtitle, translation_subtitle, bilingual_subtitle = write_item_subtitle_files(
                    manifest, item, render, root / stem,
                )
                output = root / f"{stem}.mp4"
                self._render_one(source, item, render, source_subtitle, translation_subtitle, output)
                render.video_path = str(output.relative_to(self.workspace.root))
                render.source_subtitle_path = str(source_subtitle.relative_to(self.workspace.root))
                render.translation_subtitle_path = str(translation_subtitle.relative_to(self.workspace.root))
                render.bilingual_subtitle_path = str(bilingual_subtitle.relative_to(self.workspace.root))
                render.subtitle_path = render.bilingual_subtitle_path
        return manifest

    def repair_silent_audio(self, manifest: VideoCollectionManifest) -> list[str]:
        """Re-render only outputs whose decoded audio is effectively silent."""
        if not manifest.source_media_path:
            raise FileNotFoundError("collection has no archived source media")
        source = self.workspace.root / manifest.source_media_path
        if not source.is_file():
            raise FileNotFoundError(source)
        repaired: list[str] = []
        silence_limit = (
            INTERVIEW_MAX_INTERNAL_SILENCE_SECONDS
            if manifest.editorial_mode == "known_tech_interview_clip" else 30.0
        )
        for item in manifest.items:
            for render in item.renders:
                if not render.video_path:
                    continue
                output = self.workspace.root / render.video_path
                try:
                    loudness = probe_audio_loudness(
                        output, minimum_silence_seconds=min(10.0, silence_limit),
                    )
                    probe = probe_video(output)
                    full_length = bool(
                        probe.audio_duration is not None
                        and probe.audio_duration >= item.duration - 0.25
                    )
                    if (
                        full_length and loudness.max_db > -50 and loudness.mean_db > -60
                        and loudness.longest_silence_seconds <= silence_limit
                    ):
                        continue
                except (OSError, ValueError):
                    pass
                source_subtitle = self.workspace.root / render.source_subtitle_path
                translation_subtitle = self.workspace.root / render.translation_subtitle_path
                if not source_subtitle.is_file() or not translation_subtitle.is_file():
                    raise FileNotFoundError(f"subtitle files missing for {render.video_path}")
                self._render_one(
                    source, item, render, source_subtitle, translation_subtitle, output,
                )
                verified = probe_audio_loudness(
                    output, minimum_silence_seconds=min(10.0, silence_limit),
                )
                verified_probe = probe_video(output)
                if (
                    verified_probe.audio_duration is None
                    or verified_probe.audio_duration < item.duration - 0.25
                    or verified.max_db <= -50 or verified.mean_db <= -60
                    or verified.longest_silence_seconds > silence_limit
                ):
                    raise RuntimeError(f"audio remains silent after repair: {output}")
                repaired.append(render.video_path)
        return repaired

    def _render_one(
        self, source: Path, item: CollectionItem, render: PlatformRender,
        source_subtitle: Path, translation_subtitle: Path, output: Path,
    ) -> None:
        profile = render.profile
        source_ranges = render_source_ranges(item, render)
        render_duration = sum(row.duration for row in source_ranges)
        subtitle_concat, subtitle_height = _write_subtitle_overlay_concat(
            source_subtitle, translation_subtitle, profile, render_duration, output,
        )
        hook_concat: Path | None = None
        if render.selected_hook:
            hook_concat, _ = _write_hook_overlay_concat(
                render.selected_hook, render_duration, output, profile,
            )
        slide_translation_concat: Path | None = None
        if render.slide_translations:
            slide_translation_concat, _ = _write_slide_translation_overlay_concat(
                render.slide_translations, source_ranges, profile, render_duration, output,
            )
        parts: list[str] = []
        concat_inputs: list[str] = []
        for index, source_range in enumerate(source_ranges):
            video_filters = (
                f"trim=start={source_range.start:.3f}:end={source_range.end:.3f},"
                "setpts=PTS-STARTPTS,scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,fps=25"
            )
            if (
                index == 0 and hook_concat is not None
                and source_range.framing == FramingMode.SPEAKER
            ):
                video_filters += (
                    ",zoompan=z='min(zoom+0.0005,1.06)':"
                    "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=25"
                )
            if profile == RenderProfile.WECHAT_VERTICAL:
                parts.append(f"[0:v]{video_filters}[src{index}]")
                parts.append(f"[src{index}]split=2[bg{index}][fg{index}]")
                parts.append(
                    f"[bg{index}]scale=270:480:force_original_aspect_ratio=increase,"
                    f"crop=270:480,boxblur=10:5,scale=1080:1920[blur{index}]"
                )
                if source_range.framing == FramingMode.SPLIT:
                    # A split source already contains meaningful panes on both
                    # sides. Fit the complete composite so neither pane is cut or
                    # covered, even when an editor also saved a slide crop hint.
                    foreground = (
                        "scale=1080:720:force_original_aspect_ratio=decrease,"
                        "pad=1080:720:(ow-iw)/2:(oh-ih)/2:black"
                    )
                elif source_range.has_explicit_crop:
                    foreground = (
                        f"crop={source_range.crop_width}:{source_range.crop_height}:"
                        f"{source_range.crop_x}:{source_range.crop_y},"
                        "scale=1080:720:force_original_aspect_ratio=decrease,"
                        "pad=1080:720:(ow-iw)/2:(oh-ih)/2:black"
                    )
                elif source_range.framing == FramingMode.SLIDE:
                    foreground = (
                        "scale=1080:720:force_original_aspect_ratio=decrease,"
                        "pad=1080:720:(ow-iw)/2:(oh-ih)/2:black"
                    )
                else:
                    foreground = "scale=1280:-2,crop=1080:720"
                parts.append(f"[fg{index}]{foreground}[fit{index}]")
                evidence_y = 500 if render.selected_hook and render.selected_hook.speaker_label.strip() else 380
                parts.append(
                    f"[blur{index}][fit{index}]overlay=(W-w)/2:{evidence_y}[v{index}]"
                )
            else:
                parts.append(f"[0:v]{video_filters}[v{index}]")
            parts.append(
                f"[1:a]atrim=start={source_range.start:.3f}:end={source_range.end:.3f},"
                f"asetpts=PTS-{source_range.start:.3f}/TB,"
                "aresample=async=1:first_pts=0,"
                f"apad=whole_dur={source_range.duration:.3f},"
                f"atrim=duration={source_range.duration:.3f}[a{index}]"
            )
            concat_inputs.append(f"[v{index}][a{index}]")
        parts.append("".join(concat_inputs) + f"concat=n={len(source_ranges)}:v=1:a=1[cv][outa]")
        if profile == RenderProfile.BILIBILI_LANDSCAPE:
            parts.append(
                f"[cv]scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black[base]"
            )
            parts.append("[2:v]format=rgba,setpts=PTS-STARTPTS[subs]")
            parts.append(f"[base][subs]overlay=(W-w)/2:H-{subtitle_height}-67:eof_action=pass[captioned]")
        else:
            parts.extend([
                # Each source segment is already composed into the portrait canvas so
                # slide crops and speaker framing can change at edit boundaries.
                "[cv]null[base]",
                "[2:v]format=rgba,setpts=PTS-STARTPTS[subs]",
                f"[base][subs]overlay=(W-w)/2:H-{subtitle_height}-340:eof_action=pass[captioned]",
            ])
        current_label = "captioned"
        next_input = 3
        hook_input = next_input if hook_concat is not None else None
        if hook_concat is not None:
            next_input += 1
        slide_input = next_input if slide_translation_concat is not None else None
        if slide_input is not None:
            slide_x = 0
            slide_y = 0 if profile == RenderProfile.BILIBILI_LANDSCAPE else 380
            parts.extend([
                f"[{slide_input}:v]format=rgba,setpts=PTS-STARTPTS[slidezh]",
                f"[{current_label}][slidezh]overlay={slide_x}:{slide_y}:"
                "eof_action=pass[withslide]",
            ])
            current_label = "withslide"
        if hook_input is not None:
            hook_y = 58 if profile == RenderProfile.BILIBILI_LANDSCAPE else 60
            parts.extend([
                f"[{hook_input}:v]format=rgba,setpts=PTS-STARTPTS[hook]",
                f"[{current_label}][hook]overlay=(W-w)/2:{hook_y}:"
                "eof_action=pass[outv]",
            ])
        else:
            parts.append(f"[{current_label}]null[outv]")
        filter_graph = ";\n".join(parts)
        script = output.with_suffix(".filters.txt")
        script.write_text(filter_graph, encoding="utf-8")
        preset = os.environ.get("VIDEO_FACTORY_FFMPEG_PRESET", "medium").strip()
        if preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"}:
            raise ValueError(f"unsupported VIDEO_FACTORY_FFMPEG_PRESET: {preset}")
        command = [
            "ffmpeg", "-y", "-i", str(source),
            # A dedicated audio input avoids FFmpeg emitting silent AAC when
            # a cold open seeks forward and the body then seeks backward in
            # the same source while video overlays are being scheduled.
            "-i", str(source),
            "-f", "concat", "-safe", "0", "-i", str(subtitle_concat),
        ]
        if hook_concat is not None:
            command.extend(["-f", "concat", "-safe", "0", "-i", str(hook_concat)])
        if slide_translation_concat is not None:
            command.extend([
                "-f", "concat", "-safe", "0", "-i", str(slide_translation_concat),
            ])
        command.extend([
            "-filter_complex", filter_graph,
            "-map", "[outv]", "-map", "[outa]", "-c:v", "libx264", "-preset", preset,
            "-crf", "20", "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
            "-r", "25", "-movflags", "+faststart", str(output),
        ])
        completed = self.runner(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout).strip()[-4000:])


def validate_collection(
    manifest: VideoCollectionManifest, workspace: Path | None = None,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    media_info = manifest.source_media_info
    source_1080 = bool(media_info and media_info.width >= 1920 and media_info.height >= 1080)
    checks.append(CheckResult(
        "source_media_1080p", source_1080,
        f"{media_info.width}x{media_info.height}" if media_info else "source media probe required",
    ))
    checks.append(CheckResult(
        "source_acquisition_client",
        bool(media_info and media_info.acquisition_client in {"mweb", "local"}),
        media_info.acquisition_client if media_info else "missing",
    ))
    study_mode = manifest.editorial_mode == "study" and any(item.kind in {
        CollectionItemKind.BILIBILI_CHAPTER, CollectionItemKind.WECHAT_SHORT,
    } for item in manifest.items)
    youtube_wechat_mode = manifest.editorial_mode in {
        "technical_coverage", "known_tech_interview_clip", "conference_highlights",
    }
    if study_mode:
        chapters = [item for item in manifest.items if item.kind == CollectionItemKind.BILIBILI_CHAPTER]
        shorts = [item for item in manifest.items if item.kind == CollectionItemKind.WECHAT_SHORT]
        minimum_shorts = 3 if manifest.source_duration <= 2700 else 4
        checks.append(CheckResult(
            "bilibili_chapter_count", 1 <= len(chapters) <= 8,
            f"Bilibili study chapters: {len(chapters)}",
        ))
        checks.append(CheckResult(
            "wechat_short_count", minimum_shorts <= len(shorts) <= 8,
            f"WeChat lessons: {len(shorts)}; target {minimum_shorts}–8",
        ))
        chapter_ranges = sorted(
            [source_range for item in chapters for source_range in item.source_ranges],
            key=lambda item: item.start,
        )
        covered = 0.0
        cursor = 0.0
        for source_range in chapter_ranges:
            covered += max(0.0, source_range.end - max(cursor, source_range.start))
            cursor = max(cursor, source_range.end)
        coverage_ratio = covered / manifest.source_duration if manifest.source_duration else 0.0
        checks.append(CheckResult(
            "bilibili_story_coverage", coverage_ratio >= 0.85,
            f"Bilibili chapters cover {coverage_ratio:.1%} of the source; "
            "the plan contract requires 95% of the substantive story",
        ))
        if manifest.source_duration <= 2700:
            short_ranges = sorted(
                [source_range for item in shorts for source_range in item.source_ranges],
                key=lambda item: item.start,
            )
            short_covered = 0.0
            short_cursor = 0.0
            for source_range in short_ranges:
                short_covered += max(0.0, source_range.end - max(short_cursor, source_range.start))
                short_cursor = max(short_cursor, source_range.end)
            short_ratio = short_covered / manifest.source_duration if manifest.source_duration else 0.0
            checks.append(CheckResult(
                "wechat_story_coverage", short_ratio >= 0.8,
                f"short-source WeChat lessons cover {short_ratio:.1%} of the source; "
                "the plan contract requires 90% of the substantive story",
            ))
    elif manifest.editorial_mode == "technical_coverage":
        chapters = [item for item in manifest.items if item.kind == CollectionItemKind.BILIBILI_CHAPTER]
        shorts = [item for item in manifest.items if item.kind == CollectionItemKind.WECHAT_SHORT]
        minimum_shorts = max(1, math.ceil(manifest.source_duration / 360.0))
        maximum_shorts = min(24, max(minimum_shorts, math.floor(manifest.source_duration / 180.0)))
        checks.append(CheckResult(
            "bilibili_paused", not chapters,
            "Bilibili routing paused" if not chapters else f"unexpected Bilibili items: {len(chapters)}",
        ))
        checks.append(CheckResult(
            "wechat_short_count", minimum_shorts <= len(shorts) <= maximum_shorts,
            f"WeChat technical lessons: {len(shorts)}; target {minimum_shorts}–{maximum_shorts}",
        ))
        short_ranges = sorted(
            [source_range for item in shorts for source_range in item.source_ranges],
            key=lambda item: item.start,
        )
        coverage_errors = _coverage_contract_errors(
            short_ranges, 0.0, manifest.source_duration, 0.9, 60.0,
            "WeChat lessons",
        )
        checks.append(CheckResult(
            "wechat_story_coverage", not coverage_errors,
            "chronological shorts cover the complete technical story"
            if not coverage_errors else "; ".join(coverage_errors),
        ))
    elif manifest.editorial_mode == "known_tech_interview_clip":
        chapters = [item for item in manifest.items if item.kind == CollectionItemKind.BILIBILI_CHAPTER]
        shorts = [item for item in manifest.items if item.kind == CollectionItemKind.WECHAT_SHORT]
        checks.append(CheckResult(
            "bilibili_paused", not chapters,
            "Bilibili routing paused" if not chapters else f"unexpected Bilibili items: {len(chapters)}",
        ))
        checks.append(CheckResult(
            "interview_highlight_count", len(shorts) == 1,
            f"known-tech interview highlights: {len(shorts)}; target exactly 1",
        ))
        for short in shorts:
            selected_text = " ".join(
                cue.source_text for cue in manifest.transcript
                if any(cue.end > row.start and cue.start < row.end for row in short.source_ranges)
            )
            entailment_errors = _interview_title_entailment_errors(short.title, selected_text)
            checks.append(CheckResult(
                f"item:{short.id}:title_source_entailment", not entailment_errors,
                "interview title is entailed by the selected passage"
                if not entailment_errors else "; ".join(entailment_errors),
            ))
    elif manifest.editorial_mode == "conference_highlights":
        chapters = [item for item in manifest.items if item.kind == CollectionItemKind.BILIBILI_CHAPTER]
        shorts = [item for item in manifest.items if item.kind == CollectionItemKind.WECHAT_SHORT]
        ranges = sorted(
            [source_range for item in shorts for source_range in item.source_ranges],
            key=lambda value: value.start,
        )
        overlaps = any(
            current.start < previous.end
            for previous, current in zip(ranges, ranges[1:])
        )
        total = sum(value.duration for value in ranges)
        checks.extend([
            CheckResult("bilibili_paused", not chapters, f"unexpected Bilibili items: {len(chapters)}"),
            CheckResult("conference_highlight_count", 1 <= len(shorts) <= 5, f"conference highlights: {len(shorts)}; target 1–5"),
            CheckResult("conference_highlight_distinct_ranges", not overlaps, "source ranges are non-overlapping" if not overlaps else "source ranges overlap"),
            CheckResult("conference_highlight_total_duration", total <= CONFERENCE_HIGHLIGHT_MAX_TOTAL_SECONDS, f"total duration {total:.2f}s; maximum 900s"),
        ])
    else:
        mains = [item for item in manifest.items if item.kind == CollectionItemKind.MAIN]
        episodes = [item for item in manifest.items if item.kind == CollectionItemKind.EPISODE]
        checks.append(CheckResult("main_count", len(mains) == 1, f"main items: {len(mains)}"))
        checks.append(CheckResult("episode_count", 3 <= len(episodes) <= 5, f"episode items: {len(episodes)}"))
    orders = [item.order for item in manifest.items]
    checks.append(CheckResult("collection_order", orders == list(range(1, len(orders) + 1)), f"orders: {orders}"))
    for item in manifest.items:
        in_bounds = all(0 <= source.start < source.end <= manifest.source_duration + 0.5 for source in item.source_ranges)
        checks.append(CheckResult(f"item:{item.id}:ranges", in_bounds, f"duration {item.duration:.2f}s"))
        if item.kind == CollectionItemKind.BILIBILI_CHAPTER:
            duration_ok = 480 <= item.duration <= 1800
            target = "480–1800"
        elif item.kind == CollectionItemKind.WECHAT_SHORT:
            if manifest.editorial_mode == "known_tech_interview_clip":
                duration_ok = 0 < item.duration <= INTERVIEW_MAX_SECONDS
                target = "prefer <=180; hard maximum 300"
            elif manifest.editorial_mode == "conference_highlights":
                duration_ok = (
                    CONFERENCE_HIGHLIGHT_MIN_SECONDS
                    <= item.duration
                    <= CONFERENCE_HIGHLIGHT_MAX_SECONDS
                )
                target = "45–300"
            else:
                duration_ok = 180 <= item.duration <= 360
                target = "180–360"
        elif item.kind == CollectionItemKind.MAIN:
            duration_ok = 900 <= item.duration <= 1320
            target = "900–1320"
        else:
            duration_ok = 270 <= item.duration <= 330
            target = "270–330"
        checks.append(CheckResult(
            f"item:{item.id}:duration", duration_ok,
            f"duration {item.duration:.2f}s; target {target}s",
        ))
        if item.kind in {CollectionItemKind.MAIN, CollectionItemKind.BILIBILI_CHAPTER}:
            expected_profiles = {RenderProfile.BILIBILI_LANDSCAPE}
        elif item.kind == CollectionItemKind.WECHAT_SHORT:
            expected_profiles = {RenderProfile.WECHAT_VERTICAL}
        else:
            expected_profiles = {RenderProfile.BILIBILI_LANDSCAPE, RenderProfile.WECHAT_VERTICAL}
        actual_profiles = {render.profile for render in item.renders}
        checks.append(CheckResult(
            f"item:{item.id}:profiles", actual_profiles == expected_profiles,
            "render profiles: " + ", ".join(sorted(item.value for item in actual_profiles)),
        ))
        if youtube_wechat_mode and item.kind == CollectionItemKind.WECHAT_SHORT:
            cue_text = " ".join(
                cue.source_text for cue in manifest.transcript
                if any(cue.end > source.start and cue.start < source.end for source in item.source_ranges)
            )
            visible_copy = [item.title, item.thesis]
            for render in item.renders:
                visible_copy.extend(hook.headline_zh for hook in render.hook_candidates)
                visible_copy.extend(hook.speaker_label for hook in render.hook_candidates)
                if render.selected_hook is not None:
                    visible_copy.extend([
                        render.selected_hook.headline_zh, render.selected_hook.promise,
                    ])
            political = political_markers(" ".join([cue_text, *visible_copy]))
            checks.append(CheckResult(
                f"item:{item.id}:non_political",
                not political,
                "selected clip and visible copy are non-political"
                if not political else "forbidden political signals: " + ", ".join(political[:8]),
            ))
        for render in item.renders:
            checks.append(CheckResult(
                f"render:{item.id}:{render.profile}:subtitle_mode",
                render.subtitle_mode == SubtitleMode.BILINGUAL_STACKED,
                str(render.subtitle_mode),
            ))
            visual_text_expected = render.slide_translation_required
            slide_rows = sorted(
                render.slide_translations, key=lambda row: (row.start, row.end),
            )
            slide_rows_valid = all(
                0 <= row.start < row.end <= manifest.source_duration + 0.5
                and bool(row.source_text.strip()) and bool(row.translation.strip())
                and (row.source_text_bottom is None or 0 <= row.source_text_bottom <= 1080)
                and (row.source_text_center_x is None or 0 <= row.source_text_center_x <= 1920)
                for row in slide_rows
            ) and all(
                slide_rows[index].start >= slide_rows[index - 1].end - 0.05
                for index in range(1, len(slide_rows))
            )
            checks.append(CheckResult(
                f"render:{item.id}:{render.profile}:slide_translation_layer",
                slide_rows_valid and (not visual_text_expected or bool(slide_rows)),
                f"{len(slide_rows)} source-timed slide translations"
                if slide_rows_valid and (slide_rows or not visual_text_expected)
                else "slide/split framing requires non-overlapping source-timed translations",
            ))
            needs_hook = render.profile == RenderProfile.WECHAT_VERTICAL or (
                study_mode and item.kind == CollectionItemKind.BILIBILI_CHAPTER
                and render.profile == RenderProfile.BILIBILI_LANDSCAPE
            )
            if needs_hook:
                hook_errors = (
                    ["selected hook is required"] if render.selected_hook is None
                    else hook_contract_errors(
                        render.selected_hook, item, manifest.transcript, render.profile,
                    )
                )
                checks.append(CheckResult(
                    f"render:{item.id}:{render.profile}:hook",
                    len(render.hook_candidates) == 3 and not hook_errors,
                    "three evidence-backed candidates; selected hook valid" if not hook_errors
                    and len(render.hook_candidates) == 3 else "; ".join(hook_errors or [
                        f"expected 3 candidates; got {len(render.hook_candidates)}",
                    ]),
                ))
            if not render.video_path or workspace is None:
                continue
            path = workspace / render.video_path
            exists = path.is_file()
            checks.append(CheckResult(f"render:{item.id}:{render.profile}:file", exists, str(path)))
            if exists:
                probe = probe_video(path)
                dimensions = (1920, 1080) if render.profile == RenderProfile.BILIBILI_LANDSCAPE else (1080, 1920)
                checks.extend([
                    CheckResult(f"render:{item.id}:{render.profile}:resolution", (probe.width, probe.height) == dimensions, f"{probe.width}x{probe.height}"),
                    CheckResult(f"render:{item.id}:{render.profile}:h264", probe.video_codec == "h264", probe.video_codec),
                    CheckResult(f"render:{item.id}:{render.profile}:aac", probe.audio_codec == "aac", probe.audio_codec or "missing"),
                    CheckResult(
                        f"render:{item.id}:{render.profile}:audio_duration",
                        bool(
                            probe.audio_duration is not None
                            and probe.audio_duration >= item.duration - 0.25
                        ),
                        f"{probe.audio_duration:.2f}s / target {item.duration:.2f}s"
                        if probe.audio_duration is not None else "missing audio duration",
                    ),
                    CheckResult(f"render:{item.id}:{render.profile}:pixel", probe.pixel_format == "yuv420p", probe.pixel_format),
                    CheckResult(
                        f"render:{item.id}:{render.profile}:duration",
                        abs(probe.duration - item.duration) <= 0.25,
                        f"{probe.duration:.2f}s / target {item.duration:.2f}s",
                    ),
                ])
                try:
                    silence_limit = (
                        INTERVIEW_MAX_INTERNAL_SILENCE_SECONDS
                        if manifest.editorial_mode == "known_tech_interview_clip" else 30.0
                    )
                    loudness = probe_audio_loudness(
                        path, minimum_silence_seconds=min(10.0, silence_limit),
                    )
                    audible = (
                        loudness.max_db > -50 and loudness.mean_db > -60
                        and loudness.longest_silence_seconds <= silence_limit
                    )
                    loudness_detail = (
                        f"mean {loudness.mean_db:.1f} dB; max {loudness.max_db:.1f} dB; "
                        f"longest silence {loudness.longest_silence_seconds:.1f}s"
                    )
                except (OSError, ValueError) as error:
                    audible = False
                    loudness_detail = f"loudness probe failed: {error}"
                checks.append(CheckResult(
                    f"render:{item.id}:{render.profile}:audible_audio",
                    audible, loudness_detail,
                ))
            subtitle_paths = (
                render.source_subtitle_path, render.translation_subtitle_path,
                render.bilingual_subtitle_path,
            )
            subtitle_files = all(value and (workspace / value).is_file() for value in subtitle_paths)
            checks.append(CheckResult(
                f"render:{item.id}:{render.profile}:bilingual_subtitle_files",
                subtitle_files, "English, Chinese, and bilingual SRT files present"
                if subtitle_files else "one or more bilingual subtitle files are missing",
            ))
    untranslated = [
        item.id for item in manifest.transcript
        if not item.translation.strip() and not source_is_omittable_caption_only(item.source_text)
    ]
    translated_stage_directions = [
        item.id for item in manifest.transcript
        if NON_SPEECH_DIRECTION.search(item.translation)
    ]
    missing_source = [item.id for item in manifest.transcript if not item.source_text.strip()]
    checks.append(CheckResult(
        "source_subtitles_preserved", not missing_source,
        "all original English caption cues preserved" if not missing_source
        else f"missing English: {', '.join(missing_source[:10])}",
    ))
    checks.append(CheckResult(
        "translation_complete", not untranslated,
        "all substantive transcript cues translated; filler-only cues omitted"
        if not untranslated else f"missing: {', '.join(untranslated[:10])}",
    ))
    checks.append(CheckResult(
        "non_speech_directions_omitted", not translated_stage_directions,
        "non-speech caption metadata is omitted from Chinese subtitles"
        if not translated_stage_directions
        else f"translated stage directions: {', '.join(translated_stage_directions[:10])}",
    ))
    term_errors = terminology_contract_errors(manifest.transcript, manifest.terminology)
    checks.append(CheckResult(
        "terminology_contract", not term_errors,
        "natural terminology contract passed" if not term_errors else "; ".join(term_errors),
    ))
    fast = [item.id for item in fast_translation_cues(manifest.transcript)]
    checks.append(CheckResult(
        "subtitle_reading_speed", not fast,
        "subtitle reading speed passed" if not fast else f"too fast: {', '.join(fast[:10])}",
    ))
    checks.append(CheckResult(
        "rights_review", manifest.rights_review.status != "unreviewed",
        "reuse basis reviewed" if manifest.rights_review.status != "unreviewed" else "human reuse-basis review required before publication",
    ))
    return checks


def _translation_source_key(value: str) -> str:
    without_stage_directions = re.sub(r"\[[^\]]+\]", " ", value)
    return re.sub(r"[^a-z0-9]+", "", without_stage_directions.casefold())


def _translation_source_fingerprint(cues: list[TranscriptCue]) -> str:
    keys = [
        key for cue in cues
        if (key := _translation_source_key(cue.source_text))
        and not FILLER_ONLY.fullmatch(re.sub(r"\s+", " ", cue.source_text).strip())
    ]
    return hashlib.sha256("".join(keys).encode()).hexdigest()


def _strict_interview_source_fingerprint(cues: list[TranscriptCue]) -> str:
    """Bind a review to every publishable source character, including fillers."""
    source = re.sub(
        r"\s+", " ", " ".join(
            omit_non_speech_directions(cue.source_text) for cue in cues
        ),
    ).strip()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _interview_caption_content_fingerprint(cues: list[TranscriptCue]) -> str:
    rows = [{
        "source": re.sub(r"\s+", " ", cue.source_text).strip(),
        "translation": re.sub(r"\s+", " ", cue.translation).strip(),
        "start": round(cue.start, 3), "end": round(cue.end, 3),
    } for cue in cues]
    return hashlib.sha256(json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _terminology_decision_fingerprint(
    terminology: list[TerminologyEntry],
) -> str:
    """Bind cached semantic review to the exact terminology decisions it saw."""
    rows = [{
        "source": entry.source,
        "strategy": entry.strategy.value,
        "target": entry.target,
        "first_use_explanation": entry.first_use_explanation,
        "notes": entry.notes,
        "alternatives": list(entry.alternatives),
        "rationale": entry.rationale,
    } for entry in terminology]
    return hashlib.sha256(json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _compact_translation_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep cache/recovery facts inline while moving verbose attempts to the audit."""
    compact: list[dict[str, Any]] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        step = str(item.get("step") or "")
        if step == "interview_joint_boundary_translation":
            compact.append({
                key: value for key, value in item.items()
                if key not in {"attempts"}
            })
            continue
        if step == "joint_caption_scopes_translation":
            compact.append({
                key: value for key, value in item.items()
                if key != "scope_traces"
            })
            continue
        if step in {
            "interview_joint_boundary_translation_failed",
            "joint_caption_scopes_translation_failed",
        }:
            attempts = item.get("attempts")
            attempt_rows = attempts if isinstance(attempts, list) else []
            failed_card_ids: list[str] = []
            final_errors: Any = None
            for row in reversed(attempt_rows):
                if not isinstance(row, dict):
                    continue
                if final_errors is None and row.get("errors"):
                    final_errors = row.get("errors")
                ids = row.get("failed_card_ids")
                if isinstance(ids, list):
                    failed_card_ids.extend(str(value) for value in ids)
                if final_errors is not None and failed_card_ids:
                    break
            compact.append({
                "step": step,
                "error": str(item.get("error") or "")[:2000],
                "attempt_count": len(attempt_rows),
                "failed_card_ids": list(dict.fromkeys(failed_card_ids)),
                "final_errors": final_errors,
            })
            continue
        row = dict(item)
        for verbose_key in (
            "batches", "failed_cards", "previous_proposal",
            "translation_attempt_provenance", "review_attempts",
        ):
            value = row.get(verbose_key)
            if isinstance(value, list) and len(json.dumps(value, ensure_ascii=False)) > 8000:
                row[verbose_key + "_count"] = len(value)
                row.pop(verbose_key, None)
        compact.append(row)
    return compact


def _write_translation_audit(
    job: Path, trace: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "trace": trace,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path = job / "translation-audit.json"
    path.write_bytes(encoded)
    return {
        "asset": path.name,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "schema_version": 1,
    }


def _translation_trace_from_plan(
    plan_path: Path, payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load a checksum-bound full audit, falling back to the inline cache trace."""
    inline = [
        row for row in payload.get("trace", []) if isinstance(row, dict)
    ]
    reference = payload.get("translation_audit")
    if not isinstance(reference, dict):
        return inline
    asset = str(reference.get("asset") or "").strip()
    expected_sha = str(reference.get("sha256") or "").strip()
    if not asset or Path(asset).name != asset or not expected_sha:
        return inline
    path = plan_path.parent / asset
    try:
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != expected_sha:
            return inline
        audit = json.loads(encoded)
    except (OSError, json.JSONDecodeError):
        return inline
    rows = audit.get("trace") if isinstance(audit, dict) else None
    return [row for row in rows if isinstance(row, dict)] \
        if isinstance(rows, list) else inline


def _merge_cached_translations(
    current: list[TranscriptCue], cached: list[TranscriptCue],
) -> None:
    """Reuse a reviewed translation while restoring fillers and stage directions."""
    by_key: dict[str, list[TranscriptCue]] = {}
    for cue in cached:
        key = _translation_source_key(cue.source_text)
        if key:
            by_key.setdefault(key, []).append(cue)
    offsets: dict[str, int] = {}
    for cue in current:
        key = _translation_source_key(cue.source_text)
        matches = by_key.get(key, []) if key else []
        offset = offsets.get(key, 0)
        if offset < len(matches):
            cue.translation = matches[offset].translation
            offsets[key] = offset + 1
            continue
        if source_is_omittable_caption_only(cue.source_text):
            cue.translation = ""
        else:
            raise ValueError(f"reviewed translation is missing source cue: {cue.id} {cue.source_text!r}")


def rebase_interview_clip_timeline(
    cues: list[TranscriptCue], plan: dict[str, Any], download_window: dict[str, float],
    media_duration: float, previous_clip: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Move a selected interview edit onto its downloaded clip timeline.

    Original source seconds remain on every cue/range and in the returned clip
    provenance so a cached plan can request the same remote interval later.
    """
    offset = float(download_window["download_start"])
    original_start = float(download_window["original_start"])
    original_end = float(download_window["original_end"])
    previous_offset = float((previous_clip or {}).get("download_start") or 0)
    previous_rebased = bool((previous_clip or {}).get("rebased"))
    rebased_cues: list[TranscriptCue] = []
    for cue in cues:
        cue_original_start = cue.original_start
        cue_original_end = cue.original_end
        if cue_original_start is None:
            cue_original_start = cue.start + previous_offset if previous_rebased else cue.start
        if cue_original_end is None:
            cue_original_end = cue.end + previous_offset if previous_rebased else cue.end
        if min(cue_original_end, original_end) - max(cue_original_start, original_start) < 0.5:
            continue
        local_start = max(0.0, cue_original_start - offset)
        local_end = min(media_duration, cue_original_end - offset)
        if local_end <= local_start:
            continue
        rebased_tokens: list[dict[str, Any]] = []
        for row in cue.source_tokens:
            if not isinstance(row, dict):
                continue
            try:
                token_start = float(row["start"])
                token_end = float(row["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if previous_rebased:
                token_start += previous_offset
                token_end += previous_offset
            token_local_start = max(0.0, token_start - offset)
            token_local_end = min(media_duration, token_end - offset)
            if token_local_end <= token_local_start:
                continue
            rebased_tokens.append({
                **row,
                "start": round(token_local_start, 3),
                "end": round(token_local_end, 3),
            })
        cue.start = round(local_start, 3)
        cue.end = round(local_end, 3)
        cue.original_start = round(cue_original_start, 3)
        cue.original_end = round(cue_original_end, 3)
        cue.source_tokens = rebased_tokens
        rebased_cues.append(cue)
    cues[:] = rebased_cues

    local_start = max(0.0, original_start - offset)
    local_end = min(media_duration, original_end - offset)
    if local_end - local_start <= 0:
        raise ValueError(
            "downloaded interview interval cannot contain the validated highlight"
        )
    plan["story_start"] = round(local_start, 3)
    plan["story_end"] = round(local_end, 3)
    rows = plan.get("wechat_lessons")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("known-tech interview must contain one highlight before rebasing")
    rows[0].update({
        "start": round(local_start, 3), "end": round(local_end, 3),
        "original_start": round(original_start, 3),
        "original_end": round(original_end, 3),
    })
    return {
        **download_window,
        "media_duration": round(media_duration, 3),
        "rebased": True,
    }


def _local_interview_media_window(
    cached_clip: dict[str, Any], media_duration: float, original_duration: float,
) -> tuple[dict[str, float], bool]:
    """Identify whether supplied local media is the cached bounded clip or the full source."""
    cached_duration = float(cached_clip.get("media_duration") or 0)
    tolerance = max(2.0, cached_duration * 0.03)
    required = ("original_start", "original_end", "download_start", "download_end")
    has_explicit_window = all(cached_clip.get(key) is not None for key in required)
    explicit_duration = (
        float(cached_clip["download_end"]) - float(cached_clip["download_start"])
        if has_explicit_window else 0.0
    )
    if (
        has_explicit_window
        and (
            (cached_duration > 0 and abs(media_duration - cached_duration) <= tolerance)
            or abs(media_duration - explicit_duration) <= max(2.0, explicit_duration * 0.03)
        )
    ):
        return ({key: float(cached_clip[key]) for key in required}, True)
    original_start = float(cached_clip["original_start"])
    original_end = float(cached_clip["original_end"])
    selected_duration = original_end - original_start
    if (
        not cached_clip.get("rebased")
        and media_duration < original_duration - 5.0
        and selected_duration <= media_duration <= selected_duration + 15.0
    ):
        # Fast stream-copy downloads can include a preceding video keyframe.
        # The requested end remains stable; derive the actual local start from
        # the probed duration so source seconds and subtitles stay aligned.
        download_end = min(original_duration, original_end + INTERVIEW_BOUNDARY_PADDING_SECONDS)
        return ({
            "original_start": original_start, "original_end": original_end,
            "download_start": max(0.0, download_end - media_duration),
            "download_end": download_end,
        }, True)
    return ({
        "original_start": original_start,
        "original_end": original_end,
        "download_start": 0.0,
        "download_end": float(original_duration),
    }, False)


def _previous_clip_for_local_timeline(
    cues: list[TranscriptCue], cached_clip: dict[str, Any],
    local_window: dict[str, float], media_duration: float,
) -> dict[str, Any]:
    """Mark no-render plans whose selected cues are already clip-local."""
    if not cues:
        return cached_clip
    original_start = float(local_window["original_start"])
    original_end = float(local_window["original_end"])
    fits_local_media = all(
        -0.5 <= cue.start < cue.end <= media_duration + 1.0 for cue in cues
    )
    overlaps_original_timeline = any(
        min(cue.end, original_end) - max(cue.start, original_start) >= 0.5
        for cue in cues
    )
    if overlaps_original_timeline:
        return {**cached_clip, "rebased": False}
    if fits_local_media:
        return {
            **cached_clip,
            "rebased": True,
            "download_start": float(local_window["download_start"]),
            "download_end": float(local_window["download_end"]),
            "media_duration": media_duration,
        }
    return {**cached_clip, "rebased": False}


class YouTubeCollectionFactory:
    def __init__(
        self, workspace: Workspace, writer: OpenAICompatibleStoryWriter,
        directing_writer: OpenAICompatibleStoryWriter | None = None,
        subtitle_reviewer: OpenAICompatibleStoryWriter | None = None,
        runtime_guidance: str = "",
    ) -> None:
        self.workspace = workspace
        self.writer = writer
        self.directing_writer = directing_writer
        self.subtitle_reviewer = subtitle_reviewer
        self.runtime_guidance = runtime_guidance.strip()

    def _approved_interview_hook_pair(
        self, source_video_id: str, lesson: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Reuse only a human-approved queued first-screen package for the same clip."""
        expected_start = lesson.get("original_start")
        expected_end = lesson.get("original_end")
        if not source_video_id or expected_start is None or expected_end is None:
            return None
        batches = sorted(
            self.workspace.publish_dir.glob("*/batch.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for batch_path in batches:
            try:
                batch = json.loads(batch_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                batch.get("queue_hidden")
                or not batch.get("approved_at")
                or not str(batch.get("manifest_id") or "").startswith(
                    f"youtube-{source_video_id}-"
                )
            ):
                continue
            manifest_path = (
                self.workspace.manifests_dir / "collections"
                / f"{batch['manifest_id']}.json"
            )
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(manifest.get("source_video_id") or "") != source_video_id:
                continue
            for item in manifest.get("items", []):
                if not isinstance(item, dict):
                    continue
                ranges = item.get("source_ranges")
                source_range = ranges[0] if isinstance(ranges, list) and ranges else {}
                if not isinstance(source_range, dict):
                    continue
                approved_start = source_range.get("original_start")
                approved_end = source_range.get("original_end")
                if (
                    approved_start is None or approved_end is None
                    or abs(float(approved_start) - float(expected_start)) > 1.0
                    or abs(float(approved_end) - float(expected_end)) > 1.0
                ):
                    continue
                renders = item.get("renders")
                render_row = renders[0] if isinstance(renders, list) and renders else {}
                if not isinstance(render_row, dict):
                    continue
                selected = render_row.get("selected_hook")
                if not isinstance(selected, dict):
                    continue
                headline = str(selected.get("headline_zh") or "").strip()
                context = str(selected.get("promise") or "").strip()
                if (
                    not 8 <= len(re.sub(r"\s+", "", headline)) <= 36
                    or len(re.sub(r"\s+", "", context))
                    < INTERVIEW_HOOK_CONTEXT_MIN_VISIBLE_CHARACTERS
                    or not _interview_hook_context_fits_overlay(context)
                ):
                    continue
                hook_rows = render_row.get("hook_candidates")
                hooks = [
                    str(candidate.get("headline_zh") or "").strip()
                    for candidate in hook_rows
                    if isinstance(candidate, dict)
                    and str(candidate.get("headline_zh") or "").strip()
                ] if isinstance(hook_rows, list) else []
                hooks = [headline, *[value for value in hooks if value != headline]][:3]
                return {
                    "headline": headline, "context": context,
                    "hook_headlines": hooks, "manifest_id": batch["manifest_id"],
                    "approved_at": batch["approved_at"],
                }
        return None

    def generate(
        self, url: str, job: Path, render: bool = True, local_media: Path | None = None,
        local_subtitles: Path | None = None,
        translation_plan: Path | None = None,
        editorial_mode: str = "auto",
        editorial_guidance: str | None = None,
    ) -> dict[str, Any]:
        acquirer = YouTubeAcquirer(self.workspace)
        candidate, evidence, metadata, cues, media_asset, subtitle_asset, media_info = acquirer.acquire(
            url, job, local_media=local_media, local_subtitles=local_subtitles,
            # Editorial selection must precede remote media acquisition. Local
            # media is still archived by acquire() and keeps its full timeline.
            download_media=False,
        )
        acquired_source_cues = [TranscriptCue(**asdict(cue)) for cue in cues]
        classified_mode, known_people, _ = classify_youtube_editorial(
            str(metadata.get("title") or ""),
            str(metadata.get("channel") or metadata.get("uploader") or ""),
            str(metadata.get("description") or ""),
            [item for item in (metadata.get("chapters") or []) if isinstance(item, dict)],
            [str(item) for item in (metadata.get("creators") or [])],
        )
        metadata["known_tech_people"] = known_people
        if editorial_mode == "auto":
            channel_name = str(metadata.get("channel") or metadata.get("uploader") or "").casefold()
            trusted_long_conference = bool(
                any(marker in channel_name for marker in TRUSTED_CHANNEL_MARKERS)
                and float(metadata.get("duration") or 0) > 7200
                and not metadata.get("chapters")
                and classified_mode in {"technical_coverage", "known_tech_interview_clip"}
            )
            editorial_mode = (
                "conference_highlights" if trusted_long_conference else classified_mode
            )
        if editorial_mode not in {
            "technical_coverage", "known_tech_interview_clip",
            "conference_highlights", "study",
        }:
            raise ValueError(f"YouTube source is not eligible for an editorial route: {editorial_mode}")
        metadata["editorial_mode"] = editorial_mode
        translation_plan, caption_incumbent_trace = select_caption_incumbent(
            self.workspace, str(metadata.get("id") or ""), translation_plan,
            editorial_mode, editorial_guidance,
        )
        translator = NaturalSubtitleTranslator(
            self.writer, self.directing_writer, self.subtitle_reviewer,
            runtime_guidance=self.runtime_guidance,
        )
        cached_source_clip: dict[str, Any] | None = None
        cached_interview_captions_complete = False
        cached_joint_captions_complete = False
        if translation_plan is not None:
            cues[:] = rebalance_source_cues(cues)
            cached = json.loads(translation_plan.read_text(encoding="utf-8"))
            cached_source_clip = dict(cached.get("source_clip") or {}) or None
            cached_mode = str(cached.get("editorial_mode") or "study")
            if cached_mode != editorial_mode:
                raise ValueError(
                    f"translation plan editorial mode is {cached_mode}, expected {editorial_mode}"
                )
            cached_cues = [TranscriptCue(**item) for item in cached.get("transcript", [])]
            cached_source_video_id = str(cached.get("source_video_id") or "")
            current_source_video_id = str(metadata.get("id") or "")
            source_selected_cache = (
                editorial_mode in {"known_tech_interview_clip", "conference_highlights"}
                and cached_source_video_id
                and cached_source_video_id == current_source_video_id
            )
            interview_cache = (
                editorial_mode == "known_tech_interview_clip"
                and source_selected_cache
            )
            if not cached_cues or (
                not source_selected_cache
                and _translation_source_fingerprint(cached_cues) != _translation_source_fingerprint(cues)
            ):
                raise ValueError("translation plan transcript does not match the supplied YouTube subtitles")
            if source_selected_cache or all(
                cue.source_text.strip() and cue.translation.strip() for cue in cached_cues
            ):
                # The reviewed plan may merge adjacent source fragments again to meet
                # reading-speed limits. The whole-source fingerprint above proves that
                # no English content was lost. Interview plans intentionally retain only
                # the selected clip and are instead bound to the exact source video id.
                cues[:] = cached_cues
            else:
                _merge_cached_translations(cues, cached_cues)
            editorial_plan = dict(cached.get("editorial_plan") or {})
            cached_terminology_rows = cached.get("terminology", [])
            terminology = translator._parse_terminology(cached_terminology_rows, cues)
            trace = _translation_trace_from_plan(translation_plan, cached)
            if caption_incumbent_trace:
                trace.append(caption_incumbent_trace)
            cached_interview_captions_complete = (
                editorial_mode == "known_tech_interview_clip"
                and cached_interview_caption_pipeline_complete(
                    cues, trace, media_info.sha256 if media_info else "", terminology,
                )
            )
            cached_joint_captions_complete = (
                editorial_mode != "known_tech_interview_clip"
                and cached_joint_caption_pipeline_complete(
                    cues, trace, terminology,
                )
            )
            editorial_plan, repairs = translator.ensure_editorial_plan(
                metadata, acquired_source_cues, editorial_plan, editorial_mode,
            )
            trace.extend(repairs)
            if interview_cache:
                selected_rows = editorial_plan.get("wechat_lessons")
                selected_row = (
                    selected_rows[0] if isinstance(selected_rows, list)
                    and selected_rows and isinstance(selected_rows[0], dict) else None
                )
                selected = _coerce_range(
                    selected_row, float(metadata.get("duration") or 0),
                ) if selected_row else None
                if cached_source_clip and cached_source_clip.get("original_start") is not None:
                    selected = SourceRange(
                        float(cached_source_clip["original_start"]),
                        float(cached_source_clip["original_end"]),
                    )
                if selected is None:
                    raise ValueError("cached interview plan has no valid source range")
                selected_source_cues = [
                    TranscriptCue(**asdict(cue)) for cue in acquired_source_cues
                    if min(cue.end, selected.end) - max(cue.start, selected.start) >= 0.5
                ]
                if (
                    cached_cues
                    and _strict_interview_source_fingerprint(selected_source_cues)
                    != _strict_interview_source_fingerprint(cached_cues)
                ):
                    if not interview_cache:
                        raise ValueError(
                            "cached interview plan no longer matches current YouTube subtitles"
                        )
                    cached_interview_captions_complete = False
                    trace.append({
                        "step": "cached_interview_source_challenger_rejected",
                        "reason": (
                            "cached English differs from the current authoritative "
                            "YouTube spoken source; rebuild from current subtitles"
                        ),
                        "current_source_fingerprint": (
                            _strict_interview_source_fingerprint(selected_source_cues)
                        ),
                        "cached_source_fingerprint": (
                            _strict_interview_source_fingerprint(cached_cues)
                        ),
                    })
                if not cached_interview_captions_complete:
                    cues[:] = selected_source_cues
                    for cue in cues:
                        cue.translation = ""
                    # The current authoritative subtitles may have gained or
                    # lost terms since the cached plan was written. Reparse
                    # against the challenger source so absent rows are dropped
                    # and current established/protected terms are restored
                    # before terminology review and joint translation.
                    terminology = translator._parse_terminology(
                        cached_terminology_rows, cues,
                    )
                    trace.append({
                        "step": "legacy_interview_caption_challenger",
                        "reason": "cached captions predate audio-aligned joint translation",
                    })
            elif not cached_joint_captions_complete:
                # Keep the accepted editorial plan byte-for-byte while replacing
                # only subtitles produced by the retired cue pipeline.
                cues[:] = _source_cues_for_plan(
                    acquired_source_cues, editorial_plan,
                    float(metadata.get("duration") or 0),
                )
                for cue in cues:
                    cue.translation = ""
                terminology = translator._parse_terminology(
                    cached_terminology_rows, cues,
                )
                trace.append({
                    "step": "legacy_caption_pipeline_challenger",
                    "editorial_mode": editorial_mode,
                    "reason": (
                        "preserve editorial selection and rebuild subtitles with "
                        "joint boundary translation"
                    ),
                })
        else:
            translate_kwargs: dict[str, Any] = {"plan_only": True}
            if editorial_guidance and editorial_guidance.strip():
                translate_kwargs["editorial_guidance"] = editorial_guidance
            terminology, editorial_plan, trace = translator.translate(
                metadata, cues, editorial_mode, **translate_kwargs,
            )
            if caption_incumbent_trace:
                trace.append(caption_incumbent_trace)
        original_duration = float(metadata.get("duration") or (cues[-1].end if cues else 0))
        source_clip: dict[str, Any] | None = None
        selected_interview_range: SourceRange | None = None
        frozen_hook_snapshot = (
            {}
            if editorial_mode == "known_tech_interview_clip"
            else _snapshot_plan_hooks(
                editorial_plan, original_duration, acquired_source_cues,
                [str(value) for value in metadata.get("known_tech_people", [])],
            )
        )
        if editorial_mode == "known_tech_interview_clip":
            rows = editorial_plan.get("wechat_lessons")
            raw_selected = (
                rows[0] if isinstance(rows, list) and rows
                and isinstance(rows[0], dict) else None
            )
            selected_interview_range = _coerce_range(raw_selected, original_duration)
            if cached_source_clip and cached_source_clip.get("original_start") is not None:
                selected_interview_range = SourceRange(
                    float(cached_source_clip["original_start"]),
                    float(cached_source_clip["original_end"]),
                )
            if selected_interview_range is None:
                raise ValueError("interview highlight has no valid source range before media acquisition")
            source_clip = {
                "original_start": selected_interview_range.start,
                "original_end": selected_interview_range.end,
                "rebased": False,
            }
            if not cached_interview_captions_complete:
                trace.append(translator.discover_missing_terminology(
                    cues, terminology,
                ))
                terminology_review = translator.review_terminology_decisions(
                    cues, terminology,
                )
                if terminology_review:
                    trace.append(terminology_review)
                translation_audit = _write_translation_audit(job, trace)
                (job / "translation-plan.json").write_text(json.dumps({
                    "editorial_mode": editorial_mode,
                    "source_video_id": str(metadata.get("id") or ""),
                    "source_clip": source_clip,
                    "editorial_plan": editorial_plan,
                    "terminology": [asdict(item) for item in terminology],
                    "transcript": [asdict(item) for item in cues],
                    "trace": _compact_translation_trace(trace),
                    "translation_audit": translation_audit,
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if not media_asset:
                    media_asset, media_info, video_evidence, download_window = (
                        acquirer.acquire_remote_media(
                            candidate, metadata, url, job,
                            source_range=selected_interview_range,
                        )
                    )
                    evidence.append(video_evidence)
                    if download_window is not None:
                        source_clip = rebase_interview_clip_timeline(
                            cues, editorial_plan, download_window, media_info.duration,
                            previous_clip=cached_source_clip,
                        )
                        metadata["original_duration"] = original_duration
                        metadata["duration"] = media_info.duration
                        metadata["source_clip"] = source_clip
                        candidate.metadata.update({
                            "duration": media_info.duration,
                            "original_duration": original_duration,
                            "source_clip": source_clip,
                        })
                        self.workspace.save_candidate(candidate)
                elif cached_source_clip and media_info:
                    clip_provenance = cached_source_clip
                    local_window, local_is_bounded_clip = _local_interview_media_window(
                        clip_provenance, media_info.duration, original_duration,
                    )
                    if local_is_bounded_clip:
                        previous_clip = _previous_clip_for_local_timeline(
                            cues, clip_provenance, local_window, media_info.duration,
                        )
                        source_clip = rebase_interview_clip_timeline(
                            cues, editorial_plan, local_window, media_info.duration,
                            previous_clip=previous_clip,
                        )
                        source_clip["rebased"] = True
                        source_clip["local_media_full_source"] = False
                        metadata["original_duration"] = original_duration
                        metadata["duration"] = media_info.duration
                        metadata["source_clip"] = source_clip
                elif (
                    media_info
                    and media_info.duration < original_duration - 5.0
                ):
                    raise ValueError(
                        "bounded local interview media requires a translation plan "
                        "with source_clip provenance; media duration alone cannot prove "
                        "which original interval it contains"
                    )
                media_path = Path(str(media_asset))
                if not media_path.is_absolute():
                    media_path = self.workspace.root / media_path
                cues[:] = merge_interview_source_cues(cues, maximum_duration=60.0)
                alignment = AudioWordAligner().align(
                    media_path, cues, job,
                    media_sha256=media_info.sha256 if media_info else "",
                )
                ledger_path = job / "source-word-ledger.json"
                ledger_path.write_text(json.dumps({
                    "alignment": alignment.trace(),
                    "audio_hypothesis": alignment.audio_hypothesis,
                    "source_words": [asdict(word) for word in alignment.words],
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                alignment_trace = alignment.trace()
                alignment_trace["ledger_asset"] = ledger_path.name
                alignment_trace["ledger_sha256"] = hashlib.sha256(
                    ledger_path.read_bytes()
                ).hexdigest()
                trace.append(alignment_trace)
                try:
                    trace.append(translator.translate_interview_clip_once(
                        cues, terminology, alignment.words, alignment.fingerprint,
                        alignment.audio_hypothesis,
                    ))
                except Exception as exc:
                    failure_trace = {
                        "step": "interview_joint_boundary_translation_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    if isinstance(exc, InterviewJointTranslationError):
                        failure_trace["attempts"] = exc.trace
                    trace.append(failure_trace)
                    translation_audit = _write_translation_audit(job, trace)
                    (job / "translation-plan.json").write_text(json.dumps({
                        "editorial_mode": editorial_mode,
                        "source_video_id": str(metadata.get("id") or ""),
                        "source_clip": source_clip,
                        "editorial_plan": editorial_plan,
                        "terminology": [asdict(item) for item in terminology],
                        "transcript": [asdict(item) for item in cues],
                        "trace": _compact_translation_trace(trace),
                        "translation_audit": translation_audit,
                    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    raise
        if editorial_mode == "known_tech_interview_clip" and cached_interview_captions_complete:
            trace.append({
                "step": "interview_caption_pipeline_reuse",
                "reason": (
                    "cached plan already contains audio-aligned, independently "
                    "reviewed joint cards"
                ),
            })
        if editorial_mode == "known_tech_interview_clip":
            reviewed_audio_conflicts = set(next((
                item.get("audio_conflict_entities", [])
                for item in reversed(trace) if isinstance(item, dict)
                and item.get("step") == "interview_joint_boundary_translation"
            ), []))
            if caption_errors := interview_caption_duration_errors(
                cues, terminology, reviewed_audio_conflicts,
            ):
                raise ValueError(
                    "interview caption policy gate failed; regenerate from the raw "
                    "translation plan instead of reprocessing cached cards: "
                    + "; ".join(caption_errors)
                )
        else:
            if not cached_joint_captions_complete:
                trace.append(translator.discover_missing_terminology(
                    cues, terminology,
                ))
                terminology_review = translator.review_terminology_decisions(
                    cues, terminology,
                )
                if terminology_review:
                    trace.append(terminology_review)
                try:
                    trace.append(translator.translate_caption_scopes(
                        cues, terminology, editorial_plan, original_duration,
                        editorial_mode,
                    ))
                except Exception as exc:
                    failure_trace = {
                        "step": "joint_caption_scopes_translation_failed",
                        "editorial_mode": editorial_mode,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    if isinstance(exc, InterviewJointTranslationError):
                        failure_trace["attempts"] = exc.trace
                    trace.append(failure_trace)
                    translation_audit = _write_translation_audit(job, trace)
                    (job / "translation-plan.json").write_text(json.dumps({
                        "editorial_mode": editorial_mode,
                        "source_video_id": str(metadata.get("id") or ""),
                        "source_clip": source_clip,
                        "editorial_plan": editorial_plan,
                        "terminology": [asdict(item) for item in terminology],
                        "transcript": [asdict(item) for item in cues],
                        "trace": _compact_translation_trace(trace),
                        "translation_audit": translation_audit,
                    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    raise
            else:
                trace.append({
                    "step": "joint_caption_scopes_reuse",
                    "editorial_mode": editorial_mode,
                    "reason": "cached cards match the current joint subtitle policy",
                })
            if caption_errors := interview_caption_duration_errors(cues, terminology):
                raise ValueError(
                    "joint caption policy gate failed: " + "; ".join(caption_errors)
                )
        if editorial_mode == "known_tech_interview_clip":
            approved_pair: dict[str, Any] | None = None
            pair_rows = editorial_plan.get("wechat_lessons")
            pair_row = (
                pair_rows[0] if isinstance(pair_rows, list) and pair_rows
                and isinstance(pair_rows[0], dict) else None
            )
            if pair_row is not None:
                approved_pair = self._approved_interview_hook_pair(
                    str(metadata.get("id") or candidate.metadata.get("video_id") or ""),
                    pair_row,
                )
                if approved_pair:
                    pair_row["title"] = approved_pair["headline"]
                    pair_row["hook_headlines"] = approved_pair["hook_headlines"]
                    pair_row["hook_context"] = approved_pair["context"]
                    editorial_plan["collection_title"] = approved_pair["headline"]
                    trace.append({
                        "step": "approved_first_screen_incumbent",
                        **approved_pair,
                    })
            requested_title_trace = _apply_supported_requested_title(
                editorial_plan, cues,
                float(metadata.get("duration") or (cues[-1].end if cues else 0)),
                editorial_guidance,
            )
            if requested_title_trace:
                trace.append(requested_title_trace)
            audit_rows = editorial_plan.get("wechat_lessons")
            audit_row = (
                audit_rows[0] if isinstance(audit_rows, list) and audit_rows
                and isinstance(audit_rows[0], dict) else {}
            )
            current_audited_hooks = [
                str(value).strip() for value in audit_row.get("hook_headlines", [])
                if str(value).strip()
            ] if isinstance(audit_row.get("hook_headlines"), list) else []
            current_context = str(audit_row.get("hook_context") or "").strip()
            requested_pair_is_locked = bool(
                requested_title_trace
                and requested_title_trace.get("applied")
                and current_audited_hooks
                and current_context
                and _interview_hook_context_fits_overlay(current_context)
            )
            approved_pair_is_locked = bool(
                approved_pair and current_context == approved_pair.get("context")
                and current_audited_hooks
                and current_audited_hooks[0] == approved_pair.get("headline")
            )
            completed_audit = _matching_completed_directing_audit(
                trace, current_audited_hooks, current_context,
            )
            if requested_pair_is_locked or approved_pair_is_locked or completed_audit:
                trace.append({
                    "step": "interview_directing_audit_reuse",
                    "hook_headlines": current_audited_hooks,
                    "hook_context": current_context,
                    "reason": (
                        "human-requested supported title remains paired with its fixed subtitle"
                        if requested_pair_is_locked else
                        "human-approved queued headline/subtitle pair is unchanged"
                        if approved_pair_is_locked else
                        "previously audited headline/subtitle pair is unchanged"
                    ),
                })
            else:
                trace.append(translator.audit_interview_directing(
                    editorial_plan, cues,
                    float(metadata.get("duration") or (cues[-1].end if cues else 0)),
                ))
            # Explicit human editorial guidance wins as part of the locked
            # first-screen pair. Reapplying is idempotent because the audit is
            # skipped above for a valid requested-title/context package.
            final_requested_title_trace = _apply_supported_requested_title(
                editorial_plan, cues,
                float(metadata.get("duration") or (cues[-1].end if cues else 0)),
                editorial_guidance,
            )
            if final_requested_title_trace:
                trace.append({
                    **final_requested_title_trace,
                    "step": "requested_title_final",
                })
        if filler_cue_ids := omit_spoken_fillers(cues):
            raise ValueError(
                "filler cleanup changed independently reviewed joint cards; "
                "regenerate so filler omission occurs before review: "
                + ", ".join(filler_cue_ids)
            )
        def persist_translation_plan() -> None:
            translation_audit = _write_translation_audit(job, trace)
            (job / "translation-plan.json").write_text(json.dumps({
                "editorial_mode": editorial_mode,
                "source_video_id": str(metadata.get("id") or ""),
                "source_clip": source_clip,
                "editorial_plan": editorial_plan,
                "terminology": [asdict(item) for item in terminology],
                "transcript": [asdict(item) for item in cues],
                "trace": _compact_translation_trace(trace),
                "translation_audit": translation_audit,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Preserve the expensive editorial and translation work before remote
        # media acquisition. A blocked download can then resume from this plan.
        persist_translation_plan()

        if render and not media_asset:
            media_asset, media_info, video_evidence, download_window = acquirer.acquire_remote_media(
                candidate, metadata, url, job,
                source_range=selected_interview_range
                if editorial_mode == "known_tech_interview_clip" else None,
            )
            evidence.append(video_evidence)
            if download_window is not None:
                source_clip = rebase_interview_clip_timeline(
                    cues, editorial_plan, download_window, media_info.duration,
                    previous_clip=cached_source_clip,
                )
                metadata["original_duration"] = original_duration
                metadata["duration"] = media_info.duration
                metadata["source_clip"] = source_clip
                candidate.metadata.update({
                    "duration": media_info.duration,
                    "original_duration": original_duration,
                    "source_clip": source_clip,
                })
                self.workspace.save_candidate(candidate)
        elif source_clip is not None and not source_clip.get("rebased"):
            # Rendering from a supplied local source keeps original timestamps;
            # render=False plans remain reusable for a future bounded download.
            if media_asset and cached_source_clip:
                local_window, local_is_bounded_clip = _local_interview_media_window(
                    cached_source_clip,
                    media_info.duration if media_info else original_duration,
                    original_duration,
                )
                previous_clip = _previous_clip_for_local_timeline(
                    cues, cached_source_clip, local_window,
                    media_info.duration if media_info else original_duration,
                )
                source_clip = rebase_interview_clip_timeline(
                    cues, editorial_plan, local_window,
                    media_info.duration if media_info else original_duration,
                    previous_clip=previous_clip,
                )
                source_clip["rebased"] = local_is_bounded_clip
                source_clip["local_media_full_source"] = not local_is_bounded_clip
                if local_is_bounded_clip and media_info:
                    metadata["original_duration"] = original_duration
                    metadata["duration"] = media_info.duration
                    metadata["source_clip"] = source_clip
                    candidate.metadata.update({
                        "duration": media_info.duration,
                        "original_duration": original_duration,
                        "source_clip": source_clip,
                    })
                    self.workspace.save_candidate(candidate)
            else:
                source_clip["local_media_full_source"] = bool(media_asset)

        persist_translation_plan()
        manifest = build_collection_manifest(
            candidate, metadata, cues, terminology, editorial_plan, media_asset, subtitle_asset,
            media_info, frozen_hook_snapshot,
        )
        if render:
            YouTubeCollectionRenderer(self.workspace).render(manifest)
            self._link_superseded_collection(manifest)
        checks = validate_collection(manifest, self.workspace.root)
        manifest.quality_checks = [item.to_dict() for item in checks]
        path = self.workspace.save_collection_manifest(manifest)
        job_manifest = job / "collection-manifest.json"
        shutil.copy2(path, job_manifest)
        translation_audit = _write_translation_audit(job, trace)
        result = {
            "status": "completed", "source_type": "youtube", "candidate": candidate.id,
            "editorial_mode": editorial_mode,
            "collection_manifest": str(job_manifest), "collection_id": manifest.id,
            "items": [{"id": item.id, "title": item.title, "duration": item.duration} for item in manifest.items],
            "translation_trace": _compact_translation_trace(trace),
            "translation_audit": translation_audit,
            "checks": [item.to_dict() for item in checks],
            "publishable": all(item.passed for item in checks),
            "evidence_ids": [item.id for item in evidence], "completed_at": now_iso(),
        }
        (job / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    def _link_superseded_collection(self, manifest: VideoCollectionManifest) -> None:
        """Link the newest rebuild to the previous collection without deleting either."""
        from .serde import load_collection_manifest

        prior: list[tuple[Path, VideoCollectionManifest]] = []
        for path in self.workspace.collections_dir.glob("*.json"):
            try:
                existing = load_collection_manifest(path)
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if (
                existing.id != manifest.id
                and existing.source_video_id == manifest.source_video_id
                and not existing.superseded_by_collection_id
                and any(render.video_path for item in existing.items for render in item.renders)
            ):
                prior.append((path, existing))
        if not prior:
            return
        _, previous = max(prior, key=lambda row: row[1].created_at)
        manifest.supersedes_collection_id = previous.id
        previous.superseded_by_collection_id = manifest.id
        self.workspace.save_collection_manifest(previous)
