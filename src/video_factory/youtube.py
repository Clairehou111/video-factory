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
    SourceType, SubtitleMode, TerminologyEntry, TerminologyStrategy, TranscriptCue,
    VideoCollectionManifest, now_iso,
)
from .quality import CheckResult
from .storage import Workspace
from .youtube_runtime import ManagedYouTubeRuntime


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
    "knowledge work": "知识工作",
    "vertical": "垂直行业",
    "productivity gain": "生产力提升",
    "diffusion": "普及",
    "agent": "智能体",
    "bug": "故障",
    "royalty": "版税",
    "open-source check": "开源制衡",
    "token pricing": "token 定价",
    "memory system": "记忆系统",
    "model family": "模型系列",
    "windows interrupt": "Windows 中断",
    "slop": "低质内容",
}
CONTEXTUAL_CHINESE_TERMS: dict[tuple[str, str], tuple[str, ...]] = {
    # In product/business interviews, “business model” means 商业模式.  A
    # substring-only `model -> 模型` contract corrupts an already correct
    # translation and then loops forever trying to insert the wrong noun.
    ("model", "business model"): ("商业模式",),
    (
        "open-source check", "open-source check on closed source",
    ): ("开源对闭源的制衡",),
    ("open-source check", "open-source check"): ("开源的制衡", "制衡"),
    ("royalty", "royalty"): ("版税", "收益"),
    ("app tier", "app tier"): ("应用层", "应用"),
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
INTERVIEW_MIN_SECONDS = 45.0
INTERVIEW_MAX_SECONDS = 180.0
INTERVIEW_MAX_INTERNAL_SILENCE_SECONDS = 3.0
INTERVIEW_CAPTION_POLICY_VERSION = "2026-09-21-v1"
INTERVIEW_CAPTION_TARGET_MAX_SECONDS = 5.0
INTERVIEW_CAPTION_HARD_MAX_SECONDS = 7.5
INTERVIEW_CAPTION_MIN_SECONDS = 1.2
INTERVIEW_CAPTION_TARGET_MAX_ENGLISH_WORDS = 14
INTERVIEW_CAPTION_MAX_ENGLISH_WORDS = 28
INTERVIEW_CAPTION_TARGET_MAX_CHINESE_CHARACTERS = 22
INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS = 32
INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND = 12.0
INTERVIEW_CAPTION_MAX_RENDERED_LINES = 3
INTERVIEW_HOOK_CONTEXT_MIN_VISIBLE_CHARACTERS = 22
INTERVIEW_HOOK_CONTEXT_DRAFT_TARGET_MAX_VISIBLE_CHARACTERS = 105
INTERVIEW_DIRECTING_POLICY_VERSION = "2026-09-21-v3-atomic-hook-pair"


def _interview_caption_policy_fingerprint() -> str:
    policy = {
        "version": INTERVIEW_CAPTION_POLICY_VERSION,
        "target_max_seconds": INTERVIEW_CAPTION_TARGET_MAX_SECONDS,
        "hard_max_seconds": INTERVIEW_CAPTION_HARD_MAX_SECONDS,
        "minimum_seconds": INTERVIEW_CAPTION_MIN_SECONDS,
        "target_maximum_english_words": INTERVIEW_CAPTION_TARGET_MAX_ENGLISH_WORDS,
        "maximum_english_words": INTERVIEW_CAPTION_MAX_ENGLISH_WORDS,
        "target_maximum_chinese_characters": (
            INTERVIEW_CAPTION_TARGET_MAX_CHINESE_CHARACTERS
        ),
        "maximum_chinese_characters": INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS,
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
        if not config.minimum_duration_seconds <= item.duration_seconds <= config.maximum_duration_seconds:
            reasons.append("duration_out_of_range")
        if item.duration_seconds < 900:
            reasons.append("insufficient_material_for_main_and_three_episodes")
        if not item.transcript_available:
            reasons.append("english_transcript_unavailable")
        if not item.title.strip() or not item.channel.strip():
            reasons.append("missing_identity")
        trusted_channel = any(marker in channel for marker in TRUSTED_CHANNEL_MARKERS)
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
    buffer: list[str] = []
    cue_start = cue_end = 0.0

    def flush() -> None:
        nonlocal buffer, cue_start, cue_end
        text = _join_caption_fragments(buffer)
        if text:
            cues.append(TranscriptCue(
                id=f"cue-{len(cues) + 1:04d}", start=round(cue_start, 3),
                end=round(max(cue_end, cue_start + 0.8), 3), source_text=text,
            ))
        buffer = []

    for start, end, text in tokens:
        if not buffer:
            cue_start = start
        prospective = re.sub(r"\s+", " ", "".join([*buffer, text])).strip()
        if buffer and (start - cue_end > 0.6 or start - cue_start > 6.0 or len(prospective) > 105):
            flush()
            cue_start = start
        buffer.append(text)
        cue_end = end
        if re.search(r"[.!?][\"']?$", text.strip()) and cue_end - cue_start >= 1.0:
            flush()
    if buffer:
        flush()
    for current, following in zip(cues, cues[1:]):
        if 0 < following.start - current.end <= 2.0:
            current.end = round(min(following.start - 0.05, current.start + 6.0), 3)
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
        ):
            errors.append(
                f"term:{entry.source}: established Chinese term must translate as "
                f"{established_target!r}, not {entry.strategy.value}"
            )
            continue
        if entry.strategy == TerminologyStrategy.TRANSLATE and not entry.target.strip():
            errors.append(f"term:{entry.source}: translated terms require target")
        if entry.strategy == TerminologyStrategy.TRANSLATE and entry.target.strip():
            protected_terms = [
                item.source for item in terminology
                if item.strategy == TerminologyStrategy.PRESERVE
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
        if entry.strategy == TerminologyStrategy.PRESERVE and not _contains_term(combined_target, entry.source):
            errors.append(f"term:{entry.source}: preserved English term is missing from translation")
        if entry.strategy == TerminologyStrategy.BILINGUAL_ONCE:
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
    entry: TerminologyEntry, cue: TranscriptCue,
) -> bool:
    """Accept an exact target or a declared phrase-specific Chinese meaning."""
    if entry.target and entry.target in cue.translation:
        return True
    source_key = entry.source.casefold()
    for (term, phrase), targets in CONTEXTUAL_CHINESE_TERMS.items():
        if (
            source_key == term
            and _contains_term(cue.source_text, phrase)
            and any(target in cue.translation for target in targets)
        ):
            return True
    return False


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
        return bool(re.search(
            rf"(?<![A-Za-z0-9]){re.escape(term)}{plural}(?![A-Za-z0-9])", value,
            re.IGNORECASE,
        ))
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
        needed = len(re.sub(r"\s+", "", current.translation)) / max_chars_per_second if current.translation else 0
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
        and len(re.sub(r"\s+", "", cue.translation))
        / max(cue.duration, 0.1) > max_chars_per_second
    ]


def interview_chinese_style_errors(cues: list[TranscriptCue]) -> dict[str, list[str]]:
    errors: dict[str, list[str]] = {}
    for cue in cues:
        found = [
            reason for pattern, reason in INTERVIEW_CHINESE_STYLE_PATTERNS
            if pattern.search(cue.translation)
        ]
        visible = len(re.sub(r"\s+", "", cue.translation))
        if visible > INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS:
            found.append(
                f"exceeds {INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS} visible characters"
            )
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


def normalize_interview_asr_terms(
    cues: list[TranscriptCue], terminology: list[TerminologyEntry],
) -> list[dict[str, str]]:
    """Correct a bounded homophone only when the reviewed glossary confirms it."""
    glossary = {entry.source.casefold() for entry in terminology}
    corrections: list[dict[str, str]] = []
    if "per seat pricing" not in glossary:
        return corrections
    pattern = re.compile(r"\bperceived\s+pricing\b", re.IGNORECASE)
    for cue in cues:
        corrected = pattern.sub("per seat pricing", cue.source_text)
        if corrected == cue.source_text:
            continue
        corrections.append({
            "cue_id": cue.id,
            "before": cue.source_text,
            "after": corrected,
            "basis": "reviewed glossary plus adjacent CRM pricing context",
        })
        cue.source_text = corrected
    return corrections


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
    if editorial_mode != "known_tech_interview_clip" or original_characters <= maximum_characters:
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
    r"(?<!\d)(?P<start>(?:\d{1,2}:)?\d{1,2}:[0-5]\d)\s*"
    r"(?:-|–|—|to|through)\s*"
    r"(?P<end>(?:\d{1,2}:)?\d{1,2}:[0-5]\d)(?!\d)",
    re.IGNORECASE,
)


def _guidance_clock_seconds(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    return float(parts[0] * 3600 + parts[1] * 60 + parts[2])


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
    if not INTERVIEW_MIN_SECONDS <= end - start <= INTERVIEW_MAX_SECONDS:
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
    ) -> None:
        self.writer = writer
        self.directing_writer = directing_writer
        self.subtitle_reviewer = subtitle_reviewer

    def translate(
        self, metadata: dict[str, Any], cues: list[TranscriptCue], editorial_mode: str = "study",
        editorial_guidance: str | None = None,
    ) -> tuple[list[TerminologyEntry], dict[str, Any], list[dict[str, Any]]]:
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
                "Select one continuous, self-contained 45–180 second source range with the strongest surprising, "
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
            "Every terminology item must choose translate, preserve, or bilingual_once. bilingual_once keeps the English term, adds one short Chinese explanation at first use, and then keeps English only.",
            edition_contract,
            f"Return JSON with editorial_mode, collection_title, story_start, story_end, terminology, bilibili_chapters, and wechat_lessons. Every returned lesson has title, thesis, numeric start/end source seconds, framing (auto|speaker|slide|split), and hook_headlines. For an interview highlight also return hook_context: one natural {INTERVIEW_HOOK_CONTEXT_MIN_VISIBLE_CHARACTERS}–{INTERVIEW_HOOK_CONTEXT_DRAFT_TARGET_MAX_VISIBLE_CHARACTERS} visible-character fixed explanatory subtitle placed persistently below the hook for the entire clip. This range is a drafting target; deterministic validation uses the real rendered three-line panel rather than a raw character cutoff. Together, the title and hook_context must let the audience grasp the full conflict, mechanism, and resolution; do not split essential meaning across rotating headlines or spoken captions. Write it like a sharp Chinese tech editor explaining the point to a colleague: short subject-verb clauses, spoken cause-and-effect, and concrete actors/actions. The hook states the tension; hook_context must advance the story rather than repeat it. When the selected source gives an answer, remedy, alternative, or decision, hook_context must include that payoff. It should be understandable at first glance, not read like a translated report or academic abstract. For example, prefer 'SaaS 想靠 API 涨价补收入；客户只愿为结果付费，否则就把数据搬走' over '当厂商因席位减少而提高价格时，反而会激励客户将数据移出系统'. Use split when the source simultaneously shows a speaker pane and a slide pane; this preserves the complete left speaker instead of treating the slide crop as the whole frame. Use slide only for a true slide-only shot or when the crop retains all meaningful content.",
            "Return at most 12 terminology rows, using only terms that occur inside the selected source range. Do not translate or reproduce the transcript in this planning response.",
            (
                "Advisory from YouTube Ask or a human editor (use it only to locate a stronger source-backed angle; "
                "verify every visible claim against the supplied transcript and omit unsupported details): "
                + editorial_guidance.strip()
            ) if editorial_guidance and editorial_guidance.strip() else "",
            "For interview highlights, prefer a self-contained 75–120 second passage and end on a complete thought. Use the 180-second ceiling only when the argument genuinely cannot stand alone at a shorter natural boundary.",
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
        if editorial_mode == "known_tech_interview_clip":
            selected = _coerce_range(plan["wechat_lessons"][0], duration)
            if selected is None:
                raise ValueError("interview highlight has no valid source range")
            cues[:] = [
                cue for cue in cues
                if min(cue.end, selected.end) - max(cue.start, selected.start) >= 0.5
            ]
        terminology = self._parse_terminology(plan.get("terminology", []), cues)
        glossary_rows: list[dict[str, Any]] = []
        for entry in terminology:
            row = asdict(entry)
            row["first_use_cue_id"] = next(
                (
                    cue.id for cue in cues
                    if _contains_term(cue.source_text, entry.source)
                ),
                "",
            )
            glossary_rows.append(row)
        glossary_json = json.dumps(glossary_rows, ensure_ascii=False)
        traces: list[dict[str, Any]] = [
            planning_input_trace,
            {"step": "translation_plan", "provenance": provenance}, *plan_repairs,
        ]
        for offset in range(0, len(cues), 45):
            chunk = cues[offset:offset + 45]
            expected = {item.id for item in chunk}
            translations: dict[str, str] = {}
            last_invalid = ""
            for attempt in range(3):
                pending = [item for item in chunk if item.id not in translations]
                chunk_payload = [
                    {"id": item.id, "start": item.start, "end": item.end, "source": item.source_text}
                    for item in pending
                ]
                prompt = "\n".join([
                    "Translate these English transcript cues into concise, natural Simplified Chinese subtitles for Chinese AI/software practitioners.",
                    "Translate meaning in chapter context, never English word order. Preserve uncertainty, negation, numbers, scope, and speaker attribution.",
                    "Voice: a technically fluent Chinese colleague explaining the speaker's point aloud. Prefer short subject-verb-object clauses and familiar spoken cause-and-effect such as '就、结果、可、只会'; choose everyday verbs such as '搬走、涨价、赶走、自己挖坑' when faithful. The result should sound written in Chinese, not translated from English or summarized as a formal report.",
                    "Translate launch/reveal intent from context: 'Here is/Here's [named product or version]' usually means the product has arrived, so write '[产品名] 来了' or '[产品名] 发布了', not the literal '这是 [产品名]'. Keep '这是' only when the speaker is genuinely identifying or pointing to an object.",
                    "Strict cue alignment: each Chinese text may translate only the source text carrying the same id. Never move meaning to the previous or next id, never finish a neighboring cue, and never borrow words from another cue. If a supplied source is a fragment, keep the Chinese fragment aligned instead of completing it from context.",
                    "Translate the substantive meaning of every supplied cue. The English caption remains visible, so omit speech fillers from the Chinese line: do not translate standalone or discourse uses of 'um', 'uh', 'you know', 'I mean', or similar hesitation sounds. Also omit every non-speech stage direction such as '[clears throat]', '[coughs]', '[laughter]', '[music]', or '[applause]'; never render these as Chinese words. A filler-only or non-speech-only cue may have an empty Chinese text. Keep English when the glossary says preserve. For bilingual_once, add the exact first_use_explanation only in first_use_cue_id; all later occurrences keep English without repeating the explanation.",
                    "For every glossary row marked translate, use its exact Chinese target in each corresponding cue and do not leave the English source word in the Chinese subtitle. Ordinary established words such as chips and applications are Chinese copy here, not product identifiers.",
                    "Return {translations:[{id,text}]}; return every supplied id exactly once and no extra ids.",
                    (
                        "The previous response was rejected: " + last_invalid
                        + ". Copy the supplied id strings exactly."
                    ) if last_invalid else "",
                    "Glossary: " + glossary_json,
                    "Cues: " + json.dumps(chunk_payload, ensure_ascii=False),
                ])
                draft, chunk_provenance = self.writer._request_json([
                    {"role": "system", "content": "Return one valid JSON object only."},
                    {"role": "user", "content": prompt},
                ], max_tokens=6000)
                returned = {
                    str(item.get("id")): str(item.get("text") or "").strip()
                    for item in draft.get("translations", []) if isinstance(item, dict)
                }
                pending_ids = {item.id for item in pending}
                extra = sorted(set(returned) - pending_ids)
                filler_only_ids = {
                    item.id for item in pending if source_is_omittable_caption_only(item.source_text)
                }
                empty = sorted(
                    key for key, value in returned.items()
                    if not value and key not in filler_only_ids
                )
                if extra or empty:
                    last_invalid = f"empty={empty}, extra={extra}"
                    traces.append({
                        "step": "translate_chunk_rejected", "offset": offset,
                        "attempt": attempt + 1, "reason": last_invalid,
                        "provenance": chunk_provenance,
                    })
                    continue
                translations.update(returned)
                traces.append({
                    "step": "translate_chunk", "offset": offset, "attempt": attempt + 1,
                    "requested": len(pending), "returned": len(returned),
                    "provenance": chunk_provenance,
                })
                if set(translations) == expected:
                    break
                if not returned:
                    break
            if set(translations) != expected:
                missing = sorted(expected - set(translations))
                raise ValueError(f"translation response ids mismatch after retries; missing={missing}")
            for item in chunk:
                item.translation = translations[item.id]
        for _ in range(3):
            balanced = rebalance_translated_cues(cues)
            if len(balanced) == len(cues):
                break
            cues[:] = balanced
        errors = terminology_contract_errors(cues, terminology)
        if errors:
            repair_trace = self._repair_terminology(cues, terminology, errors)
            traces.append(repair_trace)
            enforced = self._enforce_terminology_contract(cues, terminology)
            if enforced:
                traces.append({"step": "deterministic_terminology_enforcement", "terms": enforced})
            for _ in range(2):
                balanced = rebalance_translated_cues(cues)
                if len(balanced) == len(cues):
                    break
                cues[:] = balanced
            errors = terminology_contract_errors(cues, terminology)
            if errors:
                # Rebalancing can move a translated term into the neighboring
                # visual card even though the English occurrence stays here.
                # Repair the final card boundaries, not only the pre-split
                # translation, so each bilingual card remains aligned.
                post_balance_repair = self._repair_terminology(
                    cues, terminology, errors,
                )
                traces.append({
                    **post_balance_repair,
                    "step": "post_rebalance_terminology_repair",
                })
                enforced = self._enforce_terminology_contract(cues, terminology)
                if enforced:
                    traces.append({
                        "step": "post_rebalance_terminology_enforcement",
                        "terms": enforced,
                    })
                errors = terminology_contract_errors(cues, terminology)
            if errors:
                raise ValueError("; ".join(errors))
        if dense := fast_translation_cues(cues):
            traces.append(self._repair_reading_speed(
                dense, terminology=terminology, all_cues=cues,
            ))
            if remaining := fast_translation_cues(cues):
                raise ValueError(
                    "subtitle reading speed repair failed: "
                    + ", ".join(cue.id for cue in remaining[:10])
                )
            term_errors = terminology_contract_errors(cues, terminology)
            if term_errors:
                post_density_repair = self._repair_terminology(
                    cues, terminology, term_errors,
                )
                traces.append({
                    **post_density_repair,
                    "step": "post_density_terminology_repair",
                })
                enforced = self._enforce_terminology_contract(cues, terminology)
                if enforced:
                    traces.append({
                        "step": "post_density_terminology_enforcement",
                        "terms": enforced,
                    })
                term_errors = terminology_contract_errors(cues, terminology)
                if term_errors:
                    raise ValueError("; ".join(term_errors))
        return terminology, plan, traces

    def _repair_reading_speed(
        self, cues: list[TranscriptCue], max_chars_per_second: float = 12.0,
        terminology: list[TerminologyEntry] | None = None,
        all_cues: list[TranscriptCue] | None = None,
    ) -> dict[str, Any]:
        first_use_ids = {
            entry.source: next((
                cue.id for cue in (all_cues or cues)
                if _contains_term(cue.source_text, entry.source)
            ), "")
            for entry in (terminology or [])
        }
        payload = [{
            "id": cue.id,
            "source": cue.source_text,
            "current_translation": cue.translation,
            "duration_seconds": round(cue.duration, 3),
            "maximum_visible_characters": max(
                4, math.floor(cue.duration * max_chars_per_second),
            ),
            "required_terms": [
                entry.source for entry in (terminology or [])
                if entry.strategy in {
                    TerminologyStrategy.PRESERVE,
                    TerminologyStrategy.BILINGUAL_ONCE,
                } and _contains_term(cue.source_text, entry.source)
            ],
            "required_first_use_explanations": [
                entry.first_use_explanation for entry in (terminology or [])
                if entry.strategy == TerminologyStrategy.BILINGUAL_ONCE
                and entry.first_use_explanation
                and _contains_term(cue.source_text, entry.source)
                and first_use_ids.get(entry.source) == cue.id
            ],
        } for cue in cues]
        prompt = "\n".join([
            "Shorten only these Simplified Chinese interview subtitles so each can be read in its source time.",
            "Keep every factual claim, number, negation, named entity, and English technical term, but remove filler, repeated phrasing, and translated hesitation. Use concise natural spoken Chinese, not fragments copied from English word order.",
            "The visible Chinese character count after removing whitespace must not exceed maximum_visible_characters.",
            "Every string in required_terms must remain exactly in English in that cue.",
            "Every string in required_first_use_explanations must also remain exactly once in that cue.",
            "Return {translations:[{id,text}]}; return every supplied id exactly once and no extra ids.",
            "Cues: " + json.dumps(payload, ensure_ascii=False),
        ])
        expected = {cue.id for cue in cues}
        feedback = ""
        attempts: list[dict[str, Any]] = []
        translations: dict[str, str] = {}
        for attempt in range(3):
            request = prompt + feedback
            draft, provenance = self.writer._request_json([
                {"role": "system", "content": "Return one valid JSON object only."},
                {"role": "user", "content": request},
            ], max_tokens=2500)
            attempts.append(provenance)
            translations = {
                str(item.get("id")): str(item.get("text") or "").strip()
                for item in draft.get("translations", []) if isinstance(item, dict)
            }
            invalid: list[str] = []
            if set(translations) != expected or any(
                not translations.get(cue_id, "") for cue_id in expected
            ):
                invalid.append("return every required id with non-empty text")
            for cue in cues:
                proposed = translations.get(cue.id, "")
                limit = max(4, math.floor(cue.duration * max_chars_per_second))
                count = len(re.sub(r"\s+", "", proposed))
                if proposed and count > limit:
                    invalid.append(f"{cue.id} has {count} characters; maximum is {limit}")
                required = [
                    entry.source for entry in (terminology or [])
                    if entry.strategy in {
                        TerminologyStrategy.PRESERVE,
                        TerminologyStrategy.BILINGUAL_ONCE,
                    } and _contains_term(cue.source_text, entry.source)
                ]
                missing = [term for term in required if not _contains_term(proposed, term)]
                if proposed and missing:
                    invalid.append(
                        f"{cue.id} is missing required terms: {', '.join(missing)}"
                    )
                explanations = [
                    entry.first_use_explanation for entry in (terminology or [])
                    if entry.strategy == TerminologyStrategy.BILINGUAL_ONCE
                    and entry.first_use_explanation
                    and _contains_term(cue.source_text, entry.source)
                    and first_use_ids.get(entry.source) == cue.id
                ]
                invalid_explanations = [
                    explanation for explanation in explanations
                    if proposed.count(explanation) != 1
                ]
                if proposed and invalid_explanations:
                    invalid.append(
                        f"{cue.id} must keep each first-use explanation exactly once: "
                        + ", ".join(invalid_explanations)
                    )
            if not invalid:
                break
            feedback = (
                "\nYour previous response failed the deterministic limits: "
                + "; ".join(invalid)
                + ". Rewrite again, shorter, without returning commentary."
            )
        else:
            raise ValueError("subtitle reading speed repair failed after 3 attempts: " + feedback.strip())
        for cue in cues:
            cue.translation = translations[cue.id]
        return {
            "step": "subtitle_reading_speed_repair",
            "cue_ids": sorted(expected), "attempts": attempts,
        }

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

    def repair_merged_interview_translations(
        self, before: list[TranscriptCue], merged: list[TranscriptCue],
        terminology: list[TerminologyEntry],
    ) -> dict[str, Any] | None:
        """Retranslate only cached cues that became complete semantic phrases."""
        original_by_id = {cue.id: cue for cue in before}
        targets = [
            cue for cue in merged
            if cue.id in original_by_id
            and cue.source_text.strip() != original_by_id[cue.id].source_text.strip()
        ]
        if not targets:
            return None
        if len(targets) > 8:
            batch_traces: list[dict[str, Any]] = []
            for offset in range(0, len(targets), 8):
                batch_trace = self.repair_merged_interview_translations(
                    before, targets[offset:offset + 8], terminology,
                )
                if batch_trace:
                    batch_traces.append(batch_trace)
            return {
                "step": "merged_interview_translation_repair",
                "cue_ids": [cue.id for cue in targets],
                "batches": batch_traces,
            }
        glossary = [
            {"source": item.source, "strategy": item.strategy.value, "target": item.target}
            for item in terminology
        ]
        payload = [{
            "id": cue.id, "start": cue.start, "end": cue.end,
            "source": cue.source_text,
        } for cue in targets]
        last_error = ""
        copy_writer = getattr(self.writer, "fallback", self.writer)
        for attempt in range(2):
            draft, provenance = copy_writer._request_json([
                {"role": "system", "content": "Return one valid JSON object only."},
                {"role": "user", "content": "\n".join([
                    "Rewrite merged interview captions into concise, natural Simplified Chinese.",
                    "Audience: Chinese developers and technology practitioners. Write how a fluent colleague would explain the spoken sentence; do not mirror English word order or preserve awkward seams from old fragments.",
                    "Each returned caption is shown alone, so make it a self-contained spoken Chinese clause with a complete subject-predicate or action-result structure. Reorder the source naturally. Omit false starts, repeated filler, and a trailing connector that only leads into the next cue; never leave a noun such as '这个概念' or a setup such as '这就是我们' dangling at the end.",
                    "Translate the complete substantive meaning once. Preserve product names, code, APIs, and technical terms in English when that is more natural. Do not add facts or commentary.",
                    "Return JSON as {translations:[{id,text}]}; return every supplied id exactly once.",
                    "Terminology: " + json.dumps(glossary, ensure_ascii=False),
                    "Cues: " + json.dumps(payload, ensure_ascii=False),
                    ("Previous validation error: " + last_error) if last_error else "",
                ])},
            ], max_tokens=2500)
            rows = draft.get("translations")
            translated = {
                str(row.get("id") or ""): str(row.get("text") or "").strip()
                for row in rows if isinstance(row, dict)
            } if isinstance(rows, list) else {}
            expected = {cue.id for cue in targets}
            if set(translated) != expected or any(not value for value in translated.values()):
                last_error = "return every requested id exactly once with non-empty text"
                continue
            review_provenance: dict[str, Any] | None = None
            if copy_writer is not None:
                review_messages = [
                    {"role": "system", "content": "Return one valid JSON object only."},
                    {"role": "user", "content": "\n".join([
                        "You are the final read-aloud editor for Chinese subtitles in a technology interview.",
                        "Review every proposed card independently. A viewer must understand each card without reading the previous or next card. Rewrite awkward English-order Chinese, remove duplicated filler, and finish every card on a complete Chinese clause. If the English source trails into the next cue, preserve its substantive meaning but omit that unfinished connector instead of leaving the Chinese hanging.",
                        "Keep established English technical terms natural. Do not add facts, conclusions, emphasis, or commentary.",
                        "Return JSON as {translations:[{id,text}]}; return every id exactly once.",
                        "Rows: " + json.dumps([
                            {
                                "id": cue.id, "source": cue.source_text,
                                "proposed_chinese": translated[cue.id],
                            } for cue in targets
                        ], ensure_ascii=False),
                        ("Previous validation error: " + last_error) if last_error else "",
                    ])},
                ]
                reviewed, review_provenance = copy_writer._request_json(
                    review_messages, max_tokens=2500,
                )
                reviewed_rows = reviewed.get("translations")
                reviewed_translations = {
                    str(row.get("id") or ""): str(row.get("text") or "").strip()
                    for row in reviewed_rows if isinstance(row, dict)
                } if isinstance(reviewed_rows, list) else {}
                if (
                    set(reviewed_translations) != expected
                    or any(not value for value in reviewed_translations.values())
                ):
                    last_error = "independent read-aloud review omitted one or more cue ids"
                    continue
                translated = reviewed_translations
            incomplete = [
                cue.id for cue in targets
                if not re.search(r"[。！？”’.!?]$", translated[cue.id].strip())
                or re.search(
                    r"(?:这个概念|这个过程|这就是我们|所有)$",
                    translated[cue.id].strip(),
                )
            ]
            if incomplete:
                last_error = (
                    "these standalone cards still end on an unfinished setup instead of a complete "
                    "spoken Chinese sentence: " + ", ".join(incomplete)
                )
                continue
            too_fast = [
                cue.id for cue in targets
                if len(re.sub(r"\s+", "", translated[cue.id])) / max(cue.duration, 0.1) > 12
            ]
            if too_fast:
                last_error = "translations exceed 12 visible characters per second: " + ", ".join(too_fast)
                continue
            for cue in targets:
                cue.translation = translated[cue.id]
            return {
                "step": "merged_interview_translation_repair",
                "cue_ids": sorted(expected), "attempt": attempt + 1,
                "provenance": provenance,
                "review_provenance": review_provenance,
            }
        raise ValueError("merged interview translation repair failed: " + last_error)

    def segment_interview_subtitle_cards(
        self, cues: list[TranscriptCue], terminology: list[TerminologyEntry],
    ) -> dict[str, Any] | None:
        """Turn dense interview cues into speech-timed, semantic clause cards.

        Source boundaries are chosen deterministically and remain a lossless,
        ordered partition.  A Chinese-language writer translates those fixed
        spans; an independent reviewer may reject fidelity problems but never
        rewrites text or changes boundaries.
        """
        _retime_existing_semantic_card_groups(cues)
        plans: dict[str, tuple[TranscriptCue, list[str]]] = {}
        for cue in cues:
            # A stale cached plan may already contain ``-card-*`` rows from an
            # older policy. Keep compliant rows unchanged, but allow an
            # overlong/dense legacy card to become the parent of newly
            # translated fixed spans. Skipping every card here would make an
            # obsolete cache impossible to migrate without returning to the
            # coarse pre-review transcript.
            stale_card_errors = (
                interview_caption_duration_errors([cue])
                if "-card-" in cue.id else []
            )
            if "-card-" in cue.id and not stale_card_errors:
                continue
            desired = _desired_subtitle_card_count(
                cue.source_text, cue.translation, cue.duration,
            )
            parts = _semantic_english_parts(cue.source_text, desired)
            parts = _coalesce_short_semantic_parts(parts, cue.duration)
            parts = _split_overlong_semantic_parts(parts, cue.duration)
            if len(parts) <= 1:
                if stale_card_errors:
                    # Text-density failures can often be repaired by a concise
                    # card-specific translation without changing the already
                    # reviewed English span. Timing/English failures remain
                    # visible to the final gate if no semantic split exists.
                    plans[cue.id] = (cue, [cue.source_text])
                continue
            if re.sub(r"\s+", " ", " ".join(parts)).strip() != re.sub(
                r"\s+", " ", cue.source_text,
            ).strip():
                raise ValueError(f"semantic subtitle split lost source text for {cue.id}")
            plans[cue.id] = (cue, parts)
        if not plans:
            return {
                "step": "interview_semantic_subtitle_cards",
                "cue_ids": [],
                "reviewed_cue_ids": [cue.id for cue in cues],
                "policy_version": INTERVIEW_CAPTION_POLICY_VERSION,
                "policy_fingerprint": INTERVIEW_CAPTION_POLICY_FINGERPRINT,
                "fallback_used": False,
                "fallback_reason": "",
                "provenance": None,
                "translation_attempt_provenance": [],
                "fidelity_review_provenance": None,
            }

        requested: list[dict[str, Any]] = []
        for cue_id, (cue, parts) in plans.items():
            durations = _allocate_caption_durations(parts, cue.duration)
            for index, (source_part, part_duration) in enumerate(
                zip(parts, durations), start=1,
            ):
                requested.append({
                    "id": f"{cue_id}-card-{index}",
                    "parent_id": cue_id,
                    "source": source_part,
                    "full_source_context": cue.source_text,
                    "duration_seconds": round(part_duration, 3),
                })
        expected = {row["id"] for row in requested}
        translations: dict[str, str] = {}
        provenance: dict[str, Any] | None = None
        fidelity_review_provenance: dict[str, Any] | None = None
        translation_writer = (
            self.writer if hasattr(self.writer, "_request_json")
            else self.directing_writer
        )
        fidelity_reviewer = (
            self.subtitle_reviewer or self.directing_writer
        )
        fidelity_reviewer = (
            fidelity_reviewer
            if fidelity_reviewer is not None
            and fidelity_reviewer is not translation_writer
            else None
        )
        error = ""
        locked_translations: dict[str, str] = {}
        translation_attempt_provenance: list[dict[str, Any]] = []
        if translation_writer is not None:
            # Translation remains one model's responsibility.  The directing
            # critic audits the hook only; using it as a third translator made
            # provenance ambiguous and often replaced idiomatic Chinese with a
            # more literal draft.  Three bounded Kimi attempts are cheaper on
            # the coding plan and each retry contains only rejected ids.
            attempt_writers = [translation_writer] * 3
            for active_writer in attempt_writers:
                try:
                    active_requested = [
                        row for row in requested
                        if str(row["id"]) not in locked_translations
                    ]
                    active_expected = {str(row["id"]) for row in active_requested}
                    if not active_requested:
                        translations = dict(locked_translations)
                        break
                    draft, provenance = active_writer._request_json([
                        {"role": "system", "content": "Return one valid JSON object only."},
                        {"role": "user", "content": "\n".join([
                            "You are the Chinese subtitle translator for a technology interview. Write concise, idiomatic spoken Simplified Chinese, as a technically informed Chinese colleague would naturally explain the same utterance aloud.",
                            "The English source field is authoritative and its card boundaries are fixed. Preserve actors, actions, numbers, entities, negation, uncertainty, timing, and cause-effect. Do not move meaning between ids or add a stronger conclusion, opportunity, or explanation. Parent context may only resolve an elided subject, referent, or obvious ASR name.",
                            "Remove hesitation sounds and repeated filler. Interview ASR may contain duplicated words, broken repetitions, or terse domain shorthand: recover the clear meaning from this card plus parent context, omit transcription noise, and express the domain meaning naturally instead of translating corrupt word order or a shorthand noun literally. If a fixed source span is a discourse setup such as 'back to the ... point on X', write a complete topic-setting Chinese sentence that preserves X; never leave a dash fragment, and never borrow the next card's claim to complete it. Do not invent a relationship that the recoverable context does not support. Make each visible card a complete, easily understood spoken clause. Preserve established English product names, APIs, code, and identifiers; follow the supplied terminology exactly.",
                            f"Fit the wording to duration_seconds: target at most {INTERVIEW_CAPTION_TARGET_MAX_CHINESE_CHARACTERS} visible Chinese characters and never exceed {INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND:g} visible characters per second or the hard {INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS}-character ceiling. A short spoken phrase needs a short Chinese line, not an expanded explanation. End each card with Chinese punctuation. Return every id exactly once as {{translations:[{{id,text}}]}}.",
                            "Terminology: " + json.dumps([
                                {
                                    "source": item.source,
                                    "strategy": item.strategy.value,
                                    "target": item.target,
                                } for item in terminology
                            ], ensure_ascii=False),
                            "Cards: " + json.dumps(active_requested, ensure_ascii=False),
                            ("Previous validation error: " + error) if error else "",
                        ])},
                    ], max_tokens=5000)
                    translation_attempt_provenance.append(provenance)
                    rows = draft.get("translations")
                    batch_translations = {
                        str(row.get("id") or ""): str(row.get("text") or "").strip()
                        for row in rows if isinstance(row, dict)
                    } if isinstance(rows, list) else {}
                    unexpected_ids = set(batch_translations) - active_expected
                    if unexpected_ids - set(locked_translations):
                        error = (
                            "translation returned unrequested card ids: "
                            + ", ".join(sorted(unexpected_ids))
                        )
                        translations = dict(locked_translations)
                        continue
                    # Some models repeat an already accepted row on a focused
                    # retry. Ignore that harmless echo without allowing it to
                    # overwrite independently reviewed copy.
                    batch_translations = {
                        card_id: text_value
                        for card_id, text_value in batch_translations.items()
                        if card_id in active_expected
                    }
                    for card_id, text_value in list(batch_translations.items()):
                        if not text_value:
                            continue
                        if text_value.endswith(("，", "；", ",", ";")):
                            text_value = text_value[:-1].rstrip() + "。"
                        elif not re.search(r"[。！？!?]$", text_value):
                            text_value += "。"
                        batch_translations[card_id] = text_value
                    translations = {**locked_translations, **batch_translations}
                    invalid = [
                        card_id for card_id in active_expected
                        if not translations.get(card_id)
                        or len(re.sub(r"\s+", "", translations[card_id]))
                        > INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS
                    ]
                    requested_by_id = {row["id"]: row for row in requested}
                    for card_id in active_expected:
                        text_value = translations.get(card_id, "")
                        source_value = str(requested_by_id[card_id]["source"])
                        duration_value = float(
                            requested_by_id[card_id].get("duration_seconds") or 0.1
                        )
                        if (
                            len(re.sub(r"\s+", "", text_value))
                            / max(duration_value, 0.1)
                            > INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND
                        ):
                            invalid.append(card_id)
                        if _caption_entity_alignment_errors(source_value, text_value):
                            invalid.append(card_id)
                        if (
                            _contains_term(source_value, "maybe")
                            and not re.search(r"可能|也许|或许", text_value)
                        ):
                            invalid.append(card_id)
                        for term in terminology:
                            if (
                                term.strategy != TerminologyStrategy.TRANSLATE
                                or not term.target
                                or not _contains_term(source_value, term.source)
                            ):
                                continue
                            if (
                                not _translated_term_present(
                                    term, TranscriptCue(
                                        id=card_id, start=0,
                                        end=duration_value,
                                        source_text=source_value,
                                        translation=text_value,
                                    ),
                                )
                                or _contains_term(text_value, term.source)
                            ):
                                invalid.append(card_id)
                    if set(batch_translations) != active_expected:
                        error = (
                            "missing, extra, or empty cards: "
                            + ", ".join(sorted(set(invalid)))
                        )
                        translations = dict(locked_translations)
                        continue
                    invalid_set = set(invalid)
                    reviewable_ids = active_expected - invalid_set
                    if not reviewable_ids:
                        error = (
                            "missing, overlong, dependent, or terminology-invalid cards: "
                            + ", ".join(sorted(invalid_set))
                        )
                        translations = dict(locked_translations)
                        continue
                    rejected_set: set[str] = set()
                    if fidelity_reviewer is not None:
                        try:
                            reviewed, fidelity_review_provenance = fidelity_reviewer._request_json([
                                    {"role": "system", "content": "Return one valid JSON object only."},
                                    {"role": "user", "content": "\n".join([
                                        "You are an independent bilingual fidelity and spoken-Chinese auditor. Do not rewrite any caption. A complete topic-setting Chinese sentence is a faithful rendering of an English discourse setup such as 'back to the ... point on X' when it preserves X and does not import the following card's claim.",
                                        "For every fixed English/Chinese pair, score fidelity and naturalness separately from 1 to 5. Fidelity covers actor, action, numbers, entities, negation, uncertainty, timing, cause-effect, omissions, and unsupported additions. Do not penalize omission of hesitation sounds, stutters, or duplicated ASR tokens when substantive meaning is preserved. Do not call a source term untranslated when the supplied terminology explicitly marks it preserve; judge whether the surrounding Chinese is natural instead. When the English ASR is broken, pass only if the Chinese collapses the recoverable context into one coherent proposition; reject Chinese that preserves disconnected transcript debris, unclear parallel fragments, or confidently guesses an ambiguous technical word without contextual support. Naturalness means publication-ready spoken Chinese that a viewer understands on first reading; reject literal English word order, untranslated ordinary nouns not marked preserve, awkward duplicated wording, broken clauses, dangling connective openings, and surface translation of a metaphor when Chinese normally expresses its technical meaning directly.",
                                        "Set pass=true only when both scores are at least 4 and you would publish the exact Chinese unchanged. A merely understandable but avoidably awkward caption must fail. Ignore only genuinely minor preference. Return every id exactly once as {reviews:[{id,pass,fidelity_score,naturalness_score,errors}]}. errors is a short list; provide no corrected text.",
                                        "Terminology: " + json.dumps([
                                            {
                                                "source": item.source,
                                                "strategy": item.strategy.value,
                                                "target": item.target,
                                            } for item in terminology
                                        ], ensure_ascii=False),
                                        "Rows: " + json.dumps([
                                            {
                                                "id": row["id"], "source": row["source"],
                                                "parent_context": row["full_source_context"],
                                                "chinese": translations[row["id"]],
                                            } for row in active_requested
                                            if str(row["id"]) in reviewable_ids
                                        ], ensure_ascii=False),
                                    ])},
                                ], max_tokens=5000)
                            review_rows = reviewed.get("reviews")
                            review_values = {
                                str(row.get("id") or ""): row
                                for row in review_rows if isinstance(row, dict)
                            } if isinstance(review_rows, list) else {}
                            rejected_set = {
                                card_id for card_id in reviewable_ids
                                if card_id not in review_values
                                or review_values[card_id].get("pass") is not True
                                or _review_score_out_of_five(
                                    review_values[card_id].get("fidelity_score")
                                ) < 4
                                or _review_score_out_of_five(
                                    review_values[card_id].get("naturalness_score")
                                ) < 4
                            }
                        except Exception as exc:
                            # Once an independent reviewer is configured,
                            # transport or schema failure cannot silently
                            # turn an unreviewed draft into publishable copy.
                            fidelity_review_provenance = None
                            error = (
                                "fidelity reviewer failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            translations = dict(locked_translations)
                            continue
                    locked_translations.update({
                        card_id: translations[card_id]
                        for card_id in reviewable_ids - rejected_set
                    })
                    if invalid_set or rejected_set:
                        problems: list[str] = []
                        if invalid_set:
                            problems.append(
                                "missing, overlong, dependent, or terminology-invalid cards: "
                                + ", ".join(sorted(invalid_set))
                            )
                        if rejected_set:
                            problems.append("fidelity reviewer rejected: " + json.dumps({
                                card_id: {
                                    "fidelity_score": review_values.get(card_id, {}).get("fidelity_score"),
                                    "naturalness_score": review_values.get(card_id, {}).get("naturalness_score"),
                                    "errors": review_values.get(card_id, {}).get("errors", []),
                                }
                                for card_id in sorted(rejected_set)
                            }, ensure_ascii=False))
                        error = "; ".join(problems)
                        translations = dict(locked_translations)
                        continue
                    translations = dict(locked_translations)
                    if set(translations) == expected:
                        break
                    continue
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    translations = dict(locked_translations)

        if set(translations) != expected and fidelity_reviewer is not None:
            missing = sorted(expected - set(translations))
            raise ValueError(
                "reviewed subtitle translation exhausted bounded writer and fallback attempts: "
                + ", ".join(missing)
                + ("; " + error if error else "")
            )

        if set(translations) != expected and translation_writer is not None:
            # A model can treat a multi-row request as one paragraph and move
            # meaning between otherwise fixed ids. Recover card-by-card so the
            # model has no neighboring output slot into which it can shift a
            # company, action, or payoff.
            isolated: dict[str, str] = dict(translations)
            isolated_provenance: list[dict[str, Any]] = []
            isolated_failures: list[str] = []
            isolated_writers = []
            for writer in (
                translation_writer, getattr(self.writer, "fallback", None),
            ):
                if writer is not None and all(
                    writer is not existing for existing in isolated_writers
                ):
                    isolated_writers.append(writer)
            failed_parent_ids: set[str] = set()
            for row in requested:
                card_id = str(row["id"])
                if card_id in isolated:
                    continue
                duration_value = float(row.get("duration_seconds") or 0.1)
                maximum_characters = max(6, min(
                    INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS,
                    math.floor(
                        duration_value
                        * INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND
                    ),
                ))
                accepted = ""
                for isolated_writer in isolated_writers:
                    try:
                        draft, card_provenance = isolated_writer._request_json([
                            {"role": "system", "content": "Return one valid JSON object only."},
                            {"role": "user", "content": "\n".join([
                                "Translate exactly one fixed English interview caption into concise, natural spoken Simplified Chinese.",
                                "Translate only Source. Parent context may resolve an elided subject or predicate, but you must not import a neighboring fact, entity, action, or conclusion. The Chinese card must stand alone.",
                                "Preserve timing, modality, negation, numbers, and named entities. Normalize an obvious ASR spelling only when the parent context identifies it unambiguously. Remove fillers.",
                                "Use direct, idiomatic, standalone spoken Chinese rather than mirroring English word order.",
                                f"Use at most {maximum_characters} visible characters including punctuation. Return {{text:string}} only.",
                                "Terminology: " + json.dumps([
                                    {
                                        "source": item.source,
                                        "strategy": item.strategy.value,
                                        "target": item.target,
                                    } for item in terminology
                                ], ensure_ascii=False),
                                "Source: " + str(row["source"]),
                                "Parent context: " + str(row["full_source_context"]),
                            ])},
                        ], max_tokens=600)
                        candidate = str(draft.get("text") or "").strip()
                        if candidate and not re.search(r"[。！？!?]$", candidate):
                            candidate += "。"
                        card_errors = _semantic_card_translation_errors(
                            row, candidate, terminology,
                        )
                        if card_errors:
                            isolated_failures.append(
                                f"{card_id}:{','.join(card_errors)}"
                            )
                            continue
                        accepted = candidate
                        isolated_provenance.append({
                            "id": card_id, "provenance": card_provenance,
                        })
                        break
                    except Exception as exc:
                        isolated_failures.append(
                            f"{card_id}:{type(exc).__name__}:{exc}"
                        )
                if not accepted:
                    failed_parent_ids.add(str(row["parent_id"]))
                    continue
                isolated[card_id] = accepted
            if failed_parent_ids:
                isolated = {
                    card_id: text_value
                    for card_id, text_value in isolated.items()
                    if next(
                        str(row["parent_id"]) for row in requested
                        if str(row["id"]) == card_id
                    ) not in failed_parent_ids
                }
            provenance = {
                "mode": "isolated_card_recovery",
                "cards": isolated_provenance,
                "failures": isolated_failures,
                "failed_parent_ids": sorted(failed_parent_ids),
            }
            if isolated:
                translations = isolated

        fallback_used = any(
            any(
                f"{cue_id}-card-{index}" not in translations
                for index in range(1, len(parts) + 1)
            )
            for cue_id, (_, parts) in plans.items()
        )
        # A pre-existing Chinese paragraph is never repartitioned to fill
        # missing card translations. Clause-shaped target slices are not proof
        # that they align with fixed English spans. Keep the parent intact; the
        # policy gate below will fail closed when that parent is too dense.

        expanded: list[TranscriptCue] = []
        segmented_ids: list[str] = []
        for cue in cues:
            plan = plans.get(cue.id)
            if plan is None:
                expanded.append(cue)
                continue
            _, parts = plan
            card_ids = [f"{cue.id}-card-{index}" for index in range(1, len(parts) + 1)]
            if any(card_id not in translations for card_id in card_ids):
                expanded.append(cue)
                continue
            durations = _allocate_caption_durations(parts, cue.duration)
            elapsed = 0.0
            for index, (source_part, part_duration, card_id) in enumerate(
                zip(parts, durations, card_ids), start=1,
            ):
                start = cue.start + elapsed
                elapsed += part_duration
                end = cue.end if index == len(parts) else cue.start + elapsed
                expanded.append(TranscriptCue(
                    id=card_id, start=round(start, 3), end=round(end, 3),
                    source_text=source_part,
                    translation=translations[card_id], speaker=cue.speaker,
                    confidence=cue.confidence,
                ))
                segmented_ids.append(card_id)
        cues[:] = expanded
        self._enforce_terminology_contract(cues, terminology)
        if term_errors := terminology_contract_errors(cues, terminology):
            raise ValueError(
                "semantic subtitle cards violate terminology contract: "
                + "; ".join(term_errors)
            )
        return {
            "step": "interview_semantic_subtitle_cards",
            "cue_ids": segmented_ids,
            "reviewed_cue_ids": [cue.id for cue in cues],
            "policy_version": INTERVIEW_CAPTION_POLICY_VERSION,
            "policy_fingerprint": INTERVIEW_CAPTION_POLICY_FINGERPRINT,
            "fallback_used": fallback_used,
            "fallback_reason": error if fallback_used else "",
            "provenance": provenance,
            "translation_attempt_provenance": translation_attempt_provenance,
            "fidelity_review_provenance": fidelity_review_provenance,
        }

    def repair_audio_verified_cards(
        self, cues: list[TranscriptCue], terminology: list[TerminologyEntry],
        audit: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Retranslate only cards whose YouTube ASR was corrected from audio."""
        corrections = [
            row for row in audit.get("corrections", [])
            if isinstance(row, dict) and str(row.get("verified_source") or "").strip()
            and ("changed" not in row or row.get("changed") is True)
        ]
        if not corrections:
            return None
        merged_fragments: list[dict[str, str]] = []
        retained_corrections: list[dict[str, Any]] = []
        for row in corrections:
            card_id = str(row.get("cue_id") or "")
            verified = str(row.get("verified_source") or "").strip()
            index = next((
                position for position, cue in enumerate(cues) if cue.id == card_id
            ), -1)
            parent = card_id.rsplit("-card-", 1)[0]
            can_merge = (
                index >= 0
                and index + 1 < len(cues)
                and cues[index + 1].id.rsplit("-card-", 1)[0] == parent
                and cues[index].duration <= 3.0
                and bool(re.search(
                    r"\b(?:tends?|seems?)\s+to\s+be[,.\s]*$", verified,
                    re.IGNORECASE,
                ))
            )
            if not can_merge:
                retained_corrections.append(row)
                continue
            current = cues[index]
            following = cues[index + 1]
            following.start = current.start
            following.original_start = current.original_start
            following.source_text = (
                verified.rstrip(" ,.") + " " + following.source_text.lstrip()
            ).strip()
            del cues[index]
            merged_fragments.append({
                "removed_cue_id": card_id,
                "retained_cue_id": following.id,
            })
        corrections = retained_corrections
        if not corrections:
            return {
                "step": "audio_verified_caption_repair",
                "cue_ids": [],
                "merged_fragments": merged_fragments,
                "translation_attempts": [],
                "review_attempts": [],
            }
        by_id = {cue.id: cue for cue in cues}
        requested = [{
            "id": str(row["cue_id"]),
            "youtube_source": str(row.get("source") or ""),
            "audio_verified_source": str(row["verified_source"]).strip(),
            "duration_seconds": round(by_id[str(row["cue_id"])].duration, 3),
            "verification_reasons": list(row.get("reasons") or []),
        } for row in corrections if str(row.get("cue_id") or "") in by_id]
        if not requested:
            return None
        writer = self.writer if hasattr(self.writer, "_request_json") else None
        reviewer = self.subtitle_reviewer or self.directing_writer
        if writer is None or reviewer is None or reviewer is writer:
            raise RuntimeError(
                "audio-verified captions require separate translator and reviewer"
            )
        expected = {row["id"] for row in requested}
        locked: dict[str, str] = {}
        attempts: list[dict[str, Any]] = []
        reviews: list[dict[str, Any]] = []
        error = ""
        for _ in range(2):
            active = [row for row in requested if row["id"] not in locked]
            if not active:
                break
            draft, provenance = writer._request_json([
                {"role": "system", "content": "Return one valid JSON object only."},
                {"role": "user", "content": "\n".join([
                    "Translate audio-verified technology interview captions into concise, natural spoken Simplified Chinese.",
                    "Reconcile both evidence fields. audio_verified_source is authoritative for corrected content words, spellings, and entities; youtube_source may restore a short preposition or function word that Whisper omitted when it is compatible with the audio evidence. Preserve recoverable meaning, modality, entities, numbers, and cause-effect while removing stutters and false starts. Restructure fragmentary speech naturally. If a complete clause is followed by disconnected repeated noun fragments, translate the complete clause and omit the ambiguous debris; never turn those nouns into a causal, locative, or explanatory relationship. More generally, if the relationship among fragments is unclear, translate only the clear proposition rather than inventing a relationship or mirroring broken English word order.",
                    "Translate domain meaning, not the nearest dictionary word. In infrastructure or industry discussion, English 'compute' normally means 算力; reserve 计算 for an actual calculation or machine-operation sense. When industry profit or value moves 'up the stack toward the application layer', express it naturally as profit or value moving toward the application layer, never as physical 堆栈/技术栈 movement. Preserve product and company names accurately.",
                    "Each result must be understandable on first reading, end with Chinese punctuation, stay within 42 visible characters, and fit duration_seconds at no more than 12 visible characters per second. Return every requested id exactly once as {translations:[{id,text}]}",
                    "Terminology: " + json.dumps([
                        {"source": item.source, "strategy": item.strategy.value, "target": item.target}
                        for item in terminology
                    ], ensure_ascii=False),
                    "Cards: " + json.dumps(active, ensure_ascii=False),
                    ("Previous audit error: " + error) if error else "",
                ])},
            ], max_tokens=1800)
            attempts.append(provenance)
            rows = draft.get("translations")
            translated = {
                str(row.get("id") or ""): str(row.get("text") or "").strip()
                for row in rows if isinstance(row, dict)
            } if isinstance(rows, list) else {}
            for card_id, text_value in list(translated.items()):
                if text_value and not re.search(r"[。！？!?]$", text_value):
                    translated[card_id] = text_value + "。"
            active_ids = {row["id"] for row in active}
            active_by_id = {row["id"]: row for row in active}
            invalid = [
                card_id for card_id in active_ids
                if not translated.get(card_id)
                or len(re.sub(r"\s+", "", translated[card_id])) > 42
                or len(re.sub(r"\s+", "", translated[card_id]))
                / max(by_id[card_id].duration, 0.1) > 12
                or _caption_entity_alignment_errors(
                    active_by_id[card_id]["audio_verified_source"],
                    translated[card_id],
                )
                or (
                    _contains_term(
                        active_by_id[card_id]["audio_verified_source"],
                        "up the stack",
                    )
                    and re.search(
                        r"(?:堆栈|技术栈).{0,6}(?:上移|向上)", translated[card_id],
                    )
                )
            ]
            if invalid or set(translated) != active_ids:
                error = "invalid audio-verified translations: " + ", ".join(
                    sorted(set(invalid) | (active_ids - set(translated)))
                )
                continue
            verdict, review_provenance = reviewer._request_json([
                {"role": "system", "content": "Return one valid JSON object only."},
                {"role": "user", "content": "\n".join([
                    "Audit each audio-verified English/Chinese subtitle pair. Do not rewrite it.",
                    "Use both evidence fields: audio_verified_source controls corrected content words, spellings, and entities; youtube_source may preserve a compatible short preposition or function word omitted by Whisper. Pass only publication-ready spoken Chinese that preserves the recoverable substantive meaning and is immediately natural to a technical Chinese viewer. Do not penalize omission of hesitation, false starts, duplicated words, or genuinely ambiguous transcript debris, and allow structural reordering required by natural Chinese. Reject lost clear claims, literal word order, ambiguous dictionary translation, wrong entities, invented relationships, or unsupported explanation. In an industry-profit context, reject physical 堆栈/技术栈 wording for 'up the stack toward the application layer'; it must convey profit or value moving toward the application layer. fidelity_score and naturalness_score must be integers from 1 to 5. Return {reviews:[{id,pass,fidelity_score,naturalness_score,errors}]} and every id exactly once.",
                    "Rows: " + json.dumps([{
                        "id": row["id"],
                        "youtube_source": row["youtube_source"],
                        "audio_verified_source": row["audio_verified_source"],
                        "chinese": translated[row["id"]],
                    } for row in active], ensure_ascii=False),
                ])},
            ], max_tokens=1400)
            reviews.append(review_provenance)
            review_rows = verdict.get("reviews")
            review_map = {
                str(row.get("id") or ""): row
                for row in review_rows if isinstance(row, dict)
            } if isinstance(review_rows, list) else {}
            rejected = [
                card_id for card_id in active_ids
                if card_id not in review_map
                or review_map[card_id].get("pass") is not True
                or _review_score_out_of_five(
                    review_map[card_id].get("fidelity_score")
                ) < 4
                or _review_score_out_of_five(
                    review_map[card_id].get("naturalness_score")
                ) < 4
            ]
            rejected_set = set(rejected)
            locked.update({
                card_id: translated[card_id]
                for card_id in active_ids - rejected_set
            })
            if rejected:
                error = "audio-verified reviewer rejected: " + json.dumps({
                    card_id: {
                        "chinese": translated.get(card_id, ""),
                        "pass": review_map.get(card_id, {}).get("pass"),
                        "fidelity_score": review_map.get(card_id, {}).get("fidelity_score"),
                        "naturalness_score": review_map.get(card_id, {}).get("naturalness_score"),
                        "errors": review_map.get(card_id, {}).get("errors", []),
                    }
                    for card_id in rejected
                }, ensure_ascii=False)
                continue
            break
        if set(locked) != expected:
            raise ValueError(
                "audio-verified caption repair failed closed: "
                + ", ".join(sorted(expected - set(locked)))
                + ("; " + error if error else "")
            )
        source_by_id = {
            row["id"]: row["audio_verified_source"] for row in requested
        }
        for card_id in expected:
            cue = by_id[card_id]
            cue.source_text = source_by_id[card_id]
            cue.translation = locked[card_id]
        self._enforce_terminology_contract(cues, terminology)
        return {
            "step": "audio_verified_caption_repair",
            "cue_ids": sorted(expected),
            "merged_fragments": merged_fragments,
            "translation_attempts": attempts,
            "review_attempts": reviews,
        }

    def repair_interview_chinese_style(
        self, cues: list[TranscriptCue], terminology: list[TerminologyEntry],
    ) -> dict[str, Any] | None:
        """Repair explicitly detected failures without globally rewriting good copy."""
        errors = interview_chinese_style_errors(cues)
        if not errors:
            return None
        by_id = {cue.id: cue for cue in cues}
        positions = {cue.id: index for index, cue in enumerate(cues)}
        payload = []
        for cue_id, reasons in errors.items():
            index = positions[cue_id]
            cue = by_id[cue_id]
            payload.append({
                "id": cue.id,
                "source": cue.source_text,
                "current_chinese": cue.translation,
                "style_errors": reasons,
                "previous_source": cues[index - 1].source_text if index else "",
                "next_source": cues[index + 1].source_text if index + 1 < len(cues) else "",
            })
        expected = set(errors)
        last_error = ""
        provenance: dict[str, Any] | None = None
        provider_failures: list[str] = []
        repair_writers = []
        for writer in (
            getattr(self.writer, "fallback", None),
            self.directing_writer, self.writer,
        ):
            if (
                writer is not None and hasattr(writer, "_request_json")
                and all(writer is not existing for existing in repair_writers)
            ):
                repair_writers.append(writer)
        if not repair_writers:
            return None
        for repair_writer in repair_writers:
            for _ in range(2):
                try:
                    draft, provenance = repair_writer._request_json([
                        {"role": "system", "content": "Return one valid JSON object only."},
                        {"role": "user", "content": "\n".join([
                            "Act as the final Chinese translation editor for these interview captions. Rewrite every row as fluent, compact Chinese that a viewer can understand while seeing that row alone; do not merely polish or preserve English sentence structure.",
                            "Preserve the English source's actor, action, timing, uncertainty, negation, numbers, entities, and cause-effect. Do not borrow a new fact from neighboring cards. When English begins with a conjunction or relative clause, use neighboring source only to recover the omitted subject, predicate, or referent, then write a complete standalone Chinese sentence rather than a continuation. Resolve obvious ASR spellings from context. For product-stack language such as 'up the stack toward the application layer', express the movement naturally in Chinese instead of translating 'up' as supply-chain upstream. Every result must end with punctuation, fit its speaking duration, "
                            f"and remain within {INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS} visible characters and {INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND:g} visible characters per second.",
                            "Return every id exactly once as {translations:[{id,text}]}; no commentary.",
                            "Terminology: " + json.dumps([
                                {
                                    "source": item.source, "strategy": item.strategy.value,
                                    "target": item.target,
                                    "first_use_explanation": item.first_use_explanation,
                                } for item in terminology
                            ], ensure_ascii=False),
                            "Rows: " + json.dumps(payload, ensure_ascii=False),
                            ("Previous validation error: " + last_error) if last_error else "",
                        ])},
                    ], max_tokens=2500)
                except Exception as error:
                    last_error = f"{type(error).__name__}: {error}"
                    provider_failures.append(last_error)
                    break
                rows = draft.get("translations")
                translated = {
                    str(row.get("id") or ""): str(row.get("text") or "").strip()
                    for row in rows if isinstance(row, dict)
                } if isinstance(rows, list) else {}
                if set(translated) != expected or any(not value for value in translated.values()):
                    last_error = "return every flagged id exactly once with non-empty text"
                    continue
                proposed = [TranscriptCue(
                    id=cue_id, start=by_id[cue_id].start, end=by_id[cue_id].end,
                    source_text=by_id[cue_id].source_text,
                    translation=(
                        translated[cue_id]
                        if re.search(r"[。！？!?]$", translated[cue_id])
                        else translated[cue_id] + "。"
                    ),
                ) for cue_id in expected]
                validation_errors = {
                    cue.id: _semantic_card_translation_errors({
                        "source": cue.source_text,
                        "duration_seconds": cue.duration,
                    }, cue.translation, terminology)
                    for cue in proposed
                }
                validation_errors = {
                    cue_id: values for cue_id, values in validation_errors.items()
                    if values
                }
                if validation_errors:
                    last_error = json.dumps({
                        "validation_errors": validation_errors,
                    }, ensure_ascii=False)
                    continue
                previous_translations = {cue.id: cue.translation for cue in cues}
                for cue in proposed:
                    by_id[cue.id].translation = cue.translation
                self._enforce_terminology_contract(cues, terminology)
                if term_errors := terminology_contract_errors(cues, terminology):
                    for cue in cues:
                        cue.translation = previous_translations[cue.id]
                    last_error = "final Chinese style repair violated terminology: " + "; ".join(
                        term_errors
                    )
                    continue
                if caption_errors := interview_caption_duration_errors(
                    [by_id[cue_id] for cue_id in expected]
                ):
                    for cue in cues:
                        cue.translation = previous_translations[cue.id]
                    last_error = (
                        "final Chinese style repair violated caption policy: "
                        + "; ".join(caption_errors)
                    )
                    continue
                return {
                    "step": "interview_chinese_style_repair",
                    "cue_ids": sorted(expected),
                    "provenance": provenance,
                    "provider_failures": provider_failures,
                }
        raise ValueError("interview Chinese style repair failed: " + last_error)

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
                    "Return exactly one continuous 45–180 second WeChat clip and no Bilibili chapters. "
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
                "Keep the existing terminology unless it must be normalized. Terminology fields are source, strategy (translate|preserve|bilingual_once), target, first_use_explanation, notes.",
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
                    if item.strategy == TerminologyStrategy.PRESERVE
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
                    elif entry.source.casefold() == "open-source check" and re.search(
                        r"开源.{0,8}制衡", replaced,
                    ):
                        replaced = re.sub(r"\s*[（(]开源制衡[）)]", "", replaced)
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

    def _repair_terminology(
        self, cues: list[TranscriptCue], terminology: list[TerminologyEntry], errors: list[str],
    ) -> dict[str, Any]:
        affected_terms = [
            entry for entry in terminology
            if any(error.startswith(f"term:{entry.source}:") for error in errors)
        ]
        affected: dict[str, TranscriptCue] = {}
        for entry in affected_terms:
            cue = next(
                (item for item in cues if _contains_term(item.source_text, entry.source)), None,
            )
            if cue:
                affected[cue.id] = cue
        forbidden_phrases = ("挽具", "铺好的道路", "人类触摸", "技能登记处")
        for cue in cues:
            if any(phrase in cue.translation for phrase in forbidden_phrases):
                affected[cue.id] = cue
        if not affected:
            raise ValueError("; ".join(errors))

        payload = [
            {
                "id": cue.id, "source": cue.source_text,
                "current_translation": cue.translation,
            }
            for cue in affected.values()
        ]
        prompt = "\n".join([
            "Repair only these Simplified Chinese subtitle lines so they remain natural and faithful while satisfying the terminology contract.",
            "A preserve term must appear exactly in English. A bilingual_once term must keep its English source and include the exact first_use_explanation on its first source occurrence. Do not translate code, APIs, products, or unsettled technical terms into literal Chinese.",
            "For translate terms, rewrite the whole sentence as direct, idiomatic spoken Chinese; do not merely substitute one English token inside the old translated word order. Prefer actor-action phrasing and remove filler or demonstratives that Chinese would normally omit.",
            "Return {translations:[{id,text}]}; return every supplied id exactly once and no extra ids.",
            "Contract errors: " + json.dumps(errors, ensure_ascii=False),
            "Relevant terminology: " + json.dumps([asdict(item) for item in affected_terms], ensure_ascii=False),
            "Cues: " + json.dumps(payload, ensure_ascii=False),
        ])
        expected = set(affected)
        translations: dict[str, str] = {}
        provenance: dict[str, Any] | None = None
        failures: list[str] = []
        writers = []
        for writer in (
            self.directing_writer, self.writer,
            getattr(self.writer, "fallback", None),
        ):
            if writer is not None and all(writer is not existing for existing in writers):
                writers.append(writer)
        for repair_writer in writers:
            try:
                draft, provenance = repair_writer._request_json([
                    {"role": "system", "content": "Return one valid JSON object only."},
                    {"role": "user", "content": prompt},
                ], max_tokens=2500)
                candidate = {
                    str(item.get("id")): str(item.get("text") or "").strip()
                    for item in draft.get("translations", []) if isinstance(item, dict)
                }
                if set(candidate) != expected or any(not value for value in candidate.values()):
                    missing = sorted(expected - set(candidate))
                    extra = sorted(set(candidate) - expected)
                    raise ValueError(
                        f"terminology repair ids mismatch; missing={missing}, extra={extra}"
                    )
                translations = candidate
                break
            except Exception as error:
                failures.append(f"{type(error).__name__}: {error}")
        if not translations:
            # Final bounded fallback: exact-term enforcement is less elegant
            # than a rewrite, but preserves the source meaning and allows a
            # transient provider outage to self-heal without aborting the job.
            enforced = self._enforce_terminology_contract(cues, terminology)
            remaining = terminology_contract_errors(cues, terminology)
            if not remaining:
                return {
                    "step": "terminology_repair",
                    "errors": errors,
                    "fallback": "deterministic_exact_term_enforcement",
                    "enforced": enforced,
                    "provider_failures": failures,
                }
            raise RuntimeError(
                "terminology repair providers failed and deterministic fallback "
                "could not satisfy the contract: " + "; ".join([*failures, *remaining])
            )
        for cue_id, cue in affected.items():
            cue.translation = translations[cue_id]
        return {
            "step": "terminology_repair", "errors": errors,
            "provenance": provenance, "provider_failures": failures,
        }

    @staticmethod
    def _parse_terminology(raw: Any, cues: list[TranscriptCue]) -> list[TerminologyEntry]:
        entries: list[TerminologyEntry] = []
        for item in raw if isinstance(raw, list) else []:
            source = str(item.get("source") or item.get("term") or "").strip() if isinstance(item, dict) else ""
            if not isinstance(item, dict) or not source:
                continue
            try:
                strategy = TerminologyStrategy(str(item.get("strategy") or item.get("choice") or "preserve"))
            except ValueError:
                strategy = TerminologyStrategy.PRESERVE
            target = str(item.get("target") or item.get("translation") or "").strip()
            explanation = str(item.get("first_use_explanation") or "").strip()
            established_target = ESTABLISHED_CHINESE_TERMS.get(source.casefold())
            if established_target:
                strategy = TerminologyStrategy.TRANSLATE
                target = established_target
                explanation = ""
            if strategy == TerminologyStrategy.BILINGUAL_ONCE and not explanation:
                explanation = target
            if (
                strategy == TerminologyStrategy.BILINGUAL_ONCE
                and explanation
                and _contains_term(explanation, source)
                and target
            ):
                # Legacy plans sometimes stored the complete rendered label
                # ("data center（数据中心）") as the explanation. Enforcement
                # adds the English source itself, so retaining that shape
                # duplicates the term and can break the caption density gate.
                explanation = target
            # An incomplete strategy row cannot be enforced downstream. Keep
            # the exact source term instead of failing the whole video or
            # inventing a translation/explanation the model did not supply.
            if strategy == TerminologyStrategy.TRANSLATE and not target:
                strategy = TerminologyStrategy.PRESERVE
            if strategy == TerminologyStrategy.BILINGUAL_ONCE and not explanation:
                strategy = TerminologyStrategy.PRESERVE
            entries.append(TerminologyEntry(
                source=source, strategy=strategy, target=target,
                first_use_explanation=explanation,
                notes=str(item.get("notes") or "").strip(),
            ))
        source = "\n".join(item.source_text for item in cues)
        known = {item.source.casefold() for item in entries}
        for term, target in ESTABLISHED_CHINESE_TERMS.items():
            if _contains_term(source, term) and term.casefold() not in known:
                entries.append(TerminologyEntry(
                    source=term, strategy=TerminologyStrategy.TRANSLATE,
                    target=target,
                ))
                known.add(term.casefold())
        for term in PROTECTED_TERMS:
            if _contains_term(source, term) and term.casefold() not in known:
                strategy = TerminologyStrategy.BILINGUAL_ONCE if term in {"Harness"} else TerminologyStrategy.PRESERVE
                entries.append(TerminologyEntry(
                    term, strategy,
                    first_use_explanation="Agent 的执行与反馈框架" if term == "Harness" else "",
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
    if source_range is None or not INTERVIEW_MIN_SECONDS <= source_range.duration <= INTERVIEW_MAX_SECONDS:
        errors.append("interview highlight must be a complete 45–180 second source range")
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


def editorial_plan_contract_errors(
    plan: dict[str, Any], duration: float, cues: list[TranscriptCue] | None = None,
) -> list[str]:
    if str(plan.get("editorial_mode") or "") == "known_tech_interview_clip":
        return _interview_clip_contract_errors(plan, duration, cues)
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
    r"^(?:(?:um+|uh+|erm+|hmm+|mm+|you know|i mean|like|well)[\s,.;:!?-]*)+$",
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


def build_collection_manifest(
    candidate: Candidate, metadata: dict[str, Any], cues: list[TranscriptCue],
    terminology: list[TerminologyEntry], plan: dict[str, Any], source_media_path: str,
    source_subtitle_path: str, source_media_info: SourceMediaInfo | None = None,
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
            hook_candidates = build_hook_candidates(
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
            hook_candidates = build_hook_candidates(
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
                # Remove only this job's incomplete generated file and let the
                # agent retry the same auditable acquisition route once.
                downloaded.unlink(missing_ok=True)
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


def _desired_subtitle_card_count(english: str, chinese: str, duration: float) -> int:
    """Apply the shared interview-card timing and density policy."""
    english_words = len(re.findall(r"\S+", english))
    chinese_characters = len(re.sub(r"\s+", "", chinese))
    desired = max(
        1,
        math.ceil(duration / INTERVIEW_CAPTION_TARGET_MAX_SECONDS),
        math.ceil(english_words / INTERVIEW_CAPTION_TARGET_MAX_ENGLISH_WORDS),
        math.ceil(chinese_characters / INTERVIEW_CAPTION_TARGET_MAX_CHINESE_CHARACTERS),
    )
    # Do not request more cards than can receive the policy minimum. If density
    # still cannot fit, the publication gate rejects the cue instead of making
    # sub-second flashes.
    maximum_feasible = max(1, math.floor(duration / INTERVIEW_CAPTION_MIN_SECONDS))
    return min(maximum_feasible, desired)


def _semantic_english_parts(value: str, desired_count: int) -> list[str]:
    """Partition only at spoken clause boundaries; never cut by character count."""
    normalized = re.sub(r"\s+", " ", value).strip()
    words = normalized.split()
    if desired_count <= 1 or len(words) < 8:
        return [normalized]
    boundary_costs: dict[int, float] = {}
    coordinating = {"and", "but", "or", "so", "while", "whereas", "then", "yet", "maybe"}
    dependent_clause = {"if", "when", "because", "that"}
    relative_heads = {
        "domains", "factor", "given", "idea", "reason", "text", "thing", "way",
    }
    coordinated_list_words = {
        "everything", "expertise", "knowledge", "less", "meetings", "more",
    }
    weak_relative = {"where", "which"}
    spoken_fillers = {"uh", "um"}
    clean_words = [
        re.sub(r"^[\"'“‘(\[]+|[^A-Za-z]+$", "", word).casefold()
        for word in words
    ]

    def noisy_asr_near(index: int, window: int = 20) -> bool:
        nearby = clean_words[index:min(len(clean_words), index + window)]
        return any(
            left == right and left not in spoken_fillers and left
            for left, right in zip(nearby, nearby[1:])
        )

    for index in range(1, len(words)):
        previous = words[index - 1]
        current = clean_words[index]
        if re.search(r"[.!?;:][\"'”’)]?$", previous):
            boundary_costs[index] = 0.0
        elif (
            re.search(r",[\"'”’)]?$", previous)
            and current in {"he", "it", "she", "they", "we", "you"}
        ):
            boundary_costs[index] = 0.35
        elif current == "so" and index + 1 < len(words) and re.sub(
            r"^[\"'“‘(\[]+|[^A-Za-z]+$", "", words[index + 1],
        ).casefold() == "that":
            # The Chinese card can restore the governing action from parent
            # context.  Keeping this as a candidate is preferable to a
            # 10–20-second card in disfluent interview ASR.
            boundary_costs[index] = 0.75
        elif (
            current == "to" and index + 2 < len(clean_words)
            and clean_words[index + 1:index + 3] == ["your", "point"]
        ):
            # Spoken "to your point ..." starts a new response move. Keeping
            # it with the preceding ASR clause can leave that card dangling on
            # "about" and strand the actual point in the next card.
            boundary_costs[index] = 0.25
        elif (
            current == "and" and index > 0
            and clean_words[index - 1] == "lo"
            and index + 1 < len(clean_words)
            and clean_words[index + 1] == "behold"
        ):
            # “lo and behold” is one spoken idiom. Splitting before “and”
            # produces a meaningless `...well lo` card even though the full
            # parent cue is valid.
            continue
        elif (
            current == "and"
            and index <= 5
            and clean_words[0] in spoken_fillers
        ):
            # Do not strand a short interview false start such as
            # “Um so in coding essenti” before the speaker reaches the first
            # complete clause. The later conjunction remains available.
            continue
        elif (
            current == "and" and index + 1 < len(words)
            and clean_words[index + 1] == "if" and noisy_asr_near(index)
        ):
            # A boundary immediately before a corrupt repeated ASR fragment
            # strands the condition and invites literal translationese.  Keep
            # searching for the earlier action boundary (for example,
            # "and look for bottlenecks") so the noisy condition retains a
            # recoverable predicate.
            continue
        elif (
            current == "if" and clean_words[index - 1] == "and"
            and noisy_asr_near(index)
        ):
            continue
        elif current == "and" and index + 1 < len(clean_words) and (
            clean_words[index - 1] in coordinated_list_words
            or clean_words[index + 1] in coordinated_list_words
        ):
            # Do not cut a coordinated noun list (“knowledge and expertise”)
            # or a repeated comparison (“less and less”).
            continue
        elif current == "and" and "where when" in " ".join(
            clean_words[max(0, index - 12):index]
        ):
            # Do not cut a still-open relative/time clause immediately before
            # its main predicate, e.g. “where when they deploy … and they run
            # into a bug”.
            continue
        elif current == "or" and "either" in clean_words[max(0, index - 5):index]:
            # Keep the two halves of “either … or …” together. A later “or”
            # can still introduce the next complete alternative.
            continue
        elif current == "that" and clean_words[index - 1] in relative_heads:
            # Keep defining relative clauses with the noun they explain.
            continue
        elif current in coordinating:
            boundary_costs[index] = 0.25
        elif current == "like" and index >= 6:
            # In long interview ASR, a later "like" commonly restarts an
            # example or paraphrase. It is a safer forced boundary than an
            # arbitrary word-count cut, and the card translator removes the
            # discourse filler while preserving the following proposition.
            boundary_costs[index] = 1.25
        elif current == "just" and index >= 6:
            # Repeated interview emphasis often introduces a complete final
            # verdict ("just flat out, we have to do better").  It is safer
            # than cutting the governing clause after "I don't think".
            boundary_costs[index] = 1.0
        elif (
            current == "capture" and index >= 6
            and clean_words[index - 1] == "of"
            and "work" in clean_words[max(0, index - 4):index]
        ):
            # Spoken list setup: "everyday CRM work of: capture a meeting,
            # fill records, update tasks."  The ASR omits the colon.
            boundary_costs[index] = 0.5
        elif (
            current == "a" and index >= 3
            and clean_words[index - 1] == "compact"
            and index + 1 < len(clean_words)
            and clean_words[index + 1] == "compact"
        ):
            # Preserve the speaker's repeated term while giving the ownership
            # claim and named attribution separate readable cards.
            boundary_costs[index] = 0.2
        elif current == "because" and index >= 3:
            boundary_costs[index] = 0.75
        elif current in dependent_clause and index >= 6:
            boundary_costs[index] = 1.0
        elif (
            current == "where" and index + 1 < len(clean_words)
            and clean_words[index + 1] == "when"
        ):
            continue
        elif current in weak_relative and index >= 5:
            boundary_costs[index] = 1.5
        elif current in spoken_fillers:
            # A hesitation sound is not a semantic boundary. Splitting before
            # it can strand the preceding contrast (for example “but Linux”)
            # even though the next words complete that same clause.
            continue
    if not boundary_costs:
        return [normalized]
    maximum_count = min(desired_count, len(boundary_costs) + 1)
    for count in range(maximum_count, 1, -1):
        target = len(words) / count
        ranked: list[tuple[float, tuple[int, ...]]] = []
        for cuts in combinations(sorted(boundary_costs), count - 1):
            bounds = (0, *cuts, len(words))
            lengths = [bounds[i + 1] - bounds[i] for i in range(count)]
            if min(lengths) < 3:
                continue
            candidate_parts = [
                " ".join(words[bounds[i]:bounds[i + 1]]).strip()
                for i in range(count)
            ]
            discourse_only = {"and", "but", "like", "so", "uh", "um", "you", "know"}
            if any(
                not [
                    word for word in re.findall(r"[A-Za-z']+", part.casefold())
                    if word not in discourse_only
                ]
                for part in candidate_parts
            ):
                continue
            if any(
                _semantic_source_part_is_dangling(part)
                for part in candidate_parts[:-1]
            ):
                # A short card is not useful when it visibly ends on "and",
                # "so", "uh", an article, or another open connector.
                continue
            if any(
                (
                    candidate_parts[index + 1].casefold().startswith("and the ")
                    and not re.search(
                        r"\b(?:is|are|was|were|has|have|does|do|will|can)\b",
                        candidate_parts[index + 1], re.IGNORECASE,
                    )
                )
                or re.match(
                    r"^and\s+[A-Z][A-Za-z.'-]+\s+"
                    r"(?:\w+\s+){0,2}(?:is|are|was|were|has|have|does|do)\b",
                    candidate_parts[index + 1],
                )
                for index in range(count - 1)
            ):
                # Keep coordinated subjects and noun phrases together:
                # "Susan / and Sheldon are ..." and
                # "the best actor / and the best partner ..." are not two
                # independently translatable caption cards.
                continue
            if any(
                re.search(r"\bif\b[^.!?]*$", candidate_parts[i], re.IGNORECASE)
                and candidate_parts[i + 1].casefold().startswith(("where ", "which "))
                for i in range(count - 1)
            ):
                continue
            balance = sum(abs(length - target) for length in lengths)
            boundary_penalty = sum(boundary_costs[cut] for cut in cuts)
            ranked.append((balance + boundary_penalty * 2.5, cuts))
        if not ranked:
            continue
        _, cuts = min(ranked, key=lambda item: item[0])
        bounds = (0, *cuts, len(words))
        return [
            " ".join(words[bounds[index]:bounds[index + 1]]).strip()
            for index in range(count)
        ]
    return [normalized]


def _semantic_source_part_is_dangling(value: str) -> bool:
    """Reject fixed card cuts that leave a setup without its complement."""
    cleaned = re.sub(r"[-,:;]+$", "", value.strip()).strip()
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
        r"\b(?:predicts?|forecast(?:s|ed)?|found|divided)"
        r"(?:\s+(?:that|the|a|an|our|their|his|her|its|\w+)){0,3}$",
        cleaned, re.IGNORECASE,
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
    return set(re.findall(r"\b[A-Z][A-Z0-9.-]{1,}\b", value)) - {"AI", "US", "USA"}


def _caption_entity_alignment_errors(source: str, translation: str) -> list[str]:
    """Keep named entities in their fixed source card, allowing known Chinese aliases."""
    errors: list[str] = []
    source_entities = _uppercase_source_entities(source)
    target_entities = _uppercase_source_entities(translation)
    for entity in source_entities:
        aliases = CAPTION_ENTITY_ALIASES.get(entity, (entity,))
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
) -> list[str]:
    """Validate one isolated semantic card before it can be locked for rendering."""
    errors: list[str] = []
    source = str(row.get("source") or "")
    duration = max(float(row.get("duration_seconds") or 0.1), 0.1)
    visible = len(re.sub(r"\s+", "", translation))
    if not translation:
        errors.append("empty")
    if (
        visible > INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS
        or visible / duration > INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND
    ):
        errors.append("reading_speed")
    if translation and not re.search(r"[。！？!?]$", translation):
        errors.append("punctuation")
    errors.extend(_caption_entity_alignment_errors(source, translation))
    if _contains_term(source, "maybe") and not re.search(r"可能|也许|或许", translation):
        errors.append("modality")
    if (
        _contains_term(source, "up the stack")
        and re.search(r"(?:堆栈|技术栈).{0,6}(?:上移|向上)", translation)
    ):
        errors.append("literal_industry_stack")
    for term in terminology:
        if (
            term.strategy == TerminologyStrategy.TRANSLATE
            and term.target
            and _contains_term(source, term.source)
            and (
                not _translated_term_present(
                    term, TranscriptCue(
                        id=str(row.get("id") or ""), start=0, end=duration,
                        source_text=source, translation=translation,
                    ),
                )
                or _contains_term(translation, term.source)
            )
        ):
            errors.append(f"term:{term.source}")
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


def _coalesce_short_semantic_parts(
    parts: list[str], duration: float,
    minimum_seconds: float = INTERVIEW_CAPTION_MIN_SECONDS,
) -> list[str]:
    """Merge speech fragments only when the real allocator cannot fit the minimum."""
    merged = list(parts)
    while len(merged) > 1:
        durations = _allocate_caption_durations(merged, duration)
        short_index = next((
            index for index, allocated in enumerate(durations)
            if allocated < minimum_seconds - 1e-6
        ), None)
        if short_index is None:
            weights = [max(1, len(re.findall(r"\S+", part))) for part in merged]
            total = sum(weights)
            short_index = next((
                index for index, (part, weight) in enumerate(zip(merged, weights))
                if duration * weight / total < minimum_seconds
                and part.casefold().lstrip().startswith(("and ", "or "))
                and not re.search(r"[.!?][\"'”’)]?$", part.strip())
            ), None)
        if short_index is None:
            break
        if short_index == 0:
            merged[1] = f"{merged[0]} {merged[1]}".strip()
            del merged[0]
        else:
            merged[short_index - 1] = (
                f"{merged[short_index - 1]} {merged[short_index]}".strip()
            )
            del merged[short_index]
    return merged


def _allocate_caption_durations(parts: list[str], duration: float) -> list[float]:
    """Allocate a complete parent span while honoring hard card bounds.

    Source word timing is not present in ``TranscriptCue``. Word count is the
    best available speech proxy, but the final allocation is projected into
    the same minimum/maximum bounds used by validation and rendering.
    """
    if not parts:
        return []
    count = len(parts)
    minimum = INTERVIEW_CAPTION_MIN_SECONDS
    maximum = INTERVIEW_CAPTION_HARD_MAX_SECONDS
    if duration < count * minimum - 1e-6 or duration > count * maximum + 1e-6:
        weights = [max(1, len(re.findall(r"\S+", part))) for part in parts]
        total_weight = sum(weights)
        return [duration * weight / total_weight for weight in weights]

    weights = [max(1, len(re.findall(r"\S+", part))) for part in parts]
    allocated = [minimum] * count
    remaining = duration - count * minimum
    open_indices = set(range(count))
    while remaining > 1e-9 and open_indices:
        total_weight = sum(weights[index] for index in open_indices)
        consumed = 0.0
        saturated: set[int] = set()
        for index in open_indices:
            share = remaining * weights[index] / total_weight
            capacity = maximum - allocated[index]
            addition = min(share, capacity)
            allocated[index] += addition
            consumed += addition
            if capacity - addition <= 1e-9:
                saturated.add(index)
        remaining -= consumed
        open_indices -= saturated
        if consumed <= 1e-9:
            break
    if allocated:
        allocated[-1] += duration - sum(allocated)
    return allocated


def _retime_existing_semantic_card_groups(cues: list[TranscriptCue]) -> None:
    """Project stale sibling-card timings into the current hard bounds.

    Older plans allocated time purely by word count and can contain a
    sub-second final card beside a long sibling. Moving only their shared
    boundary preserves the ordered bilingual pairs and complete parent span;
    it does not invent a new English/Chinese alignment.
    """
    index = 0
    while index < len(cues):
        cue = cues[index]
        if "-card-" not in cue.id:
            index += 1
            continue
        parent_id = cue.id.rsplit("-card-", 1)[0]
        end_index = index + 1
        while (
            end_index < len(cues)
            and cues[end_index].id.rsplit("-card-", 1)[0] == parent_id
            and abs(cues[end_index].start - cues[end_index - 1].end) <= 0.1
        ):
            end_index += 1
        group = cues[index:end_index]
        if len(group) > 1:
            start = group[0].start
            end = group[-1].end
            durations = _allocate_caption_durations(
                [item.source_text for item in group], end - start,
            )
            if all(
                INTERVIEW_CAPTION_MIN_SECONDS - 1e-6 <= duration
                <= INTERVIEW_CAPTION_HARD_MAX_SECONDS + 1e-6
                for duration in durations
            ):
                elapsed = 0.0
                original_start = group[0].original_start
                original_end = group[-1].original_end
                has_original_timeline = (
                    original_start is not None and original_end is not None
                    and original_end > original_start
                )
                original_scale = (
                    (original_end - original_start) / max(end - start, 1e-9)
                    if has_original_timeline else 1.0
                )
                for position, (item, duration) in enumerate(
                    zip(group, durations), start=1,
                ):
                    item.start = round(start + elapsed, 3)
                    if has_original_timeline:
                        item.original_start = round(
                            original_start + elapsed * original_scale, 3,
                        )
                    elapsed += duration
                    item.end = end if position == len(group) else round(start + elapsed, 3)
                    if has_original_timeline:
                        item.original_end = (
                            original_end if position == len(group)
                            else round(original_start + elapsed * original_scale, 3)
                        )
        index = end_index


def _split_overlong_semantic_parts(
    parts: list[str], duration: float,
    maximum_seconds: float = INTERVIEW_CAPTION_HARD_MAX_SECONDS,
) -> list[str]:
    """Recursively split a defensible long clause after the initial partition."""
    refined = list(parts)
    while True:
        weights = [max(1, len(re.findall(r"\S+", part))) for part in refined]
        total = sum(weights)
        changed = False
        for index, (part, weight) in enumerate(zip(refined, weights)):
            if duration * weight / total <= maximum_seconds:
                continue
            children = _semantic_english_parts(part, 2)
            if len(children) <= 1:
                continue
            refined[index:index + 1] = children
            changed = True
            break
        if not changed:
            return refined


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


def interview_asr_suspicions(cues: list[TranscriptCue]) -> list[dict[str, Any]]:
    """Find caption spans that warrant audio verification, without running ASR."""
    findings: list[dict[str, Any]] = []
    for cue in cues:
        words = [
            re.sub(r"[^a-z0-9'-]+", "", word.casefold())
            for word in cue.source_text.split()
        ]
        words = [word for word in words if word]
        reasons: list[str] = []
        repeated = sorted({
            left for left, right in zip(words, words[1:])
            if left == right and left not in _ASR_REPEAT_IGNORE
        })
        if repeated:
            reasons.append("repeated_content_word:" + ",".join(repeated))
        for index, word in enumerate(words[:-1]):
            if words[index + 1] != "cycle":
                continue
            if word not in _KNOWN_CYCLE_MODIFIERS:
                reasons.append("unknown_cycle_modifier:" + word)
        if not reasons:
            continue
        findings.append({
            "cue_id": cue.id,
            "start": cue.start,
            "end": cue.end,
            "source": cue.source_text,
            "reasons": reasons,
        })
    return findings


def _whisper_text_for_range(
    payload: dict[str, Any], start: float, end: float,
) -> str:
    words = [
        word for segment in payload.get("segments", [])
        if isinstance(segment, dict)
        for word in segment.get("words", [])
        if isinstance(word, dict)
        and start <= (
            float(word.get("start") or 0) + float(word.get("end") or 0)
        ) / 2 <= end
    ]
    if words:
        return re.sub(
            r"\s+", " ", "".join(str(word.get("word") or "") for word in words),
        ).strip()
    segments = [
        str(segment.get("text") or "").strip()
        for segment in payload.get("segments", [])
        if isinstance(segment, dict)
        and float(segment.get("end") or 0) > start
        and float(segment.get("start") or 0) < end
        and str(segment.get("text") or "").strip()
    ]
    return re.sub(r"\s+", " ", " ".join(segments)).strip()


def _whisper_text_aligned_to_source(
    payload: dict[str, Any], source: str, start: float, end: float,
) -> str:
    """Use lexical alignment because semantic-card timestamps are proportional."""
    window_words = [
        word for segment in payload.get("segments", [])
        if isinstance(segment, dict)
        for word in segment.get("words", [])
        if isinstance(word, dict)
        and float(word.get("end") or 0) > start - TARGETED_WHISPER_CONTEXT_SECONDS
        and float(word.get("start") or 0) < end + TARGETED_WHISPER_CONTEXT_SECONDS
    ]

    def token(value: object) -> str:
        return re.sub(r"[^a-z0-9']+", "", str(value or "").casefold())

    source_tokens = [token(word) for word in source.split()]
    source_tokens = [word for word in source_tokens if word]
    whisper_tokens = [token(word.get("word")) for word in window_words]
    matcher = SequenceMatcher(None, source_tokens, whisper_tokens, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size >= 2]
    matched = sum(block.size for block in blocks)
    if not blocks or matched < max(3, math.ceil(len(source_tokens) * 0.4)):
        return _whisper_text_for_range(payload, start - 0.25, end + 0.05)
    first = min(block.b for block in blocks)
    last = max(block.b + block.size for block in blocks)
    # Recover a misspelled trailing entity (for example Lumenum→Lumentum)
    # that token-equality alignment intentionally could not match.
    unmatched_tail = source_tokens[max(block.a + block.size for block in blocks):]
    for source_token in unmatched_tail[-3:]:
        candidates = range(last, min(len(whisper_tokens), last + 4))
        best = max(
            candidates,
            key=lambda index: SequenceMatcher(
                None, source_token, whisper_tokens[index], autojunk=False,
            ).ratio(),
            default=-1,
        )
        if best >= 0 and SequenceMatcher(
            None, source_token, whisper_tokens[best], autojunk=False,
        ).ratio() >= 0.72:
            last = best + 1
    return re.sub(
        r"\s+", " ", "".join(
            str(word.get("word") or "") for word in window_words[first:last]
        ),
    ).strip()


def targeted_whisper_caption_audit(
    media_path: Path, cues: list[TranscriptCue], job: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any] | None:
    """Retranscribe only suspicious caption spans and cache the audio evidence."""
    findings = interview_asr_suspicions(cues)
    if not findings:
        return None
    runner = runner or subprocess.run
    whisper = shutil.which("whisper")
    if not whisper:
        raise RuntimeError(
            "suspicious YouTube captions require local Whisper, but whisper is unavailable"
        )
    policy_version = TARGETED_WHISPER_POLICY_VERSION
    source_fingerprint = hashlib.sha256(json.dumps({
        "media": str(media_path.resolve()),
        "size": media_path.stat().st_size,
        "mtime_ns": media_path.stat().st_mtime_ns,
        "findings": findings,
        "model": os.environ.get("VIDEO_FACTORY_WHISPER_MODEL", "large-v3-turbo"),
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    fingerprint = hashlib.sha256(
        f"{policy_version}:{source_fingerprint}".encode("utf-8")
    ).hexdigest()
    workspace_root = job.parents[1] if job.parent.name == "jobs" else job.parent
    output_dir = (
        workspace_root / "cache" / "targeted-whisper" / source_fingerprint[:24]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "caption-audit.json"
    if audit_path.is_file():
        cached = json.loads(audit_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            return {**cached, "cache_hit": True}

    # Semantic-card timestamps are assigned proportionally to text weight, so a
    # short child card can drift several seconds from the words it represents.
    # Keep enough surrounding audio for lexical alignment to find the actual
    # phrase instead of silently accepting a nearby repeated fragment.
    padded = sorted((
        max(0.0, float(row["start"]) - TARGETED_WHISPER_CONTEXT_SECONDS),
        float(row["end"]) + TARGETED_WHISPER_CONTEXT_SECONDS,
    ) for row in findings)
    merged: list[list[float]] = []
    for start, end in padded:
        if merged and start <= merged[-1][1] + 0.2:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    clip_timestamps = ",".join(
        f"{start:.3f},{end:.3f}" for start, end in merged
    )
    command = [
        whisper, str(media_path), "--model",
        os.environ.get("VIDEO_FACTORY_WHISPER_MODEL", "large-v3-turbo"),
        "--language", "en", "--task", "transcribe",
        "--clip_timestamps", clip_timestamps,
        "--word_timestamps", "True", "--output_format", "json",
        "--output_dir", str(output_dir), "--verbose", "False", "--fp16", "False",
        "--beam_size", os.environ.get("VIDEO_FACTORY_WHISPER_BEAM_SIZE", "1"),
        "--best_of", "1",
        "--threads", os.environ.get(
            "VIDEO_FACTORY_WHISPER_THREADS", str(os.cpu_count() or 4),
        ),
    ]
    raw_path = output_dir / "raw-transcript.json"
    raw_meta_path = output_dir / "raw-transcript-meta.json"
    raw_cache_hit = False
    if raw_path.is_file() and raw_meta_path.is_file():
        raw_meta = json.loads(raw_meta_path.read_text(encoding="utf-8"))
        raw_cache_hit = raw_meta.get("command") == command
    if raw_cache_hit:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        try:
            whisper_timeout = int(os.environ.get(
                "VIDEO_FACTORY_WHISPER_TIMEOUT_SECONDS", "900",
            ))
        except ValueError:
            whisper_timeout = 900
        whisper_timeout = max(120, min(whisper_timeout, 1800))
        completed = runner(
            command, capture_output=True, text=True,
            timeout=whisper_timeout, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "targeted Whisper verification failed: "
                + (completed.stderr or completed.stdout or "unknown error")[-800:]
            )
        whisper_path = output_dir / f"{media_path.stem}.json"
        if not whisper_path.is_file():
            raise RuntimeError("targeted Whisper verification produced no JSON transcript")
        payload = json.loads(whisper_path.read_text(encoding="utf-8"))
        raw_path.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        raw_meta_path.write_text(json.dumps({
            "fingerprint": fingerprint,
            "command": command,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    corrections: list[dict[str, Any]] = []
    for row in findings:
        verified = _whisper_text_aligned_to_source(
            payload, str(row["source"]), float(row["start"]), float(row["end"]),
        )
        if len(re.findall(r"\w+", verified)) < 2:
            raise RuntimeError(
                f"targeted Whisper could not recover caption {row['cue_id']}"
            )
        verified_words = [
            re.sub(r"[^a-z0-9'-]+", "", word.casefold())
            for word in verified.split()
        ]
        for reason in row.get("reasons", []):
            if not str(reason).startswith("unknown_cycle_modifier:"):
                continue
            original_modifier = str(reason).split(":", 1)[1]
            cycle_modifiers = {
                verified_words[index]
                for index in range(len(verified_words) - 1)
                if verified_words[index + 1] == "cycle"
            }
            resolved = (
                bool(cycle_modifiers.intersection(_KNOWN_CYCLE_MODIFIERS))
                or (
                    original_modifier not in verified_words
                    and "cycle" in verified_words
                )
            )
            if not resolved:
                raise RuntimeError(
                    "targeted Whisper did not resolve ambiguous cycle modifier "
                    f"for {row['cue_id']}: {verified}"
                )
        corrections.append({
            **row,
            "verified_source": verified,
            "changed": re.sub(r"\W+", "", verified).casefold()
            != re.sub(r"\W+", "", str(row["source"])).casefold(),
        })
    audit = {
        "step": "targeted_whisper_caption_audit",
        "policy_version": policy_version,
        "fingerprint": fingerprint,
        "cache_hit": False,
        "raw_cache_hit": raw_cache_hit,
        "model": os.environ.get("VIDEO_FACTORY_WHISPER_MODEL", "large-v3-turbo"),
        "clip_ranges": merged,
        "corrections": corrections,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return audit


def interview_caption_duration_errors(cues: list[TranscriptCue]) -> list[str]:
    """Validate every measurable invariant in the shared caption policy."""
    errors: list[str] = []
    for cue in cues:
        source = re.sub(r"\s+", " ", cue.source_text).strip()
        translation = re.sub(r"\s+", "", cue.translation)
        words = len(re.findall(r"\S+", source))
        if not source or not translation:
            errors.append(f"{cue.id} is missing bilingual caption text")
            continue
        if cue.duration < INTERVIEW_CAPTION_MIN_SECONDS - 1e-6:
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
        if len(translation) > INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS:
            errors.append(
                f"{cue.id} has {len(translation)} Chinese characters, above the "
                f"{INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS}-character maximum"
            )
        if (
            len(translation) / max(cue.duration, 0.1)
            > INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND
        ):
            errors.append(
                f"{cue.id} exceeds "
                f"{INTERVIEW_CAPTION_MAX_CHINESE_CHARACTERS_PER_SECOND:g} "
                "Chinese characters per second"
            )
        entity_errors = _caption_entity_alignment_errors(source, cue.translation)
        if entity_errors:
            errors.append(
                f"{cue.id} has bilingual entity misalignment: "
                + ", ".join(entity_errors)
            )
    return errors


def cached_interview_caption_pipeline_complete(
    cues: list[TranscriptCue], trace: list[dict[str, Any]],
) -> bool:
    """Identify an immutable plan reviewed under the exact current policy."""
    substantive_ids = {
        cue.id for cue in cues
        if cue.source_text.strip() or cue.translation.strip()
    }
    if not substantive_ids or interview_caption_duration_errors(cues):
        return False
    return any(
        item.get("step") == "interview_semantic_subtitle_cards"
        and item.get("policy_version") == INTERVIEW_CAPTION_POLICY_VERSION
        and item.get("policy_fingerprint") == INTERVIEW_CAPTION_POLICY_FINGERPRINT
        and substantive_ids <= {
            str(cue_id) for cue_id in item.get("reviewed_cue_ids", [])
        }
        for item in trace if isinstance(item, dict)
    )


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
    in ``segment_interview_subtitle_cards``, where every fixed source span gets
    its own translation and fidelity review.
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
            index += 1
        if current.source_text and current.translation:
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
        # Cached manifests can predate the current translation policy. Apply
        # deterministic cleanup on every render so rerenders self-heal without
        # another model call or an editorial rewrite.
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
        "technical_coverage", "known_tech_interview_clip",
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
                duration_ok = INTERVIEW_MIN_SECONDS <= item.duration <= INTERVIEW_MAX_SECONDS
                target = "45–180"
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
        cue.start = round(local_start, 3)
        cue.end = round(local_end, 3)
        cue.original_start = round(cue_original_start, 3)
        cue.original_end = round(cue_original_end, 3)
        rebased_cues.append(cue)
    cues[:] = rebased_cues

    local_start = max(0.0, original_start - offset)
    local_end = min(media_duration, original_end - offset)
    if local_end - local_start < INTERVIEW_MIN_SECONDS:
        raise ValueError(
            "downloaded interview interval cannot contain the validated 45-second highlight"
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
    if (
        cached_clip.get("rebased")
        and cached_duration > 0
        and abs(media_duration - cached_duration) <= tolerance
        and all(cached_clip.get(key) is not None for key in required)
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
    if cached_clip.get("rebased") or not cues:
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
    if not fits_local_media or overlaps_original_timeline:
        return cached_clip
    return {
        **cached_clip,
        "rebased": True,
        "download_start": float(local_window["download_start"]),
        "download_end": float(local_window["download_end"]),
        "media_duration": media_duration,
    }


class YouTubeCollectionFactory:
    def __init__(
        self, workspace: Workspace, writer: OpenAICompatibleStoryWriter,
        directing_writer: OpenAICompatibleStoryWriter | None = None,
        subtitle_reviewer: OpenAICompatibleStoryWriter | None = None,
    ) -> None:
        self.workspace = workspace
        self.writer = writer
        self.directing_writer = directing_writer
        self.subtitle_reviewer = subtitle_reviewer

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
        classified_mode, known_people, _ = classify_youtube_editorial(
            str(metadata.get("title") or ""),
            str(metadata.get("channel") or metadata.get("uploader") or ""),
            str(metadata.get("description") or ""),
            [item for item in (metadata.get("chapters") or []) if isinstance(item, dict)],
            [str(item) for item in (metadata.get("creators") or [])],
        )
        metadata["known_tech_people"] = known_people
        if editorial_mode == "auto":
            editorial_mode = classified_mode
        if editorial_mode not in {"technical_coverage", "known_tech_interview_clip", "study"}:
            raise ValueError(f"YouTube source is not eligible for an editorial route: {editorial_mode}")
        metadata["editorial_mode"] = editorial_mode
        translation_plan, caption_incumbent_trace = select_caption_incumbent(
            self.workspace, str(metadata.get("id") or ""), translation_plan,
            editorial_mode, editorial_guidance,
        )
        translator = NaturalSubtitleTranslator(
            self.writer, self.directing_writer, self.subtitle_reviewer,
        )
        cached_source_clip: dict[str, Any] | None = None
        cached_interview_captions_complete = False
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
            interview_cache = (
                editorial_mode == "known_tech_interview_clip"
                and cached_source_video_id
                and cached_source_video_id == current_source_video_id
            )
            if not cached_cues or (
                not interview_cache
                and _translation_source_fingerprint(cached_cues) != _translation_source_fingerprint(cues)
            ):
                raise ValueError("translation plan transcript does not match the supplied YouTube subtitles")
            if interview_cache or all(
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
            terminology = translator._parse_terminology(cached.get("terminology", []), cues)
            trace = list(cached.get("trace") or [])
            if caption_incumbent_trace:
                trace.append(caption_incumbent_trace)
            cached_interview_captions_complete = (
                editorial_mode == "known_tech_interview_clip"
                and cached_interview_caption_pipeline_complete(cues, trace)
            )
            editorial_plan, repairs = translator.ensure_editorial_plan(
                metadata, cues, editorial_plan, editorial_mode,
            )
            trace.extend(repairs)
            if cached_term_errors := terminology_contract_errors(cues, terminology):
                # During generation we still have a language model available,
                # so repair the whole Chinese clause naturally before the
                # deterministic replacement fallback. A pure rerender remains
                # model-free and fails closed if replacement cannot satisfy it.
                trace.append(translator._repair_terminology(
                    cues, terminology, cached_term_errors,
                ))
            if enforcement_trace := enforce_cached_terminology_contract(cues, terminology):
                trace.append(enforcement_trace)
            if dense := fast_translation_cues(cues):
                trace.append(translator._repair_reading_speed(
                    dense, terminology=terminology, all_cues=cues,
                ))
                NaturalSubtitleTranslator._enforce_terminology_contract(cues, terminology)
                if remaining := fast_translation_cues(cues):
                    raise ValueError(
                        "cached subtitle reading speed repair failed: "
                        + ", ".join(cue.id for cue in remaining[:10])
                    )
                if term_errors := terminology_contract_errors(cues, terminology):
                    raise ValueError(
                        "cached subtitle terminology failed after density repair: "
                        + "; ".join(term_errors)
                    )
        else:
            if editorial_guidance and editorial_guidance.strip():
                terminology, editorial_plan, trace = translator.translate(
                    metadata, cues, editorial_mode,
                    editorial_guidance=editorial_guidance,
                )
            else:
                # Preserve the original three-argument seam for adapters and
                # tests that replace the translator implementation.
                terminology, editorial_plan, trace = translator.translate(
                    metadata, cues, editorial_mode,
                )
            if caption_incumbent_trace:
                trace.append(caption_incumbent_trace)
        if editorial_mode == "known_tech_interview_clip":
            if asr_term_corrections := normalize_interview_asr_terms(cues, terminology):
                trace.append({
                    "step": "interview_asr_term_normalization",
                    "corrections": asr_term_corrections,
                })
        if editorial_mode == "known_tech_interview_clip" and cached_interview_captions_complete:
            trace.append({
                "step": "interview_caption_pipeline_reuse",
                "reason": (
                    "cached plan already contains reviewed semantic cards; "
                    "translation and boundaries are immutable during rerender"
                ),
            })
        elif editorial_mode == "known_tech_interview_clip":
            before_merge = list(cues)
            # Legacy YouTube captions can break one sentence across several
            # 10–12 second cues (for example, "every" / "layer."). Join the
            # complete dependency chain before choosing fixed card spans; the
            # downstream policy immediately repartitions it into <=7.5s cards.
            merged_cues = merge_dependent_subtitle_cues(
                cues, maximum_duration=60.0,
            )
            if any(
                cue.source_text.strip() != before_merge[index].source_text.strip()
                for index, cue in enumerate(merged_cues)
                if index < len(before_merge)
            ) or len(merged_cues) != len(before_merge):
                repair_trace = translator.repair_merged_interview_translations(
                    before_merge, merged_cues, terminology,
                )
                cues[:] = merged_cues
                if repair_trace:
                    trace.append(repair_trace)
                if post_merge_term_errors := terminology_contract_errors(
                    cues, terminology,
                ):
                    trace.append({
                        **translator._repair_terminology(
                            cues, terminology, post_merge_term_errors,
                        ),
                        "step": "post_merge_terminology_repair",
                    })
                if terminology_trace := enforce_cached_terminology_contract(cues, terminology):
                    trace.append({
                        **terminology_trace,
                        "step": "post_merge_terminology_enforcement",
                    })
            if subtitle_card_trace := translator.segment_interview_subtitle_cards(
                cues, terminology,
            ):
                trace.append(subtitle_card_trace)
            if chinese_style_trace := translator.repair_interview_chinese_style(
                cues, terminology,
            ):
                trace.append(chinese_style_trace)
        if editorial_mode == "known_tech_interview_clip":
            if caption_errors := interview_caption_duration_errors(cues):
                raise ValueError(
                    "interview caption policy gate failed; regenerate from the raw "
                    "translation plan instead of reprocessing cached cards: "
                    + "; ".join(caption_errors)
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
            trace.append({
                "step": "omit_spoken_fillers",
                "cue_ids": filler_cue_ids,
                "policy": "English source remains visible; Chinese carries substantive meaning only",
            })
        source_clip: dict[str, Any] | None = None
        original_duration = float(metadata.get("duration") or (cues[-1].end if cues else 0))
        selected_interview_range: SourceRange | None = None
        if editorial_mode == "known_tech_interview_clip":
            rows = editorial_plan.get("wechat_lessons")
            raw_selected = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None
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
        audio_verified_ids = {
            str(row.get("cue_id") or "")
            for item in trace if isinstance(item, dict)
            and item.get("step") == "targeted_whisper_caption_audit"
            and item.get("policy_version") == TARGETED_WHISPER_POLICY_VERSION
            for row in item.get("corrections", []) if isinstance(row, dict)
        }
        unverified_cues = [
            cue for cue in cues if cue.id not in audio_verified_ids
        ]
        pending_asr_suspicions = (
            interview_asr_suspicions(unverified_cues)
            if editorial_mode == "known_tech_interview_clip" else []
        )
        if pending_asr_suspicions and not any(
            item.get("step") == "targeted_whisper_pending"
            for item in trace if isinstance(item, dict)
        ):
            trace.append({
                "step": "targeted_whisper_pending",
                "cue_ids": [row["cue_id"] for row in pending_asr_suspicions],
                "reasons": {
                    row["cue_id"]: row["reasons"] for row in pending_asr_suspicions
                },
                "policy": "verify only suspicious spans after selected media is local",
            })

        def persist_translation_plan() -> None:
            (job / "translation-plan.json").write_text(json.dumps({
                "editorial_mode": editorial_mode,
                "source_video_id": str(metadata.get("id") or ""),
                "source_clip": source_clip,
                "editorial_plan": editorial_plan,
                "terminology": [asdict(item) for item in terminology],
                "transcript": [asdict(item) for item in cues],
                "trace": trace,
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
        elif source_clip is not None:
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

        if editorial_mode == "known_tech_interview_clip" and media_asset:
            media_path = Path(str(media_asset))
            if not media_path.is_absolute():
                media_path = self.workspace.root / media_path
            whisper_audit = targeted_whisper_caption_audit(
                media_path, unverified_cues, job,
            )
            if whisper_audit:
                trace.append(whisper_audit)
                repair_trace = translator.repair_audio_verified_cards(
                    cues, terminology, whisper_audit,
                )
                if repair_trace:
                    trace.append(repair_trace)
                if errors := terminology_contract_errors(cues, terminology):
                    raise ValueError(
                        "audio-verified captions violate terminology contract: "
                        + "; ".join(errors)
                    )
                if errors := interview_caption_duration_errors(cues):
                    raise ValueError(
                        "audio-verified caption policy gate failed: "
                        + "; ".join(errors)
                    )

        persist_translation_plan()
        manifest = build_collection_manifest(
            candidate, metadata, cues, terminology, editorial_plan, media_asset, subtitle_asset,
            media_info,
        )
        if render:
            YouTubeCollectionRenderer(self.workspace).render(manifest)
            self._link_superseded_collection(manifest)
        checks = validate_collection(manifest, self.workspace.root)
        manifest.quality_checks = [item.to_dict() for item in checks]
        path = self.workspace.save_collection_manifest(manifest)
        job_manifest = job / "collection-manifest.json"
        shutil.copy2(path, job_manifest)
        result = {
            "status": "completed", "source_type": "youtube", "candidate": candidate.id,
            "editorial_mode": editorial_mode,
            "collection_manifest": str(job_manifest), "collection_id": manifest.id,
            "items": [{"id": item.id, "title": item.title, "duration": item.duration} for item in manifest.items],
            "translation_trace": trace, "checks": [item.to_dict() for item in checks],
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
