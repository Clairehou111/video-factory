from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol
from urllib.parse import parse_qs, quote, urlparse, urlunparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from .factory import GenerateOptions, VideoFactory
from .models import ContentType, InformationRenderProfile, TopicType
from .observability import Observability
from .openrouter import DISCOUNTS_READER, ENDPOINTS_API, MODELS_API, parse_discounted_models
from .serde import load_collection_manifest
from .self_audit import ProblemLedger, ProblemObservation
from .storage import Workspace
from .youtube import DiscoveryConfig as YouTubeDiscoveryConfig
from .youtube import (
    SourceBelow1080Error, YouTubeAcquisitionError, YouTubeCollectionRenderer,
    YouTubeDiscoveryService, YouTubeWebAuthRequired, it_software_ai_markers,
    political_markers, technical_share_markers, youtube_editorial_subject_text,
    validate_collection,
)


class DiscoveryChannel(StrEnum):
    X = "x"
    GITHUB = "github"
    PROJECTS = "projects"
    ROBOTICS = "robotics"
    AUTONOMOUS_DRIVING = "autonomous_driving"
    # Keep the original names for the English channels so existing configs and
    # discovery state remain readable. Chinese sources have their own cadence,
    # candidate budget, quality gate, and selection slot.
    NEWS = "news"
    NEWS_ZH = "news_zh"
    OFFICIAL = "official"
    OFFICIAL_ZH = "official_zh"
    PAPER = "paper"
    YOUTUBE = "youtube"
    OPENROUTER = "openrouter"


DEFAULT_CADENCE_HOURS = {
    DiscoveryChannel.X: 2,
    DiscoveryChannel.GITHUB: 48,
    DiscoveryChannel.PROJECTS: 4,
    DiscoveryChannel.ROBOTICS: 4,
    DiscoveryChannel.AUTONOMOUS_DRIVING: 4,
    DiscoveryChannel.NEWS: 2,
    DiscoveryChannel.NEWS_ZH: 2,
    DiscoveryChannel.OFFICIAL: 2,
    DiscoveryChannel.OFFICIAL_ZH: 2,
    DiscoveryChannel.PAPER: 24,
    DiscoveryChannel.YOUTUBE: 2,
    DiscoveryChannel.OPENROUTER: 2,
}
DEFAULT_LOOKBACK_HOURS = {
    DiscoveryChannel.X: 6,
    DiscoveryChannel.GITHUB: 24 * 14,
    DiscoveryChannel.PROJECTS: 24,
    # Robotics news and official demos often publish on a weekly cadence.  A
    # one-day window made the factory miss strong, still-current events after
    # a weekend or a temporarily blocked local run.  Event de-duplication
    # keeps the wider bootstrap window from producing repeats.
    DiscoveryChannel.ROBOTICS: 24 * 7,
    DiscoveryChannel.AUTONOMOUS_DRIVING: 24 * 7,
    # News syndication can lag the original publication by several hours and
    # a 12-hour window missed an overnight US report before the Tokyo morning
    # search could resolve it. Event dedupe prevents the wider window from
    # generating the same event twice.
    DiscoveryChannel.NEWS: 24,
    DiscoveryChannel.NEWS_ZH: 12,
    # A Tokyo evening review can see a US midnight-dated release more than a
    # day after its RSS timestamp. Event de-duplication prevents the wider
    # window from selecting the same official announcement twice.
    DiscoveryChannel.OFFICIAL: 48,
    DiscoveryChannel.OFFICIAL_ZH: 12,
    DiscoveryChannel.PAPER: 24 * 3,
    DiscoveryChannel.YOUTUBE: 24 * 30,
    DiscoveryChannel.OPENROUTER: 24,
}
DEFAULT_QUERIES = {
    DiscoveryChannel.X: [
        'AI (agent OR SDK OR API OR model OR benchmark OR funding OR launch)',
        '("new AI model" OR "new model" OR "model release") (API OR developers OR "early access")',
        '("open source" OR paper OR product) AI',
        '("fastest-growing" OR viral OR "million users" OR ARR OR unicorn) (AI product OR startup)',
        '(robotics OR humanoid OR "physical AI" OR robotaxi OR "autonomous driving") (launch OR demo OR deploy)',
    ],
    DiscoveryChannel.GITHUB: [
        "AI agent created:>{date}",
        "LLM tool created:>{date}",
        "agent SDK created:>{date}",
        "AI app demo created:>{date}",
        "robotics created:>{date}",
        '"autonomous driving" created:>{date}',
    ],
    DiscoveryChannel.PROJECTS: [
        'breakout AI startup fastest-growing product users revenue',
        'AI startup (ARR OR million users OR unicorn OR viral growth)',
        'AI product startup major funding adoption launch',
    ],
    DiscoveryChannel.ROBOTICS: [
        '(robotics OR "humanoid robot" OR "physical AI") ("first ever" OR record OR autonomous OR "uncut demo" OR deployed OR ships)',
        'robot (home OR school OR hospital OR warehouse OR farm OR delivery) (autonomous OR trial OR deployed OR demo)',
    ],
    DiscoveryChannel.AUTONOMOUS_DRIVING: [
        '(robotaxi OR "autonomous driving") (launch OR highway OR airport OR "million miles" OR "without safety driver")',
        'self-driving car engineering safety field test deployment',
    ],
    DiscoveryChannel.NEWS: [
        "AI model launch OR agent API",
        '"new AI model" (developers OR API OR "early access" OR benchmark)',
        "AI startup funding OR founding team",
        "AI research benchmark",
        '(OpenAI OR Anthropic OR Google OR Meta OR xAI) (Mac OR GPU OR chip OR compute OR data center) (buy OR rent OR shortage OR training OR inference)',
    ],
    DiscoveryChannel.NEWS_ZH: [
        "(DeepSeek OR 智谱 OR GLM OR Kimi OR 通义千问 OR Qwen OR 豆包 OR 混元 OR 文心 OR MiniMax OR 阶跃星辰) (新模型 OR 新产品 OR 发布 OR 开源)",
        "(大模型 OR AI 模型) (降价 OR 涨价 OR 调价 OR 价格战 OR 对标 OR 超越 OR 争议)",
    ],
    DiscoveryChannel.OFFICIAL: [
        "AI model product launch",
        "agent SDK API announcement",
        "research benchmark release",
    ],
    DiscoveryChannel.OFFICIAL_ZH: [
        "(新模型 OR 新产品 OR 发布 OR 上线 OR 开源 OR 开放权重)",
        "(价格 OR 降价 OR 涨价 OR 调价 OR 免费 OR 套餐) (模型 OR API OR token)",
        "(API OR 上下文 OR 多模态 OR 智能体 OR Agent OR 工具调用) (升级 OR 发布 OR 开放)",
        "(下线 OR 停服 OR 迁移 OR 弃用 OR 基准 OR 榜单 OR 评测 OR 安全事件)",
    ],
    DiscoveryChannel.PAPER: [
        'all:"AI agent"',
        'all:"large language model" AND (all:benchmark OR all:reasoning)',
        'all:"embodied AI" OR all:"robot learning"',
        'all:"autonomous driving" AND (all:benchmark OR all:deployment)',
    ],
}
CHINESE_LLM_OFFICIAL_DOMAINS = [
    "deepseek.com", "zhipuai.cn", "bigmodel.cn", "z.ai", "moonshot.cn", "kimi.com",
    "qwen.ai", "aliyun.com", "volcengine.com", "doubao.com", "hunyuan.tencent.com",
    "cloud.tencent.com", "qianfan.cloud.baidu.com", "cloud.baidu.com", "minimaxi.com",
    "minimax.io", "stepfun.com", "baichuan-ai.com", "01.ai", "sensenova.cn",
    "sensetime.com", "xfyun.cn", "huaweicloud.com",
]
DEFAULT_OFFICIAL_DOMAINS = [
    "openai.com", "anthropic.com", "deepmind.google", "ai.google", "meta.com",
    "microsoft.com", "x.ai", "mistral.ai", "cohere.com", "nvidia.com",
    "aws.amazon.com", "huggingface.co", "typesafe.ai",
]
DEFAULT_OFFICIAL_FEEDS = [
    "https://openai.com/news/rss.xml",
    "https://deepmind.google/blog/rss.xml",
    "https://huggingface.co/blog/feed.xml",
]
DEFAULT_TRUSTED_NEWS = [
    "reuters.com", "bloomberg.com", "ft.com", "techcrunch.com", "theverge.com",
    "wired.com", "arstechnica.com", "axios.com", "nytimes.com", "theinformation.com",
    "tomshardware.com", "9to5mac.com",
]
DEFAULT_NEWS_FEEDS: list[str] = []
DISABLED_DISCOVERY_DOMAINS = {"macrumors.com"}
DEFAULT_TRUSTED_NEWS_ZH = [
    "36kr.com", "caixin.com", "jiemian.com", "yicai.com", "cls.cn", "stcn.com",
    "thepaper.cn", "tmtpost.com", "leiphone.com", "qbitai.com", "jiqizhixin.com",
]
DEFAULT_PROJECT_DOMAINS = [
    "reuters.com", "bloomberg.com", "ft.com", "techcrunch.com", "sifted.eu",
    "forbes.com", "cnbc.com", "venturebeat.com",
]
DEFAULT_PROJECT_FEEDS = [
    "https://techcrunch.com/category/startups/feed/",
    "https://sifted.eu/feed",
]
DEFAULT_ROBOTICS_DOMAINS = [
    "therobotreport.com", "spectrum.ieee.org", "robohub.org", "techcrunch.com",
    "reuters.com", "electrek.co", "autonomousvehicleinternational.com",
    "waymo.com", "bostondynamics.com", "figure.ai", "agilityrobotics.com",
    "1x.tech", "physicalintelligence.company", "skild.ai", "fieldai.com",
    "apptronik.com", "unitree.com", "sunday.ai", "robocup.org", "aihub.org",
    "zoox.com", "nuro.ai", "tesla.com",
]
DEFAULT_ROBOTICS_FEEDS = [
    "https://www.therobotreport.com/feed/",
    "https://spectrum.ieee.org/feeds/topic/robotics.rss",
    "https://robohub.org/feed/",
    "https://waymo.com/blog/rss.xml",
    "https://electrek.co/guides/autonomous-driving/feed/",
    "https://www.autonomousvehicleinternational.com/feed/",
]
DEFAULT_AUTONOMOUS_DRIVING_DOMAINS = [
    "waymo.com", "zoox.com", "nuro.ai", "tesla.com", "electrek.co",
    "autonomousvehicleinternational.com", "reuters.com", "techcrunch.com",
    "kodiak.ai", "aurora.tech", "maymobility.com",
]
DEFAULT_AUTONOMOUS_DRIVING_FEEDS = [
    "https://waymo.com/blog/rss.xml",
    "https://electrek.co/guides/autonomous-driving/feed/",
    "https://www.autonomousvehicleinternational.com/feed/",
]
DEFAULT_X_ACCOUNTS = [
    "OpenAI", "AnthropicAI", "GoogleDeepMind", "MetaAI", "MistralAI",
    "Cohere", "huggingface", "karpathy",
]

CHINESE_LLM_PROVIDER_MARKERS = (
    "deepseek", "深度求索", "智谱", "glm", "kimi", "月之暗面", "通义千问", "qwen",
    "豆包", "火山引擎", "混元", "文心", "千帆", "minimax", "阶跃星辰", "stepfun",
    "百川", "零一万物", "01.ai", "日日新", "商汤", "讯飞星火", "盘古",
)
CHINESE_LLM_HIGH_VALUE_MARKERS = (
    "新模型", "模型发布", "模型上线", "新产品", "产品发布", "正式发布", "开放权重",
    "开源", "降价", "涨价", "调价", "价格战", "免费", "套餐", "api 发布", "api升级",
    "api 升级", "上下文", "多模态", "工具调用", "智能体", "agent", "下线", "停服",
    "迁移", "弃用", "基准", "benchmark", "榜单", "评测", "超越", "对标", "回应",
    "争议", "冲突", "故障", "宕机", "安全事件", "泄露", "license", "许可证",
)
CHINESE_LLM_PRICE_MARKERS = (
    "降价", "涨价", "调价", "价格战", "免费", "优惠", "折扣", "套餐", "token 价格",
    "token价格", "调用价格",
)

NEWS_CHANNELS = {
    DiscoveryChannel.NEWS, DiscoveryChannel.NEWS_ZH,
    DiscoveryChannel.PROJECTS, DiscoveryChannel.ROBOTICS,
    DiscoveryChannel.AUTONOMOUS_DRIVING,
}
OFFICIAL_CHANNELS = {DiscoveryChannel.OFFICIAL, DiscoveryChannel.OFFICIAL_ZH}
WEB_DISCOVERY_CHANNELS = NEWS_CHANNELS | OFFICIAL_CHANNELS
CHINESE_DISCOVERY_CHANNELS = {DiscoveryChannel.NEWS_ZH, DiscoveryChannel.OFFICIAL_ZH}


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_date(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    try:
        if re.fullmatch(r"\d{8}", value):
            return datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        try:
            return parsedate_to_datetime(value).astimezone(UTC)
        except (TypeError, ValueError):
            return None


def canonical_url(value: str) -> str:
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").casefold()
    if host in {"twitter.com", "www.twitter.com", "www.x.com"}:
        host = "x.com"
    if host == "www.github.com":
        host = "github.com"
    if host in {"www.youtube.com", "m.youtube.com"}:
        host = "youtube.com"
    path = parsed.path.rstrip("/") or "/"
    query = parsed.query if host in {"youtube.com", "youtu.be"} else ""
    return urlunparse((parsed.scheme or "https", host, path, "", query, ""))


def _discovery_domain_disabled(value: str) -> bool:
    parsed = urlparse(value if "://" in value else "https://" + value)
    host = (parsed.hostname or "").casefold()
    return any(host == domain or host.endswith("." + domain) for domain in DISABLED_DISCOVERY_DOMAINS)


def source_identity(value: str) -> tuple[str, str]:
    """Return a stable identity even when a platform canonicalizes its URL.

    X commonly rewrites ``/i/status/<id>`` to ``/<author>/status/<id>`` after
    acquisition. Treating those URLs as different made deterministic browser
    failures restart the expensive director workflow.
    """
    normalized = canonical_url(value)
    parsed = urlparse(normalized)
    host = parsed.hostname or ""
    if host == "x.com":
        match = re.search(r"/status/(\d+)(?:/|$)", parsed.path)
        if match:
            return "x_status", match.group(1)
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
        if video_id:
            return "youtube", video_id
    if host == "youtube.com":
        video_id = (parse_qs(parsed.query).get("v") or [""])[0]
        if video_id:
            return "youtube", video_id
        match = re.match(r"/(?:shorts|live|embed)/([^/?]+)", parsed.path)
        if match:
            return "youtube", match.group(1)
    return "url", normalized


def same_source(left: str, right: str) -> bool:
    return source_identity(left) == source_identity(right)


def _rerender_requires_full_regeneration(error: BaseException) -> bool:
    """Identify cached manifests whose material contract cannot be repaired by rerendering."""
    message = f"{type(error).__name__}: {error}".casefold()
    return any(marker in message for marker in (
        "does not cite an archived image asset",
        "unidentifiedimageerror",
        "cannot identify image file",
    ))


def _retryable_adoption_error(error: BaseException) -> bool:
    """Keep temporary source outages out of the human-repair queue.

    A YouTube metadata or media fetch can fail before any editorial work is
    possible.  TLS disconnects, timeouts, throttling, and upstream 5xx errors
    are operational outages, not evidence that the candidate itself needs a
    human decision.  Permanent source-contract failures still fail closed.
    """
    if not isinstance(error, YouTubeAcquisitionError):
        return False
    if isinstance(error, (SourceBelow1080Error, YouTubeWebAuthRequired)):
        return False
    detail = f"{type(error).__name__}: {error}".casefold()
    permanent_markers = (
        "metadata has no video id", "transcript is empty", "must use yt-dlp json3",
        "has no audio track", "below 1080", "source quality",
    )
    if any(marker in detail for marker in permanent_markers):
        return False
    return any(marker in detail for marker in (
        "ssl", "unexpected_eof", "timed out", "timeout", "temporary failure",
        "connection reset", "connection aborted", "remote end closed",
        "network lookup", "name resolution", "http error 429", "too many requests",
        "http error 500", "http error 502", "http error 503", "http error 504",
        "unable to download webpage", "unable to download api page",
    ))


@dataclass(slots=True)
class ChannelConfig:
    enabled: bool = True
    cadence_hours: int = 24
    lookback_hours: int = 24
    minimum_score: float = 70.0
    max_candidates: int = 20
    probe_limit: int = 8
    queries: list[str] = field(default_factory=list)
    seed_accounts: list[str] = field(default_factory=list)
    seed_domains: list[str] = field(default_factory=list)
    feeds: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, channel: DiscoveryChannel, data: dict[str, Any]) -> "ChannelConfig":
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unsupported {channel.value} discovery config fields: {', '.join(sorted(unknown))}")
        values = dict(data)
        values.setdefault("cadence_hours", DEFAULT_CADENCE_HOURS[channel])
        values.setdefault("lookback_hours", DEFAULT_LOOKBACK_HOURS[channel])
        values.setdefault("queries", list(DEFAULT_QUERIES.get(channel, [])))
        if channel == DiscoveryChannel.OPENROUTER:
            # OpenRouter remains available as an inference provider, but price
            # promotions are not an editorial discovery source by default.
            values.setdefault("enabled", False)
        if channel == DiscoveryChannel.X:
            values.setdefault("seed_accounts", list(DEFAULT_X_ACCOUNTS))
        if channel == DiscoveryChannel.NEWS:
            values.setdefault("seed_domains", list(DEFAULT_TRUSTED_NEWS))
            values.setdefault("feeds", list(DEFAULT_NEWS_FEEDS))
        if channel == DiscoveryChannel.NEWS_ZH:
            values.setdefault("seed_domains", list(DEFAULT_TRUSTED_NEWS_ZH))
        if channel == DiscoveryChannel.PROJECTS:
            values.setdefault("seed_domains", list(DEFAULT_PROJECT_DOMAINS))
            values.setdefault("feeds", list(DEFAULT_PROJECT_FEEDS))
        if channel == DiscoveryChannel.ROBOTICS:
            values.setdefault("seed_domains", list(DEFAULT_ROBOTICS_DOMAINS))
            values.setdefault("feeds", list(DEFAULT_ROBOTICS_FEEDS))
        if channel == DiscoveryChannel.AUTONOMOUS_DRIVING:
            values.setdefault("seed_domains", list(DEFAULT_AUTONOMOUS_DRIVING_DOMAINS))
            values.setdefault("feeds", list(DEFAULT_AUTONOMOUS_DRIVING_FEEDS))
        if channel == DiscoveryChannel.OFFICIAL:
            values.setdefault("seed_domains", list(DEFAULT_OFFICIAL_DOMAINS))
            values.setdefault("feeds", list(DEFAULT_OFFICIAL_FEEDS))
        if channel == DiscoveryChannel.OFFICIAL_ZH:
            values.setdefault("seed_domains", list(CHINESE_LLM_OFFICIAL_DOMAINS))
        values["seed_domains"] = [
            domain for domain in values.get("seed_domains", [])
            if not _discovery_domain_disabled(str(domain))
        ]
        values["feeds"] = [
            feed for feed in values.get("feeds", [])
            if not _discovery_domain_disabled(str(feed))
        ]
        return cls(**values)


@dataclass(slots=True)
class AdoptionPolicy:
    """Editorial admission policy, deliberately separate from source quality."""

    general_minimum_score: float = 75.0
    high_standard_minimum_score: float = 82.0
    youtube_minimum_score: float = 80.0
    minimum_audience_consequence: float = 12.0
    llm_priority_bonus: float = 6.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdoptionPolicy":
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "unsupported adoption policy fields: " + ", ".join(sorted(unknown))
            )
        policy = cls(**{key: float(value) for key, value in data.items()})
        for name in (
            "general_minimum_score", "high_standard_minimum_score",
            "youtube_minimum_score", "minimum_audience_consequence",
        ):
            value = float(getattr(policy, name))
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be between 0 and 100")
        if not 0 <= policy.llm_priority_bonus <= 25:
            raise ValueError("llm_priority_bonus must be between 0 and 25")
        if policy.high_standard_minimum_score < policy.general_minimum_score:
            raise ValueError(
                "high_standard_minimum_score cannot be lower than general_minimum_score"
            )
        return policy


@dataclass(slots=True)
class ResourceDiscoveryConfig:
    timezone: str = "Asia/Tokyo"
    retry_backoff_seconds: list[int] = field(default_factory=lambda: [0, 30, 120])
    blocked_retry_delay_hours: int = 6
    max_blocked_retry_runs: int = 2
    event_dedupe_days: int = 30
    adoption_policy: AdoptionPolicy = field(default_factory=AdoptionPolicy)
    channels: dict[DiscoveryChannel, ChannelConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.channels:
            self.channels = {
                channel: ChannelConfig.from_dict(channel, {}) for channel in DiscoveryChannel
            }
        if not self.retry_backoff_seconds or len(self.retry_backoff_seconds) > 5:
            raise ValueError("retry_backoff_seconds must contain one to five attempts")
        if any(value < 0 for value in self.retry_backoff_seconds):
            raise ValueError("retry backoff values cannot be negative")
        if self.blocked_retry_delay_hours < 1:
            raise ValueError("blocked_retry_delay_hours must be positive")
        if self.max_blocked_retry_runs < 1:
            raise ValueError("max_blocked_retry_runs must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceDiscoveryConfig":
        unknown = set(data) - {
            "timezone", "retry_backoff_seconds", "blocked_retry_delay_hours",
            "max_blocked_retry_runs", "event_dedupe_days", "adoption_policy", "channels",
        }
        if unknown:
            raise ValueError("unsupported resource discovery config fields: " + ", ".join(sorted(unknown)))
        raw_channels = data.get("channels") or {}
        unknown_channels = set(raw_channels) - {item.value for item in DiscoveryChannel}
        if unknown_channels:
            raise ValueError("unsupported discovery channels: " + ", ".join(sorted(unknown_channels)))
        channels = {
            channel: ChannelConfig.from_dict(channel, dict(raw_channels.get(channel.value) or {}))
            for channel in DiscoveryChannel
        }
        return cls(
            timezone=str(data.get("timezone") or "Asia/Tokyo"),
            retry_backoff_seconds=[int(item) for item in data.get("retry_backoff_seconds", [0, 30, 120])],
            blocked_retry_delay_hours=int(data.get("blocked_retry_delay_hours", 6)),
            max_blocked_retry_runs=int(data.get("max_blocked_retry_runs", 2)),
            event_dedupe_days=int(data.get("event_dedupe_days", 30)),
            adoption_policy=AdoptionPolicy.from_dict(dict(data.get("adoption_policy") or {})),
            channels=channels,
        )

    @classmethod
    def from_path(cls, path: Path | None) -> "ResourceDiscoveryConfig":
        return cls() if path is None else cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(slots=True)
class DiscoveryCandidate:
    id: str
    channel: DiscoveryChannel
    url: str
    title: str
    author: str = ""
    publisher: str = ""
    published_at: str = ""
    summary: str = ""
    body_text: str = ""
    stable_id: str = ""
    topic_type: TopicType | None = None
    content_type: ContentType | None = None
    discovery_bucket: str = ""
    event_key: str = ""
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    eligible: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "discovered"
    discovered_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiscoveryCandidate":
        data = dict(payload)
        data["channel"] = DiscoveryChannel(data["channel"])
        if data.get("topic_type"):
            data["topic_type"] = TopicType(data["topic_type"])
        if data.get("content_type"):
            data["content_type"] = ContentType(data["content_type"])
        return cls(**data)


@dataclass(slots=True)
class ChannelRun:
    channel: DiscoveryChannel
    status: str
    candidates: list[DiscoveryCandidate] = field(default_factory=list)
    selected: DiscoveryCandidate | None = None
    adoption: dict[str, Any] | None = None
    selections: list[DiscoveryCandidate] = field(default_factory=list)
    adoptions: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    next_run_at: str = ""
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResourceDiscoveryRun:
    id: str
    status: str
    started_at: str
    completed_at: str = ""
    channels: dict[str, ChannelRun] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiscoveryAdapter(Protocol):
    def search(self, config: ChannelConfig, now: datetime) -> list[DiscoveryCandidate]: ...


Runner = Callable[..., subprocess.CompletedProcess[str]]
Fetcher = Callable[[str], tuple[str, str]]


def _json_rows(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    for key in ("data", "items", "results", "entries"):
        rows = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(rows, list):
            return [dict(item) for item in rows if isinstance(item, dict)]
        if isinstance(rows, dict):
            for nested_key in ("tweets", "posts", "items", "results", "entries"):
                nested = rows.get(nested_key)
                if isinstance(nested, list):
                    return [dict(item) for item in nested if isinstance(item, dict)]
    return [dict(payload)] if isinstance(payload, dict) else []


class XDiscoveryAdapter:
    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner or subprocess.run

    def _run(self, primary: list[str], fallback: list[str]) -> list[dict[str, Any]]:
        errors: list[str] = []
        for command in (primary, primary, fallback):
            completed = self.runner(command, capture_output=True, text=True, timeout=90, check=False)
            if completed.returncode == 0:
                try:
                    payload = json.loads(completed.stdout)
                    if isinstance(payload, dict) and payload.get("ok") is False:
                        detail = payload.get("error") if isinstance(payload.get("error"), dict) else {}
                        errors.append(str(detail.get("message") or "Twitter backend returned ok=false"))
                        continue
                    return _json_rows(completed.stdout)
                except json.JSONDecodeError as error:
                    errors.append(f"invalid JSON: {error}")
            else:
                errors.append((completed.stderr or completed.stdout).strip())
        raise RuntimeError("X search failed: " + "; ".join(item for item in errors if item))

    def search(self, config: ChannelConfig, now: datetime) -> list[DiscoveryCandidate]:
        rows: list[dict[str, Any]] = []
        for query in config.queries:
            rows.extend(self._run(
                ["twitter", "search", query, "-n", str(config.max_candidates), "--json"],
                ["opencli", "twitter", "search", query, "-f", "json"],
            ))
        for account in config.seed_accounts:
            rows.extend(self._run(
                ["twitter", "user-posts", account.lstrip("@"), "-n", "12", "--json"],
                ["opencli", "twitter", "tweets", account.lstrip("@"), "-f", "json"],
            ))
        found: dict[str, DiscoveryCandidate] = {}
        for row in rows:
            post_id = str(row.get("id") or row.get("rest_id") or "").strip()
            author_data = row.get("author") if isinstance(row.get("author"), dict) else {}
            author = str(
                author_data.get("screenName") or author_data.get("username") or row.get("username")
                or row.get("screen_name") or row.get("author_handle") or ""
            ).lstrip("@")
            text = str(row.get("text") or row.get("full_text") or row.get("content") or "").strip()
            url = str(row.get("url") or row.get("tweet_url") or "")
            if not url and post_id and author:
                url = f"https://x.com/{author}/status/{post_id}"
            if not url or not text:
                continue
            media = row.get("media") or row.get("attachments") or []
            key = post_id or canonical_url(url)
            found[key] = DiscoveryCandidate(
                id=f"x-{post_id or hashlib.sha256(url.encode()).hexdigest()[:12]}",
                channel=DiscoveryChannel.X, url=canonical_url(url), title=text[:180],
                author=author, publisher="X", published_at=str(
                    row.get("createdAtISO") or row.get("created_at") or row.get("date") or ""
                ), summary=text, body_text=text, stable_id=f"x:{post_id}" if post_id else canonical_url(url),
                metadata={"media_count": len(media) if isinstance(media, list) else int(bool(media)), "metrics": row.get("metrics") or {}},
                discovered_at=_iso(now),
            )
        return list(found.values())


class GitHubDiscoveryAdapter:
    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner or subprocess.run

    def search(self, config: ChannelConfig, now: datetime) -> list[DiscoveryCandidate]:
        rough: dict[str, dict[str, Any]] = {}
        since = (now - timedelta(hours=config.lookback_hours)).date().isoformat()
        for template in config.queries:
            query = template.replace("{date}", since)
            command = [
                "gh", "search", "repos", query, "--sort", "updated", "--limit", str(config.max_candidates),
                "--json", "fullName,url,description,createdAt,updatedAt,pushedAt,stargazersCount,isArchived,owner",
            ]
            completed = self.runner(command, capture_output=True, text=True, timeout=90, check=False)
            if completed.returncode != 0:
                raise RuntimeError((completed.stderr or completed.stdout).strip() or "GitHub search failed")
            for row in _json_rows(completed.stdout):
                full_name = str(row.get("fullName") or row.get("nameWithOwner") or "")
                if not full_name or bool(row.get("isArchived")):
                    continue
                rough[full_name.casefold()] = row
        # A burst of independently created integrations is a useful early
        # adoption signal even before stars accumulate. Keep the signal
        # auditable and bounded to repositories returned by configured search.
        token_repositories: dict[str, set[str]] = {}
        ignored_tokens = {
            "agent", "agents", "application", "awesome", "demo", "examples",
            "framework", "github", "integration", "language", "machine", "model",
            "models", "open", "powered", "project", "projects", "repository", "sdk",
            "software", "system", "tool", "tools", "using", "with", "workflow",
            "workflows", "decision", "decisions", "typed",
        }
        row_tokens: dict[str, set[str]] = {}
        for key, row in rough.items():
            full_name = str(row.get("fullName") or row.get("nameWithOwner") or "")
            repository_name = full_name.split("/", 1)[-1]
            token_text = f"{repository_name} {row.get('description') or ''}".casefold()
            tokens = {
                token for token in re.findall(r"\b[a-z][a-z0-9-]{2,}\b", token_text)
                if token not in ignored_tokens and not token.isdigit()
            }
            row_tokens[key] = tokens
            for token in tokens:
                token_repositories.setdefault(token, set()).add(full_name.casefold())

        ranked = sorted(
            rough.values(),
            key=lambda row: (
                _parse_date(str(row.get("pushedAt") or row.get("updatedAt") or "")) or datetime.min.replace(tzinfo=UTC),
                int(row.get("stargazersCount") or 0),
            ),
            reverse=True,
        )[: config.probe_limit]
        found: dict[str, DiscoveryCandidate] = {}
        for row in ranked:
            full_name = str(row.get("fullName") or row.get("nameWithOwner") or "")
            velocity_subject, velocity_count = max(
                (
                    (token, len(token_repositories.get(token, set())))
                    for token in row_tokens.get(full_name.casefold(), set())
                ),
                key=lambda value: (value[1], value[0]),
                default=("", 0),
            )
            url = str(row.get("url") or f"https://github.com/{full_name}")
            readme_command = [
                "gh", "api", f"repos/{full_name}/readme", "-H", "Accept: application/vnd.github.raw+json",
            ]
            readme = self.runner(readme_command, capture_output=True, text=True, timeout=60, check=False)
            body = readme.stdout if readme.returncode == 0 else ""
            owner = row.get("owner") if isinstance(row.get("owner"), dict) else {}
            found[full_name.casefold()] = DiscoveryCandidate(
                id=f"github-{full_name.replace('/', '-')}", channel=DiscoveryChannel.GITHUB,
                url=canonical_url(url), title=full_name,
                author=str(owner.get("login") or full_name.split("/", 1)[0]), publisher="GitHub",
                published_at=str(row.get("pushedAt") or row.get("updatedAt") or row.get("createdAt") or ""),
                summary=str(row.get("description") or ""), body_text=body,
                stable_id=f"github:{full_name.casefold()}",
                metadata={
                    "stars": int(row.get("stargazersCount") or 0), "created_at": row.get("createdAt"),
                    "updated_at": row.get("updatedAt"), "readme_available": bool(body),
                    "related_repo_velocity": {
                        "subject": velocity_subject,
                        "repository_count": velocity_count,
                        "window_hours": config.lookback_hours,
                    } if velocity_count >= 3 else {},
                }, discovered_at=_iso(now),
            )
        return list(found.values())


def _default_fetcher(url: str) -> tuple[str, str]:
    direct = Request(url, headers={"User-Agent": "video-factory/0.1"})
    raw = b""
    resolved = url
    try:
        with urlopen(direct, timeout=45) as response:
            raw = response.read()
            resolved = response.geturl()
    except Exception:
        pass
    target = "https://r.jina.ai/http://" + resolved.split("://", 1)[-1]
    request = Request(target, headers={"User-Agent": "video-factory/0.1"})
    try:
        with urlopen(request, timeout=45) as response:
            return response.read().decode("utf-8", errors="replace"), resolved
    except Exception:
        if raw:
            return raw.decode("utf-8", errors="replace"), resolved
        raise


def _rss_rows(payload: bytes) -> list[dict[str, str]]:
    root = ElementTree.fromstring(payload)
    rows: list[dict[str, str]] = []
    for item in root.findall(".//item"):
        source = item.find("source")
        rows.append({
            "title": (item.findtext("title") or "").strip(),
            "url": (item.findtext("link") or "").strip(),
            "published_at": (item.findtext("pubDate") or "").strip(),
            "summary": (item.findtext("description") or "").strip(),
            "publisher": (source.text or "").strip() if source is not None else "",
            "publisher_url": str(source.attrib.get("url", "")) if source is not None else "",
        })
    atom = {"a": "http://www.w3.org/2005/Atom"}
    for item in root.findall(".//a:entry", atom):
        link = item.find("a:link", atom)
        rows.append({
            "title": (item.findtext("a:title", default="", namespaces=atom) or "").strip(),
            "url": str(link.attrib.get("href", "")) if link is not None else "",
            "published_at": (
                item.findtext("a:published", default="", namespaces=atom)
                or item.findtext("a:updated", default="", namespaces=atom) or ""
            ).strip(),
            "summary": (item.findtext("a:summary", default="", namespaces=atom) or "").strip(),
            "publisher": "", "publisher_url": "",
        })
    return rows


def extract_source_video_url(value: str) -> str:
    """Return one explicit downloadable source video URL from fetched page text."""
    decoded = html.unescape(value).replace("\\/", "/")
    youtube = re.search(
        r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?[^\s\]\)\"'<>]*v=|embed/)|youtu\.be/)"
        r"([A-Za-z0-9_-]{6,})",
        decoded,
        re.IGNORECASE,
    )
    if youtube:
        return f"https://www.youtube.com/watch?v={youtube.group(1)}"
    direct = re.search(r"https?://[^\s\]\)\"'<>]+\.(?:mp4|webm)(?:\?[^\s\]\)\"'<>]*)?", decoded, re.I)
    return direct.group(0) if direct else ""


class RSSDiscoveryAdapter:
    def __init__(self, channel: DiscoveryChannel, fetcher: Fetcher | None = None) -> None:
        if channel not in WEB_DISCOVERY_CHANNELS:
            raise ValueError("RSS adapter supports news, projects, robotics, and official channels")
        self.channel = channel
        self.fetcher = fetcher or _default_fetcher
        self.last_trace: dict[str, Any] = {}

    @staticmethod
    def _download(url: str) -> bytes:
        request = Request(url, headers={"User-Agent": "video-factory/0.1"})
        try:
            with urlopen(request, timeout=40) as response:
                return response.read()
        except Exception as first_error:
            fallback = subprocess.run([
                "curl", "-fsSL", "--retry", "2", "--retry-all-errors",
                "--connect-timeout", "15", "--max-time", "45",
                "-A", "video-factory/0.1", url,
            ], capture_output=True)
            if fallback.returncode == 0 and fallback.stdout:
                return fallback.stdout
            detail = fallback.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"RSS download failed via urllib ({type(first_error).__name__}) and curl: {detail}"
            ) from first_error

    @staticmethod
    def _google_news_url(query: str) -> str:
        if re.search(r"[\u3400-\u9fff]", query):
            return (
                "https://news.google.com/rss/search?q=" + quote(query)
                + "&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
            )
        return (
            "https://news.google.com/rss/search?q=" + quote(query)
            + "&hl=en-US&gl=US&ceid=US:en"
        )

    def search(self, config: ChannelConfig, now: datetime) -> list[DiscoveryCandidate]:
        rows: list[dict[str, str]] = []
        urls: list[str] = list(config.feeds)
        source_kinds: dict[str, str] = {url: "curated_feed" for url in config.feeds}
        query_by_url: dict[str, str] = {}
        for query in config.queries:
            restricted = query
            if self.channel in OFFICIAL_CHANNELS and config.seed_domains:
                restricted = f"{query} ({' OR '.join('site:' + item for item in config.seed_domains)})"
            query_url = self._google_news_url(restricted)
            urls.append(query_url)
            source_kinds[query_url] = "google_news_query"
            query_by_url[query_url] = query
        self.last_trace = {
            "adapter": "rss",
            "channel": self.channel.value,
            "started_at": _iso(now),
            "configured_queries": list(config.queries),
            "configured_feeds": list(config.feeds),
            "trusted_domains": list(config.seed_domains),
            "sources": [],
            "funnel": {},
        }
        def download_source(url: str) -> tuple[str, list[dict[str, str]] | None, Exception | None]:
            try:
                return url, _rss_rows(self._download(url)), None
            except Exception as error:
                return url, None, error

        failures: list[str] = []
        successful_sources = 0
        workers = min(6, len(urls)) or 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            downloaded = list(pool.map(download_source, urls))
        for url, feed_rows, error in downloaded:
            if error is None and feed_rows is not None:
                successful_sources += 1
                self.last_trace["sources"].append({
                    "kind": source_kinds.get(url, "feed"),
                    "url": url,
                    "query": query_by_url.get(url, ""),
                    "status": "ok",
                    "row_count": len(feed_rows),
                })
                feed_host = (urlparse(url).hostname or "").casefold()
                for row in feed_rows:
                    # Direct curated feeds usually omit RSS <source>. Preserve
                    # the feed origin so they pass the same explicit-domain
                    # provenance gate as Google News results.
                    row.setdefault("publisher_url", "")
                    row.setdefault("publisher", "")
                    if not row["publisher_url"]:
                        row["publisher_url"] = f"https://{feed_host}" if feed_host else url
                    if not row["publisher"]:
                        row["publisher"] = feed_host
                    row["discovery_source"] = url
                rows.extend(feed_rows)
            else:
                assert error is not None
                self.last_trace["sources"].append({
                    "kind": source_kinds.get(url, "feed"),
                    "url": url,
                    "query": query_by_url.get(url, ""),
                    "status": "failed",
                    "row_count": 0,
                    "error": f"{type(error).__name__}: {error}",
                })
                failures.append(f"{url}: {type(error).__name__}: {error}")
                if len(urls) == 1:
                    raise RuntimeError(f"RSS search failed: {type(error).__name__}: {error}") from error
        if urls and successful_sources == 0:
            summary = "; ".join(failures[:3])
            raise RuntimeError(
                f"RSS search failed for all {len(urls)} sources"
                + (f": {summary}" if summary else "")
            )
        unique_rows: dict[str, dict[str, str]] = {}
        for row in rows:
            if row.get("url"):
                unique_rows.setdefault(row["url"], row)
        ranked_rows = list(unique_rows.values())
        trusted_before = len(ranked_rows)
        rejected_untrusted: list[dict[str, str]] = []
        if self.channel in NEWS_CHANNELS and config.seed_domains:
            rejected_untrusted = [
                {
                    "title": row.get("title", "")[:240],
                    "publisher": row.get("publisher", "")[:120],
                    "publisher_url": row.get("publisher_url", "")[:500],
                    "reason": "publisher_not_in_trusted_domains",
                }
                for row in ranked_rows
                if not any(
                    (urlparse(row.get("publisher_url", "")).hostname or "").casefold() == domain
                    or (urlparse(row.get("publisher_url", "")).hostname or "").casefold().endswith("." + domain)
                    for domain in config.seed_domains
                )
            ][:12]
            ranked_rows = [
                row for row in ranked_rows
                if any(
                    (urlparse(row.get("publisher_url", "")).hostname or "").casefold() == domain
                    or (urlparse(row.get("publisher_url", "")).hostname or "").casefold().endswith("." + domain)
                    for domain in config.seed_domains
                )
            ]
        ranked_rows = self._fair_source_rows(
            ranked_rows, max(config.max_candidates, config.probe_limit),
        )
        probe_rows = ranked_rows[: min(len(ranked_rows), max(config.probe_limit, config.probe_limit * 2))]

        def fetch_row(row: dict[str, str]) -> tuple[dict[str, str], str, str, str]:
            raw_url = row["url"]
            try:
                body, resolved = self.fetcher(raw_url)
                fetch_status = "fetched"
            except Exception:
                body, resolved = row.get("summary", ""), raw_url
                fetch_status = "summary_fallback"
            return row, body, resolved, fetch_status

        with ThreadPoolExecutor(max_workers=min(6, len(probe_rows)) or 1) as pool:
            fetched_rows = list(pool.map(fetch_row, probe_rows))

        found: dict[str, DiscoveryCandidate] = {}
        fetch_status_counts: dict[str, int] = {}
        rejected_resolved_domain: list[dict[str, str]] = []
        for row, body, resolved, fetch_status in fetched_rows:
            fetch_status_counts[fetch_status] = fetch_status_counts.get(fetch_status, 0) + 1
            raw_url = row["url"]
            if not raw_url:
                continue
            publisher_url = row.get("publisher_url", "")
            publisher_host = (urlparse(publisher_url).hostname or "").casefold()
            if self.channel in NEWS_CHANNELS and config.seed_domains and not any(
                publisher_host == domain or publisher_host.endswith("." + domain) for domain in config.seed_domains
            ):
                continue
            final_url = canonical_url(resolved or raw_url)
            final_host = (urlparse(final_url).hostname or "").casefold()
            if _discovery_domain_disabled(final_url):
                rejected_resolved_domain.append({
                    "title": row.get("title", "")[:240],
                    "resolved_url": final_url[:500],
                    "reason": "source_domain_disabled",
                })
                continue
            if self.channel in WEB_DISCOVERY_CHANNELS and config.seed_domains and not any(
                final_host == domain or final_host.endswith("." + domain)
                for domain in config.seed_domains
            ):
                # Google News' <source> attribution is useful for ranking but
                # cannot turn an unresolved aggregator wrapper into first-party
                # or trusted-news evidence. The fetched final URL itself must
                # land on the configured source domain.
                rejected_resolved_domain.append({
                    "title": row.get("title", "")[:240],
                    "resolved_url": final_url[:500],
                    "reason": "resolved_url_not_in_trusted_domains",
                })
                continue
            digest = hashlib.sha256(final_url.encode()).hexdigest()[:12]
            source_video_url = extract_source_video_url(body)
            found[final_url] = DiscoveryCandidate(
                id=f"{self.channel.value}-{digest}", channel=self.channel, url=final_url,
                title=row["title"], author="", publisher=row.get("publisher") or final_host,
                published_at=row.get("published_at", ""), summary=row.get("summary", ""), body_text=body,
                stable_id=f"web:{final_url}", metadata={
                    "publisher_url": publisher_url, "image_count": len(re.findall(r"!\[[^]]*\]\([^)]*\)|https?://\S+\.(?:png|jpe?g|webp)", body, re.I)),
                    "source_video_url": source_video_url,
                    "source_class": "official" if self.channel in OFFICIAL_CHANNELS else "news",
                    "language": "zh" if self.channel in CHINESE_DISCOVERY_CHANNELS else "en",
                }, discovered_at=_iso(now),
            )
            if len(found) >= config.probe_limit:
                break
        self.last_trace["funnel"] = {
            "sources_planned": len(urls),
            "sources_succeeded": successful_sources,
            "sources_failed": len(failures),
            "rows_raw": len(rows),
            "rows_unique": trusted_before,
            "rows_trusted": len(ranked_rows),
            "rows_probed": len(probe_rows),
            "candidates_emitted": len(found),
            "fetch_status": fetch_status_counts,
        }
        self.last_trace["rejections"] = [*rejected_untrusted, *rejected_resolved_domain[:12]]
        emitted = list(found.values())
        # Repetition across independent publishers is an early viral signal.
        # Cluster only the bounded, fully resolved candidates so an aggregator
        # headline can never manufacture consensus.
        assign_event_clusters(emitted)
        cluster_publishers: dict[str, set[str]] = {}
        for item in emitted:
            publisher = (urlparse(item.url).hostname or item.publisher).casefold()
            cluster_publishers.setdefault(item.event_key, set()).add(publisher)
        for item in emitted:
            mention_count = len(cluster_publishers.get(item.event_key, set()))
            if mention_count >= 2:
                item.metadata["cross_source_mentions"] = mention_count
        return emitted

    @staticmethod
    def _fair_source_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
        """Take fresh rows round-robin so one busy publication cannot own every probe."""
        buckets: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            source = row.get("discovery_source") or row.get("publisher_url") or "unknown"
            buckets.setdefault(source, []).append(row)
        for bucket in buckets.values():
            bucket.sort(
                key=lambda row: _parse_date(row.get("published_at", ""))
                or datetime.min.replace(tzinfo=UTC),
                reverse=True,
            )
        selected: list[dict[str, str]] = []
        ordered_sources = sorted(buckets)
        while len(selected) < limit:
            added = False
            for source in ordered_sources:
                bucket = buckets[source]
                if bucket:
                    selected.append(bucket.pop(0))
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
        return selected


class PaperDiscoveryAdapter:
    def __init__(self, downloader: Callable[[str], bytes] | None = None) -> None:
        self.downloader = downloader or RSSDiscoveryAdapter._download

    def search(self, config: ChannelConfig, now: datetime) -> list[DiscoveryCandidate]:
        found: dict[str, DiscoveryCandidate] = {}
        for query in config.queries:
            url = (
                "https://export.arxiv.org/api/query?search_query=" + quote(query)
                + f"&start=0&max_results={config.max_candidates}&sortBy=submittedDate&sortOrder=descending"
            )
            rows = _rss_rows(self.downloader(url))
            for row in rows:
                raw_url = row["url"]
                match = re.search(r"arxiv\.org/abs/([^?#]+)", raw_url)
                if not match:
                    continue
                paper_id = match.group(1)
                pdf_url = f"https://arxiv.org/pdf/{paper_id}"
                found[paper_id] = DiscoveryCandidate(
                    id=f"paper-{paper_id.replace('/', '-')}", channel=DiscoveryChannel.PAPER,
                    url=pdf_url, title=re.sub(r"\s+", " ", row["title"]),
                    publisher="arXiv", published_at=row.get("published_at", ""),
                    summary=re.sub(r"\s+", " ", row.get("summary", "")),
                    body_text=re.sub(r"\s+", " ", row.get("summary", "")), stable_id=f"arxiv:{paper_id}",
                    metadata={"abstract_url": raw_url, "pdf_available": True}, discovered_at=_iso(now),
                )
        return list(found.values())


class YouTubeDiscoveryAdapter:
    def __init__(self, workspace: Workspace, service: YouTubeDiscoveryService | None = None) -> None:
        self.service = service or YouTubeDiscoveryService(workspace)
        self.last_trace: dict[str, Any] = {}

    def search(self, config: ChannelConfig, now: datetime) -> list[DiscoveryCandidate]:
        settings = dict(config.settings)
        settings.update({
            "cadence_hours": config.cadence_hours, "minimum_score": config.minimum_score,
            "lookback_days": max(1, config.lookback_hours // 24), "max_source_selections": 1,
        })
        if config.queries:
            settings["query_pools"] = {"resource_discovery": config.queries}
        yt_config = YouTubeDiscoveryConfig.from_dict(settings)
        rough = self.service._search(yt_config)
        probed = self.service._choose_probe_candidates(rough, yt_config)
        hydrated = [self.service._hydrate(item) for item in probed]
        all_pools = list(dict.fromkeys([
            *yt_config.query_pools.keys(), *yt_config.channel_sources.keys(),
        ]))
        pools_with_candidates = [
            pool for pool in all_pools
            if any(pool in item.matched_pools for item in rough)
        ]
        direct_pools = [
            pool for pool in pools_with_candidates
            if any(
                pool in item.matched_pools
                and f"channel:{pool}" in item.discovery_routes
                for item in rough
            )
        ]
        self.last_trace = dict(self.service.last_trace)
        self.last_trace["probe"] = {
            "configured_limit": yt_config.metadata_probe_limit,
            "effective_limit": max(
                yt_config.metadata_probe_limit,
                len(pools_with_candidates) + len(direct_pools),
            ),
            "probed": len(probed),
            "pool_coverage": {
                pool: {
                    "rough": sum(pool in item.matched_pools for item in rough),
                    "probed": sum(pool in item.matched_pools for item in probed),
                }
                for pool in all_pools
            },
            "candidates": [{
                "video_id": item.video_id, "title": item.title,
                "matched_pools": list(item.matched_pools),
                "discovery_routes": list(item.discovery_routes),
                "source_channel_url": item.source_channel_url,
                "channel_recency_rank": item.channel_recency_rank,
            } for item in probed],
        }
        found: list[DiscoveryCandidate] = []
        for item in hydrated:
            self.service._score(item, yt_config, now)
            transcript_available = bool(getattr(item, "transcript_available", False))
            subject_text = youtube_editorial_subject_text(
                item.title, item.description, item.chapters,
            )
            markers = technical_share_markers(subject_text)
            found.append(DiscoveryCandidate(
                id=f"youtube-{item.video_id}", channel=DiscoveryChannel.YOUTUBE, url=item.url,
                title=item.title, author=item.channel, publisher=item.channel, published_at=item.published_at,
                summary=item.description, body_text=item.description, stable_id=f"youtube:{item.video_id}",
                metadata={
                    "duration_seconds": item.duration_seconds, "view_count": item.view_count,
                    "chapters": item.chapters, "creators": item.creators,
                    "transcript_available": transcript_available,
                    "technical_share": item.editorial_mode == "technical_coverage",
                    "youtube_editorial_mode": item.editorial_mode,
                    "technical_markers": markers, "known_tech_people": item.matched_known_people,
                    "it_scope_markers": list(item.scope_markers),
                    "political_signals": item.political_signals,
                    "matched_pools": list(item.matched_pools),
                    "discovery_routes": list(item.discovery_routes),
                    "source_channel_url": item.source_channel_url,
                    "channel_recency_rank": item.channel_recency_rank,
                    "youtube_audience_score": item.score_breakdown.get("audience_value", 0),
                    "youtube_audience_breakdown": dict(item.audience_breakdown),
                    "youtube_audience_matches": dict(item.audience_matches),
                    "youtube_score": item.score,
                    "youtube_rejection_reasons": list(item.rejection_reasons),
                }, discovered_at=_iso(now),
            ))
        return found


class OpenRouterDiscountDiscoveryAdapter:
    """Find price anomalies, not routine promotions, in OpenRouter endpoints."""

    OFFICIAL_PRICE_URLS = {
        "deepseek": "https://api-docs.deepseek.com/quick_start/pricing",
    }

    def __init__(self, fetch: Callable[[str], bytes] | None = None) -> None:
        self.fetch = fetch or self._fetch

    @staticmethod
    def _fetch(url: str) -> bytes:
        request = Request(url, headers={"User-Agent": "video-factory/0.1"})
        with urlopen(request, timeout=45) as response:
            return response.read()

    def _json(self, url: str) -> dict[str, Any]:
        payload = json.loads(self.fetch(url).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object from {url}")
        return payload

    @staticmethod
    def _price(pricing: dict[str, Any], key: str) -> float:
        try:
            return float(pricing.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _workload_cost(cls, pricing: dict[str, Any]) -> float:
        return cls._price(pricing, "prompt") * 18_000 + cls._price(pricing, "completion") * 4_000

    @staticmethod
    def _provider_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold())

    @staticmethod
    def _use_case_zh(model_id: str, description: str) -> str:
        text = f"{model_id} {description}".casefold()
        uses: list[str] = []
        if any(marker in text for marker in ("document", "office productivity", "long-context")):
            uses.append("长文档与办公自动化")
        if any(marker in text for marker in ("coding", "software engineering", "code")):
            uses.append("代码草稿")
        if any(marker in text for marker in ("agent", "agentic", "workflow")):
            uses.append("Agent 工作流")
        if "multimodal" in text:
            uses.append("多模态任务")
        if "reasoning" in text:
            uses.append("推理批处理")
        return "、".join(dict.fromkeys(uses)) or "高频、成本敏感的开发任务"

    def _official_comparison(
        self, model_id: str, endpoints: list[dict[str, Any]], minimum_uptime: float,
    ) -> dict[str, Any] | None:
        owner = model_id.lstrip("~").split("/", 1)[0]
        owner_key = self._provider_key(owner)
        if owner not in self.OFFICIAL_PRICE_URLS:
            return None
        healthy = [
            row for row in endpoints
            if int(row.get("status") or 0) == 0
            and float(row.get("uptime_last_30m") or 0) >= minimum_uptime
            and isinstance(row.get("pricing"), dict)
        ]
        official = next((
            row for row in healthy
            if self._provider_key(str(row.get("provider_name") or "")) == owner_key
        ), None)
        alternatives = [
            row for row in healthy if row is not official
            and self._price(row.get("pricing") or {}, "prompt") > 0
            and self._price(row.get("pricing") or {}, "completion") > 0
        ]
        if official is None or not alternatives:
            return None
        best = min(alternatives, key=lambda row: self._workload_cost(row.get("pricing") or {}))
        official_pricing = official.get("pricing") or {}
        official_cost = self._workload_cost(official_pricing)
        best_cost = self._workload_cost(best.get("pricing") or {})
        if official_cost <= 0 or best_cost >= official_cost:
            return None
        peak_cost = official_cost
        for override in official_pricing.get("overrides") or []:
            if isinstance(override, dict):
                peak_cost = max(peak_cost, self._workload_cost(override))
        return {
            "official_provider": str(official.get("provider_name") or owner),
            "official_source_url": self.OFFICIAL_PRICE_URLS[owner],
            "official_prompt_per_m": round(self._price(official_pricing, "prompt") * 1_000_000, 6),
            "official_completion_per_m": round(self._price(official_pricing, "completion") * 1_000_000, 6),
            "official_workload_cost": official_cost,
            "official_peak_workload_cost": peak_cost,
            "alternative_provider": str(best.get("provider_name") or best.get("name") or ""),
            "alternative_prompt_per_m": round(
                self._price(best.get("pricing") or {}, "prompt") * 1_000_000, 6,
            ),
            "alternative_completion_per_m": round(
                self._price(best.get("pricing") or {}, "completion") * 1_000_000, 6,
            ),
            "alternative_workload_cost": best_cost,
            "alternative_uptime": float(best.get("uptime_last_30m") or 0),
            "savings_offpeak_percent": round((1 - best_cost / official_cost) * 100, 1),
            "savings_peak_percent": round((1 - best_cost / peak_cost) * 100, 1),
        }

    @staticmethod
    def _temptation(
        discount_percent: int, comparison: dict[str, Any] | None,
        age_days: float, quality: float, coding: float, workload_cost: float,
        settings: dict[str, Any],
    ) -> tuple[float, list[str]]:
        savings = float((comparison or {}).get("savings_offpeak_percent") or 0)
        new_days = float(settings.get("new_model_days", 21))
        reasons: list[str] = []
        base = max(discount_percent * 0.7, savings * 0.8)
        if age_days <= new_days:
            base += 20
        if quality >= 50 or coding >= 65:
            base += 15
        elif quality >= 40 or coding >= 50:
            base += 8
        if workload_cost <= float(settings.get("maximum_shock_workload_cost_usd", 0.01)):
            base += 10
        if savings >= float(settings.get("minimum_vendor_savings_percent", 50)):
            reasons.append("cheaper_than_official_vendor")
        if discount_percent >= float(settings.get("extreme_discount_percent", 75)):
            reasons.append("extreme_discount")
        if (
            age_days <= new_days
            and discount_percent >= float(settings.get("new_model_discount_percent", 50))
        ):
            reasons.append("new_model_launch_discount")
        score = round(min(100.0, base), 1)
        if score < float(settings.get("minimum_temptation_score", 70)):
            reasons = []
        return score, reasons

    def search(self, config: ChannelConfig, now: datetime) -> list[DiscoveryCandidate]:
        models = self._json(MODELS_API).get("data") or []
        if not isinstance(models, list):
            raise ValueError("OpenRouter Models API returned no model list")
        try:
            reader = self.fetch(str(config.settings.get("discounts_reader_url") or DISCOUNTS_READER))
            discounts = parse_discounted_models(reader.decode("utf-8", errors="replace"))
        except Exception:
            # Vendor-vs-official comparisons still work if the public list UI
            # or its read-only rendering is temporarily unavailable.
            discounts = {}
        minimum_discount = int(config.settings.get("probe_minimum_discount_percent", 50))
        discounted_ids = [
            model_id for model_id, percent in discounts.items()
            if percent >= minimum_discount and not model_id.endswith(":batch") and not model_id.startswith("~")
        ]
        vendor_ids = [
            str(row.get("id") or "") for row in models
            if isinstance(row, dict) and str(row.get("id") or "").split("/", 1)[0] in self.OFFICIAL_PRICE_URLS
            and not str(row.get("id") or "").endswith(":batch")
            and not str(row.get("id") or "").startswith("~")
        ][: int(config.settings.get("vendor_probe_limit", 6))]
        ids = list(dict.fromkeys([*discounted_ids[: config.probe_limit], *vendor_ids]))
        model_by_id = {
            str(row.get("id")): row for row in models
            if isinstance(row, dict) and row.get("id")
        }
        minimum_uptime = float(config.settings.get("minimum_endpoint_uptime", 99.0))
        found: list[DiscoveryCandidate] = []
        for model_id in ids:
            model = model_by_id.get(model_id)
            if not isinstance(model, dict):
                continue
            architecture = model.get("architecture") or {}
            if "text" not in (architecture.get("output_modalities") or []):
                continue
            try:
                endpoint_payload = self._json(ENDPOINTS_API.format(model_id=model_id))
            except Exception:
                continue
            endpoints = (endpoint_payload.get("data") or {}).get("endpoints") or []
            if not isinstance(endpoints, list):
                continue
            healthy = [
                row for row in endpoints if isinstance(row, dict)
                and int(row.get("status") or 0) == 0
                and float(row.get("uptime_last_30m") or 0) >= minimum_uptime
                and self._price(row.get("pricing") or {}, "prompt") > 0
                and self._price(row.get("pricing") or {}, "completion") > 0
            ]
            if not healthy:
                continue
            best = min(healthy, key=lambda row: self._workload_cost(row.get("pricing") or {}))
            pricing = best.get("pricing") or {}
            workload_cost = self._workload_cost(pricing)
            endpoint_discount = round(float(pricing.get("discount") or 0) * 100)
            discount = max(int(discounts.get(model_id) or 0), endpoint_discount)
            comparison = self._official_comparison(model_id, endpoints, minimum_uptime)
            created = datetime.fromtimestamp(float(model.get("created") or 0), tz=UTC)
            age_days = max(0.0, (now - created).total_seconds() / 86400)
            benchmark = (model.get("benchmarks") or {}).get("artificial_analysis") or {}
            quality = float(benchmark.get("intelligence_index") or 0)
            coding = float(benchmark.get("coding_index") or 0)
            if quality >= 55:
                quality_verdict = "clears_internal_story_quality_gate"
            elif coding >= 70:
                quality_verdict = "coding_evaluation_only"
            else:
                quality_verdict = "cheap_trial_only_below_story_quality_gate"
            temptation, attraction_reasons = self._temptation(
                discount, comparison, age_days, quality, coding, workload_cost, config.settings,
            )
            provider = str(best.get("provider_name") or best.get("name") or "OpenRouter endpoint")
            prompt_per_m = round(self._price(pricing, "prompt") * 1_000_000, 6)
            completion_per_m = round(self._price(pricing, "completion") * 1_000_000, 6)
            catalog_pricing = model.get("pricing") or {}
            page_prompt_per_m = round(self._price(catalog_pricing, "prompt") * 1_000_000, 6)
            page_completion_per_m = round(self._price(catalog_pricing, "completion") * 1_000_000, 6)
            display_prompt_per_m = page_prompt_per_m if discount and page_prompt_per_m else prompt_per_m
            display_completion_per_m = (
                page_completion_per_m if discount and page_completion_per_m else completion_per_m
            )
            video_workload_cost = display_prompt_per_m * 0.018 + display_completion_per_m * 0.004
            name = str(model.get("name") or model_id)
            description = str(model.get("description") or "")
            use_case_zh = self._use_case_zh(model_id, description)
            if comparison and "cheaper_than_official_vendor" in attraction_reasons:
                savings = float(comparison["savings_offpeak_percent"])
                title = f"{name}：OpenRouter 可靠线路比原厂谷时便宜 {savings:.0f}%"
                price_hook = f"比原厂谷时便宜{savings:.1f}%"
            else:
                title = f"{name}：OpenRouter {discount}% 折扣，典型调用约 ${workload_cost:.4f}"
                price_hook = f"{discount}%折扣"
            if quality_verdict == "clears_internal_story_quality_gate":
                editorial_verdict_zh = (
                    "偶尔查看 OpenRouter 折扣：真实任务 A/B 测试后，作为主力订阅的低价补充。"
                )
            else:
                editorial_verdict_zh = (
                    f"偶尔查看 OpenRouter 折扣：把{use_case_zh}的低价线路补进主力订阅。"
                )
            if comparison and "cheaper_than_official_vendor" in attraction_reasons:
                required_hook_zh = (
                    f"OpenRouter 同模型便宜{float(comparison['savings_offpeak_percent']):.1f}%："
                    f"{use_case_zh}多一个 Cheaper Choice。"
                )
                required_headline_zh = f"OpenRouter Cheaper Choice｜{name}"
            else:
                required_hook_zh = (
                    f"OpenRouter {discount}% off：{use_case_zh}多一个 Cheaper Choice。"
                )
                required_headline_zh = f"OpenRouter Cheaper Choice｜{name}"
            comparison_text = ""
            linked_sources: list[str] = []
            if comparison:
                linked_sources.append(str(comparison["official_source_url"]))
                cost_share = 100 - float(comparison["savings_offpeak_percent"])
                official_multiple = float(comparison["official_workload_cost"]) / max(
                    float(comparison["alternative_workload_cost"]), 1e-12,
                )
                comparison_text = (
                    f" The official-vendor endpoint costs ${comparison['official_prompt_per_m']:.3f}/M input "
                    f"and ${comparison['official_completion_per_m']:.3f}/M output off-peak. "
                    f"For the same workload, {provider} is {comparison['savings_offpeak_percent']:.1f}% cheaper "
                    f"off-peak and {comparison['savings_peak_percent']:.1f}% cheaper at peak rates. "
                    f"That alternative is {cost_share:.1f}% of the official off-peak cost; equivalently, "
                    f"the official off-peak route costs {official_multiple:.2f} times as much."
                )
            discount_text = (
                f" A {discount}% discount leaves {100 - discount}% of the listed price; "
                f"the listed price is {100 / max(100 - discount, 1):.2f} times the discounted price."
                if discount else ""
            )
            body = (
                f"OpenRouter currently lists {name} through the healthy {provider} endpoint at "
                f"${prompt_per_m:.4f} per million input tokens and ${completion_per_m:.4f} per million output tokens. "
                f"Its last-30-minute uptime is {float(best.get('uptime_last_30m') or 0):.3f}%. "
                f"A representative 18,000-input plus 4,000-output-token developer task costs about "
                f"${workload_cost:.6f}.{discount_text}{comparison_text} The listed model has a "
                f"{int(model.get('context_length') or 0):,}-token "
                f"context window, Artificial Analysis intelligence {quality:.1f}, and coding {coding:.1f}. "
                f"The editorial quality verdict is {quality_verdict}; the internal general story-writer "
                "intelligence gate is 55, so a low price must not be presented as proof of stronger capability. "
                f"The OpenRouter model description identifies suitable workloads as: {description} "
                f"The required concise Chinese closing is: {editorial_verdict_zh} "
                f"The required Chinese hook is: {required_hook_zh} "
                f"The required Chinese headline is: {required_headline_zh} "
                f"This observation was captured from the OpenRouter model and endpoint APIs at {_iso(now)}; "
                "endpoint availability and pricing can change, so the video must show the capture time and provider name."
            )
            fingerprint = hashlib.sha256(json.dumps({
                "model": model_id, "provider": provider, "prompt": prompt_per_m,
                "completion": completion_per_m, "discount": discount,
                "vendor_savings": (comparison or {}).get("savings_offpeak_percent"),
            }, sort_keys=True).encode()).hexdigest()[:12]
            found.append(DiscoveryCandidate(
                id=f"openrouter-{re.sub(r'[^a-z0-9]+', '-', model_id.casefold()).strip('-')}-{fingerprint}",
                channel=DiscoveryChannel.OPENROUTER, url=f"https://openrouter.ai/{model_id}",
                title=title, author="OpenRouter", publisher="OpenRouter", published_at=_iso(now),
                summary=body, body_text=body, stable_id=f"openrouter-price:{model_id}:{fingerprint}",
                topic_type=TopicType.MODEL_OR_PRODUCT, content_type=ContentType.FLASH,
                metadata={
                    "model_id": model_id, "model_created_at": _iso(created), "model_age_days": round(age_days, 2),
                    "provider": provider, "endpoint_uptime": float(best.get("uptime_last_30m") or 0),
                    "prompt_per_m": prompt_per_m, "completion_per_m": completion_per_m,
                    "workload_cost_usd": workload_cost, "discount_percent": discount,
                    "page_prompt_per_m": display_prompt_per_m,
                    "page_completion_per_m": display_completion_per_m,
                    "video_workload_cost_usd": video_workload_cost,
                    "intelligence_index": quality, "coding_index": coding,
                    "model_description": description, "use_case_zh": use_case_zh,
                    "quality_verdict": quality_verdict,
                    "editorial_verdict_zh": editorial_verdict_zh,
                    "required_hook_zh": required_hook_zh,
                    "required_headline_zh": required_headline_zh,
                    "official_comparison": comparison, "linked_sources": linked_sources,
                    "temptation_score": temptation, "attraction_reasons": attraction_reasons,
                    "compelling": bool(attraction_reasons), "visual_path": "openrouter_model_pricing_page",
                }, discovered_at=_iso(now),
            ))
        return sorted(found, key=lambda item: (-float(item.metadata["temptation_score"]), item.url))


def _route_candidate(item: DiscoveryCandidate) -> tuple[TopicType, ContentType]:
    if item.channel == DiscoveryChannel.OPENROUTER:
        return TopicType.MODEL_OR_PRODUCT, ContentType.FLASH
    text = f"{item.title}\n{item.summary}\n{item.body_text[:6000]}".casefold()
    url = item.url.casefold()
    if re.search(
        r"\b(?:raised?|raises|raising|funding|series\s+[a-f]|seed\s+round|post-money|valued\s+at)\b|"
        r"\$\s?\d+(?:\.\d+)?\s?(?:m|million|b|billion)\b|融资|创始团队|团队变动",
        text,
    ) or any(marker in text for marker in ("founding team", "acquisition")):
        topic = TopicType.COMPANY_OR_TEAM
    elif item.channel == DiscoveryChannel.PAPER or sum(marker in text for marker in (
        "technical report", "benchmark", "benchmark contamination", "dataset", "methodology",
        "double-blind", "evaluation", "evaluations", "experiment", "experimental results",
    )) >= 2:
        topic = TopicType.RESEARCH_OR_BENCHMARK
    elif any(marker in text for marker in ("founded", "company", "startup")) and item.channel in NEWS_CHANNELS:
        topic = TopicType.COMPANY_OR_TEAM
    elif any(marker in text for marker in (
        "新模型", "模型发布", "模型上线", "新产品", "产品发布",
    )):
        topic = TopicType.MODEL_OR_PRODUCT
    elif any(marker in text for marker in (
        "sdk", "api", "agent", "cli", "developer tool", "quick start", "install",
        "智能体", "开发工具", "工具调用",
    )):
        topic = TopicType.GITHUB_PROJECT if item.channel == DiscoveryChannel.GITHUB else TopicType.TOOL_SDK_AGENT
    elif any(marker in text for marker in (
        "model", "product", "available today", "launching", "introducing",
        "新模型", "模型发布", "模型上线", "新产品", "产品发布", "大模型",
    )):
        topic = TopicType.MODEL_OR_PRODUCT
    elif item.channel in {DiscoveryChannel.ROBOTICS, DiscoveryChannel.AUTONOMOUS_DRIVING}:
        topic = TopicType.MODEL_OR_PRODUCT
    elif item.channel == DiscoveryChannel.PROJECTS:
        topic = TopicType.COMPANY_OR_TEAM
    elif item.channel == DiscoveryChannel.YOUTUBE:
        topic = TopicType.EXPERT_TALK
    elif item.channel == DiscoveryChannel.GITHUB:
        topic = TopicType.GITHUB_PROJECT
    elif any(marker in url for marker in ("/news", "/changelog", "/announcement")) or item.channel in OFFICIAL_CHANNELS:
        topic = TopicType.OFFICIAL_ANNOUNCEMENT
    else:
        topic = TopicType.PRACTICE_POST if item.channel == DiscoveryChannel.X else TopicType.OFFICIAL_ANNOUNCEMENT
    if topic in {TopicType.PRACTICE_POST, TopicType.COMPANY_OR_TEAM, TopicType.OFFICIAL_ANNOUNCEMENT}:
        content = ContentType.FLASH
    elif topic == TopicType.RESEARCH_OR_BENCHMARK:
        content = ContentType.DEEP_DIVE
    else:
        content = ContentType.EXPLAINER
    return topic, content


COMMON_WORDS = {
    "about", "after", "agent", "agents", "announces", "announced", "introduces", "launches",
    "latest", "model", "models", "new", "official", "open", "release", "released", "research",
    "startup", "that", "their", "this", "using", "with", "from", "into", "your", "video",
}


def _tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9][a-z0-9.+-]{2,}|[\u4e00-\u9fff]{2,}", value.casefold())
        if token not in COMMON_WORDS
    }


def _similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def _facts(text: str) -> int:
    parts = [item.strip() for item in re.split(r"[\n.!?。！？;；]+", text) if len(item.strip()) >= 18]
    return len(parts)


def _roundup_primary_source(item: DiscoveryCandidate) -> str | None:
    """Prefer a subject-specific primary page over an unstable multi-item digest."""
    text = item.body_text or ""
    youtube_count = len(re.findall(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/", text, re.I))
    roundup = bool(re.search(
        r"\b(?:video friday|roundup|weekly (?:digest|selection)|week in robotics)\b",
        item.title, re.I,
    )) or youtube_count >= 3
    if not roundup:
        return None
    subject_tokens = {
        token for token in _tokens(item.title)
        if token not in {"friday", "meet", "weekly", "roundup", "robot", "robotics"}
        and len(token) >= 4
    }
    if not subject_tokens:
        return None
    raw_links = re.findall(r"https?://[^\s)\]>'\"]+", text)
    source_host = (urlparse(item.url).hostname or "").casefold()
    ranked: list[tuple[int, str]] = []
    for raw in dict.fromkeys(raw_links):
        url = html.unescape(raw).rstrip(".,;:")
        parsed = urlparse(url)
        host = (parsed.hostname or "").casefold()
        if parsed.scheme not in {"http", "https"} or not host or host == source_host:
            continue
        if host.endswith(("youtube.com", "youtu.be")) or "/tag/" in parsed.path.casefold():
            continue
        haystack = f"{host}{parsed.path}".casefold()
        matches = sum(token in haystack for token in subject_tokens)
        if matches:
            ranked.append((matches, canonical_url(url)))
    return min(ranked, key=lambda pair: (-pair[0], pair[1]))[1] if ranked else None


def _plain_html(value: str) -> str:
    text = re.sub(r"<br\s*/?>|</p>|</blockquote>|</h\d>", "\n", value, flags=re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def atomize_robotics_roundup(item: DiscoveryCandidate) -> list[DiscoveryCandidate]:
    """Split a multi-video robotics digest into independently ranked events.

    The parent article's headline used to make only its first named toy enter
    discovery, while later home, pharmacy, and industrial robots never became
    candidates.  Pair each embedded video with the next subject-specific link
    and carry only that local section into scoring and generation.
    """
    if item.channel != DiscoveryChannel.ROBOTICS:
        return [item]
    source = item.summary or ""
    videos = list(re.finditer(
        r"https?://(?:www\.)?(?:youtube\.com/(?:embed/|watch\?v=)|youtu\.be/)([A-Za-z0-9_-]{6,})[^\"'<>\s]*",
        source, re.I,
    ))
    if len(videos) < 2:
        return [item]
    links = list(re.finditer(
        r"<a\b[^>]*href=[\"'](https?://[^\"']+)[\"'][^>]*>(.*?)</a>",
        source, re.I | re.S,
    ))
    source_host = (urlparse(item.url).hostname or "").casefold()
    children: list[DiscoveryCandidate] = []
    pending: list[re.Match[str]] = []
    section_start = 0
    tokens = sorted(
        [(match.start(), "video", match) for match in videos]
        + [(match.start(), "link", match) for match in links],
        key=lambda row: row[0],
    )
    for _, kind, match in tokens:
        if kind == "video":
            pending.append(match)
            continue
        if not pending:
            continue
        primary = canonical_url(html.unescape(match.group(1)).rstrip(".,;:"))
        parsed = urlparse(primary)
        host = (parsed.hostname or "").casefold()
        label = _plain_html(match.group(2))
        if (
            not host or host == source_host or host.endswith(("youtube.com", "youtu.be"))
            or "/tag/" in parsed.path.casefold() or len(label) < 3
        ):
            continue
        first_video = pending[0]
        # Product descriptions often sit *before* the embed, while captions
        # and availability notes sit after it. Preserve the whole local block
        # since the preceding subject link; starting at the video URL silently
        # discarded the strongest evidence for later roundup entries.
        section = source[section_start:match.end()]
        section_text = _plain_html(section)
        video_id = first_video.group(1)
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        identity = hashlib.sha256(f"{item.stable_id}|{primary}|{video_id}".encode()).hexdigest()[:12]
        children.append(DiscoveryCandidate(
            id=f"{item.channel.value}-{identity}", channel=item.channel,
            url=primary, title=label,
            author=item.author, publisher=host.removeprefix("www."),
            published_at=item.published_at, summary=section_text,
            body_text=section_text, stable_id=f"roundup:{identity}",
            metadata={
                **item.metadata,
                "roundup_parent_url": item.url,
                "source_video_url": video_url,
                "source_video_id": video_id,
                "linked_sources": list(dict.fromkeys([
                    *(str(url) for url in item.metadata.get("linked_sources") or []), item.url,
                ])),
                "atomized_roundup_event": True,
            },
            discovered_at=item.discovered_at,
        ))
        pending = []
        section_start = match.end()
    return children if len(children) >= 2 else [item]


def _robotics_bucket(text: str) -> str:
    lower = text.casefold()
    autonomous_markers = (
        "autonomous driving", "self-driving", "self driving", "robotaxi", "driverless",
        "public road", "safety driver", "safety monitor", "waymo", "zoox", "nuro",
        "kodiak", "aurora", "vehicle", "truck", "highway", "rides per week",
    )
    return "autonomous_driving" if any(marker in lower for marker in autonomous_markers) else "robotics"


def _real_world_impact(text: str) -> tuple[int, str]:
    """Rank useful physical deployment above toys and presentation demos."""
    lower = text.casefold()
    deployed = any(marker in lower for marker in (
        "deployed", "deployment", "paid rides", "deliveries",
        "operating", "in service", "customer", "public road", "without a safety driver",
        "field test", "pilot", "trial", "已部署", "交付", "投入使用", "实地测试",
    ))
    daily_setting = any(marker in lower for marker in (
        "home", "household", "school", "hospital", "pharmacy", "warehouse", "factory",
        "industrial", "worksite", "farm", "delivery", "airport", "highway", "public road",
        "kitchen", "laundry", "shirt", "cleaning", "家务", "家庭", "医院", "药房",
        "仓库", "工厂", "工业", "道路",
    ))
    useful_task = any(marker in lower for marker in (
        "fold", "pick", "weigh", "grind", "package", "clean", "cook", "carry",
        "deliver", "load", "unload", "drive", "ride", "manipulation",
        "叠", "抓取", "称重", "研磨", "包装", "清洁", "搬运", "配送", "驾驶",
    ))
    physical = any(marker in lower for marker in (
        "robot", "humanoid", "biped", "quadruped", "vehicle", "robotaxi", "self-driving",
        "physical ai", "robotic", "manipulator", "gripper", "dual-arm", "dual arm",
        "omnidirectional base", "locomotion", "机器人", "机械臂", "自动驾驶",
    ))
    if physical and deployed and daily_setting:
        return 3, "real deployment in a useful daily-life or work setting"
    if physical and daily_setting and useful_task:
        return 2, "physical demonstration of a useful daily-life or work task"
    if physical:
        return 1, "physical prototype, toy, research, or entertainment demonstration"
    return 0, "no verified physical-world action"


def _demand_exceeded_capacity_signal(text: str) -> bool:
    """Require an explicit causal link between demand and service exhaustion."""
    lower = text.casefold().replace(",", "")
    return bool(
        re.search(
            r"\b(?:demand|traffic|requests?|signups?|users?)\b[^.!?]{0,100}"
            r"\b(?:overwhelm(?:ed|ing)?|exceed(?:ed|ing)?|outstrip(?:ped|ping)?|"
            r"capacity|outage|unavailable|unable to serve|rate limits?)\b",
            lower,
        )
        or re.search(
            r"\b(?:lost|lose|unable)\b[^.!?]{0,50}\b(?:serve|serving)\b[^.!?]{0,80}"
            r"\b(?:because|due to|from)\b[^.!?]{0,30}\b(?:demand|traffic|requests?|users?)\b",
            lower,
        )
    )


def _has_breakout_project_signal(text: str) -> bool:
    """Require Lovable-class traction instead of treating every launch as a trend."""
    lower = text.casefold().replace(",", "")
    if re.search(r"\b(?:unicorn|fastest[- ]growing)\b", lower):
        return True
    # Very early launches rarely have six-figure user or revenue numbers. A
    # demand spike that exhausts API/service capacity is nevertheless concrete
    # breakout evidence and was the decisive launch signal for Jev.
    if _demand_exceeded_capacity_signal(text):
        return True

    money_signals = re.findall(
        r"\$\s*(\d+(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)?\b"
        r"[^.!?。！？]{0,50}\b(arr|mrr|revenue|sales|funding|raised|valuation)\b",
        lower,
    )
    money_signals.extend(
        (amount, unit, metric)
        for metric, amount, unit in re.findall(
            r"\b(arr|mrr|revenue|sales|funding|raised|valuation)\b"
            r"[^.!?。！？]{0,50}\$\s*(\d+(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)?\b",
            lower,
        )
    )
    for amount, unit, metric in money_signals:
        value = float(amount) * {
            "": 1, "k": 1_000, "thousand": 1_000,
            "m": 1_000_000, "million": 1_000_000,
            "b": 1_000_000_000, "billion": 1_000_000_000,
        }[unit]
        threshold = 10_000_000 if metric in {"funding", "raised", "valuation"} else 1_000_000
        if value >= threshold:
            return True

    for amount, unit, metric in re.findall(
        r"\b(\d+(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)?\s+"
        r"(users|customers|downloads|installs|signups|waitlist|stars)\b",
        lower,
    ):
        value = float(amount) * {
            "": 1, "k": 1_000, "thousand": 1_000,
            "m": 1_000_000, "million": 1_000_000,
            "b": 1_000_000_000, "billion": 1_000_000_000,
        }[unit]
        threshold = 1_000 if metric == "customers" else 10_000 if metric == "stars" else 100_000
        if value >= threshold:
            return True

    growth_rates = re.findall(
        r"(\d+(?:\.\d+)?)\s*%[^.!?。！？]{0,30}\b(?:growth|grew|increase|增长)\b", lower,
    )
    growth_rates.extend(re.findall(
        r"\b(?:growth|grew|increase|增长)\b[^.!?。！？]{0,30}(\d+(?:\.\d+)?)\s*%", lower,
    ))
    return any(
        float(percent) >= 100
        for percent in growth_rates
    )


def _is_substantive_model_launch(text: str) -> bool:
    """Recognize model releases before source-channel-specific startup gates."""
    lower = re.sub(r"\s+", " ", text.casefold())
    model = bool(re.search(
        r"\b(?:ai|frontier|foundation|reasoning|language|transformer(?:-based)?|system one) "
        r"models?\b|\bllms?\b|大模型|基础模型|推理模型|语言模型",
        lower,
    ))
    release = bool(re.search(
        r"\b(?:launch(?:ed|es|ing)?|release[ds]?|introduc(?:e|ed|es|ing)|"
        r"available|early access|public model|open(?:ed|ing)? access)\b|发布|上线|开放|公测",
        lower,
    ))
    consequence = bool(re.search(
        r"\b(?:api|developers?|software|automation|workflow|inference|latency|"
        r"benchmark|pricing|cost|faster|cheaper|structured outputs?)\b|"
        r"开发者|软件|自动化|工作流|推理|延迟|基准|价格|成本",
        lower,
    ))
    return model and release and consequence


def _viral_attention_signals(item: DiscoveryCandidate, text: str) -> list[str]:
    """Return concrete, inspectable evidence of unusually fast attention."""
    signals: list[str] = []
    if _demand_exceeded_capacity_signal(text):
        signals.append("demand_exceeded_service_capacity")
    velocity = item.metadata.get("related_repo_velocity")
    if isinstance(velocity, dict) and int(velocity.get("repository_count") or 0) >= 3:
        signals.append("rapid_integration_repository_growth")
    if int(item.metadata.get("cross_source_mentions") or 0) >= 3:
        signals.append("cross_source_repetition")
    return signals


def _is_routine_product_iteration(text: str) -> bool:
    """Reject an explicitly uneventful maintenance or beta iteration.

    This is deliberately narrower than a generic version-number filter. A
    release is rejected only when the source both describes an iteration and
    explicitly says that it contains no material change. Major features,
    security events, breaking changes, migrations, and measured performance
    improvements remain eligible.
    """
    lower = re.sub(r"\s+", " ", text.casefold())
    iteration = bool(re.search(
        r"\b(?:public|developer)?\s*beta\s*(?:version\s*)?\d+\b|"
        r"\b(?:alpha|preview|release candidate|rc)\s*\d+\b|"
        r"\bversion\s+\d+(?:\.\d+){1,3}\b|"
        r"\b(?:fifth|sixth|seventh|eighth|ninth|tenth)\s+(?:public\s+|developer\s+)?betas?\b|"
        r"第[二三四五六七八九十\d]+个?(?:公开|开发者)?测试版|小版本更新|例行更新",
        lower,
    ))
    explicitly_minor = any(marker in lower for marker in (
        "no notable new features", "no new features", "nothing notably new",
        "minor update", "routine maintenance", "maintenance release",
        "bug fixes only", "only bug fixes", "no user-facing changes",
        "没有值得注意的新功能", "没有新功能", "无明显新功能", "仅修复 bug",
        "只有错误修复", "例行维护",
    ))
    material_change = any(marker in lower for marker in (
        "major release", "new capability", "new feature", "introduces",
        "security fix", "security update", "zero-day", "cve-", "actively exploited",
        "breaking change", "incompatible", "deprecat", "migration required",
        "performance improved", "latency", "throughput", "available for the first time",
        "重大版本", "新增功能", "新能力", "安全修复", "安全更新", "零日漏洞",
        "不兼容", "弃用", "必须迁移", "性能提升", "延迟降低", "吞吐提升",
    ))
    # Negated phrases contain the lexical substring "new feature". Preserve
    # the source's explicit negation instead of treating that as novelty.
    if any(marker in lower for marker in (
        "no notable new features", "no new features", "没有值得注意的新功能",
        "没有新功能", "无明显新功能",
    )):
        material_change = False
    return iteration and explicitly_minor and not material_change


def _physical_update_rejection(item: DiscoveryCandidate) -> str:
    """Reject physical-tech announcements whose event is too small for video.

    A source video is necessary for robotics and autonomous driving, but it is
    not sufficient.  Supplier certificates and another-city rollouts can be
    legitimate trade news while still offering no new capability, conflict,
    or useful audience consequence for this product.
    """
    if item.channel not in {DiscoveryChannel.ROBOTICS, DiscoveryChannel.AUTONOMOUS_DRIVING}:
        return ""
    lead = re.sub(
        r"<[^>]+>|\s+", " ", f"{item.title}\n{item.summary}", flags=re.IGNORECASE,
    ).casefold()
    certification = bool(re.search(
        r"\b(?:certif(?:ication|ied)|accreditation|iatf\s*16949|iso\s*\d{3,})\b|"
        r"认证|资质|通过.{0,12}(?:标准|认证)",
        lead,
    ))
    certification_changes_operation = bool(re.search(
        r"\b(?:regulatory|type|vehicle) approval\b|"
        r"\bapproved (?:for|to) (?:public roads?|commercial service|operate)\b|"
        r"获准.{0,10}(?:上路|运营|载客)|取得.{0,10}(?:运营|上路)许可",
        lead,
    ))
    if certification and not certification_changes_operation:
        return "standalone_certification"

    geographic_expansion = bool(re.search(
        r"\bwelcom(?:e|ing) (?:our )?first (?:public )?riders? in\b|"
        r"\bmarking \d+ cities\b|\b(?:expand(?:s|ed|ing)?|launch(?:es|ed|ing)?) "
        r"(?:its )?(?:service )?(?:to|in|into|across) (?:\d+ |new )?(?:cities|markets)\b|"
        r"\badds? (?:\d+ |new )?(?:cities|markets)\b|"
        r"扩大到.{0,8}\d+.{0,4}城|新增.{0,12}(?:城市|市场)|进驻.{0,12}(?:城市|市场)",
        lead,
    ))
    expansion_is_a_new_boundary = bool(re.search(
        r"\b(?:first international|first country|international debut|first in (?:europe|asia|africa))\b|"
        r"\bfirst (?:paid |public )?(?:fully )?driverless (?:service|rides?)\b|"
        r"\bremov(?:e|es|ed|ing) (?:the )?safety driver\b|"
        r"首次进入.{0,8}(?:国家|海外|欧洲|亚洲)|首次.{0,12}(?:无人安全员|付费载客)",
        lead,
    ))
    if geographic_expansion and not expansion_is_a_new_boundary:
        return "routine_physical_geographic_expansion"
    return ""


def _article_subject_body(item: DiscoveryCandidate) -> str:
    """Exclude publisher chrome and related stories from Jina article scoring."""
    body = item.body_text.strip()
    if not body or "Markdown Content:" not in body:
        return body
    title = re.sub(r"\s+", " ", item.title).strip().rstrip("/")
    if not title:
        return body
    # Jina normally emits an H1, but paywalled pages sometimes flatten it to a
    # plain line.  Start only after ``Markdown Content:`` so the ``Title:``
    # metadata line cannot be mistaken for the article heading.
    markdown_start = body.find("Markdown Content:") + len("Markdown Content:")
    markdown = body[markdown_start:]
    heading = re.search(rf"(?mi)^#?\s*{re.escape(title)}\s*$", markdown)
    if not heading:
        return markdown.strip()
    article = markdown[heading.end():].strip()
    boundaries = [
        match.start() for pattern in (
            r"(?mi)^Topics\s*$",
            r"(?mi)^##\s+Related\s*$",
            r"(?mi)^###\s+Newsletters\s*$",
            r"(?mi)^ADVERTISEMENT\s*$",
            r"(?mi)^EDITOR'S PICKS\s*$",
            r"(?mi)^RECOMMENDED\s*$",
            r"(?mi)^Do Not Sell or Share My Personal Information\s*$",
            r"(?m)^©\s+\d{4}\b",
        )
        if (match := re.search(pattern, article))
    ]
    return article[:min(boundaries)].strip() if boundaries else article


def _candidate_subject_text(item: DiscoveryCandidate) -> str:
    return "\n".join(filter(None, (
        item.title.strip(), item.summary.strip(), _article_subject_body(item),
    ))).strip()


def _has_material_technology_consequence(text: str) -> bool:
    """Require a concrete technical change, not a generic 'AI-powered' label."""
    lower = re.sub(r"\s+", " ", text.casefold())
    technology = (
        r"(?:ai|artificial intelligence|llms?|foundation model|reasoning model|"
        r"model api|api|sdk|developer tool|software infrastructure|open source|"
        r"cloud infrastructure|cybersecurity|semiconductor|chip|gpu|robot|robotics|"
        r"humanoid|robotaxi|self-driving|autonomous driving|inference|training)"
    )
    change = (
        r"(?:launch(?:ed|es|ing)?|release[ds]?|ship(?:ped|s|ping)?|deploy(?:ed|s|ing)?|"
        r"open-source[ds]?|publish(?:ed|es|ing)?|introduc(?:e|ed|es|ing)|unveil(?:ed|s|ing)?|"
        r"upgrade[ds]?|cut|reduce[ds]?|improve[ds]?|increase[ds]?|benchmark(?:ed|s|ing)?)"
    )
    english = bool(
        re.search(rf"\b{change}\b.{{0,100}}\b{technology}\b", lower)
        or re.search(rf"\b{technology}\b.{{0,100}}\b{change}\b", lower)
    )
    chinese = bool(re.search(
        r"(?:发布|上线|开源|部署|交付|推出|升级|降低|提升|基准|评测).{0,60}"
        r"(?:人工智能|大模型|基础模型|智能体|API|SDK|开发者工具|软件基础设施|"
        r"云基础设施|网络安全|半导体|芯片|机器人|自动驾驶)|"
        r"(?:人工智能|大模型|基础模型|智能体|API|SDK|开发者工具|软件基础设施|"
        r"云基础设施|网络安全|半导体|芯片|机器人|自动驾驶).{0,60}"
        r"(?:发布|上线|开源|部署|交付|推出|升级|降低|提升|基准|评测)",
        lower,
    ))
    return english or chinese


def _is_finance_only_business_story(item: DiscoveryCandidate) -> bool:
    """Reject money/deal stories that have no concrete technology consequence."""
    if item.channel not in {
        DiscoveryChannel.PROJECTS, DiscoveryChannel.NEWS, DiscoveryChannel.NEWS_ZH,
    }:
        return False
    lead = re.sub(r"\s+", " ", f"{item.title}\n{item.summary}").casefold()
    text = _candidate_subject_text(item)
    finance_is_the_event = bool(re.search(
        r"\b(?:acquir(?:e[ds]?|ing|ed|er|ers|ition)|sell(?:s|ing)?|sold|sale|"
        r"valuation|cash|fund(?:s|raising)?|funding|financing|invest(?:or|ors|ment|ments)|"
        r"venture capital|private equity|revenue|profit|bank balance|shares?)\b|"
        r"\$\s*\d|收购|出售|卖给|估值|现金|融资|营收|利润|账面|股份",
        lead,
    ))
    technology_consequence = _has_material_technology_consequence(text)
    return finance_is_the_event and not technology_consequence


def evaluate_candidate(item: DiscoveryCandidate, config: ChannelConfig, now: datetime) -> DiscoveryCandidate:
    item.url = canonical_url(item.url)
    item.topic_type, item.content_type = _route_candidate(item)
    subject_body = _article_subject_body(item)
    text = _candidate_subject_text(item)
    if item.channel in WEB_DISCOVERY_CHANNELS and subject_body != item.body_text.strip():
        item.metadata["page_image_count"] = int(item.metadata.get("image_count") or 0)
        item.metadata["image_count"] = len(re.findall(
            r"!\[[^]]*\]\([^)]*\)|https?://\S+\.(?:png|jpe?g|webp)",
            subject_body,
            re.IGNORECASE,
        ))
    reasons: list[str] = []
    published = _parse_date(item.published_at)
    age_hours = (now - published).total_seconds() / 3600 if published else None
    if not item.title.strip() or not item.url.startswith(("http://", "https://")):
        reasons.append("missing_identity")
    if not item.author.strip() and not item.publisher.strip():
        reasons.append("missing_author_or_publisher")
    if published is None:
        reasons.append("missing_published_at")
    elif age_hours is not None and age_hours > config.lookback_hours:
        reasons.append("outside_lookback")

    if item.channel == DiscoveryChannel.OPENROUTER:
        # OpenRouter candidates are produced only from its model/endpoints API;
        # model names such as DeepSeek V4 need not repeat the word AI.
        scope_markers = ["openrouter_model"]
    elif item.channel == DiscoveryChannel.YOUTUBE:
        scope_markers = [
            str(marker) for marker in item.metadata.get("it_scope_markers") or []
            if str(marker).strip()
        ]
    else:
        scope_markers = it_software_ai_markers(text)
    item.metadata["it_scope_markers"] = scope_markers
    if not scope_markers:
        reasons.append("outside_it_software_ai_scope")
    if item.channel in WEB_DISCOVERY_CHANNELS and re.match(
        r"\s*(?:<!doctype\s+html|<html\b)", item.body_text, re.IGNORECASE,
    ):
        # The normal fetch path returns extracted Markdown. Raw HTML means the
        # extractor failed; navigation, ads, and metadata must not masquerade
        # as article evidence or inflate the score.
        reasons.append("unextractable_source_page")

    roundup_video = bool(
        item.metadata.get("atomized_roundup_event")
        and item.metadata.get("source_video_url")
    )
    minimum_text = {
        DiscoveryChannel.X: 80, DiscoveryChannel.GITHUB: 400, DiscoveryChannel.NEWS: 500,
        DiscoveryChannel.PROJECTS: 400, DiscoveryChannel.ROBOTICS: 400,
        DiscoveryChannel.AUTONOMOUS_DRIVING: 400,
        DiscoveryChannel.NEWS_ZH: 500, DiscoveryChannel.OFFICIAL: 280,
        DiscoveryChannel.OFFICIAL_ZH: 280, DiscoveryChannel.PAPER: 350, DiscoveryChannel.YOUTUBE: 100,
        DiscoveryChannel.OPENROUTER: 240,
    }[item.channel]
    if roundup_video:
        # A roundup child intentionally carries only its local paragraph. Its
        # subject-specific page and official video are acquired during
        # adoption, so applying the parent-news 500-character gate erases the
        # strongest real-world events from the candidate pool.
        minimum_text = 120
    required_facts = 1 if item.channel == DiscoveryChannel.X or roundup_video else 3
    if len(text) < minimum_text or _facts(text) < required_facts:
        reasons.append("insufficient_narrative_material")

    lower = text.casefold()
    chinese_llm_story = item.channel in CHINESE_DISCOVERY_CHANNELS and any(
        marker in lower for marker in CHINESE_LLM_PROVIDER_MARKERS
    )
    if chinese_llm_story and not any(marker in lower for marker in CHINESE_LLM_HIGH_VALUE_MARKERS):
        reasons.append("missing_high_value_chinese_llm_event")
    if chinese_llm_story and any(marker in lower for marker in CHINESE_LLM_PRICE_MARKERS):
        quantified_price = bool(re.search(r"\d+(?:\.\d+)?\s*(?:%|％|倍|元|美元)|免费|价格战|腰斩", lower))
        if not quantified_price:
            reasons.append("missing_quantified_price_change")
        percentages = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|％)", lower)]
        if percentages and max(percentages) < 20 and not any(
            marker in lower for marker in ("涨价", "价格战", "免费", "腰斩")
        ):
            reasons.append("routine_chinese_llm_promotion")
    viral_signals = _viral_attention_signals(item, text)
    if viral_signals:
        item.metadata["viral_attention_signals"] = viral_signals
    if (
        item.channel == DiscoveryChannel.PROJECTS
        and not _has_breakout_project_signal(text)
        and not _is_substantive_model_launch(text)
        and not viral_signals
    ):
        reasons.append("missing_breakout_traction")
    if item.channel in WEB_DISCOVERY_CHANNELS and _is_routine_product_iteration(text):
        reasons.append("routine_iteration_without_material_change")
    physical_update_rejection = _physical_update_rejection(item)
    if physical_update_rejection:
        reasons.append(physical_update_rejection)
    if _is_finance_only_business_story(item):
        reasons.append("finance_only_without_technology_consequence")
    physical_channels = {DiscoveryChannel.ROBOTICS, DiscoveryChannel.AUTONOMOUS_DRIVING}
    has_source_video = bool(str(item.metadata.get("source_video_url") or "").strip())
    if item.channel in physical_channels and not has_source_video:
        reasons.append("missing_source_video")
    if item.channel in physical_channels and all(marker in lower for marker in (
        "validated only within a simplified 2d simulation",
        "not yet been verified on physical robots",
    )):
        reasons.append("simulation_only_without_real_world_evidence")
    if item.channel in physical_channels:
        item.discovery_bucket = _robotics_bucket(text)
        item.metadata["discovery_bucket"] = item.discovery_bucket
        if item.channel == DiscoveryChannel.ROBOTICS and item.discovery_bucket != "robotics":
            reasons.append("belongs_to_autonomous_driving_bucket")
        if (
            item.channel == DiscoveryChannel.AUTONOMOUS_DRIVING
            and item.discovery_bucket != "autonomous_driving"
        ):
            reasons.append("belongs_to_robotics_bucket")
    visual = False
    if item.channel == DiscoveryChannel.X:
        visual = True  # The complete post card is a first-class real source visual.
    elif item.channel == DiscoveryChannel.GITHUB:
        visual = bool(item.body_text)
        if not any(marker in lower for marker in ("install", "usage", "quickstart", "quick start", "getting started", "npx ", "pip ", "npm ")):
            reasons.append("missing_trial_path")
        if not any(marker in lower for marker in ("demo", "example", "input", "output", "workflow", "screenshot", "![")):
            reasons.append("missing_concrete_io_or_demo")
    elif item.channel in physical_channels:
        # Motion is the product for physical technology. Screenshots and long
        # articles do not substitute for seeing the robot or vehicle act.
        visual = has_source_video
    elif item.channel in WEB_DISCOVERY_CHANNELS:
        visual = bool(
            item.metadata.get("source_video_url")
            or item.metadata.get("image_count")
            or len(subject_body) >= 900
        )
        if item.channel in OFFICIAL_CHANNELS and not any(
            marker in lower for marker in (
                "available", "launch", "introduc", "release", "rollout", "api", "model",
                "benchmark", "today", "发布", "上线", "开放", "开源", "模型", "产品",
                "价格", "降价", "涨价", "调价", "下线", "停服", "迁移", "弃用",
                "基准", "榜单", "评测", "上下文", "多模态", "智能体", "安全事件",
            )
        ):
            reasons.append("missing_official_event_or_availability")
    elif item.channel == DiscoveryChannel.PAPER:
        visual = bool(item.metadata.get("pdf_available"))
        if not any(marker in lower for marker in ("we propose", "method", "experiment", "benchmark", "evaluate", "results")):
            reasons.append("missing_method_or_results")
    elif item.channel == DiscoveryChannel.YOUTUBE:
        visual = True
        for reason in item.metadata.get("youtube_rejection_reasons") or []:
            if str(reason).strip():
                reasons.append(str(reason).strip())
        duration = float(item.metadata.get("duration_seconds") or 0)
        if not 900 <= duration <= 7200:
            reasons.append("duration_out_of_range")
        if not item.metadata.get("transcript_available"):
            reasons.append("transcript_unavailable")
        mode = str(item.metadata.get("youtube_editorial_mode") or "")
        source_text = f"{item.title} {item.summary} {item.body_text}"
        if mode != "known_tech_interview_clip" and (
            item.metadata.get("political_signals") or political_markers(source_text)
        ):
            reasons.append("political_content_forbidden")
        if mode not in {"technical_coverage", "known_tech_interview_clip"}:
            reasons.append("not_technical_share_or_known_tech_interview")
    else:
        visual = bool(item.metadata.get("visual_path"))
        if not item.metadata.get("compelling"):
            reasons.append("promotion_not_compelling")
        if float(item.metadata.get("endpoint_uptime") or 0) < float(
            config.settings.get("minimum_endpoint_uptime", 99.0)
        ):
            reasons.append("endpoint_reliability_below_gate")
    if not visual:
        reasons.append("missing_visual_path")

    evidence = 25.0 if len(text) >= minimum_text * 2 and _facts(text) >= 4 else 18.0
    authority = 20.0 if item.channel in {
        DiscoveryChannel.OFFICIAL, DiscoveryChannel.OFFICIAL_ZH,
        DiscoveryChannel.PAPER, DiscoveryChannel.OPENROUTER,
    } else 16.0
    if item.channel in NEWS_CHANNELS and any(
        (urlparse(item.url).hostname or "").endswith(domain)
        for domain in (*DEFAULT_TRUSTED_NEWS, *DEFAULT_TRUSTED_NEWS_ZH)
    ):
        authority = 20.0
    audience = 20.0 if re.search(
        r"\b(ai|llm|agent|model|api|sdk|benchmark|funding|tokens?|startup|robotics?|"
        r"humanoid|robotaxi|self-driving|autonomous driving|physical ai|embodied ai)\b|"
        r"大模型|新模型|智能体|多模态|上下文|工具调用|开源|价格战|降价|涨价|调价",
        lower,
    ) else 10.0
    visuals = 15.0 if visual else 0.0
    freshness = 10.0 if age_hours is not None and age_hours <= max(2, config.lookback_hours / 4) else 6.0
    specificity = 10.0 if re.search(
        r"\d|install|available|method|result|funding|api|sdk|demo|deploy|pilot|field test|发布|上线|开放|开源|"
        r"降价|涨价|调价|免费|下线|迁移|弃用|基准|评测",
        lower,
    ) else 6.0
    attention = 0.0
    if item.channel in physical_channels:
        attention_text = f"{item.title}\n{item.summary}".casefold()
        attention_signals = sum(bool(re.search(pattern, attention_text)) for pattern in (
            r"\bfirst(?:[- ]ever| time)?\b|history made|record[- ]breaking",
            r"\bfully autonomous\b|without (?:a )?safety (?:driver|monitor)|driverless",
            r"\b\d+(?:\.\d+)?\+?\s*(?:million|billion)\s+(?:miles|kilometers|rides|deliveries)\b",
            r"\b(?:home|school|hospital|airport|highway|public road|warehouse|farm)\b",
            r"\b(?:uncut video|highlights|video|demo|field test|pilot|deployed|available)\b",
        ))
        attention = min(8.0, attention_signals * 2.0)
    impact_tier = 0
    impact_reason = ""
    impact_value = 0.0
    if item.channel in physical_channels:
        impact_tier, impact_reason = _real_world_impact(text)
        impact_value = {0: 0.0, 1: 8.0, 2: 16.0, 3: 24.0}[impact_tier]
        item.metadata["real_world_impact_tier"] = impact_tier
        item.metadata["real_world_impact_reason"] = impact_reason
    if item.channel == DiscoveryChannel.X:
        metrics = item.metadata.get("metrics") if isinstance(item.metadata.get("metrics"), dict) else {}
        engagement = sum(int(metrics.get(key) or 0) for key in ("likes", "retweets", "replies", "like_count", "retweet_count"))
        specificity = min(10.0, specificity + (2.0 if engagement >= 100 else 0.0))
    item.score_breakdown = {
        "evidence_completeness": evidence, "source_authority": authority,
        "audience_value": audience, "visual_usability": visuals,
        "freshness": freshness, "specificity": specificity,
        **({
            "attention_value": attention,
            "real_world_impact": impact_value,
        } if item.channel in physical_channels else {}),
    }
    item.score = round(sum(item.score_breakdown.values()), 2)
    item.rejection_reasons = list(dict.fromkeys(reasons))
    item.eligible = not item.rejection_reasons and item.score >= config.minimum_score
    item.status = "eligible" if item.eligible else "rejected"
    return item


_LLM_PRIORITY_PATTERN = re.compile(
    r"\b(?:llms?|large language model|foundation model|reasoning model|openai|anthropic|"
    r"ai model|transformer(?:-based)?(?: ai)? model|system one model|"
    r"claude|gemini|deepseek|qwen|mistral|gpt(?:[- ]?\d[\w.-]*)?|"
    r"glm[- ]?\d|kimi|llama)\b|"
    r"大模型|基础模型|推理模型|语言模型|智谱|通义千问|豆包|混元|文心|阶跃星辰",
    re.IGNORECASE,
)
_HARDWARE_PATTERN = re.compile(
    r"\b(?:semiconductor|microchip|chipset|silicon|gpu|npu|tpu|accelerator|hbm|"
    r"server rack|data cent(?:er|re)|mac mini|compute cluster|wafer|foundry|fab|"
    r"macbook|iphone|ipad|smartphone|laptop|desktop|ram|memory|ssd|display|"
    r"nvidia|amd|qualcomm|broadcom|tsmc|cerebras)\b|"
    r"半导体|芯片|晶圆|显卡|加速卡|服务器|数据中心|算力集群|内存|手机|"
    r"电脑|笔记本|台式机|存储|屏幕|配色|颜色",
    re.IGNORECASE,
)
_AI_HARDWARE_CONTEXT_PATTERN = re.compile(
    r"\b(?:ai|artificial intelligence|machine learning|deep learning|neural network|"
    r"large language model|llm|foundation model|generative ai|ai agents?|"
    r"model (?:training|inference)|inference (?:server|cluster|workload)|training cluster|"
    r"tensor cores?|cuda|tokens? per second|openai|anthropic|claude|gemini|gpt)\b|"
    r"人工智能|机器学习|深度学习|神经网络|大模型|基础模型|生成式AI|智能体|"
    r"模型训练|模型推理|推理服务器|训练集群",
    re.IGNORECASE,
)
_MATERIAL_AI_HARDWARE_PATTERN = re.compile(
    r"(?:\b(?:buy|bought|purchase[ds]?|rent(?:ed|s)?|deploy(?:ed|s)?|install(?:ed|s)?|"
    r"ship(?:ped|s)? to (?:customers?|data cent(?:er|re)s?)|shortage|supply constraint)\b"
    r".{0,80}\b(?:ai|model|training|inference|agents?|llm)\b)|"
    r"(?:\b(?:ai|model|training|inference|agents?|llm)\b.{0,80}"
    r"\b(?:throughput|latency|performance|capacity|cost|energy|power|tokens? per second)\b"
    r".{0,40}\d)|"
    r"(?:购买|采购|租用|部署|安装|交付|短缺|供应受限).{0,50}"
    r"(?:AI|人工智能|模型|训练|推理|智能体|大模型)|"
    r"(?:AI|人工智能|模型|训练|推理|智能体|大模型).{0,50}"
    r"(?:吞吐|延迟|性能|容量|成本|能耗|功耗).{0,20}\d",
    re.IGNORECASE,
)


def _hardware_news_rejection(item: DiscoveryCandidate, text: str) -> str:
    """Keep hardware only when a material AI workload is the actual story."""
    if item.channel in {DiscoveryChannel.ROBOTICS, DiscoveryChannel.AUTONOMOUS_DRIVING}:
        return ""
    if not _hardware_is_central(item, text):
        return ""
    if not _AI_HARDWARE_CONTEXT_PATTERN.search(text):
        return "hardware_not_ai_related"
    if not _MATERIAL_AI_HARDWARE_PATTERN.search(text):
        return "routine_ai_hardware_product_news"
    return ""


def _hardware_is_central(item: DiscoveryCandidate, text: str) -> bool:
    """Ignore incidental hardware words buried in an otherwise software story."""
    headline_context = f"{item.title}\n{item.summary}"
    if _HARDWARE_PATTERN.search(headline_context):
        return True
    matches = {
        match.group(0).casefold()
        for match in _HARDWARE_PATTERN.finditer(text)
    }
    return len(matches) >= 2 and bool(_MATERIAL_AI_HARDWARE_PATTERN.search(text))


def _adoption_text(item: DiscoveryCandidate) -> str:
    return re.sub(
        r"\s+", " ", _candidate_subject_text(item),
    )


def _adoption_category_flags(item: DiscoveryCandidate, text: str) -> list[str]:
    flags: list[str] = []
    if item.channel in {DiscoveryChannel.ROBOTICS, DiscoveryChannel.AUTONOMOUS_DRIVING}:
        flags.append("physical_system")
    if _hardware_is_central(item, text):
        flags.append("hardware")
    if _LLM_PRIORITY_PATTERN.search(text):
        flags.append("llm_intelligence")
    return flags


def _youtube_adoption_decision(
    item: DiscoveryCandidate, policy: AdoptionPolicy,
) -> dict[str, Any]:
    """Map the dedicated YouTube discovery rubric into the adoption contract."""
    source = float(item.metadata.get("youtube_score") or 0.0)
    breakdown = {
        "youtube_editorial_score": round(max(0.0, min(100.0, source)), 2),
    }
    threshold = policy.youtube_minimum_score
    score_clears = source >= threshold
    passed = item.eligible and score_clears
    reasons = [] if passed else [
        "source_quality_gate_failed" if not item.eligible
        else "youtube_editorial_score_below_threshold"
    ]
    score_verdict = "clears" if score_clears else "does not clear"
    state_note = "" if item.eligible else " Candidate is currently held by its source/retry state."
    return {
        "version": 1, "pool": "youtube", "category_flags": [],
        "score": round(max(0.0, min(100.0, source)), 2),
        "threshold": threshold, "passed": passed,
        "breakdown": breakdown, "reasons": reasons,
        "summary": (
            f"YouTube editorial score {source:.1f} {score_verdict} {threshold:.1f}."
            f"{state_note}"
        ),
    }


def evaluate_adoption_candidate(
    item: DiscoveryCandidate, policy: AdoptionPolicy,
) -> DiscoveryCandidate:
    """Score publish-worthiness without altering the source-quality verdict."""
    if item.channel == DiscoveryChannel.YOUTUBE:
        decision = _youtube_adoption_decision(item, policy)
        item.metadata["adoption_decision"] = decision
        return item

    text = _adoption_text(item)
    lower = text.casefold()
    flags = _adoption_category_flags(item, text)

    quality = round(max(0.0, min(100.0, item.score)) * 0.25, 2)
    action = 8.0 if re.search(
        r"\b(?:launch(?:ed|es)?|release[ds]?|publish(?:ed|es)?|open[- ]source[ds]?|"
        r"deploy(?:ed|s)?|ship(?:ped|s)?|buy|bought|purchase[ds]?|rent(?:ed|s)?|"
        r"acquire[ds]?|leave|left|join(?:ed|s)?|hire[ds]?|cut|drop(?:ped|s)?|"
        r"raise[ds]?|increase[ds]?|ban(?:ned|s)?|block(?:ed|s)?|approve[ds]?)\b|"
        r"发布|上线|开源|部署|交付|购买|采购|租用|收购|离职|加入|招聘|"
        r"降价|涨价|封禁|获批|开放",
        lower,
    ) else 0.0
    metric = 7.0 if re.search(
        r"(?:\d+(?:\.\d+)?\s*(?:%|％|x\b|倍|ms\b|s\b|gb\b|tb\b|"
        r"million\b|billion\b|万|亿|美元|元)|\$\s*\d)", lower,
    ) or re.search(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b", lower) else 0.0
    immediate = 4.0 if re.search(
        r"\b(?:today|now|this week|available|effective immediately|begins?|"
        r"starting|rollout|announc(?:e|ed|es|ing))\b|今天|本周|现已|立即|开始|官宣",
        lower,
    ) else 0.0
    material_scope = 6.0 if re.search(
        r"\b(?:production|customers?|users?|developers?|api|sdk|pricing|cost|"
        r"latency|throughput|safety|security|training|inference|public roads?|"
        r"factory|warehouse|hospital|home|school|commercial|regulator)\b|"
        r"生产|客户|用户|开发者|价格|成本|延迟|吞吐|安全|训练|推理|"
        r"公共道路|工厂|仓库|医院|家庭|学校|商业化|监管",
        lower,
    ) else 0.0
    material_change = action + metric + immediate + material_scope

    audience_domain = 10.0 if re.search(
        r"\b(?:developers?|engineers?|researchers?|customers?|users?|teams?|"
        r"api|sdk|install|deploy|production|workflow|open source|pricing|cost|"
        r"safety|driverless|robotaxi|robotics?)\b|"
        r"开发者|工程师|研究者|用户|团队|安装|部署|工作流|开源|价格|成本|"
        r"安全|无人驾驶|自动驾驶|机器人",
        lower,
    ) else 0.0
    direct_effect = 8.0 if re.search(
        r"\b(?:save[sd]?|reduce[sd]?|lower(?:ed|s)?|faster|slower|replace[sd]?|"
        r"automate[sd]?|access|available|availability|use[ds]?|run[ns]?|return[sd]?|"
        r"blocked?|banned?|risk|safer|cheaper|expensive|free|"
        r"fold[sd]?|load[sd]?|clean[sd]?|carr(?:y|ies|ied)|deliver(?:ed|s)?|"
        r"without a safety driver|paid rides?)\b|"
        r"节省|降低|提速|变慢|取代|自动化|可用|开放|封禁|风险|更安全|"
        r"更便宜|免费|无人安全员|付费载客",
        lower,
    ) else 0.0
    known_actor = bool(re.search(
        r"\b(?:openai|anthropic|google|microsoft|meta|nvidia|apple|tesla|waymo|"
        r"amazon|aws|deepseek|qwen|mistral|amd|qualcomm)\b|"
        r"谷歌|微软|英伟达|苹果|特斯拉|百度|阿里|腾讯|字节|华为",
        lower,
    ))
    proper_title_tokens = {
        token for token in re.findall(r"\b[A-Z][A-Za-z0-9.+-]{2,}\b", item.title)
        if token.casefold() not in {
            "the", "this", "today", "new", "company", "team", "robot",
            "model", "developers", "engineers", "researchers",
        }
    }
    named_stakes = 3.0 if known_actor or proper_title_tokens else 0.0
    tension = 4.0 if re.search(
        r"\b(?:versus|vs\.?|compete[sd]?|rival|challenge[sd]?|conflict|"
        r"dispute[sd]?|but|while|instead|compared with|outperform(?:ed|s)?)\b|"
        r"对抗|竞争|挑战|争议|质疑|相比|超过|击败|但|却|而",
        lower,
    ) else 0.0
    audience_consequence = audience_domain + direct_effect + named_stakes + tension

    first_or_record = 5.0 if re.search(
        r"\b(?:first(?:[- ]ever| time)?|record[- ]breaking|breakthrough|"
        r"unprecedented|fully autonomous|without a safety driver)\b|"
        r"首次|首个|纪录|突破|前所未有|完全自动驾驶|无人安全员",
        lower,
    ) else 0.0
    surprising_comparison = 5.0 if tension or re.search(
        r"\b(?:from .{0,30} to|up to|as much as|more than|less than|"
        r"cheaper than|faster than|higher than|lower than)\b|"
        r"从.{0,20}(?:降到|升到)|高达|低至|超过",
        lower,
    ) else 0.0
    non_routine = 5.0 if "routine_iteration_without_material_change" not in item.rejection_reasons else 0.0
    impact_tier = int(item.metadata.get("real_world_impact_tier") or 0)
    impact_novelty = {0: 0.0, 1: 0.0, 2: 3.0, 3: 5.0}.get(impact_tier, 5.0)
    novelty_utility = min(
        15.0, first_or_record + surprising_comparison + non_routine + impact_novelty,
    )

    if item.metadata.get("source_video_url") or re.search(
        r"\b(?:gif|uncut(?:.{0,30})?(?:video|demo|demonstration))\b|动图|原视频", lower,
    ):
        visual_proof = 10.0
    elif item.channel in {DiscoveryChannel.X, DiscoveryChannel.GITHUB, DiscoveryChannel.PAPER}:
        visual_proof = 8.0
    elif item.metadata.get("image_count") or item.metadata.get("visual_path") or re.search(
        r"\b(?:chart|diagram|screenshot|benchmark table|source code)\b|图表|架构图|截图|代码",
        lower,
    ):
        visual_proof = 8.0
    else:
        visual_proof = 4.0

    llm_priority_qualified = (
        "llm_intelligence" in flags
        and material_change >= 12.0
        and audience_consequence >= policy.minimum_audience_consequence
    )
    llm_bonus = policy.llm_priority_bonus if llm_priority_qualified else 0.0
    subtotal = quality + material_change + audience_consequence + novelty_utility + visual_proof
    score = round(min(100.0, subtotal + llm_bonus), 2)
    high_standard = bool({"physical_system", "hardware"} & set(flags))
    threshold = (
        policy.high_standard_minimum_score if high_standard
        else policy.general_minimum_score
    )
    reasons: list[str] = []
    if "physical_system" in flags and not str(item.metadata.get("source_video_url") or "").strip():
        reasons.append("missing_source_video")
    physical_update_rejection = _physical_update_rejection(item)
    if physical_update_rejection:
        reasons.append(physical_update_rejection)
    if _is_finance_only_business_story(item):
        reasons.append("finance_only_without_technology_consequence")
    hardware_rejection = _hardware_news_rejection(item, text)
    if hardware_rejection:
        reasons.append(hardware_rejection)
    if not item.eligible:
        reasons.append("source_quality_gate_failed")
    if audience_consequence < policy.minimum_audience_consequence:
        reasons.append("weak_audience_consequence")
    if score < threshold:
        reasons.append("adoption_score_below_threshold")
    passed = not reasons
    breakdown = {
        "source_quality": quality,
        "material_change": material_change,
        "audience_consequence": audience_consequence,
        "novelty_or_utility": novelty_utility,
        "visual_proof": visual_proof,
        "llm_priority_bonus": llm_bonus,
    }
    decision = {
        "version": 1, "pool": "general", "category_flags": flags,
        "llm_priority_qualified": llm_priority_qualified,
        "score": score, "threshold": threshold, "passed": passed,
        "breakdown": breakdown, "reasons": reasons,
        "summary": (
            f"Adoption score {score:.1f} {'clears' if passed else 'does not clear'} "
            f"{threshold:.1f}; audience consequence {audience_consequence:.1f}/25."
        ),
    }
    item.metadata["adoption_decision"] = decision
    return item


def select_adoption_candidates(
    candidates: Iterable[DiscoveryCandidate], policy: AdoptionPolicy,
    reserved_event_keys: set[str | tuple[str, str]] | None = None,
) -> list[DiscoveryCandidate]:
    """Return every passing unique event; YouTube is an independent pool."""
    reserved = set(reserved_event_keys or set())
    winners: dict[tuple[str, str], DiscoveryCandidate] = {}
    for item in candidates:
        evaluate_adoption_candidate(item, policy)
        decision = dict(item.metadata.get("adoption_decision") or {})
        if not decision.get("passed"):
            continue
        pool = str(decision.get("pool") or "general")
        event_key = item.event_key or item.stable_id or item.url
        # A retry reserves its event only inside the same comparison pool.
        if event_key in reserved or (pool, event_key) in reserved:
            continue
        key = (pool, event_key)
        incumbent = winners.get(key)
        rank = (float(decision.get("score") or 0), item.score, item.url)
        incumbent_decision = dict(incumbent.metadata.get("adoption_decision") or {}) if incumbent else {}
        incumbent_rank = (
            float(incumbent_decision.get("score") or 0), incumbent.score, incumbent.url,
        ) if incumbent else (-1.0, -1.0, "")
        if incumbent is None or rank[:2] > incumbent_rank[:2] or (
            rank[:2] == incumbent_rank[:2] and rank[2] < incumbent_rank[2]
        ):
            winners[key] = item
    return sorted(
        winners.values(),
        key=lambda item: (
            str((item.metadata.get("adoption_decision") or {}).get("pool") or "general"),
            -float((item.metadata.get("adoption_decision") or {}).get("score") or 0),
            item.url,
        ),
    )


def _same_event(left: DiscoveryCandidate, right: DiscoveryCandidate) -> bool:
    if left.url == right.url or (left.stable_id and left.stable_id == right.stable_id):
        return True
    left_date, right_date = _parse_date(left.published_at), _parse_date(right.published_at)
    if left_date and right_date and abs((left_date - right_date).total_seconds()) > 72 * 3600:
        return False
    shared = _tokens(left.title) & _tokens(right.title)
    return bool(shared) and _similarity(left.title, right.title) >= 0.34


def assign_event_clusters(candidates: list[DiscoveryCandidate]) -> None:
    parents = list(range(len(candidates)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if _same_event(candidates[left], candidates[right]):
                parents[root(right)] = root(left)
    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(root(index), []).append(index)
    for indexes in groups.values():
        seed = "|".join(sorted(candidates[index].stable_id or candidates[index].url for index in indexes))
        event_key = "event:" + hashlib.sha256(seed.encode()).hexdigest()[:16]
        for index in indexes:
            candidates[index].event_key = event_key


def select_parallel_candidates(
    by_channel: dict[DiscoveryChannel, list[DiscoveryCandidate]],
    reserved_event_keys: set[str] | None = None,
) -> dict[DiscoveryChannel, DiscoveryCandidate]:
    def selection_score(item: DiscoveryCandidate) -> float:
        if item.channel in {DiscoveryChannel.ROBOTICS, DiscoveryChannel.AUTONOMOUS_DRIVING}:
            # Reality tier is deliberately lexicographic: an eligible robot
            # doing useful physical work must not lose to a toy because the
            # toy has a catchier title or one extra video keyword.
            return float(item.metadata.get("real_world_impact_tier") or 0) * 1000 + item.score
        return item.score

    channels = sorted(by_channel, key=lambda item: item.value)
    choices = {
        channel: sorted(
            [item for item in by_channel[channel] if item.eligible],
            key=lambda item: (-selection_score(item), item.url),
        )[:8]
        for channel in channels
    }
    best_score = -1.0
    best: dict[DiscoveryChannel, DiscoveryCandidate] = {}

    def visit(index: int, used: set[str], score: float, selected: dict[DiscoveryChannel, DiscoveryCandidate]) -> None:
        nonlocal best_score, best
        if index == len(channels):
            signature = tuple(item.url for _, item in sorted(selected.items(), key=lambda pair: pair[0].value))
            best_signature = tuple(item.url for _, item in sorted(best.items(), key=lambda pair: pair[0].value))
            if score > best_score or (score == best_score and signature < best_signature):
                best_score, best = score, dict(selected)
            return
        channel = channels[index]
        visit(index + 1, used, score, selected)
        for item in choices[channel]:
            if item.event_key in used:
                continue
            selected[channel] = item
            visit(index + 1, used | {item.event_key}, score + selection_score(item), selected)
            selected.pop(channel, None)

    visit(0, set(reserved_event_keys or set()), 0.0, {})
    return best


class ResourceDiscoveryService:
    def __init__(
        self, workspace: Workspace, adapters: dict[DiscoveryChannel, DiscoveryAdapter] | None = None,
        factory: VideoFactory | None = None, clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] | None = None,
        observability: Observability | None = None,
    ) -> None:
        self.workspace = workspace
        self.workspace.initialize()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.sleeper = sleeper or time.sleep
        self.factory = factory or VideoFactory(workspace)
        self.observability = observability or Observability(
            workspace.root, project_name="video-factory-discovery",
        )
        self.problem_ledger = ProblemLedger(workspace.root)
        self.adapters = adapters or {
            DiscoveryChannel.X: XDiscoveryAdapter(),
            DiscoveryChannel.GITHUB: GitHubDiscoveryAdapter(),
            DiscoveryChannel.PROJECTS: RSSDiscoveryAdapter(DiscoveryChannel.PROJECTS),
            DiscoveryChannel.ROBOTICS: RSSDiscoveryAdapter(DiscoveryChannel.ROBOTICS),
            DiscoveryChannel.AUTONOMOUS_DRIVING: RSSDiscoveryAdapter(DiscoveryChannel.AUTONOMOUS_DRIVING),
            DiscoveryChannel.NEWS: RSSDiscoveryAdapter(DiscoveryChannel.NEWS),
            DiscoveryChannel.NEWS_ZH: RSSDiscoveryAdapter(DiscoveryChannel.NEWS_ZH),
            DiscoveryChannel.OFFICIAL: RSSDiscoveryAdapter(DiscoveryChannel.OFFICIAL),
            DiscoveryChannel.OFFICIAL_ZH: RSSDiscoveryAdapter(DiscoveryChannel.OFFICIAL_ZH),
            DiscoveryChannel.PAPER: PaperDiscoveryAdapter(),
            DiscoveryChannel.YOUTUBE: YouTubeDiscoveryAdapter(workspace),
            DiscoveryChannel.OPENROUTER: OpenRouterDiscountDiscoveryAdapter(),
        }

    def status(self, channel: DiscoveryChannel | None = None) -> dict[str, Any]:
        state = self.workspace.load_discovery_state()
        if channel is None:
            return state
        return dict((state.get("channels") or {}).get(channel.value) or {})

    def run(
        self, config: ResourceDiscoveryConfig, scheduled: bool = True,
        channels: Iterable[DiscoveryChannel] | None = None, provider: str = "auto", model: str | None = None,
    ) -> ResourceDiscoveryRun:
        requested = [item.value for item in (channels or DiscoveryChannel)]
        with self.observability.span("factory.discovery.run", {
            "scheduled": scheduled,
            "requested_channels": requested,
        }) as span:
            result = self._run_impl(config, scheduled, channels, provider, model)
            span.set_attribute("run_id", result.id)
            span.set_attribute("status", result.status)
            for name, entry in result.channels.items():
                funnel = dict(entry.trace.get("funnel") or {})
                span.add_event("channel.completed", {
                    "channel": name,
                    "status": entry.status,
                    "candidate_count": len(entry.candidates),
                    "eligible_count": int(funnel.get("eligible") or 0),
                    "selected": bool(entry.selected),
                    "selection_count": len(entry.selections),
                    "adoption_passed": int(funnel.get("adoption_passed") or 0),
                    "adoption_rejected": int(funnel.get("adoption_rejected") or 0),
                    "sources_failed": int(funnel.get("sources_failed") or 0),
                })
            return result

    def _run_impl(
        self, config: ResourceDiscoveryConfig, scheduled: bool = True,
        channels: Iterable[DiscoveryChannel] | None = None, provider: str = "auto", model: str | None = None,
    ) -> ResourceDiscoveryRun:
        now = self.clock().astimezone(UTC)
        requested = set(channels or DiscoveryChannel)
        state = self.workspace.load_discovery_state()
        # Generated-event state is recoverable/cache-like and may be reset.
        # Remote publication is the durable truth for YouTube source reuse.
        state["published_youtube_source_ids"] = sorted(self._published_youtube_source_ids())
        run = ResourceDiscoveryRun(
            id=f"resources-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}",
            status="running", started_at=_iso(now),
        )
        eligible: dict[DiscoveryChannel, list[DiscoveryCandidate]] = {}
        blocked: dict[DiscoveryChannel, DiscoveryCandidate] = {}
        for channel in DiscoveryChannel:
            if channel not in requested or not config.channels[channel].enabled:
                continue
            channel_state = dict((state.get("channels") or {}).get(channel.value) or {})
            next_run = _parse_date(str(channel_state.get("next_run_at") or ""))
            if scheduled and next_run and now < next_run:
                run.channels[channel.value] = ChannelRun(
                    channel, "not_due", next_run_at=_iso(next_run), trace={
                        "scheduler": {
                            "decision": "not_due", "checked_at": _iso(now),
                            "next_run_at": _iso(next_run),
                            "wake_interval_minutes": 10,
                            "channel_cadence_hours": config.channels[channel].cadence_hours,
                        },
                    },
                )
                continue
            entry = ChannelRun(channel, "searching")
            entry.trace["scheduler"] = {
                "decision": "due", "checked_at": _iso(now),
                "previous_next_run_at": _iso(next_run) if next_run else "",
                "wake_interval_minutes": 10,
                "channel_cadence_hours": config.channels[channel].cadence_hours,
            }
            run.channels[channel.value] = entry
            blocked_payload = channel_state.get("blocked_candidate")
            if isinstance(blocked_payload, dict):
                blocked_item = DiscoveryCandidate.from_dict(blocked_payload)
                if channel in WEB_DISCOVERY_CHANNELS and config.channels[channel].seed_domains:
                    blocked_host = (urlparse(blocked_item.url).hostname or "").casefold()
                    if not any(
                        blocked_host == domain or blocked_host.endswith("." + domain)
                        for domain in config.channels[channel].seed_domains
                    ):
                        persisted = state.setdefault("channels", {}).setdefault(channel.value, {})
                        persisted.pop("blocked_candidate", None)
                        persisted.pop("blocked_retry_at", None)
                        persisted.pop("blocked_retry_runs", None)
                        blocked_item.status = "rejected"
                        blocked_item.eligible = False
                        blocked_item.rejection_reasons.append("unresolved_aggregator_not_primary_source")
                        self.workspace.save_discovery_candidate(blocked_item.to_dict())
                        entry.candidates = [blocked_item]
                        entry.status = "blocked_invalidated_source"
                        continue
                retry_at = _parse_date(str(channel_state.get("blocked_retry_at") or ""))
                if scheduled and retry_at is None:
                    # Import legacy blocked state without immediately spending
                    # another full LLM retry budget on the next 10-minute tick.
                    retry_at = now + timedelta(hours=config.blocked_retry_delay_hours)
                    persisted = state.setdefault("channels", {}).setdefault(channel.value, {})
                    persisted["blocked_retry_at"] = _iso(retry_at)
                    persisted["blocked_retry_runs"] = max(1, int(channel_state.get("blocked_retry_runs") or 0))
                if scheduled and retry_at and now < retry_at:
                    entry.candidates = [blocked_item]
                    entry.status = "blocked_retry_wait"
                    entry.next_run_at = _iso(retry_at)
                    continue
                blocked[channel] = blocked_item
                entry.candidates = [blocked_item]
                entry.status = "blocked_retry_pending"
                continue
            try:
                items = self.adapters[channel].search(config.channels[channel], now)
                adapter_trace = getattr(self.adapters[channel], "last_trace", None)
                if isinstance(adapter_trace, dict):
                    entry.trace["search"] = dict(adapter_trace)
                if channel == DiscoveryChannel.ROBOTICS:
                    items = [
                        child
                        for parent in items
                        for child in atomize_robotics_roundup(parent)
                    ][:config.channels[channel].max_candidates]
                transient_retries = {
                    str(candidate_id): dict(payload)
                    for candidate_id, payload in dict(
                        channel_state.get("transient_retries") or {}
                    ).items()
                    if isinstance(payload, dict)
                }
                present_ids = {item.id for item in items}
                injected_retry_ids: list[str] = []
                for candidate_id, retry in transient_retries.items():
                    retry_at = _parse_date(str(retry.get("retry_at") or ""))
                    candidate_payload = retry.get("candidate")
                    if (
                        candidate_id in present_ids
                        or (scheduled and retry_at and now < retry_at)
                        or not isinstance(candidate_payload, dict)
                    ):
                        continue
                    retry_item = DiscoveryCandidate.from_dict(candidate_payload)
                    retry_item.status = "retry_pending"
                    items.append(retry_item)
                    present_ids.add(candidate_id)
                    injected_retry_ids.append(candidate_id)
                if transient_retries:
                    entry.trace["transient_retries"] = {
                        "queued": len(transient_retries),
                        "injected_due": injected_retry_ids,
                    }
                skipped_ids = set(str(value) for value in (state.get("skipped_ids") or []))
                needs_human_ids = {
                    str(value.get("candidate_id") or "")
                    for value in (state.get("needs_human_candidates") or [])
                    if isinstance(value, dict)
                }
                seen_price_ids = set(str(value) for value in channel_state.get("seen_candidate_ids") or [])
                # Page probing has already happened inside the adapter. Once a
                # probed roundup is atomized—or YouTube has fairly distributed
                # probes across pools—every returned item must enter the same
                # comparison instead of being cut off by a second outer budget.
                evaluation_limit = (
                    len(items) if channel == DiscoveryChannel.YOUTUBE else
                    config.channels[channel].max_candidates if channel == DiscoveryChannel.ROBOTICS else
                    config.channels[channel].probe_limit
                )
                for item in items[:evaluation_limit]:
                    evaluate_candidate(item, config.channels[channel], now)
                    if channel == DiscoveryChannel.OPENROUTER and item.id in seen_price_ids:
                        item.eligible = False
                        item.status = "rejected"
                        item.rejection_reasons.append("price_event_already_seen")
                    if item.id in skipped_ids:
                        item.eligible = False
                        item.status = "skipped"
                        item.rejection_reasons.append("manually_skipped")
                    if item.id in needs_human_ids:
                        item.eligible = False
                        item.status = "needs_human"
                        item.rejection_reasons.append("automatic_repair_budget_exhausted")
                    transient_retry = transient_retries.get(item.id)
                    retry_at = _parse_date(str((transient_retry or {}).get("retry_at") or ""))
                    if transient_retry and scheduled and retry_at and now < retry_at:
                        item.eligible = False
                        item.status = "retry_wait"
                        item.rejection_reasons.append("transient_source_retry_backoff")
                    evaluate_adoption_candidate(item, config.adoption_policy)
                    self.workspace.save_discovery_candidate(item.to_dict())
                entry.candidates = items[:evaluation_limit]
                duplicate_count = sum(
                    1 for item in items[:evaluation_limit]
                    if item.eligible and self._historical_duplicate(item, state, config, now)
                )
                eligible[channel] = [
                    item for item in items[:evaluation_limit]
                    if item.eligible and item.id not in skipped_ids and item.id not in needs_human_ids
                    and not self._historical_duplicate(item, state, config, now)
                ]
                search_funnel = dict((entry.trace.get("search") or {}).get("funnel") or {})
                entry.trace["funnel"] = {
                    **search_funnel,
                    "evaluated": len(entry.candidates),
                    "eligible_before_history_dedupe": sum(item.eligible for item in entry.candidates),
                    "historical_duplicates": duplicate_count,
                    "eligible": len(eligible[channel]),
                    "adoption_evaluated": len(eligible[channel]),
                    "adoption_passed": sum(
                        bool((item.metadata.get("adoption_decision") or {}).get("passed"))
                        for item in eligible[channel]
                    ),
                    "adoption_rejected": sum(
                        not bool((item.metadata.get("adoption_decision") or {}).get("passed"))
                        for item in eligible[channel]
                    ),
                    "rejected": sum(not item.eligible for item in entry.candidates),
                }
                entry.trace["candidate_decisions"] = [{
                    "id": item.id,
                    "title": item.title[:240],
                    "url": item.url,
                    "score": item.score,
                    "eligible": item.eligible,
                    "rejection_reasons": list(item.rejection_reasons),
                    "adoption": dict(item.metadata.get("adoption_decision") or {}),
                } for item in entry.candidates]
                self._record_discovery_health(channel, entry, run.id)
                entry.status = "searched"
                entry.next_run_at = _iso(now + timedelta(hours=config.channels[channel].cadence_hours))
                state.setdefault("channels", {}).setdefault(channel.value, {})["next_run_at"] = entry.next_run_at
                if channel == DiscoveryChannel.OPENROUTER:
                    state["channels"][channel.value]["seen_candidate_ids"] = list(dict.fromkeys([
                        *seen_price_ids, *(item.id for item in items),
                    ]))[-500:]
            except Exception as error:
                entry.status = "search_failed"
                entry.error = f"{type(error).__name__}: {error}"
                adapter_trace = getattr(self.adapters[channel], "last_trace", None)
                if isinstance(adapter_trace, dict):
                    entry.trace["search"] = dict(adapter_trace)
                entry.trace["failure"] = {"stage": "search", "error": entry.error}
                self._record_discovery_source_failure(channel, entry, run.id)
                entry.next_run_at = _iso(now + timedelta(minutes=15))
                state.setdefault("channels", {}).setdefault(channel.value, {})["next_run_at"] = entry.next_run_at

        all_eligible = [item for items in eligible.values() for item in items]
        assign_event_clusters([*all_eligible, *blocked.values()])
        passing_blocked: dict[DiscoveryChannel, DiscoveryCandidate] = {}
        for channel, item in blocked.items():
            if self._historical_duplicate(item, state, config, now):
                channel_state = state.setdefault("channels", {}).setdefault(channel.value, {})
                channel_state.pop("blocked_candidate", None)
                channel_state.pop("blocked_retry_at", None)
                channel_state.pop("blocked_retry_runs", None)
                item.status = "not_adopted"
                item.eligible = False
                item.rejection_reasons = list(dict.fromkeys([
                    *item.rejection_reasons, "source_already_published",
                ]))
                self.workspace.save_discovery_candidate(item.to_dict())
                run.channels[channel.value].status = "historical_duplicate"
                continue
            evaluate_adoption_candidate(item, config.adoption_policy)
            decision = dict(item.metadata.get("adoption_decision") or {})
            entry = run.channels[channel.value]
            entry.trace["blocked_adoption_recheck"] = decision
            if decision.get("passed"):
                passing_blocked[channel] = item
                continue
            channel_state = state.setdefault("channels", {}).setdefault(channel.value, {})
            channel_state.pop("blocked_candidate", None)
            channel_state.pop("blocked_retry_at", None)
            channel_state.pop("blocked_retry_runs", None)
            item.status = "not_adopted"
            self.workspace.save_discovery_candidate(item.to_dict())
            entry.status = "adoption_rejected"

        selectable = [
            item for channel, items in eligible.items() if channel not in passing_blocked
            for item in items
        ]
        reserved = {
            (
                str((item.metadata.get("adoption_decision") or {}).get("pool") or "general"),
                item.event_key,
            )
            for item in passing_blocked.values() if item.event_key
        }
        selected_items = [
            *select_adoption_candidates(selectable, config.adoption_policy, reserved),
            *passing_blocked.values(),
        ]
        selected_ids = {item.id for item in selected_items}
        selected_pool_events = {
            (
                str((item.metadata.get("adoption_decision") or {}).get("pool") or "general"),
                item.event_key,
            )
            for item in selected_items
        }
        for channel, items in eligible.items():
            entry = run.channels[channel.value]
            traced = {
                str(row.get("id") or ""): row
                for row in entry.trace.get("candidate_decisions") or []
            }
            for item in items:
                decision = dict(item.metadata.get("adoption_decision") or {})
                if item.id in selected_ids:
                    decision["selection_status"] = "selected"
                elif decision.get("passed"):
                    pool_event = (
                        str(decision.get("pool") or "general"), item.event_key,
                    )
                    decision["selection_status"] = (
                        "duplicate_event_lower_rank"
                        if pool_event in selected_pool_events
                        else "reserved_event"
                    )
                    item.status = "not_selected"
                else:
                    decision["selection_status"] = "adoption_rejected"
                    item.status = "not_adopted"
                item.metadata["adoption_decision"] = decision
                if item.id in traced:
                    traced[item.id]["adoption"] = decision
                self.workspace.save_discovery_candidate(item.to_dict())
        adoption_statuses: dict[DiscoveryChannel, list[str]] = {}
        for item in selected_items:
            channel = item.channel
            entry = run.channels[channel.value]
            entry.selections.append(item)
            if entry.selected is None:
                entry.selected = item
            selection_trace = {
                "candidate_id": item.id, "event_key": item.event_key,
                "quality_score": item.score,
                "adoption": dict(item.metadata.get("adoption_decision") or {}),
                "reason": "passed pool-specific adoption threshold as a unique event",
            }
            entry.trace.setdefault("selections", []).append(selection_trace)
            if "selection" not in entry.trace:
                entry.trace["selection"] = selection_trace
            item.status = "selected"
            self.workspace.save_discovery_candidate(item.to_dict())
            adoption = self._adopt(item, config, provider, model)
            entry.adoptions.append(adoption)
            if entry.adoption is None:
                entry.adoption = adoption
            adoption_statuses.setdefault(channel, []).append(str(adoption["status"]))
            channel_state = state.setdefault("channels", {}).setdefault(channel.value, {})
            if adoption["status"] == "generated":
                channel_state.pop("blocked_candidate", None)
                channel_state.pop("blocked_retry_at", None)
                channel_state.pop("blocked_retry_runs", None)
                transient_retries = dict(channel_state.get("transient_retries") or {})
                transient_retries.pop(item.id, None)
                if transient_retries:
                    channel_state["transient_retries"] = transient_retries
                else:
                    channel_state.pop("transient_retries", None)
                state.setdefault("generated_events", []).append({
                    "event_key": item.event_key, "title": item.title, "published_at": item.published_at,
                    "topic_type": item.topic_type.value if item.topic_type else "", "url": item.url,
                    "generated_at": _iso(self.clock()), "candidate_id": item.id,
                    "channel": channel.value,
                    "pool": str((item.metadata.get("adoption_decision") or {}).get("pool") or "general"),
                    "adoption_score": float(
                        (item.metadata.get("adoption_decision") or {}).get("score") or 0
                    ),
                })
            else:
                queue_status = self._record_blocked_candidate(
                    state, item, config, self.clock(),
                    retryable=bool(adoption.get("retryable")),
                    last_error=str(adoption.get("last_error") or ""),
                )
                if queue_status in {"needs_human", "retry_pending"}:
                    adoption_statuses[channel][-1] = queue_status
        for channel, statuses_for_channel in adoption_statuses.items():
            entry = run.channels[channel.value]
            unique_statuses = set(statuses_for_channel)
            entry.status = (
                statuses_for_channel[0] if len(unique_statuses) == 1
                else "partially_generated" if "generated" in unique_statuses
                else "multiple_blocked"
            )
        for channel, entry in run.channels.items():
            if entry.status == "searched":
                funnel = dict(entry.trace.get("funnel") or {})
                quality_eligible = int(funnel.get("eligible") or 0)
                adoption_passed = int(funnel.get("adoption_passed") or 0)
                entry.status = (
                    "adoption_rejected"
                    if quality_eligible > 0 and adoption_passed == 0
                    else "no_selection"
                )
        state["generated_events"] = (state.get("generated_events") or [])[-500:]
        run.completed_at = _iso(self.clock())
        statuses = {entry.status for entry in run.channels.values()}
        run.status = "failed" if statuses and statuses <= {"search_failed"} else "completed"
        self.workspace.save_discovery_run(run.id, run.to_dict())
        state.setdefault("history", []).append({
            "id": run.id, "started_at": run.started_at, "completed_at": run.completed_at,
            "status": run.status, "channels": {key: value.status for key, value in run.channels.items()},
        })
        state["history"] = state["history"][-100:]
        self.workspace.save_discovery_state(state)
        return run

    def _record_discovery_health(
        self, channel: DiscoveryChannel, entry: ChannelRun, run_id: str,
    ) -> None:
        """Turn observable recall failures into asynchronous audit inputs.

        A run with no selected story is normal. We record only structural
        funnel failures: sources returned rows but the trust/resolution gate
        removed all of them, or all configured sources failed.
        """
        funnel = dict(entry.trace.get("funnel") or {})
        raw = int(funnel.get("rows_raw") or 0)
        trusted = int(funnel.get("rows_trusted") or 0)
        emitted = int(funnel.get("candidates_emitted") or len(entry.candidates))
        planned = int(funnel.get("sources_planned") or 0)
        succeeded = int(funnel.get("sources_succeeded") or 0)
        observed = ""
        if planned and succeeded == 0:
            observed = f"all {planned} configured sources failed before candidate extraction"
        elif raw and trusted == 0:
            observed = f"{raw} fetched rows were reduced to zero by source trust coverage"
        elif trusted and emitted == 0:
            observed = f"{trusted} trusted rows produced zero resolved candidates"
        if not observed:
            return
        try:
            self.problem_ledger.record(ProblemObservation(
                stage="discovery",
                category="coverage_gap",
                severity="high" if planned and succeeded == 0 else "medium",
                reporter="discovery-funnel",
                job_id=run_id,
                expected=(
                    f"{channel.value} discovery preserves at least one auditable candidate when configured sources return relevant rows"
                ),
                observed=observed,
                artifact_refs=[f"discovery/runs/{run_id}.json"],
                metadata={"channel": channel.value, "funnel": funnel},
            ))
        except Exception:
            # Auditability must not turn an optional discovery run into a
            # production outage; the full funnel remains in the run artifact.
            pass

    def _record_discovery_source_failure(
        self, channel: DiscoveryChannel, entry: ChannelRun, run_id: str,
    ) -> None:
        """Escalate source outages, especially silent social-auth expiry."""
        detail = entry.error or "discovery source failed without an error detail"
        lower = detail.casefold()
        auth_failure = channel == DiscoveryChannel.X and any(marker in lower for marker in (
            "auth_required", "not_authenticated", "not logged", "no ct0 cookie",
        ))
        try:
            self.problem_ledger.record(ProblemObservation(
                stage="discovery",
                category="source_auth_unavailable" if auth_failure else "source_unavailable",
                severity="high" if auth_failure else "medium",
                reporter="discovery-source-health",
                job_id=run_id,
                expected=f"{channel.value} discovery source is available on its configured cadence",
                observed=detail,
                artifact_refs=[f"discovery/runs/{run_id}.json"],
                metadata={"channel": channel.value, "auth_failure": auth_failure},
            ))
        except Exception:
            # The run artifact still preserves the complete failure trace.
            pass

    def adopt_candidate(
        self, candidate_id: str, config: ResourceDiscoveryConfig,
        provider: str = "auto", model: str | None = None,
    ) -> dict[str, Any]:
        item = DiscoveryCandidate.from_dict(self.workspace.load_discovery_candidate(candidate_id))
        if not item.eligible and item.status not in {
            "blocked", "needs_human", "retry_pending", "retry_wait",
        }:
            raise ValueError(f"candidate {candidate_id} did not pass its channel quality gate")
        result = self._adopt(item, config, provider, model)
        state = self.workspace.load_discovery_state()
        channel_state = state.setdefault("channels", {}).setdefault(item.channel.value, {})
        if result["status"] == "generated":
            channel_state.pop("blocked_candidate", None)
            channel_state.pop("blocked_retry_at", None)
            channel_state.pop("blocked_retry_runs", None)
            transient_retries = dict(channel_state.get("transient_retries") or {})
            transient_retries.pop(item.id, None)
            if transient_retries:
                channel_state["transient_retries"] = transient_retries
            else:
                channel_state.pop("transient_retries", None)
            state.setdefault("generated_events", []).append({
                "event_key": item.event_key, "title": item.title, "published_at": item.published_at,
                "topic_type": item.topic_type.value if item.topic_type else "", "url": item.url,
                "generated_at": _iso(self.clock()), "candidate_id": item.id,
                "channel": item.channel.value,
                "pool": str((item.metadata.get("adoption_decision") or {}).get("pool") or "general"),
                "adoption_score": float(
                    (item.metadata.get("adoption_decision") or {}).get("score") or 0
                ),
            })
            state["needs_human_candidates"] = [
                value for value in (state.get("needs_human_candidates") or [])
                if not isinstance(value, dict) or value.get("candidate_id") != item.id
            ]
        else:
            result["queue_status"] = self._record_blocked_candidate(
                state, item, config, self.clock(),
                retryable=bool(result.get("retryable")),
                last_error=str(result.get("last_error") or ""),
            )
        self.workspace.save_discovery_state(state)
        return result

    def _record_blocked_candidate(
        self, state: dict[str, Any], item: DiscoveryCandidate,
        config: ResourceDiscoveryConfig, now: datetime, *, retryable: bool = False,
        last_error: str = "",
    ) -> str:
        channel_state = state.setdefault("channels", {}).setdefault(item.channel.value, {})
        previous = channel_state.get("blocked_candidate")

        def move_to_human(candidate: DiscoveryCandidate, reason: str) -> None:
            candidate.status = "needs_human"
            self.workspace.save_discovery_candidate(candidate.to_dict())
            rows = state.setdefault("needs_human_candidates", [])
            if not any(
                isinstance(value, dict) and value.get("candidate_id") == candidate.id
                for value in rows
            ):
                rows.append({
                    "candidate_id": candidate.id, "channel": candidate.channel.value,
                    "title": candidate.title, "url": candidate.url,
                    "recorded_at": _iso(now), "reason": reason,
                })
            state["needs_human_candidates"] = rows[-100:]

        transient_retries = dict(channel_state.get("transient_retries") or {})
        if retryable:
            previous_retry = dict(transient_retries.get(item.id) or {})
            retry_runs = int(previous_retry.get("retry_runs") or 0) + 1
            retry_at = now.astimezone(UTC) + timedelta(hours=config.blocked_retry_delay_hours)
            item.status = "retry_pending"
            item.metadata["transient_retry"] = {
                "retry_runs": retry_runs, "retry_at": _iso(retry_at),
                "last_error": last_error[:1000],
            }
            self.workspace.save_discovery_candidate(item.to_dict())
            transient_retries[item.id] = {
                "candidate": item.to_dict(), "retry_runs": retry_runs,
                "retry_at": _iso(retry_at), "last_error": last_error[:1000],
            }
            channel_state["transient_retries"] = transient_retries
            return "retry_pending"

        if item.id in transient_retries:
            transient_retries.pop(item.id, None)
            if transient_retries:
                channel_state["transient_retries"] = transient_retries
            else:
                channel_state.pop("transient_retries", None)

        if isinstance(previous, dict) and previous.get("id") != item.id:
            previous_item = DiscoveryCandidate.from_dict(previous)
            previous_score = float(
                (previous_item.metadata.get("adoption_decision") or {}).get("score") or 0
            )
            current_score = float(
                (item.metadata.get("adoption_decision") or {}).get("score") or 0
            )
            if previous_score >= current_score:
                move_to_human(
                    item,
                    "another higher-scoring candidate already occupies the bounded automatic retry slot",
                )
                return "needs_human"
            move_to_human(
                previous_item,
                "a higher-scoring candidate replaced it in the bounded automatic retry slot",
            )
            channel_state.pop("blocked_retry_at", None)
            channel_state.pop("blocked_retry_runs", None)
            previous = None
        previous_runs = int(channel_state.get("blocked_retry_runs") or 0)
        retry_runs = previous_runs + 1 if isinstance(previous, dict) and previous.get("id") == item.id else 1
        if retry_runs >= config.max_blocked_retry_runs:
            move_to_human(item, "bounded automatic generation retries exhausted")
            channel_state.pop("blocked_candidate", None)
            channel_state.pop("blocked_retry_at", None)
            channel_state.pop("blocked_retry_runs", None)
            return "needs_human"
        channel_state["blocked_candidate"] = item.to_dict()
        channel_state["blocked_retry_runs"] = retry_runs
        channel_state["blocked_retry_at"] = _iso(
            now.astimezone(UTC) + timedelta(hours=config.blocked_retry_delay_hours)
        )
        return "blocked"

    def skip(self, candidate_id: str, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("skip requires a non-empty reason")
        item = DiscoveryCandidate.from_dict(self.workspace.load_discovery_candidate(candidate_id))
        item.status = "skipped"
        item.metadata["skip_reason"] = reason.strip()
        item.metadata["skipped_at"] = _iso(self.clock())
        self.workspace.save_discovery_candidate(item.to_dict())
        state = self.workspace.load_discovery_state()
        channel_state = state.setdefault("channels", {}).setdefault(item.channel.value, {})
        blocked = channel_state.get("blocked_candidate")
        if isinstance(blocked, dict) and blocked.get("id") == item.id:
            channel_state.pop("blocked_candidate", None)
            channel_state.pop("blocked_retry_at", None)
            channel_state.pop("blocked_retry_runs", None)
        transient_retries = dict(channel_state.get("transient_retries") or {})
        transient_retries.pop(item.id, None)
        if transient_retries:
            channel_state["transient_retries"] = transient_retries
        else:
            channel_state.pop("transient_retries", None)
        state.setdefault("skipped_ids", []).append(item.id)
        state["needs_human_candidates"] = [
            value for value in (state.get("needs_human_candidates") or [])
            if not isinstance(value, dict) or value.get("candidate_id") != item.id
        ]
        self.workspace.save_discovery_state(state)
        return {"status": "skipped", "candidate_id": item.id, "reason": reason.strip()}

    def _adopt(
        self, item: DiscoveryCandidate, config: ResourceDiscoveryConfig,
        provider: str, model: str | None,
    ) -> dict[str, Any]:
        completed = self._latest_completed_generation(item.url)
        if completed is not None:
            item.status = "generated"
            self.workspace.save_discovery_candidate(item.to_dict())
            return {
                "status": "generated", "candidate_id": item.id,
                "attempts": [{
                    "attempt": 0, "mode": "reuse_completed_generation",
                    "status": "generated", "result": completed,
                }],
                "result": completed,
            }
        attempts: list[dict[str, Any]] = []
        roundup_primary = _roundup_primary_source(item)
        generation_url = roundup_primary or item.url
        manifest: Path | None = (
            self._latest_failed_manifest(generation_url)
            if item.status in {"blocked", "needs_human"} else None
        )
        youtube_media: Path | None = None
        youtube_subtitles: Path | None = None
        youtube_translation_plan: Path | None = None
        if item.channel == DiscoveryChannel.YOUTUBE:
            youtube_media, youtube_subtitles = self._latest_youtube_assets(item.url)
            youtube_translation_plan = self._latest_youtube_translation_plan(item.url)
        for attempt, delay in enumerate(config.retry_backoff_seconds, start=1):
            retry_mode = "deterministic_rerender" if manifest and manifest.is_file() else "full_generation"
            if delay and retry_mode == "full_generation":
                self.sleeper(delay)
            try:
                if manifest and manifest.is_file():
                    source_video_url = str(item.metadata.get("source_video_url") or "").strip()
                    result = (
                        self.factory.rerender(manifest, source_video_url=source_video_url)
                        if source_video_url else self.factory.rerender(manifest)
                    )
                else:
                    selected_links = list(dict.fromkeys([
                        *(str(url) for url in item.metadata.get("linked_sources") or []),
                        *([item.url] if roundup_primary else []),
                    ]))
                    result = self.factory.generate(generation_url, GenerateOptions(
                        provider=provider, model=model, topic=item.topic_type,
                        content_type=item.content_type, render=True,
                        research=item.channel != DiscoveryChannel.OPENROUTER,
                        youtube_media=str(youtube_media) if youtube_media else None,
                        youtube_subtitles=str(youtube_subtitles) if youtube_subtitles else None,
                        youtube_translation_plan=(
                            str(youtube_translation_plan) if youtube_translation_plan else None
                        ),
                        youtube_editorial_mode=str(
                            item.metadata.get("youtube_editorial_mode") or "auto"
                        ),
                        linked_sources=tuple(selected_links),
                        discovery_context=(
                            (
                                "DISCOVERY SELECTED EVENT — BINDING STORY SCOPE\n"
                                f"Title: {item.title}\n"
                                f"Publisher: {item.publisher}\n"
                                f"Published: {item.published_at}\n"
                                f"Source URL: {item.url}\n"
                                + (
                                    f"Preferred subject-specific primary source: {roundup_primary}\n"
                                    if roundup_primary else ""
                                )
                                + (
                                    f"Official source video: {item.metadata.get('source_video_url')}\n"
                                    if item.metadata.get("source_video_url") else ""
                                )
                                + f"Selected summary: {item.summary}\n"
                                + "If the source page is a roundup, digest, or multi-video page, ignore every unrelated item. "
                                + "Every shot, hook, and conclusion must remain about the selected title and summary."
                            )
                            if item.channel != DiscoveryChannel.OPENROUTER else None
                        ),
                        discovery_source_url=(
                            str(item.metadata.get("roundup_parent_url") or "").strip() or None
                            if item.metadata.get("atomized_roundup_event") else None
                        ),
                        discovery_source_quote=(
                            item.summary.strip() or item.body_text.strip() or None
                            if item.metadata.get("atomized_roundup_event") else None
                        ),
                        discovery_published_at=(
                            item.published_at if item.channel != DiscoveryChannel.OPENROUTER else None
                        ),
                        discovery_channel=item.channel.value,
                        source_video_url=(
                            str(item.metadata.get("source_video_url") or "").strip() or None
                        ),
                        render_profile=InformationRenderProfile.RADAR_V2.value,
                        supplemental_context=(
                            f"Discovery headline: {item.title}\n\n{item.body_text}\n\n"
                            f"Price-event metadata: {json.dumps(item.metadata, ensure_ascii=False, sort_keys=True)}"
                            if item.channel == DiscoveryChannel.OPENROUTER else None
                        ),
                        price_event_metadata=(
                            dict(item.metadata) if item.channel == DiscoveryChannel.OPENROUTER else None
                        ),
                    ))
                manifest_value = result.get("manifest")
                if manifest_value:
                    manifest = Path(str(manifest_value))
                failed_checks = [
                    check for check in [
                        *(result.get("checks") or []), *(result.get("video_checks") or []),
                    ]
                    if isinstance(check, dict) and not check.get("passed", False)
                    and str(check.get("name") or "") not in {
                        "music_license_record", "editorial_safety_review", "rights_review",
                    }
                ]
                collection_value = result.get("collection_manifest")
                audio_failures = [
                    check for check in failed_checks
                    if any(marker in str(check.get("name") or "") for marker in (
                        ":aac", ":audio_duration", ":audible_audio",
                    ))
                ]
                if item.channel == DiscoveryChannel.YOUTUBE and collection_value and audio_failures:
                    collection_path = Path(str(collection_value))
                    collection = load_collection_manifest(collection_path)
                    repaired = YouTubeCollectionRenderer(self.workspace).repair_silent_audio(collection)
                    repaired_checks = validate_collection(collection, self.workspace.root)
                    collection.quality_checks = [check.to_dict() for check in repaired_checks]
                    self.workspace.save_collection_manifest(collection)
                    collection_path.write_text(
                        json.dumps(collection.to_dict(), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    result["checks"] = collection.quality_checks
                    result.setdefault("automatic_repairs", []).append({
                        "kind": "silent_or_truncated_audio", "outputs": repaired,
                    })
                    failed_checks = [
                        check for check in collection.quality_checks
                        if not check.get("passed", False)
                        and str(check.get("name") or "") not in {
                            "music_license_record", "editorial_safety_review", "rights_review",
                        }
                    ]
                output_created = bool(result.get("video") or result.get("collection_manifest"))
                success = result.get("status") == "completed" and output_created and not failed_checks
                attempts.append({
                    "attempt": attempt, "mode": retry_mode,
                    "status": "generated" if success else "quality_failed", "result": result,
                })
                if success:
                    item.status = "generated"
                    self.workspace.save_discovery_candidate(item.to_dict())
                    return {"status": "generated", "candidate_id": item.id, "attempts": attempts, "result": result}
                if manifest is None or not manifest.is_file():
                    attempts[-1]["recovery"] = "stop_non_retryable_quality_failure"
                    break
            except Exception as error:
                discard_cached_manifest = (
                    retry_mode == "deterministic_rerender"
                    and _rerender_requires_full_regeneration(error)
                )
                retryable_error = _retryable_adoption_error(error)
                attempts.append({
                    "attempt": attempt, "mode": retry_mode,
                    "status": "failed", "error": f"{type(error).__name__}: {error}",
                    "retryable": retryable_error,
                    **({"recovery": "discard_invalid_manifest_and_regenerate"} if discard_cached_manifest else {}),
                })
                if discard_cached_manifest:
                    manifest = None
                else:
                    possible = getattr(error, "manifest", None)
                    if possible:
                        manifest = Path(str(possible))
                    if manifest is None:
                        manifest = self._latest_failed_manifest(item.url)
                if item.channel == DiscoveryChannel.YOUTUBE:
                    youtube_media, youtube_subtitles = self._latest_youtube_assets(item.url)
                    youtube_translation_plan = self._latest_youtube_translation_plan(item.url)
                if isinstance(error, (NameError, UnboundLocalError, SyntaxError, ImportError)):
                    attempts[-1]["recovery"] = "stop_non_retryable_internal_error"
                    break
                if not retryable_error and not discard_cached_manifest and (
                    manifest is None or not manifest.is_file()
                ):
                    attempts[-1]["recovery"] = "stop_non_retryable_generation_failure"
                    break
        item.status = "blocked"
        self.workspace.save_discovery_candidate(item.to_dict())
        failed_attempts = [attempt for attempt in attempts if attempt.get("status") == "failed"]
        retryable = bool(failed_attempts) and len(failed_attempts) == len(attempts) and all(
            bool(attempt.get("retryable")) for attempt in failed_attempts
        )
        last_error = str((failed_attempts[-1] if failed_attempts else {}).get("error") or "")
        return {
            "status": "blocked", "candidate_id": item.id, "attempts": attempts,
            "retryable": retryable, "last_error": last_error,
            "failure_kind": "transient_source_acquisition" if retryable else "generation_or_quality",
        }

    def _latest_youtube_assets(self, source_url: str) -> tuple[Path | None, Path | None]:
        """Reuse complete source assets across planning/render retries.

        YouTube planning can fail before a collection manifest exists. The
        downloaded 1080p source and json3 transcript are still valid inputs,
        so retrying should not spend bandwidth downloading them again.
        """
        jobs = self.workspace.root / "jobs"
        if not jobs.is_dir():
            return None, None
        results = sorted(jobs.glob("*/result.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for result_path in results[:12]:
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not same_source(str(payload.get("url") or ""), source_url):
                continue
            job = result_path.parent
            media = next(
                (path for suffix in ("*.mkv", "*.mp4", "*.webm", "*.mov")
                 for path in sorted(job.glob(suffix)) if path.is_file()),
                None,
            )
            subtitles = next(
                (path for path in sorted(job.glob("*.json3"))
                 if path.is_file() and not path.name.endswith(".part")),
                None,
            )
            if media and subtitles:
                return media, subtitles
        return None, None

    def _latest_youtube_translation_plan(self, source_url: str) -> Path | None:
        jobs = self.workspace.root / "jobs"
        if not jobs.is_dir():
            return None
        results = sorted(jobs.glob("*/result.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for result_path in results[:12]:
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not same_source(str(payload.get("url") or ""), source_url):
                continue
            plan = result_path.parent / "translation-plan.json"
            if plan.is_file():
                return plan
        return None

    def _latest_failed_manifest(self, source_url: str) -> Path | None:
        jobs = self.workspace.root / "jobs"
        if not jobs.is_dir():
            return None
        results = sorted(jobs.glob("*/result.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in results[:30]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not same_source(str(payload.get("url") or ""), source_url):
                continue
            manifest = payload.get("manifest")
            if manifest and Path(str(manifest)).is_file():
                return Path(str(manifest))
        return None

    def _latest_completed_generation(self, source_url: str) -> dict[str, Any] | None:
        """Reuse a validated direct generation when discovery later adopts it.

        Manual/editor-guided generation is a normal recovery path for a
        discovery candidate. Treating that completed artifact as invisible
        leaves the old ``needs_human`` row active and can generate the same
        source twice on the next discovery run.
        """
        jobs = self.workspace.root / "jobs"
        if not jobs.is_dir():
            return None
        results = sorted(
            jobs.glob("*/result.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        ignored_checks = {
            "music_license_record", "editorial_safety_review", "rights_review",
        }
        for path in results[:100]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                payload.get("status") != "completed"
                or not same_source(str(payload.get("url") or ""), source_url)
            ):
                continue
            failed_checks = [
                check for check in [
                    *(payload.get("checks") or []),
                    *(payload.get("video_checks") or []),
                ]
                if isinstance(check, dict)
                and not check.get("passed", False)
                and str(check.get("name") or "") not in ignored_checks
            ]
            if failed_checks:
                continue
            output = payload.get("collection_manifest") or payload.get("video")
            if not output:
                continue
            output_path = Path(str(output))
            if not output_path.is_absolute():
                output_path = self.workspace.root / output_path
            if not output_path.is_file():
                continue
            return payload
        return None

    @staticmethod
    def _historical_duplicate(
        item: DiscoveryCandidate, state: dict[str, Any], config: ResourceDiscoveryConfig,
        now: datetime,
    ) -> bool:
        item_pool = "youtube" if item.channel == DiscoveryChannel.YOUTUBE else "general"
        source_kind, source_id = source_identity(item.url)
        if (
            item.channel == DiscoveryChannel.YOUTUBE
            and source_kind == "youtube"
            and source_id in set(state.get("published_youtube_source_ids") or [])
        ):
            return True
        for row in state.get("generated_events") or []:
            row_pool = str(row.get("pool") or "")
            if not row_pool:
                row_channel = str(row.get("channel") or "")
                row_candidate = str(row.get("candidate_id") or "")
                row_identity = source_identity(str(row.get("url") or ""))[0]
                row_pool = (
                    "youtube"
                    if row_channel == DiscoveryChannel.YOUTUBE.value
                    or row_candidate.startswith("youtube-")
                    or row_identity == "youtube"
                    else "general"
                )
            if row_pool != item_pool:
                continue
            if item.channel == DiscoveryChannel.OPENROUTER:
                if str(row.get("candidate_id") or "") == item.id:
                    return True
            elif canonical_url(str(row.get("url") or "")) == item.url:
                return True
            generated = _parse_date(str(row.get("generated_at") or ""))
            if generated and now - generated > timedelta(days=config.event_dedupe_days):
                continue
            shadow = DiscoveryCandidate(
                id="history", channel=item.channel, url=str(row.get("url") or ""),
                title=str(row.get("title") or ""), published_at=str(row.get("published_at") or ""),
            )
            if _same_event(item, shadow):
                return True
        return False

    def _published_youtube_source_ids(self) -> set[str]:
        """Build a durable source-level dedupe ledger from remote publish facts."""
        published: set[str] = set()
        remote_item_states = {"submitted", "collected", "uploaded_uncollected"}
        for batch_path in self.workspace.publish_dir.glob("*/batch.json"):
            try:
                batch = json.loads(batch_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if batch.get("batch_type") != "collection":
                continue
            remotely_used = str(batch.get("state") or "") == "succeeded" or any(
                str(row.get("state") or "") in remote_item_states
                for row in batch.get("items") or [] if isinstance(row, dict)
            )
            if not remotely_used:
                continue
            manifest_id = str(batch.get("manifest_id") or "").strip()
            if not manifest_id:
                continue
            try:
                manifest = self.workspace.load_collection_manifest(manifest_id)
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                collection_path = self.workspace.collections_dir / f"{manifest_id}.json"
                if not collection_path.is_file():
                    continue
                try:
                    manifest = load_collection_manifest(collection_path)
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
            source_video_id = str(manifest.source_video_id or "").strip()
            if source_video_id:
                published.add(source_video_id)
        return published
