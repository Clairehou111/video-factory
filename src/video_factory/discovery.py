from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol
from urllib.parse import quote, urlparse, urlunparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from .factory import GenerateOptions, VideoFactory
from .models import ContentType, TopicType
from .openrouter import DISCOUNTS_READER, ENDPOINTS_API, MODELS_API, parse_discounted_models
from .storage import Workspace
from .youtube import DiscoveryConfig as YouTubeDiscoveryConfig
from .youtube import YouTubeDiscoveryService


class DiscoveryChannel(StrEnum):
    X = "x"
    GITHUB = "github"
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
    DiscoveryChannel.NEWS: 2,
    DiscoveryChannel.NEWS_ZH: 2,
    DiscoveryChannel.OFFICIAL: 2,
    DiscoveryChannel.OFFICIAL_ZH: 2,
    DiscoveryChannel.PAPER: 24,
    DiscoveryChannel.YOUTUBE: 48,
    DiscoveryChannel.OPENROUTER: 2,
}
DEFAULT_LOOKBACK_HOURS = {
    DiscoveryChannel.X: 6,
    DiscoveryChannel.GITHUB: 24 * 14,
    DiscoveryChannel.NEWS: 12,
    DiscoveryChannel.NEWS_ZH: 12,
    DiscoveryChannel.OFFICIAL: 12,
    DiscoveryChannel.OFFICIAL_ZH: 12,
    DiscoveryChannel.PAPER: 24 * 3,
    DiscoveryChannel.YOUTUBE: 24 * 30,
    DiscoveryChannel.OPENROUTER: 24,
}
DEFAULT_QUERIES = {
    DiscoveryChannel.X: [
        'AI (agent OR SDK OR API OR model OR benchmark OR funding OR launch)',
        '("open source" OR paper OR product) AI',
    ],
    DiscoveryChannel.GITHUB: [
        "AI agent created:>{date}",
        "LLM tool created:>{date}",
        "agent SDK created:>{date}",
    ],
    DiscoveryChannel.NEWS: [
        "AI model launch OR agent API",
        "AI startup funding OR founding team",
        "AI research benchmark",
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
    "aws.amazon.com", "huggingface.co",
]
DEFAULT_OFFICIAL_FEEDS = [
    "https://openai.com/news/rss.xml",
    "https://deepmind.google/blog/rss.xml",
    "https://huggingface.co/blog/feed.xml",
]
DEFAULT_TRUSTED_NEWS = [
    "reuters.com", "bloomberg.com", "ft.com", "techcrunch.com", "theverge.com",
    "wired.com", "arstechnica.com", "axios.com", "nytimes.com",
]
DEFAULT_TRUSTED_NEWS_ZH = [
    "36kr.com", "caixin.com", "jiemian.com", "yicai.com", "cls.cn", "stcn.com",
    "thepaper.cn", "tmtpost.com", "leiphone.com", "qbitai.com", "jiqizhixin.com",
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

NEWS_CHANNELS = {DiscoveryChannel.NEWS, DiscoveryChannel.NEWS_ZH}
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
        if channel == DiscoveryChannel.X:
            values.setdefault("seed_accounts", list(DEFAULT_X_ACCOUNTS))
        if channel == DiscoveryChannel.NEWS:
            values.setdefault("seed_domains", list(DEFAULT_TRUSTED_NEWS))
        if channel == DiscoveryChannel.NEWS_ZH:
            values.setdefault("seed_domains", list(DEFAULT_TRUSTED_NEWS_ZH))
        if channel == DiscoveryChannel.OFFICIAL:
            values.setdefault("seed_domains", list(DEFAULT_OFFICIAL_DOMAINS))
            values.setdefault("feeds", list(DEFAULT_OFFICIAL_FEEDS))
        if channel == DiscoveryChannel.OFFICIAL_ZH:
            values.setdefault("seed_domains", list(CHINESE_LLM_OFFICIAL_DOMAINS))
        return cls(**values)


@dataclass(slots=True)
class ResourceDiscoveryConfig:
    timezone: str = "Asia/Tokyo"
    retry_backoff_seconds: list[int] = field(default_factory=lambda: [0, 30, 120])
    event_dedupe_days: int = 30
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceDiscoveryConfig":
        unknown = set(data) - {"timezone", "retry_backoff_seconds", "event_dedupe_days", "channels"}
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
            event_dedupe_days=int(data.get("event_dedupe_days", 30)),
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
    error: str = ""
    next_run_at: str = ""


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


class RSSDiscoveryAdapter:
    def __init__(self, channel: DiscoveryChannel, fetcher: Fetcher | None = None) -> None:
        if channel not in WEB_DISCOVERY_CHANNELS:
            raise ValueError("RSS adapter supports news and official channels")
        self.channel = channel
        self.fetcher = fetcher or _default_fetcher

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
        for query in config.queries:
            restricted = query
            if self.channel in OFFICIAL_CHANNELS and config.seed_domains:
                restricted = f"{query} ({' OR '.join('site:' + item for item in config.seed_domains)})"
            urls.append(self._google_news_url(restricted))
        for url in urls:
            try:
                rows.extend(_rss_rows(self._download(url)))
            except Exception as error:
                if len(urls) == 1:
                    raise RuntimeError(f"RSS search failed: {type(error).__name__}: {error}") from error
        unique_rows: dict[str, dict[str, str]] = {}
        for row in rows:
            if row.get("url"):
                unique_rows.setdefault(row["url"], row)
        ranked_rows = list(unique_rows.values())
        if self.channel in NEWS_CHANNELS and config.seed_domains:
            ranked_rows = [
                row for row in ranked_rows
                if any(
                    (urlparse(row.get("publisher_url", "")).hostname or "").casefold() == domain
                    or (urlparse(row.get("publisher_url", "")).hostname or "").casefold().endswith("." + domain)
                    for domain in config.seed_domains
                )
            ]
        ranked_rows.sort(
            key=lambda row: _parse_date(row.get("published_at", ""))
            or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        # Filter before probing. Otherwise eight newer untrusted results can
        # occupy the probe window and make a healthy trusted-news channel look empty.
        ranked_rows = ranked_rows[: max(config.max_candidates, config.probe_limit)]
        found: dict[str, DiscoveryCandidate] = {}
        for row in ranked_rows:
            raw_url = row["url"]
            if not raw_url:
                continue
            publisher_url = row.get("publisher_url", "")
            publisher_host = (urlparse(publisher_url).hostname or "").casefold()
            if self.channel in NEWS_CHANNELS and config.seed_domains and not any(
                publisher_host == domain or publisher_host.endswith("." + domain) for domain in config.seed_domains
            ):
                continue
            try:
                body, resolved = self.fetcher(raw_url)
            except Exception:
                body, resolved = row.get("summary", ""), raw_url
            final_url = canonical_url(resolved or raw_url)
            final_host = (urlparse(final_url).hostname or "").casefold()
            if self.channel in OFFICIAL_CHANNELS and config.seed_domains and not any(
                final_host == domain or final_host.endswith("." + domain)
                or publisher_host == domain or publisher_host.endswith("." + domain)
                for domain in config.seed_domains
            ):
                continue
            digest = hashlib.sha256(final_url.encode()).hexdigest()[:12]
            found[final_url] = DiscoveryCandidate(
                id=f"{self.channel.value}-{digest}", channel=self.channel, url=final_url,
                title=row["title"], author="", publisher=row.get("publisher") or final_host,
                published_at=row.get("published_at", ""), summary=row.get("summary", ""), body_text=body,
                stable_id=f"web:{final_url}", metadata={
                    "publisher_url": publisher_url, "image_count": len(re.findall(r"!\[[^]]*\]\([^)]*\)|https?://\S+\.(?:png|jpe?g|webp)", body, re.I)),
                    "source_class": "official" if self.channel in OFFICIAL_CHANNELS else "news",
                    "language": "zh" if self.channel in CHINESE_DISCOVERY_CHANNELS else "en",
                }, discovered_at=_iso(now),
            )
            if len(found) >= config.probe_limit:
                break
        return list(found.values())


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
        found: list[DiscoveryCandidate] = []
        for item in hydrated:
            transcript_available = bool(getattr(item, "transcript_available", False))
            found.append(DiscoveryCandidate(
                id=f"youtube-{item.video_id}", channel=DiscoveryChannel.YOUTUBE, url=item.url,
                title=item.title, author=item.channel, publisher=item.channel, published_at=item.published_at,
                summary=item.description, body_text=item.description, stable_id=f"youtube:{item.video_id}",
                metadata={
                    "duration_seconds": item.duration_seconds, "view_count": item.view_count,
                    "chapters": item.chapters, "transcript_available": transcript_available,
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


def evaluate_candidate(item: DiscoveryCandidate, config: ChannelConfig, now: datetime) -> DiscoveryCandidate:
    item.url = canonical_url(item.url)
    item.topic_type, item.content_type = _route_candidate(item)
    text = f"{item.summary}\n{item.body_text}".strip()
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

    minimum_text = {
        DiscoveryChannel.X: 80, DiscoveryChannel.GITHUB: 400, DiscoveryChannel.NEWS: 500,
        DiscoveryChannel.NEWS_ZH: 500, DiscoveryChannel.OFFICIAL: 280,
        DiscoveryChannel.OFFICIAL_ZH: 280, DiscoveryChannel.PAPER: 350, DiscoveryChannel.YOUTUBE: 100,
        DiscoveryChannel.OPENROUTER: 240,
    }[item.channel]
    if len(text) < minimum_text or _facts(text) < (1 if item.channel == DiscoveryChannel.X else 3):
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
    visual = False
    if item.channel == DiscoveryChannel.X:
        visual = True  # The complete post card is a first-class real source visual.
    elif item.channel == DiscoveryChannel.GITHUB:
        visual = bool(item.body_text)
        if not any(marker in lower for marker in ("install", "usage", "quickstart", "quick start", "getting started", "npx ", "pip ", "npm ")):
            reasons.append("missing_trial_path")
        if not any(marker in lower for marker in ("demo", "example", "input", "output", "workflow", "screenshot", "![")):
            reasons.append("missing_concrete_io_or_demo")
    elif item.channel in WEB_DISCOVERY_CHANNELS:
        visual = bool(item.metadata.get("image_count")) or len(item.body_text) >= 900
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
        duration = float(item.metadata.get("duration_seconds") or 0)
        if not 900 <= duration <= 7200:
            reasons.append("duration_out_of_range")
        if not item.metadata.get("transcript_available"):
            reasons.append("transcript_unavailable")
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
        r"\b(ai|llm|agent|model|api|sdk|benchmark|funding|tokens?)\b|"
        r"大模型|新模型|智能体|多模态|上下文|工具调用|开源|价格战|降价|涨价|调价",
        lower,
    ) else 10.0
    visuals = 15.0 if visual else 0.0
    freshness = 10.0 if age_hours is not None and age_hours <= max(2, config.lookback_hours / 4) else 6.0
    specificity = 10.0 if re.search(
        r"\d|install|available|method|result|funding|api|sdk|发布|上线|开放|开源|"
        r"降价|涨价|调价|免费|下线|迁移|弃用|基准|评测",
        lower,
    ) else 6.0
    if item.channel == DiscoveryChannel.X:
        metrics = item.metadata.get("metrics") if isinstance(item.metadata.get("metrics"), dict) else {}
        engagement = sum(int(metrics.get(key) or 0) for key in ("likes", "retweets", "replies", "like_count", "retweet_count"))
        specificity = min(10.0, specificity + (2.0 if engagement >= 100 else 0.0))
    item.score_breakdown = {
        "evidence_completeness": evidence, "source_authority": authority,
        "audience_value": audience, "visual_usability": visuals,
        "freshness": freshness, "specificity": specificity,
    }
    item.score = round(sum(item.score_breakdown.values()), 2)
    item.rejection_reasons = list(dict.fromkeys(reasons))
    item.eligible = not item.rejection_reasons and item.score >= config.minimum_score
    item.status = "eligible" if item.eligible else "rejected"
    return item


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
    channels = sorted(by_channel, key=lambda item: item.value)
    choices = {
        channel: sorted(
            [item for item in by_channel[channel] if item.eligible],
            key=lambda item: (-item.score, item.url),
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
            visit(index + 1, used | {item.event_key}, score + item.score, selected)
            selected.pop(channel, None)

    visit(0, set(reserved_event_keys or set()), 0.0, {})
    return best


class ResourceDiscoveryService:
    def __init__(
        self, workspace: Workspace, adapters: dict[DiscoveryChannel, DiscoveryAdapter] | None = None,
        factory: VideoFactory | None = None, clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.workspace = workspace
        self.workspace.initialize()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.sleeper = sleeper or time.sleep
        self.factory = factory or VideoFactory(workspace)
        self.adapters = adapters or {
            DiscoveryChannel.X: XDiscoveryAdapter(),
            DiscoveryChannel.GITHUB: GitHubDiscoveryAdapter(),
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
        now = self.clock().astimezone(UTC)
        requested = set(channels or DiscoveryChannel)
        state = self.workspace.load_discovery_state()
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
                run.channels[channel.value] = ChannelRun(channel, "not_due", next_run_at=_iso(next_run))
                continue
            entry = ChannelRun(channel, "searching")
            run.channels[channel.value] = entry
            blocked_payload = channel_state.get("blocked_candidate")
            if isinstance(blocked_payload, dict):
                blocked[channel] = DiscoveryCandidate.from_dict(blocked_payload)
            try:
                items = self.adapters[channel].search(config.channels[channel], now)
                skipped_ids = set(str(value) for value in (state.get("skipped_ids") or []))
                seen_price_ids = set(str(value) for value in channel_state.get("seen_candidate_ids") or [])
                for item in items[: config.channels[channel].probe_limit]:
                    evaluate_candidate(item, config.channels[channel], now)
                    if channel == DiscoveryChannel.OPENROUTER and item.id in seen_price_ids:
                        item.eligible = False
                        item.status = "rejected"
                        item.rejection_reasons.append("price_event_already_seen")
                    if item.id in skipped_ids:
                        item.eligible = False
                        item.status = "skipped"
                        item.rejection_reasons.append("manually_skipped")
                    self.workspace.save_discovery_candidate(item.to_dict())
                entry.candidates = items
                eligible[channel] = [
                    item for item in items
                    if item.eligible and item.id not in skipped_ids
                    and not self._historical_duplicate(item, state, config, now)
                ]
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
                entry.next_run_at = _iso(now + timedelta(minutes=15))
                state.setdefault("channels", {}).setdefault(channel.value, {})["next_run_at"] = entry.next_run_at

        all_eligible = [item for items in eligible.values() for item in items]
        assign_event_clusters([*all_eligible, *blocked.values()])
        selectable = {channel: items for channel, items in eligible.items() if channel not in blocked}
        selected = select_parallel_candidates(
            selectable, {item.event_key for item in blocked.values() if item.event_key},
        )
        selected.update(blocked)
        for channel, item in selected.items():
            entry = run.channels[channel.value]
            entry.selected = item
            item.status = "selected"
            self.workspace.save_discovery_candidate(item.to_dict())
            adoption = self._adopt(item, config, provider, model)
            entry.adoption = adoption
            entry.status = str(adoption["status"])
            channel_state = state.setdefault("channels", {}).setdefault(channel.value, {})
            if adoption["status"] == "generated":
                channel_state.pop("blocked_candidate", None)
                state.setdefault("generated_events", []).append({
                    "event_key": item.event_key, "title": item.title, "published_at": item.published_at,
                    "topic_type": item.topic_type.value if item.topic_type else "", "url": item.url,
                    "generated_at": _iso(self.clock()), "candidate_id": item.id,
                })
            else:
                channel_state["blocked_candidate"] = item.to_dict()
        for channel, entry in run.channels.items():
            if entry.status == "searched":
                entry.status = "no_selection"
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

    def adopt_candidate(
        self, candidate_id: str, config: ResourceDiscoveryConfig,
        provider: str = "auto", model: str | None = None,
    ) -> dict[str, Any]:
        item = DiscoveryCandidate.from_dict(self.workspace.load_discovery_candidate(candidate_id))
        if not item.eligible and item.status != "blocked":
            raise ValueError(f"candidate {candidate_id} did not pass its channel quality gate")
        result = self._adopt(item, config, provider, model)
        state = self.workspace.load_discovery_state()
        channel_state = state.setdefault("channels", {}).setdefault(item.channel.value, {})
        if result["status"] == "generated":
            channel_state.pop("blocked_candidate", None)
            state.setdefault("generated_events", []).append({
                "event_key": item.event_key, "title": item.title, "published_at": item.published_at,
                "topic_type": item.topic_type.value if item.topic_type else "", "url": item.url,
                "generated_at": _iso(self.clock()), "candidate_id": item.id,
            })
        else:
            channel_state["blocked_candidate"] = item.to_dict()
        self.workspace.save_discovery_state(state)
        return result

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
        state.setdefault("skipped_ids", []).append(item.id)
        self.workspace.save_discovery_state(state)
        return {"status": "skipped", "candidate_id": item.id, "reason": reason.strip()}

    def _adopt(
        self, item: DiscoveryCandidate, config: ResourceDiscoveryConfig,
        provider: str, model: str | None,
    ) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        manifest: Path | None = None
        for attempt, delay in enumerate(config.retry_backoff_seconds, start=1):
            if delay:
                self.sleeper(delay)
            try:
                if manifest and manifest.is_file():
                    result = self.factory.rerender(manifest)
                else:
                    result = self.factory.generate(item.url, GenerateOptions(
                        provider=provider, model=model, topic=item.topic_type,
                        content_type=item.content_type, render=True,
                        research=item.channel != DiscoveryChannel.OPENROUTER,
                        linked_sources=tuple(str(url) for url in item.metadata.get("linked_sources") or []),
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
                output_created = bool(result.get("video") or result.get("collection_manifest"))
                success = result.get("status") == "completed" and output_created and not failed_checks
                attempts.append({"attempt": attempt, "status": "generated" if success else "quality_failed", "result": result})
                if success:
                    item.status = "generated"
                    self.workspace.save_discovery_candidate(item.to_dict())
                    return {"status": "generated", "candidate_id": item.id, "attempts": attempts, "result": result}
            except Exception as error:
                attempts.append({"attempt": attempt, "status": "failed", "error": f"{type(error).__name__}: {error}"})
                possible = getattr(error, "manifest", None)
                if possible:
                    manifest = Path(str(possible))
                if manifest is None:
                    manifest = self._latest_failed_manifest(item.url)
        item.status = "blocked"
        self.workspace.save_discovery_candidate(item.to_dict())
        return {"status": "blocked", "candidate_id": item.id, "attempts": attempts}

    def _latest_failed_manifest(self, source_url: str) -> Path | None:
        jobs = self.workspace.root / "jobs"
        if not jobs.is_dir():
            return None
        results = sorted(jobs.glob("*/result.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in results[:8]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if canonical_url(str(payload.get("url") or "")) != canonical_url(source_url):
                continue
            manifest = payload.get("manifest")
            if manifest and Path(str(manifest)).is_file():
                return Path(str(manifest))
        return None

    @staticmethod
    def _historical_duplicate(
        item: DiscoveryCandidate, state: dict[str, Any], config: ResourceDiscoveryConfig,
        now: datetime,
    ) -> bool:
        for row in state.get("generated_events") or []:
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
