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
from pathlib import Path
from typing import Any, Callable

from .llm import OpenAICompatibleStoryWriter
from .media import probe_video
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
    "popular_ai": [
        "AI engineering agents talk",
        "coding agents platform engineering",
        "AI developer productivity conference",
    ],
}

AUDIENCE_MARKERS = (
    "ai", "agent", "coding", "developer", "engineering", "llm", "model", "startup",
    "platform", "software", "workflow", "team", "perplexity", "karpathy", "y combinator",
)
INSIGHT_MARKERS = (
    "how", "why", "engineering", "build", "system", "architecture", "workflow", "lessons",
    "team", "scale", "agentic", "technical", "from", "future", "inside",
)
FEATURED_IDENTITIES = (
    "aravind srinivas", "perplexity", "andrej karpathy", "karpathy", "y combinator",
)
TRUSTED_CHANNEL_MARKERS = (
    "andrej karpathy", "perplexity", "y combinator", "sequoia capital",
    "stanford online", "stanford graduate school of business", "ai engineer",
    "lex fridman", "ted", "20vc", "founders forum", "dwarkesh patel",
    "lenny's podcast", "no priors", "cnbc", "bloomberg technology",
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
FILLER_ONLY = re.compile(r"^(?:um+|uh+|you know|like|well|so)[,.!? ]*$", re.IGNORECASE)
NON_SPEECH_TRANSLATIONS = {
    "music": "[音乐]", "applause": "[掌声]", "laughter": "[笑声]",
}


class YouTubeAcquisitionError(RuntimeError):
    pass


class YouTubeWebAuthRequired(YouTubeAcquisitionError):
    pass


class SourceBelow1080Error(YouTubeAcquisitionError):
    pass


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    cadence_hours: int = 48
    max_source_selections: int = 1
    minimum_score: float = 70.0
    minimum_duration_seconds: int = 900
    maximum_duration_seconds: int = 7200
    lookback_days: int = 30
    results_per_query: int = 8
    metadata_probe_limit: int = 10
    timezone: str = "Asia/Tokyo"
    query_pools: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_QUERY_POOLS))

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
    transcript_available: bool = False
    source_width: int = 0
    source_height: int = 0
    source_quality_verified: bool = False
    matched_pools: list[str] = field(default_factory=list)
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    eligible: bool = False
    rejection_reasons: list[str] = field(default_factory=list)

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
        current_year = self.clock().astimezone(UTC).year
        for pool, queries in config.query_pools.items():
            for query in queries:
                dated_query = query if str(current_year) in query else f"{query} {current_year} latest"
                command = [
                    self._executable(), "--flat-playlist", "--dump-single-json", "--skip-download",
                    *self.runtime.extractor_arguments("gvs"),
                    f"ytsearch{config.results_per_query}:{dated_query}",
                ]
                completed = self.runner(command, check=False, capture_output=True, text=True)
                if completed.returncode != 0:
                    raise RuntimeError((completed.stderr or completed.stdout).strip() or f"YouTube search failed: {query}")
                payload = json.loads(completed.stdout)
                for raw in payload.get("entries", []):
                    video_id = str(raw.get("id") or "").strip()
                    if not video_id:
                        continue
                    raw_url = str(raw.get("webpage_url") or raw.get("url") or "")
                    canonical_url = raw_url if raw_url.startswith(("http://", "https://")) else f"https://www.youtube.com/watch?v={video_id}"
                    item = results.setdefault(video_id, YouTubeCandidate(
                        video_id=video_id,
                        url=canonical_url,
                        title=str(raw.get("title") or ""),
                        channel=str(raw.get("channel") or raw.get("uploader") or ""),
                        duration_seconds=float(raw.get("duration") or 0),
                        view_count=int(raw.get("view_count") or 0),
                    ))
                    if pool not in item.matched_pools:
                        item.matched_pools.append(pool)
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
        limit = max(0, config.metadata_probe_limit)
        if not limit:
            return []

        def priority(item: YouTubeCandidate) -> tuple[int, int, int, int, str]:
            channel = item.channel.casefold()
            identity = f"{item.title} {item.channel}".casefold()
            return (
                int(any(marker in channel for marker in TRUSTED_CHANNEL_MARKERS)),
                int(any(marker in identity for marker in FEATURED_IDENTITIES)),
                len(item.matched_pools), item.view_count, item.video_id,
            )

        ranked = sorted(candidates, key=priority, reverse=True)
        pools = list(config.query_pools)
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
        haystack = f"{item.title} {item.channel} {item.description}".casefold()
        channel = item.channel.casefold()
        reasons = list(item.rejection_reasons)
        if not any(marker in haystack for marker in AUDIENCE_MARKERS):
            reasons.append("audience_mismatch")
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

        audience = 25.0 if any(marker in haystack for marker in AUDIENCE_MARKERS) else 0.0
        if trusted_channel:
            authority = 20.0
        elif any(marker in haystack for marker in FEATURED_IDENTITIES):
            authority = 9.0
        else:
            authority = 6.0
        insight_hits = sum(1 for marker in INSIGHT_MARKERS if marker in haystack)
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
        if entry.strategy == TerminologyStrategy.TRANSLATE and not entry.target.strip():
            errors.append(f"term:{entry.source}: translated terms require target")
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


def _contains_term(value: str, term: str) -> bool:
    """Prevent acronym matches such as RAG→organization and LLM→willmake."""
    if term.isupper() and re.fullmatch(r"[A-Z0-9.+-]+", term):
        return bool(re.search(
            rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", value,
            re.IGNORECASE,
        ))
    return term.casefold() in value.casefold()


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


SOURCE_DANGLING_END = re.compile(
    r"(?:\b(?:a|an|and|are|as|at|because|but|by|can|could|for|from|if|in|is|my|"
    r"of|on|or|our|that|the|their|then|this|to|uh|um|was|were|when|where|which|"
    r"while|will|with|would|your)|[-,:;])\s*[\"'”’]?$",
    re.IGNORECASE,
)


def rebalance_source_cues(cues: list[TranscriptCue], maximum_duration: float = 8.0) -> list[TranscriptCue]:
    """Merge caption fragments whose visible English cannot stand on its own."""
    balanced: list[TranscriptCue] = []
    index = 0
    while index < len(cues):
        current = cues[index]
        if index + 1 < len(cues):
            following = cues[index + 1]
            combined_duration = following.end - current.start
            dangling = bool(SOURCE_DANGLING_END.search(current.source_text.strip()))
            connector_only = bool(re.fullmatch(
                r"(?:and|but|or|so|then|because|uh|um)[,\s]*",
                current.source_text.strip(), re.IGNORECASE,
            ))
            if (
                (dangling or connector_only) and following.start - current.end <= 0.8
                and combined_duration <= maximum_duration
            ):
                balanced.append(TranscriptCue(
                    id=current.id, start=current.start, end=following.end,
                    source_text=f"{current.source_text} {following.source_text}".strip(),
                    translation="", speaker=current.speaker or following.speaker,
                    confidence=current.confidence,
                ))
                index += 2
                continue
        balanced.append(current)
        index += 1
    return balanced


def _required_short_source_ranges(
    cues: list[TranscriptCue], story_start: float, story_end: float,
) -> list[dict[str, float]]:
    """Choose complete short-source lesson boundaries; the model only names them."""
    span = story_end - story_start
    count = min(8, max(3, math.ceil(span / 330.0)))
    if not 180 <= span / count <= 360:
        raise ValueError(f"short source cannot be divided into 3–8 lessons of 180–360 seconds: {span:.1f}s")
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


class NaturalSubtitleTranslator:
    def __init__(self, writer: OpenAICompatibleStoryWriter) -> None:
        self.writer = writer

    def translate(
        self, metadata: dict[str, Any], cues: list[TranscriptCue],
    ) -> tuple[list[TerminologyEntry], dict[str, Any], list[dict[str, Any]]]:
        cues[:] = rebalance_source_cues(cues)
        transcript = [
            {"id": item.id, "start": item.start, "end": item.end, "text": item.source_text}
            for item in cues
        ]
        duration = float(metadata.get("duration") or (cues[-1].end if cues else 0))
        mandatory_layout = ""
        if duration <= 2700:
            lesson_ranges = _required_short_source_ranges(cues, 0.0, duration)
            mandatory_layout = (
                "Mandatory short-source layout: set story_start=0 and story_end="
                f"{duration:.3f}; return exactly one Bilibili chapter with that same range; "
                "return WeChat lessons using these exact ranges in this exact order, changing "
                "only title, thesis, framing, and hook_headlines: "
                + json.dumps(lesson_ranges, ensure_ascii=False)
            )
        planning_prompt = "\n".join([
            "You are the senior Chinese editor for an evidence-bound AI engineering video channel.",
            "Audience: Chinese developers, AI builders, tech leads, and platform leaders.",
            "Create a curated but faithful edition. Natural Chinese matters more than mirroring English word order.",
            "Never invent a Chinese term merely to make the subtitle fully Chinese. Product names, code, APIs, and emerging terms without a settled Chinese equivalent stay in English.",
            "Every terminology item must choose translate, preserve, or bilingual_once. bilingual_once keeps the English term, adds one short Chinese explanation at first use, and then keeps English only.",
            "Plan two independent editions from the same source. Bilibili is a complete study collection: chronological, non-overlapping chapters must cover the whole substantive story, excluding only greetings, sponsor messages, dead air, or credits. Each chapter should normally be 10–35 minutes. If the substantive story is at most 35 minutes, return exactly one Bilibili chapter spanning story_start through story_end; do not turn it into highlights. WeChat uses 3–6 minute mini-lessons.",
            "If the substantive source is at most 45 minutes, the WeChat lessons must also be chronological and collectively cover the whole story. If it is longer than 45 minutes, choose only the strongest 4–8 standalone WeChat lessons; do not force full coverage.",
            "Return JSON with collection_title, story_start, story_end, terminology, bilibili_chapters, and wechat_lessons. Every chapter and lesson has title, thesis, numeric start/end source seconds, framing (auto|speaker|slide|split), and hook_headlines. Use split when the source simultaneously shows a speaker pane and a slide pane; this preserves the complete left speaker instead of treating the slide crop as the whole frame. Use slide only for a true slide-only shot or when the crop retains all meaningful content.",
            "hook_headlines contains exactly three distinct, evidence-faithful Simplified Chinese headlines of 8–26 visible characters: one tension or contrarian claim, one concrete technical choice with stakes, and one consequence or outcome claim. Make each one specific enough to earn the next 8 seconds, not merely label the topic. Every headline must be a complete clause; never cut an English term, append a generic 为什么, ask an empty question, or use sensational clickbait.",
            "Titles and thesis must be natural Simplified Chinese; each theme title must be a complete phrase of at most 30 visible characters.",
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
        ], max_tokens=7000)
        plan, plan_repairs = self.ensure_editorial_plan(metadata, cues, plan)
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
            {"step": "translation_plan", "provenance": provenance}, *plan_repairs,
        ]
        for offset in range(0, len(cues), 45):
            chunk = cues[offset:offset + 45]
            expected = {item.id for item in chunk}
            translations: dict[str, str] = {}
            for attempt in range(3):
                pending = [item for item in chunk if item.id not in translations]
                chunk_payload = [
                    {"id": item.id, "start": item.start, "end": item.end, "source": item.source_text}
                    for item in pending
                ]
                prompt = "\n".join([
                    "Translate these English transcript cues into concise, natural Simplified Chinese subtitles for Chinese AI/software practitioners.",
                    "Translate meaning in chapter context, never English word order. Preserve uncertainty, negation, numbers, scope, and speaker attribution.",
                    "Strict cue alignment: each Chinese text may translate only the source text carrying the same id. Never move meaning to the previous or next id, never finish a neighboring cue, and never borrow words from another cue. If a supplied source is a fragment, keep the Chinese fragment aligned instead of completing it from context.",
                    "Translate every supplied cue, including brief spoken fillers, because the original English caption remains visible. Render fillers naturally in Chinese without adding claims. Keep English when the glossary says preserve. For bilingual_once, add the exact first_use_explanation only in first_use_cue_id; all later occurrences keep English without repeating the explanation.",
                    "Return {translations:[{id,text}]}; return every supplied id exactly once and no extra ids.",
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
                empty = sorted(key for key, value in returned.items() if not value)
                if extra or empty:
                    raise ValueError(f"translation response invalid; empty={empty}, extra={extra}")
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
                raise ValueError("; ".join(errors))
        return terminology, plan, traces

    def ensure_editorial_plan(
        self, metadata: dict[str, Any], cues: list[TranscriptCue], plan: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        duration = float(metadata.get("duration") or (cues[-1].end if cues else 0))
        errors = editorial_plan_contract_errors(plan, duration)
        traces: list[dict[str, Any]] = []
        if not errors:
            return plan, traces
        transcript = [
            {"id": item.id, "start": item.start, "end": item.end, "text": item.source_text}
            for item in cues
        ]
        original_terminology = plan.get("terminology", [])
        for attempt in range(2):
            mandatory_layout = ""
            if duration <= 2700:
                lesson_ranges = _required_short_source_ranges(cues, 0.0, duration)
                mandatory_layout = (
                    "This is a mandatory deterministic boundary contract: story_start=0, "
                    f"story_end={duration:.3f}, one Bilibili chapter from 0 to {duration:.3f}, "
                    "and WeChat lessons with exactly these ordered start/end pairs: "
                    + json.dumps(lesson_ranges, ensure_ascii=False)
                    + ". Copy every number exactly; write only the titles, theses, framing, and hooks."
                )
            prompt = "\n".join([
                "Repair this editorial plan for a Chinese AI engineering video collection. Do not translate subtitles.",
                "Return the complete plan with collection_title, story_start, story_end, terminology, bilibili_chapters, and wechat_lessons.",
                "Bilibili chapters must be chronological, non-overlapping, normally 600–2100 seconds each, and cover at least 95% of the complete substantive story with no gap above 60 seconds. If story_end-story_start is at most 2100 seconds, return exactly one chapter from story_start to story_end. story_start/story_end may exclude only greetings, sponsor messages, dead air, or credits and must retain at least 90% of the source.",
                "WeChat lessons must be 180–360 seconds with distinct natural Chinese titles and concrete theses. For sources at most 45 minutes, return 3–8 chronological, non-overlapping lessons covering at least 90% of the substantive story. For longer sources, select 4–8 strongest standalone lessons without a coverage requirement.",
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
            ], max_tokens=7000)
            if not repaired.get("terminology") and original_terminology:
                repaired["terminology"] = original_terminology
            for key in ("collection_title",):
                if not str(repaired.get(key) or "").strip():
                    repaired[key] = plan.get(key, "")
            plan = repaired
            errors = editorial_plan_contract_errors(plan, duration)
            traces.append({
                "step": "editorial_plan_repair", "attempt": attempt + 1,
                "errors_after": errors, "provenance": provenance,
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
            elif entry.strategy == TerminologyStrategy.BILINGUAL_ONCE:
                needs_source = not _contains_term(first.translation, entry.source)
                needs_explanation = entry.first_use_explanation not in first.translation
                if needs_source or needs_explanation:
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
            "Return {translations:[{id,text}]}; return every supplied id exactly once and no extra ids.",
            "Contract errors: " + json.dumps(errors, ensure_ascii=False),
            "Relevant terminology: " + json.dumps([asdict(item) for item in affected_terms], ensure_ascii=False),
            "Cues: " + json.dumps(payload, ensure_ascii=False),
        ])
        draft, provenance = self.writer._request_json([
            {"role": "system", "content": "Return one valid JSON object only."},
            {"role": "user", "content": prompt},
        ], max_tokens=2500)
        translations = {
            str(item.get("id")): str(item.get("text") or "").strip()
            for item in draft.get("translations", []) if isinstance(item, dict)
        }
        expected = set(affected)
        if set(translations) != expected or any(not value for value in translations.values()):
            missing = sorted(expected - set(translations))
            extra = sorted(set(translations) - expected)
            raise ValueError(f"terminology repair ids mismatch; missing={missing}, extra={extra}")
        for cue_id, cue in affected.items():
            cue.translation = translations[cue_id]
        return {"step": "terminology_repair", "errors": errors, "provenance": provenance}

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
            if strategy == TerminologyStrategy.BILINGUAL_ONCE and not explanation:
                explanation = target
            entries.append(TerminologyEntry(
                source=source, strategy=strategy, target=target,
                first_use_explanation=explanation,
                notes=str(item.get("notes") or "").strip(),
            ))
        source = "\n".join(item.source_text for item in cues)
        known = {item.source.casefold() for item in entries}
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
            source_range and 480 <= source_range.duration <= 2100
            and 4 <= visible <= 30 and thesis
            and len(hooks) == 3 and len(valid_hooks) == 3 and len(set(hooks)) == 3
            and not title.endswith(("，", "、", "：", "；", ",", ":", ";"))
        ):
            chapters.append(source_range)
            chapter_titles.append(_normalized_title(title))
    if len(raw_chapters) != len(chapters) or not 1 <= len(chapters) <= 8:
        errors.append(
            "bilibili_chapters must contain 1–8 complete study chapters of 480–2100 seconds"
        )
    if story_end - story_start <= 2100 and len(chapters) != 1:
        errors.append("a story of 35 minutes or less must use exactly one complete Bilibili chapter")
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
    minimum_lessons = 3 if duration <= 2700 else 4
    if len(raw_lessons) != len(lessons) or not minimum_lessons <= len(lessons) <= 8:
        errors.append(
            f"wechat_lessons must contain {minimum_lessons}–8 complete lessons of 180–360 seconds"
        )
    if len(set(lesson_titles)) != len(lesson_titles):
        errors.append("WeChat lesson titles must be distinct")
    if duration <= 2700 and lessons:
        errors.extend(_coverage_contract_errors(
            lessons, story_start, story_end, 0.9, 60.0, "short-source WeChat lessons",
        ))
    return errors


def editorial_plan_contract_errors(plan: dict[str, Any], duration: float) -> list[str]:
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


def _headline_fragment(value: str, limit: int = 26) -> str:
    compact = re.sub(r"\s+", " ", value).strip().rstrip("。")
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


def build_hook_candidates(
    title: str, thesis: str, source_range: SourceRange, cues: list[TranscriptCue], item_id: str,
    proposed_headlines: Any = None,
) -> list[HookSpec]:
    eligible = [
        cue for cue in cues
        if cue.start >= source_range.start and cue.end <= source_range.end
        and cue.source_text.strip() and not HOOK_GREETING.search(cue.source_text)
    ]
    windows: list[tuple[float, list[TranscriptCue]]] = []
    for index, cue in enumerate(eligible):
        group = [cue]
        end_index = index + 1
        while group[-1].end - group[0].start < 6.0 and end_index < len(eligible):
            following = eligible[end_index]
            if following.start - group[-1].end > 0.8 or following.end - group[0].start > 10.0:
                break
            group.append(following)
            end_index += 1
        duration = group[-1].end - group[0].start
        if not 6.0 <= duration <= 10.0:
            continue
        combined = " ".join(item.source_text for item in group).casefold()
        signal_hits = sum(1 for marker in HOOK_SIGNAL_MARKERS if marker in combined)
        concrete_hits = sum(1 for marker in HOOK_CONCRETE_MARKERS if marker in combined)
        weak_hits = sum(1 for marker in HOOK_WEAK_MARKERS if marker in combined)
        specificity = min(4, len(re.findall(r"\b[A-Z][A-Za-z0-9.+-]*\b", " ".join(item.source_text for item in group))))
        score = (
            signal_hits * 3 + concrete_hits * 2 + specificity
            + min(5, len(combined) / 35) - weak_hits * 4
        )
        windows.append((score, group))
    windows.sort(key=lambda row: (-row[0], row[1][0].start))
    selected_windows: list[list[TranscriptCue]] = []
    for _, group in windows:
        if any(abs(group[0].start - existing[0].start) < 2.0 for existing in selected_windows):
            continue
        selected_windows.append(group)
        if len(selected_windows) == 3:
            break
    if not selected_windows:
        raise ValueError(f"{item_id}: no evidence-backed 6–10 second hook window")
    while len(selected_windows) < 3:
        selected_windows.append(selected_windows[-1])

    title_headline = _headline_fragment(title, 30)
    if not title_headline:
        raise ValueError(f"{item_id}: title cannot be shortened without breaking a clause or English term")
    subject = _headline_fragment(title.split("：", 1)[0], 18) or title_headline
    fallback_headlines = [
        title_headline,
        _headline_fragment(thesis, 30) or f"{subject}的关键取舍是什么？",
        _headline_fragment(f"{subject}真正改变了什么？", 30) or "真正的工程代价是什么？",
    ]
    supplied = [str(item).strip() for item in proposed_headlines] if isinstance(proposed_headlines, list) else []
    headlines: list[str] = []
    for raw in [*supplied, *fallback_headlines]:
        headline = _headline_fragment(raw, 30)
        if headline and headline not in headlines:
            headlines.append(headline)
        if len(headlines) == 3:
            break
    if len(headlines) != 3:
        raise ValueError(f"{item_id}: three complete, distinct hook headlines are required")
    strategies = [HookStrategy.CONTRARIAN, HookStrategy.QUESTION, HookStrategy.OUTCOME]
    hooks: list[HookSpec] = []
    for index, (group, headline, strategy) in enumerate(zip(selected_windows, headlines, strategies), start=1):
        hooks.append(HookSpec(
            id=f"{item_id}-hook-{index}", strategy=strategy,
            headline_zh=headline, promise=thesis,
            source_range=SourceRange(
                group[0].start, group[-1].end,
                source_range.framing if source_range.framing != FramingMode.AUTO
                else (FramingMode.SPEAKER if group[0].speaker else FramingMode.AUTO),
                "evidence-backed cold open",
                source_range.crop_x, source_range.crop_y,
                source_range.crop_width, source_range.crop_height,
            ),
            source_cue_ids=[item.id for item in group],
            payoff_cue_ids=[item.id for item in group],
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
    headline_length = len(re.sub(r"\s+", "", hook.headline_zh))
    if not 6 <= headline_length <= 30:
        errors.append("hook headline must contain 6–30 visible characters")
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
    contract_errors = editorial_plan_contract_errors(plan, duration)
    if contract_errors:
        raise ValueError("editorial plan contract failed: " + "; ".join(contract_errors))
    if "bilibili_chapters" in plan or "wechat_lessons" in plan:
        collection_id = f"youtube-{candidate.metadata.get('video_id') or candidate.id}-{uuid.uuid4().hex[:8]}"
        source_line = f"来源：{candidate.author or metadata.get('channel') or 'YouTube'}｜{candidate.source_url}"
        items: list[CollectionItem] = []
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
            )
            items.append(CollectionItem(
                id=item_id, kind=CollectionItemKind.WECHAT_SHORT,
                order=len(chapter_rows) + index, title=title, thesis=thesis,
                source_ranges=[source_range],
                renders=[PlatformRender(
                    RenderProfile.WECHAT_VERTICAL, 1080, 1920, title=title,
                    description=source_line, tags=["AI", "开发者", "技术团队", "短课"],
                    hook_candidates=hook_candidates, selected_hook=hook_candidates[0],
                )],
            ))
        return VideoCollectionManifest(
            id=collection_id, candidate_id=candidate.id, source_url=candidate.source_url,
            source_video_id=str(candidate.metadata.get("video_id") or ""),
            source_title=candidate.title, source_channel=candidate.author or "",
            collection_title=str(plan.get("collection_title") or f"{candidate.author or 'AI'} 中文学习合集").strip(),
            transcript=cues, terminology=terminology, items=items,
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
        metadata = self._metadata(url)
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
            subtitle = self._subtitle(url, video_id, job)
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

    def _download_media(self, url: str, video_id: str, job: Path) -> Path:
        output = str(job / "%(id)s.%(ext)s")
        command = [
            self._executable(), "--no-playlist",
            "-f", "bestvideo[height>=1080][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[height>=1080]+bestaudio/best[height>=1080]",
            "--merge-output-format", "mkv", *self._extractor_args(), *self._auth_args(),
            "-o", output, url,
        ]
        completed = self.runner(command, check=False, capture_output=True, text=True)
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
    hook = render.selected_hook
    if hook is None:
        return list(item.source_ranges)
    hook_range = hook.source_range
    for source_range in item.source_ranges:
        if source_range.start <= hook_range.start and hook_range.end <= source_range.end:
            hook_range = SourceRange(
                hook_range.start, hook_range.end,
                source_range.framing if hook_range.framing == FramingMode.AUTO else hook_range.framing,
                hook_range.reason,
                hook_range.crop_x if hook_range.has_explicit_crop else source_range.crop_x,
                hook_range.crop_y if hook_range.has_explicit_crop else source_range.crop_y,
                hook_range.crop_width if hook_range.has_explicit_crop else source_range.crop_width,
                hook_range.crop_height if hook_range.has_explicit_crop else source_range.crop_height,
            )
            break
    body: list[SourceRange] = []
    removed_duration = 0.0
    for source_range in item.source_ranges:
        overlap_start = max(source_range.start, hook_range.start)
        overlap_end = min(source_range.end, hook_range.end)
        if overlap_end <= overlap_start:
            body.append(source_range)
            continue
        removed_duration += overlap_end - overlap_start
        if source_range.start < overlap_start:
            body.append(SourceRange(
                source_range.start, overlap_start, source_range.framing, source_range.reason,
                source_range.crop_x, source_range.crop_y,
                source_range.crop_width, source_range.crop_height,
            ))
        if overlap_end < source_range.end:
            body.append(SourceRange(
                overlap_end, source_range.end, source_range.framing, source_range.reason,
                source_range.crop_x, source_range.crop_y,
                source_range.crop_width, source_range.crop_height,
            ))
    if abs(removed_duration - hook_range.duration) > 0.05 or not body:
        raise ValueError(f"{item.id}: selected hook must be fully contained in the episode ranges")
    return [hook_range, *body]


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
    if len(lines) > 2:
        midpoint = max(1, len(words) // 2)
        lines = [" ".join(words[:midpoint]), " ".join(words[midpoint:])]
    return "\n".join(lines)


def _subtitle_rows(
    manifest: VideoCollectionManifest, ranges: list[SourceRange], profile: RenderProfile,
) -> list[tuple[float, float, str, str]]:
    chinese_limit = 22 if profile == RenderProfile.BILIBILI_LANDSCAPE else 16
    english_limit = 72 if profile == RenderProfile.BILIBILI_LANDSCAPE else 42
    rows: list[tuple[float, float, str, str]] = []
    offset = 0.0
    for source_range in ranges:
        for cue in manifest.transcript:
            if cue.end <= source_range.start or cue.start >= source_range.end or not cue.translation.strip():
                continue
            start = offset + max(0.0, cue.start - source_range.start)
            end = offset + min(source_range.duration, cue.end - source_range.start)
            if end > start:
                rows.append((
                    start, max(end, start + 0.8),
                    _wrap_english(cue.source_text, english_limit),
                    wrap_subtitle(cue.translation, chinese_limit),
                ))
        offset += source_range.duration
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
    height = 250 if profile == RenderProfile.BILIBILI_LANDSCAPE else 330
    english_size = 24 if profile == RenderProfile.BILIBILI_LANDSCAPE else 28
    chinese_size = 40 if profile == RenderProfile.BILIBILI_LANDSCAPE else 44
    configured_font = Path(os.environ.get(
        "VIDEO_FACTORY_FONT",
        "/Users/clairehou/pyProjects/MoneyPrinterTurbo/resource/fonts/STHeitiMedium.ttc",
    ))
    if not configured_font.is_file():
        configured_font = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    if not configured_font.is_file():
        raise FileNotFoundError("set VIDEO_FACTORY_FONT to a Chinese-capable TTF/TTC")
    english_font = ImageFont.truetype(str(configured_font), english_size)
    chinese_font = ImageFont.truetype(str(configured_font), chinese_size)
    frame_dir = output.with_suffix(".subtitle-frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    blank = frame_dir / "blank.png"
    Image.new("RGBA", (width, height), (0, 0, 0, 0)).save(blank)

    rendered: dict[tuple[str, str], Path] = {}
    sequence: list[tuple[Path, float]] = []
    cursor = 0.0
    source_rows = _read_srt_rows(source_subtitle)
    translation_rows = _read_srt_rows(translation_subtitle)
    if len(source_rows) != len(translation_rows):
        raise ValueError("English and Chinese subtitle row counts differ")
    for source_row, translation_row in zip(source_rows, translation_rows):
        start, end, english = source_row
        zh_start, zh_end, chinese = translation_row
        if abs(start - zh_start) > 0.02 or abs(end - zh_end) > 0.02:
            raise ValueError("English and Chinese subtitle timing differs")
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
            english_bbox = draw.multiline_textbbox(
                (width / 2, 0), english, font=english_font, anchor="ma",
                align="center", spacing=6, stroke_width=1,
            )
            chinese_bbox = draw.multiline_textbbox(
                (width / 2, 0), chinese, font=chinese_font, anchor="ma",
                align="center", spacing=8, stroke_width=2,
            )
            english_height = english_bbox[3] - english_bbox[1]
            chinese_height = chinese_bbox[3] - chinese_bbox[1]
            gap = 12
            total_height = english_height + gap + chinese_height
            top = max(12, (height - total_height) / 2)
            max_text_width = max(
                english_bbox[2] - english_bbox[0], chinese_bbox[2] - chinese_bbox[0],
            )
            padding_x, padding_y = 28, 18
            draw.rounded_rectangle(
                (
                    width / 2 - max_text_width / 2 - padding_x, top - padding_y,
                    width / 2 + max_text_width / 2 + padding_x,
                    top + total_height + padding_y,
                ),
                radius=18, fill=(0, 0, 0, 170),
            )
            draw.multiline_text(
                (width / 2, top), english, font=english_font, fill=(215, 222, 233, 255),
                anchor="ma", align="center", spacing=6, stroke_width=1,
                stroke_fill=(0, 0, 0, 235),
            )
            draw.multiline_text(
                (width / 2, top + english_height + gap), chinese,
                font=chinese_font, fill=(255, 255, 255, 255),
                anchor="ma", align="center", spacing=8, stroke_width=2,
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
    configured_font = Path(os.environ.get(
        "VIDEO_FACTORY_FONT",
        "/Users/clairehou/pyProjects/MoneyPrinterTurbo/resource/fonts/STHeitiMedium.ttc",
    ))
    if not configured_font.is_file():
        configured_font = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    font = ImageFont.truetype(str(configured_font), font_size)
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


def _write_hook_overlay_concat(
    hook: HookSpec, duration: float, output: Path,
    profile: RenderProfile = RenderProfile.WECHAT_VERTICAL,
) -> tuple[Path, int]:
    from PIL import Image, ImageDraw, ImageFont

    is_bilibili = profile == RenderProfile.BILIBILI_LANDSCAPE
    width, height = (1920, 240) if is_bilibili else (1080, 250)
    configured_font = Path(os.environ.get(
        "VIDEO_FACTORY_FONT",
        "/Users/clairehou/pyProjects/MoneyPrinterTurbo/resource/fonts/STHeitiMedium.ttc",
    ))
    if not configured_font.is_file():
        configured_font = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    hero_font = ImageFont.truetype(str(configured_font), 66 if is_bilibili else 58)
    compact_font = ImageFont.truetype(str(configured_font), 32)
    frame_dir = output.with_suffix(".hook-frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    hero_frame = frame_dir / "hook.png"
    compact_frame = frame_dir / "hook-compact.png"
    blank_frame = frame_dir / "blank.png"

    hero_canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hero_canvas)
    headline = wrap_subtitle(hook.headline_zh, 28 if is_bilibili else 18)
    bbox = draw.multiline_textbbox(
        (width / 2, height / 2), headline, font=hero_font, anchor="mm",
        align="center", spacing=10, stroke_width=2,
    )
    draw.rounded_rectangle(
        (bbox[0] - 34, bbox[1] - 22, bbox[2] + 34, bbox[3] + 22),
        radius=22, fill=(3, 8, 20, 218), outline=(255, 224, 99, 220), width=3,
    )
    draw.multiline_text(
        (width / 2, height / 2), headline, font=hero_font, fill=(255, 255, 255, 255),
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
    hero_duration = min(duration, hook.source_range.duration if is_bilibili else 7.0)
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
                parts.append(
                    f"[blur{index}][fit{index}]overlay=(W-w)/2:380[v{index}]"
                )
            else:
                parts.append(f"[0:v]{video_filters}[v{index}]")
            parts.append(
                f"[0:a]atrim=start={source_range.start:.3f}:end={source_range.end:.3f},asetpts=PTS-STARTPTS[a{index}]"
            )
            concat_inputs.append(f"[v{index}][a{index}]")
        parts.append("".join(concat_inputs) + f"concat=n={len(source_ranges)}:v=1:a=1[cv][outa]")
        if profile == RenderProfile.BILIBILI_LANDSCAPE:
            parts.append(
                f"[cv]scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black[base]"
            )
            parts.append("[1:v]format=rgba,setpts=PTS-STARTPTS[subs]")
            parts.append(f"[base][subs]overlay=(W-w)/2:H-{subtitle_height}-67:eof_action=pass[captioned]")
        else:
            parts.extend([
                # Each source segment is already composed into the portrait canvas so
                # slide crops and speaker framing can change at edit boundaries.
                "[cv]null[base]",
                "[1:v]format=rgba,setpts=PTS-STARTPTS[subs]",
                f"[base][subs]overlay=(W-w)/2:H-{subtitle_height}-340:eof_action=pass[captioned]",
            ])
        current_label = "captioned"
        next_input = 2
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
    study_mode = any(item.kind in {
        CollectionItemKind.BILIBILI_CHAPTER, CollectionItemKind.WECHAT_SHORT,
    } for item in manifest.items)
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
            duration_ok = 480 <= item.duration <= 2100
            target = "480–2100"
        elif item.kind == CollectionItemKind.WECHAT_SHORT:
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
                    CheckResult(f"render:{item.id}:{render.profile}:pixel", probe.pixel_format == "yuv420p", probe.pixel_format),
                    CheckResult(
                        f"render:{item.id}:{render.profile}:duration",
                        abs(probe.duration - item.duration) <= 0.25,
                        f"{probe.duration:.2f}s / target {item.duration:.2f}s",
                    ),
                ])
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
    untranslated = [item.id for item in manifest.transcript if not item.translation.strip()]
    missing_source = [item.id for item in manifest.transcript if not item.source_text.strip()]
    checks.append(CheckResult(
        "source_subtitles_preserved", not missing_source,
        "all original English caption cues preserved" if not missing_source
        else f"missing English: {', '.join(missing_source[:10])}",
    ))
    checks.append(CheckResult(
        "translation_complete", not untranslated,
        "all transcript cues translated" if not untranslated else f"missing: {', '.join(untranslated[:10])}",
    ))
    term_errors = terminology_contract_errors(manifest.transcript, manifest.terminology)
    checks.append(CheckResult(
        "terminology_contract", not term_errors,
        "natural terminology contract passed" if not term_errors else "; ".join(term_errors),
    ))
    fast = [
        item.id for item in manifest.transcript
        if item.translation.strip() and len(re.sub(r"\s+", "", item.translation)) / max(item.duration, 0.1) > 12
    ]
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
        stage = re.fullmatch(r"\[([^\]]+)\]", cue.source_text.strip().casefold())
        if stage:
            cue.translation = NON_SPEECH_TRANSLATIONS.get(stage.group(1), f"[{stage.group(1)}]")
        elif FILLER_ONLY.fullmatch(re.sub(r"\s+", " ", cue.source_text).strip()):
            cue.translation = "嗯" if cue.source_text.strip().casefold().startswith(("um", "uh")) else "那么，"
        else:
            raise ValueError(f"reviewed translation is missing source cue: {cue.id} {cue.source_text!r}")


class YouTubeCollectionFactory:
    def __init__(self, workspace: Workspace, writer: OpenAICompatibleStoryWriter) -> None:
        self.workspace = workspace
        self.writer = writer

    def generate(
        self, url: str, job: Path, render: bool = True, local_media: Path | None = None,
        local_subtitles: Path | None = None,
        translation_plan: Path | None = None,
    ) -> dict[str, Any]:
        acquirer = YouTubeAcquirer(self.workspace)
        candidate, evidence, metadata, cues, media_asset, subtitle_asset, media_info = acquirer.acquire(
            url, job, local_media=local_media, local_subtitles=local_subtitles,
            download_media=render,
        )
        translator = NaturalSubtitleTranslator(self.writer)
        if translation_plan is not None:
            cues[:] = rebalance_source_cues(cues)
            cached = json.loads(translation_plan.read_text(encoding="utf-8"))
            cached_cues = [TranscriptCue(**item) for item in cached.get("transcript", [])]
            if not cached_cues or _translation_source_fingerprint(cached_cues) != _translation_source_fingerprint(cues):
                raise ValueError("translation plan transcript does not match the supplied YouTube subtitles")
            if all(cue.source_text.strip() and cue.translation.strip() for cue in cached_cues):
                # The reviewed plan may merge adjacent source fragments again to meet
                # reading-speed limits. The whole-source fingerprint above proves that
                # no English content was lost, so preserve its final aligned timeline.
                cues[:] = cached_cues
            else:
                _merge_cached_translations(cues, cached_cues)
            editorial_plan = dict(cached.get("editorial_plan") or {})
            terminology = translator._parse_terminology(cached.get("terminology", []), cues)
            trace = list(cached.get("trace") or [])
            editorial_plan, repairs = translator.ensure_editorial_plan(metadata, cues, editorial_plan)
            trace.extend(repairs)
        else:
            terminology, editorial_plan, trace = translator.translate(metadata, cues)
        (job / "translation-plan.json").write_text(json.dumps({
            "editorial_plan": editorial_plan,
            "terminology": [asdict(item) for item in terminology],
            "transcript": [asdict(item) for item in cues],
            "trace": trace,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
