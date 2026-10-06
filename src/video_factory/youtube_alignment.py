from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from .models import SourceWord, TranscriptCue
from .youtube_runtime import ManagedYouTubeRuntime, STABLE_TS_COMMIT


ALIGNMENT_POLICY_VERSION = 3


def source_ledger_fingerprint(words: list[SourceWord]) -> str:
    return hashlib.sha256(json.dumps(
        [
            (word.id, word.raw, round(word.youtube_start, 3), round(word.youtube_end, 3))
            for word in words
        ],
        ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()


@dataclass(slots=True)
class SourceAlignment:
    words: list[SourceWord]
    status: str
    fingerprint: str
    engine: str
    model: str
    cache_hit: bool = False
    diagnostic: str = ""
    media_sha256: str = ""
    audio_hypothesis: str = ""

    def trace(self) -> dict[str, Any]:
        aligned = sum(word.alignment_status == "aligned" for word in self.words)
        low_confidence = sum(word.alignment_status == "low_confidence" for word in self.words)
        json3_timed = sum(word.alignment_status == "youtube_token" for word in self.words)
        linear_fallback = sum(word.alignment_status == "youtube" for word in self.words)
        return {
            "step": "interview_source_word_alignment",
            "policy_version": ALIGNMENT_POLICY_VERSION,
            "fingerprint": self.fingerprint,
            "source_ledger_fingerprint": source_ledger_fingerprint(self.words),
            "media_sha256": self.media_sha256,
            "engine": self.engine,
            "model": self.model,
            "status": self.status,
            "cache_hit": self.cache_hit,
            "word_count": len(self.words),
            "aligned_words": aligned,
            "low_confidence_words": low_confidence,
            "json3_timed_words": json3_timed,
            "linear_fallback_words": linear_fallback,
            "diagnostic": self.diagnostic,
            "audio_hypothesis_sha256": (
                hashlib.sha256(self.audio_hypothesis.encode("utf-8")).hexdigest()
                if self.audio_hypothesis else ""
            ),
        }


def _normalized_word(value: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", value.casefold().replace("’", "'"))


def source_words_from_cues(cues: list[TranscriptCue]) -> list[SourceWord]:
    """Build stable word identities without changing the authoritative text."""
    result: list[SourceWord] = []
    ordinal = 0
    for cue in cues:
        raw_words = re.findall(r"\S+", cue.source_text)
        if not raw_words:
            continue
        duration = max(0.001, cue.end - cue.start)
        timed_rows = [
            row for row in cue.source_tokens
            if isinstance(row, dict) and str(row.get("raw") or "").strip()
        ]
        use_token_timing = (
            [str(row.get("raw") or "").strip() for row in timed_rows] == raw_words
            and all(
                isinstance(row.get("start"), (int, float))
                and isinstance(row.get("end"), (int, float))
                and cue.start - 0.05 <= float(row["start"]) < float(row["end"]) <= cue.end + 0.05
                for row in timed_rows
            )
            and all(
                float(left["start"]) <= float(right["start"])
                and float(left["end"]) <= float(right["end"])
                for left, right in zip(timed_rows, timed_rows[1:])
            )
        )
        for local_index, raw in enumerate(raw_words):
            if use_token_timing:
                start = float(timed_rows[local_index]["start"])
                end = float(timed_rows[local_index]["end"])
            else:
                start = cue.start + duration * local_index / len(raw_words)
                end = cue.start + duration * (local_index + 1) / len(raw_words)
            result.append(SourceWord(
                id=f"{cue.id}:w{local_index + 1}", cue_id=cue.id,
                ordinal=ordinal, raw=raw, normalized=_normalized_word(raw),
                youtube_start=round(start, 3), youtube_end=round(end, 3),
                alignment_status="youtube_token" if use_token_timing else "youtube",
            ))
            ordinal += 1
    return result


def _flatten_aligned_words(payload: dict[str, Any]) -> list[dict[str, Any]]:
    direct = payload.get("words")
    if isinstance(direct, list):
        return [row for row in direct if isinstance(row, dict)]
    return [
        word
        for segment in payload.get("segments", [])
        if isinstance(segment, dict)
        for word in segment.get("words", [])
        if isinstance(word, dict)
    ]


def apply_alignment_payload(
    source_words: list[SourceWord], payload: dict[str, Any],
) -> tuple[str, str]:
    """Attach times only when stable-ts preserved the YouTube word sequence."""
    youtube_timing_status = {
        word.id: (
            "youtube_token" if word.alignment_status == "youtube_token" else "youtube"
        )
        for word in source_words
    }
    rows = _flatten_aligned_words(payload)
    aligned = [
        (
            _normalized_word(str(row.get("word") or row.get("text") or "")),
            float(row.get("start") or 0.0),
            float(row.get("end") or 0.0),
            row.get("probability", row.get("confidence")),
        )
        for row in rows
    ]
    aligned = [row for row in aligned if row[0] and row[2] > row[1]]
    expected_pairs = [
        (index, word.normalized) for index, word in enumerate(source_words)
        if word.normalized
    ]
    expected = [value for _, value in expected_pairs]
    observed = [row[0] for row in aligned]
    matcher = SequenceMatcher(None, expected, observed, autojunk=False)
    matching = sum(block.size for block in matcher.get_matching_blocks())
    coverage = matching / max(1, len(expected))
    credible_small_difference = matching >= max(2, len(expected) - 2)
    if not expected or (coverage < 0.8 and not credible_small_difference):
        return "audio_alignment_degraded", (
            f"stable-ts word sequence differed from authoritative YouTube text "
            f"({len(observed)} aligned versus {len(expected)} source words; "
            f"exact coverage {coverage:.1%})"
        )
    for tag, source_start, source_end, aligned_start, aligned_end in matcher.get_opcodes():
        source_indices = [expected_pairs[index][0] for index in range(source_start, source_end)]
        aligned_rows = aligned[aligned_start:aligned_end]
        if tag == "equal":
            for source_index, (_, start, end, raw_confidence) in zip(source_indices, aligned_rows):
                confidence = (
                    float(raw_confidence)
                    if isinstance(raw_confidence, (int, float)) else None
                )
                word = source_words[source_index]
                word.aligned_start = round(start, 3)
                word.aligned_end = round(end, 3)
                word.confidence = confidence
                word.alignment_status = (
                    "low_confidence"
                    if confidence is not None and confidence < 0.5 else "aligned"
                )
            continue
        if not source_indices:
            continue
        if aligned_rows:
            interval_start = aligned_rows[0][1]
            interval_end = aligned_rows[-1][2]
        else:
            previous = aligned[aligned_start - 1][2] if aligned_start else None
            following = aligned[aligned_start][1] if aligned_start < len(aligned) else None
            interval_start = previous if previous is not None else source_words[source_indices[0]].youtube_start
            interval_end = following if following is not None else source_words[source_indices[-1]].youtube_end
        interval_end = max(interval_start + 0.001, interval_end)
        for position, source_index in enumerate(source_indices):
            word = source_words[source_index]
            word.aligned_start = round(
                interval_start + (interval_end - interval_start) * position / len(source_indices), 3,
            )
            word.aligned_end = round(
                interval_start + (interval_end - interval_start) * (position + 1) / len(source_indices), 3,
            )
            word.confidence = 0.0
            word.alignment_status = "low_confidence"
    # Punctuation-only whitespace tokens are rare, but keeping them in the
    # immutable ledger is safer than deleting source text. Give them their
    # JSON3 timing and mark them for review.
    for word in source_words:
        if not word.normalized:
            word.alignment_status = "low_confidence"
    previous_start = -1.0
    previous_end = -1.0
    clip_end = max((word.youtube_end for word in source_words), default=0.0)
    for word in source_words:
        start, end = word.start, word.end
        if (
            start < -0.05 or end <= start
            or start + 0.05 < previous_start
            or end + 0.05 < previous_end
            or end > clip_end + 2.0
        ):
            for candidate in source_words:
                candidate.aligned_start = None
                candidate.aligned_end = None
                candidate.confidence = None
                candidate.alignment_status = youtube_timing_status[candidate.id]
            return "audio_alignment_degraded", (
                "stable-ts returned non-monotonic or out-of-bounds timing at "
                f"{word.id}: {start:.3f}-{end:.3f}"
            )
        previous_start, previous_end = start, end
    return "aligned", ""


class AudioWordAligner:
    """Narrow adapter around stable-ts; failure preserves JSON3 timing."""

    def __init__(
        self, runtime: ManagedYouTubeRuntime | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.runtime = runtime or ManagedYouTubeRuntime()
        self.runner = runner or subprocess.run

    def align(
        self, media_path: Path, cues: list[TranscriptCue], job: Path,
        media_sha256: str = "",
    ) -> SourceAlignment:
        words = source_words_from_cues(cues)
        model = os.environ.get("VIDEO_FACTORY_ALIGNMENT_MODEL", "base.en")
        fingerprint = hashlib.sha256(json.dumps({
            "policy": ALIGNMENT_POLICY_VERSION,
            "stable_ts_commit": STABLE_TS_COMMIT,
            "model": model,
            "media_sha256": media_sha256,
            "media_size": media_path.stat().st_size if media_path.is_file() and not media_sha256 else 0,
            "media_mtime_ns": media_path.stat().st_mtime_ns if media_path.is_file() and not media_sha256 else 0,
            "source_ledger_fingerprint": source_ledger_fingerprint(words),
        }, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cache_dir = job / "audio-alignment"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{fingerprint[:24]}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_words = [SourceWord(**row) for row in cached.get("source_words", [])]
            if cached_words:
                return SourceAlignment(
                    cached_words, str(cached.get("status") or "audio_alignment_degraded"),
                    fingerprint, f"stable-ts@{STABLE_TS_COMMIT}", model,
                    cache_hit=True, diagnostic=str(cached.get("diagnostic") or ""),
                    media_sha256=media_sha256,
                    audio_hypothesis=str(cached.get("audio_hypothesis") or ""),
                )
        transcript_path = cache_dir / f"{fingerprint[:24]}.txt"
        result_path = cache_dir / f"{fingerprint[:24]}.stable-ts.json"
        transcript_path.write_text(" ".join(word.raw for word in words), encoding="utf-8")
        python = self.runtime.settings.python
        helper = Path(__file__).with_name("stable_ts_align_worker.py")
        status = "audio_alignment_degraded"
        diagnostic = ""
        audio_hypothesis = ""
        if not media_path.is_file():
            diagnostic = f"media is unavailable for alignment: {media_path}"
        elif not python.is_file():
            diagnostic = "managed YouTube runtime has no Python interpreter"
        else:
            command = [
                str(python), str(helper), "--media", str(media_path),
                "--transcript", str(transcript_path), "--output", str(result_path),
                "--model", model,
                "--clip-start", str(min((word.youtube_start for word in words), default=0.0)),
                "--clip-end", str(max((word.youtube_end for word in words), default=0.0)),
            ]
            try:
                completed = self.runner(
                    command, capture_output=True, text=True, check=False,
                    timeout=int(os.environ.get("VIDEO_FACTORY_ALIGNMENT_TIMEOUT_SECONDS", "1200")),
                )
                if completed.returncode != 0:
                    diagnostic = (
                        completed.stderr or completed.stdout or "stable-ts failed"
                    )[-800:]
                elif not result_path.is_file():
                    diagnostic = "stable-ts produced no word alignment JSON"
                else:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    audio_hypothesis = str(payload.get("audio_hypothesis") or "").strip()
                    status, diagnostic = apply_alignment_payload(words, payload)
            except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
                diagnostic = f"{type(exc).__name__}: {exc}"
        cache_path.write_text(json.dumps({
            "status": status, "diagnostic": diagnostic,
            "audio_hypothesis": audio_hypothesis,
            "source_words": [asdict(word) for word in words],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return SourceAlignment(
            words, status, fingerprint, f"stable-ts@{STABLE_TS_COMMIT}", model,
            diagnostic=diagnostic, media_sha256=media_sha256,
            audio_hypothesis=audio_hypothesis,
        )


def card_times_from_words(
    parts: list[str], parent: TranscriptCue, source_words: list[SourceWord],
) -> list[tuple[float, float]]:
    """Use aligned word gaps for cuts while retaining the complete parent span."""
    parent_words = [word for word in source_words if word.cue_id == parent.id]
    counts = [len(re.findall(r"\S+", part)) for part in parts]
    if len(parent_words) != sum(counts) or not parent_words:
        return []
    boundaries = [parent.start]
    cursor = 0
    for count in counts[:-1]:
        cursor += count
        left = parent_words[cursor - 1]
        right = parent_words[cursor]
        boundary = (left.end + right.start) / 2
        boundaries.append(round(max(boundaries[-1], boundary), 3))
    boundaries.append(parent.end)
    return list(zip(boundaries, boundaries[1:]))
