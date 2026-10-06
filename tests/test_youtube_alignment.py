import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from video_factory.models import TranscriptCue
from video_factory.youtube_alignment import (
    AudioWordAligner, apply_alignment_payload, card_times_from_words,
    source_words_from_cues,
)
from video_factory.youtube_runtime import ManagedYouTubeRuntime, YouTubeRuntimeSettings


class YouTubeAlignmentTest(unittest.TestCase):
    def test_source_words_have_stable_ids_and_exact_text(self) -> None:
        cues = [TranscriptCue("cue-1", 2, 6, "Hello, retrieval agent.")]
        words = source_words_from_cues(cues)
        self.assertEqual([word.id for word in words], ["cue-1:w1", "cue-1:w2", "cue-1:w3"])
        self.assertEqual(" ".join(word.raw for word in words), cues[0].source_text)
        self.assertEqual([word.normalized for word in words], ["hello", "retrieval", "agent"])

    def test_source_words_prefer_authoritative_json3_token_times(self) -> None:
        cue = TranscriptCue(
            "cue-1", 2, 6, "one two three", source_tokens=[
                {"raw": "one", "start": 2.1, "end": 2.4},
                {"raw": "two", "start": 3.0, "end": 3.2},
                {"raw": "three", "start": 5.1, "end": 5.6},
            ],
        )

        words = source_words_from_cues([cue])

        self.assertEqual(
            [(word.youtube_start, word.youtube_end) for word in words],
            [(2.1, 2.4), (3.0, 3.2), (5.1, 5.6)],
        )
        self.assertTrue(all(word.alignment_status == "youtube_token" for word in words))

    def test_invalid_json3_token_ledger_falls_back_for_whole_cue(self) -> None:
        cue = TranscriptCue(
            "cue-1", 2, 5, "one two three", source_tokens=[
                {"raw": "one", "start": 2.1, "end": 2.4},
                {"raw": "wrong", "start": 3.0, "end": 3.2},
                {"raw": "three", "start": 4.1, "end": 4.6},
            ],
        )

        words = source_words_from_cues([cue])

        self.assertEqual(
            [(word.youtube_start, word.youtube_end) for word in words],
            [(2.0, 3.0), (3.0, 4.0), (4.0, 5.0)],
        )
        self.assertTrue(all(word.alignment_status == "youtube" for word in words))

    def test_exact_stable_ts_payload_attaches_audio_times(self) -> None:
        words = source_words_from_cues([TranscriptCue("cue-1", 0, 3, "one two three")])
        payload = {"segments": [{"words": [
            {"word": " one", "start": 0.2, "end": 0.7, "probability": 0.9},
            {"word": " two", "start": 0.8, "end": 1.3, "probability": 0.4},
            {"word": " three", "start": 1.4, "end": 2.0, "probability": 0.8},
        ]}]}
        status, diagnostic = apply_alignment_payload(words, payload)
        self.assertEqual(status, "aligned")
        self.assertEqual(diagnostic, "")
        self.assertEqual(words[0].start, 0.2)
        self.assertEqual(words[1].alignment_status, "low_confidence")

    def test_changed_asr_text_cannot_overwrite_youtube_source(self) -> None:
        words = source_words_from_cues([TranscriptCue("cue-1", 0, 2, "YouTube wording")])
        status, diagnostic = apply_alignment_payload(words, {"words": [
            {"word": "different", "start": 0.1, "end": 0.8},
            {"word": "wording", "start": 0.9, "end": 1.5},
        ]})
        self.assertEqual(status, "audio_alignment_degraded")
        self.assertIn("differed", diagnostic)
        self.assertEqual(" ".join(word.raw for word in words), "YouTube wording")
        self.assertTrue(all(word.aligned_start is None for word in words))

    def test_one_merged_audio_token_is_interpolated_without_changing_source(self) -> None:
        words = source_words_from_cues([
            TranscriptCue("cue-1", 0, 3, "we can not ship"),
        ])
        status, diagnostic = apply_alignment_payload(words, {"words": [
            {"word": "we", "start": 0.1, "end": 0.4, "probability": 0.9},
            {"word": "cannot", "start": 0.5, "end": 1.2, "probability": 0.7},
            {"word": "ship", "start": 1.3, "end": 1.8, "probability": 0.9},
        ]})
        self.assertEqual(status, "aligned")
        self.assertEqual(diagnostic, "")
        self.assertEqual(" ".join(word.raw for word in words), "we can not ship")
        self.assertEqual(
            [word.alignment_status for word in words],
            ["aligned", "low_confidence", "low_confidence", "aligned"],
        )

    def test_non_monotonic_audio_times_degrade_to_youtube_timing(self) -> None:
        words = source_words_from_cues([
            TranscriptCue("cue-1", 0, 3, "one two three"),
        ])

        status, diagnostic = apply_alignment_payload(words, {"words": [
            {"word": "one", "start": 0.2, "end": 0.7},
            {"word": "two", "start": 1.2, "end": 1.6},
            {"word": "three", "start": 0.8, "end": 1.1},
        ]})

        self.assertEqual(status, "audio_alignment_degraded")
        self.assertIn("non-monotonic", diagnostic)
        self.assertTrue(all(word.aligned_start is None for word in words))

    def test_card_boundary_uses_midpoint_between_aligned_words(self) -> None:
        cue = TranscriptCue("cue-1", 0, 4, "one two three four")
        words = source_words_from_cues([cue])
        for word, start, end in zip(words, (0.1, 0.8, 2.2, 3.0), (0.5, 1.2, 2.6, 3.5)):
            word.aligned_start = start
            word.aligned_end = end
            word.alignment_status = "aligned"
        self.assertEqual(
            card_times_from_words(["one two", "three four"], cue, words),
            [(0, 1.7), (1.7, 4)],
        )

    def test_missing_alignment_runtime_degrades_and_caches(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            media = root / "clip.mp4"
            media.write_bytes(b"media")
            job = root / "job"
            runtime = ManagedYouTubeRuntime(YouTubeRuntimeSettings(root / "missing-runtime"))
            aligner = AudioWordAligner(runtime=runtime)
            first = aligner.align(media, [TranscriptCue("cue-1", 0, 2, "one two")], job)
            second = aligner.align(media, [TranscriptCue("cue-1", 0, 2, "one two")], job)
            self.assertEqual(first.status, "audio_alignment_degraded")
            self.assertIn("no Python", first.diagnostic)
            self.assertTrue(second.cache_hit)
            cache_files = list((job / "audio-alignment").glob("*.json"))
            self.assertEqual(len(cache_files), 1)
            self.assertEqual(json.loads(cache_files[0].read_text())["status"], first.status)


if __name__ == "__main__":
    unittest.main()
