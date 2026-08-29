from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.factory import GenerateOptions, VideoFactory
from video_factory.models import (
    ColdOpenBeat, ContentType, Evidence, MaterialRole, RenderManifest, Scene,
)
from video_factory.storage import Workspace


def basic_manifest() -> RenderManifest:
    evidence = Evidence("e-1", "candidate-1", "https://github.com/acme/demo", "claim", "github:readme")
    return RenderManifest(
        id="render-1", candidate_id="candidate-1", content_type=ContentType.EXPLAINER,
        scenes=[Scene(
            "scene-1", 0, 20, "claim", "claim", [evidence.id],
            MaterialRole.PROOF, "show source",
        )],
        evidence=[evidence], source_urls=[evidence.url], fixed_hook="hook", fixed_footer="conclusion",
    )


class VideoFactoryTest(unittest.TestCase):
    def test_github_generation_cache_is_reused_and_refreshable(self) -> None:
        with TemporaryDirectory() as temp:
            factory = VideoFactory(Workspace(Path(temp) / "workspace"))
            manifest = basic_manifest()

            def generate_github(url, job, options, result):
                path = job / "manifest.json"
                path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
                result["manifest"] = str(path)
                result["checks"] = []
                result["publishable"] = True

            url = "https://github.com/acme/demo"
            with (
                patch.object(factory, "_generate_github", side_effect=generate_github) as generate,
                patch("video_factory.factory.validate_manifest", return_value=[]),
            ):
                first = factory.generate(url, GenerateOptions(render=False))
                second = factory.generate(url, GenerateOptions(render=False))
                refreshed = factory.generate(url, GenerateOptions(render=False, refresh=True))

            self.assertEqual(generate.call_count, 2)
            self.assertEqual(
                next(stage for stage in second["stages"] if stage["name"] == "generation_cache")["status"],
                "hit",
            )
            self.assertEqual(
                next(stage for stage in first["stages"] if stage["name"] == "generation_cache")["status"],
                "stored",
            )
            self.assertEqual(
                next(stage for stage in refreshed["stages"] if stage["name"] == "generation_cache")["status"],
                "stored",
            )

    def test_github_render_applies_license_and_budgets_cold_open(self) -> None:
        with TemporaryDirectory() as temp:
            factory = VideoFactory(Workspace(Path(temp) / "workspace"))
            manifest = basic_manifest()
            manifest.github_brief = object()  # The capture request is intercepted before inspecting the brief.
            manifest.cold_open_beats = [
                ColdOpenBeat("one", "one", "event_hook", 1.1, ["e-1"]),
                ColdOpenBeat("two", "two", "capability_reveal", 1.2, ["e-1"]),
                ColdOpenBeat("three", "three", "editorial_verdict", 1.3, ["e-1"]),
            ]
            requested_durations: list[float] = []

            def stop_after_request(*args):
                requested_durations.append(float(args[-1]))
                raise RuntimeError("capture intercepted")

            bgm = Path("workspace/assets/music/858ccdf31193/better-times-are-coming-mixkit-173.mp3").resolve()
            with (
                patch.dict(os.environ, {"VIDEO_FACTORY_BGM_FILE": str(bgm)}, clear=False),
                patch(
                    "video_factory.factory.WebScrollVideoAdapter.github_story_request",
                    side_effect=stop_after_request,
                ),
                self.assertRaisesRegex(RuntimeError, "capture intercepted"),
            ):
                factory._render_github_manifest(
                    manifest, "https://github.com/acme/demo", Path(temp), {"stages": []},
                )

            self.assertEqual(manifest.music_license_status, "royalty_free_verified")
            self.assertTrue(manifest.license_records)
            self.assertAlmostEqual(requested_durations[0], 16.4)

    def test_archive_asset_hashes_without_reading_entire_file(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = Workspace(root / "workspace")
            workspace.initialize()
            source = root / "large.mp4"
            payload = b"a" * (2 * 1024 * 1024 + 17)
            source.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()

            with patch.object(Path, "read_bytes", side_effect=AssertionError("unbounded read")):
                archived, digest = workspace.archive_asset(source, "youtube-video")

            self.assertEqual(digest, expected)
            self.assertEqual((workspace.root / archived).stat().st_size, len(payload))


if __name__ == "__main__":
    unittest.main()
