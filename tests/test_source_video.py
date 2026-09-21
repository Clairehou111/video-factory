from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from video_factory.media import probe_video
from video_factory.source_video import (
    download_official_youtube_video, recommend_source_clip_duration,
    render_source_video_clip, select_action_clip,
)


class SourceVideoTest(unittest.TestCase):
    def test_source_clip_duration_varies_with_action_scenario(self) -> None:
        self.assertEqual(
            recommend_source_clip_duration(
                "An industrial whole-body robot picks, carries, and deploys at a customer site",
            ),
            12.0,
        )
        self.assertEqual(
            recommend_source_clip_duration("A tiny robot performs one jump"),
            5.0,
        )
        self.assertEqual(
            recommend_source_clip_duration("成立18个月，向500强工业客户完成首批交付"),
            12.0,
        )
        self.assertEqual(
            recommend_source_clip_duration("A robot is introduced in an official demo"),
            9.0,
        )
    def test_download_falls_back_to_managed_mweb_runtime_after_plain_yt_dlp_failure(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = MagicMock()
            runtime.require_executable.return_value = "/managed/yt-dlp"
            runtime.extractor_arguments.return_value = ["--extractor-args", "youtube:player_client=mweb"]
            calls = []

            # Compute the real cache key instead of coupling the test to a
            # made-up filename.
            import hashlib
            url = "https://www.youtube.com/watch?v=official"
            expected = root / f"official-source-{hashlib.sha256(url.encode()).hexdigest()[:12]}.mp4"

            def corrected_run(command, **kwargs):
                calls.append(command)
                if command[0] == "yt-dlp":
                    raise subprocess.CalledProcessError(1, command)
                expected.write_bytes(b"video")
                return subprocess.CompletedProcess(command, 0)

            with (
                patch("video_factory.source_video.ManagedYouTubeRuntime", return_value=runtime),
                patch("video_factory.source_video.subprocess.run", side_effect=corrected_run),
            ):
                path, route = download_official_youtube_video(url, root)

            self.assertEqual(path, expected)
            self.assertEqual(route, "managed_mweb_po_token_fallback")
            self.assertEqual([command[0] for command in calls], ["yt-dlp", "/managed/yt-dlp"])

    def test_motion_fallback_prefers_sustained_action_over_static_intro(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            static = root / "static.mp4"
            action = root / "action.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=640x360:d=5:r=25",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(static),
            ], check=True)
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=s=640x360:d=7:r=25",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(action),
            ], check=True)
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(static), "-i", str(action),
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
            ], check=True)

            selected = select_action_clip(source, clip_duration=4.0)

            self.assertEqual(selected["method"], "sustained_motion_without_hard_cuts")
            self.assertGreaterEqual(float(selected["start"]), 5.0)

    def test_rendered_source_clip_is_continuous_and_matches_requested_viewport(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            output = root / "clip.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=s=640x360:d=8:r=25",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
            ], check=True)

            render_source_video_clip(
                source, output, start=1.5, duration=5.0, width=540, height=800,
            )
            probe = probe_video(output)

            self.assertEqual((probe.width, probe.height), (540, 800))
            self.assertAlmostEqual(probe.duration, 5.0, places=1)

    def test_default_source_video_clip_uses_wide_evidence_pane(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            output = root / "clip.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=s=640x360:d=1:r=25",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
            ], check=True)

            render_source_video_clip(source, output, start=0, duration=1.0)
            probe = probe_video(output)

            self.assertEqual((probe.width, probe.height), (1384, 1092))

    def test_short_animation_can_loop_for_the_planned_proof_hold(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "animation.mp4"
            output = root / "looped.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=s=320x180:d=0.32:r=25",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
            ], check=True)

            render_source_video_clip(
                source, output, start=0, duration=3.5,
                width=540, height=800, loop=True,
            )

            self.assertAlmostEqual(probe_video(output).duration, 3.5, places=1)


if __name__ == "__main__":
    unittest.main()
