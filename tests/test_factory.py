from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from video_factory.agent import ContentAgentError
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
    def test_native_media_master_is_primary_and_does_not_invoke_mpt(self) -> None:
        with TemporaryDirectory() as temp:
            job = Path(temp)
            framed = job / "framed.mp4"
            mastered = job / "native-ffmpeg-master.mp4"
            result: dict[str, object] = {"stages": []}
            with (
                patch("video_factory.factory.MPTSettings.from_environment", return_value=MagicMock()),
                patch(
                    "video_factory.factory.NativeFFmpegAssemblyAdapter.assemble",
                    return_value=mastered,
                ) as native,
                patch("video_factory.factory.MPTAssemblyAdapter.assemble") as mpt,
            ):
                selected = VideoFactory._assemble_master(basic_manifest(), framed, job, result)

            self.assertEqual(selected, mastered)
            native.assert_called_once()
            mpt.assert_not_called()
            self.assertEqual(result["stages"][0]["backend"], "native_ffmpeg")
            self.assertFalse(result["stages"][0]["fallback_used"])

    def test_media_master_records_native_failure_before_mpt_fallback(self) -> None:
        with TemporaryDirectory() as temp:
            job = Path(temp)
            framed = job / "framed.mp4"
            fallback = job / "mpt-master.mp4"
            result: dict[str, object] = {"stages": []}
            with (
                patch("video_factory.factory.MPTSettings.from_environment", return_value=MagicMock()),
                patch(
                    "video_factory.factory.NativeFFmpegAssemblyAdapter.assemble",
                    side_effect=RuntimeError("native encoder failed"),
                ),
                patch(
                    "video_factory.factory.MPTAssemblyAdapter.assemble",
                    return_value=fallback,
                ) as mpt,
            ):
                selected = VideoFactory._assemble_master(basic_manifest(), framed, job, result)

            self.assertEqual(selected, fallback)
            mpt.assert_called_once()
            failure = json.loads((job / "assembly-primary-error.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["fallback"], "money_printer_turbo")
            self.assertEqual([stage["status"] for stage in result["stages"]], ["fallback", "ok"])

    def test_editorial_agent_escalates_after_primary_semantic_repairs_are_exhausted(self) -> None:
        with TemporaryDirectory() as temp:
            factory = VideoFactory(Workspace(Path(temp) / "workspace"))
            job = Path(temp) / "job"
            job.mkdir()
            primary_writer = MagicMock()
            primary_writer.settings.model = "z-ai/glm-5.3-flash"
            primary_reviewer = MagicMock()
            fallback_writer = MagicMock()
            fallback_reviewer = MagicMock()
            primary_agent = MagicMock()
            fallback_agent = MagicMock()
            primary_agent.run.side_effect = ContentAgentError(
                "critic rejected duplicate hook", [{"step": "copy_review", "status": "failed"}],
            )
            expected_run = object()
            fallback_agent.run.return_value = expected_run
            selection: dict[str, object] = {}

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False),
                patch.object(
                    factory, "_editorial_agent", side_effect=[primary_agent, fallback_agent],
                ) as editorial_agent,
                patch("video_factory.factory.LLMSettings.from_environment", return_value=MagicMock()),
                patch("video_factory.factory.OpenAICompatibleStoryWriter", return_value=fallback_writer),
                patch.object(
                    factory, "_copy_reviewer",
                    return_value=(fallback_reviewer, {"provider": "openrouter", "model": "critic"}),
                ),
            ):
                run = factory._run_editorial_agent_with_fallback(
                    object(), primary_writer, primary_reviewer,
                    GenerateOptions(provider="deepseek"), job, selection,
                )

            self.assertIs(run, expected_run)
            self.assertTrue((job / "content-agent-primary-error.json").is_file())
            self.assertEqual(selection["fallback"]["model"], "google/gemini-3.7-flash")
            self.assertIn("semantic-copy repairs", selection["fallback"]["reason"])
            self.assertEqual(editorial_agent.call_args_list[0].kwargs["max_llm_calls"], 6)
            self.assertEqual(editorial_agent.call_args_list[1].kwargs["max_llm_calls"], 14)

    def test_explicit_trigger_uncertainty_requires_adjacent_visible_qualification(self) -> None:
        evidence = [Evidence(
            "e-uncertain", "tweet-1", "https://x.com/example/status/1",
            "Shortly afterward my account was suspended. I don't know yet whether this setup was the trigger.",
            "x:thread_post",
        )]

        direction = VideoFactory._causal_uncertainty_direction(evidence)

        self.assertIn("同一句或同一屏", direction)
        self.assertIn("是否由此触发尚无定论", direction)
        self.assertIn("禁止用‘导致、触发、秒封、随即被封、照做就被封’", direction)

    def test_no_causal_uncertainty_rule_without_explicit_source_boundary(self) -> None:
        evidence = [Evidence(
            "e-causal", "tweet-1", "https://x.com/example/status/1",
            "The vendor confirmed that the policy caused the suspension.", "x:thread_post",
        )]
        self.assertEqual(VideoFactory._causal_uncertainty_direction(evidence), "")

    def test_employee_reply_cannot_be_promoted_to_official_company_response(self) -> None:
        evidence = [Evidence(
            "e-employee", "tweet-1", "https://x.com/employee/status/1",
            "An Anthropic employee replied from a personal account: We are hiring.",
            "x:visual_analysis",
        )]

        direction = VideoFactory._source_identity_direction(evidence)

        self.assertIn("个人回复不等于公司官方账号或公司声明", direction)
        self.assertIn("禁止写‘官方回应、官方表态、第一条官方回应’", direction)

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

    def test_generation_cache_isolated_by_render_profile(self) -> None:
        with TemporaryDirectory() as temp:
            factory = VideoFactory(Workspace(Path(temp) / "workspace"))
            classic = factory._generation_cache_path(
                "https://x.com/example/status/1", GenerateOptions(render=False),
            )
            radar = factory._generation_cache_path(
                "https://x.com/example/status/1",
                GenerateOptions(render=False, render_profile="radar_v2"),
            )
            self.assertNotEqual(classic, radar)

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
