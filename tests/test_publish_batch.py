import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.media import VideoProbe
from video_factory.models import Candidate, ContentType, Evidence, MaterialRole, RenderManifest, Scene, SourceType
from video_factory.publish import (
    SOCIAL_AUTO_UPLOAD_COMMIT,
    BackendResult,
    PublishBatchService,
    PublishBatchState,
    PublishPlatform,
    PublishTarget,
    PublishTargetState,
    SocialAutoUploadBackend,
    SocialAutoUploadSettings,
    _apply_upstream_compatibility_patches,
    _redact_output,
    create_publish_batch,
    targets_from_spec,
)
from video_factory.storage import Workspace


class FakeBackend:
    commit = SOCIAL_AUTO_UPLOAD_COMMIT

    def __init__(self) -> None:
        self.invalid_accounts: set[PublishPlatform] = set()
        self.uncertain_submissions: set[PublishPlatform] = set()
        self.crashing_submissions: set[PublishPlatform] = set()
        self.checked: list[PublishPlatform] = []
        self.submitted: list[PublishPlatform] = []

    def check_account(self, target: PublishTarget) -> BackendResult:
        self.checked.append(target.platform)
        if target.platform in self.invalid_accounts:
            return BackendResult(["sau", target.platform.value, "check"], 1, stderr="expired", started=True)
        return BackendResult(["sau", target.platform.value, "check"], 0, stdout="valid", started=True)

    def submit_video(self, target: PublishTarget, video_path: Path) -> BackendResult:
        self.submitted.append(target.platform)
        if target.platform in self.crashing_submissions:
            raise RuntimeError("browser crashed token=secret-value")
        if target.platform in self.uncertain_submissions:
            return BackendResult(
                ["sau", target.platform.value, "upload-video"], 1,
                stderr="browser closed after click", started=True,
            )
        return BackendResult(["sau", target.platform.value, "upload-video"], 0, stdout="submitted", started=True)

    def runtime_metadata(self):
        return {"social_auto_upload_commit": self.commit}


def valid_manifest(video: Path) -> RenderManifest:
    candidate = Candidate("candidate-1", SourceType.TWEET, "https://x.com/example/status/1", "example")
    evidence = Evidence("e-1", candidate.id, candidate.source_url, "A source-backed claim", "tweet")
    return RenderManifest(
        id="render-1",
        candidate_id=candidate.id,
        content_type=ContentType.FLASH,
        source_urls=[candidate.source_url],
        evidence=[evidence],
        fixed_footer="清楚结论",
        scenes=[
            Scene("s-1", 0, 4, "事实", "事实", [evidence.id], MaterialRole.PROOF, "show source"),
            Scene("s-2", 4, 8, "结论", "结论", [evidence.id], MaterialRole.EXPLANATION, "show conclusion"),
        ],
        video_path=str(video),
        music_license_status="verified",
        license_records=[{"track": "test", "license": "original"}],
    )


class PublishBatchTest(unittest.TestCase):
    @staticmethod
    def valid_probe(path: Path) -> VideoProbe:
        return VideoProbe(path, 8.0, 1080, 1920, "h264", "yuv420p", "aac")

    def make_batch(self, temp: str, platforms=None):
        root = Path(temp)
        video = root / "video.mp4"
        video.write_bytes(b"video")
        platforms = platforms or [PublishPlatform.TENCENT, PublishPlatform.DOUYIN]
        targets = [
            PublishTarget(platform, "main", f"{platform.value} title", tags=["测试"], options={"tid": 249} if platform == PublishPlatform.BILIBILI else {})
            for platform in platforms
        ]
        with patch("video_factory.publish.probe_video", side_effect=self.valid_probe):
            batch = create_publish_batch(valid_manifest(video), targets, root / "workspace")
        workspace = Workspace(root / "workspace")
        workspace.initialize()
        workspace.save_publish_batch(batch)
        return workspace, batch

    def test_spec_applies_common_fields_and_resolves_relative_files(self) -> None:
        with TemporaryDirectory() as temp:
            cover = Path(temp) / "cover.png"
            cover.write_bytes(b"image")
            targets = targets_from_spec({
                "title": "共同标题",
                "description": "共同简介",
                "tags": ["AI", "工具"],
                "targets": [
                    {"platform": "tencent", "account": "main", "options": {"thumbnail": "cover.png"}},
                    {"platform": "bilibili", "account": "main", "title": "B站标题", "options": {"tid": 249}},
                ],
            }, Path(temp))
            self.assertEqual(targets[0].title, "共同标题")
            self.assertEqual(targets[0].options["thumbnail"], str(cover.resolve()))
            self.assertEqual(targets[1].title, "B站标题")
            self.assertEqual(targets[1].options["tid"], 249)

    def test_batch_resolves_manifest_video_relative_to_workspace(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            video = workspace / "jobs" / "render-1" / "final.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            manifest = valid_manifest(Path("jobs/render-1/final.mp4"))
            target = PublishTarget(PublishPlatform.TENCENT, "main", "测试标题")

            with patch("video_factory.publish.probe_video", side_effect=self.valid_probe):
                batch = create_publish_batch(manifest, [target], workspace)

            self.assertEqual(batch.state, PublishBatchState.READY_FOR_REVIEW)
            self.assertEqual(batch.video_path, str(video.resolve()))
            self.assertTrue(batch.video_sha256)

    def test_human_safety_review_is_narrow_and_bound_into_approval(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp, [PublishPlatform.TENCENT])
            batch.state = PublishBatchState.BLOCKED
            batch.checks = [{
                "name": "editorial_safety_review", "passed": False,
                "detail": "Sensitive security terms found: jailbreak",
            }]

            batch.record_review_override(
                "editorial_safety_review", "claire", "defensive AI safety analysis only",
            )

            self.assertEqual(batch.state, PublishBatchState.READY_FOR_REVIEW)
            self.assertTrue(batch.checks[0]["passed"])
            self.assertEqual(batch.review_overrides["editorial_safety_review"]["actor"], "claire")
            approval_before = batch.compute_approval_digest()
            batch.review_overrides["editorial_safety_review"]["reason"] = "changed"
            self.assertNotEqual(approval_before, batch.compute_approval_digest())

    def test_batch_blocks_corrupt_or_noncompliant_video(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "video.mp4"
            video.write_bytes(b"not-a-video")
            target = PublishTarget(PublishPlatform.TENCENT, "main", "测试标题")
            with patch("video_factory.publish.probe_video", side_effect=RuntimeError("invalid data")):
                corrupt = create_publish_batch(valid_manifest(video), [target], root)
            self.assertEqual(corrupt.state, PublishBatchState.BLOCKED)
            self.assertFalse(next(check for check in corrupt.checks if check["name"] == "video_probe")["passed"])

            wrong = VideoProbe(video, 16.0, 1920, 1080, "hevc", "yuv444p", None)
            with patch("video_factory.publish.probe_video", return_value=wrong):
                noncompliant = create_publish_batch(valid_manifest(video), [target], root)
            self.assertEqual(noncompliant.state, PublishBatchState.BLOCKED)
            failed = {check["name"] for check in noncompliant.checks if not check["passed"]}
            self.assertTrue({"resolution", "h264", "yuv420p", "aac", "duration"} <= failed)

    def test_batch_approval_is_bound_to_exact_video_and_metadata(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp)
            backend = FakeBackend()
            service = PublishBatchService(workspace, backend)
            service.approve(batch, "editor@example.com")
            batch.targets[0].title = "审批后被修改"
            with self.assertRaisesRegex(ValueError, "new human approval"):
                service.run(batch)
            self.assertEqual(batch.state, PublishBatchState.READY_FOR_REVIEW)
            reloaded = workspace.load_publish_batch(batch.id)
            self.assertIsNone(reloaded.approval_digest)

    def test_batch_approval_is_bound_to_thumbnail_bytes(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp, [PublishPlatform.TENCENT])
            thumbnail = Path(temp) / "cover.png"
            thumbnail.write_bytes(b"approved-cover")
            batch.targets[0].options["thumbnail"] = str(thumbnail)
            service = PublishBatchService(workspace, FakeBackend())
            service.approve(batch, "editor")
            thumbnail.write_bytes(b"replacement-cover")
            with self.assertRaisesRegex(ValueError, "new human approval"):
                service.run(batch)
            self.assertEqual(batch.state, PublishBatchState.READY_FOR_REVIEW)

    def test_approved_batch_preflights_then_submits_sequentially(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp)
            backend = FakeBackend()
            service = PublishBatchService(workspace, backend)
            service.approve(batch, "editor")
            service.run(batch)
            self.assertEqual(batch.state, PublishBatchState.SUCCEEDED)
            self.assertEqual(backend.checked, [PublishPlatform.TENCENT, PublishPlatform.DOUYIN])
            self.assertEqual(backend.submitted, [PublishPlatform.TENCENT, PublishPlatform.DOUYIN])
            self.assertTrue(all(target.state == PublishTargetState.SUBMITTED for target in batch.targets))
            attempts = list((workspace.publish_dir / batch.id / "attempts").glob("*.json"))
            self.assertEqual(len(attempts), 4)

    def test_any_preflight_failure_blocks_all_submissions_and_can_be_retried(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp)
            backend = FakeBackend()
            backend.invalid_accounts.add(PublishPlatform.DOUYIN)
            service = PublishBatchService(workspace, backend)
            service.approve(batch, "editor")
            service.run(batch)
            self.assertEqual(batch.state, PublishBatchState.FAILED)
            self.assertEqual(backend.submitted, [])
            failed = next(target for target in batch.targets if target.platform == PublishPlatform.DOUYIN)
            self.assertEqual(failed.state, PublishTargetState.FAILED_PRE_SUBMIT)

            backend.invalid_accounts.clear()
            service.retry(batch, PublishPlatform.DOUYIN)
            self.assertEqual(batch.state, PublishBatchState.SUCCEEDED)
            self.assertEqual(backend.submitted, [PublishPlatform.TENCENT, PublishPlatform.DOUYIN])

    def test_retry_keeps_global_preflight_barrier_for_other_failures(self) -> None:
        with TemporaryDirectory() as temp:
            platforms = [PublishPlatform.TENCENT, PublishPlatform.DOUYIN, PublishPlatform.XIAOHONGSHU]
            workspace, batch = self.make_batch(temp, platforms)
            backend = FakeBackend()
            backend.invalid_accounts.update({PublishPlatform.DOUYIN, PublishPlatform.XIAOHONGSHU})
            service = PublishBatchService(workspace, backend)
            service.approve(batch, "editor")
            service.run(batch)
            self.assertEqual(backend.submitted, [])

            backend.invalid_accounts.remove(PublishPlatform.DOUYIN)
            service.retry(batch, PublishPlatform.DOUYIN)
            self.assertEqual(backend.submitted, [])
            self.assertEqual(
                next(item for item in batch.targets if item.platform == PublishPlatform.XIAOHONGSHU).state,
                PublishTargetState.FAILED_PRE_SUBMIT,
            )

            backend.invalid_accounts.clear()
            service.retry(batch, PublishPlatform.XIAOHONGSHU)
            self.assertEqual(batch.state, PublishBatchState.SUCCEEDED)
            self.assertEqual(backend.submitted, platforms)

    def test_submit_failure_is_uncertain_and_never_retryable(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp, [PublishPlatform.TENCENT])
            backend = FakeBackend()
            backend.uncertain_submissions.add(PublishPlatform.TENCENT)
            service = PublishBatchService(workspace, backend)
            service.approve(batch, "editor")
            service.run(batch)
            self.assertEqual(batch.targets[0].state, PublishTargetState.UNCERTAIN)
            with self.assertRaisesRegex(ValueError, "only failed_pre_submit"):
                service.retry(batch, PublishPlatform.TENCENT)

    def test_backend_exception_after_submitting_state_is_uncertain_and_redacted(self) -> None:
        with TemporaryDirectory() as temp:
            workspace, batch = self.make_batch(temp, [PublishPlatform.TENCENT])
            backend = FakeBackend()
            backend.crashing_submissions.add(PublishPlatform.TENCENT)
            service = PublishBatchService(workspace, backend)
            service.approve(batch, "editor")
            service.run(batch)
            self.assertEqual(batch.targets[0].state, PublishTargetState.UNCERTAIN)
            self.assertNotIn("secret-value", batch.targets[0].last_error)
            self.assertIn("<redacted>", batch.targets[0].last_error)

    def test_bilibili_requires_a_positive_tid(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            PublishTarget(PublishPlatform.BILIBILI, "main", "title")

    def test_backend_maps_only_known_options_and_uses_isolated_home(self) -> None:
        with TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            settings = SocialAutoUploadSettings(runtime_home=runtime)
            settings.source_dir.mkdir(parents=True)
            settings.executable.parent.mkdir(parents=True)
            settings.executable.write_bytes(b"executable")
            (settings.venv_dir / "bin" / "python").write_bytes(b"python")
            settings.biliup_source_dir.mkdir(parents=True)
            video = Path(temp) / "video.mp4"
            cover = Path(temp) / "cover.png"
            video.write_bytes(b"video")
            cover.write_bytes(b"cover")
            target = PublishTarget(
                PublishPlatform.BILIBILI, "main", "标题", description="简介", tags=["AI"],
                schedule_at="2030-01-01 10:00", options={"tid": 249, "thumbnail": str(cover)},
            )
            completed = subprocess.CompletedProcess([], 0, "submitted", "")
            with patch("video_factory.publish.subprocess.run", return_value=completed) as run:
                result = SocialAutoUploadBackend(settings).submit_video(target, video)
            command = run.call_args.args[0]
            self.assertIn("--tid", command)
            self.assertIn("249", command)
            self.assertIn("--schedule", command)
            self.assertEqual(run.call_args.kwargs["env"]["HOME"], str(runtime / "home"))
            self.assertTrue(result.succeeded)

    def test_setup_reuses_an_existing_partial_virtual_environment(self) -> None:
        with TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            settings = SocialAutoUploadSettings(runtime_home=runtime)
            (settings.source_dir / ".git").mkdir(parents=True)
            python = settings.venv_dir / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python")
            patchright = settings.venv_dir / "bin" / "patchright"
            patchright.write_bytes(b"patchright")
            legacy = settings.source_dir / "uploader" / "legacy.py"
            legacy.parent.mkdir()
            legacy.write_text("from playwright.async_api import async_playwright\n", encoding="utf-8")
            with (
                patch("video_factory.publish.shutil.which", side_effect=lambda command: f"/usr/bin/{command}"),
                patch("video_factory.publish._detect_local_chrome_path", return_value=Path("/Applications/Test Chrome")),
                patch.object(SocialAutoUploadBackend, "_checked") as checked,
            ):
                metadata = SocialAutoUploadBackend(settings).setup()
            commands = [call.args[0] for call in checked.call_args_list]
            self.assertFalse(any(command[1:2] == ["venv"] for command in commands))
            self.assertTrue(any(command[1:3] == ["pip", "install"] for command in commands))
            self.assertFalse(any(command[-1:] == ["playwright==1.52.0"] for command in commands))
            self.assertIn('LOCAL_CHROME_PATH = "/Applications/Test Chrome"', (settings.source_dir / "conf.py").read_text())
            self.assertFalse(any(command[-2:] == ["install", "chromium"] for command in commands))
            self.assertIn("from patchright.async_api", legacy.read_text())
            self.assertEqual(metadata["compatibility_patches"], "uploader/legacy.py")

    def test_publisher_policy_patch_disables_ai_label_and_uses_creator_auth_check(self) -> None:
        with TemporaryDirectory() as temp:
            source = Path(temp)
            tencent = source / "uploader" / "tencent_uploader" / "main.py"
            tencent.parent.mkdir(parents=True)
            tencent.write_text(
                "    async def apply_original_statement(self, page: Page) -> None:\n"
                "        label_text = '含AI生成内容'\n",
                encoding="utf-8",
            )
            cli = source / "sau_cli.py"
            cli.write_text(
                'result = run_biliup_command(["-u", str(account_file), "renew"])\n',
                encoding="utf-8",
            )

            changed = _apply_upstream_compatibility_patches(source)

            self.assertEqual(changed, ["sau_cli.py", "uploader/tencent_uploader/main.py"])
            self.assertIn("return None", tencent.read_text(encoding="utf-8"))
            self.assertIn('"list", "--max-pages", "1"', cli.read_text(encoding="utf-8"))

    def test_setup_installs_managed_chromium_when_local_chrome_is_unavailable(self) -> None:
        with TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            settings = SocialAutoUploadSettings(runtime_home=runtime)
            (settings.source_dir / ".git").mkdir(parents=True)
            python = settings.venv_dir / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python")
            patchright = settings.venv_dir / "bin" / "patchright"
            patchright.write_bytes(b"patchright")
            with (
                patch("video_factory.publish.shutil.which", side_effect=lambda command: f"/usr/bin/{command}"),
                patch("video_factory.publish._detect_local_chrome_path", return_value=None),
                patch.dict("video_factory.publish.os.environ", {"VIDEO_FACTORY_PLAYWRIGHT_DOWNLOAD_HOST": "https://mirror.example"}, clear=False),
                patch.object(SocialAutoUploadBackend, "_checked") as checked,
            ):
                SocialAutoUploadBackend(settings).setup()
            browser_install = next(call for call in checked.call_args_list if call.args[0][-2:] == ["install", "chromium"])
            self.assertEqual(browser_install.kwargs["env"]["PLAYWRIGHT_DOWNLOAD_HOST"], "https://mirror.example")

    def test_bilibili_login_does_not_receive_unsupported_browser_flags(self) -> None:
        with TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            settings = SocialAutoUploadSettings(runtime_home=runtime)
            settings.source_dir.mkdir(parents=True)
            settings.executable.parent.mkdir(parents=True)
            settings.executable.write_bytes(b"executable")
            (settings.venv_dir / "bin" / "python").write_bytes(b"python")
            settings.biliup_source_dir.mkdir(parents=True)
            completed = subprocess.CompletedProcess([], 0, "", "")
            with patch("video_factory.publish.subprocess.run", return_value=completed) as run:
                SocialAutoUploadBackend(settings).login_account(PublishPlatform.BILIBILI, "main")
            command = run.call_args.args[0]
            self.assertNotIn("--headed", command)
            self.assertNotIn("--headless", command)

    def test_bilibili_account_check_does_not_require_publish_tid(self) -> None:
        with TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            settings = SocialAutoUploadSettings(runtime_home=runtime)
            settings.source_dir.mkdir(parents=True)
            settings.executable.parent.mkdir(parents=True)
            settings.executable.write_bytes(b"executable")
            (settings.venv_dir / "bin" / "python").write_bytes(b"python")
            settings.biliup_source_dir.mkdir(parents=True)
            completed = subprocess.CompletedProcess([], 0, "valid", "")
            with patch("video_factory.publish.subprocess.run", return_value=completed) as run:
                result = SocialAutoUploadBackend(settings).check_login(PublishPlatform.BILIBILI, "main")
            self.assertTrue(result.succeeded)
            self.assertEqual(run.call_args.args[0][-1], "check")

    def test_bilibili_direct_runtime_must_be_installed_before_submission(self) -> None:
        with TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime"
            settings = SocialAutoUploadSettings(runtime_home=runtime)
            settings.source_dir.mkdir(parents=True)
            settings.executable.parent.mkdir(parents=True)
            settings.executable.write_bytes(b"executable")
            backend = SocialAutoUploadBackend(settings)
            video = Path(temp) / "video.mp4"
            video.write_bytes(b"video")
            target = PublishTarget(PublishPlatform.BILIBILI, "main", "标题", options={"tid": 249})
            with patch("video_factory.publish.subprocess.run") as run:
                result = backend.submit_video(target, video)
            self.assertFalse(result.started)
            self.assertIn("direct Bilibili publisher is not installed", result.stderr)
            run.assert_not_called()

    def test_audit_output_redacts_runtime_paths_and_secret_values(self) -> None:
        runtime = Path("/private/runtime")
        raw = "/private/runtime/source/cookies/main.json cookie=abc token:xyz authorization=BearerSecret"
        redacted = _redact_output(raw, runtime)
        self.assertNotIn("/private/runtime", redacted)
        self.assertNotIn("abc", redacted)
        self.assertNotIn("xyz", redacted)
        self.assertNotIn("BearerSecret", redacted)
        self.assertIn("<redacted>", redacted)


if __name__ == "__main__":
    unittest.main()
