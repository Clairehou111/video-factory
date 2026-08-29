from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .media import probe_video, validate_wechat_mp4
from .models import ContentType, RenderManifest, now_iso
from .quality import CheckResult, is_publishable, validate_manifest


SOCIAL_AUTO_UPLOAD_REPOSITORY = "https://github.com/dreammis/social-auto-upload.git"
SOCIAL_AUTO_UPLOAD_COMMIT = "1c66b7db4b30585bbb40c58eb0aa572ffa3cce97"


class PublicationState(StrEnum):
    DRAFT = "draft"
    READY_FOR_HUMAN_REVIEW = "ready_for_human_review"
    AWAITING_FINAL_PUBLISH_CLICK = "awaiting_final_publish_click"
    BLOCKED = "blocked"


@dataclass(slots=True)
class PublishDraft:
    id: str
    manifest_id: str
    title: str
    description: str
    video_path: str
    state: PublicationState
    final_publish_requires_human: bool = True
    checks: list[dict[str, object]] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CreatorStudioDriver(Protocol):
    """UI boundary with no submit/publish operation by design."""

    def open(self, url: str) -> None: ...
    def upload(self, selector: str, file_path: str) -> None: ...
    def fill(self, selector: str, value: str) -> None: ...
    def wait_until_visible(self, selector: str) -> None: ...


@dataclass(frozen=True, slots=True)
class WeChatVideoAccountSelectors:
    studio_url: str
    upload_input: str
    title_input: str
    description_input: str
    final_publish_button: str


class WeChatVideoAccountUploader:
    """Upload and fill a draft, then intentionally stop before final publication."""

    def prepare_for_human_final_click(
        self, draft: PublishDraft, driver: CreatorStudioDriver, selectors: WeChatVideoAccountSelectors,
    ) -> PublishDraft:
        if draft.state != PublicationState.READY_FOR_HUMAN_REVIEW:
            raise ValueError(f"draft must pass review before upload; current state is {draft.state}")
        if not Path(draft.video_path).is_file():
            raise FileNotFoundError(draft.video_path)
        driver.open(selectors.studio_url)
        driver.upload(selectors.upload_input, draft.video_path)
        driver.fill(selectors.title_input, draft.title)
        driver.fill(selectors.description_input, draft.description)
        driver.wait_until_visible(selectors.final_publish_button)
        draft.state = PublicationState.AWAITING_FINAL_PUBLISH_CLICK
        return draft


def prepare_publish_draft(
    manifest: RenderManifest,
    title: str,
    description: str,
    workspace: Path | None = None,
) -> PublishDraft:
    """Produce an upload-ready review package; deliberately never submits it."""
    checks = validate_manifest(manifest, workspace)
    video_path = Path(manifest.video_path or "")
    if manifest.video_path and workspace is not None and not video_path.is_absolute():
        video_path = workspace / video_path
    has_video = bool(manifest.video_path) and video_path.is_file()
    checks.append(CheckResult("video_file", has_video, "成片文件存在" if has_video else "未找到成片文件"))
    if has_video:
        checks.extend(_final_video_checks(video_path, manifest.content_type))
    ready = is_publishable(checks)
    return PublishDraft(
        id=f"publish-{manifest.id}",
        manifest_id=manifest.id,
        title=title,
        description=description,
        video_path=str(video_path.resolve()) if has_video else str(video_path),
        state=PublicationState.READY_FOR_HUMAN_REVIEW if ready else PublicationState.BLOCKED,
        checks=[check.to_dict() for check in checks],
    )


class PublishPlatform(StrEnum):
    TENCENT = "tencent"
    DOUYIN = "douyin"
    XIAOHONGSHU = "xiaohongshu"
    BILIBILI = "bilibili"


class PublishBatchState(StrEnum):
    BLOCKED = "blocked"
    READY_FOR_REVIEW = "ready_for_review"
    APPROVED = "approved"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class PublishTargetState(StrEnum):
    PENDING = "pending"
    PREFLIGHT_PASSED = "preflight_passed"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    FAILED_PRE_SUBMIT = "failed_pre_submit"
    UNCERTAIN = "uncertain"


PLATFORM_OPTION_NAMES: dict[PublishPlatform, set[str]] = {
    PublishPlatform.TENCENT: {
        "thumbnail", "thumbnail_landscape", "thumbnail_portrait", "collection", "short_title", "category",
    },
    PublishPlatform.DOUYIN: {
        "thumbnail", "thumbnail_landscape", "thumbnail_portrait", "collection", "declaration",
        "product_link", "product_title",
    },
    PublishPlatform.XIAOHONGSHU: {"thumbnail"},
    PublishPlatform.BILIBILI: {"tid", "thumbnail"},
}

FILE_OPTION_NAMES = {"thumbnail", "thumbnail_landscape", "thumbnail_portrait"}


@dataclass(slots=True)
class PublishTarget:
    platform: PublishPlatform
    account_name: str
    title: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    schedule_at: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    state: PublishTargetState = PublishTargetState.PENDING
    attempts: int = 0
    last_error: str | None = None
    submitted_at: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.platform, PublishPlatform):
            self.platform = PublishPlatform(self.platform)
        if not isinstance(self.state, PublishTargetState):
            self.state = PublishTargetState(self.state)
        for name in FILE_OPTION_NAMES & set(self.options):
            if isinstance(self.options[name], Path):
                self.options[name] = str(self.options[name])
        self.validate()

    def validate(self) -> None:
        if not self.account_name.strip():
            raise ValueError(f"{self.platform.value}: account must not be empty")
        if not self.title.strip():
            raise ValueError(f"{self.platform.value}: title must not be empty")
        if self.schedule_at:
            try:
                datetime.strptime(self.schedule_at, "%Y-%m-%d %H:%M")
            except ValueError as error:
                raise ValueError(
                    f"{self.platform.value}: schedule_at must use YYYY-MM-DD HH:MM"
                ) from error
        unknown = set(self.options) - PLATFORM_OPTION_NAMES[self.platform]
        if unknown:
            raise ValueError(f"{self.platform.value}: unsupported options: {', '.join(sorted(unknown))}")
        if self.platform == PublishPlatform.BILIBILI:
            tid = self.options.get("tid")
            if not isinstance(tid, int) or isinstance(tid, bool) or tid <= 0:
                raise ValueError("bilibili: options.tid must be a positive integer")
        for name in FILE_OPTION_NAMES & set(self.options):
            value = self.options[name]
            if value and not Path(str(value)).is_file():
                raise FileNotFoundError(f"{self.platform.value}: {name} file not found: {value}")

    def approval_payload(self) -> dict[str, Any]:
        option_file_sha256 = {
            name: _sha256_file(Path(str(self.options[name])))
            for name in sorted(FILE_OPTION_NAMES & set(self.options))
            if self.options[name]
        }
        return {
            "platform": self.platform.value,
            "account_name": self.account_name,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "schedule_at": self.schedule_at,
            "options": self.options,
            "option_file_sha256": option_file_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PublishTarget:
        return cls(**data)


@dataclass(slots=True)
class PublishBatch:
    id: str
    manifest_id: str
    video_path: str
    video_sha256: str
    targets: list[PublishTarget]
    state: PublishBatchState
    backend_commit: str = SOCIAL_AUTO_UPLOAD_COMMIT
    checks: list[dict[str, Any]] = field(default_factory=list)
    approval_digest: str | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        if not isinstance(self.state, PublishBatchState):
            self.state = PublishBatchState(self.state)
        self.targets = [target if isinstance(target, PublishTarget) else PublishTarget.from_dict(target) for target in self.targets]
        platforms = [target.platform for target in self.targets]
        if not platforms:
            raise ValueError("publish batch must contain at least one target")
        if len(set(platforms)) != len(platforms):
            raise ValueError("publish batch may contain only one target per platform")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PublishBatch:
        return cls(**{**data, "targets": [PublishTarget.from_dict(item) for item in data.get("targets", [])]})

    def approval_payload(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "video_path": str(Path(self.video_path).resolve()),
            "video_sha256": self.video_sha256,
            "backend_commit": self.backend_commit,
            "targets": [target.approval_payload() for target in self.targets],
        }

    def compute_approval_digest(self) -> str:
        encoded = json.dumps(self.approval_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def approve(self, actor: str) -> None:
        if self.state != PublishBatchState.READY_FOR_REVIEW:
            raise ValueError(f"batch must be ready_for_review before approval; current state is {self.state}")
        if not actor.strip():
            raise ValueError("approval actor must not be empty")
        for target in self.targets:
            target.validate()
        self.approval_digest = self.compute_approval_digest()
        self.approved_by = actor.strip()
        self.approved_at = now_iso()
        self.state = PublishBatchState.APPROVED
        self.updated_at = now_iso()

    def verify_approval(self) -> None:
        try:
            for target in self.targets:
                target.validate()
            video_path = Path(self.video_path)
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            current_video_hash = _sha256_file(video_path)
            current_approval_digest = self.compute_approval_digest()
        except Exception:
            self._invalidate_approval()
            raise
        if current_video_hash != self.video_sha256 or self.approval_digest != current_approval_digest:
            self._invalidate_approval()
            raise ValueError("approved publish payload changed; a new human approval is required")

    def _invalidate_approval(self) -> None:
        self.state = PublishBatchState.READY_FOR_REVIEW
        self.approval_digest = None
        self.approved_by = None
        self.approved_at = None
        self.updated_at = now_iso()


@dataclass(frozen=True, slots=True)
class BackendResult:
    command: list[str]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    started: bool = False
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.started and not self.timed_out and self.returncode == 0

    def to_audit_dict(self) -> dict[str, Any]:
        return asdict(self)


class PublisherBackend(Protocol):
    commit: str

    def check_account(self, target: PublishTarget) -> BackendResult: ...
    def submit_video(self, target: PublishTarget, video_path: Path) -> BackendResult: ...


@dataclass(frozen=True, slots=True)
class SocialAutoUploadSettings:
    runtime_home: Path
    repository: str = SOCIAL_AUTO_UPLOAD_REPOSITORY
    commit: str = SOCIAL_AUTO_UPLOAD_COMMIT
    python_version: str = "3.11"
    check_timeout_seconds: int = 120
    submit_timeout_seconds: int = 1800

    @classmethod
    def from_environment(cls) -> SocialAutoUploadSettings:
        configured = os.environ.get("VIDEO_FACTORY_SAU_HOME")
        runtime_home = Path(configured).expanduser() if configured else Path.home() / ".video-factory" / "social-auto-upload"
        return cls(runtime_home=runtime_home.resolve())

    @property
    def source_dir(self) -> Path:
        return self.runtime_home / "source"

    @property
    def venv_dir(self) -> Path:
        return self.runtime_home / "venv"

    @property
    def executable(self) -> Path:
        if os.name == "nt":
            return self.venv_dir / "Scripts" / "sau.exe"
        return self.venv_dir / "bin" / "sau"


class SocialAutoUploadBackend:
    """Pinned, isolated subprocess adapter for social-auto-upload's public CLI."""

    def __init__(self, settings: SocialAutoUploadSettings | None = None) -> None:
        self.settings = settings or SocialAutoUploadSettings.from_environment()
        self.commit = self.settings.commit

    def setup(self) -> dict[str, str]:
        git = shutil.which("git")
        uv = shutil.which("uv")
        if not git or not uv:
            raise RuntimeError("publisher setup requires git and uv")
        self.settings.runtime_home.mkdir(parents=True, exist_ok=True)
        _restrict_directory(self.settings.runtime_home)
        source = self.settings.source_dir
        if not source.exists():
            self._checked([git, "clone", "--filter=blob:none", "--no-checkout", self.settings.repository, str(source)])
        elif not (source / ".git").is_dir():
            raise RuntimeError(f"managed publisher source is not a git checkout: {source}")
        self._checked([git, "-C", str(source), "fetch", "--depth", "1", "origin", self.settings.commit])
        self._checked([git, "-C", str(source), "checkout", "--detach", self.settings.commit])
        compatibility_patches = _apply_upstream_compatibility_patches(source)
        local_chrome = _detect_local_chrome_path()
        config_path = source / "conf.py"
        if not config_path.exists():
            config_path.write_text(
                "from pathlib import Path\n\n"
                "BASE_DIR = Path(__file__).parent.resolve()\n"
                "XHS_SERVER = \"http://127.0.0.1:11901\"\n"
                f"LOCAL_CHROME_PATH = {json.dumps(str(local_chrome) if local_chrome else '')}\n"
                "LOCAL_CHROME_HEADLESS = True\n"
                "DEBUG_MODE = True\n"
                "YT_PROXY = None\n",
                encoding="utf-8",
            )
        python = self.settings.venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            self._checked([uv, "venv", "--python", self.settings.python_version, str(self.settings.venv_dir)])
        self._checked([uv, "pip", "install", "--python", str(python), "-e", str(source)])
        patchright = self.settings.venv_dir / ("Scripts/patchright.exe" if os.name == "nt" else "bin/patchright")
        install_managed_browser = not local_chrome or os.environ.get("VIDEO_FACTORY_INSTALL_MANAGED_CHROMIUM") == "1"
        if install_managed_browser:
            browser_environment = os.environ.copy()
            browser_environment["PLAYWRIGHT_BROWSERS_PATH"] = str(self.settings.runtime_home / "browsers")
            if mirror := os.environ.get("VIDEO_FACTORY_PLAYWRIGHT_DOWNLOAD_HOST"):
                browser_environment["PLAYWRIGHT_DOWNLOAD_HOST"] = mirror
            self._checked([str(patchright), "install", "chromium"], env=browser_environment)
        cookies = source / "cookies"
        cookies.mkdir(exist_ok=True)
        _restrict_directory(cookies)
        metadata = {
            "repository": self.settings.repository,
            "commit": self.settings.commit,
            "python": self.settings.python_version,
            "executable": str(self.settings.executable),
            "local_chrome": str(local_chrome) if local_chrome else "",
            "browser_runtime": str(self.settings.runtime_home / "browsers"),
            "managed_browser_installed": str(install_managed_browser).lower(),
            "compatibility_patches": ",".join(compatibility_patches),
        }
        (self.settings.runtime_home / "installation.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        return metadata

    def login_account(self, platform: PublishPlatform, account_name: str, headless: bool = False) -> BackendResult:
        flags = [] if platform == PublishPlatform.BILIBILI else (["--headless"] if headless else ["--headed"])
        result = self._run([platform.value, "login", "--account", account_name, *flags], timeout=900, interactive=True)
        self._restrict_account_files()
        return result

    def check_account(self, target: PublishTarget) -> BackendResult:
        return self.check_login(target.platform, target.account_name)

    def check_login(self, platform: PublishPlatform, account_name: str) -> BackendResult:
        return self._run(
            [platform.value, "check", "--account", account_name],
            timeout=self.settings.check_timeout_seconds,
        )

    def submit_video(self, target: PublishTarget, video_path: Path) -> BackendResult:
        if target.platform == PublishPlatform.BILIBILI:
            metadata = self.runtime_metadata()
            if metadata.get("biliup_integrity") == "mismatch":
                return BackendResult(
                    ["sau", "bilibili", "upload-video"], None,
                    stderr="biliup binary differs from the recorded runtime lock; run publisher setup and revalidate",
                )
        command = [
            target.platform.value, "upload-video", "--account", target.account_name,
            "--file", str(video_path.resolve()), "--title", target.title,
        ]
        if target.description:
            command.extend(["--desc", target.description])
        if target.tags:
            command.extend(["--tags", ",".join(target.tags)])
        if target.schedule_at:
            command.extend(["--schedule", target.schedule_at])
        option_flags = {
            "thumbnail": "--thumbnail",
            "thumbnail_landscape": "--thumbnail-landscape",
            "thumbnail_portrait": "--thumbnail-portrait",
            "collection": "--collection",
            "short_title": "--short-title",
            "category": "--category",
            "declaration": "--declaration",
            "product_link": "--product-link",
            "product_title": "--product-title",
            "tid": "--tid",
        }
        for name, value in target.options.items():
            if value not in (None, ""):
                command.extend([option_flags[name], str(value)])
        result = self._run(command, timeout=self.settings.submit_timeout_seconds)
        self._restrict_account_files()
        return result

    def submit_collection_video(self, target: PublishTarget, video_path: Path) -> BackendResult:
        """Submit one approved collection item and require a machine-readable remote id.

        The collection workflow deliberately uses a separate CLI contract so
        an older pinned runtime fails before upload instead of succeeding
        without a BVID and tempting the factory to upload the same file again.
        """
        if target.platform != PublishPlatform.BILIBILI:
            return self.submit_video(target, video_path)
        command = [
            target.platform.value, "upload-video", "--account", target.account_name,
            "--file", str(video_path.resolve()), "--title", target.title,
            "--desc", target.description, "--tid", str(target.options["tid"]), "--json",
        ]
        if target.tags:
            command.extend(["--tags", ",".join(target.tags)])
        result = self._run(command, timeout=self.settings.submit_timeout_seconds)
        self._restrict_account_files()
        return result

    def ensure_bilibili_collection(self, account_name: str, title: str) -> BackendResult:
        return self._run(
            ["bilibili", "collection-ensure", "--account", account_name, "--title", title, "--json"],
            timeout=self.settings.check_timeout_seconds,
        )

    def add_bilibili_collection(
        self, account_name: str, collection_id: str, bvid: str, position: int,
    ) -> BackendResult:
        return self._run(
            [
                "bilibili", "collection-add", "--account", account_name,
                "--collection-id", collection_id, "--bvid", bvid,
                "--position", str(position), "--json",
            ],
            timeout=self.settings.check_timeout_seconds,
        )

    def runtime_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {"social_auto_upload_commit": self.commit}
        root = self.settings.runtime_home / "home" / ".social-auto-upload" / "tools" / "biliup"
        binaries = [path for path in root.rglob("biliup*") if path.is_file() and path.name != "version.txt"] if root.exists() else []
        if binaries:
            binary = sorted(binaries)[0]
            metadata["biliup_path"] = str(binary.relative_to(self.settings.runtime_home))
            metadata["biliup_sha256"] = _sha256_file(binary)
            version_file = binary.with_name("version.txt")
            if version_file.is_file():
                metadata["biliup_version"] = version_file.read_text(encoding="utf-8").strip()
            lock_path = self.settings.runtime_home / "biliup-lock.json"
            locked = {
                "biliup_path": metadata["biliup_path"],
                "biliup_sha256": metadata["biliup_sha256"],
                "biliup_version": metadata.get("biliup_version"),
            }
            if lock_path.is_file():
                expected = json.loads(lock_path.read_text(encoding="utf-8"))
                metadata["biliup_integrity"] = "verified" if expected == locked else "mismatch"
            else:
                lock_path.write_text(json.dumps(locked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                metadata["biliup_integrity"] = "locked"
        return metadata

    def _run(self, arguments: list[str], timeout: int, interactive: bool = False) -> BackendResult:
        executable = self.settings.executable
        safe_command = ["sau", *arguments]
        if not executable.is_file():
            return BackendResult(safe_command, None, stderr="publisher backend is not installed; run publisher setup")
        environment = os.environ.copy()
        isolated_home = self.settings.runtime_home / "home"
        isolated_home.mkdir(parents=True, exist_ok=True)
        _restrict_directory(isolated_home)
        environment["HOME"] = str(isolated_home)
        environment["PLAYWRIGHT_BROWSERS_PATH"] = str(self.settings.runtime_home / "browsers")
        if os.name == "nt":
            environment["USERPROFILE"] = str(isolated_home)
        try:
            completed = subprocess.run(
                [str(executable), *arguments], cwd=self.settings.source_dir, env=environment,
                capture_output=not interactive, text=True, timeout=timeout, check=False,
            )
            return BackendResult(
                safe_command,
                completed.returncode,
                _redact_output(completed.stdout or "", self.settings.runtime_home),
                _redact_output(completed.stderr or "", self.settings.runtime_home),
                started=True,
            )
        except subprocess.TimeoutExpired as error:
            return BackendResult(
                safe_command,
                None,
                _redact_output(_timeout_text(error.stdout), self.settings.runtime_home),
                _redact_output(_timeout_text(error.stderr), self.settings.runtime_home),
                started=True,
                timed_out=True,
            )
        except OSError as error:
            return BackendResult(safe_command, None, stderr=_redact_output(str(error), self.settings.runtime_home))

    @staticmethod
    def _checked(command: list[str], env: dict[str, str] | None = None) -> None:
        subprocess.run(command, check=True, env=env)

    def _restrict_account_files(self) -> None:
        cookies = self.settings.source_dir / "cookies"
        if not cookies.exists():
            return
        _restrict_directory(cookies)
        for path in cookies.iterdir():
            if path.is_file():
                try:
                    path.chmod(0o600)
                except OSError:
                    pass


def targets_from_spec(data: dict[str, Any], base_dir: Path | None = None) -> list[PublishTarget]:
    allowed_top_level = {"title", "description", "tags", "schedule_at", "targets"}
    unknown_top_level = set(data) - allowed_top_level
    if unknown_top_level:
        raise ValueError(f"unsupported publish spec fields: {', '.join(sorted(unknown_top_level))}")
    raw_targets = data.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("publish spec targets must be a non-empty list")
    common = {name: data.get(name) for name in ("title", "description", "tags", "schedule_at")}
    result: list[PublishTarget] = []
    for raw in raw_targets:
        if not isinstance(raw, dict):
            raise ValueError("each publish target must be an object")
        allowed_target = {"platform", "account", "title", "description", "tags", "schedule_at", "options"}
        unknown = set(raw) - allowed_target
        if unknown:
            raise ValueError(f"unsupported publish target fields: {', '.join(sorted(unknown))}")
        options = dict(raw.get("options") or {})
        for name in FILE_OPTION_NAMES & set(options):
            if options[name] and base_dir and not Path(str(options[name])).is_absolute():
                options[name] = str((base_dir / str(options[name])).resolve())
        tags = raw.get("tags", common["tags"] if common["tags"] is not None else [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError("publish tags must be a list of strings")
        result.append(PublishTarget(
            platform=PublishPlatform(raw.get("platform")),
            account_name=str(raw.get("account") or ""),
            title=str(raw.get("title", common["title"] or "")),
            description=str(raw.get("description", common["description"] or "")),
            tags=tags,
            schedule_at=raw.get("schedule_at", common["schedule_at"]),
            options=options,
        ))
    return result


def create_publish_batch(
    manifest: RenderManifest,
    targets: list[PublishTarget],
    workspace: Path | None = None,
    backend_commit: str = SOCIAL_AUTO_UPLOAD_COMMIT,
) -> PublishBatch:
    checks = validate_manifest(manifest, workspace)
    video_path = Path(manifest.video_path or "")
    if manifest.video_path and workspace is not None and not video_path.is_absolute():
        video_path = workspace / video_path
    has_video = bool(manifest.video_path) and video_path.is_file()
    checks.append(CheckResult("video_file", has_video, "成片文件存在" if has_video else "未找到成片文件"))
    if has_video:
        checks.extend(_final_video_checks(video_path, manifest.content_type))
    video_hash = _sha256_file(video_path) if has_video else ""
    state = PublishBatchState.READY_FOR_REVIEW if is_publishable(checks) else PublishBatchState.BLOCKED
    return PublishBatch(
        id=f"publish-{manifest.id}-{uuid4().hex[:8]}",
        manifest_id=manifest.id,
        video_path=str(video_path.resolve()) if has_video else str(video_path),
        video_sha256=video_hash,
        targets=targets,
        state=state,
        backend_commit=backend_commit,
        checks=[check.to_dict() for check in checks],
    )


class PublishBatchService:
    def __init__(self, workspace: Any, backend: PublisherBackend) -> None:
        self.workspace = workspace
        self.backend = backend

    def save(self, batch: PublishBatch) -> None:
        self.workspace.save_publish_batch(batch)

    def approve(self, batch: PublishBatch, actor: str) -> PublishBatch:
        batch.approve(actor)
        self.save(batch)
        return batch

    def run(self, batch: PublishBatch) -> PublishBatch:
        if batch.backend_commit != self.backend.commit:
            raise ValueError(
                f"batch requires social-auto-upload {batch.backend_commit}; configured backend is {self.backend.commit}"
            )
        if batch.state not in {PublishBatchState.APPROVED, PublishBatchState.FAILED, PublishBatchState.PARTIAL_SUCCESS}:
            raise ValueError(f"batch is not runnable from state {batch.state}")
        pending = [target for target in batch.targets if target.state == PublishTargetState.PENDING]
        if not pending:
            raise ValueError("batch has no pending targets")
        try:
            batch.verify_approval()
        except Exception:
            self.save(batch)
            raise
        batch.state = PublishBatchState.RUNNING
        batch.updated_at = now_iso()
        self.save(batch)

        preflight_failed = False
        for target in pending:
            try:
                result = self.backend.check_account(target)
            except Exception as error:
                result = BackendResult(
                    ["publisher", target.platform.value, "check"], None,
                    stderr=f"backend preflight error: {_redact_secret_values(str(error))[-800:]}",
                )
            self._record(batch, target, "preflight", result)
            if result.succeeded:
                target.state = PublishTargetState.PREFLIGHT_PASSED
                target.last_error = None
            else:
                target.state = PublishTargetState.FAILED_PRE_SUBMIT
                target.last_error = _result_error(result, "account preflight failed")
                preflight_failed = True
            batch.updated_at = now_iso()
            self.save(batch)

        all_targets_preflighted = all(
            target.state in {PublishTargetState.PREFLIGHT_PASSED, PublishTargetState.SUBMITTED}
            for target in batch.targets
        )
        if preflight_failed or not all_targets_preflighted:
            self._finish(batch)
            return batch

        for target in batch.targets:
            if target.state != PublishTargetState.PREFLIGHT_PASSED:
                continue
            target.state = PublishTargetState.SUBMITTING
            target.attempts += 1
            batch.updated_at = now_iso()
            self.save(batch)
            try:
                result = self.backend.submit_video(target, Path(batch.video_path))
            except Exception as error:
                # The target was durably marked SUBMITTING before the backend call. A backend
                # exception cannot prove that the final platform click did not happen.
                result = BackendResult(
                    ["publisher", target.platform.value, "upload-video"], None,
                    stderr=f"backend submission error: {_redact_secret_values(str(error))[-800:]}", started=True,
                )
            self._record(batch, target, "submit", result)
            if result.succeeded:
                target.state = PublishTargetState.SUBMITTED
                target.submitted_at = now_iso()
                target.last_error = None
            elif result.started:
                target.state = PublishTargetState.UNCERTAIN
                target.last_error = _result_error(result, "submission result is uncertain")
            else:
                target.state = PublishTargetState.FAILED_PRE_SUBMIT
                target.last_error = _result_error(result, "submission did not start")
            batch.updated_at = now_iso()
            self.save(batch)
        self._finish(batch)
        return batch

    def retry(self, batch: PublishBatch, platform: PublishPlatform) -> PublishBatch:
        target = next((item for item in batch.targets if item.platform == platform), None)
        if not target:
            raise KeyError(f"batch has no {platform.value} target")
        if target.state != PublishTargetState.FAILED_PRE_SUBMIT:
            raise ValueError(
                f"only failed_pre_submit targets can be retried; {platform.value} is {target.state.value}"
            )
        target.state = PublishTargetState.PENDING
        target.last_error = None
        batch.updated_at = now_iso()
        self.save(batch)
        return self.run(batch)

    def _record(self, batch: PublishBatch, target: PublishTarget, action: str, result: BackendResult) -> None:
        try:
            runtime_metadata = self.backend.runtime_metadata() if hasattr(self.backend, "runtime_metadata") else {"commit": self.backend.commit}
        except Exception as error:
            runtime_metadata = {
                "commit": self.backend.commit,
                "metadata_error": _redact_secret_values(str(error))[-800:],
            }
        self.workspace.append_publish_attempt(
            batch.id, target.platform.value, action,
            {**result.to_audit_dict(), "runtime": runtime_metadata, "recorded_at": now_iso()},
        )

    def _finish(self, batch: PublishBatch) -> None:
        states = {target.state for target in batch.targets}
        if states == {PublishTargetState.SUBMITTED}:
            batch.state = PublishBatchState.SUCCEEDED
        elif PublishTargetState.SUBMITTED in states:
            batch.state = PublishBatchState.PARTIAL_SUCCESS
        else:
            batch.state = PublishBatchState.FAILED
        batch.updated_at = now_iso()
        self.save(batch)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_video_checks(path: Path, content_type: ContentType) -> list[CheckResult]:
    try:
        probe = probe_video(path)
        raw_checks = validate_wechat_mp4(
            probe, max_duration=15 if content_type == ContentType.FLASH else None,
            require_audio=True,
        )
    except Exception as error:
        detail = _redact_secret_values(f"{type(error).__name__}: {error}")[-800:]
        return [CheckResult("video_probe", False, f"成片媒体探测失败：{detail}")]
    return [
        CheckResult(str(item["name"]), bool(item["passed"]), str(item["detail"]))
        for item in raw_checks
    ]


def _redact_output(value: str, runtime_home: Path) -> str:
    return _redact_secret_values(value.replace(str(runtime_home), "<sau-runtime>"))[-4000:]


def _redact_secret_values(value: str) -> str:
    return re.sub(
        r"(?i)(cookie|token|authorization|password)(\s*[:=]\s*)([^\s,;]+)",
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>",
        value,
    )


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _result_error(result: BackendResult, fallback: str) -> str:
    detail = (result.stderr or result.stdout).strip()
    return detail[-1000:] if detail else fallback


def _restrict_directory(path: Path) -> None:
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _detect_local_chrome_path() -> Path | None:
    configured = os.environ.get("VIDEO_FACTORY_CHROME_PATH")
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"VIDEO_FACTORY_CHROME_PATH does not exist: {candidate}")
        return candidate.resolve()
    candidates = [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    ]
    if os.name == "nt":
        for root_name in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            if root := os.environ.get(root_name):
                candidates.extend([
                    Path(root) / "Google/Chrome/Application/chrome.exe",
                    Path(root) / "Microsoft/Edge/Application/msedge.exe",
                ])
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _apply_upstream_compatibility_patches(source: Path) -> list[str]:
    """Route undeclared legacy Playwright imports through upstream's declared Patchright runtime."""
    changed: list[str] = []
    uploader_root = source / "uploader"
    if not uploader_root.exists():
        return changed
    replacements = {
        "from playwright.async_api": "from patchright.async_api",
        "from playwright.sync_api": "from patchright.sync_api",
    }
    for path in uploader_root.rglob("*.py"):
        original = path.read_text(encoding="utf-8")
        patched = original
        for before, after in replacements.items():
            patched = patched.replace(before, after)
        if patched != original:
            path.write_text(patched, encoding="utf-8")
            changed.append(str(path.relative_to(source)))
    return sorted(changed)
