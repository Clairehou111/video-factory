from __future__ import annotations

import json
import fcntl
import hashlib
import os
import subprocess
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .collection_publish import create_collection_publish_batch
from .discovery import ChannelRun, DiscoveryChannel, ResourceDiscoveryRun
from .models import InformationRenderProfile, RenderManifest, TopicType, VideoCollectionManifest
from .publish import PublishBatchState, PublishPlatform, PublishTarget, create_publish_batch
from .radar import build_tencent_radar_copy
from .serde import load_collection_manifest, load_manifest


TOPIC_TAGS: dict[TopicType, list[str]] = {
    TopicType.PRACTICE_POST: ["AI", "开发者", "实践"],
    TopicType.GITHUB_PROJECT: ["AI", "GitHub", "开源项目"],
    TopicType.TOOL_SDK_AGENT: ["AI", "开发工具", "Agent"],
    TopicType.MODEL_OR_PRODUCT: ["AI", "大模型", "新产品"],
    TopicType.COMPANY_OR_TEAM: ["AI", "科技公司", "团队动态"],
    TopicType.RESEARCH_OR_BENCHMARK: ["AI", "论文", "Benchmark"],
    TopicType.OFFICIAL_ANNOUNCEMENT: ["AI", "官方公告", "产品动态"],
    TopicType.LINKED_EXTERNAL_SOURCE: ["AI", "技术动态", "开发者"],
    TopicType.EXPERT_TALK: ["AI", "技术分享", "软件工程"],
}


@contextmanager
def pipeline_lock(workspace_root: Path):
    """Prevent two schedulers or manual commands from generating the same run."""
    lock_path = workspace_root / "pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another pipeline run already holds {lock_path}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def is_pipeline_lock_collision(error: BaseException) -> bool:
    return isinstance(error, RuntimeError) and "another pipeline run already holds" in str(error)


@dataclass(frozen=True, slots=True)
class PipelinePublishConfig:
    tencent_account: str = "main"
    # Retained only so existing config files continue to load. New automatic
    # batches never route to Bilibili while that channel is paused.
    bilibili_account: str = "main"
    bilibili_tid: int = 231
    tencent_collection: str = "AI 前沿动态"
    tencent_tags: list[str] = field(default_factory=lambda: ["AI", "开发者", "技术动态"])
    youtube_tencent_tags: list[str] = field(
        default_factory=lambda: ["AI", "开发者", "技术分享", "软件工程"],
    )
    youtube_interview_tencent_tags: list[str] = field(
        default_factory=lambda: ["AI", "科技人物", "对谈高光", "技术洞见"],
    )
    youtube_bilibili_tags: list[str] = field(
        default_factory=lambda: ["AI", "软件工程", "AI编程", "中文字幕"],
    )

    def __post_init__(self) -> None:
        if not self.tencent_account.strip():
            raise ValueError("pipeline tencent_account must not be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelinePublishConfig":
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("unsupported pipeline publish fields: " + ", ".join(sorted(unknown)))
        return cls(**data)

    @classmethod
    def from_path(cls, path: Path | None) -> "PipelinePublishConfig":
        return cls() if path is None else cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(slots=True)
class PreparedPublishBatch:
    channel: str
    candidate_id: str
    manifest_id: str
    batch_id: str = ""
    batch_state: str = ""
    platforms: list[str] = field(default_factory=list)
    created: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiscoveryPublishBridge:
    """Turn generated discovery outputs into reviewable, platform-routed batches.

    The bridge never approves or submits. It preserves the existing human
    approval digest while making discovery -> generation -> publish queue a
    single idempotent operation.
    """

    def __init__(self, workspace: Any, config: PipelinePublishConfig) -> None:
        self.workspace = workspace
        self.config = config

    def prepare(self, run: ResourceDiscoveryRun) -> list[PreparedPublishBatch]:
        prepared: list[PreparedPublishBatch] = []
        for channel_name, entry in run.channels.items():
            selected = entry.selected
            adoption = entry.adoption or {}
            if selected is None or adoption.get("status") != "generated":
                continue
            result = adoption.get("result") if isinstance(adoption.get("result"), dict) else {}
            if selected.channel == DiscoveryChannel.YOUTUBE:
                prepared.append(self._prepare_youtube(channel_name, selected, result))
            else:
                prepared.append(self._prepare_tencent(channel_name, selected, result))
        return prepared

    def _prepare_tencent(self, channel: str, candidate: Any, result: dict[str, Any]) -> PreparedPublishBatch:
        path_value = str(result.get("manifest") or "")
        if not path_value:
            return PreparedPublishBatch(channel, candidate.id, "", reason="generated result has no manifest")
        manifest = load_manifest(Path(path_value))
        publisher = (candidate.publisher or candidate.author or "原始来源").strip()
        if getattr(manifest, "render_profile", "classic") == InformationRenderProfile.RADAR_V2.value:
            title, description = build_tencent_radar_copy(
                manifest, fallback_title=candidate.title, publisher=publisher,
                source_url=candidate.url,
            )
        else:
            title = (manifest.fixed_title or manifest.fixed_hook or candidate.title).strip()[:30]
            description = f"来源：{publisher}｜{candidate.url}"
        topic = manifest.topic_type or candidate.topic_type or TopicType.LINKED_EXTERNAL_SOURCE
        tags = list(dict.fromkeys([*self.config.tencent_tags, *TOPIC_TAGS[topic]]))[:10]
        target = PublishTarget(
            platform=PublishPlatform.TENCENT,
            account_name=self.config.tencent_account,
            title=title,
            description=description,
            tags=tags,
            options={"collection": self.config.tencent_collection},
        )
        batch = create_publish_batch(manifest, [target], self.workspace.root)
        return self._save_or_reuse(channel, candidate.id, manifest, batch, [PublishPlatform.TENCENT.value])

    def _prepare_youtube(self, channel: str, candidate: Any, result: dict[str, Any]) -> PreparedPublishBatch:
        path_value = str(result.get("collection_manifest") or "")
        if not path_value:
            return PreparedPublishBatch(channel, candidate.id, "", reason="generated result has no collection manifest")
        manifest = load_collection_manifest(Path(path_value))
        editorial_mode = str(
            result.get("editorial_mode")
            or candidate.metadata.get("youtube_editorial_mode")
            or ("technical_coverage" if candidate.metadata.get("technical_share") else "")
            or getattr(manifest, "editorial_mode", "")
            or ""
        )
        if editorial_mode not in {"technical_coverage", "known_tech_interview_clip"}:
            return PreparedPublishBatch(
                channel, candidate.id, manifest.id,
                reason="YouTube candidate is neither technical coverage nor a known-tech interview clip",
            )
        description = f"来源：{manifest.source_channel}｜{manifest.source_url}"
        tags = (
            self.config.youtube_interview_tencent_tags
            if editorial_mode == "known_tech_interview_clip"
            else self.config.youtube_tencent_tags
        )
        spec = {
            "collection_title": manifest.collection_title,
            "targets": {
                "tencent": {
                    "account": self.config.tencent_account,
                    "collection": manifest.collection_title,
                    "tags": tags,
                    "description": description,
                },
            },
        }
        batch = create_collection_publish_batch(manifest, spec, self.workspace.root)
        return self._save_or_reuse(
            channel, candidate.id, manifest, batch,
            [PublishPlatform.TENCENT.value],
        )

    def _save_or_reuse(
        self, channel: str, candidate_id: str,
        manifest: RenderManifest | VideoCollectionManifest, batch: Any, platforms: list[str],
    ) -> PreparedPublishBatch:
        existing = self._existing_batch(manifest.id)
        if existing is not None:
            existing_state = str(existing.get("state") or "")
            existing_platforms = self._existing_platforms(existing)
            if set(existing_platforms) == set(platforms) and (
                existing_state != PublishBatchState.BLOCKED.value
                or batch.state == PublishBatchState.BLOCKED
            ):
                return PreparedPublishBatch(
                    channel, candidate_id, manifest.id,
                    batch_id=str(existing["id"]), batch_state=existing_state,
                    platforms=existing_platforms, created=False,
                    reason="existing publish batch reused",
                )
        self.workspace.save_publish_batch(batch)
        return PreparedPublishBatch(
            channel, candidate_id, manifest.id, batch_id=batch.id,
            batch_state=batch.state.value, platforms=platforms, created=True,
            reason="queued for human review" if batch.state == PublishBatchState.READY_FOR_REVIEW
            else "blocked by publish quality checks",
        )

    def _existing_batch(self, manifest_id: str) -> dict[str, Any] | None:
        found: list[dict[str, Any]] = []
        for path in self.workspace.publish_dir.glob("*/batch.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(payload.get("manifest_id") or "") == manifest_id:
                found.append(payload)
        if not found:
            return None
        return max(found, key=lambda item: str(item.get("created_at") or ""))

    @staticmethod
    def _existing_platforms(batch: dict[str, Any]) -> list[str]:
        rows = batch.get("items") if batch.get("batch_type") == "collection" else batch.get("targets")
        return list(dict.fromkeys(
            str(item.get("platform") or "") for item in (rows or []) if isinstance(item, dict)
        ))


@dataclass(frozen=True, slots=True)
class AutomationPolicy:
    """Operator boundary for unattended production.

    The first seven days are a supervised launch period. Automatic repair is
    still bounded by the content/render agents, while publication always
    remains behind the existing digest-backed human approval gate.
    """

    trial_days: int = 7
    human_publish_approval_required: bool = True

    def __post_init__(self) -> None:
        if self.trial_days < 1:
            raise ValueError("automation trial_days must be positive")


class AutomationAuditService:
    """Persist one human-readable operational audit for every pipeline run."""

    def __init__(
        self, workspace: Any, policy: AutomationPolicy | None = None,
        clock: Any | None = None,
    ) -> None:
        self.workspace = workspace
        self.policy = policy or AutomationPolicy()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.root = workspace.root / "automation"
        self.runs_dir = self.root / "runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self, run: ResourceDiscoveryRun, prepared: list[PreparedPublishBatch],
        requested_provider: str, requested_model: str | None,
    ) -> dict[str, Any]:
        now = self.clock().astimezone(UTC)
        state = self._load_or_start_state(now)
        trial_started = _parse_utc(str(state["trial_started_at"]))
        trial_ends = trial_started + timedelta(days=self.policy.trial_days)
        jobs = self._job_results_since(_parse_utc(run.started_at))
        problems, fixes = self._problems_and_fixes(run, jobs)
        usage = self._llm_usage(jobs, _parse_utc(run.started_at))
        discovery_state = self.workspace.load_discovery_state()
        needs_human_candidates = list(discovery_state.get("needs_human_candidates") or [])
        pending = [
            item.to_dict() for item in prepared
            if item.batch_state == PublishBatchState.READY_FOR_REVIEW.value
        ]
        blocked = [item.to_dict() for item in prepared if item.batch_state == PublishBatchState.BLOCKED.value]
        report: dict[str, Any] = {
            "id": run.id,
            "recorded_at": _iso_utc(now),
            "pipeline_status": run.status,
            "trial": {
                "started_at": _iso_utc(trial_started),
                "ends_at": _iso_utc(trial_ends),
                "day": max(1, (now.date() - trial_started.date()).days + 1),
                "active": now < trial_ends,
                "automatic_fixing": "bounded",
                "next_phase": "remote_deployment_ready" if now >= trial_ends else "supervised_local_launch",
            },
            "publication_gate": {
                "requires_human_confirmation": self.policy.human_publish_approval_required,
                "automatic_publish": False,
                "ready_batch_ids": [item["batch_id"] for item in pending],
                "blocked_batch_ids": [item["batch_id"] for item in blocked if item.get("batch_id")],
                "review_commands": [
                    f"video-factory --workspace {self.workspace.root} publish-status {item['batch_id']}"
                    for item in pending
                ],
                "approval_commands": [
                    f"video-factory --workspace {self.workspace.root} publish-approve {item['batch_id']} --actor <reviewer>"
                    for item in pending
                ],
                "publish_commands": [
                    f"video-factory --workspace {self.workspace.root} publish-run {item['batch_id']}"
                    for item in pending
                ],
            },
            "requested_model_route": {
                "provider": requested_provider, "model": requested_model or "capability-qualified cheapest",
            },
            "model_routes": [
                {"job_id": str(job.get("job_id") or ""), **dict(job.get("model_selection") or {})}
                for job in jobs if isinstance(job.get("model_selection"), dict)
            ],
            "llm_usage": usage,
            "problems": problems,
            "automatic_fixes": fixes,
            "needs_human_candidates": needs_human_candidates,
            "publish_queue": [item.to_dict() for item in prepared],
            "human_action_required": bool(
                pending or needs_human_candidates
                or any(item.get("status") == "unresolved" for item in problems)
            ),
            "human_message": self._human_message(
                pending, problems, needs_human_candidates, trial_ends,
            ),
        }
        self._write_report(report)
        self._append_jsonl(self.root / "llm-costs.jsonl", {
            "run_id": run.id, "recorded_at": report["recorded_at"], **usage,
        })
        for problem in problems:
            self._append_jsonl(self.root / "problems.jsonl", {
                "run_id": run.id, "recorded_at": report["recorded_at"], **problem,
            })
        return report

    def load(self, run_id: str | None = None) -> dict[str, Any]:
        path = self.runs_dir / f"{run_id}.json" if run_id else self.root / "latest.json"
        if not path.is_file():
            raise KeyError("no automation audit has been recorded")
        return json.loads(path.read_text(encoding="utf-8"))

    def record_crash(
        self, error: Exception, requested_provider: str, requested_model: str | None,
    ) -> dict[str, Any]:
        """Leave an audit trail even when orchestration fails before discovery can save a run."""
        now = self.clock().astimezone(UTC)
        run = ResourceDiscoveryRun(
            id=f"pipeline-crash-{now.strftime('%Y%m%d%H%M%S')}",
            status="failed", started_at=_iso_utc(now), completed_at=_iso_utc(now),
            channels={
                "pipeline": ChannelRun(
                    DiscoveryChannel.X, "failed",
                    error=f"{type(error).__name__}: {error}",
                ),
            },
        )
        return self.record(run, [], requested_provider, requested_model)

    def notify(
        self, report: dict[str, Any], runner: Any | None = None, force: bool = False,
    ) -> dict[str, Any]:
        """Notify the operator locally and/or by webhook without granting publish authority."""
        signature_payload = {
            "ready_batch_ids": (report.get("publication_gate") or {}).get("ready_batch_ids") or [],
            "problems": [
                {"scope": item.get("scope"), "kind": item.get("kind"), "detail": item.get("detail")}
                for item in (report.get("problems") or []) if item.get("status") == "unresolved"
            ],
            "needs_human_candidates": [
                item.get("candidate_id") for item in (report.get("needs_human_candidates") or [])
            ],
        }
        signature = hashlib.sha256(json.dumps(
            signature_payload, ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest()
        notification_state_path = self.root / "notification-state.json"
        previous: dict[str, Any] = {}
        if notification_state_path.is_file():
            try:
                previous = json.loads(notification_state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        last_sent = _parse_utc_or_none(str(previous.get("sent_at") or ""))
        reminder_hours = max(1, int(os.environ.get("VIDEO_FACTORY_NOTIFICATION_REMINDER_HOURS", "6")))
        duplicate = (
            previous.get("signature") == signature and last_sent is not None
            and self.clock().astimezone(UTC) < last_sent + timedelta(hours=reminder_hours)
        )
        if not force and (not report.get("human_action_required") or duplicate):
            status = {
                "attempted": False, "delivered": False, "channels": [], "error": "",
                "skipped_reason": "no_human_action" if not report.get("human_action_required") else "duplicate_reminder_cooldown",
            }
            report["notification"] = status
            self._write_report(report)
            return status
        execute = runner or subprocess.run
        ready = len((report.get("publication_gate") or {}).get("ready_batch_ids") or [])
        problems = len(report.get("problems") or [])
        message = f"{ready} 个批次待人工审核，{problems} 个问题已记录。"
        channels: list[dict[str, Any]] = []
        if sys.platform == "darwin" or runner is not None:
            command = [
                "/usr/bin/osascript", "-e",
                (
                    f"display notification {json.dumps(message, ensure_ascii=False)} "
                    f"with title {json.dumps('Video Factory 待审核', ensure_ascii=False)}"
                ),
            ]
            try:
                completed = execute(command, capture_output=True, text=True, timeout=20, check=False)
                channels.append({
                    "channel": "macos", "delivered": completed.returncode == 0,
                    "error": "" if completed.returncode == 0 else (completed.stderr or completed.stdout)[-500:],
                })
            except Exception as error:
                channels.append({
                    "channel": "macos", "delivered": False,
                    "error": f"{type(error).__name__}: {error}",
                })
        webhook = os.environ.get("VIDEO_FACTORY_AUDIT_WEBHOOK_URL", "").strip()
        if webhook:
            payload = {
                "event": "video_factory_human_review_required", "run_id": report.get("id"),
                "message": message, "human_message": report.get("human_message"),
                "ready_batch_ids": (report.get("publication_gate") or {}).get("ready_batch_ids") or [],
                "problem_count": problems,
            }
            try:
                request = Request(
                    webhook, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    method="POST", headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=20) as response:
                    delivered = 200 <= int(response.status) < 300
                channels.append({"channel": "webhook", "delivered": delivered, "error": ""})
            except Exception as error:
                channels.append({
                    "channel": "webhook", "delivered": False,
                    "error": f"{type(error).__name__}: {error}"[-500:],
                })
        status = {
            "attempted": bool(channels), "delivered": any(item["delivered"] for item in channels),
            "channels": channels,
            "error": "; ".join(item["error"] for item in channels if item["error"]),
        }
        report["notification"] = status
        if status["delivered"]:
            _atomic_json(notification_state_path, {
                "signature": signature, "sent_at": _iso_utc(self.clock().astimezone(UTC)),
                "run_id": report.get("id"),
            })
        self._write_report(report)
        return status

    def _load_or_start_state(self, now: datetime) -> dict[str, Any]:
        path = self.root / "state.json"
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8"))
        else:
            state = {"trial_started_at": _iso_utc(now)}
        state.update({
            "trial_days": self.policy.trial_days,
            "human_publish_approval_required": self.policy.human_publish_approval_required,
            "updated_at": _iso_utc(now),
        })
        _atomic_json(path, state)
        return state

    def _job_results_since(self, started_at: datetime) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        jobs_dir = self.workspace.root / "jobs"
        if not jobs_dir.is_dir():
            return rows
        for path in jobs_dir.glob("*/result.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                started = _parse_utc(str(payload.get("started_at") or ""))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if started >= started_at - timedelta(seconds=2):
                payload["_result_path"] = str(path)
                rows.append(payload)
        return sorted(rows, key=lambda item: str(item.get("started_at") or ""))

    def _problems_and_fixes(
        self, run: ResourceDiscoveryRun, jobs: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        problems: list[dict[str, Any]] = []
        fixes: list[dict[str, Any]] = []
        for channel, entry in run.channels.items():
            if entry.status == "blocked_invalidated_source":
                candidate = entry.candidates[0] if entry.candidates else None
                title = candidate.title if candidate else "candidate"
                problems.append({
                    "scope": channel, "kind": "unresolved_aggregator_source",
                    "status": "resolved_automatically",
                    "detail": f"{title} was removed from retry because its final URL was not a configured source domain",
                })
                fixes.append({
                    "scope": channel, "kind": "source_gate",
                    "detail": "discarded unresolved aggregator wrapper before another generation attempt",
                })
            if entry.status == "blocked_retry_wait":
                candidate = entry.candidates[0] if entry.candidates else None
                problems.append({
                    "scope": channel, "kind": "generation_retry_cooldown", "status": "unresolved",
                    "detail": (
                        f"{candidate.title if candidate else 'candidate'} will retry after {entry.next_run_at}; "
                        "cooldown prevents repeated LLM spend"
                    ),
                })
            if entry.error:
                problems.append({
                    "scope": channel, "kind": "discovery", "status": "unresolved", "detail": entry.error,
                })
            attempts = list((entry.adoption or {}).get("attempts") or [])
            for attempt in attempts:
                attempt_result = attempt.get("result") if isinstance(attempt.get("result"), dict) else {}
                for repair in attempt_result.get("automatic_repairs") or []:
                    if not isinstance(repair, dict):
                        continue
                    kind = str(repair.get("kind") or "automatic_repair")
                    problems.append({
                        "scope": channel, "kind": kind, "status": "resolved_automatically",
                        "detail": "quality gate detected a repairable output defect",
                    })
                    fixes.append({
                        "scope": channel, "kind": kind,
                        "detail": str(repair.get("outputs") or "repair completed and revalidated"),
                    })
                status = str(attempt.get("status") or "")
                if status not in {"failed", "quality_failed"}:
                    continue
                detail = str(attempt.get("error") or "")
                if not detail:
                    failed = [
                        str(check.get("detail") or check.get("name") or "quality check failed")
                        for check in (attempt.get("result") or {}).get("checks", [])
                        if isinstance(check, dict) and not check.get("passed", False)
                    ]
                    detail = "; ".join(failed) or status
                resolved = any(
                    int(next_attempt.get("attempt") or 0) > int(attempt.get("attempt") or 0)
                    and next_attempt.get("status") == "generated"
                    for next_attempt in attempts
                )
                problems.append({
                    "scope": channel, "kind": status,
                    "status": "resolved_automatically" if resolved else "unresolved", "detail": detail,
                })
                if resolved:
                    fixes.append({
                        "scope": channel, "kind": "pipeline_retry", "detail": "reused archived inputs and regenerated/rerendered successfully",
                    })
        for job in jobs:
            if job.get("status") == "failed":
                detail = str(job.get("error") or "job failed")
                if not any(item["detail"] == detail for item in problems):
                    problems.append({
                        "scope": str(job.get("source_type") or "job"), "kind": "generation",
                        "status": "unresolved", "detail": detail,
                    })
            for row in _trace_rows(job, self.workspace.root):
                step = str(row.get("step") or "")
                if "repair" in step and row.get("status") not in {"failed", "invalid"}:
                    fixes.append({
                        "scope": str(job.get("job_id") or job.get("source_type") or "job"),
                        "kind": step, "detail": str(row.get("reason") or row.get("errors") or "model/deterministic repair applied"),
                    })
        return _unique_dicts(problems), _unique_dicts(fixes)

    def _llm_usage(self, jobs: list[dict[str, Any]], started_at: datetime) -> dict[str, Any]:
        prices = _price_map(jobs)
        calls: list[dict[str, Any]] = []
        for job in jobs:
            for provenance in _provenances({
                "trace": _trace_rows(job, self.workspace.root),
                "stages": job.get("stages") or [],
            }):
                generated = _parse_utc_or_none(str(provenance.get("generated_at") or ""))
                if generated is not None and generated < started_at - timedelta(seconds=2):
                    continue
                usage = provenance.get("usage")
                if not isinstance(usage, dict):
                    continue
                prompt = _int_value(usage, "prompt_tokens", "input_tokens")
                completion = _int_value(usage, "completion_tokens", "output_tokens")
                total = _int_value(usage, "total_tokens") or prompt + completion
                model = str(
                    provenance.get("model") or provenance.get("actual_model")
                    or provenance.get("requested_model") or "unknown"
                )
                actual = _float_value(usage, "cost", "cost_usd")
                estimate = None
                if model in prices:
                    input_price, output_price = prices[model]
                    estimate = prompt * input_price + completion * output_price
                calls.append({
                    "provider": str(provenance.get("provider") or "unknown"), "model": model,
                    "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total,
                    "actual_cost_usd": actual, "estimated_cost_usd": estimate,
                })
        return {
            "calls": len(calls),
            "prompt_tokens": sum(item["prompt_tokens"] for item in calls),
            "completion_tokens": sum(item["completion_tokens"] for item in calls),
            "total_tokens": sum(item["total_tokens"] for item in calls),
            "actual_cost_usd": round(sum(item["actual_cost_usd"] or 0 for item in calls), 8),
            "estimated_cost_usd": round(sum(item["estimated_cost_usd"] or 0 for item in calls), 8),
            "accounted_cost_usd": round(sum(
                item["actual_cost_usd"]
                if item["actual_cost_usd"] is not None else (item["estimated_cost_usd"] or 0)
                for item in calls
            ), 8),
            "unpriced_calls": sum(
                1 for item in calls if item["actual_cost_usd"] is None and item["estimated_cost_usd"] is None
            ),
            "by_model": _usage_by_model(calls),
        }

    def _human_message(
        self, pending: list[dict[str, Any]], problems: list[dict[str, Any]],
        needs_human_candidates: list[dict[str, Any]], trial_ends: datetime,
    ) -> str:
        unresolved = sum(item.get("status") == "unresolved" for item in problems)
        if pending:
            return (
                f"请人工查看 {len(pending)} 个待发布批次；确认内容、声音、字幕、标题和平台后，"
                "先执行 approval_commands，再执行 publish_commands。"
            )
        if unresolved:
            return f"本轮没有可发布批次，仍有 {unresolved} 个未解决问题需要检查。"
        if needs_human_candidates:
            return f"有 {len(needs_human_candidates)} 个候选已耗尽自动修复预算，需要人工决定跳过或调整规则。"
        return f"本轮无需人工发布操作；7 天本地观察期截至 {_iso_utc(trial_ends)}。"

    def _write_report(self, report: dict[str, Any]) -> None:
        _atomic_json(self.runs_dir / f"{report['id']}.json", report)
        _atomic_json(self.root / "latest.json", report)
        markdown = self._markdown(report)
        (self.runs_dir / f"{report['id']}.md").write_text(markdown, encoding="utf-8")
        (self.root / "latest.md").write_text(markdown, encoding="utf-8")

    @staticmethod
    def _markdown(report: dict[str, Any]) -> str:
        gate = report["publication_gate"]
        usage = report["llm_usage"]
        lines = [
            f"# Video Factory run {report['id']}", "",
            f"- Status: {report['pipeline_status']}",
            f"- Trial day: {report['trial']['day']} (active: {report['trial']['active']})",
            f"- Ready for human review: {len(gate['ready_batch_ids'])}",
            f"- LLM tokens: {usage['total_tokens']}",
            f"- LLM accounted cost: ${usage['accounted_cost_usd']:.6f}",
            f"- Unpriced LLM calls: {usage['unpriced_calls']}", "",
            "## Human action", "", report["human_message"], "",
        ]
        for heading, key in (("Review", "review_commands"), ("Approve", "approval_commands"), ("Publish", "publish_commands")):
            commands = gate[key]
            if commands:
                lines.extend([f"### {heading}", "", *[f"- `{item}`" for item in commands], ""])
        lines.extend(["## Problems", ""])
        lines.extend(
            f"- [{item['status']}] {item['scope']}: {item['detail']}" for item in report["problems"]
        )
        if not report["problems"]:
            lines.append("- None")
        lines.extend(["", "## Automatic fixes", ""])
        lines.extend(
            f"- {item['scope']}: {item['kind']} — {item['detail']}" for item in report["automatic_fixes"]
        )
        if not report["automatic_fixes"]:
            lines.append("- None")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _trace_rows(job: dict[str, Any], workspace_root: Path) -> list[dict[str, Any]]:
    rows = list(job.get("translation_trace") or [])
    manifest_path = job.get("manifest")
    if manifest_path:
        try:
            manifest = load_manifest(Path(str(manifest_path)))
            for check in manifest.quality_checks:
                if check.get("name") == "content_agent" and isinstance(check.get("detail"), dict):
                    rows.extend(check["detail"].get("trace") or [])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    for key in ("content_agent_error",):
        path_value = job.get(key)
        if path_value:
            try:
                payload = json.loads(Path(str(path_value)).read_text(encoding="utf-8"))
                rows.extend(payload.get("trace") or [])
            except (OSError, TypeError, json.JSONDecodeError):
                pass
    return [item for item in rows if isinstance(item, dict)]


def _provenances(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        usage = value.get("usage")
        if isinstance(usage, dict) and (value.get("provider") or value.get("model")):
            yield value
        for key, item in value.items():
            if key != "usage":
                yield from _provenances(item)
    elif isinstance(value, list):
        for item in value:
            yield from _provenances(item)


def _price_map(jobs: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    prices: dict[str, tuple[float, float]] = {}
    for quote in _dicts_with_keys(jobs, {"model_id", "prompt_price", "completion_price"}):
        try:
            prices[str(quote["model_id"])] = (float(quote["prompt_price"]), float(quote["completion_price"]))
        except (TypeError, ValueError):
            continue
    return prices


def _dicts_with_keys(value: Any, keys: set[str]) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if keys <= set(value):
            yield value
        for item in value.values():
            yield from _dicts_with_keys(item, keys)
    elif isinstance(value, list):
        for item in value:
            yield from _dicts_with_keys(item, keys)


def _usage_by_model(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for call in calls:
        key = (call["provider"], call["model"])
        row = grouped.setdefault(key, {
            "provider": key[0], "model": key[1], "calls": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "actual_cost_usd": 0.0, "estimated_cost_usd": 0.0,
        })
        row["calls"] += 1
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            row[name] += call[name]
        row["actual_cost_usd"] += call["actual_cost_usd"] or 0
        row["estimated_cost_usd"] += call["estimated_cost_usd"] or 0
    for row in grouped.values():
        row["actual_cost_usd"] = round(row["actual_cost_usd"], 8)
        row["estimated_cost_usd"] = round(row["estimated_cost_usd"], 8)
    return sorted(grouped.values(), key=lambda item: (item["provider"], item["model"]))


def _int_value(values: dict[str, Any], *names: str) -> int:
    for name in names:
        value = values.get(name)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _float_value(values: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = values.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _parse_utc_or_none(value: str) -> datetime | None:
    try:
        return _parse_utc(value)
    except ValueError:
        return None


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _unique_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            result.append(row)
            seen.add(key)
    return result
