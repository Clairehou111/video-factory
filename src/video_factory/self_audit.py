from __future__ import annotations

"""Bounded, asynchronous self-improvement for the video factory.

The module deliberately has no network, subprocess, publishing, or discovery
dependencies.  Callers inject a model and (optionally) a verifier and Phoenix
sink.  This keeps an audit replayable from archived workspace material and
makes the safety boundary straightforward to test.
"""

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


LIFECYCLE_STATES = {
    "queued", "diagnosed", "testing", "fixed", "backed_off", "human_blocked",
}
SEVERITY_WEIGHT = {"low": 1, "medium": 2, "high": 3, "critical": 4}
BACKOFF_DAYS = (1, 3, 7)
ALLOWED_POLICY_FIELDS: dict[str, set[str]] = {
    "runtime": {"narrative_guidance", "evaluator_thresholds"},
    "writer": {
        "system_prompt", "style_prompt", "hook_rules", "narrative_rules",
        "max_headline_chars", "max_screen_chars",
    },
    "editorial": {
        "critic_prompt", "quality_rules", "min_evidence_count", "thresholds",
    },
    "director": {
        "prompt", "narrative_rules", "pacing", "clip_duration_min",
        "clip_duration_max",
    },
    "compositor": {
        "layout_profile", "typography", "translation_alignment",
    },
    "generation": {"max_repair_attempts", "story_policy"},
}
ALLOWED_EVALUATOR_THRESHOLDS = {
    "story_axis", "audience_recognition", "event_chain_clarity",
    "evidence_coverage", "readability", "grounding", "artifact_consistency",
    "blinded_confidence",
}
FORBIDDEN_POLICY_WORDS = {
    "publish", "login", "credential", "secret", "token", "network", "shell",
    "command", "discovery", "approval", "account", "webhook", "budget",
}
ALLOWED_PATCH_ROOTS = {
    "src", "tests", "examples", "deploy", "docs", "regressions.json",
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _workspace_root(root_or_workspace: Path | str | Any) -> Path:
    # pathlib.Path.root is the filesystem anchor ("/"), not a workspace
    # attribute. Only unwrap domain workspace objects.
    value = root_or_workspace if isinstance(root_or_workspace, (Path, str)) else root_or_workspace.root
    return Path(value).resolve()


def _normalized(value: str) -> str:
    # Volatile IDs/timestamps should not split one recurring failure into many
    # clusters. Meaningful numbers (prices, dimensions, metrics) are retained.
    value = value.strip().lower()
    value = re.sub(r"\b[0-9a-f]{12,64}\b", "<id>", value)
    value = re.sub(r"\b20\d\d[-/]\d\d[-/]\d\d[t ][0-9:.+z-]+", "<time>", value)
    value = re.sub(r"\s+", " ", value)
    return value


def problem_fingerprint(stage: str, category: str, expected: str, observed: str) -> str:
    canonical = "\n".join(_normalized(item) for item in (stage, category, expected, observed))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


@dataclass(slots=True)
class ProblemObservation:
    stage: str
    category: str
    expected: str
    observed: str
    severity: str = "medium"
    reporter: str = "agent"
    job_id: str = ""
    manifest_id: str = ""
    artifact_refs: list[str] = field(default_factory=list)
    timecode: str = ""
    hashes: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stage.strip() or not self.category.strip():
            raise ValueError("problem stage and category are required")
        if not self.expected.strip() or not self.observed.strip():
            raise ValueError("problem expected and observed behavior are required")
        if self.severity not in SEVERITY_WEIGHT:
            raise ValueError(f"unsupported problem severity: {self.severity}")


class AuditModel(Protocol):
    """Injected model adapter. It may use OpenRouter; the core never does."""

    def estimate_cost(self, problem: Mapping[str, Any]) -> float: ...

    def propose(
        self, problem: Mapping[str, Any], archive: Mapping[str, Any],
        budget_remaining_usd: float,
    ) -> Mapping[str, Any]: ...


class CandidateVerifier(Protocol):
    def verify(
        self, problem: Mapping[str, Any], proposal: Mapping[str, Any],
        archive: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class CodeRepairExecutor(Protocol):
    """Materialize a validated diff in an isolated review branch/worktree."""

    def stage(self, plan: Mapping[str, Any]) -> Mapping[str, Any]: ...


class PhoenixSink(Protocol):
    """Small optional surface implemented by observability/Phoenix adapters."""

    def record_event(self, name: str, attributes: Mapping[str, Any]) -> None: ...


class ProblemLedger:
    """Append-only observations plus an atomic materialized cluster index."""

    def __init__(self, root_or_workspace: Path | str | Any, clock: Callable[[], datetime] | None = None):
        self.workspace_root = _workspace_root(root_or_workspace)
        self.root = self.workspace_root / "automation" / "self-audit"
        self.observations_path = self.root / "observations.jsonl"
        self.events_path = self.root / "events.jsonl"
        self.index_path = self.root / "problems.json"
        self.clock = clock or _utc_now
        self.root.mkdir(parents=True, exist_ok=True)

    def record(
        self, observation: ProblemObservation | Mapping[str, Any] | None = None, **values: Any,
    ) -> dict[str, Any]:
        if observation is None:
            observation = ProblemObservation(**values)
        elif isinstance(observation, Mapping):
            observation = ProblemObservation(**dict(observation))
        elif values:
            raise ValueError("pass either observation or keyword fields, not both")
        now = self.clock().astimezone(UTC)
        fingerprint = problem_fingerprint(
            observation.stage, observation.category,
            observation.expected, observation.observed,
        )
        row = {
            "fingerprint": fingerprint, "recorded_at": _iso(now), **asdict(observation),
        }
        _append_jsonl(self.observations_path, row)
        problems = self._load_index()
        current = problems.get(fingerprint)
        if current is None:
            current = {
                **row, "id": f"problem-{fingerprint}", "first_seen_at": _iso(now),
                "last_seen_at": _iso(now), "occurrence_count": 1,
                "state": "queued", "failure_count": 0, "next_attempt_at": None,
                "attempts": [], "regression_ref": "",
            }
        else:
            current["last_seen_at"] = _iso(now)
            current["occurrence_count"] = int(current.get("occurrence_count", 0)) + 1
            # Preserve the strongest severity and merge provenance without
            # erasing the representative expected/observed report.
            if SEVERITY_WEIGHT[observation.severity] > SEVERITY_WEIGHT.get(str(current.get("severity")), 0):
                current["severity"] = observation.severity
            current["artifact_refs"] = list(dict.fromkeys([
                *list(current.get("artifact_refs") or []), *observation.artifact_refs,
            ]))
            current["hashes"] = {**dict(current.get("hashes") or {}), **observation.hashes}
            if current.get("state") == "fixed":
                current["state"] = "queued"
                current["next_attempt_at"] = None
        problems[fingerprint] = current
        self._save_index(problems)
        self._event("problem.recorded", {
            "problem_id": current["id"], "fingerprint": fingerprint,
            "occurrence_count": current["occurrence_count"],
        })
        return dict(current)

    def import_legacy(self, path: Path | str | None = None) -> int:
        source = Path(path) if path is not None else self.workspace_root / "automation" / "problems.jsonl"
        if not source.is_file():
            return 0
        import_state_path = self.root / "legacy-imports.json"
        if import_state_path.is_file():
            import_state = json.loads(import_state_path.read_text(encoding="utf-8"))
        else:
            import_state = {"row_hashes": []}
        seen = set(import_state.get("row_hashes") or [])
        imported = 0
        for line in source.read_text(encoding="utf-8").splitlines():
            row_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
            if row_hash in seen:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            detail = str(row.get("detail") or row.get("observed") or "legacy problem")
            self.record(ProblemObservation(
                stage=str(row.get("scope") or row.get("stage") or "pipeline"),
                category=str(row.get("kind") or row.get("category") or "legacy"),
                expected=str(row.get("expected") or "pipeline stage completes without this problem"),
                observed=detail,
                severity=str(row.get("severity") or "medium")
                if str(row.get("severity") or "medium") in SEVERITY_WEIGHT else "medium",
                reporter="legacy-import",
                job_id=str(row.get("job_id") or row.get("run_id") or ""),
                metadata={"legacy_status": str(row.get("status") or "")},
            ))
            imported += 1
            seen.add(row_hash)
        _atomic_json(import_state_path, {
            "source": str(source.resolve()), "row_hashes": sorted(seen),
            "updated_at": _iso(self.clock()),
        })
        self._event("legacy.imported", {"source": str(source), "rows": imported})
        return imported

    def status(self) -> dict[str, Any]:
        rows = list(self._load_index().values())
        counts = {state: 0 for state in sorted(LIFECYCLE_STATES)}
        for row in rows:
            counts[str(row.get("state") or "queued")] = counts.get(str(row.get("state") or "queued"), 0) + 1
        return {
            "root": str(self.root), "total": len(rows), "counts": counts,
            "problems": sorted(rows, key=lambda item: str(item.get("last_seen_at") or ""), reverse=True),
        }

    def get(self, problem_id: str) -> dict[str, Any]:
        _, row = self._find(self._load_index(), problem_id)
        return dict(row)

    def reproduction_bundle(self, problem_id: str) -> dict[str, Any]:
        return self._archived_reproduction(self.get(problem_id))

    def _archived_reproduction(self, problem: Mapping[str, Any]) -> dict[str, Any]:
        assets: list[dict[str, Any]] = []
        for reference in problem.get("artifact_refs") or []:
            if not isinstance(reference, str) or "://" in reference:
                raise ValueError("audit reproduction accepts archived local paths only")
            candidate = Path(reference)
            if not candidate.is_absolute():
                candidate = self.workspace_root / candidate
            resolved = candidate.resolve()
            try:
                resolved.relative_to(self.workspace_root)
            except ValueError as error:
                raise ValueError("audit artifact escapes workspace") from error
            if not resolved.is_file():
                raise FileNotFoundError(reference)
            raw = resolved.read_bytes()
            asset: dict[str, Any] = {
                "path": str(resolved.relative_to(self.workspace_root)), "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            if resolved.suffix.lower() in {".json", ".jsonl", ".txt", ".md", ".srt"}:
                asset["text"] = raw[:250_000].decode("utf-8", errors="replace")
            assets.append(asset)
        return {
            "problem_id": problem["id"], "mode": "archived_only",
            "external_actions_allowed": False, "assets": assets,
            "hashes": dict(problem.get("hashes") or {}),
        }

    def scan_artifact_drift(self) -> int:
        """Record when a saved manifest/video no longer matches its run identity."""
        jobs = self.workspace_root / "jobs"
        if not jobs.is_dir():
            return 0
        recorded = 0
        for result_path in jobs.glob("*/result.json"):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            identity = result.get("artifact_identity")
            if not isinstance(identity, Mapping):
                continue
            for label, result_key, hash_key in (
                ("manifest", "manifest", "manifest_sha256"),
                ("video", "video", "video_sha256"),
            ):
                expected = str(identity.get(hash_key) or "")
                raw_path = result.get(result_key)
                if not expected or not raw_path:
                    continue
                path = Path(str(raw_path))
                if not path.is_absolute():
                    path = self.workspace_root / path
                try:
                    resolved = path.resolve()
                    resolved.relative_to(self.workspace_root)
                except (OSError, ValueError):
                    continue
                if not resolved.is_file():
                    continue
                current = hashlib.sha256(resolved.read_bytes()).hexdigest()
                if current == expected:
                    continue
                self.record(ProblemObservation(
                    stage="artifact", category="artifact_drift",
                    expected=f"saved {label} remains byte-identical to the completed run",
                    observed=f"{label} hash changed after completion",
                    severity="high", reporter="artifact-scanner",
                    job_id=str(result.get("job_id") or result_path.parent.name),
                    artifact_refs=[str(resolved.relative_to(self.workspace_root))],
                    hashes={f"expected_{label}_sha256": expected, f"current_{label}_sha256": current},
                ))
                recorded += 1
        return recorded

    def eligible(self, limit: int, now: datetime | None = None) -> list[dict[str, Any]]:
        current_time = (now or self.clock()).astimezone(UTC)
        eligible: list[dict[str, Any]] = []
        for row in self._load_index().values():
            if row.get("state") in {"fixed", "human_blocked"}:
                continue
            next_attempt = _parse_time(row.get("next_attempt_at"))
            if next_attempt is not None and next_attempt > current_time:
                continue
            eligible.append(dict(row))
        # Severity first, then recurrence, then age (oldest first).
        eligible.sort(key=lambda item: (
            -SEVERITY_WEIGHT.get(str(item.get("severity")), 0),
            -int(item.get("occurrence_count") or 0),
            str(item.get("first_seen_at") or ""),
        ))
        return eligible[:max(0, limit)]

    def transition(
        self, problem_id: str, state: str, *, attempt: Mapping[str, Any] | None = None,
        regression_ref: str = "",
    ) -> dict[str, Any]:
        if state not in LIFECYCLE_STATES:
            raise ValueError(f"unsupported lifecycle state: {state}")
        problems = self._load_index()
        fingerprint, row = self._find(problems, problem_id)
        if state == "fixed" and not (regression_ref or row.get("regression_ref")):
            raise ValueError("fixed problems require an executable regression reference")
        row["state"] = state
        row["updated_at"] = _iso(self.clock())
        if regression_ref:
            row["regression_ref"] = regression_ref
        if attempt is not None:
            row.setdefault("attempts", []).append(dict(attempt))
        problems[fingerprint] = row
        self._save_index(problems)
        self._event("problem.transitioned", {"problem_id": row["id"], "state": state})
        return dict(row)

    def fail_attempt(self, problem_id: str, detail: str, cost_usd: float = 0.0) -> dict[str, Any]:
        problems = self._load_index()
        fingerprint, row = self._find(problems, problem_id)
        failures = int(row.get("failure_count") or 0) + 1
        delay = BACKOFF_DAYS[min(failures - 1, len(BACKOFF_DAYS) - 1)]
        row["failure_count"] = failures
        row["next_attempt_at"] = _iso(self.clock() + timedelta(days=delay))
        row["state"] = "human_blocked" if failures >= 3 else "backed_off"
        row.setdefault("attempts", []).append({
            "at": _iso(self.clock()), "status": "failed", "detail": detail,
            "cost_usd": max(0.0, float(cost_usd)), "backoff_days": delay,
        })
        problems[fingerprint] = row
        self._save_index(problems)
        self._event("problem.attempt_failed", {
            "problem_id": row["id"], "failure_count": failures,
            "state": row["state"], "backoff_days": delay,
        })
        return dict(row)

    def defer(self, problem_id: str, detail: str, days: int = 7) -> dict[str, Any]:
        """Delay a non-failing shadow/diagnostic case without inflating failures."""
        problems = self._load_index()
        fingerprint, row = self._find(problems, problem_id)
        row["next_attempt_at"] = _iso(self.clock() + timedelta(days=max(1, days)))
        row.setdefault("attempts", []).append({
            "at": _iso(self.clock()), "status": "deferred", "detail": detail,
            "backoff_days": max(1, days),
        })
        problems[fingerprint] = row
        self._save_index(problems)
        self._event("problem.deferred", {
            "problem_id": row["id"], "state": row.get("state"), "days": max(1, days),
        })
        return dict(row)

    def _find(self, problems: dict[str, dict[str, Any]], problem_id: str) -> tuple[str, dict[str, Any]]:
        for fingerprint, row in problems.items():
            if row.get("id") == problem_id or fingerprint == problem_id:
                return fingerprint, row
        raise KeyError(problem_id)

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.is_file():
            return {}
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        return dict(payload.get("problems") or {})

    def _save_index(self, problems: Mapping[str, Mapping[str, Any]]) -> None:
        _atomic_json(self.index_path, {
            "schema_version": 1, "updated_at": _iso(self.clock()), "problems": problems,
        })

    def _event(self, name: str, attributes: Mapping[str, Any]) -> None:
        _append_jsonl(self.events_path, {
            "at": _iso(self.clock()), "event": name, **dict(attributes),
        })


class PolicyStore:
    def __init__(self, root: Path, clock: Callable[[], datetime] | None = None):
        self.root = root
        self.versions_dir = root / "policies" / "versions"
        self.active_path = root / "policies" / "active.json"
        self.clock = clock or _utc_now
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, policy: Mapping[str, Any]) -> None:
        if not policy:
            raise ValueError("candidate policy must not be empty")
        unknown_sections = set(policy) - set(ALLOWED_POLICY_FIELDS)
        if unknown_sections:
            raise ValueError("unsafe policy sections: " + ", ".join(sorted(unknown_sections)))
        for section, values in policy.items():
            if not isinstance(values, Mapping):
                raise ValueError(f"policy section {section} must be an object")
            unknown = set(values) - ALLOWED_POLICY_FIELDS[section]
            if unknown:
                raise ValueError(f"unsafe {section} policy fields: " + ", ".join(sorted(unknown)))
            for key in values:
                words = set(re.findall(r"[a-z]+", key.lower()))
                if words & FORBIDDEN_POLICY_WORDS:
                    raise ValueError(f"unsafe policy field: {section}.{key}")
        runtime = policy.get("runtime")
        if not isinstance(runtime, Mapping):
            raise ValueError("candidate policy requires a runtime section")
        guidance = runtime.get("narrative_guidance")
        if not isinstance(guidance, str) or not guidance.strip():
            raise ValueError("runtime.narrative_guidance must be a non-empty string")
        if len(guidance) > 4_000:
            raise ValueError("runtime.narrative_guidance exceeds 4000 characters")
        thresholds = runtime.get("evaluator_thresholds")
        if not isinstance(thresholds, Mapping) or not thresholds:
            raise ValueError("runtime.evaluator_thresholds must be a non-empty object")
        unknown_thresholds = set(thresholds) - ALLOWED_EVALUATOR_THRESHOLDS
        if unknown_thresholds:
            raise ValueError(
                "unsafe evaluator thresholds: " + ", ".join(sorted(unknown_thresholds)),
            )
        for name, value in thresholds.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                raise ValueError(f"evaluator threshold {name} must be between 0 and 1")

    def promote(
        self, policy: Mapping[str, Any], *, problem_id: str,
        verification: Mapping[str, Any], version: str | None = None,
    ) -> dict[str, Any]:
        self.validate(policy)
        required = {"reproduction_fixed", "deterministic_gates", "grounding", "safety", "golden_regressions"}
        if not required.issubset(verification) or not all(bool(verification[key]) for key in required):
            raise ValueError("policy promotion requires all deterministic safety and regression gates")
        if float(verification.get("confidence") or 0.0) < 0.65:
            raise ValueError("policy promotion confidence must be at least 0.65")
        digest = hashlib.sha256(json.dumps(policy, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        identifier = version or f"{self.clock().strftime('%Y%m%d%H%M%S')}-{digest[:10]}"
        if not re.fullmatch(r"[A-Za-z0-9._-]+", identifier):
            raise ValueError("invalid policy version")
        record = {
            "version": identifier, "created_at": _iso(self.clock()), "problem_id": problem_id,
            "digest": digest, "policy": dict(policy), "verification": dict(verification),
        }
        path = self.versions_dir / f"{identifier}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("digest") != digest:
                raise ValueError("policy version already exists with different content")
        else:
            _atomic_json(path, record)
        _atomic_json(self.active_path, {
            "version": identifier, "digest": digest, "activated_at": _iso(self.clock()),
            "previous_version": self.active_version(),
        })
        return record

    def rollback(self, version: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", version):
            raise ValueError("invalid policy version")
        path = self.versions_dir / f"{version}.json"
        if not path.is_file():
            raise KeyError(version)
        record = json.loads(path.read_text(encoding="utf-8"))
        previous = self.active_version()
        _atomic_json(self.active_path, {
            "version": version, "digest": record["digest"],
            "activated_at": _iso(self.clock()), "previous_version": previous,
            "rollback": True,
        })
        return {"version": version, "previous_version": previous, "policy": record["policy"]}

    def active_version(self) -> str | None:
        if not self.active_path.is_file():
            return None
        return str(json.loads(self.active_path.read_text(encoding="utf-8")).get("version") or "") or None


def load_active_policy(root_or_workspace: Path | str | Any) -> dict[str, Any]:
    """Load and integrity-check the small runtime-facing policy projection.

    Generation code receives only narrative guidance and evaluator thresholds,
    never the broader audit record or an unchecked arbitrary configuration.
    """
    workspace_root = _workspace_root(root_or_workspace)
    root = workspace_root / "automation" / "self-audit"
    store = PolicyStore(root)
    version = store.active_version()
    if version is None:
        return {
            "version": None, "digest": "", "narrative_guidance": "",
            "evaluator_thresholds": {},
        }
    pointer = json.loads(store.active_path.read_text(encoding="utf-8"))
    record_path = store.versions_dir / f"{version}.json"
    if not record_path.is_file():
        raise ValueError(f"active policy version is missing: {version}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    policy = record.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("active policy payload is invalid")
    store.validate(policy)
    digest = hashlib.sha256(
        json.dumps(policy, ensure_ascii=False, sort_keys=True).encode("utf-8"),
    ).hexdigest()
    if digest != record.get("digest") or digest != pointer.get("digest"):
        raise ValueError("active policy integrity check failed")
    runtime = policy["runtime"]
    return {
        "version": version, "digest": digest,
        "narrative_guidance": str(runtime["narrative_guidance"]),
        "evaluator_thresholds": dict(runtime["evaluator_thresholds"]),
    }


class SelfAuditService:
    """Select, replay, diagnose, and safely stage bounded repairs."""

    def __init__(
        self, root_or_workspace: Path | str | Any, model: AuditModel,
        verifier: CandidateVerifier | None = None, phoenix: PhoenixSink | None = None,
        code_executor: CodeRepairExecutor | None = None,
        clock: Callable[[], datetime] | None = None, max_issues: int = 5,
        max_cost_usd: float = 1.0, shadow_only: bool = True,
    ) -> None:
        self.workspace_root = _workspace_root(root_or_workspace)
        self.clock = clock or _utc_now
        self.ledger = ProblemLedger(self.workspace_root, self.clock)
        self.policy_store = PolicyStore(self.ledger.root, self.clock)
        self.model = model
        self.verifier = verifier
        self.phoenix = phoenix
        self.code_executor = code_executor
        self.max_issues = max(0, min(5, int(max_issues)))
        self.max_cost_usd = max(0.0, min(1.0, float(max_cost_usd)))
        self.shadow_only = shadow_only

    def run(self) -> dict[str, Any]:
        started = self.clock().astimezone(UTC)
        drift_observations = self.ledger.scan_artifact_drift()
        report: dict[str, Any] = {
            "started_at": _iso(started), "max_issues": self.max_issues,
            "max_cost_usd": self.max_cost_usd, "spent_usd": 0.0,
            "processed": [], "skipped": [], "shadow_only": self.shadow_only,
            "artifact_drift_observations": drift_observations,
        }
        self._phoenix("self_improvement.run.started", report)
        for problem in self.ledger.eligible(self.max_issues, started):
            remaining = self.max_cost_usd - float(report["spent_usd"])
            estimate = max(0.0, float(self.model.estimate_cost(problem)))
            if estimate > remaining:
                report["skipped"].append({"problem_id": problem["id"], "reason": "cost_budget"})
                continue
            try:
                archive = self._archived_reproduction(problem)
                self.ledger.transition(problem["id"], "diagnosed")
                proposal = dict(self.model.propose(problem, archive, remaining))
                cost = max(0.0, float(proposal.get("cost_usd") or estimate))
                if cost > remaining:
                    raise ValueError("model call exceeded remaining audit budget")
                report["spent_usd"] = round(float(report["spent_usd"]) + cost, 8)
                proposal_path = (
                    self.ledger.root / "proposals" / str(problem["id"])
                    / f"{started.strftime('%Y%m%dT%H%M%SZ')}.json"
                )
                _atomic_json(proposal_path, proposal)
                outcome = self._handle_proposal(problem, archive, proposal)
                report["processed"].append({
                    "problem_id": problem["id"], "cost_usd": cost,
                    "proposal_artifact": str(proposal_path.relative_to(self.workspace_root)),
                    "proposal_sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
                    **outcome,
                })
                self._phoenix("self_improvement.problem.processed", report["processed"][-1])
            except Exception as error:
                failed = self.ledger.fail_attempt(problem["id"], f"{type(error).__name__}: {error}")
                report["processed"].append({
                    "problem_id": problem["id"], "status": failed["state"],
                    "error": f"{type(error).__name__}: {error}",
                })
                self._phoenix("self_improvement.problem.failed", report["processed"][-1])
        report["completed_at"] = _iso(self.clock())
        _atomic_json(self.ledger.root / "runs" / f"{started.strftime('%Y%m%dT%H%M%SZ')}.json", report)
        _atomic_json(self.ledger.root / "latest-run.json", report)
        self._phoenix("self_improvement.run.completed", report)
        return report

    def rollback(self, version: str) -> dict[str, Any]:
        result = self.policy_store.rollback(version)
        self._phoenix("self_improvement.policy.rolled_back", result)
        return result

    def _archived_reproduction(self, problem: Mapping[str, Any]) -> dict[str, Any]:
        return self.ledger._archived_reproduction(problem)

    def _handle_proposal(
        self, problem: Mapping[str, Any], archive: Mapping[str, Any], proposal: Mapping[str, Any],
    ) -> dict[str, Any]:
        kind = str(proposal.get("kind") or "diagnosis")
        if kind == "diagnosis":
            self.ledger.defer(str(problem["id"]), "diagnosis recorded; await new evidence or human review")
            return {"status": "diagnosed", "rationale": str(proposal.get("rationale") or "")}
        regression_ref = str(proposal.get("regression_ref") or "")
        if kind == "policy":
            candidate = proposal.get("policy")
            if not isinstance(candidate, Mapping):
                raise ValueError("policy proposal requires a policy object")
            self.policy_store.validate(candidate)
            self.ledger.transition(problem["id"], "testing")
            if self.verifier is None:
                return {"status": "testing", "reason": "no verifier configured"}
            verification = dict(self.verifier.verify(problem, proposal, archive))
            if self.shadow_only:
                self.ledger.defer(
                    str(problem["id"]), "shadow candidate recorded; promotion remains disabled",
                )
                return {"status": "testing", "reason": "shadow_only", "verification": verification}
            record = self.policy_store.promote(
                candidate, problem_id=str(problem["id"]), verification=verification,
                version=str(proposal.get("version") or "") or None,
            )
            self.ledger.transition(problem["id"], "fixed", regression_ref=regression_ref)
            return {"status": "fixed", "policy_version": record["version"]}
        if kind == "code":
            diff = str(proposal.get("diff") or "")
            self._validate_code_diff(diff)
            if not regression_ref:
                raise ValueError("code repair requires an executable regression reference")
            branch = f"agent-fix/{problem['id']}"
            if proposal.get("branch") and proposal.get("branch") != branch:
                raise ValueError("code repair branch must use the isolated agent-fix namespace")
            if self.shadow_only:
                self.ledger.defer(
                    str(problem["id"]), "shadow code candidate validated but not materialized",
                )
                return {
                    "status": "testing", "reason": "shadow_only",
                    "proposed_branch": branch,
                }
            plan = {
                "problem_id": problem["id"], "created_at": _iso(self.clock()),
                "branch": branch, "status": "review_required", "diff": diff,
                "regression_ref": regression_ref, "merge": False, "push": False,
            }
            _atomic_json(self.ledger.root / "code-repairs" / f"{problem['id']}.json", plan)
            if self.code_executor is not None:
                staged = dict(self.code_executor.stage(plan))
                plan.update(staged)
                _atomic_json(
                    self.ledger.root / "code-repairs" / f"{problem['id']}.json", plan,
                )
            self.ledger.transition(problem["id"], "testing", attempt={
                "at": _iso(self.clock()), "status": "review_required", "branch": branch,
            })
            return {
                "status": "review_required", "branch": branch,
                **({"worktree": plan["worktree"], "commit": plan.get("commit", "")}
                   if plan.get("worktree") else {}),
            }
        raise ValueError(f"unsupported repair proposal kind: {kind}")

    @staticmethod
    def _validate_code_diff(diff: str) -> None:
        if not diff.strip() or "diff --git " not in diff:
            raise ValueError("code proposal requires a unified git diff")
        lowered = diff.lower()
        if any(command in lowered for command in ("git push", "git merge", "git checkout", "rm -rf")):
            raise ValueError("code proposal contains a forbidden command")
        paths = re.findall(r"^\+\+\+ b/(.+)$", diff, flags=re.MULTILINE)
        if not paths:
            raise ValueError("code proposal has no target paths")
        for value in paths:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or any(part.startswith(".") for part in path.parts):
                raise ValueError(f"unsafe patch path: {value}")
            if path.parts[0] not in ALLOWED_PATCH_ROOTS:
                raise ValueError(f"patch target is outside the review allowlist: {value}")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".zip", ".sqlite", ".db"}:
                raise ValueError(f"binary patch target is not allowed: {value}")
        if not any(Path(value).parts[0] == "tests" for value in paths):
            raise ValueError("code repair must include a regression test")

    def _phoenix(self, name: str, attributes: Mapping[str, Any]) -> None:
        if self.phoenix is None:
            return
        try:
            self.phoenix.record_event(name, attributes)
        except Exception:
            # Observability is best effort and never blocks production/audit.
            pass


__all__ = [
    "AuditModel", "CandidateVerifier", "CodeRepairExecutor", "PhoenixSink", "PolicyStore",
    "ProblemLedger", "ProblemObservation", "SelfAuditService",
    "load_active_policy", "problem_fingerprint",
]
