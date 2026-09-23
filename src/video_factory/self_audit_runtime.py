from __future__ import annotations

"""Runtime adapters for the bounded, asynchronous self-audit core."""

import json
import os
import re
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

from .llm import OpenAICompatibleStoryWriter


PRODUCT_POSITION = (
    "This product is a BGM-only visual technology-intelligence radar for Chinese developers, "
    "vibe coders, and technology-curious viewers, including people interested in robotics and "
    "autonomous driving. A short item should make the event and its consequence understandable "
    "in roughly 10–15 seconds, use real source evidence as the visual proof, and leave the source "
    "link in the platform caption. It is not a narrated lesson."
)


class OpenRouterGeminiAuditModel:
    """Ask a pinned Gemini model to diagnose one archived problem cluster."""

    def __init__(
        self, writer: OpenAICompatibleStoryWriter, repo_root: Path,
        estimated_cost_usd: float | None = None,
    ) -> None:
        self.writer = writer
        self.repo_root = repo_root.resolve()
        configured = estimated_cost_usd
        if configured is None:
            try:
                configured = float(os.environ.get("VIDEO_FACTORY_AUDIT_ESTIMATED_COST_USD", "0.15"))
            except ValueError:
                configured = 0.15
        self.estimated_cost_usd = max(0.01, min(float(configured), 1.0))

    def estimate_cost(self, problem: Mapping[str, Any]) -> float:
        return self.estimated_cost_usd

    def propose(
        self, problem: Mapping[str, Any], archive: Mapping[str, Any],
        budget_remaining_usd: float,
    ) -> Mapping[str, Any]:
        stage = str(problem.get("stage") or "pipeline")
        source_context = self._source_context(stage)
        discovery_problem = "discovery" in stage.casefold()
        schema = {
            "kind": "diagnosis|policy|code",
            "rationale": "specific root cause and why the current agent produced it",
            "policy": {
                "runtime": {
                    "narrative_guidance": "bounded guidance applied to later generations",
                    "evaluator_thresholds": {
                        "story_axis": 0.8, "grounding": 1.0,
                        "blinded_confidence": 0.65,
                    },
                },
            },
            "diff": "unified git diff including source fix and regression test, or empty",
            "regression_ref": "tests/test_file.py::TestClass::test_method, required for policy/code",
            "verification": {
                "reproduction_fixed": True, "grounding": True, "safety": True,
                "golden_regressions": True, "confidence": 0.0,
            },
        }
        prompt = "\n\n".join([
            PRODUCT_POSITION,
            "Criticize this recorded factory problem and identify the smallest durable root-cause fix. "
            "This is an asynchronous repair pass, not a request to rewrite one finished video. "
            "Prefer a bounded runtime policy when the defect is editorial; use a code patch only when "
            "the behavior cannot be corrected by the allowed policy. "
            + (
                "This is a discovery-stage problem: audit only the archived scheduler/source/query/funnel trace. "
                "You may propose a code-and-test repair to source coverage, query recall, filtering, scoring, or trace logic, "
                "but do not perform a fresh network search inside the audit and do not change login, publishing, credentials, or budgets. "
                if discovery_problem else
                "Never propose discovery, login, publishing, network, credential, or budget changes. "
            )
            + "Do not hide a problem with a one-off replacement sentence.",
            "Problem:\n" + json.dumps(dict(problem), ensure_ascii=False),
            "Archived reproduction (external actions are forbidden):\n"
            + json.dumps(dict(archive), ensure_ascii=False),
            "Relevant repository context:\n" + source_context,
            f"Remaining nightly budget: ${budget_remaining_usd:.4f}.",
            "Return one JSON object matching this shape. For diagnosis, omit policy/diff. For policy, "
            "use only runtime.narrative_guidance and runtime.evaluator_thresholds plus an optional "
            "director.narrative_rules list. For code, return an exact unified diff that also adds a "
            "unittest regression.\n" + json.dumps(schema, ensure_ascii=False),
        ])
        transport = getattr(self.writer, "transport", None)
        with transport.stage("nightly_self_audit") if transport is not None else nullcontext():
            result, provenance = self.writer._request_json([
                {"role": "system", "content": "Return one valid JSON object without markdown fences."},
                {"role": "user", "content": prompt},
            ], max_tokens=7_000)
        usage = provenance.get("usage") if isinstance(provenance.get("usage"), Mapping) else {}
        actual_cost = 0.0
        if isinstance(usage, Mapping):
            try:
                actual_cost = float(usage.get("cost") or usage.get("cost_usd") or 0.0)
            except (TypeError, ValueError):
                actual_cost = 0.0
        result["cost_usd"] = actual_cost or self.estimated_cost_usd
        result["provenance"] = provenance
        return result

    def _source_context(self, stage: str) -> str:
        normalized = stage.casefold()
        groups = {
            "discovery": ("discovery.py",),
            "publish": ("publish.py", "automation.py"),
            "render": ("compositor.py", "webcapture.py", "quality.py"),
            "composition": ("compositor.py", "quality.py"),
            "story": ("agent.py", "llm.py", "writer.py", "editorial.py", "director.py"),
            "generation": ("agent.py", "llm.py", "writer.py", "editorial.py", "factory.py"),
            "pipeline": ("automation.py", "factory.py"),
        }
        names = next((value for key, value in groups.items() if key in normalized), groups["pipeline"])
        parts: list[str] = []
        remaining = 60_000
        for name in names:
            path = self.repo_root / "src" / "video_factory" / name
            if not path.is_file() or remaining <= 0:
                continue
            source = path.read_text(encoding="utf-8")[:remaining]
            parts.append(f"--- {path.relative_to(self.repo_root)} ---\n{source}")
            remaining -= len(source)
        return "\n".join(parts)


class RepositoryCandidateVerifier:
    """Run fixed local gates; never execute commands supplied by the model."""

    def __init__(self, repo_root: Path, timeout_seconds: int = 240) -> None:
        self.repo_root = repo_root.resolve()
        self.timeout_seconds = timeout_seconds
        self._test_result: subprocess.CompletedProcess[str] | None = None

    def verify(
        self, problem: Mapping[str, Any], proposal: Mapping[str, Any],
        archive: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._test_result is None:
            self._test_result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                cwd=self.repo_root,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"},
                capture_output=True, text=True, timeout=self.timeout_seconds, check=False,
            )
        completed = self._test_result
        supplied = proposal.get("verification")
        evaluation = dict(supplied) if isinstance(supplied, Mapping) else {}
        confidence = _float(evaluation.get("confidence"))
        return {
            "reproduction_fixed": bool(evaluation.get("reproduction_fixed")),
            "deterministic_gates": completed.returncode == 0,
            "grounding": bool(evaluation.get("grounding")),
            "safety": bool(evaluation.get("safety")),
            "golden_regressions": bool(evaluation.get("golden_regressions")),
            "confidence": confidence,
            "test_returncode": completed.returncode,
            "test_output": (completed.stdout + completed.stderr)[-4_000:],
        }


class ReviewBranchExecutor:
    """Apply a model diff only inside an isolated local review worktree."""

    TEST_REF = re.compile(
        r"^(tests/test_[^:]+\.py)::([A-Za-z_]\w*)::(test_[A-Za-z_]\w*)$",
    )

    def __init__(self, repo_root: Path, workspace_root: Path, timeout_seconds: int = 300) -> None:
        self.repo_root = repo_root.resolve()
        self.workspace_root = workspace_root.resolve()
        self.timeout_seconds = timeout_seconds

    def stage(self, plan: Mapping[str, Any]) -> Mapping[str, Any]:
        branch = str(plan["branch"])
        problem_id = str(plan["problem_id"])
        worktree = self.workspace_root / "automation" / "self-audit" / "worktrees" / problem_id
        if worktree.exists():
            raise RuntimeError(f"review worktree already exists: {worktree}")
        worktree.parent.mkdir(parents=True, exist_ok=True)
        self._run(["git", "worktree", "add", "-b", branch, str(worktree), "HEAD"], self.repo_root)
        try:
            diff = str(plan["diff"])
            self._run(["git", "apply", "--check", "-"], worktree, input_text=diff)
            self._run(["git", "apply", "-"], worktree, input_text=diff)
            module = self._unittest_name(str(plan["regression_ref"]))
            env = {
                **os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(worktree / "src"),
            }
            self._run([sys.executable, "-m", "unittest", module], worktree, env=env)
            self._run(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                worktree, env=env,
            )
            self._run(["git", "diff", "--check"], worktree)
            paths = re.findall(r"^\+\+\+ b/(.+)$", diff, flags=re.MULTILINE)
            self._run(["git", "add", "--", *paths], worktree)
            self._run([
                "git", "-c", "user.name=Video Factory Agent",
                "-c", "user.email=video-factory-agent@localhost",
                "commit", "-m", f"fix: {problem_id}",
            ], worktree)
            commit = self._run(["git", "rev-parse", "HEAD"], worktree).stdout.strip()
            return {
                "worktree": str(worktree), "commit": commit,
                "tests_passed": True, "merge": False, "push": False,
            }
        except Exception:
            # Keep the failed worktree and diff for diagnosis; never mutate the
            # caller's worktree or delete evidence automatically.
            raise

    @classmethod
    def _unittest_name(cls, reference: str) -> str:
        match = cls.TEST_REF.fullmatch(reference)
        if not match:
            raise ValueError("invalid executable regression reference")
        file_name, class_name, method_name = match.groups()
        module = file_name.removesuffix(".py").replace("/", ".")
        return f"{module}.{class_name}.{method_name}"

    def _run(
        self, command: list[str], cwd: Path, *, input_text: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            command, cwd=cwd, env=dict(env) if env is not None else None,
            input=input_text, capture_output=True, text=True,
            timeout=self.timeout_seconds, check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"fixed review command failed ({command[0]} {command[1]}): "
                + (completed.stderr or completed.stdout)[-2_000:]
            )
        return completed


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "OpenRouterGeminiAuditModel", "PRODUCT_POSITION",
    "RepositoryCandidateVerifier", "ReviewBranchExecutor",
]
