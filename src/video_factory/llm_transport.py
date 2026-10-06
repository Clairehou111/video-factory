"""One auditable transport boundary for every billable LLM request.

Workflow-level call counters are useful for explaining an agent trace, but
they cannot see provider retries or multiple models hidden behind a composite
reviewer.  This module counts the HTTP attempts themselves and writes the
provider-reported usage/cost before control returns to the workflow.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar
from urllib.request import Request

from .observability import NoopObservability, Observability, redact


T = TypeVar("T")


def _run_with_deadline(operation: Callable[[], T], timeout: float) -> T:
    """Return an operation's result within one wall-clock deadline.

    ``urllib`` applies ``timeout`` to individual socket operations.  A server
    can therefore keep a chunked response alive forever by sending a small
    amount of data before each socket timeout.  Run the complete open/read
    operation in a daemon thread so the calling pipeline still has a hard
    wall-clock bound.  The worker owns and closes its response context; a
    timed-out worker cannot hold up interpreter shutdown.
    """
    deadline = max(0.001, float(timeout))
    outcome: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            outcome.put((True, operation()))
        except BaseException as error:
            outcome.put((False, error))

    worker = threading.Thread(
        target=invoke, name="video-factory-llm-request", daemon=True,
    )
    worker.start()
    worker.join(deadline)
    if worker.is_alive():
        raise TimeoutError(
            f"LLM request exceeded the {deadline:g}s total response deadline"
        )
    succeeded, value = outcome.get_nowait()
    if not succeeded:
        assert isinstance(value, BaseException)
        raise value
    return value  # type: ignore[return-value]


class LLMBudgetExceeded(RuntimeError):
    """Raised before another HTTP request can exceed a job's configured guard."""


@dataclass(slots=True)
class _BudgetState:
    job_id: str
    candidate_id: str
    max_requests: int
    max_cost_usd: float
    max_openrouter_semantic_reviews: int | None
    requests: int = 0
    cost_usd: float = 0.0
    stage_requests: dict[str, int] = field(default_factory=dict)


_ACTIVE_BUDGET: ContextVar[_BudgetState | None] = ContextVar(
    "video_factory_llm_budget", default=None,
)
_ACTIVE_STAGE: ContextVar[str] = ContextVar("video_factory_llm_stage", default="unspecified")


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _number(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value or "0"))
    except ValueError:
        return 0.0


class LLMTransport:
    """Execute and meter JSON LLM requests without retaining prompt contents."""

    def __init__(
        self, workspace_root: Path | None = None, observability: Observability | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root) if workspace_root is not None else None
        self.observability = observability or (
            Observability(self.workspace_root) if self.workspace_root is not None else NoopObservability()
        )
        self.ledger_path = (
            self.workspace_root / "observability" / "llm-calls.jsonl"
            if self.workspace_root is not None else None
        )
        self._lock = threading.Lock()

    @contextmanager
    def scope(
        self, *, job_id: str, candidate_id: str = "", max_requests: int,
        max_cost_usd: float, max_openrouter_semantic_reviews: int | None = None,
    ) -> Iterator[_BudgetState]:
        state = _BudgetState(
            job_id=job_id, candidate_id=candidate_id, max_requests=max_requests,
            max_cost_usd=max_cost_usd,
            max_openrouter_semantic_reviews=max_openrouter_semantic_reviews,
        )
        token: Token[_BudgetState | None] = _ACTIVE_BUDGET.set(state)
        try:
            yield state
        finally:
            _ACTIVE_BUDGET.reset(token)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        token = _ACTIVE_STAGE.set(name)
        try:
            yield
        finally:
            _ACTIVE_STAGE.reset(token)

    def request_json(
        self, request: Request, *, timeout: float, provider: str, requested_model: str,
        opener: Callable[..., Any], validator: Callable[[dict[str, object]], T] | None = None,
    ) -> tuple[dict[str, object], T | None]:
        """Run one physical POST and record it even when validation fails."""
        stage = _ACTIVE_STAGE.get()
        budget = _ACTIVE_BUDGET.get()
        self._reserve(budget, provider, stage)
        call_id = uuid.uuid4().hex
        started = time.monotonic_ns()
        result: dict[str, object] | None = None
        error: BaseException | None = None
        validated: T | None = None
        try:
            def open_and_read() -> bytes:
                with opener(request, timeout=timeout) as response:
                    return response.read()

            decoded = json.loads(_run_with_deadline(open_and_read, timeout).decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("LLM response body is not a JSON object")
            result = decoded
            if validator is not None:
                validated = validator(result)
            return result, validated
        except BaseException as caught:
            error = caught
            raise
        finally:
            usage = dict(result.get("usage") or {}) if isinstance(result, dict) else {}
            cost = _number(usage.get("cost")) if usage.get("cost") is not None else None
            if budget is not None:
                budget.cost_usd += max(0.0, cost or 0.0)
            record = {
                "schema_version": 1,
                "event": "llm.request",
                "recorded_at": _now(),
                "call_id": call_id,
                "job_id": budget.job_id if budget else "",
                "candidate_id": budget.candidate_id if budget else "",
                "stage": stage,
                "provider": provider,
                "requested_model": requested_model,
                "actual_model": (
                    str(result.get("model") or requested_model) if result else requested_model
                ),
                "status": "error" if error is not None else "ok",
                "error_type": type(error).__name__ if error is not None else "",
                "error": str(error)[:800] if error is not None else "",
                "latency_ms": round((time.monotonic_ns() - started) / 1_000_000, 3),
                "prompt_sha256": hashlib.sha256(request.data or b"").hexdigest(),
                "usage": usage,
                "cost_usd": cost,
                "budget": {
                    "request_number": budget.requests if budget else None,
                    "max_requests": budget.max_requests if budget else None,
                    "cost_usd_after": round(budget.cost_usd, 9) if budget else None,
                    "max_cost_usd": budget.max_cost_usd if budget else None,
                },
            }
            self._append(record)
            self.observability.record_event("llm.request", record)

    def _reserve(self, budget: _BudgetState | None, provider: str, stage: str) -> None:
        if budget is None:
            return
        if budget.requests >= budget.max_requests:
            raise LLMBudgetExceeded(
                f"LLM request budget exhausted ({budget.requests}/{budget.max_requests})"
            )
        if budget.cost_usd >= budget.max_cost_usd:
            raise LLMBudgetExceeded(
                f"LLM cost budget exhausted (${budget.cost_usd:.4f}/${budget.max_cost_usd:.4f})"
            )
        review_key = "openrouter:semantic_review"
        if provider == "openrouter" and stage == "semantic_review":
            used = budget.stage_requests.get(review_key, 0)
            limit = budget.max_openrouter_semantic_reviews
            if limit is not None and used >= limit:
                raise LLMBudgetExceeded(
                    f"OpenRouter semantic-review request budget exhausted ({used}/{limit})"
                )
            budget.stage_requests[review_key] = used + 1
        budget.requests += 1
        budget.stage_requests[stage] = budget.stage_requests.get(stage, 0) + 1

    def _append(self, record: dict[str, object]) -> None:
        if self.ledger_path is None:
            return
        try:
            payload = (
                json.dumps(redact(record), ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            with self._lock:
                self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(
                    self.ledger_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600,
                )
                try:
                    os.write(descriptor, payload)
                finally:
                    os.close(descriptor)
        except Exception:
            # Cost telemetry must never turn a successful provider response
            # into a failed production job. Phoenix/events.jsonl is a second
            # best-effort sink when this dedicated ledger is unavailable.
            return
