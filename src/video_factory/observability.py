"""Best-effort observability with a local, redacted source of truth.

Phoenix is deliberately an optional export destination.  Production code can
always create spans through :class:`Observability`; unavailable dependencies,
an offline collector, or an unwritable spool must never fail the operation
being observed.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import threading
import time
import uuid
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping


_REDACTED = "[REDACTED]"
_SENSITIVE_KEY = re.compile(
    r"(?:^|[^a-z0-9])(?:api[-_]?key|authorization|cookie|credential|key|password|secret|session|token)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
_SENSITIVE_TEXT = (
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)([?&](?:api[-_]?key|access[-_]?token|token|key)=)[^&#\s]+"),
    re.compile(r"(?i)\b(sk-[A-Za-z0-9_-]{8,})\b"),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def redact(value: Any) -> Any:
    """Return a JSON-compatible copy with common credentials removed.

    Redaction happens before an event reaches either disk or Phoenix.  Unknown
    objects are represented with ``repr`` so observability never forces callers
    to pre-serialize their attributes.
    """
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED if _SENSITIVE_KEY.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        clean = value
        for pattern in _SENSITIVE_TEXT:
            clean = pattern.sub(lambda match: f"{match.group(1)}{_REDACTED}" if match.lastindex else _REDACTED, clean)
        return clean
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return repr(value)


class EventSpool:
    """Append-only JSONL ledger used even when Phoenix is disabled."""

    def __init__(self, workspace_root: Path):
        self.path = Path(workspace_root) / "observability" / "events.jsonl"
        self._lock = threading.Lock()
        self.last_error: str | None = None

    def append(self, event: Mapping[str, Any]) -> bool:
        try:
            payload = (
                json.dumps(redact(dict(event)), ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
                try:
                    os.write(descriptor, payload)
                finally:
                    os.close(descriptor)
            self.last_error = None
            return True
        except Exception as error:  # Observability must never break production.
            self.last_error = f"{type(error).__name__}: {error}"
            return False


class _PhoenixExporter:
    """Small adapter isolated from the rest of the application.

    Imports and registration are deferred until explicitly enabled.  The
    adapter intentionally uses only the stable OpenTelemetry span methods, so
    Phoenix client upgrades do not leak into generation code.
    """

    def __init__(self, project_name: str, endpoint: str, client_endpoint: str):
        phoenix_otel = importlib.import_module("phoenix.otel")
        phoenix_client = importlib.import_module("phoenix.client")
        provider = phoenix_otel.register(
            project_name=project_name,
            endpoint=endpoint,
            auto_instrument=False,
            batch=True,
        )
        self.tracer = provider.get_tracer(project_name)
        self.trace_api = importlib.import_module("opentelemetry.trace")
        # Keep a client ready for later dataset/experiment annotations.  It is
        # optional at runtime and not used in the synchronous recording path.
        self.client = phoenix_client.Client(base_url=client_endpoint)

    def start_span(self, name: str, attributes: Mapping[str, Any], parent_span: Any = None) -> Any:
        context = self.trace_api.set_span_in_context(parent_span) if parent_span is not None else None
        span = self.tracer.start_span(name, context=context)
        for key, value in attributes.items():
            try:
                span.set_attribute(str(key), value)
            except Exception:
                span.set_attribute(str(key), repr(value))
        return span


_ACTIVE_SPANS: ContextVar[tuple["Span", ...]] = ContextVar(
    "video_factory_active_observability_spans", default=(),
)


class Span:
    """Manual nested span that works with or without OpenTelemetry installed."""

    def __init__(self, recorder: "Observability", name: str, attributes: Mapping[str, Any] | None = None):
        self.recorder = recorder
        self.name = name
        self.attributes: dict[str, Any] = dict(attributes or {})
        self.events: list[dict[str, Any]] = []
        self.span_id = uuid.uuid4().hex
        self.trace_id = self.span_id
        self.parent_span_id: str | None = None
        self.started_at = ""
        self._started_ns = 0
        self._token: Token[tuple[Span, ...]] | None = None
        self._phoenix_span: Any = None
        self._entered = False

    def __enter__(self) -> "Span":
        if self._entered:
            return self
        active = _ACTIVE_SPANS.get()
        parent = next((item for item in reversed(active) if item.recorder is self.recorder), None)
        if parent:
            self.parent_span_id = parent.span_id
            self.trace_id = parent.trace_id
        self.started_at = _utc_now()
        self._started_ns = time.monotonic_ns()
        self._token = _ACTIVE_SPANS.set((*active, self))
        self._entered = True
        self._phoenix_span = self.recorder._start_phoenix_span(
            self.name, self.attributes, parent._phoenix_span if parent is not None else None,
        )
        return self

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[str(key)] = value
        if self._phoenix_span is not None:
            try:
                self._phoenix_span.set_attribute(str(key), redact(value))
            except Exception:
                self.recorder._disable_exporter()

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        event = {"name": name, "at": _utc_now(), "attributes": dict(attributes or {})}
        self.events.append(event)
        if self._phoenix_span is not None:
            try:
                self._phoenix_span.add_event(name, redact(dict(attributes or {})))
            except Exception:
                self.recorder._disable_exporter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if not self._entered:
            return False
        ended_at = _utc_now()
        elapsed_ms = max(0.0, (time.monotonic_ns() - self._started_ns) / 1_000_000)
        status = "error" if exc is not None else "ok"
        if exc is not None:
            self.events.append({
                "name": "exception",
                "at": ended_at,
                "attributes": {"type": type(exc).__name__, "message": str(exc)},
            })
        event = {
            "schema_version": 1,
            "event": "span.completed",
            "recorded_at": ended_at,
            "process_id": os.getpid(),
            "service": self.recorder.service_name,
            "project": self.recorder.project_name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "status": status,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "duration_ms": round(elapsed_ms, 3),
            "attributes": self.attributes,
            "events": self.events,
        }
        self.recorder.record(event)
        if self._phoenix_span is not None:
            try:
                if exc is not None:
                    self._phoenix_span.record_exception(exc)
                self._phoenix_span.set_attribute("video_factory.status", status)
                self._phoenix_span.end()
            except Exception:
                self.recorder._disable_exporter()
        if self._token is not None:
            try:
                _ACTIVE_SPANS.reset(self._token)
            except (ValueError, RuntimeError):
                pass
        self._entered = False
        return False


class Observability:
    """Create local spans and optionally mirror them to a Phoenix collector."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        service_name: str = "video-factory",
        project_name: str = "video-factory",
        phoenix_enabled: bool | None = None,
        phoenix_endpoint: str | None = None,
        phoenix_client_endpoint: str | None = None,
    ):
        self.service_name = service_name
        self.project_name = project_name
        self.spool = EventSpool(Path(workspace_root))
        self._exporter: _PhoenixExporter | None = None
        self.export_error: str | None = None
        enabled = phoenix_enabled if phoenix_enabled is not None else _env_enabled("VIDEO_FACTORY_PHOENIX_ENABLED")
        if enabled:
            try:
                self._exporter = _PhoenixExporter(
                    project_name,
                    phoenix_endpoint or os.environ.get(
                        "VIDEO_FACTORY_PHOENIX_OTLP_ENDPOINT", "http://127.0.0.1:6006/v1/traces",
                    ),
                    phoenix_client_endpoint or os.environ.get(
                        "VIDEO_FACTORY_PHOENIX_CLIENT_ENDPOINT", "http://127.0.0.1:6006",
                    ),
                )
            except Exception as error:
                self.export_error = f"{type(error).__name__}: {error}"

    @property
    def phoenix_available(self) -> bool:
        return self._exporter is not None

    def span(self, name: str, attributes: Mapping[str, Any] | None = None) -> Span:
        return Span(self, name, attributes)

    def record(self, event: Mapping[str, Any]) -> bool:
        """Append an already structured event to the redacted local ledger."""
        return self.spool.append(event)

    def record_event(self, name: str, attributes: Mapping[str, Any]) -> None:
        """Phoenix-sink compatible event API used by the nightly auditor."""
        cleaned = redact(dict(attributes))
        self.record({
            "schema_version": 1, "event": name, "recorded_at": _utc_now(),
            "service": self.service_name, "project": self.project_name,
            "attributes": cleaned,
        })
        span = self._start_phoenix_span(name, cleaned)
        if span is not None:
            try:
                span.set_attribute("video_factory.event", name)
                span.end()
            except Exception:
                self._disable_exporter()

    def _start_phoenix_span(
        self, name: str, attributes: Mapping[str, Any], parent_span: Any = None,
    ) -> Any:
        if self._exporter is None:
            return None
        try:
            return self._exporter.start_span(name, redact(dict(attributes)), parent_span)
        except Exception as error:
            self.export_error = f"{type(error).__name__}: {error}"
            self._exporter = None
            return None

    def _disable_exporter(self) -> None:
        self.export_error = self.export_error or "Phoenix exporter failed while recording"
        self._exporter = None


class NoopObservability:
    """Explicit no-op implementation for callers without a workspace."""

    phoenix_available = False
    export_error = None

    def span(self, name: str, attributes: Mapping[str, Any] | None = None) -> "NoopSpan":
        return NoopSpan()

    def record(self, event: Mapping[str, Any]) -> bool:
        return False

    def record_event(self, name: str, attributes: Mapping[str, Any]) -> None:
        return None


class NoopSpan:
    def __enter__(self) -> "NoopSpan":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        return None


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
