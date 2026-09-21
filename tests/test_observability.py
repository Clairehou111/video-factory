from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.observability import NoopObservability, Observability, redact


class ObservabilityTests(unittest.TestCase):
    def test_audit_event_sink_records_redacted_local_event(self) -> None:
        with TemporaryDirectory() as temp:
            observer = Observability(Path(temp), phoenix_enabled=False)
            observer.record_event("self_improvement.problem.processed", {
                "problem_id": "p1", "api_key": "secret",
            })
            event = json.loads(observer.spool.path.read_text(encoding="utf-8"))
            self.assertEqual(event["event"], "self_improvement.problem.processed")
            self.assertEqual(event["attributes"]["api_key"], "[REDACTED]")

    def test_disabled_phoenix_still_records_local_span(self) -> None:
        with TemporaryDirectory() as temp:
            observer = Observability(Path(temp), phoenix_enabled=False)
            with observer.span("factory.run", {"job_id": "job-1"}) as span:
                span.add_event("render.started", {"frame": 1})

            self.assertFalse(observer.phoenix_available)
            lines = observer.spool.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["name"], "factory.run")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["events"][0]["name"], "render.started")

    def test_missing_phoenix_packages_never_block_local_recording(self) -> None:
        with TemporaryDirectory() as temp, patch(
            "video_factory.observability.importlib.import_module",
            side_effect=ModuleNotFoundError("phoenix is optional"),
        ):
            observer = Observability(Path(temp), phoenix_enabled=True)
            with observer.span("factory.run"):
                pass

            self.assertFalse(observer.phoenix_available)
            self.assertIn("ModuleNotFoundError", observer.export_error or "")
            self.assertTrue(observer.spool.path.is_file())

    def test_redaction_happens_before_local_recording(self) -> None:
        with TemporaryDirectory() as temp:
            observer = Observability(Path(temp), phoenix_enabled=False)
            with observer.span("llm.call", {
                "api_key": "super-secret",
                "headers": {"Authorization": "Bearer abc123", "Accept": "application/json"},
                "url": "https://example.test/run?token=query-secret&mode=safe",
                "message": "send Bearer second-secret",
            }):
                pass

            raw = observer.spool.path.read_text(encoding="utf-8")
            self.assertNotIn("super-secret", raw)
            self.assertNotIn("abc123", raw)
            self.assertNotIn("query-secret", raw)
            self.assertNotIn("second-secret", raw)
            payload = json.loads(raw)
            self.assertEqual(payload["attributes"]["api_key"], "[REDACTED]")
            self.assertEqual(payload["attributes"]["headers"]["Authorization"], "[REDACTED]")
            self.assertIn("mode=safe", payload["attributes"]["url"])

    def test_nested_spans_share_trace_and_record_parent(self) -> None:
        with TemporaryDirectory() as temp:
            observer = Observability(Path(temp), phoenix_enabled=False)
            with observer.span("factory.run") as root:
                with observer.span("writing") as child:
                    child.set_attribute("attempt", 2)

            payloads = [
                json.loads(line)
                for line in observer.spool.path.read_text(encoding="utf-8").splitlines()
            ]
            child_event, root_event = payloads
            self.assertEqual(child_event["trace_id"], root_event["trace_id"])
            self.assertEqual(child_event["parent_span_id"], root.span_id)
            self.assertIsNone(root_event["parent_span_id"])
            self.assertEqual(child_event["attributes"]["attempt"], 2)

    def test_span_records_exception_without_suppressing_it(self) -> None:
        with TemporaryDirectory() as temp:
            observer = Observability(Path(temp), phoenix_enabled=False)
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                with observer.span("render"):
                    raise RuntimeError("render failed")

            payload = json.loads(observer.spool.path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "error")
            self.assertEqual(payload["events"][0]["attributes"]["type"], "RuntimeError")

    def test_explicit_noop_observer_is_context_manager_compatible(self) -> None:
        observer = NoopObservability()
        with observer.span("anything") as span:
            span.set_attribute("secret", "value")
            span.add_event("ignored")
        self.assertFalse(observer.record({"anything": True}))

    def test_redact_handles_nested_non_json_values(self) -> None:
        payload = redact({"path": Path("artifact.mp4"), "values": {1, 2}, "monkey": "visible"})
        self.assertEqual(payload["path"], "artifact.mp4")
        self.assertCountEqual(payload["values"], [1, 2])
        self.assertEqual(payload["monkey"], "visible")

    def test_unwritable_spool_never_raises_into_observed_code(self) -> None:
        with TemporaryDirectory() as temp, patch(
            "video_factory.observability.os.open", side_effect=PermissionError("read only"),
        ):
            observer = Observability(Path(temp), phoenix_enabled=False)
            with observer.span("factory.run"):
                pass
            self.assertIn("PermissionError", observer.spool.last_error or "")


if __name__ == "__main__":
    unittest.main()
