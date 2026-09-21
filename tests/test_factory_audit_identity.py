from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from video_factory.factory import GenerateOptions, VideoFactory
from video_factory.self_audit import PolicyStore
from video_factory.storage import Workspace


PASSING = {
    "reproduction_fixed": True, "deterministic_gates": True,
    "grounding": True, "safety": True, "golden_regressions": True,
    "confidence": 0.8,
}


class FactoryAuditIdentityTest(unittest.TestCase):
    def test_completed_artifacts_bind_hash_revision_and_active_policy(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            policy = {
                "runtime": {
                    "narrative_guidance": "Preserve the audience-recognizable event as the story axis.",
                    "evaluator_thresholds": {"story_axis": 0.8, "grounding": 1.0},
                },
            }
            PolicyStore(
                workspace.root / "automation" / "self-audit",
            ).promote(policy, problem_id="problem-1", verification=PASSING, version="policy-1")
            manifest = workspace.root / "manifest.json"
            video = workspace.root / "final.mp4"
            manifest.write_text('{"id":"story"}', encoding="utf-8")
            video.write_bytes(b"video")
            factory = VideoFactory(workspace)

            identity = factory._artifact_identity({"manifest": str(manifest), "video": str(video)})

            self.assertEqual(identity["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(identity["video_sha256"], hashlib.sha256(video.read_bytes()).hexdigest())
            self.assertEqual(identity["prompt_policy_version"], "policy-1")
            self.assertEqual(len(str(identity["prompt_policy_digest"])), 64)
            self.assertIn("code_revision", identity)

    def test_generation_root_span_records_job_and_stage_without_phoenix(self) -> None:
        with TemporaryDirectory() as temp:
            workspace = Workspace(Path(temp) / "workspace")
            workspace.initialize()
            factory = VideoFactory(workspace)
            generated = {
                "job_id": "job-1", "status": "completed", "source_type": "web",
                "stages": [{"name": "content_agent", "status": "ok"}],
            }
            with patch.object(factory, "_generate", return_value=generated):
                result = factory.generate("https://example.com/story", GenerateOptions(render=False))

            self.assertEqual(result["job_id"], "job-1")
            events = [json.loads(line) for line in (
                workspace.root / "observability" / "events.jsonl"
            ).read_text(encoding="utf-8").splitlines()]
            span = events[-1]
            self.assertEqual(span["name"], "factory.generation.job")
            self.assertEqual(span["attributes"]["job_id"], "job-1")
            self.assertEqual(span["events"][0]["name"], "factory.stage")


if __name__ == "__main__":
    unittest.main()
