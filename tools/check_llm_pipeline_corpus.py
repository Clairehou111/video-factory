#!/usr/bin/env python3
"""Offline acceptance gate for a direct LLM-pipeline cutover.

The gate deliberately uses only archived manifests and assets. It never calls
an LLM or mutates the workspace. Recent outputs that were publishable at the
time they were produced must still pass today's deterministic publication
contract. The report also exposes how many legacy outputs needed more than the
new two semantic-review passes; those items now fail to human review instead
of weakening the acceptance bar or silently spending more.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_factory.quality import is_publishable, validate_manifest
from video_factory.serde import load_manifest


SEMANTIC_REVIEW_STEPS = {
    "copy_review", "copy_review_verify", "copy_review_final_verify",
    "copy_review_last_verify", "copy_review_convergence_verify",
}


def _manifest_path(raw: object, repository_root: Path) -> Path:
    path = Path(str(raw or ""))
    return path if path.is_absolute() else repository_root / path


def evaluate(workspace: Path, limit: int) -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[1]
    candidates: list[tuple[str, Path, dict[str, object]]] = []
    for result_path in (workspace / "jobs").glob("*/result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not (
            result.get("status") == "completed"
            and result.get("publishable") is True
            and result.get("manifest")
        ):
            continue
        completed = str(result.get("completed_at") or result.get("started_at") or "")
        candidates.append((completed, result_path, result))
    candidates.sort(key=lambda row: row[0], reverse=True)

    checked: list[dict[str, object]] = []
    for completed, result_path, result in candidates[:limit]:
        manifest_path = _manifest_path(result.get("manifest"), repository_root)
        row: dict[str, object] = {
            "job_id": result_path.parent.name,
            "completed_at": completed,
            "manifest": str(manifest_path),
        }
        try:
            manifest = load_manifest(manifest_path, normalize_story=False)
            checks = validate_manifest(manifest, workspace)
            failures = [check.name for check in checks if not check.passed]
            row["passes_current_quality_gate"] = is_publishable(checks)
            row["failed_checks"] = failures
            trace = next((
                list(check.get("detail", {}).get("trace") or [])
                for check in manifest.quality_checks
                if check.get("name") == "content_agent"
            ), [])
            review_calls = sum(
                isinstance(item, dict) and item.get("step") in SEMANTIC_REVIEW_STEPS
                for item in trace
            )
            row["legacy_semantic_review_rounds"] = review_calls
            row["new_pipeline_disposition"] = (
                "automatic" if review_calls <= 2 else "human_review_if_not_fixed_after_verification"
            )
        except Exception as error:
            row.update({
                "passes_current_quality_gate": False,
                "failed_checks": [f"{type(error).__name__}: {error}"],
                "legacy_semantic_review_rounds": None,
                "new_pipeline_disposition": "invalid_archive",
            })
        checked.append(row)

    passed = sum(bool(row["passes_current_quality_gate"]) for row in checked)
    legacy_over_two = sum(
        isinstance(row.get("legacy_semantic_review_rounds"), int)
        and int(row["legacy_semantic_review_rounds"]) > 2
        for row in checked
    )
    return {
        "workspace": str(workspace),
        "selection": "most recent completed jobs recorded as publishable",
        "requested": limit,
        "checked": len(checked),
        "passed_current_quality_gate": passed,
        "failed_current_quality_gate": len(checked) - passed,
        "legacy_items_over_new_two_review_limit": legacy_over_two,
        "gate_passed": bool(checked) and passed == len(checked),
        "items": checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    report = evaluate(args.workspace.resolve(), max(1, args.limit))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
