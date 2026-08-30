from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


TEST_REF = re.compile(r"^(tests/test_[^:]+\.py)::([A-Za-z_]\w*)::(test_[A-Za-z_]\w*)$")


def load_regression_registry(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / "regressions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("regressions.json must contain a JSON array")
    return [dict(item) for item in payload]


def validate_regression_registry(project_root: Path) -> list[str]:
    """Require every fixed production bug to own a discoverable unittest.

    The registry is intentionally source-controlled instead of living only in
    workspace audit logs. A bug is not considered fixed until its regression
    reference points to an actual test method collected by our test suite.
    """
    errors: list[str] = []
    rows = load_regression_registry(project_root)
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        bug_id = str(row.get("id") or "").strip()
        if not bug_id:
            errors.append(f"row {index}: id is required")
            continue
        if bug_id in seen_ids:
            errors.append(f"{bug_id}: duplicate id")
        seen_ids.add(bug_id)
        status = str(row.get("status") or "").strip()
        if status not in {"open", "fixed"}:
            errors.append(f"{bug_id}: status must be open or fixed")
        for field in ("title", "stage", "reproduction", "impact"):
            if not str(row.get(field) or "").strip():
                errors.append(f"{bug_id}: {field} is required")
        refs = row.get("tests") or []
        if status == "fixed" and not refs:
            errors.append(f"{bug_id}: fixed bug must reference at least one regression test")
        for ref in refs:
            match = TEST_REF.fullmatch(str(ref))
            if not match:
                errors.append(f"{bug_id}: invalid test reference {ref!r}")
                continue
            relative, class_name, method_name = match.groups()
            test_path = project_root / relative
            if not test_path.is_file():
                errors.append(f"{bug_id}: missing test file {relative}")
                continue
            source = test_path.read_text(encoding="utf-8")
            if not re.search(rf"^class\s+{re.escape(class_name)}\b", source, re.MULTILINE):
                errors.append(f"{bug_id}: missing test class {class_name}")
            if not re.search(rf"^\s+def\s+{re.escape(method_name)}\(", source, re.MULTILINE):
                errors.append(f"{bug_id}: missing test method {method_name}")
    return errors
