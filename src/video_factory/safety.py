from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Evidence


SENSITIVE_TERMS = (
    "vulnerability", "exploit", "jailbreak", "private data",
    "prompt injection", "watermark", "provenance mark", "c2pa",
    "绕过", "漏洞", "注入", "越狱", "去水印", "溯源标记",
)


@dataclass(frozen=True, slots=True)
class EditorialSafetyReview:
    requires_human_review: bool
    reasons: list[str]
    allowed_angle: str
    prohibited_angle: str | None = None


def review_evidence(evidence: list[Evidence]) -> EditorialSafetyReview:
    text = "\n".join(item.quote.lower() for item in evidence)
    hits = [term for term in SENSITIVE_TERMS if term in text]
    danger = (
        r"vulnerabilit(?:y|ies)|exploit|expos(?:e|ed|ure)|leak(?:ed|age)?|"
        r"steal|stolen|theft|dump|harvest|compromis(?:e|ed)|hardcod(?:e|ed)|plaintext|"
        r"漏洞|利用|暴露|泄露|窃取|明文|硬编码"
    )
    credential = r"credentials?|凭证"
    credential_risk = bool(re.search(
        rf"(?:{danger}).{{0,80}}(?:{credential})|(?:{credential}).{{0,80}}(?:{danger})",
        text,
        re.IGNORECASE | re.DOTALL,
    ))
    if credential_risk:
        hits.append("credentials" if "credentials" in text else "凭证")
    hits = list(dict.fromkeys(hits))
    if hits:
        return EditorialSafetyReview(
            requires_human_review=True,
            reasons=[f"Sensitive security terms found: {', '.join(hits)}"],
            allowed_angle="Report the disclosure, scope, affected users, provider response, and defensive actions only.",
            prohibited_angle="Do not generate reproduction steps, exploit payloads, extraction methods, or live demonstrations.",
        )
    return EditorialSafetyReview(False, [], "Normal evidence-bound editorial flow.")
