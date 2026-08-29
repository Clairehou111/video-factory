from __future__ import annotations

from dataclasses import dataclass

from .models import Evidence


SENSITIVE_TERMS = (
    "vulnerability", "exploit", "jailbreak", "credentials", "private data",
    "prompt injection", "watermark", "provenance mark", "c2pa",
    "绕过", "漏洞", "凭证", "注入", "越狱", "去水印", "溯源标记",
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
    if hits:
        return EditorialSafetyReview(
            requires_human_review=True,
            reasons=[f"Sensitive security terms found: {', '.join(hits)}"],
            allowed_angle="Report the disclosure, scope, affected users, provider response, and defensive actions only.",
            prohibited_angle="Do not generate reproduction steps, exploit payloads, extraction methods, or live demonstrations.",
        )
    return EditorialSafetyReview(False, [], "Normal evidence-bound editorial flow.")
