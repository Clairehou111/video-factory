from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .models import Candidate, Evidence, now_iso
from .storage import Workspace


class CaptureKind(StrEnum):
    TWEET = "tweet"
    WEB = "web"
    GITHUB = "github"


@dataclass(slots=True)
class BrowserCaptureRequest:
    """A browser-side request. It never contains cookies or credentials."""

    kind: CaptureKind
    url: str
    candidate: Candidate
    quote: str
    selectors: list[str] = field(default_factory=list)
    record_steps: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CapturedBrowserArtifact:
    screenshot: Path
    page_text: str
    title: str | None = None
    recording: Path | None = None


class BrowserCaptureImporter:
    """Turns a capture produced through the user's signed-in browser into evidence.

    The browser controller is deliberately outside the domain layer: it may be a
    Chrome extension session, Playwright, or a human-operated export. This
    keeps login state out of this repository and makes capture artifacts
    reproducible once imported.
    """

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def import_capture(self, request: BrowserCaptureRequest, artifact: CapturedBrowserArtifact) -> Evidence:
        screenshot_path, digest = self.workspace.archive_asset(artifact.screenshot, request.kind.value)
        if artifact.recording:
            self.workspace.archive_asset(artifact.recording, f"{request.kind.value}-recordings")
        evidence = Evidence(
            id=f"evidence-{request.candidate.id}-{digest[:12]}",
            candidate_id=request.candidate.id,
            url=request.url,
            quote=request.quote,
            source_kind=f"browser:{request.kind.value}",
            captured_asset=screenshot_path,
            captured_at=now_iso(),
            sha256=digest,
            notes="Captured from an authenticated browser session; credentials are not archived.",
        )
        self.workspace.save_candidate(request.candidate)
        self.workspace.save_evidence(evidence)
        return evidence

