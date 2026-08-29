from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from .models import Candidate, Evidence
from .storage import Workspace


_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_CONTEXT_PATH = re.compile(
    r"(?:vendor[-_ ]?notes?|background|context|announcement|official|references?/.+\.md$)",
    re.IGNORECASE,
)
_SOURCE_LABEL = re.compile(r"source|official|announcement|docs?|公告|官方|来源", re.IGNORECASE)


def markdown_links(text: str) -> list[tuple[str, str]]:
    return [(label.strip(), target.strip().split(" ", 1)[0]) for label, target in _MARKDOWN_LINK.findall(text)]


def enrich_github_context(
    candidate: Candidate,
    readme_text: str,
    owner: str,
    repo: str,
    default_branch: str,
    workspace: Workspace,
    job: Path,
    fetch_content,
    max_documents: int = 2,
) -> tuple[list[Evidence], list[str], list[dict[str, object]]]:
    """Archive small README-linked context documents before story planning.

    This is deliberately bounded and deterministic.  It does not web-search or
    let the model invent a URL; it opens only relative Markdown documents that
    the repository itself names as vendor/background/reference material.
    External source links found inside those documents are returned for the
    bounded research agent to open as primary sources.
    """
    evidence: list[Evidence] = []
    external_sources: list[str] = []
    actions: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for label, target in markdown_links(readme_text):
        parsed = urlparse(target)
        if parsed.scheme or parsed.netloc or target.startswith("#"):
            continue
        path = unquote(parsed.path).lstrip("./")
        if not path or path in seen_paths or not _CONTEXT_PATH.search(f"{label} {path}"):
            continue
        normalized = str(PurePosixPath(path))
        if normalized.startswith("../") or normalized == "..":
            continue
        seen_paths.add(normalized)
        if len(evidence) >= max_documents:
            break
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{normalized}?ref={default_branch}"
        browser_url = f"https://github.com/{owner}/{repo}/blob/{default_branch}/{normalized}"
        try:
            body = fetch_content(api_url, "application/vnd.github.raw+json")
            text = body.decode("utf-8", errors="replace")[:250_000]
            if not text.strip():
                raise ValueError("context document is empty")
            local = job / ("context-" + re.sub(r"[^a-zA-Z0-9._-]", "-", Path(normalized).name))
            local.write_bytes(body)
            asset, digest = workspace.archive_asset(local, "github-context")
            identifier = hashlib.sha256(browser_url.encode("utf-8")).hexdigest()[:12]
            item = Evidence(
                id=f"{candidate.id}-context-{identifier}", candidate_id=candidate.id,
                url=browser_url, quote=text, source_kind="github:linked_context",
                captured_asset=asset, sha256=digest,
                notes="README-linked background/vendor document; factual context, not a browser target.",
            )
            workspace.save_evidence(item)
            evidence.append(item)
            for source_label, source_url in markdown_links(text):
                source = urlparse(source_url)
                if source.scheme in {"http", "https"} and source.netloc and (
                    _SOURCE_LABEL.search(source_label) or "support." in source.netloc.casefold()
                ):
                    external_sources.append(source_url)
            actions.append({"url": browser_url, "status": "archived", "evidence_id": item.id})
        except Exception as error:
            actions.append({"url": browser_url, "status": "failed", "error": str(error)})
    return evidence, list(dict.fromkeys(external_sources)), actions
