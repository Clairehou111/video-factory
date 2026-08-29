from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from .models import Candidate, Evidence, SourceType
from .narrative import extract_external_urls, route_external_source
from .storage import Workspace


@dataclass(slots=True)
class IngestResult:
    candidate: Candidate
    evidence: list[Evidence]
    linked_candidates: list[Candidate]


def _source_type_for_url(url: str) -> SourceType:
    hostname = urlparse(url).hostname or ""
    if hostname.endswith("github.com"):
        return SourceType.GITHUB
    if hostname.endswith("arxiv.org") or hostname.endswith("openreview.net"):
        return SourceType.PAPER
    return SourceType.WEB


def _linked_candidates(parent: Candidate, urls: list[str], resolver: Callable[[str], str] | None = None) -> list[Candidate]:
    candidates: list[Candidate] = []
    for index, url in enumerate(dict.fromkeys(urls), start=1):
        resolved_url = resolver(url) if resolver else url
        source_type = _source_type_for_url(resolved_url)
        candidates.append(Candidate(
            id=f"{parent.id}-linked-{index}", source_type=source_type, source_url=resolved_url,
            title=f"Linked source {index}", parent_candidate_id=parent.id,
            metadata={"original_url": url, "routed_topic": route_external_source(source_type).value},
        ))
    return candidates


class TwitterCliIngestor:
    """Ingest public, structured captures produced by twitter-cli, not a tweet summary."""

    def ingest(self, capture: Path, workspace: Workspace, resolve_link: Callable[[str], str] | None = None) -> IngestResult:
        data = json.loads(capture.read_text(encoding="utf-8"))
        posts = data.get("data", [])
        if not posts:
            raise ValueError("twitter-cli capture has no posts")
        root = posts[0]
        author = root.get("author", {})
        # twitter-cli returns the author's contiguous thread first, then public
        # replies. Replies must never silently become the author's evidence.
        author_name = author.get("screenName")
        thread_posts = []
        for post in posts:
            if post.get("author", {}).get("screenName") != author_name:
                break
            thread_posts.append(post)
        primary_thread_posts = [post for post in thread_posts if not post.get("context_only")]
        related_posts = [post for post in thread_posts if post.get("context_only")]
        candidate = Candidate(
            id=f"tweet-{root['id']}", source_type=SourceType.TWEET,
            source_url=f"https://x.com/{author.get('screenName', 'unknown')}/status/{root['id']}",
            title=(root.get("text") or "")[:120], author=author.get("screenName"),
            published_at=root.get("createdAtISO"), dedupe_key=f"x:{root['id']}",
            metadata={
                "author_verified": bool(author.get("verified")), "author_name": author.get("name"),
                "thread_length": len(primary_thread_posts), "related_post_count": len(related_posts),
                "reply_count_not_ingested": len(posts) - len(thread_posts),
                "metrics": root.get("metrics") or {}, "media": root.get("media") or [],
                "author_bio": str(root.get("bio") or ""),
            },
        )
        raw_path, raw_hash = workspace.archive_asset(capture, "twitter-captures")
        evidence: list[Evidence] = []
        urls: list[str] = []
        for index, post in enumerate(thread_posts, start=1):
            text = post.get("text") or ""
            post_urls = [*post.get("urls", []), *extract_external_urls(text)]
            urls.extend(post_urls)
            context_only = bool(post.get("context_only"))
            evidence.append(Evidence(
                id=f"{candidate.id}-post-{post['id']}", candidate_id=candidate.id,
                url=f"https://x.com/{author.get('screenName', 'unknown')}/status/{post['id']}", quote=text,
                source_kind="x:related_post" if context_only else "x:thread_post", captured_asset=raw_path, sha256=raw_hash,
                notes=(
                    "Related same-author post selected from the captured timeline; not part of the original thread."
                    if context_only else (
                        f"Thread position {index}; structured capture retains visible author/date/metrics."
                        + (f" Author bio: {root.get('bio')}" if root.get("bio") else "")
                    )
                ),
                metadata={
                    "author_handle": str(post.get("author", {}).get("screenName") or author.get("screenName") or ""),
                    "author_name": str(post.get("author", {}).get("name") or author.get("name") or ""),
                    "published_at": str(post.get("createdAtISO") or post.get("created_at") or ""),
                    "metrics": post.get("metrics") or {},
                },
            ))
            quoted = post.get("quoted_tweet")
            if isinstance(quoted, dict) and quoted.get("text"):
                quoted_url = str(quoted.get("url") or "")
                evidence.append(Evidence(
                    id=f"{candidate.id}-quoted-{quoted.get('id', index)}", candidate_id=candidate.id,
                    url=quoted_url or candidate.source_url, quote=str(quoted["text"]),
                    source_kind="x:quoted_post", captured_asset=raw_path, sha256=raw_hash,
                    notes="Quoted post is first-class required context, not hidden tweet metadata.",
                    metadata={
                        "author_handle": str(quoted.get("author") or ""),
                        "author_name": str(quoted.get("name") or quoted.get("author") or ""),
                        "published_at": str(quoted.get("createdAtISO") or quoted.get("created_at") or ""),
                        "metrics": quoted.get("metrics") or {},
                    },
                ))
                if quoted_url:
                    urls.append(quoted_url)
        candidate.linked_sources = list(dict.fromkeys(urls))
        linked = _linked_candidates(candidate, candidate.linked_sources, resolve_link)
        workspace.save_candidate(candidate)
        for item in evidence:
            workspace.save_evidence(item)
        for item in linked:
            workspace.save_candidate(item)
        return IngestResult(candidate, evidence, linked)


class GitHubIngestor:
    """Ingest GitHub API metadata plus the exact README returned at capture time."""

    def ingest(self, repo_json: Path, readme: Path, workspace: Workspace) -> IngestResult:
        repo = json.loads(repo_json.read_text(encoding="utf-8"))
        full_name = repo["full_name"]
        candidate = Candidate(
            id=f"github-{full_name.replace('/', '-')}", source_type=SourceType.GITHUB,
            source_url=repo["html_url"], title=full_name, author=repo.get("owner", {}).get("login"),
            published_at=repo.get("created_at"), dedupe_key=f"github:{full_name}",
            metadata={
                "description": repo.get("description"), "default_branch": repo.get("default_branch"),
                "license": (repo.get("license") or {}).get("spdx_id"), "stars_at_capture": repo.get("stargazers_count"),
                "pushed_at": repo.get("pushed_at"),
            },
        )
        repo_asset, repo_hash = workspace.archive_asset(repo_json, "github-api")
        readme_asset, readme_hash = workspace.archive_asset(readme, "github-readme")
        description = repo.get("description") or "No repository description"
        readme_text = readme.read_text(encoding="utf-8", errors="replace")
        evidence = [
            Evidence(f"{candidate.id}-metadata", candidate.id, candidate.source_url, description, "github:repository", repo_asset, sha256=repo_hash),
            Evidence(f"{candidate.id}-readme", candidate.id, f"{candidate.source_url}#readme", readme_text, "github:readme", readme_asset, sha256=readme_hash),
        ]
        workspace.save_candidate(candidate)
        for item in evidence:
            workspace.save_evidence(item)
        return IngestResult(candidate, evidence, [])


class WebPageIngestor:
    """Ingest an already-captured primary webpage and retain its parent post link."""

    def ingest(
        self, url: str, content: Path, workspace: Workspace, title: str,
        parent_candidate_id: str | None = None,
    ) -> IngestResult:
        if not content.is_file():
            raise FileNotFoundError(content)
        hostname = (urlparse(url).hostname or "web").replace(".", "-")
        slug = re.sub(r"[^a-z0-9]+", "-", hostname.casefold()).strip("-")
        url_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
        candidate = Candidate(
            id=f"web-{slug}-{url_key}", source_type=SourceType.WEB, source_url=url, title=title,
            parent_candidate_id=parent_candidate_id, dedupe_key=f"web:{url}",
        )
        asset, digest = workspace.archive_asset(content, "web-pages")
        text = content.read_text(encoding="utf-8", errors="replace")
        evidence = [Evidence(
            id=f"{candidate.id}-page", candidate_id=candidate.id, url=url, quote=text,
            source_kind="web:primary_page", captured_asset=asset, sha256=digest,
            notes="Primary page captured for linked-source extension.",
        )]
        workspace.save_candidate(candidate)
        workspace.save_evidence(evidence[0])
        return IngestResult(candidate, evidence, [])
