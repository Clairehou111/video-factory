from __future__ import annotations

import re
import subprocess
import json
import time
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from .acquisition import URLAcquirer
from .ingest import WebPageIngestor
from .models import Candidate, Evidence, TopicType
from .agent import ResearchOutcome
from .llm import EditorialPlan
from .writer import StoryWriterPacket
from .storage import Workspace


TRUSTED_REPORTING = (
    "axios.com", "techcrunch.com", "wired.com", "reuters.com", "bloomberg.com",
    "ft.com", "nytimes.com", "theverge.com", "arstechnica.com",
)


def _run_opencli_json(args: list[str], timeout: int) -> list[dict[str, object]]:
    """Run one read-only OpenCLI command with a bounded daemon-start retry."""
    last_detail = ""
    for attempt in range(3):
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        returncode = int(getattr(completed, "returncode", 0))
        if returncode == 0:
            rows = json.loads(completed.stdout)
            if not isinstance(rows, list):
                raise ValueError("OpenCLI output is not a list")
            return rows
        last_detail = (
            getattr(completed, "stderr", "") or completed.stdout or f"exit {returncode}"
        ).strip()
        if attempt < 2:
            time.sleep(1 + attempt)
    raise RuntimeError(last_detail or "OpenCLI failed")


def _context_headline_specificity_score(title: str) -> int:
    """Prefer a concrete personnel move over an abstract industry headline.

    Search headlines are often the only safely archived text when Google News
    wraps a publisher URL.  A pattern beat is useful only when that headline
    carries audience-facing facts: named people/organizations and an explicit
    move or destination.  Score those positive signals so the bounded search
    selects the most informative result without inventing article-body detail.
    """
    movement = len(re.findall(
        r"\b(?:leave|leaves|leaving|left|join|joins|joined|joining|depart|departs|"
        r"departed|departing|found|founds|founded|launch|launches|launched|"
        r"acqui-?hire|acqui-?hires|acqui-?hired)\b|离开|离职|转投|加入|创办|创业|收购",
        title, re.IGNORECASE,
    ))
    destination = len(re.findall(
        r"\b(?:for|to|at)\s+(?:OpenAI|Anthropic|Google|DeepMind|Meta|Microsoft|"
        r"Nvidia|Apple|Amazon)\b|转投|加入|创办|创业",
        title, re.IGNORECASE,
    ))
    named_tokens = {
        token for token in re.findall(r"\b[A-Z][A-Za-z0-9.-]{2,}\b", title)
        if token.casefold() not in {"the", "and", "new", "why", "how", "ai"}
    }
    return movement * 5 + destination * 4 + min(len(named_tokens), 6)


class BoundedContextResearcher:
    """Small web-search tool for event context, never an unconstrained crawler."""

    def __init__(self, workspace: Workspace, max_sources: int = 2):
        self.workspace = workspace
        self.max_sources = max_sources

    def research(
        self, candidate: Candidate, topic: TopicType, job: Path, seed_evidence: list[Evidence] | None = None,
        planned_queries: list[str] | None = None, pattern_query: str | None = None,
        identity_query: str | None = None,
    ) -> tuple[list[Evidence], list[dict[str, object]]]:
        if topic == TopicType.GITHUB_PROJECT:
            return [], []
        subject = str(candidate.metadata.get("author_name") or candidate.author or candidate.title[:80])
        first_line = next((line.strip() for line in candidate.title.splitlines() if line.strip()), candidate.title[:60])
        planned = [item.strip() for item in (planned_queries or []) if item.strip()]
        query_specs: list[tuple[str, str]] = []
        if pattern_query:
            query_specs.append((pattern_query, "incumbent_history"))
        if identity_query:
            query_specs.append((identity_query, "identity_anchor"))
        query_specs.extend((item, "event_context") for item in planned[: 3 - len(query_specs)])
        queries = [item[0] for item in query_specs]
        if not queries:
            queries = [f'"{subject}" "{first_line[:60]}" AI']
            query_specs = [(queries[0], "event_context")]
        actions: list[dict[str, object]] = []
        items: list[tuple[int, str, str, str, str]] = []
        for query_index, query in enumerate(queries):
            search_url = "https://news.google.com/rss/search?q=" + quote(query) + "&hl=en-US&gl=US&ceid=US:en"
            action: dict[str, object] = {"query": query, "url": search_url, "status": "searched"}
            actions.append(action)
            try:
                payload = subprocess.run(
                    ["curl", "-fsSL", "--max-time", "30", search_url], check=True,
                    capture_output=True, timeout=35,
                ).stdout
                root = ElementTree.fromstring(payload)
            except Exception as error:
                action.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                continue
            query_role = query_specs[query_index][1]
            for item in root.findall(".//item"):
                url = (item.findtext("link") or "").strip()
                title = (item.findtext("title") or "").strip()
                published = (item.findtext("pubDate") or "").strip()
                source = item.find("source")
                publisher_url = str(source.attrib.get("url", "")) if source is not None else ""
                host = (urlparse(publisher_url).hostname or "").casefold()
                if not url or not any(host == domain or host.endswith("." + domain) for domain in TRUSTED_REPORTING):
                    continue
                if query_role == "identity_anchor" and not _identity_context_matches_subject(
                    subject, title,
                ):
                    continue
                if query_role == "event_context" and not _reported_context_matches_root(
                    candidate, seed_evidence or [], title,
                ):
                    continue
                if query_role == "incumbent_history":
                    if not _is_prior_distinct_event(candidate, seed_evidence or [], title, published):
                        continue
                    # A generic "talent war" headline cannot prove the
                    # concrete earlier move the audience needs.  Prefer a
                    # named action-bearing result; when none exists, omit the
                    # optional pattern instead of forcing a vague card.
                    if _context_headline_specificity_score(title) < 7:
                        continue
                items.append((query_index, url, title, published, publisher_url))
        evidence: list[Evidence] = []
        acquirer = URLAcquirer(self.workspace)
        seen_titles: set[str] = set()
        ranked: list[tuple[int, str, str, str, str]] = []
        items.sort(key=lambda item: (item[0], -_context_headline_specificity_score(item[2])))
        # First take one result per director query so a broad first query does
        # not consume the whole source budget; then fill remaining slots.
        for query_index in range(len(queries)):
            item = next((entry for entry in items if entry[0] == query_index and entry[2] not in seen_titles), None)
            if item:
                ranked.append(item)
                seen_titles.add(item[2])
        for item in items:
            if item[2] not in seen_titles:
                ranked.append(item)
                seen_titles.add(item[2])
        for index, (query_index, url, title, published, publisher_url) in enumerate(ranked[: self.max_sources], start=1):
            # Google News RSS links are redirect wrappers. Fetching one as if
            # it were the publisher article can return hundreds of KB of
            # Google application JavaScript, which both pollutes evidence and
            # explodes LLM cost. Preserve the trusted outlet headline/date as
            # bounded reported context; the director may use exactly that
            # event, but no unsupported article-body detail.
            if (urlparse(url).hostname or "").casefold() == "news.google.com":
                path = job / f"context-search-{index}.json"
                path.write_text(json.dumps({
                    "title": title, "published_at": published,
                    "publisher": publisher_url, "discovery_url": url,
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                asset, digest = self.workspace.archive_asset(path, "search-results")
                found = Evidence(
                    id=f"{candidate.id}-reported-context-{index}", candidate_id=candidate.id,
                    url=url, quote=f"{title}\nPublished: {published}",
                    source_kind="web:reported_context", captured_asset=asset, sha256=digest,
                    notes=(
                        "Trusted-outlet search headline and publication date only. "
                        "It supports only the event stated in the headline, not unseen article details."
                    ),
                    metadata={
                        "publisher": publisher_url, "published_at": published,
                        "context_role": query_specs[query_index][1],
                    },
                )
                self.workspace.save_evidence(found)
                evidence.append(found)
                candidate.linked_sources.append(url)
                actions.append({
                    "url": url, "status": "archived_reported_context",
                    "evidence": found.id, "publisher": publisher_url,
                })
                continue
            try:
                body, content_type, method = acquirer._fetch_readable(url)
                text = body.decode("utf-8", errors="replace")
                if "html" in content_type:
                    from .acquisition import _TextExtractor

                    parser = _TextExtractor()
                    parser.feed(text)
                    text = parser.text()
                text = text[:80_000]
                if re.search(r"window\.[A-Z_]+\s*=|function\([a-z],?[a-z]?\)\{|boq_", text[:5000]):
                    raise ValueError("source extraction returned application JavaScript, not an article")
                path = job / f"context-{index}.md"
                path.write_text(text + "\n", encoding="utf-8")
                ingest = WebPageIngestor().ingest(url, path, self.workspace, title or url, candidate.id)
                evidence.extend(ingest.evidence)
                candidate.linked_sources.append(url)
                actions.append({"url": url, "status": "archived", "method": method, "evidence": ingest.evidence[0].id})
            except Exception as error:
                # Search discovery remains auditable but is explicitly marked
                # as context-only when the publisher blocks extraction.
                path = job / f"context-search-{index}.json"
                path.write_text(json.dumps({
                    "title": title, "published_at": published, "publisher": publisher_url,
                    "discovery_url": url,
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                asset, digest = self.workspace.archive_asset(path, "search-results")
                found = Evidence(
                    id=f"{candidate.id}-search-{index}", candidate_id=candidate.id,
                    url=publisher_url or url, quote=f"{title}\nPublished: {published}",
                    source_kind="web:search_result", captured_asset=asset, sha256=digest,
                    notes="Discovery context only; open the publisher page before using details beyond the headline.",
                )
                self.workspace.save_evidence(found)
                evidence.append(found)
                actions.append({"url": url, "status": "archived_search_result", "error": f"{type(error).__name__}: {error}", "evidence": found.id})
        candidate.linked_sources = list(dict.fromkeys(candidate.linked_sources))
        self.workspace.save_candidate(candidate)
        return evidence, actions


class DirectorContextToolbox:
    """Execute only the bounded context questions selected by the director planner."""

    name = "director_context_toolbox"

    def __init__(self, workspace: Workspace, job: Path):
        self.researcher = BoundedContextResearcher(workspace)
        self.job = job

    def run(self, packet: StoryWriterPacket, plan: EditorialPlan, limit: int) -> ResearchOutcome:
        prior_evidence, prior_actions = _archive_same_author_setup(
            self.researcher.workspace, self.job, packet, limit=1,
        )
        actor_evidence, actor_actions = _archive_visual_actor_context(
            self.researcher.workspace, self.job, packet,
            limit=min(2, max(0, limit - len(prior_evidence))),
        )
        self.researcher.max_sources = max(0, limit - len(prior_evidence) - len(actor_evidence))
        pattern_query = _incumbent_history_query(packet, plan)
        identity_query = _identity_anchor_query(packet, plan)
        found, actions = self.researcher.research(
            packet.candidate, packet.topic_type, self.job, packet.evidence,
            planned_queries=plan.search_queries, pattern_query=pattern_query,
            identity_query=identity_query,
        )
        existing = {item.id for item in packet.evidence}
        evidence = [
            *packet.evidence,
            *(item for item in prior_evidence if item.id not in existing),
            *(item for item in actor_evidence if item.id not in existing),
            *(item for item in found if item.id not in existing),
        ]
        for action in [*prior_actions, *actor_actions, *actions]:
            action.setdefault("dimensions", plan.expansion_dimensions)
            action.setdefault("questions", plan.context_questions)
        return ResearchOutcome(evidence, [*prior_actions, *actor_actions, *actions])


def _parse_source_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            return None


def _root_published_at(candidate: Candidate, evidence: list[Evidence]) -> datetime | None:
    for item in evidence:
        if item.url.rstrip("/") == candidate.source_url.rstrip("/"):
            parsed = _parse_source_date(str(item.metadata.get("published_at") or ""))
            if parsed:
                return parsed
    return _parse_source_date(str(candidate.metadata.get("published_at") or candidate.metadata.get("created_at") or ""))


def _is_prior_distinct_event(
    candidate: Candidate, evidence: list[Evidence], title: str, published: str,
) -> bool:
    """A trend node must predate the root; three headlines about today are not a pattern."""
    root_date = _root_published_at(candidate, evidence)
    event_date = _parse_source_date(published)
    if root_date and event_date and event_date >= root_date - timedelta(days=2):
        return False
    author = str(candidate.metadata.get("author_name") or candidate.author or "")
    author = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", author).strip()
    if len(author) >= 5 and re.search(re.escape(author), title, re.IGNORECASE):
        return False
    return True


def _incumbent_history_query(packet: StoryWriterPacket, plan: EditorialPlan) -> str | None:
    """Add one mandatory pattern check for a people move without another LLM call."""
    if plan.story_archetype != "people_change":
        return None
    signal = "\n".join([
        plan.angle, plan.why_now, plan.why_audience,
        *(item.quote + "\n" + (item.notes or "") for item in packet.evidence),
    ])
    known = (
        "Google", "OpenAI", "Anthropic", "Meta", "Microsoft", "Apple", "Amazon",
        "NVIDIA", "DeepMind", "字节跳动", "阿里", "腾讯", "百度", "华为", "谷歌",
    )
    incumbent = next((name for name in known if re.search(rf"\b{re.escape(name)}\b", signal, re.IGNORECASE)), None)
    if not incumbent:
        match = re.search(
            r"(?:Former|leav(?:e|ing)|left|depart(?:ure|ing)?|离开|离职|出走).{0,50}?([A-Z][A-Za-z0-9.&-]{2,})",
            signal,
        )
        incumbent = match.group(1) if match else None
    if not incumbent:
        return None
    author = str(packet.candidate.metadata.get("author_name") or packet.candidate.author or "").strip("@")
    author = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", author).strip()
    exclusion = f' -"{author}"' if author else ""
    return f'"{incumbent}" AI researchers departures {datetime.now(UTC).year}{exclusion}'


def _identity_anchor_query(packet: StoryWriterPacket, plan: EditorialPlan) -> str | None:
    """Research why a moving person matters instead of assuming celebrity."""
    if plan.story_archetype != "people_change":
        return None
    root = next((
        item for item in packet.evidence
        if item.url.rstrip("/") == packet.candidate.source_url.rstrip("/")
    ), None)
    name = str(
        (root.metadata.get("author_name") if root else "")
        or packet.candidate.metadata.get("author_name")
        or packet.candidate.author
        or ""
    ).strip("@ ")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).strip()
    if not name:
        return None
    return f'"{name}" known for built designed projects role'


def _identity_context_matches_subject(subject: str, title: str) -> bool:
    normalized_subject = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", subject.casefold())
    normalized_title = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", title.casefold())
    return bool(normalized_subject and normalized_subject in normalized_title)


def _topic_terms(text: str) -> set[str]:
    stop = {
        "this", "that", "with", "have", "will", "from", "your", "what", "there", "landed",
        "amazing", "weekend", "repeat", "users", "paid", "before", "after", "today",
        "researcher", "researchers", "company", "companies", "model", "models", "team",
        "technology", "technical", "artificial", "intelligence",
        # Broad AI/news vocabulary is not an entity bridge.  In particular,
        # "drug discovery" must never become context for an agent-discovery
        # protocol, and two unrelated posts from an open-source account must
        # not match merely on generic ecosystem words.
        "agent", "agents", "agentic", "build", "discovery", "github", "open",
        "resource", "source", "spec", "standard",
    }
    return {
        item.casefold() for item in re.findall(r"[A-Za-z][A-Za-z0-9.-]{3,}", text)
        if item.casefold() not in stop
    }


def _reported_context_matches_root(
    candidate: Candidate, evidence: list[Evidence], title: str,
) -> bool:
    """Require a headline-level entity bridge to the root event.

    A trusted headline proves its own event, not its relationship to the post.
    Without a shared distinctive token the item remains research noise and is
    never eligible for the writer.
    """
    root_text = "\n".join([
        candidate.title,
        str(candidate.author or ""),
        *(item.quote for item in evidence if item.source_kind.startswith("x:")),
    ])
    return bool(_topic_terms(root_text) & _topic_terms(title))


def _archive_same_author_setup(
    workspace: Workspace, job: Path, packet: StoryWriterPacket, limit: int = 1,
) -> tuple[list[Evidence], list[dict[str, object]]]:
    """Use the author's bounded recent timeline to recover a setup the root assumes."""
    if packet.candidate.source_type.value != "tweet" or packet.topic_type not in {
        TopicType.OFFICIAL_ANNOUNCEMENT, TopicType.PRACTICE_POST,
        TopicType.MODEL_OR_PRODUCT, TopicType.TOOL_SDK_AGENT,
    }:
        return [], []
    root = next((item for item in packet.evidence if item.url.rstrip("/") == packet.candidate.source_url.rstrip("/")), None)
    if not root:
        return [], []
    handle = str(root.metadata.get("author_handle") or packet.candidate.metadata.get("author_handle") or packet.candidate.author or "").strip("@")
    if not handle:
        return [], []
    action: dict[str, object] = {
        "step": "same_author_context", "handle": handle, "status": "searched",
        "method": "opencli-twitter-timeline",
    }
    try:
        rows = _run_opencli_json(
            ["opencli", "twitter", "tweets", handle, "--limit", "20", "--page-delay", "0", "-f", "json"],
            timeout=45,
        )
    except Exception as error:
        action.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        return [], [action]
    root_date = _root_published_at(packet.candidate, packet.evidence)
    root_terms = _topic_terms("\n".join(item.quote for item in packet.evidence if item.source_kind.startswith("x:")))
    known_ids = {item.url.rstrip("/").rsplit("/", 1)[-1] for item in packet.evidence if item.url}
    ranked: list[tuple[float, dict[str, object]]] = []
    for row in rows:
        if not isinstance(row, dict) or str(row.get("id") or "") in known_ids:
            continue
        occurred = _parse_source_date(str(row.get("created_at") or ""))
        if root_date and (not occurred or occurred >= root_date or occurred < root_date - timedelta(days=21)):
            continue
        text = str(row.get("text") or "").strip()
        overlap = root_terms & _topic_terms(text)
        if len(overlap) < 2:
            continue
        score = len(overlap) * 4 + min(3, len(text) / 220)
        ranked.append((score, row))
    selected = [row for _, row in sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]]
    archived: list[Evidence] = []
    for index, row in enumerate(selected, start=1):
        path = job / f"same-author-context-{index}.json"
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        asset, digest = workspace.archive_asset(path, "twitter-captures")
        item = Evidence(
            id=f"{packet.candidate.id}-same-author-prior-{row['id']}", candidate_id=packet.candidate.id,
            url=str(row.get("url") or f"https://x.com/{handle}/status/{row['id']}"), quote=str(row.get("text") or ""),
            source_kind="x:related_prior_post", captured_asset=asset, sha256=digest,
            notes="Earlier same-author post selected by topic overlap; use only as chronological setup.",
            metadata={
                "author_handle": str(row.get("author") or handle), "author_name": str(row.get("name") or handle),
                "published_at": str(row.get("created_at") or ""), "metrics": {
                    key: int(row.get(key) or 0) for key in ("likes", "retweets", "replies", "views")
                }, "context_role": "same_author_setup",
            },
        )
        workspace.save_evidence(item)
        archived.append(item)
        packet.candidate.linked_sources.append(item.url)
    if archived:
        workspace.save_candidate(packet.candidate)
        action.update({"status": "archived", "evidence": [item.id for item in archived]})
    else:
        action.update({"status": "no_relevant_prior_post"})
    return archived, [action]


def _archive_visual_actor_context(
    workspace: Workspace, job: Path, packet: StoryWriterPacket, limit: int = 2,
) -> tuple[list[Evidence], list[dict[str, object]]]:
    """Recover the concrete post behind a screenshot-described X dispute.

    This is intentionally narrow: only a screenshot that visibly mentions a
    setup plus an account suspension/appeal can trigger it. The tool reads a
    few named actors' recent timelines and archives the exact posts; it never
    treats a screenshot interpretation as proof of an unseen configuration.
    """
    if limit <= 0 or packet.candidate.source_type.value != "tweet":
        return [], []
    # Exact referenced posts are durable acquisition evidence for this same
    # candidate. Reuse them before asking the multimodal service to rediscover
    # the screenshot text. Otherwise a transient vision/API failure can erase
    # the causal setup from a later run and make the writer invent a vague
    # summary from the root tweet alone.
    cached = [
        item for item in workspace.evidence_for_candidate(packet.candidate.id)
        if item.source_kind == "x:referenced_context_post"
        and str(item.metadata.get("context_role") or "") == "referenced_setup"
        and item.quote.strip()
    ]
    if cached:
        cached.sort(key=lambda item: str(item.metadata.get("published_at") or ""))
        selected = cached[:limit]
        for item in selected:
            if item.url not in packet.candidate.linked_sources:
                packet.candidate.linked_sources.append(item.url)
        workspace.save_candidate(packet.candidate)
        return selected, [{
            "step": "visual_actor_context", "status": "workspace_cache",
            "evidence": [item.id for item in selected],
        }]
    analyses = [item for item in packet.evidence if item.source_kind == "x:visual_analysis"]
    visual_text = "\n".join(item.quote for item in analyses)
    if not re.search(r"suspend|suspens|bann?ed|appeal|封禁|封号|申诉", visual_text, re.IGNORECASE):
        return [], []
    if not re.search(r"setup|config|model|配置|模型", visual_text, re.IGNORECASE):
        return [], []
    root_handle = str(
        packet.candidate.metadata.get("author_handle") or packet.candidate.author or ""
    ).strip("@").casefold()
    handles: list[tuple[int, str]] = []
    for match in re.finditer(r"@([A-Za-z0-9_]{2,15})", visual_text):
        handle = match.group(1)
        if handle.casefold() == root_handle:
            continue
        window = visual_text[max(0, match.start() - 260):match.end() + 320]
        signal_score = len(re.findall(
            r"suspend|suspens|bann?ed|appeal|setup|config|account|proxy|oauth|封禁|封号|申诉|配置",
            window, re.IGNORECASE,
        ))
        handles.append((signal_score, handle))
    ordered_handles: list[str] = []
    for _, handle in sorted(handles, key=lambda item: item[0], reverse=True):
        if handle.casefold() not in {item.casefold() for item in ordered_handles}:
            ordered_handles.append(handle)
    if not ordered_handles:
        return [], []

    actions: list[dict[str, object]] = []
    ranked: list[tuple[float, str, dict[str, object]]] = []
    root_date = _root_published_at(packet.candidate, packet.evidence)
    visual_terms = _topic_terms(visual_text)
    for handle in ordered_handles[:3]:
        action: dict[str, object] = {
            "step": "visual_actor_context", "handle": handle,
            "method": "opencli-twitter-search", "status": "searched",
        }
        actions.append(action)
        try:
            rows = _run_opencli_json(
                [
                    "opencli", "twitter", "search",
                    f"from:{handle} (suspended OR banned OR appeal OR setup)", "-f", "json",
                ],
                timeout=60,
            )
        except Exception as error:
            action.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            occurred = _parse_source_date(str(row.get("created_at") or ""))
            if root_date and (not occurred or occurred > root_date or occurred < root_date - timedelta(days=45)):
                continue
            quoted = row.get("quoted_tweet") if isinstance(row.get("quoted_tweet"), dict) else {}
            card = row.get("card") if isinstance(row.get("card"), dict) else {}
            blob = "\n".join((
                str(row.get("text") or ""), str(quoted.get("text") or ""),
                str(card.get("title") or ""), str(card.get("description") or ""),
            ))
            signal_hits = len(re.findall(
                r"suspend|suspens|bann?ed|appeal|setup|proxy|oauth|account|another model|different model|封禁|封号|申诉",
                blob, re.IGNORECASE,
            ))
            overlap = len(visual_terms & _topic_terms(blob))
            if signal_hits < 2 or overlap < 1:
                continue
            ranked.append((signal_hits * 5 + overlap, handle, row))
        action["candidates"] = sum(1 for _, actor, _ in ranked if actor == handle)

    archived: list[Evidence] = []
    seen_ids: set[str] = set()
    for _, handle, row in sorted(ranked, key=lambda item: item[0], reverse=True):
        row_id = str(row.get("id") or "")
        if not row_id or row_id in seen_ids or len(archived) >= limit:
            continue
        seen_ids.add(row_id)
        quoted = row.get("quoted_tweet") if isinstance(row.get("quoted_tweet"), dict) else {}
        card = row.get("card") if isinstance(row.get("card"), dict) else {}
        quote_parts = [str(row.get("text") or "").strip()]
        if quoted:
            quote_parts.append(
                f"Quoted post by @{quoted.get('author') or 'unknown'}:\n{str(quoted.get('text') or '').strip()}"
            )
        if card:
            quote_parts.append("Linked card:\n" + "\n".join(
                value for value in (
                    str(card.get("title") or "").strip(),
                    str(card.get("description") or "").strip(),
                    str(card.get("url") or "").strip(),
                ) if value
            ))
        path = job / f"visual-actor-context-{len(archived) + 1}.json"
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        asset, digest = workspace.archive_asset(path, "twitter-captures")
        item = Evidence(
            id=f"{packet.candidate.id}-referenced-{row_id}", candidate_id=packet.candidate.id,
            url=str(row.get("url") or f"https://x.com/{handle}/status/{row_id}"),
            quote="\n\n".join(part for part in quote_parts if part),
            source_kind="x:referenced_context_post", captured_asset=asset, sha256=digest,
            notes="Exact X post recovered from a screenshot-named actor; use as event setup, not as platform-wide policy proof.",
            metadata={
                "author_handle": str(row.get("author") or handle),
                "author_name": str(row.get("name") or row.get("author") or handle),
                "published_at": str(row.get("created_at") or ""),
                "context_role": "referenced_setup",
            },
        )
        workspace.save_evidence(item)
        archived.append(item)
        packet.candidate.linked_sources.append(item.url)
    if archived:
        workspace.save_candidate(packet.candidate)
    actions.append({
        "step": "visual_actor_context", "status": "archived" if archived else "no_relevant_posts",
        "evidence": [item.id for item in archived],
    })
    return archived, actions
