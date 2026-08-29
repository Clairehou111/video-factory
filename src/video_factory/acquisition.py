from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .ingest import IngestResult, TwitterCliIngestor, WebPageIngestor
from .links import ExternalLinkResolver
from .models import Candidate, Evidence, SourceType
from .narrative import extract_external_urls
from .storage import Workspace


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    ingest: IngestResult
    artifact: Path
    method: str


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "pre", "code", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "pre"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            self.parts.append(value + " ")
            if self._in_title:
                self.title += value

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self.parts)).strip()


class URLAcquirer:
    """Program-owned one-URL acquisition; no Codex step is involved."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def acquire(self, url: str, job: Path) -> AcquisitionResult:
        host = (urlparse(url).hostname or "").casefold()
        if host in {"x.com", "twitter.com", "www.x.com", "www.twitter.com"}:
            return self._acquire_x(url, job)
        if url.casefold().split("?", 1)[0].endswith(".pdf") or host.endswith(("arxiv.org", "openreview.net")):
            return self._acquire_pdf(url, job)
        return self._acquire_web(url, job)

    def _acquire_x(self, url: str, job: Path) -> AcquisitionResult:
        match = re.search(r"/(?:i/web/)?status/(\d+)", url)
        path_parts = [part for part in urlparse(url).path.split("/") if part]
        if not match or not path_parts:
            raise ValueError("X URL must contain a numeric status id")
        status_id = match.group(1)
        author = path_parts[0] if path_parts[0] != "i" else ""
        capture = job / "x-capture.json"
        configured = os.environ.get("VIDEO_FACTORY_X_CAPTURE_COMMAND", "").strip()
        attempts: list[tuple[list[str], str]] = []
        if configured:
            attempts.append(([part.replace("{url}", url).replace("{author}", author).replace("{id}", status_id) for part in configured.split()], "configured"))
        attempts.append(([
            "opencli", "twitter", "thread", status_id, "-f", "json",
            "--window", "background", "--trace", "retain-on-failure",
        ], "opencli-twitter-thread"))
        if author:
            attempts.append(([
                "opencli", "twitter", "tweets", author, "-f", "json",
                "--window", "background", "--trace", "retain-on-failure",
            ], "opencli-twitter"))
        errors: list[str] = []
        payload: dict | None = None
        method = ""
        for command, name in attempts:
            try:
                completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
                parsed = self._parse_json_output(completed.stdout)
                payload = self._normalize_x_payload(parsed, status_id, author)
                if payload.get("data"):
                    method = name
                    break
                errors.append(f"{name}: exact status {status_id} was not present")
            except Exception as error:
                errors.append(f"{name}: {type(error).__name__}: {error}")
        if not payload or not payload.get("data"):
            raise RuntimeError("X acquisition failed; " + "; ".join(errors))
        capture.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        resolver = ExternalLinkResolver()
        ingest = TwitterCliIngestor().ingest(
            capture, self.workspace,
            lambda value: resolver.resolve(value).resolved_url,
        )
        self._archive_x_media(ingest, payload, job)
        return AcquisitionResult(ingest, capture, method)

    def _archive_x_media(self, ingest: IngestResult, payload: dict, job: Path) -> None:
        """Promote root and quoted-post photos to immutable visual evidence."""
        from PIL import Image

        root_id = ingest.candidate.id.removeprefix("tweet-")
        root = next((item for item in payload.get("data", []) if str(item.get("id")) == root_id), None)
        if not isinstance(root, dict):
            return
        media_owners: list[tuple[dict, str, str]] = [(root, ingest.candidate.source_url, "root")]
        quoted = root.get("quoted_tweet")
        if isinstance(quoted, dict):
            quoted_url = str(quoted.get("url") or ingest.candidate.source_url)
            media_owners.append((quoted, quoted_url, "quoted"))
        seen_urls: set[str] = set()
        media_index = 0
        for owner, parent_url, relationship in media_owners:
            source_text = str(owner.get("text") or "")
            folded = source_text.casefold()
            visual_role = (
                "architecture" if any(term in folded for term in ("architecture", "diagram", "架构"))
                else "benchmark" if any(term in folded for term in ("benchmark", "chart", "result", "基准", "图表"))
                else "quoted_context" if relationship == "quoted"
                else "product"
            )
            for media in owner.get("media") or []:
                media_index += 1
                if not isinstance(media, dict) or str(media.get("type") or "").casefold() not in {"photo", "image"}:
                    continue
                media_url = str(media.get("url") or media.get("media_url_https") or "").strip()
                if not media_url or media_url in seen_urls:
                    continue
                seen_urls.add(media_url)
                body, content_type = self._fetch(media_url)
                if len(body) > 12_000_000:
                    raise ValueError("attached X image exceeds 12 MB")
                with Image.open(io.BytesIO(body)) as image:
                    width, height = image.size
                    image_format = (image.format or "JPEG").casefold()
                if width < 240 or height < 180:
                    continue
                extension = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "webp": ".webp"}.get(
                    image_format, ".img"
                )
                digest = hashlib.sha256(body).hexdigest()
                local = job / f"x-media-{media_index}-{digest[:10]}{extension}"
                local.write_bytes(body)
                archived_path, archived_hash = self.workspace.archive_asset(local, "twitter-media")
                evidence = Evidence(
                    id=f"{ingest.candidate.id}-media-{digest[:16]}",
                    candidate_id=ingest.candidate.id,
                    url=media_url,
                    quote=(
                        f"Image attached to the {relationship} X post. "
                        + re.sub(r"\s+", " ", source_text).strip()[:500]
                    ),
                    source_kind="x:media_photo",
                    captured_asset=archived_path,
                    sha256=archived_hash,
                    notes=f"Image attached to the exact {relationship} X post; archived as first-class visual evidence.",
                    metadata={
                        "parent_source_url": parent_url,
                        "post_relationship": relationship,
                        "content_type": content_type,
                        "width": width,
                        "height": height,
                        "visual_role": visual_role,
                        "editorial_priority": "high",
                    },
                )
                self.workspace.save_evidence(evidence)
                ingest.evidence.append(evidence)

    @staticmethod
    def _parse_json_output(output: str):
        stripped = output.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            starts = [index for index, char in enumerate(stripped) if char in "{["]
            for index in reversed(starts):
                try:
                    return json.loads(stripped[index:])
                except json.JSONDecodeError:
                    continue
        raise ValueError("capture command returned no JSON")

    @staticmethod
    def _normalize_x_payload(payload, status_id: str, author: str) -> dict:
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            posts = payload["data"]
        elif isinstance(payload, list):
            posts = payload
        elif isinstance(payload, dict):
            posts = payload.get("tweets") or payload.get("items") or []
        else:
            posts = []
        normalized: list[dict] = []
        for raw in posts:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            identifier = str(item.get("id") or item.get("tweet_id") or "")
            raw_author = item.get("author")
            author_payload = dict(raw_author) if isinstance(raw_author, dict) else {}
            screen_name = str(
                author_payload.get("screenName") or author_payload.get("username")
                or (raw_author if isinstance(raw_author, str) else "")
                or item.get("username") or author
            )
            item["id"] = identifier
            item["text"] = str(item.get("text") or item.get("full_text") or item.get("content") or "")
            item["author"] = {
                **author_payload,
                "screenName": screen_name,
                "name": author_payload.get("name") or item.get("name") or screen_name,
            }
            item.setdefault("createdAtISO", item.get("created_at") or item.get("createdAt"))
            item["metrics"] = item.get("metrics") or {
                key: item.get(key, 0) for key in ("likes", "retweets", "replies", "views")
            }
            if not item.get("media") and item.get("media_urls"):
                item["media"] = [{"type": "video" if str(value).endswith(".mp4") else "photo", "url": value} for value in item["media_urls"]]
            quoted = item.get("quoted_tweet")
            if isinstance(quoted, dict) and not quoted.get("media") and quoted.get("media_urls"):
                quoted["media"] = [
                    {"type": "video" if str(value).endswith(".mp4") else "photo", "url": value}
                    for value in quoted["media_urls"]
                ]
            urls = item.get("urls") or extract_external_urls(item["text"])
            item["urls"] = [value.get("expanded_url") if isinstance(value, dict) else value for value in urls]
            normalized.append(item)

        # Some X backends return the handle in ``name`` for the root while a
        # quoted/sibling post from the same payload carries the real display
        # name. Resolve one canonical display name per handle so cards from
        # the same author cannot alternate between e.g. Tibo/thsottiaux.
        display_names: dict[str, str] = {}

        def remember_display_name(handle: object, name: object) -> None:
            normalized_handle = str(handle or "").lstrip("@").casefold()
            display_name = str(name or "").strip()
            if not normalized_handle or not display_name:
                return
            if display_name.lstrip("@").casefold() == normalized_handle:
                return
            display_names.setdefault(normalized_handle, display_name)

        for item in normalized:
            item_author = item.get("author") or {}
            if isinstance(item_author, dict):
                remember_display_name(item_author.get("screenName"), item_author.get("name"))
            quoted = item.get("quoted_tweet")
            if isinstance(quoted, dict):
                quoted_author = quoted.get("author")
                quoted_handle = (
                    quoted_author.get("screenName") or quoted_author.get("username")
                    if isinstance(quoted_author, dict) else quoted_author
                )
                remember_display_name(quoted_handle, quoted.get("name"))

        for item in normalized:
            item_author = item.get("author") or {}
            if not isinstance(item_author, dict):
                continue
            handle_key = str(item_author.get("screenName") or "").lstrip("@").casefold()
            if handle_key in display_names:
                item["author"] = {**item_author, "name": display_names[handle_key]}
        root = next((item for item in normalized if item["id"] == status_id), None)
        if not root:
            return {"ok": False, "data": []}
        # Keep the exact post first. Only posts explicitly carrying the same
        # conversation id may be appended; a user's timeline is not a thread.
        conversation = str(root.get("conversationId") or root.get("conversation_id") or "")
        thread = [root]
        if conversation:
            thread.extend(item for item in normalized if item is not root and str(item.get("conversationId") or item.get("conversation_id") or "") == conversation)
        else:
            root_text = root.get("text", "")
            entities = {
                value.casefold() for value in re.findall(r"@[A-Za-z0-9_]+|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+", root_text)
                if len(value) >= 5
            }
            related: list[dict] = []
            for item in normalized:
                if item is root or item.get("isRetweet") or item.get("is_retweet"):
                    continue
                text = str(item.get("text") or "")
                folded = text.casefold()
                if any(entity in folded for entity in entities):
                    context = dict(item)
                    context["context_only"] = True
                    related.append(context)
                if len(related) == 3:
                    break
            thread.extend(related)
        return {"ok": True, "schema_version": "1", "data": thread}

    def _acquire_web(self, url: str, job: Path) -> AcquisitionResult:
        content, content_type, method = self._fetch_readable(url)
        page = job / "page.md"
        if "html" in content_type:
            parser = _TextExtractor()
            parser.feed(content.decode("utf-8", errors="replace"))
            text = parser.text()
            title = parser.title or urlparse(url).hostname or "Official page"
        else:
            text = content.decode("utf-8", errors="replace")
            title_match = re.search(r"^\s*(?:Title:\s*)?(.+)$", text, re.MULTILINE)
            title = (title_match.group(1).strip() if title_match else urlparse(url).hostname) or "Official page"
        page.write_text(text + "\n", encoding="utf-8")
        ingest = WebPageIngestor().ingest(url, page, self.workspace, title)
        linked = extract_external_urls(text)
        ingest.candidate.linked_sources = [item for item in dict.fromkeys(linked) if item != url][:12]
        self.workspace.save_candidate(ingest.candidate)
        return AcquisitionResult(ingest, page, method)

    def _acquire_pdf(self, url: str, job: Path) -> AcquisitionResult:
        data = self._fetch(url)[0]
        pdf = job / "source.pdf"
        pdf.write_bytes(data)
        text_path = job / "source.txt"
        if shutil.which("pdftotext"):
            subprocess.run(["pdftotext", "-layout", str(pdf), str(text_path)], check=True, timeout=120)
            text = text_path.read_text(encoding="utf-8", errors="replace")
            method = "pdf+pdftotext"
        else:
            try:
                from pypdf import PdfReader
            except ImportError as error:
                raise RuntimeError(
                    "PDF text extraction requires the declared pypdf dependency or the pdftotext binary"
                ) from error
            pages: list[str] = []
            for page_number, page in enumerate(PdfReader(str(pdf)).pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    pages.append(f"--- PDF page {page_number} ---\n{page_text}")
            text = "\n\n".join(pages)
            if len(text.strip()) < 200:
                raise RuntimeError("PDF contains no usable text; OCR or multimodal extraction is required")
            text_path.write_text(text + "\n", encoding="utf-8")
            method = "pdf+pypdf"
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        candidate = Candidate(
            id=f"paper-{digest}", source_type=SourceType.PAPER, source_url=url,
            title=next((line.strip() for line in text.splitlines() if line.strip()), "Technical paper")[:160],
            dedupe_key=f"paper:{url}", metadata={"pdf_file": pdf.name},
        )
        pdf_asset, pdf_hash = self.workspace.archive_asset(pdf, "papers")
        text_asset, _ = self.workspace.archive_asset(text_path, "paper-text")
        evidence = [Evidence(
            id=f"{candidate.id}-paper", candidate_id=candidate.id, url=url, quote=text,
            source_kind="paper:pdf", captured_asset=pdf_asset, sha256=pdf_hash,
            notes=f"Extracted text archived at {text_asset}; claims must retain PDF page/figure context.",
        )]
        candidate.linked_sources = [item for item in extract_external_urls(text) if item != url][:12]
        self.workspace.save_candidate(candidate)
        self.workspace.save_evidence(evidence[0])
        return AcquisitionResult(IngestResult(candidate, evidence, []), pdf, method)

    def _fetch_readable(self, url: str) -> tuple[bytes, str, str]:
        if os.environ.get("VIDEO_FACTORY_DISABLE_JINA", "0") != "1":
            jina_url = "https://r.jina.ai/http://" + url.split("://", 1)[-1]
            try:
                body, content_type = self._fetch(jina_url)
                if len(body) > 200:
                    return body, content_type or "text/plain", "jina-reader"
            except Exception:
                pass
        body, content_type = self._fetch(url)
        return body, content_type, "direct-http"

    @staticmethod
    def _fetch(url: str) -> tuple[bytes, str]:
        request = Request(url, headers={"User-Agent": "video-factory/0.1", "Accept": "text/html,text/plain,application/pdf,*/*"})
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=45) as response:
                    return response.read(), response.headers.get_content_type()
            except HTTPError as error:
                if error.code not in {408, 429, 500, 502, 503, 504}:
                    raise
                last_error = error
            except (URLError, OSError, http.client.HTTPException) as error:
                last_error = error
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"media/source download failed after 3 attempts: {last_error}") from last_error
