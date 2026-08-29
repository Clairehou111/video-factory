from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from urllib.request import Request, build_opener


@dataclass(frozen=True, slots=True)
class ResolvedLink:
    original_url: str
    resolved_url: str


class ExternalLinkResolver:
    """Follow a redirect chain without collecting page content or credentials."""

    def __init__(self, opener: Callable[[Request], object] | None = None):
        self._opener = opener or build_opener().open

    def resolve(self, url: str) -> ResolvedLink:
        request = Request(url, method="HEAD", headers={"User-Agent": "video-factory/0.1"})
        try:
            response = self._opener(request)
        except Exception:
            request = Request(url, headers={"Range": "bytes=0-0", "User-Agent": "video-factory/0.1"})
            response = self._opener(request)
        return ResolvedLink(url, getattr(response, "url", url))
