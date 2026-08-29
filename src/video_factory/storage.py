from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from .models import Candidate, Evidence, RenderManifest, SourceType, VideoCollectionManifest

if TYPE_CHECKING:
    from .publish import PublishBatch


class Workspace:
    """Filesystem asset store with a small SQLite index; no cloud dependency."""

    def __init__(self, root: Path):
        self.root = root
        self.assets_dir = root / "assets"
        self.manifests_dir = root / "manifests"
        self.collections_dir = self.manifests_dir / "collections"
        self.renders_dir = root / "renders"
        self.publish_dir = root / "publish"
        self.db_path = root / "index.sqlite3"

    def initialize(self) -> None:
        for directory in (
            self.root, self.assets_dir, self.manifests_dir, self.collections_dir,
            self.renders_dir, self.publish_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    kind TEXT NOT NULL,
                    id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (kind, id)
                )
            """)
            db.commit()

    def save_candidate(self, candidate: Candidate) -> None:
        self._save("candidate", candidate.id, asdict(candidate))

    def save_evidence(self, evidence: Evidence) -> None:
        self._save("evidence", evidence.id, asdict(evidence))

    def save_discovery_candidate(self, payload: dict) -> None:
        self._save("discovery_candidate", str(payload["id"]), payload)

    def load_discovery_candidate(self, identifier: str) -> dict:
        return self._load("discovery_candidate", identifier)

    def save_discovery_run(self, identifier: str, payload: dict) -> Path:
        self._save("discovery_run", identifier, payload)
        directory = self.root / "discovery" / "runs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{identifier}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def load_discovery_state(self) -> dict:
        """Load unified state and import the legacy YouTube history once."""
        self.initialize()
        try:
            return self._load("discovery_state", "current")
        except KeyError:
            state: dict = {"channels": {}, "generated_events": [], "history": [], "skipped_ids": []}
            legacy = self.root / "youtube-discovery-state.json"
            if legacy.is_file():
                payload = json.loads(legacy.read_text(encoding="utf-8"))
                state["channels"]["youtube"] = {
                    "next_run_at": payload.get("next_run_at", ""),
                    "legacy_selected_ids": payload.get("selected_ids", []),
                    "legacy_selected_titles": payload.get("selected_titles", []),
                }
                for title in payload.get("selected_titles", []):
                    state["generated_events"].append({
                        "title": str(title), "url": "", "published_at": "",
                        "generated_at": payload.get("last_completed_at", ""),
                        "event_key": "legacy:youtube",
                    })
            return state

    def save_discovery_state(self, payload: dict) -> None:
        self._save("discovery_state", "current", payload)
        directory = self.root / "discovery"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "state.json"
        temporary = directory / "state.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def save_manifest(self, manifest: RenderManifest) -> Path:
        self._save("manifest", manifest.id, manifest.to_dict())
        path = self.manifests_dir / f"{manifest.id}.json"
        path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def save_collection_manifest(self, manifest: VideoCollectionManifest) -> Path:
        self._save("collection_manifest", manifest.id, manifest.to_dict())
        path = self.collections_dir / f"{manifest.id}.json"
        path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def load_collection_manifest(self, identifier: str) -> VideoCollectionManifest:
        from .serde import collection_manifest_from_dict

        return collection_manifest_from_dict(self._load("collection_manifest", identifier))

    def save_publish_batch(self, batch: PublishBatch) -> Path:
        payload = batch.to_dict()
        self._save("publish_batch", batch.id, payload)
        batch_dir = self.publish_dir / batch.id
        batch_dir.mkdir(parents=True, exist_ok=True)
        path = batch_dir / "batch.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def load_publish_batch(self, identifier: str) -> PublishBatch:
        from .publish import PublishBatch

        payload = self._load("publish_batch", identifier)
        if payload.get("batch_type") == "collection":
            from .collection_publish import CollectionPublishBatch

            return CollectionPublishBatch.from_dict(payload)  # type: ignore[return-value]
        return PublishBatch.from_dict(payload)

    def append_publish_attempt(
        self, batch_id: str, platform: str, action: str, payload: dict,
    ) -> Path:
        self.initialize()
        attempts_dir = self.publish_dir / batch_id / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        sequence = len(list(attempts_dir.glob("*.json"))) + 1
        path = attempts_dir / f"{sequence:04d}-{platform}-{action}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def archive_asset(self, source: Path, category: str, name: str | None = None) -> tuple[str, str]:
        if not source.is_file():
            raise FileNotFoundError(source)
        hasher = hashlib.sha256()
        with source.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        target_dir = self.assets_dir / category / digest[:12]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / (name or source.name)
        shutil.copy2(source, target)
        return str(target.relative_to(self.root)), digest

    def load_candidate(self, identifier: str) -> Candidate:
        payload = self._load("candidate", identifier)
        payload["source_type"] = SourceType(payload["source_type"])
        return Candidate(**payload)

    def candidate_for_source_url(self, source_url: str) -> Candidate | None:
        """Return an already archived root candidate without touching the network."""
        target = source_url.rstrip("/")
        with closing(sqlite3.connect(self.db_path)) as db:
            rows = db.execute("SELECT payload FROM records WHERE kind = 'candidate'").fetchall()
        for (raw,) in rows:
            payload = json.loads(raw)
            if str(payload.get("source_url") or "").rstrip("/") == target:
                payload["source_type"] = SourceType(payload["source_type"])
                return Candidate(**payload)
        return None

    def evidence_for_candidate(self, candidate_id: str) -> list[Evidence]:
        return self.evidence_for_candidates([candidate_id])

    def evidence_for_candidates(self, candidate_ids: list[str]) -> list[Evidence]:
        identifiers = set(candidate_ids)
        with closing(sqlite3.connect(self.db_path)) as db:
            rows = db.execute("SELECT payload FROM records WHERE kind = 'evidence'").fetchall()
        return [Evidence(**payload) for (raw,) in rows if (payload := json.loads(raw)).get("candidate_id") in identifiers]

    def _save(self, kind: str, identifier: str, payload: dict) -> None:
        self.initialize()
        with closing(sqlite3.connect(self.db_path)) as db:
            db.execute(
                "INSERT OR REPLACE INTO records(kind, id, payload, created_at) VALUES (?, ?, ?, datetime('now'))",
                (kind, identifier, json.dumps(payload, ensure_ascii=False)),
            )
            db.commit()

    def _load(self, kind: str, identifier: str) -> dict:
        self.initialize()
        with closing(sqlite3.connect(self.db_path)) as db:
            row = db.execute("SELECT payload FROM records WHERE kind = ? AND id = ?", (kind, identifier)).fetchone()
        if not row:
            raise KeyError(f"no {kind} record: {identifier}")
        return json.loads(row[0])
