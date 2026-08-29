from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    AttentionStrategy, CaptureCue, ColdOpenBeat, ContentType, ContextEvent, ContextGraph, CueAction, DirectorBrief, EditorialBrief,
    Evidence, EvidenceShot, EvidenceShotKind, GitHubFocusCandidate, GitHubModuleFocus,
    GitHubProjectBrief, GitHubWalkthrough, MaterialRole, RenderManifest, Scene, SelectionReason,
    CollectionItem, CollectionItemKind, EditorialOpportunity, FramingMode, PlatformRender,
    HookSpec, HookStrategy, RenderProfile, RightsReview, SlideTranslation, SourceMediaInfo, SourceRange,
    StoryArcBeat, StoryBeat, StorySubject, SubtitleMode,
    TerminologyEntry, TerminologyStrategy, TopicType, TranscriptCue, VideoCollectionManifest,
)
from .github_editor import canonicalize_github_brief
from .editorial import canonicalize_editorial_brief


def manifest_from_dict(data: dict[str, Any]) -> RenderManifest:
    evidence = [Evidence(**item) for item in data.get("evidence", [])]
    story_beats = [StoryBeat(**item) for item in data.get("story_beats", [])]
    walkthrough_data = data.get("github_walkthrough")
    walkthrough = None if not walkthrough_data else GitHubWalkthrough(
        **{**walkthrough_data, "key_modules": [GitHubModuleFocus(**item) for item in walkthrough_data.get("key_modules", [])]},
    )
    brief_data = data.get("github_brief")
    if brief_data:
        brief_data = dict(brief_data)
        focus_items = []
        for item in brief_data.get("focus_candidates", []):
            focus = dict(item)
            if "editorial_role" not in focus and "role" in focus:
                focus["editorial_role"] = focus.pop("role")
            focus_items.append(GitHubFocusCandidate(**focus))
    brief = None if not brief_data else GitHubProjectBrief(
        **{**brief_data, "focus_candidates": focus_items},
    )
    if brief:
        canonicalize_github_brief(brief, evidence)
    editorial_data = data.get("editorial_brief")
    editorial_brief = None
    if editorial_data:
        editorial_data = dict(editorial_data)
        opportunity_data = editorial_data.get("opportunity")
        opportunity = None if not opportunity_data else EditorialOpportunity(
            **{**opportunity_data, "selection_reasons": [SelectionReason(**item) for item in opportunity_data.get("selection_reasons", [])]},
        )
        graph_data = editorial_data.get("context_graph")
        context_graph = None if not graph_data else ContextGraph(
            **{**graph_data, "events": [ContextEvent(**item) for item in graph_data.get("events", [])]},
        )
        director_data = editorial_data.get("director_brief")
        director_brief = None if not director_data else DirectorBrief(
            **{**director_data, "story_arc": [StoryArcBeat(**item) for item in director_data.get("story_arc", [])]},
        )
        editorial_brief = EditorialBrief(
            headline=str(editorial_data.get("headline", "")),
            subheadline=str(editorial_data.get("subheadline", "")),
            fixed_conclusion=str(editorial_data.get("fixed_conclusion", "")),
            attention_strategy=AttentionStrategy(**editorial_data["attention_strategy"]),
            subjects=[StorySubject(**item) for item in editorial_data.get("subjects", [])],
            context_events=[ContextEvent(**item) for item in editorial_data.get("context_events", [])],
            evidence_shots=[EvidenceShot(**{**item, "kind": EvidenceShotKind(item["kind"])}) for item in editorial_data.get("evidence_shots", [])],
            duration_target=float(editorial_data.get("duration_target", 0)),
            opportunity=opportunity,
            context_graph=context_graph,
            director_brief=director_brief,
        )
        canonicalize_editorial_brief(editorial_brief, evidence)
    scenes = []
    for item in data.get("scenes", []):
        scene = dict(item)
        scene["material_role"] = MaterialRole(scene["material_role"])
        scene["recording_cues"] = [CaptureCue(**{**cue, "action": CueAction(cue["action"])}) for cue in scene.get("recording_cues", [])]
        scenes.append(Scene(**scene))
    cold_open_beats = [ColdOpenBeat(**item) for item in data.get("cold_open_beats", [])]
    values = {key: value for key, value in data.items() if key not in {"evidence", "scenes", "story_beats", "github_walkthrough", "github_brief", "editorial_brief", "cold_open_beats", "topic_type", "content_type"}}
    manifest = RenderManifest(
        **values,
        content_type=ContentType(data["content_type"]),
        topic_type=TopicType(data["topic_type"]) if data.get("topic_type") else None,
        evidence=evidence,
        scenes=scenes,
        story_beats=story_beats,
        github_walkthrough=walkthrough,
        github_brief=brief,
        editorial_brief=editorial_brief,
        cold_open_beats=cold_open_beats,
    )
    if manifest.github_brief:
        repo_url = next((url for url in manifest.source_urls if "github.com/" in url), "")
        repo_name = repo_url.rstrip("/").rsplit("/", 1)[-1] if repo_url else ""
        current_title = (manifest.fixed_title or manifest.github_brief.project_title).strip()
        if repo_name and repo_name.casefold() not in current_title.casefold():
            manifest.fixed_title = f"{repo_name}｜{current_title}"
    elif manifest.editorial_brief:
        strategy = manifest.editorial_brief.attention_strategy
        selected = strategy.selected_hook or (strategy.hook_candidates[0] if strategy.hook_candidates else manifest.fixed_hook)
        manifest.fixed_hook = selected
        manifest.fixed_title = selected
        manifest.fixed_footer = manifest.editorial_brief.fixed_conclusion
    return manifest


def load_manifest(path: Path) -> RenderManifest:
    return manifest_from_dict(json.loads(path.read_text(encoding="utf-8")))


def collection_manifest_from_dict(data: dict[str, Any]) -> VideoCollectionManifest:
    def hook_from_dict(item: dict[str, Any]) -> HookSpec:
        source_range = item.get("source_range") or {}
        return HookSpec(**{
            **{key: value for key, value in item.items() if key not in {"strategy", "source_range"}},
            "strategy": HookStrategy(item["strategy"]),
            "source_range": SourceRange(**{
                **source_range,
                "framing": FramingMode(source_range.get("framing", "auto")),
            }),
        })

    items: list[CollectionItem] = []
    for raw_item in data.get("items", []):
        ranges = [
            SourceRange(**{**item, "framing": FramingMode(item.get("framing", "auto"))})
            for item in raw_item.get("source_ranges", [])
        ]
        renders = []
        for item in raw_item.get("renders", []):
            selected = item.get("selected_hook")
            renders.append(PlatformRender(**{
                **{key: value for key, value in item.items() if key not in {
                    "profile", "subtitle_mode", "hook_candidates", "selected_hook",
                    "slide_translations",
                }},
                "profile": RenderProfile(item["profile"]),
                "subtitle_mode": SubtitleMode(item.get("subtitle_mode", "bilingual_stacked")),
                "hook_candidates": [hook_from_dict(hook) for hook in item.get("hook_candidates", [])],
                "selected_hook": hook_from_dict(selected) if selected else None,
                "slide_translations": [
                    SlideTranslation(**row) for row in item.get("slide_translations", [])
                ],
            }))
        items.append(CollectionItem(
            **{
                **{key: value for key, value in raw_item.items() if key not in {"kind", "source_ranges", "renders"}},
                "kind": CollectionItemKind(raw_item["kind"]),
                "source_ranges": ranges,
                "renders": renders,
            },
        ))
    terminology = [
        TerminologyEntry(**{**item, "strategy": TerminologyStrategy(item["strategy"])})
        for item in data.get("terminology", [])
    ]
    return VideoCollectionManifest(
        **{
            **{key: value for key, value in data.items() if key not in {
                "transcript", "terminology", "items", "rights_review", "source_media_info",
            }},
            "transcript": [TranscriptCue(**item) for item in data.get("transcript", [])],
            "terminology": terminology,
            "items": items,
            "rights_review": RightsReview(**(data.get("rights_review") or {})),
            "source_media_info": SourceMediaInfo(**data["source_media_info"]) if data.get("source_media_info") else None,
        },
    )


def load_collection_manifest(path: Path) -> VideoCollectionManifest:
    return collection_manifest_from_dict(json.loads(path.read_text(encoding="utf-8")))
