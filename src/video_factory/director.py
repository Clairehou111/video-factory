from __future__ import annotations

from dataclasses import dataclass, field

from .models import (
    Candidate, CaptureCue, ColdOpenBeat, ContentType, CueAction, EditorialBrief, Evidence,
    GitHubProjectBrief, GitHubWalkthrough, MaterialRole, RenderManifest, Scene, StoryBeat, TopicType,
)
from .github_editor import compose_github_hook, validate_github_brief
from .narrative import requirements_for


@dataclass(slots=True)
class NarrativeAnswer:
    beat_id: str
    answer: str
    evidence_ids: list[str]


@dataclass(slots=True)
class SceneProposal:
    stage_name: str
    narration: str
    caption: str
    material_role: MaterialRole
    visual_action: str
    evidence_ids: list[str]
    beat_ids: list[str]
    recording_cues: list[CaptureCue] = field(default_factory=list)
    overlay_labels: list[str] = field(default_factory=list)
    sound_hint: str | None = None
    duration_hint: float | None = None
    screen_fact: str | None = None
    screen_interpretation: str | None = None
    highlight_translation: str | None = None
    source_excerpt: str | None = None
    visual_family: str = ""
    retention_job: str = ""


@dataclass(slots=True)
class StoryboardRequest:
    id: str
    candidate: Candidate
    topic_type: TopicType
    content_type: ContentType
    evidence: list[Evidence]
    footer: str
    answers: list[NarrativeAnswer]
    scenes: list[SceneProposal]
    target_duration: float
    github_walkthrough: GitHubWalkthrough | None = None
    github_brief: GitHubProjectBrief | None = None
    editorial_brief: EditorialBrief | None = None
    fixed_hook: str = ""


class StoryboardDirector:
    """Deterministic guardrails around an editorial/LLM scene proposal.

    It does not invent facts. A future script model may propose the answers and
    scenes, but this service schedules them only after their evidence and
    topic-specific narrative contract are present.
    """

    # No-voice videos still need a stable visual cadence.  Text density is
    # checked independently from this scheduling fallback.
    words_per_second = 7.0

    def direct(self, request: StoryboardRequest) -> RenderManifest:
        evidence_ids = {item.id for item in request.evidence}
        answer_map = {answer.beat_id: answer for answer in request.answers}
        required = requirements_for(request.topic_type, request.content_type)
        missing = [requirement.id for requirement in required if requirement.id not in answer_map]
        if missing:
            raise ValueError(f"missing narrative answers: {', '.join(missing)}")
        if any(not answer.answer.strip() or not set(answer.evidence_ids) <= evidence_ids for answer in request.answers):
            raise ValueError("every narrative answer needs text and valid evidence ids")
        if not request.footer.strip():
            raise ValueError("a fixed event conclusion is required")
        if not request.scenes:
            raise ValueError("at least one scene is required")
        if request.topic_type.value == "github_project":
            if request.github_brief:
                brief_errors = validate_github_brief(request.github_brief, request.evidence)
                if brief_errors:
                    raise ValueError("invalid GitHub project brief: " + "; ".join(brief_errors))
            else:
                self._validate_github_walkthrough(request.github_walkthrough, request.scenes, evidence_ids, request.candidate.source_url)
        elif request.editorial_brief:
            from .editorial import validate_editorial_structure

            brief_errors = validate_editorial_structure(
                request.editorial_brief, request.candidate, request.evidence,
                request.topic_type, request.content_type,
            )
            if brief_errors:
                raise ValueError("invalid editorial brief: " + "; ".join(brief_errors))
        # Hand-authored legacy StoryboardRequest objects remain loadable. The
        # production LLM adapter, however, only accepts EditorialBrief for all
        # non-GitHub topics and never exposes Scene fields to the model.

        self._pad_editorial_runtime(request)

        scenes: list[Scene] = []
        cursor = 0.0
        for index, proposal in enumerate(request.scenes, start=1):
            if not proposal.caption.strip() or not proposal.visual_action.strip():
                raise ValueError(f"scene {index} has no single readable message or visual action")
            screen_fact = (proposal.screen_fact or proposal.caption).strip()
            # A second audience line is optional. Never fall back to narration:
            # narration is commonly an internal director note in BGM-only mode.
            screen_interpretation = (proposal.screen_interpretation or "").strip()
            if not screen_fact:
                raise ValueError(f"scene {index} must contain a self-explanatory visible fact")
            if "highlight" in proposal.visual_action.casefold() and not (proposal.highlight_translation or "").strip():
                raise ValueError(f"scene {index} highlights source language but has no Chinese translation")
            if not set(proposal.evidence_ids) <= evidence_ids:
                raise ValueError(f"scene {index} references unknown evidence")
            duration = proposal.duration_hint or max(2.0, len(proposal.narration) / self.words_per_second)
            if duration > 5.0:
                raise ValueError(f"scene {index} is {duration:.1f}s; split it so visual change occurs within 5 seconds")
            scenes.append(Scene(
                id=f"scene-{index}", start=round(cursor, 3), end=round(cursor + duration, 3),
                narration=proposal.narration, caption=proposal.caption, evidence_ids=proposal.evidence_ids,
                material_role=proposal.material_role, visual_action=proposal.visual_action,
                overlay_labels=proposal.overlay_labels, stage_name=proposal.stage_name,
                sound_hint=proposal.sound_hint, recording_cues=proposal.recording_cues,
                screen_fact=screen_fact, screen_interpretation=screen_interpretation,
                highlight_translation=proposal.highlight_translation,
                source_excerpt=proposal.source_excerpt,
                visual_family=proposal.visual_family,
                retention_job=proposal.retention_job,
            ))
            cursor += duration
        # Decimal scene schedules such as 3.2 + 4 * 2.2 can be represented as
        # 12.000000000000002. Treat that as the requested 12.0 seconds rather
        # than sending a valid story back through another expensive LLM loop.
        if cursor > request.target_duration + 0.001:
            raise ValueError(f"proposal needs {cursor:.1f}s but target is {request.target_duration:.1f}s; choose a longer format instead of deleting causality")
        if scenes[0].start > 0 or not scenes[0].caption.strip():
            raise ValueError("the first scene must state the topic, source, or result immediately")

        fixed_hook = (
            compose_github_hook(request.github_brief)
            if request.topic_type.value == "github_project" and request.github_brief
            else request.fixed_hook.strip() or scenes[0].caption
        )
        cold_open_beats: list[ColdOpenBeat] = []
        fixed_title = fixed_hook
        if request.github_brief:
            if all((request.github_brief.hook_opening, request.github_brief.hook_reveal, request.github_brief.hook_verdict)):
                cold_open_beats = [
                    ColdOpenBeat("cold-open-event", request.github_brief.hook_opening, "event_hook", 1.1, request.github_brief.hook_evidence_ids),
                    ColdOpenBeat("cold-open-reveal", request.github_brief.hook_reveal, "capability_reveal", 1.2, request.github_brief.hook_evidence_ids),
                    ColdOpenBeat("cold-open-verdict", request.github_brief.hook_verdict, "editorial_verdict", 1.3, request.github_brief.hook_evidence_ids),
                ]
            else:  # archived manifests only
                fact = request.github_brief.hook_fact.strip()
                split_at = next((fact.find(mark) for mark in ("，", "；", "但", "却") if mark in fact), -1)
                first_fact, second_fact = (fact[:split_at], fact[split_at + 1:]) if split_at >= 0 else (fact[:len(fact)//2], fact[len(fact)//2:])
                cold_open_beats = [
                    ColdOpenBeat("cold-open-reaction", request.github_brief.hook_stance, "reaction", 1.0, request.github_brief.hook_evidence_ids),
                    ColdOpenBeat("cold-open-conflict", first_fact, "conflict", 1.0, request.github_brief.hook_evidence_ids),
                    ColdOpenBeat("cold-open-payoff", second_fact, "payoff", 1.2, request.github_brief.hook_evidence_ids),
                ]
            repo_name = request.candidate.title.rsplit("/", 1)[-1].strip()
            proposed_title = request.github_brief.project_title.strip()
            fixed_title = proposed_title if repo_name.casefold() in proposed_title.casefold() else f"{repo_name}｜{proposed_title}"
        elif request.editorial_brief:
            fixed_hook = request.editorial_brief.attention_strategy.selected_hook or request.editorial_brief.attention_strategy.hook_candidates[0]
            fixed_title = fixed_hook
        return RenderManifest(
            id=request.id, candidate_id=request.candidate.id, content_type=request.content_type,
            scenes=scenes, evidence=request.evidence, source_urls=[request.candidate.source_url, *request.candidate.linked_sources],
            topic_type=request.topic_type,
            story_beats=[StoryBeat(answer.beat_id, answer.answer, answer.evidence_ids) for answer in request.answers],
            github_walkthrough=request.github_walkthrough,
            github_brief=request.github_brief,
            editorial_brief=request.editorial_brief,
            fixed_hook=fixed_hook,
            fixed_title=fixed_title,
            cold_open_beats=cold_open_beats,
            fixed_footer=request.footer,
        )

    @staticmethod
    def _pad_editorial_runtime(request: StoryboardRequest) -> None:
        """Extend evidence holds to the format floor without inventing copy.

        A semantic copy repair can legitimately shorten a deep-dive while
        leaving every selected proof intact.  The director owns timing, so it
        distributes that small gap across the existing shots (never beyond
        the five-second cadence ceiling) and persists the durations back into
        the editorial brief so a later rerender compiles the same schedule.
        """
        if request.content_type == ContentType.FLASH:
            return
        floor = 15.0 if request.content_type == ContentType.EXPLAINER else 25.0
        durations = [float(scene.duration_hint or max(2.0, len(scene.narration) / 7.0)) for scene in request.scenes]
        total = sum(durations)
        if total >= floor - 0.001:
            return
        if request.target_duration < floor or sum(max(0.0, 5.0 - value) for value in durations) < floor - total - 0.001:
            return

        remaining = floor - total
        for index, (proposal, duration) in enumerate(zip(request.scenes, durations)):
            scenes_left = len(durations) - index
            addition = min(5.0 - duration, remaining / scenes_left)
            if addition <= 0:
                continue
            updated = round(duration + addition, 3)
            actual_addition = updated - duration
            proposal.duration_hint = updated
            remaining -= actual_addition

            timed = {
                CueAction.WAIT, CueAction.SCROLL, CueAction.HIGHLIGHT, CueAction.ZOOM,
            }
            cue = next((item for item in reversed(proposal.recording_cues) if item.action in timed), None)
            if cue is not None:
                cue.wait_ms = int(cue.wait_ms or 1000) + round(actual_addition * 1000)

            if request.editorial_brief and index < len(request.editorial_brief.evidence_shots):
                request.editorial_brief.evidence_shots[index].duration = updated

    @staticmethod
    def _validate_github_walkthrough(
        walkthrough: GitHubWalkthrough | None, scenes: list[SceneProposal], evidence_ids: set[str], repo_url: str,
    ) -> None:
        if walkthrough is None:
            raise ValueError("GitHub walkthrough is required: repo home, file tree, README opening, and two modules")
        indices = [
            walkthrough.repo_home_scene_index, walkthrough.file_tree_scene_index,
            walkthrough.readme_entry_scene_index, walkthrough.readme_end_scene_index,
        ]
        if any(index < 1 or index > len(scenes) for index in indices):
            raise ValueError("GitHub walkthrough scene indexes must reference real scenes")
        if len(walkthrough.key_modules) != 2:
            raise ValueError("GitHub walkthrough requires exactly two key modules")
        purposes = {module.purpose for module in walkthrough.key_modules}
        if not purposes & {"minimal_usable_example", "core_usage"}:
            raise ValueError("one GitHub module must show the minimum usable example or core usage")
        if not purposes & {"architecture", "implementation", "benchmark", "limitation", "roadmap"}:
            raise ValueError("one GitHub module must show architecture, implementation, benchmark, limitation, or roadmap")
        for module in walkthrough.key_modules:
            if not module.anchor.strip() or module.scene_index < 1 or module.scene_index > len(scenes):
                raise ValueError("GitHub module must use a real scene and visible page anchor")
            if not set(module.evidence_ids) <= evidence_ids:
                raise ValueError("GitHub module references unknown evidence")
            scene = scenes[module.scene_index - 1]
            rendered_text = " ".join([scene.stage_name, scene.caption, scene.visual_action, scene.narration]).casefold()
            if module.anchor.casefold() not in rendered_text:
                raise ValueError(f"GitHub module scene does not actually present its anchor: {module.anchor}")
        # The executor owns the required live-browser path (repo home → file
        # tree → README opening → two selected anchors). A separate audit
        # capture may traverse the full README, but the final cut must not.
        # The model chooses only two real anchors and their evidence-backed
        # explanation.
