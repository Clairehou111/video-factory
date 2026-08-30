from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class SourceType(StrEnum):
    TWEET = "tweet"
    WEB = "web"
    GITHUB = "github"
    PAPER = "paper"
    OFFICIAL_ANNOUNCEMENT = "official_announcement"
    YOUTUBE = "youtube"


class MaterialRole(StrEnum):
    PROOF = "proof"
    ILLUSTRATION = "illustration"
    EXPLANATION = "explanation"
    TRANSITION = "transition"


class CueAction(StrEnum):
    OPEN = "open"
    WAIT = "wait"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    HIGHLIGHT = "highlight"
    ZOOM = "zoom"
    WAIT_FOR = "wait_for"


class ContentType(StrEnum):
    FLASH = "flash"
    EXPLAINER = "explainer"
    DEEP_DIVE = "deep_dive"


class InformationRenderProfile(StrEnum):
    """Presentation profile for generated information shorts.

    ``classic`` remains the compatibility default. ``radar_v2`` opts into
    the evidence-focused layout and motion system without changing archived
    manifests when they are loaded again.
    """

    CLASSIC = "classic"
    RADAR_V2 = "radar_v2"


class OpeningMode(StrEnum):
    DIRECT_FACT = "direct_fact"
    CONFLICT = "conflict"
    COUNTER_INTUITIVE = "counter_intuitive"
    DEVELOPER_ROI = "developer_roi"


class NarrativeBeat(StrEnum):
    OPENING = "opening"
    PROOF = "proof"
    TAKEAWAY = "takeaway"


class TopicType(StrEnum):
    PRACTICE_POST = "practice_post"
    GITHUB_PROJECT = "github_project"
    TOOL_SDK_AGENT = "tool_sdk_agent"
    MODEL_OR_PRODUCT = "model_or_product"
    COMPANY_OR_TEAM = "company_or_team"
    RESEARCH_OR_BENCHMARK = "research_or_benchmark"
    OFFICIAL_ANNOUNCEMENT = "official_announcement"
    LINKED_EXTERNAL_SOURCE = "linked_external_source"
    EXPERT_TALK = "expert_talk"


class TerminologyStrategy(StrEnum):
    TRANSLATE = "translate"
    PRESERVE = "preserve"
    BILINGUAL_ONCE = "bilingual_once"


class CollectionItemKind(StrEnum):
    MAIN = "main"
    EPISODE = "episode"
    BILIBILI_CHAPTER = "bilibili_chapter"
    WECHAT_SHORT = "wechat_short"


class RenderProfile(StrEnum):
    BILIBILI_LANDSCAPE = "bilibili_landscape"
    WECHAT_VERTICAL = "wechat_vertical"


class SubtitleMode(StrEnum):
    BILINGUAL_STACKED = "bilingual_stacked"


class HookStrategy(StrEnum):
    CONTRARIAN = "contrarian"
    PAIN_GAP = "pain_gap"
    OUTCOME = "outcome"
    QUESTION = "question"


class FramingMode(StrEnum):
    AUTO = "auto"
    SPEAKER = "speaker"
    SLIDE = "slide"
    SPLIT = "split"


class EvidenceShotKind(StrEnum):
    """Editorial evidence choices, deliberately separate from render roles."""

    TWEET_CARD = "tweet_card"
    BROWSER_SECTION = "browser_section"
    IMAGE = "image"
    PDF_PAGE = "pdf_page"
    FIGURE = "figure"
    BENCHMARK_CHART = "benchmark_chart"
    CODE_EXAMPLE = "code_example"
    TERMINAL_DEMO = "terminal_demo"


@dataclass(slots=True)
class Candidate:
    id: str
    source_type: SourceType
    source_url: str
    title: str
    author: str | None = None
    linked_sources: list[str] = field(default_factory=list)
    published_at: str | None = None
    dedupe_key: str | None = None
    parent_candidate_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    captured_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class Evidence:
    id: str
    candidate_id: str
    url: str
    quote: str
    source_kind: str
    captured_asset: str | None = None
    captured_at: str = field(default_factory=now_iso)
    sha256: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranscriptCue:
    id: str
    start: float
    end: float
    source_text: str
    translation: str = ""
    speaker: str = ""
    confidence: float | None = None
    # Clip-local YouTube transcripts keep the source timeline for audit/rebuilds.
    original_start: float | None = None
    original_end: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class TerminologyEntry:
    source: str
    strategy: TerminologyStrategy
    target: str = ""
    first_use_explanation: str = ""
    notes: str = ""


@dataclass(slots=True)
class SourceRange:
    start: float
    end: float
    framing: FramingMode = FramingMode.AUTO
    reason: str = ""
    crop_x: int | None = None
    crop_y: int | None = None
    crop_width: int | None = None
    crop_height: int | None = None
    # Set when a remote interview is downloaded as a bounded source clip.
    original_start: float | None = None
    original_end: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def has_explicit_crop(self) -> bool:
        return all(
            value is not None and value >= 0
            for value in (self.crop_x, self.crop_y)
        ) and all(
            value is not None and value > 0
            for value in (self.crop_width, self.crop_height)
        )


@dataclass(slots=True)
class SourceMediaInfo:
    width: int
    height: int
    duration: float
    video_codec: str
    audio_codec: str = ""
    format_id: str = ""
    acquisition_client: str = "mweb"
    sha256: str = ""
    runtime: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HookSpec:
    id: str
    strategy: HookStrategy
    headline_zh: str
    promise: str
    source_range: SourceRange
    source_cue_ids: list[str]
    payoff_cue_ids: list[str]
    speaker_label: str = ""
    motion: str = "push_in"
    persistent_title: bool = True
    selected: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, HookStrategy):
            self.strategy = HookStrategy(self.strategy)
        if not isinstance(self.source_range, SourceRange):
            raw = dict(self.source_range)
            raw["framing"] = FramingMode(raw.get("framing", "auto"))
            self.source_range = SourceRange(**raw)


@dataclass(slots=True)
class SlideTranslation:
    """A source-timed translation of meaningful text embedded in video pixels."""

    start: float
    end: float
    source_text: str
    translation: str
    source_text_bottom: int | None = None
    source_text_center_x: int | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class PlatformRender:
    profile: RenderProfile
    width: int
    height: int
    video_path: str = ""
    subtitle_path: str = ""
    source_subtitle_path: str = ""
    translation_subtitle_path: str = ""
    bilingual_subtitle_path: str = ""
    subtitle_mode: SubtitleMode = SubtitleMode.BILINGUAL_STACKED
    hook_candidates: list[HookSpec] = field(default_factory=list)
    selected_hook: HookSpec | None = None
    slide_translations: list[SlideTranslation] = field(default_factory=list)
    slide_translation_required: bool = False
    cover_path: str = ""
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CollectionItem:
    id: str
    kind: CollectionItemKind
    order: int
    title: str
    thesis: str
    source_ranges: list[SourceRange]
    renders: list[PlatformRender] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(item.duration for item in self.source_ranges)


@dataclass(slots=True)
class RightsReview:
    status: str = "unreviewed"
    basis: str = "educational_noncommercial"
    reviewed_by: str = ""
    reviewed_at: str = ""
    notes: str = ""


@dataclass(slots=True)
class VideoCollectionManifest:
    id: str
    candidate_id: str
    source_url: str
    source_video_id: str
    source_title: str
    source_channel: str
    collection_title: str
    transcript: list[TranscriptCue]
    terminology: list[TerminologyEntry]
    items: list[CollectionItem]
    editorial_mode: str = "study"
    source_media_path: str = ""
    source_subtitle_path: str = ""
    source_duration: float = 0.0
    source_media_info: SourceMediaInfo | None = None
    supersedes_collection_id: str = ""
    superseded_by_collection_id: str = ""
    rights_review: RightsReview = field(default_factory=RightsReview)
    quality_checks: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CaptureCue:
    action: CueAction
    instruction: str
    target: str | None = None
    selector: str | None = None
    value: str | None = None
    wait_ms: int | None = None
    shot_id: str | None = None
    translation: str | None = None


@dataclass(slots=True)
class Scene:
    id: str
    start: float
    end: float
    narration: str
    caption: str
    evidence_ids: list[str]
    material_role: MaterialRole
    visual_action: str
    asset_query: str | None = None
    asset_path: str | None = None
    pointer_track: list[dict[str, float]] = field(default_factory=list)
    zoom_track: list[dict[str, float]] = field(default_factory=list)
    overlay_labels: list[str] = field(default_factory=list)
    stage_name: str | None = None
    sound_hint: str | None = None
    recording_cues: list[CaptureCue] = field(default_factory=list)
    # The factory is intentionally usable without a narrator.  These fields
    # are rendered as readable on-screen copy, rather than inferred from TTS.
    screen_fact: str | None = None
    screen_interpretation: str | None = None
    highlight_translation: str | None = None
    # Exact source-language excerpt selected by the editor for a derived
    # evidence card. It is audience evidence, unlike narration/director notes.
    source_excerpt: str | None = None
    # Semantic presentation choices selected by the editor. They are not raw
    # browser/media commands: the renderer maps the small allow-list below to
    # deterministic treatments.
    visual_family: str = ""
    retention_job: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class StoryBeat:
    """One answer to a topic-specific editorial question, tied to evidence."""

    id: str
    answer: str
    evidence_ids: list[str]
    scene_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GitHubModuleFocus:
    scene_index: int
    anchor: str
    purpose: str
    evidence_ids: list[str]


@dataclass(slots=True)
class GitHubWalkthrough:
    repo_home_scene_index: int
    file_tree_scene_index: int
    readme_entry_scene_index: int
    readme_end_scene_index: int
    key_modules: list[GitHubModuleFocus]


@dataclass(slots=True)
class GitHubFocusCandidate:
    """One exact, browser-visible proof candidate proposed by the editor."""

    id: str
    editorial_role: str
    target: str
    summary: str
    why_it_matters: str
    evidence_ids: list[str]
    translation: str = ""
    source_url: str | None = None
    browser_target: str = ""
    browser_translation: str = ""
    viewer_value: int = 0
    visual_proof: int = 0
    distinctiveness: int = 0
    actionability: int = 0
    risk_importance: int = 0
    redundancy_penalty: int = 0

    @property
    def score(self) -> int:
        return (
            self.viewer_value * 3
            + self.visual_proof * 3
            + self.distinctiveness * 2
            + self.actionability * 2
            + self.risk_importance
            - self.redundancy_penalty * 3
        )


@dataclass(slots=True)
class GitHubProjectBrief:
    """Evidence-backed editorial analysis; the recorder never invents it."""

    project_kind: str
    core_job: str
    input_output: str
    adoption_path: str
    unique_edge: str
    boundary: str
    verdict: str
    repo_description_target: str
    readme_claim_target: str
    file_tree_target: str
    focus_candidates: list[GitHubFocusCandidate]
    repo_description_translation: str = ""
    readme_claim_translation: str = ""
    selected_focus_ids: list[str] = field(default_factory=list)
    # Keep the editorial reaction separate from the verifiable claim.  The
    # renderer composes these fields deterministically instead of trusting a
    # model-written blob that can quietly collapse back into a repo summary.
    hook_strategy: str = ""
    hook_stance: str = ""
    hook_fact: str = ""
    hook_evidence_ids: list[str] = field(default_factory=list)
    project_title: str = ""
    # The model writes three independently meaningful beats. Legacy
    # hook_stance/hook_fact remain readable for archived manifests only.
    hook_opening: str = ""
    hook_reveal: str = ""
    hook_verdict: str = ""
    # Explicit story entities prevent technically correct but ownerless copy
    # such as "now one command can do it".  Background is optional: it is
    # populated when the repository responds to a named vendor/person/event.
    subject_name: str = ""
    subject_type: str = "project"
    subject_action: str = ""
    subject_consequence: str = ""
    background_actor: str = ""
    background_action: str = ""
    background_consequence: str = ""
    background_evidence_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AttentionStrategy:
    """The factual tension and editorial payoff that earn the first second."""

    hook_fact: str
    conflict: str
    surprise: str
    stakes: str
    stance: str
    payoff: str
    hook_candidates: list[str]
    hook_evidence_ids: list[str]
    selected_hook: str = ""


@dataclass(slots=True)
class StorySubject:
    name: str
    subject_type: str
    action: str
    consequence: str
    evidence_ids: list[str]


@dataclass(slots=True)
class ContextEvent:
    actor: str
    action: str
    occurred_at: str
    relation: str
    evidence_ids: list[str]
    id: str = ""


@dataclass(slots=True)
class SelectionReason:
    id: str
    dimension: str
    rationale: str
    evidence_ids: list[str]


@dataclass(slots=True)
class EditorialOpportunity:
    """Why this item deserves scarce channel attention."""

    event_claim: str
    why_now: str
    why_audience: str
    audience_pain_or_desire: str
    selection_reasons: list[SelectionReason]
    context_hypotheses: list[str] = field(default_factory=list)
    context_gaps: list[str] = field(default_factory=list)
    story_archetype: str = ""


@dataclass(slots=True)
class ContextGraph:
    """Evidence-linked background that changes how the root event is understood."""

    events: list[ContextEvent] = field(default_factory=list)
    required_context_ids: list[str] = field(default_factory=list)
    pattern_context_ids: list[str] = field(default_factory=list)
    discarded_context_ids: list[str] = field(default_factory=list)
    expansion_dimensions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StoryArcBeat:
    role: str
    claim: str
    why_here: str
    evidence_ids: list[str]
    selection_reason_ids: list[str] = field(default_factory=list)
    context_event_ids: list[str] = field(default_factory=list)
    suggested_visual: str = ""


@dataclass(slots=True)
class DirectorBrief:
    """Semantic direction only; browser and media mechanics stay deterministic."""

    editorial_thesis: str
    viewer_tension: str
    emotion: str
    emotion_intensity: int
    selected_context_ids: list[str]
    story_arc: list[StoryArcBeat]
    recommended_duration: float
    attention_trigger: str = ""


@dataclass(slots=True)
class EvidenceShot:
    """What the editor wants to prove; a compiler owns how it is rendered."""

    id: str
    kind: EvidenceShotKind
    question: str
    fact: str
    interpretation: str
    evidence_ids: list[str]
    beat_ids: list[str]
    source_url: str = ""
    target: str = ""
    translation: str = ""
    duration: float = 3.0
    relation_to_previous: str = ""
    visual_family: str = ""
    retention_job: str = ""
    selection_reason_ids: list[str] = field(default_factory=list)
    context_event_ids: list[str] = field(default_factory=list)
    full_translation: str = ""
    # Optional second line written for the audience. `question`,
    # `interpretation`, and `relation_to_previous` are director metadata and
    # must never be rendered. Keeping the audience field explicit prevents a
    # model's editorial instructions from leaking into the finished video.
    audience_copy: str = ""
    narrative_beat: str = ""


@dataclass(slots=True)
class EditorialBrief:
    """Structured non-GitHub story. It never exposes low-level Scene fields."""

    headline: str
    subheadline: str
    fixed_conclusion: str
    attention_strategy: AttentionStrategy
    subjects: list[StorySubject]
    context_events: list[ContextEvent]
    evidence_shots: list[EvidenceShot]
    duration_target: float
    opportunity: EditorialOpportunity | None = None
    context_graph: ContextGraph | None = None
    director_brief: DirectorBrief | None = None
    # Radar V2 fields are optional for archived/classic manifests. New
    # writers fill them; deterministic validation then enforces the compact
    # sound-off contract.
    opening_mode: str = ""
    category_label: str = ""
    direct_identifier: str = ""
    editorial_inference: str = ""


@dataclass(slots=True)
class ColdOpenBeat:
    """A short editorial beat rendered before evidence walkthrough scenes."""

    id: str
    text: str
    emphasis: str
    duration: float
    evidence_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RenderManifest:
    id: str
    candidate_id: str
    content_type: ContentType
    scenes: list[Scene]
    evidence: list[Evidence]
    source_urls: list[str]
    topic_type: TopicType | None = None
    story_beats: list[StoryBeat] = field(default_factory=list)
    github_walkthrough: GitHubWalkthrough | None = None
    github_brief: GitHubProjectBrief | None = None
    editorial_brief: EditorialBrief | None = None
    fixed_hook: str | None = None
    fixed_title: str | None = None
    cold_open_beats: list[ColdOpenBeat] = field(default_factory=list)
    fixed_footer: str | None = None
    footer_shows_source_url: bool = False
    audio_mode: str = "bgm_only"
    music_license_status: str = "unreviewed"
    audio_path: str | None = None
    subtitle_path: str | None = None
    video_path: str | None = None
    license_records: list[dict[str, Any]] = field(default_factory=list)
    quality_checks: list[dict[str, Any]] = field(default_factory=list)
    render_profile: str = InformationRenderProfile.CLASSIC.value
    created_at: str = field(default_factory=now_iso)

    @property
    def duration(self) -> float:
        return max((scene.end for scene in self.scenes), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
