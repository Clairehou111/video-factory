from __future__ import annotations

import json
import http.client
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .director import NarrativeAnswer, SceneProposal, StoryboardRequest
from .models import (
    AttentionStrategy, Candidate, CaptureCue, ContentType, ContextEvent, ContextGraph, CueAction, DirectorBrief, EditorialBrief,
    Evidence, EvidenceShot, EvidenceShotKind, GitHubFocusCandidate, GitHubModuleFocus,
    GitHubProjectBrief, GitHubWalkthrough, MaterialRole, SourceType, StoryArcBeat, StorySubject, TopicType,
)
from .writer import StoryWriterPacket
from .github_editor import canonicalize_github_brief, compose_github_hook, copy_width, select_github_focuses
from .editorial import canonicalize_editorial_brief, compile_evidence_shots
from .translation import IT_TRANSLATION_CONTRACT, PLAIN_CHINESE_CONTRACT


def _coerce_model_float(value: object, default: float) -> float:
    """Accept one model-written number while ignoring surrounding prose punctuation."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value).replace(",", ""))
    if not match:
        return default
    try:
        return float(match.group(0))
    except ValueError:
        return default


def _same_source_document(left: str, right: str) -> bool:
    return left.split("#", 1)[0].rstrip("/") == right.split("#", 1)[0].rstrip("/")


def _compile_evidence_shot_kind(raw: dict[str, object], evidence_by_id: dict[str, Evidence]) -> EvidenceShotKind:
    """Compile semantic presentation into a material kind owned by the program.

    Cheap models otherwise put values such as ``impact_card`` into ``kind``.
    That is the same category error as using ``trial`` as a material role: the
    first is editorial meaning, the second is an execution primitive.
    """
    family = str(raw.get("visual_family") or "").strip()
    cited = [evidence_by_id[item] for item in raw.get("evidence_ids", []) if item in evidence_by_id]
    kinds = {item.source_kind for item in cited}
    # Semantic presentation wins when a shot cites a complete X post plus its
    # attached image/analysis. The attachment enriches the tweet card; it must
    # not silently turn the mandatory root-post shot into a standalone image.
    if family in {"tweet", "quoted_post"}:
        return EvidenceShotKind.TWEET_CARD
    if family == "source_image" or any(kind in {"web:source_image", "x:media_photo"} for kind in kinds):
        return EvidenceShotKind.IMAGE
    if family == "paper":
        return EvidenceShotKind.PDF_PAGE
    if family == "chart":
        return EvidenceShotKind.BENCHMARK_CHART
    if family == "code":
        return EvidenceShotKind.CODE_EXAMPLE
    if family in {"quote_card", "timeline", "impact_card", "stat_card"}:
        if any(kind.startswith("x:") for kind in kinds):
            return EvidenceShotKind.TWEET_CARD
        if any(kind.startswith("paper:") for kind in kinds):
            return EvidenceShotKind.PDF_PAGE
        return EvidenceShotKind.BROWSER_SECTION
    if any(kind.startswith("x:") for kind in kinds):
        return EvidenceShotKind.TWEET_CARD
    if any(kind.startswith("paper:") for kind in kinds):
        return EvidenceShotKind.PDF_PAGE
    if family == "product_ui":
        return EvidenceShotKind.IMAGE if any("image" in kind or "media" in kind for kind in kinds) else EvidenceShotKind.BROWSER_SECTION
    return EvidenceShotKind.BROWSER_SECTION


def _planning_excerpt(value: str, limit: int = 5000) -> str:
    if len(value) <= limit:
        return value
    useful = re.compile(
        r"(?:^#{1,4}\s|quick|start|install|usage|feature|limit|license|warning|agent|cli|input|output|"
        r"快速|安装|使用|功能|限制|许可|版权|注意|警告|命令|输入|输出)", re.IGNORECASE,
    )
    pieces = [value[:1200]]
    seen = {pieces[0]}
    for line in value.splitlines():
        stripped = line.strip()
        if stripped and useful.search(stripped) and stripped not in seen:
            pieces.append(stripped)
            seen.add(stripped)
        if sum(len(item) for item in pieces) >= limit - 700:
            break
    pieces.append(value[-600:])
    return "\n".join(pieces)[:limit]


def _adds_unsupported_product_concept(copy: str, source: str) -> bool:
    concepts = (
        (r"\bAPI\b", r"\bAPI\b"),
        (r"成本|预算|(?<!付)费用|\bcost\b|\bbudget\b", r"成本|预算|(?<!付)费用|\bprice|\bpricing\b|\bcost\b|\bbudget\b"),
        (r"配额|额度|用量限制|usage limit|quota", r"配额|额度|用量限制|usage limit|quota"),
        (r"累积|结转|roll.?over|accumulat", r"累积|结转|roll.?over|accumulat"),
        (r"更灵活|灵活的", r"更灵活|灵活的|flexib|at (?:your|their) own leisure"),
    )
    return any(
        re.search(visible, copy, re.IGNORECASE) and not re.search(proof, source, re.IGNORECASE)
        for visible, proof in concepts
    )


class StoryDraftError(ValueError):
    """Preserve malformed model JSON so the provider can repair it once."""

    def __init__(self, draft: dict[str, object], error: Exception):
        super().__init__(str(error))
        self.draft = draft


@dataclass(frozen=True, slots=True)
class EditorialPlan:
    """A cheap first-pass decision about what the story still needs.

    This is deliberately not a storyboard.  Keeping research intent separate
    from scene/material fields prevents editorial concepts such as ``trial``
    and ``boundary`` from leaking into the renderer contract.
    """

    angle: str
    audience_value: str
    selected_evidence_ids: list[str]
    requested_urls: list[str]
    unresolved_questions: list[str]
    ready_to_write: bool
    why_now: str = ""
    why_audience: str = ""
    audience_pain_or_desire: str = ""
    selection_reasons: list[dict[str, object]] = field(default_factory=list)
    expansion_dimensions: list[str] = field(default_factory=list)
    context_questions: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    story_archetype: str = ""


@dataclass(frozen=True, slots=True)
class LLMSettings:
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 45
    provider_preferences: dict[str, object] | None = None

    @classmethod
    def from_environment(cls, provider: str, model: str | None = None) -> "LLMSettings":
        try:
            timeout_seconds = int(os.environ.get("VIDEO_FACTORY_LLM_TIMEOUT_SECONDS", "45"))
        except ValueError:
            timeout_seconds = 45
        timeout_seconds = max(15, min(timeout_seconds, 120))
        if provider == "deepseek":
            key = os.environ.get("DEEPSEEK_API_KEY")
            # The content agent uses the inexpensive chat tier by default.
            # A stronger account-specific model is configured separately as
            # an escalation writer, never silently selected here.
            selected_model = model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
            return cls(
                provider, os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                key or "", selected_model, timeout_seconds=timeout_seconds,
            )
        if provider == "kimi":
            key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
            selected_model = model or os.environ.get("KIMI_MODEL")
            if not selected_model:
                raise ValueError("Kimi requires --model or KIMI_MODEL; model availability is account-specific")
            return cls(
                provider, os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
                key or "", selected_model, timeout_seconds=timeout_seconds,
            )
        if provider == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY")
            selected_model = model or os.environ.get("OPENROUTER_MODEL")
            if not selected_model:
                raise ValueError("OpenRouter requires a selected model; use --provider auto or --model")
            preferences: dict[str, object] = {
                "sort": "price", "allow_fallbacks": True,
                "require_parameters": True,
                "data_collection": os.environ.get("OPENROUTER_DATA_COLLECTION", "deny"),
            }
            if os.environ.get("OPENROUTER_ZDR", "0") == "1":
                preferences["zdr"] = True
            return cls(
                provider, os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                key or "", selected_model, timeout_seconds=timeout_seconds,
                provider_preferences=preferences,
            )
        raise ValueError(f"unsupported story provider: {provider}")


class OpenAICompatibleStoryWriter:
    """Remote generation is optional and provider-neutral; no Codex runtime is involved."""

    def __init__(self, settings: LLMSettings):
        if not settings.api_key:
            raise ValueError(f"missing API key for {settings.provider}")
        self.settings = settings

    def generate(self, packet: StoryWriterPacket) -> tuple[StoryboardRequest, dict[str, object], dict[str, object]]:
        return self._generate_from_messages(packet, [
            {"role": "system", "content": "Return a single valid JSON object. Never add markdown fences."},
            {"role": "user", "content": packet.prompt()},
        ])

    def review_visible_copy(
        self, packet: StoryWriterPacket, draft: dict[str, object],
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        """Use a bounded semantic critic instead of example-specific regexes."""
        editorial = dict(draft.get("editorial_brief") or {})
        github = dict(draft.get("github_brief") or {})
        visible = {
            "headline": editorial.get("headline"),
            "subheadline": editorial.get("subheadline"),
            "fixed_conclusion": editorial.get("fixed_conclusion") or draft.get("footer"),
            "opening_mode": editorial.get("opening_mode"),
            "category_label": editorial.get("category_label"),
            "editorial_inference": editorial.get("editorial_inference"),
            "attention_strategy": editorial.get("attention_strategy"),
            "subjects": editorial.get("subjects"),
            "evidence_shots": [{
                key: item.get(key) for key in (
                    "id", "fact", "audience_copy", "target", "translation",
                    "full_translation", "evidence_ids", "context_event_ids", "narrative_beat",
                )
            } for item in editorial.get("evidence_shots", []) if isinstance(item, dict)],
            "github_brief": github or None,
        }
        if github:
            fields: dict[str, object] = {
                "github_brief.hook_opening": github.get("hook_opening"),
                "github_brief.hook_reveal": github.get("hook_reveal"),
                "github_brief.hook_verdict": github.get("hook_verdict"),
                "github_brief.project_title": github.get("project_title"),
                "github_brief.footer": draft.get("footer"),
                "github_brief.repo_description_translation": github.get("repo_description_translation"),
                "github_brief.readme_claim_translation": github.get("readme_claim_translation"),
            }
            selected_focus_ids = {
                str(value) for value in (github.get("selected_focus_ids") or [])
            }
            for index, focus in enumerate(github.get("focus_candidates") or []):
                if not isinstance(focus, dict) or str(focus.get("id") or "") not in selected_focus_ids:
                    continue
                translation_key = "browser_translation" if focus.get("browser_translation") else "translation"
                fields[f"github_brief.focus_candidates[{index}].{translation_key}"] = focus.get(translation_key)
        else:
            fields = {
                "editorial_brief.headline": visible["headline"],
                "editorial_brief.subheadline": visible["subheadline"],
                "editorial_brief.fixed_conclusion": visible["fixed_conclusion"],
                "editorial_brief.attention_strategy.selected_hook": (
                    (visible["attention_strategy"] or {}).get("selected_hook")
                    if isinstance(visible["attention_strategy"], dict) else None
                ),
            }
        fields = {path: value for path, value in fields.items() if value}
        for index, shot in enumerate(visible["evidence_shots"]):
            for name in ("fact", "audience_copy", "target", "translation", "full_translation"):
                if shot.get(name):
                    fields[f"editorial_brief.evidence_shots[{index}].{name}"] = shot[name]
        referenced = {
            str(evidence_id)
            for shot in visible["evidence_shots"]
            for evidence_id in (shot.get("evidence_ids") or [])
        }
        if github:
            referenced.update(str(value) for value in (github.get("hook_evidence_ids") or []))
            selected_focus_ids = {str(value) for value in (github.get("selected_focus_ids") or [])}
            for focus in github.get("focus_candidates") or []:
                if isinstance(focus, dict) and str(focus.get("id") or "") in selected_focus_ids:
                    referenced.update(str(value) for value in (focus.get("evidence_ids") or []))
        evidence = [{
            "id": item.id, "kind": item.source_kind, "url": item.url,
            "quote": item.quote[:2200],
        } for item in packet.evidence if item.id in referenced or len(referenced) == 0]
        schema = {
            "approved": True,
            "field_reviews": [{
                "field_path": "one exact key from fields",
                "verdict": "pass|fail",
                "actor_action_object_recipient": "plain extraction, or empty when not applicable",
                "certainty": "fact|reported_claim|inference|unknown|not_applicable",
                "naturalness_score": "1|2|3|4|5",
                "attention_score": "1|2|3|4|5; required for headline, selected_hook, and fixed_conclusion; otherwise 0",
                "evidence_ids": ["existing evidence id"],
                "category": "entity_relation|causal_certainty|source_support|technical_specificity|natural_chinese|story_clarity|retention_hook|payoff|none",
                "problem": "precise issue, empty only for pass",
                "repair_instruction": "content-neutral correction instruction using only supplied evidence; never suggest a new date, metric, scope, superlative, policy, or mechanism",
            }],
        }
        review_messages = [
            {"role": "system", "content": "Return one strict JSON review only. You are a skeptical Chinese technical-video copy editor, not the original writer."},
            {"role": "user", "content": "\n".join([
                "Review every field listed in fields against its cited evidence and the story as a whole. Return exactly one field_reviews item for every field path; do not omit easy fields.",
                "A field path ending in .target is the exact source-language proof the browser will highlight, not Chinese copy. Ignore Chinese naturalness for targets. Pass it only when it is an exact contiguous substring of its cited evidence and directly supports that same shot's fact; topical proximity elsewhere on the page is insufficient. If the page supports the fact but this target points at a different claim, fail source_support and instruct replacement with the smallest exact supporting excerpt from the same cited evidence.",
                "Reject when an actor, action, object, recipient, chronology, causal strength, or certainty differs from the evidence; when a concrete technical name is replaced by a vague category that makes the event harder to understand; when Chinese reads like literal translation, a report, or abstract consultant language; or when a screen cannot explain itself without narration. For a model/product story, reject a selected hook that omits the exact model/product name and names only its vendor, publisher, host, or benchmark.",
                "For each field, first extract actor-action-object-recipient and certainty, then compare them with evidence. A naturalness score below 4 is fail. Unexplained English technical nouns inside Chinese prose are fail when the evidence lets you explain the concrete action. Keeping an official English feature name does not exempt it: the first audience-facing occurrence must immediately explain what the feature concretely gives or does in natural Chinese. The same rule applies to specialist Chinese metrics: if a hook/fact says 拒绝率、幻觉率、激活参数、上下文窗口 or a similarly non-obvious metric, the first relevant evidence shot must say in plain Chinese what it measures or means in practice; numbers alone are not an explanation. Preserve quantity, duration, recurrence, permission, and guarantee strength exactly: a one-time credit, reset, trial, exception, or temporary rollout cannot become permanent freedom from recurring limits or costs; free availability or free use does not prove commercial-use permission; support does not prove a guarantee; and an open-source repository does not transfer third-party asset licenses. Unsupported mechanisms, policies, risks, permissions, or advice are fail.",
                "A specialist term needs one adjacent explanation at its first relevant evidence shot, not repetition in every persistent rail and field. If that first shot explains it, do not fail the headline, hook, conclusion, or later shots merely for using the same term without repeating the definition.",
                "For headline, selected_hook, fixed_conclusion, GitHub hook_opening, and GitHub footer, also judge short-video attention. A score below 4 is fail. Judge selected_hook as the first 1.5 seconds: when the cited evidence contains an exact number/contrast, a named consequential actor, concrete developer pain/ROI, or an honest open question inside the selected story, a neutral announcement label scores only 3. Do not require shock: when opening_mode is direct_fact, a crisp named actor + important concrete change can score 4 without conflict or an information gap. conflict, counter_intuitive, and developer_roi must be earned by evidence. GitHub hook_reveal and hook_verdict must add a clear new capability and payoff, but they do not each need another standalone conflict. Calibrate strictly: 1 is vague filler; 2 is a generic topic; 3 is an accurate but low-stakes label; 4 is immediately clear and gives the intended audience a concrete reason to keep watching through the event itself, a verified contrast, surprise, consequence, relief, scale, ROI, or honest open question; 5 is unusually memorable without enlarging the claim. Generic phrases such as 引发讨论、值得关注、注意风险、生态竞争 or a neutral topic label are not a payoff. The fixed conclusion or GitHub footer must deliver a clear evidence-backed view or memorable consequence, not merely repeat the event or give ritual caution.",
                "If editorial_inference is non-empty, it must be the selected question-form hook. Judge the proof and takeaway as false if they silently promote that question into a fact. category_label is optional factual navigation only; never demand one and reject evaluative labels that are not source facts.",
                "Every repair_instruction is evidence-bound too. Never propose an example sentence containing a date, quantity, rollout scope, default behavior, first/only/all/complete superlative, licensing permission, official policy, mechanism, or capability absent from the supplied evidence. If stronger attention cannot be earned by another verified fact, improve the stance and concrete wording around the existing fact instead of inventing one.",
                "For a fixed_conclusion repair, end on the strongest verified impact, payoff, or concrete action. Never instruct the writer to append 未知、有待观察、有待验证、进一步研究、需关注后续 or an equivalent ritual caveat; move a decision-critical scope limit to its own evidence field instead.",
                "Treat Chinese scope and novelty words as claims that require exact support. 全面、全部、所有、任何、彻底、完全、首次、第一次、唯一 must fail when the cited evidence does not prove the same actor, object, time, and coverage. In particular, supported models, where supported, new models, or across products cannot be broadened to 全面/全部, and a newly visible repository cannot become 首次/第一次/唯一. Do not excuse these words as emotion or style.",
                "Short-video fields are selective, not encyclopedic. A self-contained supported claim does not fail merely because it omits another capability, format, condition, or limitation from the same README. For GitHub, do not demand that hook_reveal, hook_verdict, and footer each repeat limitations; a limitation belongs on screen only when the selected story makes it decision-critical. Never turn a README limitation into a ritual caveat in every field. A less specific true date or scope is not false merely because the source is more precise, unless the broader wording changes who, when, or how much. Official vendor documentation and support pages remain primary vendor evidence even when they are not press-release announcements.",
                "A repair must not make Chinese harder to understand. Do not introduce an English implementation noun such as harness into Chinese copy unless the field already needs that exact name and immediately explains its concrete role in natural Chinese. Preserving technical accuracy does not require copying every source term.",
                "Energy must come from the event's actual stakes and the editor's evidence-backed stance. Do not manufacture outrage, certainty, scale, or conflict. When the facts are genuinely dramatic, allow direct emotional Chinese instead of flattening them into report language.",
                "Do not enforce a preferred opinion or wording. Do not invent facts. Judge whether a Chinese technical viewer can understand who did what and what happened next on the first read. If any field fails, approved must be false.",
                "Audience: Chinese developers and technically curious vibe coders watching a BGM-only WeChat Channels video.",
                "Fields: " + json.dumps(fields, ensure_ascii=False),
                "Visible copy: " + json.dumps(visible, ensure_ascii=False),
                "Evidence: " + json.dumps(evidence, ensure_ascii=False),
                "Return JSON matching: " + json.dumps(schema, ensure_ascii=False),
            ])},
        ]
        def normalized_field_reviews(payload: dict[str, object]) -> list[dict[str, object]]:
            rows: list[dict[str, object]] = []
            github_aliases = {
                "fixed_conclusion": "github_brief.footer",
                "footer": "github_brief.footer",
                "github_brief.fixed_conclusion": "github_brief.footer",
                **{
                    name: f"github_brief.{name}" for name in (
                        "hook_opening", "hook_reveal", "hook_verdict", "project_title",
                        "repo_description_translation", "readme_claim_translation",
                    )
                },
            }
            editorial_aliases = {
                "headline": "editorial_brief.headline",
                "subheadline": "editorial_brief.subheadline",
                "fixed_conclusion": "editorial_brief.fixed_conclusion",
                "selected_hook": "editorial_brief.attention_strategy.selected_hook",
            }
            if not github:
                for index, shot in enumerate(visible["evidence_shots"]):
                    shot_id = str(shot.get("id") or "").strip()
                    if not shot_id:
                        continue
                    for name in ("fact", "audience_copy", "target", "translation", "full_translation"):
                        canonical = f"editorial_brief.evidence_shots[{index}].{name}"
                        editorial_aliases[f"{shot_id}.{name}"] = canonical
                        editorial_aliases[f"evidence_shots[{shot_id}].{name}"] = canonical
                        editorial_aliases[f"editorial_brief.evidence_shots[{shot_id}].{name}"] = canonical
            aliases = github_aliases if github else editorial_aliases
            for item in payload.get("field_reviews") or []:
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                path = str(row.get("field_path") or "")
                row["field_path"] = aliases.get(path, path)
                if not github and re.fullmatch(r"evidence_shots\[\d+\]\.[a-z_]+", row["field_path"]):
                    row["field_path"] = "editorial_brief." + row["field_path"]
                rows.append(row)
            return rows

        # A non-GitHub deep dive can expose roughly thirty independently
        # reviewable fields. Reasoning-capable OpenRouter models count hidden
        # reasoning against max_tokens even when it is excluded from the
        # response, so 3.6K can end with finish_reason=length and no JSON.
        review, provenance = self._request_json(review_messages, max_tokens=7200)
        field_reviews = normalized_field_reviews(review)
        reviews_by_path = {
            str(item.get("field_path") or ""): item for item in field_reviews
            if str(item.get("field_path") or "") in fields
        }
        missing_paths = set(fields) - set(reviews_by_path)
        if missing_paths:
            first_provenance = provenance
            missing_fields = {path: fields[path] for path in fields if path in missing_paths}
            retry_messages = deepcopy(review_messages)
            original_fields = "Fields: " + json.dumps(fields, ensure_ascii=False)
            retry_fields = "Fields: " + json.dumps(missing_fields, ensure_ascii=False)
            retry_messages[-1]["content"] = retry_messages[-1]["content"].replace(
                original_fields, retry_fields,
            )
            review, provenance = self._request_json(retry_messages, max_tokens=3600)
            provenance = {
                **provenance,
                "review_retried": True,
                "missing_fields": sorted(missing_paths),
                "first_attempt": first_provenance,
            }
            for item in normalized_field_reviews(review):
                path = str(item.get("field_path") or "")
                if path in missing_paths:
                    reviews_by_path[path] = item
            unresolved = sorted(set(fields) - set(reviews_by_path))
            if unresolved:
                raise StoryDraftError(
                    review,
                    ValueError(
                        "copy critic omitted viewer-facing fields after one supplemental review; "
                        f"missing={unresolved}"
                    ),
                )
        field_reviews = [reviews_by_path[path] for path in fields]
        issues = [{
            "field_path": item.get("field_path"), "category": item.get("category"),
            "problem": item.get("problem"), "evidence_ids": item.get("evidence_ids") or [],
            "repair_instruction": item.get("repair_instruction"),
        } for item in field_reviews if str(item.get("verdict") or "").casefold() == "fail"]
        model_reported_failure = bool(issues)
        attention_paths = {
            "editorial_brief.headline", "editorial_brief.fixed_conclusion",
            "editorial_brief.attention_strategy.selected_hook",
            "github_brief.hook_opening", "github_brief.footer",
        }
        failed_paths = {str(item.get("field_path") or "") for item in issues}
        for item in field_reviews:
            path = str(item.get("field_path") or "")
            if path not in attention_paths or path in failed_paths:
                continue
            try:
                attention_score = int(item.get("attention_score") or 0)
            except (TypeError, ValueError):
                attention_score = 0
            if attention_score < 4:
                issues.append({
                    "field_path": path, "category": "retention_hook",
                    "problem": "short-video attention score is below 4/5",
                    "evidence_ids": item.get("evidence_ids") or [],
                    "repair_instruction": "use the strongest evidence-backed contradiction, surprise, consequence, or stance",
                })
        # The deterministic audience-glossary compiler explains harness once
        # at its first relevant evidence shot. Some critics nevertheless ask
        # every rail and translation to repeat the same definition, which is
        # incompatible with a ten-second mobile layout. Keep semantic/source
        # failures, but delegate this one presentation concern to the compiler.
        issues = [
            item for item in issues
            if not (
                str(item.get("category") or "") == "technical_specificity"
                and "harness" in (
                    str(item.get("problem") or "")
                    + str(item.get("repair_instruction") or "")
                ).casefold()
                and re.search(r"(?:未解释|解释|测试框架|运行框架)", (
                    str(item.get("problem") or "")
                    + str(item.get("repair_instruction") or "")
                ))
            )
        ]
        if bool(review.get("approved")) and model_reported_failure:
            raise StoryDraftError(review, ValueError("copy critic approval conflicts with field verdicts"))
        # The field-level verdicts are the authoritative structured decision.
        # Some providers occasionally emit approved=false while marking every
        # required field as pass.  That boolean carries no actionable repair
        # target, so treating it as a fatal rejection makes a valid generation
        # nondeterministically fail.  We still fail closed on any explicit field
        # failure or low attention score above.
        return issues, provenance

    def plan(self, packet: StoryWriterPacket) -> tuple[EditorialPlan, dict[str, object]]:
        allowed_urls = list(dict.fromkeys([packet.candidate.source_url, *packet.candidate.linked_sources]))
        evidence_index = []
        for item in packet.evidence:
            quote = _planning_excerpt(item.quote)
            evidence_index.append({
                "id": item.id, "url": item.url, "kind": item.source_kind,
                "character_count": len(item.quote), "excerpt": quote,
            })
        schema = {
            "angle": "specific evidence-bounded editorial angle in Chinese",
            "audience_value": "what developers or vibe coders learn or can decide",
            "selected_evidence_ids": ["evidence id needed for writing"],
            "requested_urls": ["only an allowed linked URL that still needs opening"],
            "unresolved_questions": ["important fact that remains unknown"],
            "ready_to_write": True,
            "why_now": "why this event deserves attention now",
            "why_audience": "why developers, vibe coders, or technical viewers care",
            "audience_pain_or_desire": "cost, capability, efficiency, opportunity, identity, or risk",
            "selection_reasons": [{"id": "stable-id", "dimension": "important_person|audience_pain|capability_shift|price_change|competition|trend|security|workflow", "rationale": "specific reason", "evidence_ids": ["evidence-id"]}],
            "expansion_dimensions": ["time|cause|competition|product_evolution|people|technical|cost|workflow|ecosystem|later_update|contrast|historical_pattern"],
            "context_questions": ["question whose answer could materially change the story"],
            "search_queries": ["at most three bounded entity/event queries"],
            "story_archetype": "event_chain|people_change|research_disclosure|capability_shift|price_competition|workflow_change|official_update|other",
        }
        messages = [
            {"role": "system", "content": "Return one valid JSON object only. This is research planning, not scene writing."},
            {"role": "user", "content": "\n".join([
                "You are the low-cost research planner for an evidence-bound Chinese technical-video editor.",
                "Choose the smallest useful evidence set. Request at most three URLs and only from allowed_urls. Do not invent URLs, facts, scenes, rendering fields, material roles, or a category of missing information. The editorial angle must be strictly faithful to the primary source facts and the author's stated conclusions: do not make associative causal leaps (for example, better architecture/inference efficiency cannot be assumed to mean cheaper deployment or lower financial cost unless the evidence explicitly provides financial or cost data). A license, price, benchmark, deployment cost, funding amount, or limitation may be unresolved only when the supplied evidence actually raises that category; silence is not evidence that it is a story-worthy unknown. Explain why this candidate deserves publication before deciding how to tell it. Select 2–4 expansion dimensions, ask only questions that could change interpretation, and propose at most three search queries. Context investigation is mandatory; context inclusion is selective.",
                "Attention is non-negotiable: identify a concrete audience pain, gain, conflict, surprise, winner/loser, or capability shift. This is not permission for empty hype. The eventual story must preserve the selection reason and deliver a strong evidence-backed stance suitable for WeChat Channels.",
                "For a GitHub repository, identify the concrete job, visible result or I/O, shortest trial, distinctive mechanism, and adoption value. Do not look for a boundary unless README explicitly states one. Also identify the strongest named tension or payoff for the opening: when a repo removes/counters/audits/bypasses a vendor capability, the editorial angle should name both sides instead of leading with its implementation stack. Missing facts stay unresolved.",
                "For a person/team move, use one context question and one search query to test recent related moves from the same incumbent; this is how one departure becomes an evidenced pattern or stays a single event. For a price/limit/reset update, search the author/company's preceding announcement and the rollout scope before inventing market context.",
                f"Topic: {packet.topic_type.value}; format: {packet.content_type.value}; target duration: {packet.target_duration}s.",
                "Existing editorial context: " + (packet.editorial_direction or "none"),
                "Candidate: " + json.dumps({
                    "title": packet.candidate.title, "url": packet.candidate.source_url,
                    "linked_sources": packet.candidate.linked_sources,
                }, ensure_ascii=False),
                "Allowed URLs: " + json.dumps(allowed_urls, ensure_ascii=False),
                "Evidence index: " + json.dumps(evidence_index, ensure_ascii=False),
                "Return JSON matching: " + json.dumps(schema, ensure_ascii=False),
            ])},
        ]
        # Reasoning-capable discount models account for hidden reasoning in
        # max_tokens; 1.8K can leave no room for the small JSON plan itself.
        draft, provenance = self._request_json(messages, max_tokens=4000)
        try:
            plan = self._parse_plan(packet, draft)
        except (KeyError, TypeError, ValueError) as error:
            raise StoryDraftError(draft, error) from error
        return plan, provenance

    def repair(
        self, packet: StoryWriterPacket, invalid_draft: dict[str, object], validation_error: str,
    ) -> tuple[StoryboardRequest, dict[str, object], dict[str, object]]:
        if packet.topic_type == TopicType.GITHUB_PROJECT and self._hook_only_repair(validation_error):
            return self._repair_github_hook(packet, invalid_draft, validation_error)
        visible_size_clauses = [clause for clause in validation_error.split(";") if clause.strip()]
        if packet.topic_type != TopicType.GITHUB_PROJECT and visible_size_clauses and all(
            "root-post Chinese translation" in clause or "fixed conclusion must fit" in clause
            for clause in visible_size_clauses
        ):
            return self._repair_root_translation(packet, invalid_draft, validation_error)
        context_structure_error = (
            "required setup/context event" in validation_error
            or "incumbent-history pattern context" in validation_error
        )
        if packet.topic_type != TopicType.GITHUB_PROJECT and not context_structure_error and (
            "missing mechanism/details" in validation_error or "unsupported concepts" in validation_error
            or "quoted earlier post" in validation_error or "branded feature name" in validation_error
            or "near-duplicates" in validation_error or "absence-of-limit" in validation_error
            or "audience_copy" in validation_error
            or "release/launch wording" in validation_error
            or "quantified_claims" in validation_error
            or "fixed conclusion is bureaucratic" in validation_error
            or "final changing shot must end" in validation_error
            or "flash final shot must deliver" in validation_error
            or "flash story must spend its last seconds" in validation_error
            or "semantic copy critic issues" in validation_error
            or "selected_hook must name the concrete model subject" in validation_error
            or "must explain specialist metric" in validation_error
        ):
            return self._repair_editorial_copy(packet, invalid_draft, validation_error)
        repair_contract = (
            "For non-GitHub output, repair editorial_brief only. Keep attention_strategy, subjects, context_events, and evidence_shots distinct. "
            "Never add scenes, kind, material_role, visual_action, recording_cues, trial materials, or boundary materials. visual_family is presentation; it is never a material kind. "
            "The three hook candidates must be specific and different; selected_hook must exactly match one. "
            "An X-rooted story begins with one complete tweet_card. Every later shot adds a causal step, contrast, evidence, consequence, or payoff. "
            "Targets are exact source substrings and translations contain natural technical Chinese only."
            if packet.topic_type != TopicType.GITHUB_PROJECT else
            "Do not select boundary for a GitHub final cut; choose technical_edge/adoption and explain an implemented capability. "
            "Browser targets must be exact contiguous substrings from one source line or intact paragraph."
        )
        if packet.topic_type != TopicType.GITHUB_PROJECT and "release/launch wording" in validation_error:
            repair_contract += (
                " The cited page has no release event. Remove 发布、上线、宣布、正式推出 from headline, "
                "hook_fact, selected_hook, every hook_candidate, subject actions, and shot copy. Describe the "
                "verified workflow/capability as available documentation; do not merely change 发布 to 文档上线."
            )
        if packet.topic_type != TopicType.GITHUB_PROJECT and "fixed conclusion must end" in validation_error:
            repair_contract += (
                " End fixed_conclusion on the evidence-backed payoff, impact, or action. Scope/unknown may stay in its "
                "own evidence shot, but the fixed conclusion must not end with 未知、有待验证、进一步研究、需关注后续."
            )
        if packet.topic_type != TopicType.GITHUB_PROJECT and (
            "required setup/context event" in validation_error or "incumbent-history pattern context" in validation_error
        ):
            repair_contract += (
                " Rebuild the complete editorial brief so director_brief.selected_context_ids includes every required_context_id. "
                "For a people_change story with pattern_context_ids, select and visibly use at least one verified earlier incumbent-history event. "
                "Add its evidence id and context event id to a distinct background/turn shot; state chronology explicitly and do not merge it with the root event."
            )
        if packet.topic_type != TopicType.GITHUB_PROJECT and "root-post Chinese translation" in validation_error:
            repair_contract += (
                " Compress only the root full_translation to 60–120 Chinese characters. Preserve every decisive actor, "
                "action, number, scope, and result; remove handles, URL, emoji, greetings, and repeated wording."
            )
        if packet.topic_type != TopicType.GITHUB_PROJECT and (
            "missing mechanism/details" in validation_error or "unsupported concepts" in validation_error
        ):
            repair_contract += (
                " Delete every unsupported API/cost/quota/accumulation/flexibility claim and every missing-details card. "
                "Do not replace them with synonyms. Rebuild the final beat from a cited actor, rollout time, coverage, "
                "capability, result, or action that the source explicitly states."
            )
        if packet.topic_type != TopicType.GITHUB_PROJECT and "quantified_claims" in validation_error:
            repair_contract += (
                " Replace every flagged quantity with the exact value and unit stated in its cited evidence, "
                "including currency scale, or delete that quantity. Never approximate, round, or change 万/亿 scale."
            )
        return self._generate_from_messages(packet, [
            {"role": "system", "content": "Return a single corrected valid JSON object. Never add markdown fences."},
            {"role": "user", "content": packet.prompt()},
            {"role": "assistant", "content": json.dumps(invalid_draft, ensure_ascii=False)},
            {"role": "user", "content": "Your JSON failed deterministic validation. Return the complete corrected JSON and fix EVERY semicolon-separated error simultaneously. Never return the same invalid target/role pair or the same overlong browser translation. Do not shorten a title or selected_hook by deleting an exact person, company, project, model, or product name: the renderer reduces title font size and adds lines. Rewrite an overlong browser-highlight translation to at most 36 total characters; root full_translation follows its separate 60–120-character rule. " + repair_contract + " Error: " + validation_error},
        ])

    def _repair_root_translation(
        self, packet: StoryWriterPacket, invalid_draft: dict[str, object], validation_error: str,
    ) -> tuple[StoryboardRequest, dict[str, object], dict[str, object]]:
        editorial = dict(invalid_draft.get("editorial_brief") or {})
        shots = editorial.get("evidence_shots") or invalid_draft.get("evidence_shots") or []
        if not shots:
            raise StoryDraftError(invalid_draft, ValueError("root translation repair has no evidence shots"))
        first = shots[0]
        root_ids = set(str(item) for item in first.get("evidence_ids", []))
        root = next((item for item in packet.evidence if item.id in root_ids), packet.evidence[0])
        needs_translation = "root-post Chinese translation" in validation_error
        needs_footer = "fixed conclusion must fit" in validation_error
        requested_fields = []
        if needs_translation:
            requested_fields.append('"full_translation":"60–120 readable Chinese characters"')
        if needs_footer:
            requested_fields.append('"fixed_conclusion":"specific conclusion, at most 62 Chinese-character-equivalents"')
        evidence_excerpt = [
            {"id": item.id, "quote": item.quote[:1800]} for item in packet.evidence[:8]
        ]
        patch, provenance = self._request_json([
            {"role": "system", "content": "Return one JSON object containing only the requested visible-copy fields."},
            {"role": "user", "content": "\n".join([
                "Repair only fields that overflow fixed rails in a BGM-only WeChat short video. Do not change evidence, order, shots, targets, visual families, or facts.",
                "For full_translation, compress the complete root X post into 60–120 readable Chinese characters beside the original. Count every Chinese character, Latin letter, digit, space, and punctuation mark toward the 120-character maximum. Preserve only the decisive named actor, action, scope, result, and essential visible numbers. Remove secondary component details, handles, URL, emoji, greeting, and repeated wording. Do not say 翻译/译为.",
                "For fixed_conclusion, preserve the existing evidence-backed payoff while compressing it to 28–62 Chinese-character-equivalents. It must be a complete assertive sentence, not a caveat, label, or truncated fragment.",
                "Original post: " + root.quote,
                "Current overlong translation: " + str(first.get("full_translation") or first.get("translation") or ""),
                "Current conclusion: " + str(editorial.get("fixed_conclusion") or invalid_draft.get("footer") or ""),
                "Evidence excerpts: " + json.dumps(evidence_excerpt, ensure_ascii=False),
                "Validation: " + validation_error,
                "Return: {" + ",".join(requested_fields) + "}",
            ])},
        ], max_tokens=700)
        translation = str(patch.get("full_translation") or "").strip()
        conclusion = str(patch.get("fixed_conclusion") or "").strip()
        if needs_translation and not 40 <= len(translation) <= 140:
            # Preserve the complete draft so the bounded second visible-copy
            # cleanup retries this field instead of mistakenly treating a
            # two-field patch as a whole storyboard.
            raise StoryDraftError(invalid_draft, ValueError(validation_error))
        if needs_footer and (not conclusion or copy_width(conclusion) > 62):
            raise StoryDraftError(invalid_draft, ValueError(validation_error))
        repaired = deepcopy(invalid_draft)
        repaired_editorial = dict(repaired.get("editorial_brief") or {})
        repaired_shots = repaired_editorial.get("evidence_shots") or repaired.get("evidence_shots") or []
        if needs_translation:
            repaired_shots[0]["full_translation"] = translation
        if needs_footer:
            repaired_editorial["fixed_conclusion"] = conclusion
            repaired["editorial_brief"] = repaired_editorial
            repaired["footer"] = conclusion
        try:
            request = self._to_storyboard_request(packet, repaired)
        except (KeyError, TypeError, ValueError) as error:
            raise StoryDraftError(repaired, error) from error
        return request, provenance, repaired

    def _repair_editorial_copy(
        self, packet: StoryWriterPacket, invalid_draft: dict[str, object], validation_error: str,
    ) -> tuple[StoryboardRequest, dict[str, object], dict[str, object]]:
        """Patch only semantic copy that failed grounding, preserving scene evidence mechanics."""
        editorial = dict(invalid_draft.get("editorial_brief") or {})
        shot_source = editorial.get("evidence_shots") or invalid_draft.get("evidence_shots") or []
        director_source = editorial.get("director_brief") or invalid_draft.get("director_brief") or {}
        current = {
            "shots": [{
                "id": item.get("id"), "question": item.get("question"),
                "evidence_ids": item.get("evidence_ids"), "beat_ids": item.get("beat_ids"),
            } for item in shot_source],
        }
        evidence_excerpt = [{
            "id": item.id, "kind": item.source_kind, "quote": item.quote[:2400], "notes": item.notes or "",
        } for item in packet.evidence]
        patch_schema = {
            "headline": "corrected concrete Chinese headline",
            "subheadline": "new source-backed information",
            "fixed_conclusion": "assertive verified payoff, no unknown/missing-detail clause",
            "opening_mode": "direct_fact|conflict|counter_intuitive|developer_roi",
            "category_label": "optional factual label or empty",
            "direct_identifier": "exact evidence-backed identifier or empty",
            "editorial_inference": "question-form inference or empty",
            "hook_fact": "verified event", "conflict": "verified tension or rollout sequence",
            "surprise": "verified surprising detail or empty", "stakes": "verified affected audience/scope",
            "stance": "specific evidence-backed view", "payoff": "distinct verified answer",
            "hook_candidates": ["exactly three distinct hooks"],
            "hook_evidence_ids": ["existing evidence id"], "selected_hook": "one exact candidate",
            "shot_updates": [{
                "id": "existing shot id", "fact": "verified fact only",
                "interpretation": "internal editorial rationale; never rendered",
                "audience_copy": "optional declarative viewer-facing fact/impact; empty instead of reading or director advice",
                "target": "smallest exact contiguous excerpt from the same cited evidence that directly proves fact",
                "translation": "natural Chinese beside the exact selected target; empty when target is Chinese or absent",
                "full_translation": "natural complete Chinese translation for the non-Chinese root post only; empty for later shots",
                "relation_to_previous": "explicit new causal/chronological relation; use 此前/随后 for an earlier quoted post",
            }],
            "director_copy": {
                "editorial_thesis": "verified thesis", "viewer_tension": "answered viewing question",
                "attention_trigger": "named actor + verified event + verified stake",
                "emotion": "surprise|excitement|relief|alarm|conflict|opportunity", "emotion_intensity": "2|3",
            },
            "story_arc_updates": [{"role": "existing role", "claim": "verified replacement claim", "why_here": "correct progression"}],
        }
        patch, provenance = self._request_json([
            {"role": "system", "content": "Return one strict JSON patch only."},
            {"role": "user", "content": "\n".join([
                "Repair only the failing Chinese editorial copy for a BGM-only WeChat short video.",
                "Keep every shot id, evidence id, URL, visual_family, and ordering unchanged. "
                "For every shot, return the smallest exact contiguous target from its same cited evidence that directly proves the repaired fact; never point the highlight at a merely adjacent or topically related claim. "
                "translation and full_translation are viewer-facing Chinese copy: repair them when validation identifies literal, vague, or unnatural language.",
                "Delete unsupported concepts completely; do not replace them with synonyms or caveats. "
                "Do not mention missing mechanisms/details anywhere. When a feature name is opaque, tell only the verified actor, rollout, time, coverage, quote, capability, or result. End on the strongest verified scope/payoff. "
                "Keep branded feature and product identifiers exactly as written in evidence unless an official Chinese name is cited, but do not leave an opaque English identifier unexplained. At its first audience-facing occurrence, immediately state in natural Chinese what the evidence says it gives or does; never invent a dictionary-style Chinese product name. "
                "For model/product stories, keep the exact model/product name in headline and selected_hook. If a visible fact uses a specialist metric such as 拒绝率, add a concise plain-Chinese meaning in that first relevant shot (for 拒绝率: the proportion of prompts the model declines to answer), while preserving the sourced number. "
                "Preserve every finite scope exactly: one credited reset, trial, grant, exception, or temporary rollout stays finite and cannot become a permanent removal of recurring limits, costs, or waiting periods. "
                "Every shot must add a different fact. Do not paraphrase the same scope or result twice. "
                "Keep production metadata separate from visible copy. question, interpretation, relation_to_previous, retention_job, and director_brief are internal. audience_copy is the only optional second line shown to viewers: make it a declarative subject/fact/comparison/impact sentence, or return an empty string. Never put reading guidance, viewing guidance, learning advice, expected-weight instructions, or editor commands in audience_copy. "
                "When the root quotes an earlier post, shot 2 relation_to_previous must explicitly say 此前先预告/承诺, so the current root reads as 随后落地. "
                "Retain energy through specificity and named actors—not mystery about absent details. For retention_hook/payoff issues, choose direct_fact when the event itself is strong; otherwise use only a verified contradiction, surprise, consequence, ROI, or an explicit unresolved question. Do not manufacture outrage, an incumbent comparison, or certainty. Preserve opening_mode/category_label/direct_identifier/editorial_inference using the Radar contract.",
                (
                    "The last changing shot currently ends on an unknown or caution. Rewrite it to the strongest completed, verified event/result already in evidence (named actor + action + result). Do not mention pending appeals, missing policy, risk advice, or wait-and-see language in the final shot."
                    if "final changing shot" in validation_error else ""
                ),
                (
                    "The cited evidence proves availability/documentation, not a release event. Remove 发布、上线、宣布、正式推出 from every visible and internal editorial-copy field. Describe only the verified capability and current availability; do not replace 发布 with 文档上线."
                    if "release/launch wording" in validation_error else ""
                ),
                IT_TRANSLATION_CONTRACT,
                PLAIN_CHINESE_CONTRACT,
                "Validation errors: " + validation_error,
                "Required existing structure (write fresh copy; the invalid old wording is intentionally omitted): " + json.dumps(current, ensure_ascii=False),
                "Evidence: " + json.dumps(evidence_excerpt, ensure_ascii=False),
                "Return JSON matching: " + json.dumps(patch_schema, ensure_ascii=False),
            ])},
        # This patch covers every shot plus the attention loop.  Once exact
        # browser targets are included, reasoning-capable providers can spend
        # more than 2.4K tokens before emitting the strict JSON payload.
        ], max_tokens=4800)
        required_patch_fields = {
            "headline", "subheadline", "fixed_conclusion", "hook_fact", "conflict", "surprise",
            "stakes", "stance", "payoff", "hook_candidates", "hook_evidence_ids", "selected_hook",
            "shot_updates", "director_copy", "story_arc_updates",
        }
        missing_patch_fields = required_patch_fields - set(patch)
        expected_shot_ids = {str(item.get("id")) for item in shot_source if isinstance(item, dict) and item.get("id")}
        returned_shot_ids = {
            str(item.get("id")) for item in patch.get("shot_updates", [])
            if isinstance(item, dict) and item.get("id")
        }
        expected_arc = [item for item in director_source.get("story_arc", []) if isinstance(item, dict)]
        returned_arc = [item for item in patch.get("story_arc_updates", []) if isinstance(item, dict)]
        if missing_patch_fields or returned_shot_ids != expected_shot_ids:
            problem = "semantic repair patch is incomplete"
            if missing_patch_fields:
                problem += ": missing " + ", ".join(sorted(missing_patch_fields))
            if returned_shot_ids != expected_shot_ids:
                problem += "; shot ids must be " + ", ".join(sorted(expected_shot_ids))
            raise StoryDraftError(patch, ValueError(problem))
        repaired = deepcopy(invalid_draft)
        repaired_editorial = dict(repaired.get("editorial_brief") or {})
        for name in ("headline", "subheadline", "fixed_conclusion"):
            if patch.get(name):
                repaired_editorial[name] = patch[name]
        for name in ("opening_mode", "category_label", "direct_identifier", "editorial_inference"):
            if name in patch:
                repaired_editorial[name] = str(patch.get(name) or "")
        repaired_editorial["attention_strategy"] = {
            name: patch[name] for name in (
                "hook_fact", "conflict", "surprise", "stakes", "stance", "payoff",
                "hook_candidates", "hook_evidence_ids", "selected_hook",
            )
        }
        repaired["editorial_brief"] = repaired_editorial
        repaired["footer"] = repaired_editorial.get("fixed_conclusion", repaired.get("footer", ""))
        updates = {
            str(item.get("id")): item for item in patch.get("shot_updates", [])
            if isinstance(item, dict) and item.get("id")
        }
        shot_container = repaired_editorial.get("evidence_shots")
        if not isinstance(shot_container, list):
            shot_container = repaired.get("evidence_shots")
        if isinstance(shot_container, list):
            for shot in shot_container:
                update = updates.get(str(shot.get("id"))) if isinstance(shot, dict) else None
                if update:
                    shot["fact"] = str(update.get("fact") or shot.get("fact") or "")
                    shot["interpretation"] = str(update.get("interpretation") or shot.get("interpretation") or "")
                    shot["audience_copy"] = str(update.get("audience_copy") or "")
                    if "target" in update:
                        shot["target"] = str(update.get("target") or "")
                    if "translation" in update:
                        shot["translation"] = str(update.get("translation") or "")
                    if "full_translation" in update:
                        shot["full_translation"] = str(update.get("full_translation") or "")
                    shot["relation_to_previous"] = str(update.get("relation_to_previous") or shot.get("relation_to_previous") or "")
        director_copy = patch.get("director_copy")
        director_container = repaired_editorial.get("director_brief") or repaired.get("director_brief")
        if isinstance(director_copy, dict) and isinstance(director_container, dict):
            for name in ("editorial_thesis", "viewer_tension", "attention_trigger", "emotion", "emotion_intensity"):
                if director_copy.get(name):
                    director_container[name] = director_copy[name]
            fallback_updates = [
                item for item in patch.get("shot_updates", []) if isinstance(item, dict)
            ]
            for index, beat in enumerate(director_container.get("story_arc", [])):
                update = returned_arc[index] if index < len(returned_arc) else (
                    fallback_updates[min(index, len(fallback_updates) - 1)] if fallback_updates else {}
                )
                if isinstance(beat, dict):
                    beat["claim"] = str(update.get("claim") or beat.get("claim") or "")
                    if not update.get("claim") and update.get("fact"):
                        beat["claim"] = str(update["fact"])
                    beat["why_here"] = str(
                        update.get("why_here") or update.get("relation_to_previous")
                        or beat.get("why_here") or ""
                    )
        try:
            request = self._to_storyboard_request(packet, repaired)
        except (KeyError, TypeError, ValueError) as error:
            raise StoryDraftError(repaired, error) from error
        return request, provenance, repaired

    @staticmethod
    def _hook_only_repair(validation_error: str) -> bool:
        allowed = (
            "hook_opening", "hook_reveal", "hook_verdict", "composed GitHub hook",
            "the three cold-open screens", "ownerless capability openings",
            "keep GitHub hook_verdict/footer", "project_title",
            "repo_description_translation", "readme_claim_translation",
            "禁止译法", "hygiene 在", "honestly 在",
        )
        clauses = [item.strip() for item in validation_error.split(";") if item.strip()]
        return bool(clauses) and all(any(marker in clause for marker in allowed) for clause in clauses)

    def _repair_github_hook(
        self, packet: StoryWriterPacket, invalid_draft: dict[str, object], validation_error: str,
    ) -> tuple[StoryboardRequest, dict[str, object], dict[str, object]]:
        brief = dict(invalid_draft.get("github_brief") or {})
        repo_name = packet.candidate.title.rsplit("/", 1)[-1]
        compact = {
            key: brief.get(key, "") for key in (
                "subject_name", "subject_action", "subject_consequence",
                "background_actor", "background_action", "background_consequence",
                "hook_opening", "hook_reveal", "hook_verdict", "project_title",
                "repo_description_target", "repo_description_translation",
                "readme_claim_target", "readme_claim_translation",
            )
        }
        schema = {
            "hook_opening": "10–28 chars",
            "hook_reveal": "10–34 chars",
            "hook_verdict": "8–30 chars",
            "footer": "positive project-specific conclusion",
            "repo_description_translation": "concise technical Chinese, <=36 chars",
            "readme_claim_translation": "concise technical Chinese, <=36 chars",
        }
        patch, provenance = self._request_json([
            {"role": "system", "content": "Return one valid JSON object containing exactly six corrected strings."},
            {"role": "user", "content": "\n".join([
                "Repair only six visible-copy fields in a Chinese WeChat Channels GitHub story. Do not change facts or add caveats.",
                "The three screens must add different information: named event/action → concrete input/output or response → affirmative consequence/opinion.",
                f"Repository name: {repo_name}. Across hook_opening and hook_reveal, name the repository at least once. If background_actor is non-empty, opening names that actor/action and reveal names the repository response. Otherwise opening or reveal names the repository directly.",
                "Never repeat or paraphrase one screen in another. Footer must end on implemented value, never 但/不过/无法/不能/尽力/权衡/需验证.",
                "The two translations must translate their paired source target in concise IT Chinese. In privacy/content engineering, hygiene means 清理/净化, never 卫生. Do not write 翻译/译为.",
                "Length uses visual Chinese-character equivalents: one Han character is 1, ASCII letters/digits about 0.55, spaces 0.35. Keep the repository's real name; do not remove it merely because it has many ASCII letters. Verify all four lengths before returning; a phrase such as 自动生成高清短视频 is too short for the 10-unit reveal minimum and must include a new concrete detail.",
                "Current structured facts: " + json.dumps(compact, ensure_ascii=False),
                "Validation errors: " + validation_error,
                "Return: " + json.dumps(schema, ensure_ascii=False),
            ])},
        ], max_tokens=700)
        repaired = deepcopy(invalid_draft)
        repaired_brief = dict(repaired.get("github_brief") or {})
        for key in ("hook_opening", "hook_reveal", "hook_verdict"):
            repaired_brief[key] = str(patch.get(key, repaired_brief.get(key, ""))).strip()
        for key in ("repo_description_translation", "readme_claim_translation"):
            repaired_brief[key] = str(patch.get(key, repaired_brief.get(key, ""))).strip()
        repaired["github_brief"] = repaired_brief
        repaired["footer"] = str(patch.get("footer", repaired.get("footer", ""))).strip()
        try:
            request = self._to_storyboard_request(packet, repaired)
        except (KeyError, TypeError, ValueError) as error:
            raise StoryDraftError(repaired, error) from error
        return request, provenance, repaired

    def _generate_from_messages(
        self, packet: StoryWriterPacket, messages: list[dict[str, str]],
    ) -> tuple[StoryboardRequest, dict[str, object], dict[str, object]]:
        draft, provenance = self._request_json(messages, max_tokens=6000)
        try:
            request_object = self._to_storyboard_request(packet, draft)
        except (KeyError, TypeError, ValueError) as error:
            raise StoryDraftError(draft, error) from error
        return request_object, provenance, draft

    def _request_json(
        self, messages: list[dict[str, str]], max_tokens: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        if self.settings.provider_preferences:
            payload["provider"] = self.settings.provider_preferences
        if self.settings.provider == "openrouter":
            payload["reasoning"] = {
                "effort": os.environ.get("OPENROUTER_REASONING_EFFORT", "low"),
                "exclude": True,
            }
        endpoint = self.settings.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"}
        if self.settings.provider == "openrouter":
            headers.update({
                "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://github.com/video-factory"),
                "X-Title": os.environ.get("OPENROUTER_APP_NAME", "Video Factory"),
            })
        request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), method="POST", headers=headers)
        result: dict[str, object] | None = None
        draft: dict[str, object] | None = None
        last_error: Exception | None = None
        # Discounted providers occasionally return HTTP 200 with
        # finish_reason=error, empty content, or a prose error instead of the
        # requested JSON object. Treat those as bounded transient failures in
        # the same way as 429/5xx responses.
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = result.get("choices", [{}])[0].get("message", {}).get("content")
                finish_reason = result.get("choices", [{}])[0].get("finish_reason")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"empty provider content (finish_reason={finish_reason})")
                decoded = self._decode_json_object(content)
                if not isinstance(decoded, dict):
                    raise ValueError("provider content is not a JSON object")
                draft = decoded
                break
            except HTTPError as error:
                if error.code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"{self.settings.provider} story request failed: HTTP {error.code}") from error
                last_error = error
            except (URLError, OSError, http.client.HTTPException, json.JSONDecodeError, ValueError) as error:
                last_error = error
            if attempt < max_attempts - 1:
                time.sleep(0.6 * (attempt + 1))
        if result is None or draft is None:
            if isinstance(last_error, HTTPError):
                detail = f"HTTP {last_error.code}"
            else:
                detail = str(getattr(last_error, "reason", last_error))
            raise RuntimeError(
                f"{self.settings.provider} story request failed after {max_attempts} attempts: {detail}"
            ) from last_error
        provenance = {
            "provider": self.settings.provider, "model": result.get("model", self.settings.model),
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "usage": result.get("usage"),
        }
        return draft, provenance

    @staticmethod
    def _decode_json_object(content: str) -> dict[str, object]:
        """Accept a JSON object even when a provider wraps it in a code fence."""
        stripped = content.strip()
        try:
            value = json.loads(stripped)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
        try:
            value = json.loads(fenced)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise ValueError("provider content contains no JSON object")

    @staticmethod
    def _parse_plan(packet: StoryWriterPacket, draft: dict[str, object]) -> EditorialPlan:
        evidence_ids = {item.id for item in packet.evidence}
        selected = [str(item) for item in draft.get("selected_evidence_ids", [])]
        unknown_evidence = sorted(set(selected) - evidence_ids)
        if unknown_evidence:
            raise ValueError("research plan references unknown evidence ids: " + ", ".join(unknown_evidence))
        if not selected:
            selected = [item.id for item in packet.evidence]
        allowed_urls = [packet.candidate.source_url, *packet.candidate.linked_sources]
        requested = list(dict.fromkeys(str(item) for item in draft.get("requested_urls", [])))
        invented_urls = [url for url in requested if not any(_same_source_document(url, allowed) for allowed in allowed_urls)]
        if invented_urls:
            raise ValueError("research plan requested URLs outside the candidate: " + ", ".join(invented_urls))
        if len(requested) > 3:
            raise ValueError("research plan may request at most three URLs")
        angle = str(draft.get("angle", "")).strip()
        audience_value = str(draft.get("audience_value", "")).strip()
        if not angle or not audience_value:
            raise ValueError("research plan needs angle and audience_value")
        selection_reasons = [dict(item) for item in draft.get("selection_reasons", []) if isinstance(item, dict)][:5]
        source_text = "\n".join(item.quote for item in packet.evidence)
        selection_reasons = [
            reason for reason in selection_reasons
            if not _adds_unsupported_product_concept(str(reason.get("rationale", "")), source_text)
        ]
        for reason in selection_reasons:
            unknown = set(str(item) for item in reason.get("evidence_ids", [])) - evidence_ids
            if unknown:
                raise ValueError("selection reason references unknown evidence ids: " + ", ".join(sorted(unknown)))
        why_audience = str(draft.get("why_audience", "")).strip()
        audience_pain = str(draft.get("audience_pain_or_desire", "")).strip()
        if _adds_unsupported_product_concept(angle, source_text):
            angle = "围绕已归档原始事件：" + packet.candidate.title
        if _adds_unsupported_product_concept(audience_value, source_text):
            audience_value = "帮助技术观众确认已发生的变化、适用对象和直接可验证的结果"
        if _adds_unsupported_product_concept(why_audience, source_text):
            why_audience = "技术观众需要及时确认这项变化是否覆盖自己"
        if _adds_unsupported_product_concept(audience_pain, source_text):
            audience_pain = "及时知道新能力的落地时间和适用范围"
        if not selection_reasons:
            selection_reasons = [{
                "id": "verified-change", "dimension": "capability_shift",
                "rationale": "原始来源确认了一项刚刚发生、具有明确适用范围的变化",
                "evidence_ids": selected[:2],
            }]
        story_archetype = str(draft.get("story_archetype", "")).strip()
        expansion_dimensions = [str(item) for item in draft.get("expansion_dimensions", []) if str(item).strip()][:4]
        search_queries = [str(item) for item in draft.get("search_queries", []) if str(item).strip()][:3]
        if story_archetype == "people_change" and not re.search(
            r"\b(?:leave|leaving|left|depart(?:ed|ure|ing)?|join(?:ed|ing)?|"
            r"poach(?:ed|ing)?|found(?:ed|ing)?|co-?founder)\b|离开|离职|出走|加入|入职|挖角|创立|创办|联合创始",
            source_text, re.IGNORECASE,
        ):
            # A named person, compliment or CEO reply is not a people move.
            # Keep bounded entity research, but remove the unsupported talent-
            # flow pattern that would otherwise summon unrelated departures.
            story_archetype = (
                "event_chain"
                if (
                    any("quoted" in item.source_kind for item in packet.evidence)
                    and re.search(
                        r"\brepl(?:y|ied)|respond(?:ed|s|ing)?|suspend(?:ed|s|ing)?|appeal|"
                        r"we are hiring|回复|回应|质疑|封禁|封号|申诉|招聘",
                        source_text, re.IGNORECASE,
                    )
                )
                else "other"
            )
            expansion_dimensions = [
                item for item in expansion_dimensions if item not in {"people", "historical_pattern"}
            ]
            search_queries = [
                item for item in search_queries
                if not re.search(r"departure|leav(?:e|ing)|talent\s+(?:war|loss|flow)|离职|出走|人才流失", item, re.IGNORECASE)
            ]
        return EditorialPlan(
            angle=angle,
            audience_value=audience_value,
            selected_evidence_ids=selected,
            requested_urls=requested,
            unresolved_questions=[str(item) for item in draft.get("unresolved_questions", []) if str(item).strip()],
            ready_to_write=bool(draft.get("ready_to_write", False)),
            why_now=str(draft.get("why_now", "")).strip(),
            why_audience=why_audience,
            audience_pain_or_desire=audience_pain,
            selection_reasons=selection_reasons,
            expansion_dimensions=expansion_dimensions,
            context_questions=[str(item) for item in draft.get("context_questions", []) if str(item).strip()][:5],
            search_queries=search_queries,
            story_archetype=story_archetype,
        )
    @staticmethod
    def _to_storyboard_request(packet: StoryWriterPacket, draft: dict[str, object]) -> StoryboardRequest:
        answers = [NarrativeAnswer(**item) for item in draft.get("answers", [])]
        github_data = draft.get("github_walkthrough")
        walkthrough = None if not github_data else GitHubWalkthrough(
            **{**github_data, "key_modules": [GitHubModuleFocus(**item) for item in github_data.get("key_modules", [])]},
        )
        brief_data = draft.get("github_brief")
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
            canonicalize_github_brief(brief, packet.evidence)
        editorial_data = draft.get("editorial_brief")
        editorial_brief = None
        if editorial_data:
            editorial_data = dict(editorial_data)
            strategy_data = dict(editorial_data.get("attention_strategy") or {})
            if not strategy_data.get("selected_hook"):
                strategy_data["selected_hook"] = editorial_data.get("selected_hook") or draft.get("selected_hook") or ""
            strategy = AttentionStrategy(**strategy_data)
            shots: list[EvidenceShot] = []
            evidence_by_id = {item.id: item for item in packet.evidence}
            for raw in editorial_data.get("evidence_shots", draft.get("evidence_shots", [])):
                shot = dict(raw)
                linked_ids = list(shot.get("evidence_ids", []))
                for evidence_id in list(linked_ids):
                    evidence_item = evidence_by_id.get(evidence_id)
                    if evidence_item and evidence_item.source_kind == "x:visual_analysis":
                        parent_id = str(evidence_item.metadata.get("parent_image_id") or "")
                        if parent_id in evidence_by_id and parent_id not in linked_ids:
                            linked_ids.append(parent_id)
                shot["evidence_ids"] = linked_ids
                # Ignore a model-supplied material kind. The compiler derives
                # it from cited evidence plus semantic visual_family.
                shot["kind"] = _compile_evidence_shot_kind(shot, evidence_by_id)
                shot["duration"] = _coerce_model_float(shot.get("duration", 3), 3.0)
                shots.append(EvidenceShot(**shot))
            director_data = editorial_data.get("director_brief") or draft.get("director_brief")
            director_brief = None
            if director_data:
                director_data = dict(director_data)
                director_brief = DirectorBrief(
                    editorial_thesis=str(director_data.get("editorial_thesis", "")),
                    viewer_tension=str(director_data.get("viewer_tension", "")),
                    emotion=str(director_data.get("emotion", "")),
                    emotion_intensity=int(director_data.get("emotion_intensity", 1)),
                    selected_context_ids=[str(item) for item in director_data.get("selected_context_ids", [])],
                    story_arc=[StoryArcBeat(**item) for item in director_data.get("story_arc", [])],
                    recommended_duration=_coerce_model_float(
                        director_data.get("recommended_duration", packet.target_duration),
                        packet.target_duration,
                    ),
                    attention_trigger=str(director_data.get("attention_trigger", "")),
                )
            context_events: list[ContextEvent] = []
            for index, raw in enumerate(editorial_data.get("context_events", draft.get("context_events", [])), start=1):
                event = dict(raw)
                event["id"] = str(event.get("id") or f"context-model-{index}")
                context_events.append(ContextEvent(**event))
            graph = packet.context_graph or ContextGraph()
            existing_context_ids = {item.id for item in graph.events if item.id}
            new_context = [item for item in context_events if item.id not in existing_context_ids]
            graph.events.extend(new_context)
            # The research layer, not the prose model, decides which context
            # is mandatory. Otherwise every optional model-written event is
            # silently upgraded to required and crowds out the real setup.
            graph.discarded_context_ids = list(dict.fromkeys([
                *graph.discarded_context_ids, *(item.id for item in new_context if item.id),
            ]))
            editorial_brief = EditorialBrief(
                headline=str(editorial_data.get("headline", "")),
                subheadline=str(editorial_data.get("subheadline", "")),
                fixed_conclusion=str(editorial_data.get("fixed_conclusion", draft.get("footer", ""))),
                attention_strategy=strategy,
                subjects=[StorySubject(**item) for item in editorial_data.get("subjects", draft.get("subjects", []))],
                context_events=context_events,
                evidence_shots=shots,
                duration_target=_coerce_model_float(
                    editorial_data.get("duration_target", packet.target_duration), packet.target_duration,
                ),
                opportunity=packet.opportunity,
                context_graph=graph,
                director_brief=director_brief,
                opening_mode=str(editorial_data.get("opening_mode", "")),
                category_label=str(editorial_data.get("category_label", "")),
                direct_identifier=str(editorial_data.get("direct_identifier", "")),
                editorial_inference=str(editorial_data.get("editorial_inference", "")),
            )
            canonicalize_editorial_brief(editorial_brief, packet.evidence)
        if packet.topic_type == TopicType.GITHUB_PROJECT and brief:
            scenes = OpenAICompatibleStoryWriter._github_scenes_from_draft(
                packet, brief, draft.get("github_scenes", []),
            )
        elif editorial_brief:
            scenes = compile_evidence_shots(editorial_brief, packet.candidate)
        else:
            raise ValueError("non-GitHub model output must contain editorial_brief, not low-level scenes")
        fixed_hook = (
            compose_github_hook(brief) if packet.topic_type == TopicType.GITHUB_PROJECT and brief
            else (editorial_brief.attention_strategy.selected_hook if editorial_brief else "")
        )
        footer = editorial_brief.fixed_conclusion if editorial_brief else str(draft.get("footer", ""))
        return StoryboardRequest(
            id=str(draft.get("id", f"story-{packet.candidate.id}")), candidate=packet.candidate,
            topic_type=packet.topic_type, content_type=packet.content_type, evidence=packet.evidence,
            footer=footer, answers=answers, scenes=scenes,
            target_duration=packet.target_duration, github_walkthrough=walkthrough,
            github_brief=brief,
            editorial_brief=editorial_brief,
            fixed_hook=fixed_hook,
        )

    @staticmethod
    def _github_scenes_from_draft(
        packet: StoryWriterPacket, brief: GitHubProjectBrief, scene_drafts: list[dict[str, object]],
    ) -> list[SceneProposal]:
        errors: list[str] = []
        expected_stages = ["repo_identity", "readme_claim", "selected_focus", "selected_focus"]
        if len(scene_drafts) != 4:
            errors.append(f"github_scenes must contain exactly 4 items, got {len(scene_drafts)}")
        focus_map = {item.id: item for item in brief.focus_candidates}
        selected_ids = brief.selected_focus_ids
        used_focus_ids: list[str] = []
        scenes: list[SceneProposal] = []
        for index, item in enumerate(scene_drafts):
            stage = str(item.get("stage", ""))
            expected = expected_stages[index] if index < len(expected_stages) else ""
            if stage != expected:
                errors.append(f"github_scenes[{index}].stage must be {expected}, got {stage or 'missing'}")
            message = str(item.get("message", "")).strip()
            interpretation = str(item.get("interpretation", "")).strip()
            if not message or not interpretation:
                errors.append(f"github_scenes[{index}] needs message and interpretation")
            evidence_ids = [str(value) for value in item.get("evidence_ids", [])]
            beat_ids = [str(value) for value in item.get("beat_ids", [])]
            duration = _coerce_model_float(item.get("duration_hint", 0), 0.0)
            if duration <= 0 or duration > 5:
                errors.append(f"github_scenes[{index}].duration_hint must be >0 and <=5")
            focus_id = item.get("focus_id")
            translation = ""
            material_role = MaterialRole.PROOF
            if stage == "repo_identity":
                target = brief.repo_description_target
                translation = brief.repo_description_translation
                if focus_id not in {None, "", "null"}:
                    errors.append("repo_identity.focus_id must be null")
            elif stage == "readme_claim":
                target = brief.readme_claim_target
                translation = brief.readme_claim_translation
                if focus_id not in {None, "", "null"}:
                    errors.append("readme_claim.focus_id must be null")
            else:
                focus_key = str(focus_id or "")
                focus = focus_map.get(focus_key)
                if not focus:
                    errors.append(f"github_scenes[{index}].focus_id is not a candidate: {focus_key or 'missing'}")
                    target = focus_key
                else:
                    used_focus_ids.append(focus_key)
                    target = focus.target
                    translation = focus.translation
                    if focus.editorial_role in {"technical_edge", "adoption", "boundary"}:
                        material_role = MaterialRole.EXPLANATION
            scenes.append(SceneProposal(
                stage_name=stage or f"invalid-{index}",
                narration="internal editorial note: " + (message or stage),
                caption=message,
                material_role=material_role,
                visual_action=f"real browser focus: {target}",
                evidence_ids=evidence_ids,
                beat_ids=beat_ids,
                duration_hint=duration or 1,
                screen_fact=message,
                screen_interpretation=interpretation,
                highlight_translation=translation,
            ))
        if used_focus_ids != selected_ids:
            errors.append(f"selected_focus scenes must use selected_focus_ids in order: {selected_ids}")
        if errors:
            raise ValueError("; ".join(errors))
        return scenes


def packet_from_json(path: Path) -> StoryWriterPacket:
    data = json.loads(path.read_text(encoding="utf-8"))
    candidate_data = dict(data["candidate"])
    candidate_data["source_type"] = SourceType(candidate_data["source_type"])
    return StoryWriterPacket(
        candidate=Candidate(**candidate_data), evidence=[Evidence(**item) for item in data["evidence"]],
        topic_type=TopicType(data["topic_type"]), content_type=ContentType(data["content_type"]),
        target_duration=float(data["target_duration"]),
        editorial_direction=str(data.get("editorial_direction", "")),
    )
