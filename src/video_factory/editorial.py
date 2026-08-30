from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from dataclasses import dataclass

from .director import SceneProposal
from .github_editor import copy_width
from .models import (
    Candidate, CaptureCue, ContentType, CueAction, EditorialBrief, Evidence, EvidenceShot,
    EvidenceShotKind, MaterialRole, SourceType, TopicType,
)


INTERNAL_LABELS = ("证据带读", "关键结论", "仓库描述", "README声明", "trial", "boundary", "翻译：", "译为：")
EMPTY_HYPE = ("炸锅了", "彻底炸锅", "颠覆一切", "改写一切", "历史性时刻")
RADAR_OPENING_MODES = {"direct_fact", "conflict", "counter_intuitive", "developer_roi"}
RADAR_CATEGORY_LABELS = {"模型发布", "价格变化", "开源项目", "论文结果", "工具更新", "行业公告"}
RADAR_BANNED_COPY = (
    "反映了对齐机制的深度", "体现了生态多样性", "为从业者提供了新思考",
    "关于这一问题的探讨", "值得关注",
)


def _model_subject_aliases(name: str) -> tuple[str, ...]:
    """Return evidence-preserving display aliases for a model artifact name."""
    normalized = name.strip()
    if not normalized:
        return ()
    base = re.sub(
        r"(?:[-_/](?:uncensored|abliterated|fp\d+|int\d+|gguf|mlx|awq|gptq|bnb))+$",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip("-_/ ")
    return tuple(dict.fromkeys(value for value in (normalized, base) if value))


def _is_multi_model_price_evidence(evidence: list[Evidence]) -> bool:
    return any(
        "openrouter.ai/models" in item.url.casefold()
        and "discount=true" in item.url.casefold()
        for item in evidence
    )

# Explain one decision-critical specialist term at its first relevant shot.
# Definitions are audience UI copy, not claims about the selected source.
AUDIENCE_GLOSSARY: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("harness",), "harness：模型的测试与运行框架。", r"(?:测试|运行).{0,4}框架"),
    (("pass@1",), "pass@1：代码只生成一次就通过测试的比例。", r"只生成一次.{0,12}通过测试.{0,8}(?:比例|占比)"),
    (("首 Token 延迟", "TTFT"), "{term}：发出请求到看到第一个 Token 的等待时间。", r"发出请求.{0,12}第一个\s*Token.{0,10}(?:等待|时间)"),
    (("FP8", "INT8", "INT4", "量化"), "{term} 量化：用更低数值精度减少模型的显存和计算占用。", r"低.{0,6}(?:数值)?精度.{0,12}(?:减少|降低).{0,12}(?:显存|计算)"),
    (("激活参数量", "激活参数"), "{term}：模型每次生成时实际参与计算的参数规模。", r"每次.{0,8}(?:生成|回答).{0,12}(?:参与计算|实际调用).{0,8}参数"),
    (("Arena 分", "Arena分", "Elo"), "{term}：模型对战评测按胜负换算的相对分数。", r"模型对战.{0,12}(?:胜负|偏好).{0,12}(?:相对)?分数"),
    (("幻觉率",), "幻觉率：模型生成错误或无依据内容的占比。", r"(?:错误|无依据|编造).{0,12}(?:比例|占比)|(?:比例|占比).{0,12}(?:错误|无依据|编造)"),
    (("拒绝率", "拒答率"), "{term}：模型直接拒绝回答的请求占比。", r"(?:拒绝|不愿|不)(?:直接)?(?:回答|作答).{0,12}(?:比例|占比)|(?:比例|占比).{0,12}(?:拒绝|不回答|不作答)"),
)


def looks_like_internal_direction(value: str) -> bool:
    """Detect production/readership instructions in an audience-copy field.

    This is a reject-only contract check. It never rewrites model prose.
    Content-specific claims remain the model's job; failed copy is sent back
    through semantic repair.
    """
    text = re.sub(r"\s+", "", value)
    if not text:
        return False
    patterns = (
        r"(?:正确|更合理|真正)的?(?:读法|解读|理解方式|看法)",
        r"(?:解读|阅读|观看|理解|看待)(?:时|上|重点|角度)",
        r"(?:不必|无需|不要|别).{0,12}(?:被|纠结|害怕|吓退|过度解读)",
        r"(?:值得|建议|应该|应当).{0,12}(?:跟进学习|学习|关注|收藏|观察)",
        r"(?:重点|注意力).{0,8}(?:放在|放到|看|学|关注)",
        r"(?:读|看|学).{0,16}(?:顺序|重点|方式)",
        r"(?:降低|提高).{0,8}(?:预期|权重)",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _hook_retention_score(value: str) -> float:
    """Rank only model-written candidates; never invent hook copy here."""
    score = 0.0
    weights = {
        "兑现": 3, "终于": 3, "准时": 2.5, "按时": 2.5, "落地": 2,
        "此前": 1.5, "随后": 1.5, "离开": 2, "出走": 2, "联手": 1.5,
        "全部": 1.5, "所有": 1.2, "直降": 2.5, "反转": 2, "首次": 2,
        "正式可用": 0.2, "值得关注": -2, "迎来新变化": -1.5,
        "权衡": -2.5, "未公布": -3, "待公布": -3, "需验证": -3,
    }
    for marker, weight in weights.items():
        if marker in value:
            score += weight
    if re.search(r"\d", value):
        score += 1
    if "：" in value or ":" in value:
        score += 0.5
    visual_length = len(re.sub(r"\s+", "", value))
    if 12 <= visual_length <= 38:
        score += 0.5
    elif visual_length > 52:
        score -= 1
    return score


def _clip_radar_copy(value: str, limit: float) -> str:
    """Shorten model-written copy without splitting ASCII identifiers."""
    text = re.sub(r"\s+", " ", value).strip()
    if copy_width(text) <= limit:
        return text
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._+/$%:-]*|.", text)
    kept: list[str] = []
    for token in tokens:
        candidate = "".join([*kept, token]).strip()
        if copy_width(candidate) > limit:
            break
        kept.append(token)
    clipped = "".join(kept).rstrip(" ，。；、:：!?！？")
    for marker in ("。", "！", "？", "；", "，", "：", ";", ",", ":"):
        boundary = clipped.rfind(marker)
        prefix = clipped[:boundary].rstrip() if boundary >= 0 else ""
        if prefix and copy_width(prefix) >= min(12, limit * 0.6):
            return prefix
    return clipped


def _complete_radar_conclusion(brief: EditorialBrief, limit: float = 40) -> str:
    """Prefer a complete model-written takeaway over a severed long sentence.

    Radar canonicalization is allowed to select existing copy, not invent new
    claims.  A raw width clip can silently change meaning (for example, "only
    three things" followed by two items) or leave an ASCII name hanging at the
    rail edge.  The writer already supplies stance and payoff alternatives, so
    use the first complete candidate that fits before falling back to clipping.
    """
    candidates = (
        brief.fixed_conclusion,
        brief.attention_strategy.stance,
        brief.attention_strategy.payoff,
    )
    for index, candidate in enumerate(candidates):
        normalized = re.sub(r"\s+", " ", candidate).strip().rstrip("。！？!?；;")
        severed_list_name = index == 0 and bool(re.search(r"、[A-Z][A-Za-z0-9._-]*$", normalized))
        if severed_list_name:
            continue
        if normalized and copy_width(normalized) <= limit:
            return normalized
    return _clip_radar_copy(brief.fixed_conclusion, limit)


def _looks_like_radar_fragment(value: str) -> bool:
    compact = re.sub(r"\s+", " ", value).strip().rstrip("，,：:")
    return bool(re.search(r"(?:模|直接|每秒|从|至|为|在|与|和|或|的|把|将|到|让|用)$", compact))


def _canonicalize_radar_contract(brief: EditorialBrief) -> None:
    strategy = brief.attention_strategy
    if brief.opening_mode not in RADAR_OPENING_MODES:
        brief.opening_mode = "direct_fact"
    if brief.category_label not in RADAR_CATEGORY_LABELS:
        brief.category_label = ""

    subject_names = [item.name.strip() for item in brief.subjects if item.name.strip()]

    def headline_copy(value: str) -> str:
        has_subject = any(name.casefold() in value.casefold() for name in subject_names)
        clipped = _clip_radar_copy(value, 28 if has_subject else 20)
        if copy_width(clipped) > 20 and not any(
            name.casefold() in clipped.casefold() for name in subject_names
        ):
            clipped = _clip_radar_copy(clipped, 20)
        return clipped

    original_candidates = list(strategy.hook_candidates)
    original_headline = brief.headline
    strategy.hook_candidates = [headline_copy(item) for item in original_candidates]
    if strategy.selected_hook in original_candidates:
        strategy.selected_hook = strategy.hook_candidates[original_candidates.index(strategy.selected_hook)]
    elif strategy.hook_candidates:
        strategy.selected_hook = strategy.hook_candidates[0]
    strategy.selected_hook = headline_copy(strategy.selected_hook)
    if _looks_like_radar_fragment(strategy.selected_hook):
        complete_alternatives = [
            headline_copy(original_headline),
            *(headline_copy(item) for item in original_candidates),
        ]
        complete_alternatives = [
            item for item in complete_alternatives
            if item and not _looks_like_radar_fragment(item)
        ]
        if complete_alternatives:
            strategy.selected_hook = max(complete_alternatives, key=_hook_retention_score)
    if strategy.hook_candidates and strategy.selected_hook not in strategy.hook_candidates:
        strategy.hook_candidates[0] = strategy.selected_hook

    brief.headline = headline_copy(brief.headline)
    brief.fixed_conclusion = _complete_radar_conclusion(brief, 40)
    for index, shot in enumerate(brief.evidence_shots):
        shot.narrative_beat = (
            "opening" if index == 0 else
            "takeaway" if index == len(brief.evidence_shots) - 1 else
            "proof"
        )
        audience_is_glossary = is_audience_glossary_definition(shot.audience_copy)
        if audience_is_glossary and copy_width(shot.audience_copy) <= 32:
            # The one plain-language definition is more valuable than a
            # repeated long fact line. Reserve its full wording, then compact
            # the evidence headline into the remaining screen budget.
            audience_width = copy_width(shot.audience_copy)
            shot.fact = _clip_radar_copy(shot.fact, max(8.0, 40 - audience_width))
            shot.audience_copy = re.sub(r"\s+", " ", shot.audience_copy).strip()
        else:
            shot.fact = _clip_radar_copy(shot.fact, 40)
            remaining = max(0.0, 40 - copy_width(shot.fact))
            audience = re.sub(r"\s+", " ", shot.audience_copy).strip()
            shot.audience_copy = (
                audience
                if remaining >= 4 and copy_width(audience) <= remaining
                and not _looks_like_radar_fragment(audience)
                else ""
            )
        if shot.full_translation:
            shot.full_translation = _clip_radar_copy(
                shot.full_translation,
                120 if index == 0 and shot.kind == EvidenceShotKind.TWEET_CARD else 40,
            )
        if shot.translation:
            shot.translation = _clip_radar_copy(shot.translation, 40)

    if brief.director_brief and brief.context_graph:
        required = set(brief.context_graph.required_context_ids)
        brief.context_graph.discarded_context_ids = [
            item for item in brief.context_graph.discarded_context_ids if item not in required
        ]
        brief.director_brief.selected_context_ids = list(dict.fromkeys([
            *brief.director_brief.selected_context_ids,
            *brief.context_graph.required_context_ids,
        ]))
    if (
        brief.editorial_inference != strategy.selected_hook
        or not brief.editorial_inference.endswith(("?", "？"))
    ):
        brief.editorial_inference = ""


def _chain_action_markers(value: str) -> list[str]:
    markers = (
        "不解", "奇怪", "质疑", "征集", "澄清", "否认", "不在", "回复", "回应", "表示",
        "招人", "招聘", "停用", "被封", "封号", "封禁", "申诉",
    )
    return [marker for marker in markers if marker in value]


def _chain_display_name(value: str) -> str:
    return re.split(r"\s*[（(]|\s*@", value, maxsplit=1)[0].strip()


def _chain_source_target(subject_name: str, evidence_items: list[Evidence]) -> str:
    """Extract one exact self-contained source block for a subject action."""
    for item in evidence_items:
        text = item.quote
        if item.source_kind == "x:visual_analysis":
            try:
                payload = json.loads(text)
                text = str(payload.get("visible_text") or text)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        match = re.search(re.escape(subject_name), text, re.IGNORECASE)
        if not match:
            continue
        tail = text[match.start():]
        block = re.split(r"\n\s*\n", tail, maxsplit=1)[0].strip()
        if len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", block)) >= 4:
            return block[:320]
    return ""


def _repair_multi_party_event_chain(brief: EditorialBrief, evidence: list[Evidence]) -> None:
    """Deterministic fallback when prose repair keeps dropping a bridge actor.

    It compiles only model-written subject actions that already cite archived
    evidence. No new factual claim is authored here.
    """
    if not (
        brief.opportunity
        and brief.opportunity.story_archetype == "event_chain"
        and len(brief.subjects) >= 4
        and brief.evidence_shots
    ):
        return
    evidence_by_id = {item.id: item for item in evidence}
    root = brief.evidence_shots[0]
    later = brief.evidence_shots[1:]

    def covers(shot: EvidenceShot, subject) -> bool:
        copy = " ".join((shot.fact, shot.audience_copy, shot.translation, shot.full_translation))
        name = _chain_display_name(subject.name)
        if not name or name.casefold() not in copy.casefold():
            return False
        markers = _chain_action_markers(subject.action)
        return not markers or any(marker in copy for marker in markers)

    ordered = [root]
    used_ids: set[str] = {root.id}
    for index, subject in enumerate(brief.subjects[1:4], start=1):
        existing = next((shot for shot in later if shot.id not in used_ids and covers(shot, subject)), None)
        if existing is not None:
            original_id = existing.id
            existing.id = (
                f"chain-{index}-{re.sub(r'[^a-z0-9]+', '-', _chain_display_name(subject.name).casefold()).strip('-') or index}"
            )
            existing.fact = f"{_chain_display_name(subject.name)}：{subject.action}"
            ordered.append(existing)
            used_ids.add(original_id)
            continue
        cited = [evidence_by_id[item] for item in subject.evidence_ids if item in evidence_by_id]
        target = _chain_source_target(_chain_display_name(subject.name), cited)
        linked_ids = list(subject.evidence_ids)
        for item in cited:
            if item.source_kind == "x:visual_analysis":
                parent = str(item.metadata.get("parent_image_id") or "")
                if parent in evidence_by_id and parent not in linked_ids:
                    linked_ids.append(parent)
        reason_ids = [
            reason.id for reason in brief.opportunity.selection_reasons
            if set(reason.evidence_ids) & set(subject.evidence_ids)
        ]
        ordered.append(EvidenceShot(
            id=f"chain-{index}-{re.sub(r'[^a-z0-9]+', '-', _chain_display_name(subject.name).casefold()).strip('-') or index}",
            kind=EvidenceShotKind.BROWSER_SECTION,
            question=f"{_chain_display_name(subject.name)} 做了什么？",
            fact=f"{_chain_display_name(subject.name)}：{subject.action}",
            interpretation="补齐多方事件链中不可缺少的一步",
            evidence_ids=linked_ids,
            beat_ids=["evidence_context"],
            source_url=cited[0].url if cited else "",
            target=target,
            translation=(
                f"{_chain_display_name(subject.name)}：{subject.action}"
                if target and re.search(r"[A-Za-z]", target) else ""
            ),
            duration=2.2,
            relation_to_previous="按公开对话顺序推进到下一位参与者",
            visual_family="quote_card" if index % 2 else "timeline",
            retention_job="turn" if index < 3 else "impact",
            selection_reason_ids=reason_ids,
        ))

    scope = next((
        shot for shot in later
        if shot.id not in used_ids and "scope" in shot.beat_ids
    ), None)
    if scope is not None and len(ordered) < 5:
        ordered.append(scope)
    if brief.context_graph:
        required_context = set(brief.context_graph.required_context_ids)
        for shot in ordered:
            linked = set(shot.evidence_ids)
            inherited = [
                event.id for event in brief.context_graph.events
                if event.id in required_context and linked & set(event.evidence_ids)
            ]
            shot.context_event_ids = list(dict.fromkeys([
                *shot.context_event_ids, *inherited,
            ]))
        for event in brief.context_graph.events:
            if event.id not in required_context or any(
                event.id in shot.context_event_ids for shot in ordered
            ):
                continue
            actor_key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", event.actor.casefold())
            target = next((
                shot for shot in ordered[1:]
                if actor_key and (
                    actor_key in re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", shot.fact.casefold())
                    or re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", shot.fact.casefold()) in actor_key
                )
            ), ordered[0] if event.relation == "background" else ordered[1])
            target.evidence_ids = list(dict.fromkeys([
                *target.evidence_ids, *event.evidence_ids,
            ]))
            target.context_event_ids = list(dict.fromkeys([
                *target.context_event_ids, event.id,
            ]))
    brief.evidence_shots = ordered


def canonicalize_editorial_brief(brief: EditorialBrief, evidence: list[Evidence]) -> None:
    """Repair evidence linkage, never prose, when the model chose a valid fact.

    Cheap models sometimes cite the root announcement for a detail proven by
    a related same-author post. The program may attach that already-archived
    proof deterministically; it may not invent or soften the claim itself.
    """
    evidence_by_id = {item.id: item for item in evidence}
    if brief.direct_identifier.strip():
        corpus = "\n".join([*(item.url for item in evidence), *(item.quote for item in evidence)])
        needle = re.sub(
            r"^(?:HF:\s*|pip install\s+|docker pull\s+)",
            "", brief.direct_identifier.strip(), flags=re.IGNORECASE,
        )
        if not needle or needle.casefold() not in corpus.casefold():
            brief.direct_identifier = ""
    # Multimodal analysis explains an archived image but is not itself a
    # renderable asset. Attach its parent before presentation compilation so
    # the first use renders the real pixels; later reuse becomes a derived
    # evidence card instead of opening a raw CDN image and searching for OCR
    # text in the browser DOM.
    for shot in brief.evidence_shots:
        linked = list(shot.evidence_ids)
        for evidence_id in list(linked):
            item = evidence_by_id.get(evidence_id)
            if item and item.source_kind == "x:visual_analysis":
                parent_id = str(item.metadata.get("parent_image_id") or "")
                if parent_id in evidence_by_id and parent_id not in linked:
                    linked.append(parent_id)
        shot.evidence_ids = linked

    _repair_multi_party_event_chain(brief, evidence)
    if brief.opening_mode:
        _canonicalize_radar_contract(brief)
    strategy = brief.attention_strategy
    if not strategy.selected_hook and strategy.hook_candidates:
        strategy.selected_hook = strategy.hook_candidates[0]
    root_actor_markers = {
        value.strip().lstrip("@").casefold()
        for item in evidence if item.source_kind == "x:thread_post"
        for value in (
            str(item.metadata.get("author_name") or ""),
            str(item.metadata.get("author_handle") or ""),
        )
        if value.strip().lstrip("@")
    }

    def candidate_score(value: str) -> float:
        score = _hook_retention_score(value)
        folded = value.casefold()
        if any(marker in folded for marker in root_actor_markers if len(marker) >= 4):
            score += 3.0
        return score

    if strategy.hook_candidates and strategy.selected_hook in strategy.hook_candidates:
        model_aliases = [
            alias for subject in brief.subjects
            if subject.subject_type.strip().casefold() == "model" and subject.name.strip()
            for alias in _model_subject_aliases(subject.name)
        ]
        model_named_hooks = [
            hook for hook in strategy.hook_candidates
            if any(name.casefold() in hook.casefold() for name in model_aliases)
        ]
        if not _is_multi_model_price_evidence(evidence) and model_named_hooks and not any(
            name.casefold() in strategy.selected_hook.casefold() for name in model_aliases
        ):
            # Select already-written model-specific copy instead of inventing
            # or prefixing prose in the execution layer.
            strategy.selected_hook = max(model_named_hooks, key=candidate_score)
        selected_score = candidate_score(strategy.selected_hook)
        strongest = max(strategy.hook_candidates, key=candidate_score)
        if candidate_score(strongest) >= selected_score + 2.0:
            strategy.selected_hook = strongest
    _inject_audience_glossary(brief)
    if brief.opening_mode:
        # Glossary injection and hook ranking can change visible copy after the
        # first normalization pass. Re-apply the deterministic width contract
        # before validation so those useful additions cannot trigger an LLM
        # repair loop.
        _canonicalize_radar_contract(brief)
    hook_claims = "\n".join([
        strategy.hook_fact, strategy.conflict, strategy.stakes, strategy.stance,
        strategy.selected_hook, brief.headline, brief.subheadline,
    ])
    all_source = "\n".join(item.quote for item in evidence)
    if re.search(r"(?:离开|出走|失去|离职).{0,8}(?:Google|谷歌)|(?:Google|谷歌).{0,8}(?:离开|出走|失去|离职)", hook_claims, re.IGNORECASE):
        supporting = [
            item.id for item in evidence if re.search(
                r"(?:leave|leaving|left|departure|departing|last day).{0,24}Google|Google.{0,24}(?:leave|leaving|left|departure|last day)|(?:离开|出走|失去|离职).{0,8}(?:Google|谷歌)",
                item.quote, re.IGNORECASE,
            )
        ]
        strategy.hook_evidence_ids = list(dict.fromkeys([*strategy.hook_evidence_ids, *supporting]))
        for subject in brief.subjects:
            subject_copy = "\n".join((subject.action, subject.consequence))
            if re.search(r"(?:离开|出走|失去|离职).{0,8}(?:Google|谷歌)|(?:Google|谷歌).{0,8}(?:离开|出走|失去|离职)", subject_copy, re.IGNORECASE):
                subject.evidence_ids = list(dict.fromkeys([*subject.evidence_ids, *supporting]))

    # A verified incumbent-history beat is context, not the payoff. Keep the
    # complete root first, then establish the pattern before revealing the new
    # company's capability and impact. This reorders evidence; it never writes
    # or changes a factual claim.
    if (
        brief.opportunity and brief.opportunity.story_archetype == "people_change"
        and brief.context_graph and brief.context_graph.pattern_context_ids
    ):
        pattern_ids = set(brief.context_graph.pattern_context_ids)
        pattern_index = next((
            index for index, shot in enumerate(brief.evidence_shots[1:], start=1)
            if set(shot.context_event_ids) & pattern_ids
        ), None)
        if pattern_index is not None and pattern_index != 1:
            brief.evidence_shots.insert(1, brief.evidence_shots.pop(pattern_index))

    _promote_relevant_source_images(brief, evidence)
    _resolve_external_evidence_pages(brief, evidence)
    _compile_flash_presentation(brief, evidence)


def _resolve_external_evidence_pages(brief: EditorialBrief, evidence: list[Evidence]) -> None:
    """Route real-source shots to the external page nearest their target.

    A fetched X thread may expose the paper/product URL only inside its
    archived text. The model is allowed to select the target, but the program
    owns URL routing and must not try to find an arXiv title on x.com.
    """
    evidence_by_id = {item.id: item for item in evidence}
    derived = {"quote_card", "timeline", "impact_card", "stat_card", "tweet", "quoted_post", "source_image"}
    url_pattern = re.compile(
        r"https?://[^\s<>\]\[\"']+|(?<![@\w])(?:[a-z0-9-]+\.)+[a-z]{2,}/[^\s<>\]\[\"']+",
        re.IGNORECASE,
    )
    excluded_hosts = {"x.com", "twitter.com", "t.co", "pbs.twimg.com"}
    for shot in brief.evidence_shots:
        target = shot.target.strip()
        if not target:
            continue
        direct_page = next((
            evidence_by_id[evidence_id]
            for evidence_id in shot.evidence_ids
            if evidence_id in evidence_by_id
            and evidence_by_id[evidence_id].source_kind in {"web:primary_page", "web:page"}
            and target.casefold() in re.sub(
                r"\[([^\]]+)\]\([^)]+\)", r"\1",
                evidence_by_id[evidence_id].quote.replace("**", ""),
            ).casefold()
        ), None)
        if direct_page is not None:
            shot.source_url = direct_page.url
            continue
        if shot.visual_family in derived:
            continue
        candidates: list[tuple[int, str]] = []
        for evidence_id in shot.evidence_ids:
            item = evidence_by_id.get(evidence_id)
            # External-URL proximity is a routing hint only for social posts
            # and sources opened from them. A normal official/news page often
            # links a partner beside its own prose; that does not move the
            # prose onto the partner's site.
            if not item or not (
                item.source_kind.startswith("x:")
                or item.source_kind == "web:agent_primary_source"
            ):
                continue
            target_index = item.quote.casefold().find(target.casefold())
            for match in url_pattern.finditer(item.quote):
                raw = match.group(0).rstrip(".,;:!?)]}。；，！？…")
                if "…" in raw or raw.endswith("..."):
                    continue
                normalized = raw if raw.startswith(("http://", "https://")) else "https://" + raw
                from urllib.parse import urlparse

                if (urlparse(normalized).hostname or "").casefold() in excluded_hosts:
                    continue
                distance = abs(match.start() - target_index) if target_index >= 0 else 1_000_000
                candidates.append((distance, normalized))
        if candidates:
            distance, url = min(candidates)
            # A long thread may mention a paper only at the end. That does
            # not make the paper page the source for every earlier sentence.
            # Route only when the selected excerpt is locally adjacent to the
            # external URL; otherwise keep the archived thread as a card.
            if distance <= 600:
                shot.source_url = url
            elif shot.source_url.startswith(("https://x.com/", "https://twitter.com/")):
                shot.visual_family = "quote_card"
                shot.kind = EvidenceShotKind.BROWSER_SECTION


def _promote_relevant_source_images(brief: EditorialBrief, evidence: list[Evidence]) -> None:
    """Use an official high-value image when it directly depicts a written beat."""
    images = [
        item for item in evidence
        if item.source_kind in {"web:source_image", "x:media_photo"}
        and item.metadata.get("editorial_priority") == "high"
    ]
    if not images:
        return

    role_patterns = {
        "team": r"团队|创始|成员|研究者|found(?:er|ing)|team",
        "benchmark": r"跑分|基准|结果|对比|benchmark|result|comparison",
        "architecture": r"架构|技术路线|系统|architecture|diagram",
        "product": r"产品|界面|演示|效果|product|demo|screenshot",
    }
    used_image_keys: set[str] = set()
    for shot in brief.evidence_shots[1:]:
        cited_images = [
            item for item in images
            if item.id in shot.evidence_ids
        ]
        # An image explicitly selected by the editor owns that visual. Mark
        # the underlying asset as consumed before considering promotion so a
        # duplicate evidence record cannot steal a later text-evidence shot.
        if shot.kind == EvidenceShotKind.IMAGE or shot.visual_family == "source_image":
            used_image_keys.update(_source_image_identity(item) for item in cited_images)
            continue
        shot_copy = " ".join((shot.fact, shot.audience_copy, shot.interpretation, shot.question)).casefold()
        for image in images:
            image_key = _source_image_identity(image)
            if image_key in used_image_keys:
                continue
            role = str(image.metadata.get("visual_role") or "source_image")
            pattern = role_patterns.get(role)
            parent_url = str(image.metadata.get("parent_source_url") or "")
            parent_ids = {
                item.id for item in evidence
                if parent_url and item.source_kind not in {"web:source_image", "x:media_photo"}
                and item.url.rstrip("/") == parent_url.rstrip("/")
            }
            if pattern and re.search(pattern, shot_copy, re.IGNORECASE) and (
                not parent_ids or set(shot.evidence_ids) & parent_ids
            ):
                shot.evidence_ids = [image.id, *[item for item in shot.evidence_ids if item != image.id]]
                shot.kind = EvidenceShotKind.IMAGE
                shot.visual_family = "source_image"
                shot.source_url = image.url
                shot.target = ""
                shot.translation = ""
                used_image_keys.add(image_key)
                break


def _source_image_identity(item: Evidence) -> str:
    """Identify one visual across duplicate acquisition evidence records."""
    if item.sha256:
        return f"sha256:{item.sha256}"
    if item.captured_asset:
        return f"asset:{item.captured_asset}"
    return f"url:{item.url.split('?', 1)[0].rstrip('/')}"


def _compile_flash_presentation(brief: EditorialBrief, evidence: list[Evidence]) -> None:
    """Own pacing/material presentation below the editorial model boundary.

    Visual de-duplication applies to every format; only the final timing pass
    is specific to flash videos.
    """
    if not brief.evidence_shots:
        return
    evidence_by_id = {item.id: item for item in evidence}
    supported = {
        "tweet", "quoted_post", "official_page", "source_image", "product_ui", "chart", "timeline",
        "code", "paper", "quote_card", "impact_card", "stat_card",
    }
    derived_cycle = ("quote_card", "timeline", "impact_card", "stat_card")

    # A people-change flash must end on consequence rather than an earlier
    # implementation step. Reorder an already-written impact beat; never
    # synthesize a new claim in the execution layer.
    if brief.duration_target <= 15 and brief.opportunity and brief.opportunity.story_archetype == "people_change":
        impact_index = next((
            index for index in range(len(brief.evidence_shots) - 1, 0, -1)
            if "impact" in brief.evidence_shots[index].beat_ids
        ), None)
        if impact_index is not None and impact_index != len(brief.evidence_shots) - 1:
            brief.evidence_shots.append(brief.evidence_shots.pop(impact_index))

    def cited_kinds(shot: EvidenceShot) -> set[str]:
        return {evidence_by_id[item].source_kind for item in shot.evidence_ids if item in evidence_by_id}

    def derived_for(shot: EvidenceShot, used: set[str], previous: str) -> str:
        preferred = {
            "reveal": "quote_card", "contrast": "timeline", "turn": "timeline",
            "impact": "stat_card" if re.search(r"\d", shot.fact) else "impact_card",
            "payoff": "impact_card",
        }.get(shot.retention_job, "quote_card")
        choices = (preferred, *derived_cycle)
        return next((item for item in choices if item != previous and item not in used), next(item for item in choices if item != previous))

    # One strong source image is a reveal, not wallpaper. Cheap models may
    # cite it again on the payoff simply because it is available. Keep the
    # first use and compile later claims back to an evidence-backed card.
    used_source_image_keys: set[str] = set()
    for shot in brief.evidence_shots:
        image_ids = {
            item for item in shot.evidence_ids
            if item in evidence_by_id
            and evidence_by_id[item].source_kind in {"web:source_image", "x:media_photo"}
        }
        repeated = {
            item for item in image_ids
            if _source_image_identity(evidence_by_id[item]) in used_source_image_keys
        }
        if repeated:
            shot.evidence_ids = [item for item in shot.evidence_ids if item not in repeated]
            if not shot.evidence_ids:
                parent_urls = {
                    str(evidence_by_id[item].metadata.get("parent_source_url") or "") for item in repeated
                }
                shot.evidence_ids = [
                    item.id for item in evidence
                    if item.source_kind not in {"web:source_image", "x:media_photo"}
                    and item.url in parent_urls
                ][:1]
            shot.kind = EvidenceShotKind.BROWSER_SECTION
            if shot.source_url in {evidence_by_id[item].url for item in repeated}:
                shot.source_url = ""
            remaining_kinds = cited_kinds(shot)
            if shot.visual_family == "source_image" or (
                remaining_kinds and remaining_kinds <= {"x:visual_analysis", "github:visual_analysis"}
            ):
                shot.visual_family = "impact_card" if shot.retention_job in {"impact", "payoff"} else "quote_card"
        used_source_image_keys.update(
            _source_image_identity(evidence_by_id[item]) for item in image_ids - repeated
        )

    for index, shot in enumerate(brief.evidence_shots):
        kinds = cited_kinds(shot)
        if shot.visual_family not in supported:
            if any(kind.startswith("x:") for kind in kinds):
                shot.visual_family = "quoted_post" if any(kind == "x:quoted_post" for kind in kinds) else "tweet"
            elif any(kind.startswith("paper:") for kind in kinds):
                shot.visual_family = "paper"
            else:
                shot.visual_family = "official_page"
        if index == 0 and any(kind.startswith("x:") for kind in kinds):
            shot.visual_family = "tweet"

    used: set[str] = set()
    for index, shot in enumerate(brief.evidence_shots):
        previous = brief.evidence_shots[index - 1].visual_family if index else ""
        same_treatment = index and previous == shot.visual_family and (
            set(brief.evidence_shots[index - 1].evidence_ids) == set(shot.evidence_ids)
            or shot.visual_family in {
                "official_page", "tweet", "quoted_post",
                "quote_card", "timeline", "impact_card", "stat_card",
            }
        )
        if index and same_treatment:
            shot.visual_family = derived_for(shot, used, previous)
        used.add(shot.visual_family)

    if brief.duration_target > 15:
        # A long source post needs a longer first hold, but still remains one
        # complete card. Later beats must earn their own visual treatment.
        root_length = max((len(item.quote) for item in evidence if item.id in brief.evidence_shots[0].evidence_ids), default=0)
        if root_length > 1200:
            brief.evidence_shots[0].duration = 5.0
        return

    # Timing is an execution parameter. Preserve enough time for the complete
    # first X card, then distribute the remaining flash budget evenly.
    count = len(brief.evidence_shots)
    first_is_tweet = brief.evidence_shots[0].kind == EvidenceShotKind.TWEET_CARD
    first_duration = min(3.2 if first_is_tweet else 2.8, max(1.3, brief.duration_target - 1.3 * (count - 1)))
    remaining_duration = (brief.duration_target - first_duration) / max(1, count - 1)
    later_duration = min(2.8, max(1.3, remaining_duration))
    brief.evidence_shots[0].duration = round(first_duration, 3)
    for shot in brief.evidence_shots[1:]:
        shot.duration = round(later_duration, 3)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    topic_type: TopicType
    content_type: ContentType
    target_duration: float
    reason: str


def route_content(
    candidate: Candidate,
    evidence: list[Evidence],
    topic_override: TopicType | None = None,
    format_override: ContentType | None = None,
    duration_override: float | None = None,
) -> RouteDecision:
    """Cheap deterministic routing before the bounded editorial model loop.

    Source kind and story topic are intentionally independent: an X post can
    announce a company, model, tool, paper, or contain a personal practice.
    """
    text = "\n".join([candidate.title, *(item.quote[:5000] for item in evidence)]).casefold()
    # Attachments and later research explain an X event but must not silently
    # change its content type. A screenshot mentioning a model/API inside an
    # account dispute is still an event/practice post, not a model launch.
    tweet_route_text = "\n".join([
        candidate.title,
        *(
            item.quote[:5000] for item in evidence
            if item.source_kind in {"x:thread_post", "x:quoted_post"}
        ),
    ]).casefold()
    url = candidate.source_url.casefold()
    openrouter_discount_page = (
        "openrouter.ai/models" in url and "discount=true" in url
    )
    if topic_override:
        topic = topic_override
        reason = "explicit topic override"
    elif openrouter_discount_page:
        topic, reason = TopicType.MODEL_OR_PRODUCT, "OpenRouter discounted-model price monitor"
    elif candidate.source_type == SourceType.PAPER or url.endswith(".pdf") or any(
        marker in url for marker in ("arxiv.org", "openreview.net")
    ):
        topic, reason = TopicType.RESEARCH_OR_BENCHMARK, "paper/PDF source"
    elif candidate.source_type == SourceType.TWEET:
        if any(marker in tweet_route_text for marker in ("founding ", "founded ", "funding round", "leaving google", "company", "startup")):
            topic, reason = TopicType.COMPANY_OR_TEAM, "X post announces a company/team change"
        elif any(marker in tweet_route_text for marker in ("paper", "benchmark", "arxiv", "research result")) or (
            "vulnerab" in tweet_route_text and any(marker in tweet_route_text for marker in ("we found", "research", "verified", "experiment"))
        ):
            topic, reason = TopicType.RESEARCH_OR_BENCHMARK, "X post leads with research"
        elif any(marker in tweet_route_text for marker in ("paid users", "rate limit", "pricing", "reset has landed", "takes effect", "rollout")):
            topic, reason = TopicType.OFFICIAL_ANNOUNCEMENT, "X post communicates an operational product update"
        elif (
            any(marker in tweet_route_text for marker in (
                "model", "weights", "checkpoint", "parameters", "parameter model",
                "fp8", "bf16", "gguf", "mlx", "llm",
            ))
            and any(marker in tweet_route_text for marker in (
                "released", "release", "available", "launching", "open source", "hugging face",
            ))
        ):
            topic, reason = TopicType.MODEL_OR_PRODUCT, "X post announces a model/product"
        elif any(marker in tweet_route_text for marker in ("sdk", "api", "agent", "tool", "cli", "open source")):
            topic, reason = TopicType.TOOL_SDK_AGENT, "X post leads with a tool or developer interface"
        elif any(marker in tweet_route_text for marker in ("model", "product", "available today", "launching")):
            topic, reason = TopicType.MODEL_OR_PRODUCT, "X post announces a model/product"
        else:
            topic, reason = TopicType.PRACTICE_POST, "X post is a claim or practice viewpoint"
    elif re.search(
        r"\b(?:raised?|raises|raising|funding|series\s+[a-f]|seed\s+round|post-money|valued\s+at)\b|"
        r"\$\s?\d+(?:\.\d+)?\s?(?:m|million|b|billion)\b",
        text,
    ) or any(marker in text for marker in ("founding team", "acquisition")):
        topic, reason = TopicType.COMPANY_OR_TEAM, "company formation, funding, or acquisition language"
    elif sum(marker in text for marker in (
        "technical report", "benchmark", "benchmark contamination", "dataset", "methodology",
        "double-blind", "evaluation", "evaluations", "experiment", "experimental results",
    )) >= 2:
        topic, reason = TopicType.RESEARCH_OR_BENCHMARK, "research/evaluation method and evidence language"
    elif any(marker in url for marker in ("/docs", "developers.", "/sdk", "/api/")) or any(
        marker in text for marker in ("quick start", "getting started", "install", "sdk", "api reference")
    ):
        topic, reason = TopicType.TOOL_SDK_AGENT, "developer documentation source"
    elif any(marker in text for marker in ("technical report", "benchmark", "dataset", "methodology")):
        topic, reason = TopicType.RESEARCH_OR_BENCHMARK, "research/benchmark language"
    elif any(marker in text for marker in ("retired", "deprecat", "migration", "effective ", "生效", "迁移")) and any(
        marker in url for marker in ("/news/", "/changelog", "/announcement")
    ):
        topic, reason = TopicType.OFFICIAL_ANNOUNCEMENT, "official change with migration/effective scope"
    elif any(marker in text for marker in ("introducing ", "new model", "model card", "available today", "model is available")):
        topic, reason = TopicType.MODEL_OR_PRODUCT, "model/product release language"
    elif any(marker in url for marker in ("/news/", "/changelog", "/announcement")):
        topic, reason = TopicType.OFFICIAL_ANNOUNCEMENT, "official news or changelog URL"
    else:
        topic, reason = TopicType.OFFICIAL_ANNOUNCEMENT, "generic primary web announcement"

    if format_override:
        content_type = format_override
    elif openrouter_discount_page or topic in {
        TopicType.PRACTICE_POST, TopicType.COMPANY_OR_TEAM, TopicType.OFFICIAL_ANNOUNCEMENT,
    }:
        content_type = ContentType.FLASH
    elif topic == TopicType.RESEARCH_OR_BENCHMARK:
        content_type = ContentType.DEEP_DIVE
    else:
        content_type = ContentType.EXPLAINER

    defaults = {ContentType.FLASH: 12.0, ContentType.EXPLAINER: 28.0, ContentType.DEEP_DIVE: 50.0}
    duration = float(duration_override or defaults[content_type])
    if content_type == ContentType.FLASH and duration > 15:
        duration = 15.0
    if duration <= 0:
        raise ValueError("duration must be positive")
    return RouteDecision(topic, content_type, duration, reason)


def _validate_radar_contract(brief: EditorialBrief, evidence: list[Evidence]) -> list[str]:
    """Validate only opted-in Radar copy; archived briefs remain compatible."""
    if not brief.opening_mode.strip():
        return []
    errors: list[str] = []
    mode = brief.opening_mode.strip()
    if mode not in RADAR_OPENING_MODES:
        errors.append("opening_mode must be direct_fact, conflict, counter_intuitive, or developer_roi")
    label = brief.category_label.strip()
    if label and label not in RADAR_CATEGORY_LABELS:
        errors.append("category_label must be an optional neutral factual Radar label")

    strategy = brief.attention_strategy
    headline = (strategy.selected_hook or brief.headline).strip()
    width = copy_width(headline)
    exact_subject = any(
        subject.name.strip() and subject.name.casefold() in headline.casefold()
        for subject in brief.subjects
    )
    if width > 28 or (width > 20 and not exact_subject):
        errors.append("Radar headline must fit 20 equivalents, or 28 only to preserve an exact subject name")
    if _looks_like_radar_fragment(headline):
        errors.append("Radar selected hook must be a complete phrase, not a mechanically clipped fragment")
    visible_fields = [
        brief.headline, brief.subheadline, brief.fixed_conclusion,
        strategy.selected_hook, *strategy.hook_candidates,
        *(shot.fact for shot in brief.evidence_shots),
        *(shot.audience_copy for shot in brief.evidence_shots),
        *(shot.translation for shot in brief.evidence_shots),
        *(shot.full_translation for shot in brief.evidence_shots),
    ]
    for phrase in RADAR_BANNED_COPY:
        if any(phrase in value for value in visible_fields):
            errors.append(f"Radar visible copy contains banned vague phrase: {phrase}")

    if copy_width(brief.fixed_conclusion) > 40:
        errors.append("Radar fixed_conclusion must fit the 40-equivalent screen budget")
    for index, shot in enumerate(brief.evidence_shots):
        compact_copy = "".join(filter(None, (shot.fact.strip(), shot.audience_copy.strip())))
        if copy_width(compact_copy) > 40:
            errors.append(f"Radar shot {shot.id} fact plus audience_copy exceeds 40 equivalents")
        gloss = (shot.full_translation or shot.translation).strip()
        gloss_limit = 120 if index == 0 and shot.kind == EvidenceShotKind.TWEET_CARD else 40
        if gloss and copy_width(gloss) > gloss_limit:
            errors.append(
                f"Radar shot {shot.id} Chinese gloss must fit at most {gloss_limit} equivalents"
            )
    beats = [shot.narrative_beat.strip() for shot in brief.evidence_shots]
    if beats:
        if beats[0] != "opening" or beats[-1] != "takeaway":
            errors.append("Radar evidence shots must start with opening and end with takeaway")
        if any(beat not in {"opening", "proof", "takeaway"} for beat in beats):
            errors.append("Radar narrative_beat must be opening, proof, or takeaway")
        if len(beats) >= 3 and not any(beat == "proof" for beat in beats[1:-1]):
            errors.append("Radar story needs at least one proof beat between opening and takeaway")

    inference = brief.editorial_inference.strip()
    if inference and (
        inference != strategy.selected_hook.strip() or not inference.endswith(("?", "？"))
    ):
        errors.append("editorial_inference must exactly match a question-form selected_hook")
    identifier = brief.direct_identifier.strip()
    if identifier:
        corpus = "\n".join([*(item.url for item in evidence), *(item.quote for item in evidence)])
        needle = re.sub(r"^(?:HF:\s*|pip install\s+|docker pull\s+)", "", identifier, flags=re.IGNORECASE)
        if not needle or needle.casefold() not in corpus.casefold():
            errors.append("direct_identifier must be copied from archived evidence")
    return errors


def validate_editorial_brief(
    brief: EditorialBrief, candidate: Candidate, evidence: list[Evidence],
    topic: TopicType, content_type: ContentType,
) -> list[str]:
    errors: list[str] = []
    errors.extend(_validate_radar_contract(brief, evidence))
    evidence_ids = {item.id for item in evidence}
    strategy = brief.attention_strategy
    required_attention = {
        "hook_fact": strategy.hook_fact, "conflict": strategy.conflict,
        "stakes": strategy.stakes, "stance": strategy.stance, "payoff": strategy.payoff,
    }
    for name, value in required_attention.items():
        if not value.strip():
            errors.append(f"attention_strategy.{name} is required")
    unknown_hook_evidence = set(strategy.hook_evidence_ids) - evidence_ids
    if unknown_hook_evidence:
        errors.append("hook references unknown evidence ids: " + ", ".join(sorted(unknown_hook_evidence)))
    if not strategy.hook_evidence_ids:
        errors.append("hook must cite evidence")
    evidence_by_id = {item.id: item.quote for item in evidence}
    evidence_items_by_id = {item.id: item for item in evidence}
    hook_source = "\n".join(evidence_by_id.get(item, "") for item in strategy.hook_evidence_ids)
    hook_claims = "\n".join([
        strategy.hook_fact, strategy.conflict, strategy.stakes, strategy.stance,
        strategy.selected_hook, brief.headline, brief.subheadline,
    ])
    root_source_text = next((
        item.quote for item in evidence
        if item.url.rstrip("/") == candidate.source_url.rstrip("/")
    ), "")
    editorial_center = "\n".join([
        brief.headline, brief.subheadline, brief.fixed_conclusion,
        strategy.hook_fact, strategy.conflict, strategy.stakes,
        strategy.stance, strategy.payoff, strategy.selected_hook,
    ])
    if (
        len(root_source_text) > 1200
        and len(re.findall(r"(?m)^\s*\d+[.)]", root_source_text)) >= 3
        and re.search(r"great release overall|overall.{0,20}great|which is great", root_source_text, re.IGNORECASE)
        and len(re.findall(r"成本|代价|权衡|\boverhead\b|\bcost\b", editorial_center, re.IGNORECASE)) >= 4
    ):
        errors.append(
            "editorial salience is distorted: a secondary component cost dominates an overall-positive technical roundup"
        )
    all_source_text = "\n".join(item.quote for item in evidence)
    visible_editorial = "\n".join([
        editorial_center,
        *(shot.question + " " + shot.fact + " " + shot.audience_copy + " " + shot.interpretation for shot in brief.evidence_shots),
        *(
            beat.claim + " " + beat.why_here
            for beat in (brief.director_brief.story_arc if brief.director_brief else [])
        ),
    ])
    if re.search(r"许可证|商用许可|\blicen[cs]e\b", visible_editorial, re.IGNORECASE) and not re.search(
        r"许可证|商用许可|\blicen[cs]e\b", all_source_text, re.IGNORECASE,
    ):
        errors.append("license is an invented unknown category; supplied evidence never raises licensing")
    lower_cost_claim = re.search(
        r"降低.{0,8}成本|成本.{0,8}降低|更低.{0,6}成本|更便宜.{0,6}推理|"
        r"砍(?:掉|低)?.{0,6}推理成本|lower.{0,12}cost|reduce.{0,12}cost|cheaper.{0,12}inference",
        visible_editorial, re.IGNORECASE,
    )
    lower_cost_proof = re.search(
        r"降低.{0,8}成本|成本.{0,8}降低|更低.{0,6}成本|更便宜.{0,6}推理|"
        r"lower.{0,12}cost|reduce.{0,12}cost|cheaper.{0,12}inference",
        all_source_text, re.IGNORECASE,
    )
    if lower_cost_claim and not lower_cost_proof:
        errors.append("better inference efficiency cannot be rewritten as verified lower deployment cost")
    if re.search(r"自部署|self[- ]host|self[- ]deploy", visible_editorial, re.IGNORECASE) and not re.search(
        r"自部署|self[- ]host|self[- ]deploy|deployment", all_source_text, re.IGNORECASE,
    ):
        errors.append("self-deployment is not supported by the supplied evidence")
    if re.search(r"(?:离开|出走|失去|离职).{0,8}(?:Google|谷歌)|(?:Google|谷歌).{0,8}(?:离开|出走|失去|离职)", hook_claims, re.IGNORECASE) and not re.search(
        r"(?:leave|leaving|left|departure|departing|last day).{0,24}Google|Google.{0,24}(?:leave|leaving|left|departure|last day)|(?:离开|出走|失去|离职).{0,8}(?:Google|谷歌)",
        hook_source, re.IGNORECASE,
    ):
        errors.append("Google departure/exit hook is not supported by hook_evidence_ids")
    if re.search(r"持续|接连|集体流失|一波离职|人才外流", hook_claims) and len(brief.context_events) < 2:
        errors.append("trend language needs at least two dated context_events")
    if re.search(r"股价|市值|蒸发|上涨|下跌", hook_claims) and not re.search(
        r"stock|share price|market cap|市值|股价|上涨|下跌", hook_source, re.IGNORECASE,
    ):
        errors.append("market-move hook needs direct market evidence")
    if re.search(r"计费.{0,4}漏洞|billing.{0,12}vulnerab", hook_claims, re.IGNORECASE) and not re.search(
        r"billing.{0,16}(?:vulnerab|exploit)|计费.{0,4}漏洞", hook_source, re.IGNORECASE,
    ):
        errors.append("billing-token correlation cannot be rewritten as the vulnerability mechanism")
    if re.search(r"发布|上线|推出|宣布", hook_claims) and not re.search(
        r"\breleas(?:e|ed|es|ing)\b|\blaunch(?:ed|es|ing)?\b|\bannounc(?:e|ed|es|ing|ement)\b|"
        r"\bavailable today\b|\bintroducing\b|\bland(?:ed|s|ing)?\b|发布|上线|推出|宣布|已落地|已到账",
        hook_source, re.IGNORECASE,
    ):
        errors.append(
            "release/launch wording needs explicit release evidence; documentation alone proves availability and capability"
        )
    if len(strategy.hook_candidates) != 3:
        errors.append("attention_strategy must contain exactly three hook_candidates")
    selected = strategy.selected_hook.strip()
    if selected and selected not in strategy.hook_candidates:
        errors.append("selected_hook must be one of hook_candidates")
    visible_hook = selected or (strategy.hook_candidates[0] if strategy.hook_candidates else "")
    if not visible_hook:
        errors.append("selected hook is required")
    if any(label.casefold() in visible_hook.casefold() for label in INTERNAL_LABELS):
        errors.append("hook exposes an internal production label")
    if any(label in visible_hook for label in EMPTY_HYPE) and not re.search(r"\d|[A-Za-z]{2,}", visible_hook):
        errors.append("hook uses empty hype without a concrete subject or number")
    if strategy.payoff.strip() == visible_hook:
        errors.append("hook and payoff must not repeat the same sentence")
    if not brief.headline.strip() or not brief.subheadline.strip() or not brief.fixed_conclusion.strip():
        errors.append("headline, subheadline, and fixed_conclusion are required")
    if content_type == ContentType.FLASH and copy_width(brief.fixed_conclusion) > 64:
        errors.append("fixed conclusion must fit the persistent bottom rail in at most 64 Chinese-character-equivalents")
    if not brief.subjects:
        errors.append("at least one named story subject is required")
    for subject in brief.subjects:
        if not subject.name.strip() or not subject.action.strip() or not subject.consequence.strip():
            errors.append("every subject needs name, action, and consequence")
        if not set(subject.evidence_ids) <= evidence_ids:
            errors.append(f"subject {subject.name or '<missing>'} references unknown evidence")

    # Story-axis lock: the hook may become sharper, but it cannot replace a
    # people-move story with a secondary product capability or metric. The
    # first selection reason is the publication promise chosen before writing.
    if (
        brief.opportunity
        and brief.opportunity.story_archetype == "people_change"
        and brief.opportunity.selection_reasons
    ):
        primary_reason = brief.opportunity.selection_reasons[0]
        movement = re.compile(
            r"离开|离职|出走|失去|加入|入职|创立|创办|创业|联合创始|另起炉灶|挖角|"
            r"押在|集体|\b(?:leave|left|depart|join|found|co-founder|poach)",
            re.IGNORECASE,
        )
        people_frame = re.compile(
            r"四位|三位|两位|团队|创始人|老将|老搭档|人才|核心人物|研究者|员工",
            re.IGNORECASE,
        )
        primary_names = [
            subject.name for subject in brief.subjects
            if set(subject.evidence_ids) & set(primary_reason.evidence_ids)
        ]

        def keeps_people_axis(copy: str) -> bool:
            named = any(name and name.casefold() in copy.casefold() for name in primary_names)
            return bool(movement.search(copy) and (named or people_frame.search(copy)))

        if not keeps_people_axis(visible_hook):
            errors.append(
                "people-change hook abandoned the primary person/team move for a secondary story"
            )
        if not keeps_people_axis(brief.fixed_conclusion):
            errors.append(
                "people-change conclusion must resolve the primary person/team move, not a secondary capability"
            )
        if (
            brief.evidence_shots
            and brief.evidence_shots[-1].selection_reason_ids
            and primary_reason.id not in brief.evidence_shots[-1].selection_reason_ids
        ):
            errors.append(
                "people-change final payoff must return to the primary selection reason"
            )

    # A multi-party reply chain needs its bridge on screen. Naming a responder
    # in the root headline is not enough when their actual reply explains why
    # the final reaction happened.
    reply_subjects = [
        subject for subject in brief.subjects
        if re.search(r"回复|回应|质疑|评论|表示|澄清|征集|reply|respond|question", subject.action, re.IGNORECASE)
    ]
    is_multi_party_chain = (
        candidate.source_type == SourceType.TWEET
        and len(brief.subjects) >= 3
        and (
            len(reply_subjects) >= 2
            or bool(brief.opportunity and brief.opportunity.story_archetype == "event_chain")
        )
    )
    if is_multi_party_chain:
        if len(brief.evidence_shots) < 4:
            errors.append(
                "multi-party reply chain needs at least four shots so the intervening response remains visible"
            )
        later_copy = "\n".join(
            " ".join((shot.fact, shot.audience_copy, shot.translation, shot.full_translation))
            for shot in brief.evidence_shots[1:]
        )
        action_markers = (
            "不解", "质疑", "征集", "澄清", "否认", "不在", "回复", "回应", "表示",
            "招人", "招聘", "停用", "封号", "封禁", "申诉",
        )
        for subject in brief.subjects[1:]:
            display_name = re.split(r"\s*\(|\s*@", subject.name, maxsplit=1)[0].strip()
            subject_shots = [
                " ".join((shot.fact, shot.audience_copy, shot.translation, shot.full_translation))
                for shot in brief.evidence_shots[1:]
                if display_name and display_name.casefold() in " ".join((
                    shot.fact, shot.audience_copy, shot.translation, shot.full_translation,
                )).casefold()
            ]
            if not subject_shots:
                errors.append(
                    f"multi-party reply chain omits {display_name or subject.name}'s visible action"
                )
                continue
            required_markers = [marker for marker in action_markers if marker in subject.action]
            if required_markers and not any(
                marker in "\n".join(subject_shots) for marker in required_markers
            ):
                errors.append(
                    f"multi-party reply chain names {display_name or subject.name} but omits their actual action"
                )
        if re.match(r"^(?:截图|画面).{0,8}(?:只|仅).{0,8}证明", brief.fixed_conclusion):
            errors.append(
                "causal-safety note replaced the story payoff; keep the event meaning first and qualify only the disputed link"
            )
    if content_type == ContentType.FLASH and brief.duration_target > 15:
        errors.append("flash duration must not exceed 15 seconds")
    if not brief.evidence_shots:
        errors.append("at least one evidence shot is required")
    if candidate.source_type == SourceType.TWEET and brief.evidence_shots and brief.evidence_shots[0].kind != EvidenceShotKind.TWEET_CARD:
        errors.append("an X-rooted story must begin with one complete tweet_card shot")
    seen_facts: set[str] = set()
    for index, shot in enumerate(brief.evidence_shots, start=1):
        if not shot.fact.strip() or not shot.interpretation.strip() or not shot.question.strip():
            errors.append(f"evidence_shots[{index}] needs question, fact, and interpretation")
        if not shot.evidence_ids or not set(shot.evidence_ids) <= evidence_ids:
            errors.append(f"evidence_shots[{index}] must cite valid evidence")
        if shot.duration <= 0 or shot.duration > 5:
            errors.append(f"evidence_shots[{index}].duration must be >0 and <=5")
        normalized = re.sub(r"\s+", "", shot.fact).casefold()
        if normalized in seen_facts:
            errors.append(f"evidence_shots[{index}] repeats a previous fact")
        seen_facts.add(normalized)
        if index > 1:
            previous = re.sub(r"[\s，。；：、！？,.!?;:]", "", brief.evidence_shots[index - 2].fact).casefold()
            current = re.sub(r"[\s，。；：、！？,.!?;:]", "", shot.fact).casefold()
            if min(len(previous), len(current)) >= 12 and SequenceMatcher(None, previous, current).ratio() >= 0.76:
                errors.append(f"evidence_shots[{index}] near-duplicates the previous on-screen fact")
        if shot.translation and re.search(r"(?:翻译|译为)\s*[:：]", shot.translation):
            errors.append(f"evidence_shots[{index}].translation must contain only natural Chinese")
        target_latin = len(re.findall(r"[A-Za-z]", shot.target))
        target_han = len(re.findall(r"[\u4e00-\u9fff]", shot.target))
        if (
            target_latin >= 12 and target_latin > target_han * 2
            and not (shot.translation.strip() or shot.full_translation.strip())
        ):
            errors.append(
                f"evidence_shots[{index}] has an English browser target but no adjacent Chinese translation"
            )
        if any(label.casefold() in (shot.fact + shot.interpretation).casefold() for label in INTERNAL_LABELS):
            errors.append(f"evidence_shots[{index}] exposes an internal production label")
        if looks_like_internal_direction(shot.audience_copy):
            errors.append(
                f"evidence_shots[{index}].audience_copy contains internal reading/director instructions; "
                "replace it with a declarative audience fact/impact or leave it empty"
            )
        normalized_audience = re.sub(r"[\s，。；：、！？,.!?;:]", "", shot.audience_copy).casefold()
        normalized_internal = re.sub(r"[\s，。；：、！？,.!?;:]", "", shot.interpretation).casefold()
        if normalized_audience and normalized_audience == normalized_internal:
            errors.append(
                f"evidence_shots[{index}].audience_copy duplicates internal interpretation; "
                "write audience information separately or leave it empty"
            )
        if shot.visual_family in {"quote_card", "timeline", "impact_card", "stat_card"}:
            cited_text_sources = [
                evidence_items_by_id[item]
                for item in shot.evidence_ids
                if item in evidence_items_by_id
                and evidence_items_by_id[item].source_kind not in {"web:source_image", "x:media_photo"}
            ]
            if cited_text_sources and not shot.target.strip():
                errors.append(
                    f"evidence_shots[{index}] derived evidence card needs a self-contained exact source excerpt"
                )
            elif shot.target.strip() and re.search(r"[A-Za-z]", shot.target):
                word_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9+._%'-]*", shot.target))
                if word_count < 4:
                    errors.append(
                        f"evidence_shots[{index}].target is a fragment; use a self-contained exact source excerpt"
                    )
    total = sum(shot.duration for shot in brief.evidence_shots)
    if total > brief.duration_target + 0.01:
        errors.append(f"evidence shots need {total:.1f}s but duration_target is {brief.duration_target:.1f}s")
    if len(brief.evidence_shots) > 1 and not any(shot.relation_to_previous.strip() for shot in brief.evidence_shots[1:]):
        errors.append("story has no explicit information progression between shots")
    if content_type == ContentType.FLASH and brief.evidence_shots:
        last_beats = set(brief.evidence_shots[-1].beat_ids)
        expected_payoffs = {
            TopicType.COMPANY_OR_TEAM: {"impact"},
            TopicType.MODEL_OR_PRODUCT: {"practical_change", "adoption_choice"},
            TopicType.OFFICIAL_ANNOUNCEMENT: {"impact", "migration", "action", "effective_scope", "availability"},
            TopicType.TOOL_SDK_AGENT: {"labor_saved", "workflow_fit", "trial_task"},
        }
        required_payoff = expected_payoffs.get(topic)
        if required_payoff and not last_beats & required_payoff:
            errors.append("flash final shot must deliver impact/payoff/action instead of an unknown-information card")
    if topic == TopicType.PRACTICE_POST and not any("scope" in shot.beat_ids for shot in brief.evidence_shots):
        errors.append("practice story must include the evidence-backed scope beat")
    dull = re.search(
        r"标志着|重心转移.{0,10}(?:信号|趋势)|值得.{0,6}关注|密切关注|关注后续|后续说明",
        brief.fixed_conclusion,
    )
    ritual_fixed = re.search(
        r"(?:但|不过|然而).{0,40}(?:未知|未公开|有待|需关注|待披露|需验证|待公布|尚未.{0,8}公开)|"
        r"(?:未知|尚未.{0,8}公开|影响范围.{0,10}(?:未知|未公开)|有待验证|需验证|待公布|"
        r"值得关注.{0,12}(?:后续|影响)|需(?:持续)?关注|仍待.{0,8}(?:公开|验证|研究)|进一步研究)[。！]?$",
        brief.fixed_conclusion,
    )
    if dull or ritual_fixed:
        errors.append("fixed conclusion is bureaucratic; name the concrete winner, loser, gain, cost, or capability shift")
    if brief.opportunity:
        reason_ids = {item.id for item in brief.opportunity.selection_reasons}
        if not brief.opportunity.why_now.strip() or not brief.opportunity.why_audience.strip():
            errors.append("editorial opportunity must explain why now and why this audience")
        for reason in brief.opportunity.selection_reasons:
            if not reason.rationale.strip() or not set(reason.evidence_ids) <= evidence_ids:
                errors.append(f"selection reason {reason.id or '<missing>'} must cite valid evidence")
        director = brief.director_brief
        if not director:
            errors.append("contextual stories require director_brief")
        else:
            if not director.editorial_thesis.strip() or not director.viewer_tension.strip() or not director.attention_trigger.strip():
                errors.append("director_brief needs thesis, viewer_tension, and attention_trigger")
            if director.emotion_intensity not in {1, 2, 3}:
                errors.append("director emotion_intensity must be 1, 2, or 3")
            arc_evidence = {item for beat in director.story_arc for item in beat.evidence_ids}
            if not director.story_arc or not arc_evidence or not arc_evidence <= evidence_ids:
                errors.append("director story_arc must cite valid evidence")
            arc_reasons = {item for beat in director.story_arc for item in beat.selection_reason_ids}
            if reason_ids and not arc_reasons & reason_ids:
                errors.append("story arc must preserve at least one reason this item was selected")
        graph_ids = {item.id for item in (brief.context_graph.events if brief.context_graph else []) if item.id}
        if director and not set(director.selected_context_ids) <= graph_ids:
            errors.append("director_brief references unknown context events")
        if director and brief.context_graph:
            selected_context = set(director.selected_context_ids)
            missing_required = set(brief.context_graph.required_context_ids) - selected_context
            if missing_required:
                errors.append(
                    "director must use every required setup/context event: " + ", ".join(sorted(missing_required))
                )
            visibly_used_context = {
                item for shot in brief.evidence_shots for item in shot.context_event_ids
            }
            causal_setup_ids = {
                event.id for event in brief.context_graph.events
                if event.relation.startswith("exact earlier post")
            }
            invisible_required = causal_setup_ids - visibly_used_context
            if invisible_required:
                errors.append(
                    "every exact causal setup post must appear in a visible evidence shot: "
                    + ", ".join(sorted(invisible_required))
                )
            leaked_discarded = (
                selected_context | visibly_used_context
            ) & set(brief.context_graph.discarded_context_ids)
            if leaked_discarded:
                errors.append(
                    "discarded research context leaked into the visible story: "
                    + ", ".join(sorted(leaked_discarded))
                )
            if brief.opportunity.story_archetype == "people_change" and brief.context_graph.pattern_context_ids:
                pattern_ids = set(brief.context_graph.pattern_context_ids)
                visibly_used = {
                    item for shot in brief.evidence_shots for item in shot.context_event_ids
                }
                if not selected_context & pattern_ids or not visibly_used & pattern_ids:
                    errors.append("people-change story must visibly use the verified incumbent-history pattern context")
        for shot in brief.evidence_shots:
            if shot.selection_reason_ids and not set(shot.selection_reason_ids) <= reason_ids:
                errors.append(f"evidence shot {shot.id} references unknown selection reason")
            if shot.context_event_ids and not set(shot.context_event_ids) <= graph_ids:
                errors.append(f"evidence shot {shot.id} references unknown context event")
        if brief.evidence_shots:
            ending_copy = " ".join((
                brief.evidence_shots[-1].fact,
                brief.evidence_shots[-1].interpretation,
            ))
            if re.search(
                r"未公开|未说明|尚未|未知|等待|待验证|需验证|待公布|需谨慎|关注后续|值得关注",
                ending_copy,
            ):
                errors.append("final changing shot must end on verified capability/impact, not an unknown or ritual caution")
        if content_type == ContentType.FLASH:
            if len(brief.evidence_shots) < 3:
                errors.append("high-retention flash needs at least three semantic visual changes")
            families = {shot.visual_family for shot in brief.evidence_shots if shot.visual_family}
            if len(families) < 3:
                errors.append("high-retention flash needs at least three visual families")
            supported_families = {
                "tweet", "quoted_post", "official_page", "source_image", "product_ui", "chart", "timeline",
                "code", "paper", "quote_card", "impact_card", "stat_card",
            }
            unknown_families = families - supported_families
            if unknown_families:
                errors.append("visual families are not renderable: " + ", ".join(sorted(unknown_families)))
            for index, shot in enumerate(brief.evidence_shots, start=1):
                if not shot.visual_family:
                    errors.append(f"evidence_shots[{index}] needs a renderable visual_family")
                if shot.visual_family in {"tweet", "quoted_post"} and shot.kind != EvidenceShotKind.TWEET_CARD:
                    errors.append(f"evidence_shots[{index}] uses an X-card family without tweet_card evidence")
                if shot.kind == EvidenceShotKind.TWEET_CARD and shot.visual_family not in {
                    "tweet", "quoted_post", "quote_card", "timeline", "impact_card", "stat_card",
                }:
                    errors.append(f"evidence_shots[{index}] assigns a browser family to tweet evidence")
                if shot.visual_family == "source_image" and shot.kind != EvidenceShotKind.IMAGE:
                    errors.append(f"evidence_shots[{index}] uses source_image without image evidence")
            for previous, current in zip(brief.evidence_shots, brief.evidence_shots[1:]):
                if previous.visual_family == current.visual_family and set(previous.evidence_ids) == set(current.evidence_ids):
                    errors.append("consecutive flash shots cannot repeat the same evidence treatment")
            if any(shot.duration > (3.2 if shot.kind == EvidenceShotKind.TWEET_CARD else 2.8) for shot in brief.evidence_shots):
                errors.append("flash shots must change every 1.3–2.8 seconds; a complete tweet may use at most 3.2 seconds")
            if any(shot.duration < 1.3 for shot in brief.evidence_shots):
                errors.append("flash shots shorter than 1.3 seconds are unreadable without narration")
            first = brief.evidence_shots[0]
            last = brief.evidence_shots[-1]
            if len(brief.evidence_shots) > 2 and set(first.evidence_ids) == set(last.evidence_ids):
                first_copy = re.sub(r"[\s，。；：、！？,.!?;:]", "", first.fact).casefold()
                last_copy = re.sub(r"[\s，。；：、！？,.!?;:]", "", last.fact).casefold()
                if SequenceMatcher(None, first_copy, last_copy).ratio() >= 0.55:
                    errors.append(
                        "final changing shot repeats the opening event; the fixed bottom rail already carries the payoff"
                    )
            visible_flash = "\n".join([
                brief.headline, brief.subheadline, brief.fixed_conclusion,
                strategy.hook_fact, strategy.conflict, strategy.surprise, strategy.stakes,
                strategy.stance, strategy.payoff,
                *(shot.fact + " " + shot.interpretation for shot in brief.evidence_shots),
            ])
            ending_copy = "\n".join([
                brief.fixed_conclusion, strategy.payoff,
                brief.evidence_shots[-1].fact, brief.evidence_shots[-1].interpretation,
                director.story_arc[-1].claim if director and director.story_arc else "",
            ])
            if re.search(r"具体.{0,8}(?:未知|未说明|未公开)|(?:未知|未说明|未公开).{0,8}(?:机制|规则|细节)|等待官方|关注后续", ending_copy):
                errors.append("flash story must spend its last seconds on verified payoff, not missing mechanism/details")
            director_copy = ""
            if director:
                director_copy = "\n".join([
                    director.editorial_thesis, director.viewer_tension, director.attention_trigger,
                    *(beat.claim + " " + beat.why_here for beat in director.story_arc),
                ])
            grounded_copy = visible_flash + "\n" + director_copy
            source_text = "\n".join(item.quote for item in evidence)
            unsupported_concepts = []
            for name, visible_pattern, source_pattern in (
                ("API", r"\bAPI\b", r"\bAPI\b"),
                ("cost/budget", r"成本|预算|(?<!付)费用|\bcost\b|\bbudget\b", r"成本|预算|(?<!付)费用|\bprice|\bpricing\b|\bcost\b|\bbudget\b"),
                ("quota/usage limit", r"配额|额度|用量限制|usage limit|quota", r"配额|额度|用量限制|usage limit|quota"),
                ("accumulation", r"累积|结转|roll.?over|accumulat", r"累积|结转|roll.?over|accumulat"),
                ("flexibility", r"更灵活|灵活的", r"更灵活|灵活的|flexib|at (?:your|their) own leisure"),
                ("official account", r"官方账号|官方帐号", r"官方账号|官方帐号|official account"),
            ):
                if re.search(visible_pattern, grounded_copy, re.IGNORECASE) and not re.search(source_pattern, source_text, re.IGNORECASE):
                    unsupported_concepts.append(name)
            if unsupported_concepts:
                errors.append("visible copy introduces unsupported concepts: " + ", ".join(unsupported_concepts))
            if re.search(r"banked reset", source_text, re.IGNORECASE) and re.search(r"银行.{0,3}重置", grounded_copy):
                errors.append("branded feature name banked reset must remain untranslated without an official Chinese name")
            if re.search(r"无.{0,6}限制|没有.{0,6}限制|不受.{0,6}限制", grounded_copy) and not re.search(
                r"无.{0,6}限制|没有.{0,6}限制|不受.{0,6}限制|unlimited|no .{0,10}limit", source_text, re.IGNORECASE,
            ):
                errors.append("absence-of-limit claim needs explicit source evidence")
        high_value_team_images = {
            item.id for item in evidence
            if item.source_kind in {"web:source_image", "x:media_photo"}
            and item.metadata.get("editorial_priority") == "high"
            and item.metadata.get("visual_role") == "team"
        }
        if topic == TopicType.COMPANY_OR_TEAM and high_value_team_images and not any(
            set(shot.evidence_ids) & high_value_team_images for shot in brief.evidence_shots
        ):
            errors.append("company/team story omitted an official high-value team image")
        if candidate.source_type == SourceType.TWEET and brief.evidence_shots:
            root_text = next((item.quote for item in evidence if item.url.rstrip("/") == candidate.source_url.rstrip("/")), evidence[0].quote if evidence else "")
            han = len(re.findall(r"[\u4e00-\u9fff]", root_text))
            latin = len(re.findall(r"[A-Za-z]", root_text))
            first = brief.evidence_shots[0]
            if latin > han * 2 and not (first.full_translation.strip() or first.translation.strip()):
                errors.append("a non-Chinese root post requires adjacent readable Chinese translation in the first shot")
            root_translation = first.full_translation.strip() or first.translation.strip()
            if len(root_translation) > 140:
                errors.append("root-post Chinese translation must fit beside the source in at most 140 characters")
            quoted_ids = {item.id for item in evidence if item.source_kind == "x:quoted_post"}
            quoted_shots = [shot for shot in brief.evidence_shots[1:] if set(shot.evidence_ids) & quoted_ids]
            if quoted_shots:
                chronology = "\n".join(
                    shot.relation_to_previous + " " + shot.interpretation for shot in quoted_shots
                )
                if not re.search(r"此前|先前|前一条|先预告|承诺|随后|随后落地|earlier|before|then", chronology, re.IGNORECASE):
                    errors.append("a quoted earlier post must be framed as chronology/background, not a detached detail card")
    return errors


def _validate_story_axis_structure(brief: EditorialBrief, candidate: Candidate) -> list[str]:
    """Keep retention optimization inside the selected story promise."""
    errors: list[str] = []
    strategy = brief.attention_strategy
    visible_hook = strategy.selected_hook.strip()
    if (
        brief.opportunity
        and brief.opportunity.story_archetype == "people_change"
        and brief.opportunity.selection_reasons
    ):
        primary_reason = brief.opportunity.selection_reasons[0]
        movement = re.compile(
            r"离开|离职|出走|失去|加入|入职|创立|创办|创业|联合创始|另起炉灶|挖角|"
            r"押在|集体|\b(?:leave|left|depart|join|found|co-founder|poach)",
            re.IGNORECASE,
        )
        people_frame = re.compile(
            r"四位|三位|两位|团队|创始人|老将|老搭档|人才|核心人物|研究者|员工",
            re.IGNORECASE,
        )
        primary_names = [
            subject.name for subject in brief.subjects
            if set(subject.evidence_ids) & set(primary_reason.evidence_ids)
        ]

        def keeps_people_axis(copy: str) -> bool:
            named = any(name and name.casefold() in copy.casefold() for name in primary_names)
            return bool(movement.search(copy) and (named or people_frame.search(copy)))

        if not keeps_people_axis(visible_hook):
            errors.append(
                "people-change hook abandoned the primary person/team move for a secondary story"
            )
        if not keeps_people_axis(brief.fixed_conclusion):
            errors.append(
                "people-change conclusion must resolve the primary person/team move, not a secondary capability"
            )
        if (
            brief.evidence_shots
            and brief.evidence_shots[-1].selection_reason_ids
            and primary_reason.id not in brief.evidence_shots[-1].selection_reason_ids
        ):
            errors.append("people-change final payoff must return to the primary selection reason")

    reply_subjects = [
        subject for subject in brief.subjects
        if re.search(
            r"回复|回应|质疑|评论|表示|澄清|征集|reply|respond|question",
            subject.action, re.IGNORECASE,
        )
    ]
    is_multi_party_chain = (
        candidate.source_type == SourceType.TWEET
        and len(brief.subjects) >= 3
        and (
            len(reply_subjects) >= 2
            or bool(brief.opportunity and brief.opportunity.story_archetype == "event_chain")
        )
    )
    if not is_multi_party_chain:
        return errors
    if len(brief.evidence_shots) < 4:
        errors.append(
            "multi-party reply chain needs at least four shots so the intervening response remains visible"
        )
    action_markers = (
        "不解", "奇怪", "质疑", "征集", "澄清", "否认", "不在", "回复", "回应", "表示",
        "招人", "招聘", "停用", "封号", "封禁", "申诉",
    )
    for subject in brief.subjects[1:]:
        display_name = re.split(r"\s*[（(]|\s*@", subject.name, maxsplit=1)[0].strip()
        subject_shot_objects = [
            shot
            for shot in brief.evidence_shots[1:]
            if display_name and display_name.casefold() in " ".join((
                shot.fact, shot.audience_copy, shot.translation, shot.full_translation,
            )).casefold()
        ]
        if not subject_shot_objects:
            errors.append(f"multi-party reply chain omits {display_name or subject.name}'s visible action")
            continue
        if any(shot.id.startswith("chain-") for shot in subject_shot_objects):
            continue
        subject_shots = [
            " ".join((shot.fact, shot.audience_copy, shot.translation, shot.full_translation))
            for shot in subject_shot_objects
        ]
        required_markers = [marker for marker in action_markers if marker in subject.action]
        if required_markers and not any(
            marker in "\n".join(subject_shots) for marker in required_markers
        ):
            errors.append(
                f"multi-party reply chain names {display_name or subject.name} but omits their actual action"
            )
    if re.match(r"^(?:截图|画面).{0,8}(?:只|仅).{0,8}证明", brief.fixed_conclusion):
        errors.append(
            "causal-safety note replaced the story payoff; keep the event meaning first and qualify only the disputed link"
        )
    return errors


def validate_editorial_structure(
    brief: EditorialBrief, candidate: Candidate, evidence: list[Evidence],
    topic: TopicType, content_type: ContentType,
) -> list[str]:
    """Validate only machine-enforceable structure and evidence linkage.

    Meaning, factual entailment, entity relations, causal certainty, and
    Chinese naturalness require the bounded semantic copy critic. Keeping
    them out of this function prevents example-specific rules from becoming
    permanent execution logic.
    """
    errors: list[str] = []
    errors.extend(_validate_radar_contract(brief, evidence))
    errors.extend(_validate_story_axis_structure(brief, candidate))
    evidence_ids = {item.id for item in evidence}
    strategy = brief.attention_strategy
    for name, value in {
        "hook_fact": strategy.hook_fact,
        "stakes": strategy.stakes, "stance": strategy.stance, "payoff": strategy.payoff,
    }.items():
        if not value.strip():
            errors.append(f"attention_strategy.{name} is required")
    if brief.opening_mode in {"conflict", "counter_intuitive"} and not strategy.conflict.strip():
        errors.append("attention_strategy.conflict is required for the selected opening_mode")
    if len(strategy.hook_candidates) != 3:
        errors.append("attention_strategy must contain exactly three hook_candidates")
    if not strategy.selected_hook or strategy.selected_hook not in strategy.hook_candidates:
        errors.append("selected_hook must be one of hook_candidates")
    if not strategy.hook_evidence_ids or not set(strategy.hook_evidence_ids) <= evidence_ids:
        errors.append("hook must cite valid evidence")
    if not all((brief.headline.strip(), brief.subheadline.strip(), brief.fixed_conclusion.strip())):
        errors.append("headline, subheadline, and fixed_conclusion are required")
    if topic == TopicType.MODEL_OR_PRODUCT:
        model_names = [
            subject.name.strip() for subject in brief.subjects
            if subject.subject_type.strip().casefold() == "model" and subject.name.strip()
        ]
        model_aliases = [alias for name in model_names for alias in _model_subject_aliases(name)]
        if not _is_multi_model_price_evidence(evidence) and model_names and not any(
            name.casefold() in strategy.selected_hook.casefold() for name in model_aliases
        ):
            errors.append(
                "selected_hook must name the concrete model subject: " + ", ".join(model_names)
            )
    if content_type == ContentType.FLASH and copy_width(brief.fixed_conclusion) > 64:
        errors.append("fixed conclusion must fit the persistent bottom rail in at most 64 Chinese-character-equivalents")
    if not brief.subjects:
        errors.append("at least one named story subject is required")
    for subject in brief.subjects:
        if not all((subject.name.strip(), subject.action.strip(), subject.consequence.strip())):
            errors.append("every subject needs name, action, and consequence")
        if not set(subject.evidence_ids) <= evidence_ids:
            errors.append(f"subject {subject.name or '<missing>'} references unknown evidence")
    if content_type == ContentType.FLASH and brief.duration_target > 15:
        errors.append("flash duration must not exceed 15 seconds")
    if not brief.evidence_shots:
        return [*errors, "at least one evidence shot is required"]
    if candidate.source_type == SourceType.TWEET and brief.evidence_shots[0].kind != EvidenceShotKind.TWEET_CARD:
        errors.append("an X-rooted story must begin with one complete tweet_card shot")
    missing_glossary = _missing_audience_glossary(brief)
    if missing_glossary:
        shot_index, term = missing_glossary
        errors.append(
            f"evidence_shots[{shot_index}].audience_copy must explain specialist metric {term} in plain Chinese"
        )

    seen_facts: set[str] = set()
    for index, shot in enumerate(brief.evidence_shots, start=1):
        if not all((shot.question.strip(), shot.fact.strip(), shot.interpretation.strip())):
            errors.append(f"evidence_shots[{index}] needs question, fact, and interpretation")
        if not shot.evidence_ids or not set(shot.evidence_ids) <= evidence_ids:
            errors.append(f"evidence_shots[{index}] must cite valid evidence")
        if shot.duration <= 0 or shot.duration > 5:
            errors.append(f"evidence_shots[{index}].duration must be >0 and <=5")
        normalized = re.sub(r"\s+", "", shot.fact).casefold()
        if normalized in seen_facts:
            errors.append(f"evidence_shots[{index}] repeats a previous fact")
        seen_facts.add(normalized)
        if any(label.casefold() in (shot.fact + shot.audience_copy).casefold() for label in INTERNAL_LABELS):
            errors.append(f"evidence_shots[{index}] exposes an internal production label")
        if shot.visual_family in {"tweet", "quoted_post"} and shot.kind != EvidenceShotKind.TWEET_CARD:
            errors.append(f"evidence_shots[{index}] uses an X-card family without tweet_card evidence")
        if shot.visual_family == "source_image" and shot.kind != EvidenceShotKind.IMAGE:
            errors.append(f"evidence_shots[{index}] source_image must compile to image material")
    if sum(shot.duration for shot in brief.evidence_shots) > brief.duration_target + 0.01:
        errors.append("evidence shot durations exceed duration_target")
    if len(brief.evidence_shots) > 1 and not any(
        shot.relation_to_previous.strip() for shot in brief.evidence_shots[1:]
    ):
        errors.append("story has no explicit information progression between shots")

    if brief.opportunity:
        reason_ids = {item.id for item in brief.opportunity.selection_reasons}
        graph_ids = {item.id for item in (brief.context_graph.events if brief.context_graph else []) if item.id}
        director = brief.director_brief
        if not director:
            errors.append("contextual stories require director_brief")
        else:
            if not director.story_arc:
                errors.append("director_brief requires story_arc")
            if not set(director.selected_context_ids) <= graph_ids:
                errors.append("director_brief references unknown context events")
            arc_evidence = {item for beat in director.story_arc for item in beat.evidence_ids}
            if not arc_evidence or not arc_evidence <= evidence_ids:
                errors.append("director story_arc must cite valid evidence")
            if reason_ids and not any(
                set(beat.selection_reason_ids) & reason_ids for beat in director.story_arc
            ):
                errors.append("story arc must preserve at least one selection reason")
        if brief.context_graph and director:
            selected = set(director.selected_context_ids)
            missing = set(brief.context_graph.required_context_ids) - selected
            if missing:
                errors.append("director must select every required context event")
            visible_context = {item for shot in brief.evidence_shots for item in shot.context_event_ids}
            exact_setup = {
                event.id for event in brief.context_graph.events
                if event.relation.startswith("exact earlier post")
            }
            if exact_setup - visible_context:
                errors.append("exact causal setup evidence must appear in a visible shot")
            if (selected | visible_context) & set(brief.context_graph.discarded_context_ids):
                errors.append("discarded research context leaked into the visible story")
            if (
                brief.opportunity.story_archetype == "people_change"
                and brief.context_graph.pattern_context_ids
            ):
                pattern_ids = set(brief.context_graph.pattern_context_ids)
                if not selected & pattern_ids or not visible_context & pattern_ids:
                    errors.append(
                        "people-change story must visibly use the verified incumbent-history pattern context"
                    )

    if content_type == ContentType.FLASH and brief.opportunity:
        if len(brief.evidence_shots) < 3:
            errors.append("flash needs at least three semantic visual changes")
        families = {shot.visual_family for shot in brief.evidence_shots if shot.visual_family}
        if len(families) < 3:
            errors.append("flash needs at least three visual families")
    if candidate.source_type == SourceType.TWEET:
        root = next((
            item.quote for item in evidence
            if item.url.rstrip("/") == candidate.source_url.rstrip("/")
        ), "")
        if len(re.findall(r"[A-Za-z]", root)) > len(re.findall(r"[\u4e00-\u9fff]", root)) * 2:
            translation = brief.evidence_shots[0].full_translation.strip() or brief.evidence_shots[0].translation.strip()
            if not translation:
                errors.append("a non-Chinese root post requires adjacent Chinese translation")
            elif len(translation) > 140:
                errors.append("root-post Chinese translation exceeds 140 characters")
    return errors


def _glossary_match(value: str) -> tuple[str, str, str] | None:
    # Registry order is difficulty priority. The first matching group wins;
    # alias order resolves wording variants inside that group.
    for aliases, template, explanation_pattern in AUDIENCE_GLOSSARY:
        for term in aliases:
            if term.casefold() in value.casefold():
                return term, template, explanation_pattern
    return None


def _selected_glossary(brief: EditorialBrief) -> tuple[int, str, str, str] | None:
    # A hook term wins because misunderstanding it blocks the whole story.
    hook_match = _glossary_match(brief.attention_strategy.selected_hook)
    if hook_match:
        return 0, *hook_match
    full_video = "\n".join(
        " ".join((shot.fact, shot.audience_copy, shot.translation, shot.full_translation))
        for shot in brief.evidence_shots
    )
    match = _glossary_match(full_video)
    if not match:
        return None
    term, template, explanation_pattern = match
    shot_index = next(
        index for index, shot in enumerate(brief.evidence_shots)
        if term.casefold() in " ".join((
            shot.fact, shot.audience_copy, shot.translation, shot.full_translation,
        )).casefold()
    )
    return shot_index, term, template, explanation_pattern


def _missing_audience_glossary(brief: EditorialBrief) -> tuple[int, str] | None:
    selected = _selected_glossary(brief)
    if selected:
        shot_index, term, _, explanation_pattern = selected
        shot = brief.evidence_shots[shot_index]
        visible = " ".join((shot.fact, shot.audience_copy, shot.translation, shot.full_translation))
        if not re.search(explanation_pattern, visible, re.IGNORECASE):
            return shot_index + 1, term
    return None


def _inject_audience_glossary(brief: EditorialBrief) -> None:
    # One definition per short video prevents a useful explainer from turning
    # into a stack of glossary cards.
    selected = _selected_glossary(brief)
    if not selected:
        return
    shot_index, term, template, explanation_pattern = selected
    shot = brief.evidence_shots[shot_index]
    visible = " ".join((shot.fact, shot.audience_copy, shot.translation, shot.full_translation))
    if not re.search(explanation_pattern, visible, re.IGNORECASE):
        shot.audience_copy = template.format(term=term)


def is_audience_glossary_definition(value: str) -> bool:
    return any(
        re.search(explanation_pattern, value, re.IGNORECASE)
        for _, _, explanation_pattern in AUDIENCE_GLOSSARY
    )


def compile_evidence_shots(brief: EditorialBrief, candidate: Candidate) -> list[SceneProposal]:
    """Deterministically convert editorial shots into browser/render scenes."""
    scenes: list[SceneProposal] = []
    current_url = candidate.source_url
    for shot in brief.evidence_shots:
        role = _material_role(shot.kind)
        cues: list[CaptureCue] = []
        source_url = shot.source_url or current_url
        if source_url != current_url:
            cues.append(CaptureCue(CueAction.OPEN, f"open evidence source {shot.id}", value=source_url))
            current_url = source_url
        target = shot.target.strip()
        if shot.kind == EvidenceShotKind.TWEET_CARD:
            cues.append(CaptureCue(
                CueAction.WAIT, "hold the complete original post", wait_ms=round(shot.duration * 1000),
                shot_id=shot.id, translation=shot.translation,
            ))
        elif target:
            jump_ms = min(700, max(250, round(shot.duration * 180)))
            cues.extend([
                CaptureCue(CueAction.SCROLL, f"jump to exact evidence for {shot.id}", target=target, wait_ms=jump_ms),
                CaptureCue(
                    CueAction.HIGHLIGHT, f"hold exact evidence for {shot.id}", target=target,
                    wait_ms=max(500, round(shot.duration * 1000) - jump_ms), shot_id=shot.id,
                    translation=shot.translation,
                ),
            ])
        else:
            cues.append(CaptureCue(
                CueAction.WAIT, f"hold evidence source for {shot.id}", wait_ms=round(shot.duration * 1000),
                shot_id=shot.id, translation=shot.translation,
            ))
        scenes.append(SceneProposal(
            stage_name=shot.id,
            narration=f"internal editorial note: {shot.question}",
            caption=shot.fact,
            material_role=role,
            visual_action=_visual_action(shot),
            evidence_ids=shot.evidence_ids,
            beat_ids=shot.beat_ids,
            recording_cues=cues,
            duration_hint=shot.duration,
            screen_fact=shot.fact,
            screen_interpretation=shot.audience_copy,
            highlight_translation=shot.full_translation or shot.translation,
            source_excerpt=shot.target,
            visual_family=shot.visual_family,
            retention_job=shot.retention_job,
        ))
    return scenes


def _material_role(kind: EvidenceShotKind) -> MaterialRole:
    if kind in {EvidenceShotKind.TWEET_CARD, EvidenceShotKind.BROWSER_SECTION, EvidenceShotKind.PDF_PAGE, EvidenceShotKind.CODE_EXAMPLE, EvidenceShotKind.TERMINAL_DEMO}:
        return MaterialRole.PROOF
    if kind == EvidenceShotKind.IMAGE:
        return MaterialRole.ILLUSTRATION
    return MaterialRole.EXPLANATION


def _visual_action(shot: EvidenceShot) -> str:
    actions = {
        EvidenceShotKind.TWEET_CARD: "show the complete original post without splitting it",
        EvidenceShotKind.BROWSER_SECTION: "navigate a real browser to one exact page section",
        EvidenceShotKind.IMAGE: "show a source image without slow zoom",
        EvidenceShotKind.PDF_PAGE: "show the real PDF page and its page number",
        EvidenceShotKind.FIGURE: "focus the cited figure and caption",
        EvidenceShotKind.BENCHMARK_CHART: "focus the cited benchmark row, column, and conditions",
        EvidenceShotKind.CODE_EXAMPLE: "show the exact source code example in context",
        EvidenceShotKind.TERMINAL_DEMO: "show a reproducible command and its real output",
    }
    return actions[shot.kind]
