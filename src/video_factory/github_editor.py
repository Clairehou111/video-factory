from __future__ import annotations

import html
import re

from .models import Evidence, GitHubFocusCandidate, GitHubProjectBrief


PROJECT_KINDS = {
    "tool", "cli_sdk", "agent_framework", "model_research", "infrastructure",
    "security_privacy", "template_ui", "other",
}
PROOF_ROLES = {"demo", "input_output", "trial"}
DECISION_ROLES = {"technical_edge", "adoption"}
FOCUS_ROLES = PROOF_ROLES | DECISION_ROLES | {"boundary"}
BOUNDARY_SIGNAL = re.compile(
    r"(?:不(?:支持|保证|适用|提供|允许|能|可|会|含|包括)|无法|不能|不可|未|仅|只|但|(?<!不)需|必须|依赖|限制|风险|成本|许可|版权|侵权|删除|门槛|警告|"
    r"\bnot\b|\bno\b|\bonly\b|\bcannot\b|\bcan't\b|\brequires?\b|\bwarning\b|\blimit|"
    r"\breview\b|\bbefore\b|\bmanual\b|\bhuman\b|\bcheck\b|\bcopyright\b|\bdelete\b|\bremove\b)",
    re.IGNORECASE,
)


def normalize_source_text(value: str) -> str:
    """Make Markdown/HTML evidence comparable with browser-visible targets."""
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"[`*_#>\[\]()]", "", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def browser_visible_markdown(value: str) -> str:
    """Best-effort conversion of one Markdown line to GitHub-visible text."""
    value = html.unescape(value)
    value = re.sub(r"^\s*[-*+]\s+(?:\[[ xX]\]\s*)?", "", value)
    value = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"^\s*#{1,6}\s*", "", value)
    value = re.sub(r"^\s*>\s*", "", value)
    value = re.sub(r"\*\*|__|`", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _match_key(value: str) -> str:
    return re.sub(r"\s+", "", normalize_source_text(value))


def _visible_substring(visible: str, target: str) -> str | None:
    target_visible = browser_visible_markdown(target)
    direct_index = visible.casefold().find(target_visible.casefold())
    if direct_index >= 0:
        return visible[direct_index:direct_index + len(target_visible)]
    compact_chars: list[str] = []
    original_indexes: list[int] = []
    for index, character in enumerate(visible):
        if not character.isspace():
            compact_chars.append(character.casefold())
            original_indexes.append(index)
    wanted = "".join(character.casefold() for character in target_visible if not character.isspace())
    start = "".join(compact_chars).find(wanted)
    if start < 0 or not wanted:
        return None
    end = start + len(wanted) - 1
    return visible[original_indexes[start]:original_indexes[end] + 1]


def resolve_grounded_target(target: str, evidence_ids: list[str], evidence: list[Evidence]) -> str | None:
    """Return actual browser-visible source text for a model-selected target.

    Models often omit whitespace surrounding Markdown emphasis.  The evidence
    gate should correct that presentation detail without accepting a
    paraphrase or invented capability.  Matching is line-local so unrelated
    words from different README sections can never be joined.
    """
    if not _match_key(target):
        return None
    allowed = set(evidence_ids)
    for item in evidence:
        if item.id not in allowed:
            continue
        for raw_line in item.quote.splitlines() or [item.quote]:
            visible = browser_visible_markdown(raw_line)
            resolved = _visible_substring(visible, target) if visible else None
            if resolved:
                return resolved
    return None


def resolve_adjacent_target_lines(
    target: str, evidence_ids: list[str], evidence: list[Evidence], prefer_boundary: bool = False,
) -> str | None:
    """Recover a real line when the model accidentally joined adjacent lines."""
    wanted = _match_key(browser_visible_markdown(target))
    if not wanted:
        return None
    allowed = set(evidence_ids)
    for item in evidence:
        if item.id not in allowed:
            continue
        lines = [browser_visible_markdown(line) for line in item.quote.splitlines()]
        lines = [line for line in lines if line]
        for window_size in (2, 3):
            for index in range(0, len(lines) - window_size + 1):
                window = lines[index:index + window_size]
                if wanted not in _match_key(" ".join(window)):
                    continue
                if prefer_boundary:
                    boundary = next((line for line in window if BOUNDARY_SIGNAL.search(line)), None)
                    if boundary:
                        return boundary
                return max(window, key=len)
    return None


def _looks_like_code(value: str) -> bool:
    return bool(re.search(
        r"(?:^|\s)(?:python\d*|uv|pip|npm|pnpm|yarn|cargo|docker|curl|git|make|sh|bash)(?:\s|$)|"
        r"--[a-z][\w-]*|\$[A-Z_]",
        value, re.IGNORECASE,
    ))


def _preceding_prose_target(target: str, evidence_ids: list[str], evidence: list[Evidence]) -> str | None:
    allowed = set(evidence_ids)
    for item in evidence:
        if item.id not in allowed:
            continue
        lines = item.quote.splitlines()
        for index, raw_line in enumerate(lines):
            visible = browser_visible_markdown(raw_line)
            if not visible or not _visible_substring(visible, target):
                continue
            fallback = None
            inside_fence = sum(1 for line in lines[:index] if line.strip().startswith("```")) % 2 == 1
            for previous in reversed(lines[max(0, index - 24):index]):
                if previous.strip().startswith("```"):
                    inside_fence = not inside_fence
                    continue
                if inside_fence:
                    continue
                prose = browser_visible_markdown(previous)
                if not prose or _looks_like_code(prose):
                    continue
                fallback = fallback or prose
                if prose.endswith(("：", ":")) or re.search(r"命令|示例|运行|使用|command|example|run|use", prose, re.IGNORECASE):
                    return prose
            return fallback
    return None


def _markdown_table_cell_target(
    target: str, evidence_ids: list[str], evidence: list[Evidence],
) -> str | None:
    """Resolve a model-selected Markdown table row to one real DOM cell.

    GitHub renders each pipe-delimited cell as a separate element, so a full
    row is valid content evidence but cannot be highlighted as one browser
    target.  Prefer the longest non-separator cell because it normally carries
    the explanatory claim rather than the short mode/name column.
    """
    if "|" not in target:
        return None
    allowed = set(evidence_ids)
    wanted = _match_key(target)
    for item in evidence:
        if item.id not in allowed:
            continue
        for raw_line in item.quote.splitlines():
            if "|" not in raw_line or wanted not in _match_key(browser_visible_markdown(raw_line)):
                continue
            cells = [
                browser_visible_markdown(cell)
                for cell in raw_line.strip().strip("|").split("|")
            ]
            cells = [
                cell for cell in cells
                if cell and not re.fullmatch(r":?-{3,}:?", cell)
            ]
            if cells:
                return max(cells, key=len)
    return None


def target_is_grounded(target: str, evidence_ids: list[str], evidence: list[Evidence]) -> bool:
    return resolve_grounded_target(target, evidence_ids, evidence) is not None


def canonicalize_github_brief(brief: GitHubProjectBrief, evidence: list[Evidence]) -> None:
    """Replace fuzzy Markdown selections with the exact visible source line."""
    if re.search(r"[/\\]", brief.file_tree_target.strip()):
        # A nested path proves which top-level directory contains the useful
        # module, but GitHub's repository home only renders that first segment.
        brief.file_tree_target = re.split(r"[/\\]", brief.file_tree_target.strip(), maxsplit=1)[0]
    metadata_ids = [item.id for item in evidence if item.source_kind in {"github:metadata", "github:repository"}]
    readme_ids = [item.id for item in evidence if item.source_kind == "github:readme"]
    brief.repo_description_target = resolve_grounded_target(brief.repo_description_target, metadata_ids, evidence) or brief.repo_description_target
    brief.readme_claim_target = resolve_grounded_target(brief.readme_claim_target, readme_ids, evidence) or brief.readme_claim_target
    for focus in brief.focus_candidates:
        resolved = resolve_grounded_target(focus.target, focus.evidence_ids, evidence)
        if not resolved and "\n" in focus.target:
            line_matches = [
                match for line in focus.target.splitlines()
                if line.strip() and (match := resolve_grounded_target(line, focus.evidence_ids, evidence))
            ]
            if focus.editorial_role == "boundary":
                resolved = next((line for line in line_matches if BOUNDARY_SIGNAL.search(line)), None)
            resolved = resolved or (line_matches[0] if line_matches else None)
        if not resolved:
            resolved = resolve_adjacent_target_lines(
                focus.target, focus.evidence_ids, evidence,
                prefer_boundary=focus.editorial_role == "boundary",
            )
        focus.target = resolved or focus.target
        if focus.browser_target:
            browser_resolved = resolve_grounded_target(focus.browser_target, focus.evidence_ids, evidence)
            if not browser_resolved:
                browser_resolved = resolve_adjacent_target_lines(
                    focus.browser_target, focus.evidence_ids, evidence,
                    prefer_boundary=focus.editorial_role == "boundary",
                )
            focus.browser_target = browser_resolved or focus.browser_target
            if "|" in focus.browser_target:
                focus.browser_target = _markdown_table_cell_target(
                    focus.browser_target, focus.evidence_ids, evidence,
                ) or focus.browser_target
            if _looks_like_code(focus.browser_target):
                focus.browser_target = _preceding_prose_target(
                    focus.target, focus.evidence_ids, evidence,
                ) or focus.browser_target
        elif _looks_like_code(focus.target):
            focus.browser_target = _preceding_prose_target(focus.target, focus.evidence_ids, evidence) or ""
        if not focus.browser_target:
            focus.browser_target = _markdown_table_cell_target(
                focus.target, focus.evidence_ids, evidence,
            ) or ""


def select_github_focuses(brief: GitHubProjectBrief) -> list[GitHubFocusCandidate]:
    """Choose one concrete proof plus one edge/adoption/boundary deterministically."""
    ordered = sorted(brief.focus_candidates, key=lambda item: (-item.score, item.id))
    if brief.selected_focus_ids:
        by_id = {item.id: item for item in brief.focus_candidates}
        return [by_id[identifier] for identifier in brief.selected_focus_ids if identifier in by_id]
    proof = next((item for item in ordered if item.editorial_role in PROOF_ROLES), None)
    if proof is None and ordered:
        proof = ordered[0]
    decision = next((item for item in ordered if item is not proof and item.editorial_role in DECISION_ROLES), None)
    if decision is None:
        decision = next((item for item in ordered if item is not proof), None)
    return [item for item in (proof, decision) if item is not None]


HOOK_STRATEGIES = {
    "conflict", "surprise", "practical_win", "warning", "counterintuitive", "verdict",
}
SUBJECTIVE_STANCE_SIGNAL = re.compile(
    r"太|真|够|别|先|值得|有意思|漂亮|解气|可惜|意外|警惕|谨慎|好消息|坏消息|"
    r"我(?:更|不)|看好|看衰|扎眼|大胆|狠|妙"
)
MILD_STANCES = {"这就有意思了", "值得一看", "可以看看", "有点意思", "不妨看看"}


def copy_width(value: str) -> float:
    """Approximate CJK display width instead of counting ASCII as Hanzi.

    Repository names are long in code points but materially narrower on the
    rendered canvas.  This mirrors the compositor's dynamic font fitting and
    stops valid names from being repaired into vague aliases.
    """
    width = 0.0
    for character in value.strip():
        if "\u4e00" <= character <= "\u9fff":
            width += 1.0
        elif character.isspace():
            width += 0.35
        elif character.isascii() and character.isalnum():
            width += 0.55
        else:
            width += 0.65
    return width


def _hook_similarity(left: str, right: str) -> float:
    def grams(value: str) -> set[str]:
        compact = re.sub(r"[^\w\u4e00-\u9fff]", "", value.casefold())
        return {compact[index:index + 2] for index in range(max(0, len(compact) - 1))}
    one, two = grams(left), grams(right)
    return len(one & two) / len(one | two) if one and two else 0.0


def compose_github_hook(brief: GitHubProjectBrief) -> str:
    """Build the visible hook from separately reviewable editorial fields."""
    if brief.hook_opening.strip():
        return brief.hook_opening.strip()
    stance = brief.hook_stance.strip().rstrip("：:，,。！!")
    fact = brief.hook_fact.strip().lstrip("：:，,")
    if not stance:
        return fact
    if not fact:
        return stance
    return f"{stance}：{fact}"


def validate_github_brief(brief: GitHubProjectBrief, evidence: list[Evidence]) -> list[str]:
    errors: list[str] = []
    evidence_ids = {item.id for item in evidence}
    browser_evidence = [item for item in evidence if item.source_kind in {"github:metadata", "github:repository", "github:readme"}]
    factual_evidence_ids = {
        item.id for item in evidence if item.source_kind in {
            "github:metadata", "github:repository", "github:readme", "github:linked_context",
            "web:agent_primary_source", "web:official_background",
        }
    }
    if brief.project_kind not in PROJECT_KINDS:
        errors.append(f"unsupported project_kind: {brief.project_kind}")
    if brief.hook_strategy not in HOOK_STRATEGIES:
        errors.append(f"unsupported hook_strategy: {brief.hook_strategy or 'missing'}")
    structured_hook = any((brief.hook_opening, brief.hook_reveal, brief.hook_verdict))
    if structured_hook:
        hook_parts = {
            "hook_opening": (brief.hook_opening.strip(), 10, 28),
            "hook_reveal": (brief.hook_reveal.strip(), 10, 34),
            "hook_verdict": (brief.hook_verdict.strip(), 8, 30),
        }
        for name, (text, minimum, maximum) in hook_parts.items():
            units = copy_width(text)
            if not minimum <= units <= maximum:
                errors.append(
                    f"{name} must be {minimum}–{maximum} Chinese-character-equivalents "
                    f"and fit one complete screen (current {units:.1f})"
                )
        if brief.hook_opening.strip("！!。？?") in MILD_STANCES or brief.hook_opening.strip("！!。？?") == brief.hook_stance.strip("！!。"):
            errors.append("hook_opening must contain the event/job, not a bare emotional reaction")
        values = [brief.hook_opening, brief.hook_reveal, brief.hook_verdict]
        for index, left in enumerate(values):
            for right in values[index + 1:]:
                if left.strip() == right.strip() or _hook_similarity(left, right) > 0.58:
                    errors.append("the three cold-open screens must add new information instead of repeating or paraphrasing")
    else:
        if not 2 <= len(brief.hook_stance.strip()) <= 12:
            errors.append("hook_stance must be a 2–12 character subjective reaction")
        elif not SUBJECTIVE_STANCE_SIGNAL.search(brief.hook_stance):
            errors.append("hook_stance must contain an actual subjective reaction, not another neutral label")
        elif brief.hook_stance.strip("！!。") in MILD_STANCES:
            errors.append("hook_stance is too mild for a cold open; use a specific high-emotion reaction")
    if not 6 <= len(brief.project_title.strip()) <= 40:
        errors.append("project_title must be a 6–40 character stable GitHub walkthrough title")
    if brief.project_title.strip() == compose_github_hook(brief):
        errors.append("project_title must identify the walkthrough instead of repeating the cold-open hook")
    if not structured_hook:
        if not 6 <= len(brief.hook_fact.strip()) <= 36:
            errors.append("hook_fact must be a 6–36 character concrete conflict, result, or benefit")
        if brief.hook_fact.strip().startswith(("一个", "这个", "该项目", "本项目")):
            errors.append("hook_fact must lead with the named subject or payoff, not a generic project summary")
        if not any(marker in brief.hook_fact for marker in ("，", "但", "却", "已经", "直接", "只要", "也能", "开始", "还没")):
            errors.append("hook_fact must expose a concrete tension or payoff, not another neutral description")
        if brief.hook_fact.strip().startswith(("Python", "HTTP", "CLI", "SDK", "标准库")):
            errors.append("hook_fact must lead with the viewer-facing conflict or payoff; keep the implementation stack in the walkthrough")
    if not brief.hook_evidence_ids or not set(brief.hook_evidence_ids) <= evidence_ids:
        errors.append("hook_fact must cite existing hook_evidence_ids")
    elif not set(brief.hook_evidence_ids) <= factual_evidence_ids:
        errors.append("hook_fact must cite repository, linked-context, or primary-web evidence, not multimodal analysis alone")
    if not brief.subject_name.strip() or not brief.subject_action.strip() or not brief.subject_consequence.strip():
        errors.append("subject_name, subject_action, and subject_consequence must identify who does what and why it matters")
    background_values = [brief.background_actor, brief.background_action, brief.background_consequence]
    if any(value.strip() for value in background_values):
        if not all(value.strip() for value in background_values):
            errors.append("background_actor, background_action, and background_consequence must be filled together")
        if not brief.background_evidence_ids or not set(brief.background_evidence_ids) <= factual_evidence_ids:
            errors.append("background context must cite repository-linked or primary evidence")
    elif brief.background_evidence_ids:
        errors.append("background_evidence_ids require a named background actor/action/consequence")
    if copy_width(compose_github_hook(brief)) > 32:
        errors.append("composed GitHub hook must be at most 32 characters")
    required_text = {
        "core_job": brief.core_job,
        "input_output": brief.input_output,
        "adoption_path": brief.adoption_path,
        "unique_edge": brief.unique_edge,
        "verdict": brief.verdict,
        "repo_description_target": brief.repo_description_target,
        "readme_claim_target": brief.readme_claim_target,
        "file_tree_target": brief.file_tree_target,
    }
    errors.extend(f"missing {name}" for name, value in required_text.items() if not value.strip())
    if re.search(r"[/\\]", brief.file_tree_target.strip()):
        errors.append(
            "file_tree_target must be one top-level file or directory name visible on the repository home; nested paths are not renderable"
        )
    translations = {
        "repo_description_translation": brief.repo_description_translation,
        "readme_claim_translation": brief.readme_claim_translation,
        **{f"focus {focus.id} translation": focus.browser_translation or focus.translation for focus in brief.focus_candidates},
    }
    errors.extend(
        f"{name} must be at most 44 characters for adjacent display"
        for name, value in translations.items() if len(value.strip()) > 44
    )
    if not 2 <= len(brief.focus_candidates) <= 5:
        errors.append("focus_candidates must contain 2–5 ranked candidates")
    seen: set[str] = set()
    for focus in brief.focus_candidates:
        if not focus.id.strip() or focus.id in seen:
            errors.append(f"duplicate or empty focus id: {focus.id}")
        seen.add(focus.id)
        if focus.editorial_role not in FOCUS_ROLES:
            errors.append(f"unsupported editorial_role: {focus.editorial_role}")
        if not set(focus.evidence_ids) <= evidence_ids:
            errors.append(f"focus {focus.id} references unknown evidence")
        if not target_is_grounded(focus.target, focus.evidence_ids, browser_evidence):
            errors.append(f"focus {focus.id} target is not present in cited evidence: {focus.target}")
        if focus.browser_target and not target_is_grounded(focus.browser_target, focus.evidence_ids, browser_evidence):
            errors.append(f"focus {focus.id} browser_target is not present in cited evidence: {focus.browser_target}")
        if "|" in focus.browser_target:
            errors.append(
                f"focus {focus.id} browser_target spans Markdown table cells; select one rendered cell"
            )
        if focus.editorial_role == "boundary" and not BOUNDARY_SIGNAL.search(focus.target):
            errors.append(
                f"focus {focus.id} is labelled boundary but its source text contains no actual condition or limitation; "
                "choose a real boundary or change editorial_role to technical_edge/adoption"
            )
        score_values = (
            focus.viewer_value, focus.visual_proof, focus.distinctiveness,
            focus.actionability, focus.risk_importance, focus.redundancy_penalty,
        )
        if any(value < 0 or value > 3 for value in score_values):
            errors.append(f"focus {focus.id} scores must be integers from 0 to 3")
    if not target_is_grounded(
        brief.repo_description_target,
        [item.id for item in evidence if item.source_kind in {"github:metadata", "github:repository"}],
        evidence,
    ):
        errors.append("repo_description_target is not present in GitHub metadata evidence")
    readme_ids = [item.id for item in evidence if item.source_kind == "github:readme"]
    if not target_is_grounded(brief.readme_claim_target, readme_ids, evidence):
        errors.append("readme_claim_target is not present in README evidence")
    selected = select_github_focuses(brief)
    if len(brief.selected_focus_ids) != 2 or len(set(brief.selected_focus_ids)) != 2:
        errors.append("selected_focus_ids must contain exactly two distinct candidate ids")
    if len(selected) != 2:
        errors.append("brief must yield exactly two selected focuses")
    else:
        if not any(item.editorial_role in PROOF_ROLES for item in selected):
            errors.append("selected focuses must include a demo, input/output, or trial proof")
        if not any(item.editorial_role in DECISION_ROLES for item in selected):
            errors.append("selected focuses must include a technical edge or adoption point; boundary is not a default final-cut focus")
    return errors
