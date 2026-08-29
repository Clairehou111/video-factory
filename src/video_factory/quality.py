from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .models import ContentType, MaterialRole, RenderManifest
from .github_editor import compose_github_hook, select_github_focuses, validate_github_brief
from .github_editor import normalize_source_text
from .narrative import requirements_for
from .safety import review_evidence
from .editorial import validate_editorial_structure


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def validate_manifest(manifest: RenderManifest, workspace: Path | None = None) -> list[CheckResult]:
    """Validate publish-blocking provenance and basic edit constraints."""
    evidence_ids = {item.id for item in manifest.evidence}
    checks: list[CheckResult] = []
    checks.append(CheckResult("source_urls", bool(manifest.source_urls), "至少保留一个可追溯来源 URL"))
    checks.append(CheckResult(
        "footer_policy",
        not manifest.footer_shows_source_url,
        "画面 footer 只能表达结论；来源保存在证据包中",
    ))
    checks.append(CheckResult(
        "fixed_footer",
        bool(manifest.fixed_footer),
        "首版模板必须提供全程固定的事件结论",
    ))
    checks.append(CheckResult(
        "fixed_hook",
        bool(manifest.fixed_hook or (manifest.scenes and manifest.scenes[0].caption.strip())),
        "必须提供可编排为独立冷开场的钩子",
    ))
    checks.append(CheckResult(
        "fixed_title",
        bool(manifest.fixed_title or (manifest.scenes and manifest.scenes[0].caption.strip())),
        "冷开场结束后保留稳定的主题标题",
    ))
    visible_copy = "\n".join(filter(None, [
        manifest.fixed_hook, manifest.fixed_title, manifest.fixed_footer,
        *(scene.caption for scene in manifest.scenes),
        *(scene.screen_fact for scene in manifest.scenes),
        *(scene.screen_interpretation for scene in manifest.scenes),
    ]))
    editorial_strategy_copy = ""
    if manifest.editorial_brief:
        strategy = manifest.editorial_brief.attention_strategy
        editorial_strategy_copy = "\n".join(filter(None, [
            strategy.hook_fact, strategy.conflict, strategy.surprise, strategy.stakes,
            strategy.stance, strategy.payoff, *strategy.hook_candidates,
        ]))
    internal_labels = [
        label for label in ("仓库描述", "README声明", "证据带读", "关键结论", "翻译：", "译为：")
        if label.casefold() in visible_copy.casefold()
    ]
    hook = (manifest.fixed_hook or "").strip()
    generic_hook = any(hook.startswith(prefix) for prefix in ("一个工具", "这个项目", "这个仓库", "一款工具"))
    generic_advice = [
        phrase for phrase in ("替你决定", "仍是你的责任", "最终还是要你", "不能代替你", "仍需你负责")
        if phrase in visible_copy
    ]
    checks.append(CheckResult(
        "viewer_facing_copy",
        not internal_labels and not generic_hook and not generic_advice,
        "可见文案不暴露内部标签，钩子与结论均指向具体项目" if not internal_labels and not generic_hook and not generic_advice
        else "删除内部标签、空泛钩子或泛化说教：" + "、".join([*internal_labels, *generic_advice, hook if generic_hook else ""]),
    ))
    unsupported_downgrades = [
        phrase for phrase in (
            "适合快速原型而非成品交付", "只适合原型", "仅适合原型", "不适合成品",
            "不适合生产", "不能用于正式项目", "质量仍需人工把关", "原创性和质量控制仍需人工",
        ) if phrase in visible_copy
    ]
    checks.append(CheckResult(
        "unsupported_editorial_downgrade",
        not unsupported_downgrades,
        "未把外部依赖或开源属性擅自解释为只能做原型" if not unsupported_downgrades
        else "删除无来源的负面泛化，改写为具体限制：" + "、".join(unsupported_downgrades),
    ))
    ritual_uncertainty = [
        phrase for phrase in (
            "仅停留在概念层面", "只是一个概念", "不宜过度期待", "实际能力有待验证",
            "影响有待验证", "尚未成熟", "仍是未知数",
        )
        if phrase in visible_copy and phrase not in "\n".join(item.quote for item in manifest.evidence)
    ]
    checks.append(CheckResult(
        "unsupported_uncertainty",
        not ritual_uncertainty,
        "未知信息只按来源具体陈述，不用习惯性负面结论收尾" if not ritual_uncertainty
        else "删除无来源的泼冷水式结论，改写为已实现能力或具体未知项：" + "、".join(ritual_uncertainty),
    ))
    closing = (manifest.fixed_footer or "").strip()
    ritual_closing = bool(re.search(
        r"(?:但|不过|然而).{0,40}(?:未知|未公开|有待|需关注|待披露|尚未.{0,8}公开)|"
        r"(?:尚未.{0,8}公开|影响范围.{0,10}(?:未知|未公开)|值得关注.{0,12}(?:后续|影响)|"
        r"需(?:持续)?关注|仍待.{0,8}(?:公开|验证|研究)|有待验证|进一步研究)[。！]?$",
        closing, re.IGNORECASE,
    ))
    checks.append(CheckResult(
        "value_forward_closing",
        not ritual_closing,
        "结尾落在明确影响、观点或行动，而不是未知项和仪式性保留" if not ritual_closing
        else "把明确限制留在对应证据镜头，结尾必须回答开头并给出观点/影响",
    ))
    raw_evidence_text = "\n".join(item.quote for item in manifest.evidence)
    visible_manual_claim = re.search(r"(?:仍|还|必须|需要|需).{0,10}人工.{0,6}(?:检查|审核|把关|复核)", visible_copy)
    sourced_manual_boundary = re.search(
        r"人工.{0,12}(?:检查|审核|把关|复核)|(?:manual|human).{0,12}(?:check|review)|review.{0,12}(?:before|manually)",
        raw_evidence_text, re.IGNORECASE,
    )
    checks.append(CheckResult(
        "manual_review_grounding",
        not visible_manual_claim or bool(sourced_manual_boundary),
        "人工审核边界有明确来源或未被擅自添加" if not visible_manual_claim or sourced_manual_boundary
        else "项目没有明确要求人工检查，不能自动补成负面边界",
    ))
    if manifest.topic_type and manifest.topic_type.value == "github_project":
        strong_attack = re.search(r"绕过|破解|攻破|击破|干碎|失效", visible_copy)
        opening_source = "\n".join(
            item.quote if item.source_kind in {"github:metadata", "github:repository"}
            else "\n".join(item.quote.splitlines()[:80])
            for item in manifest.evidence
            if item.source_kind in {"github:metadata", "github:repository", "github:readme"}
        )
        explicit_attack = re.search(
            r"绕过|破解|攻破|击破|干碎|(?:^|\W)(?:bypass|crack|defeat|evade)(?:\W|$)",
            opening_source, re.IGNORECASE,
        )
        checks.append(CheckResult(
            "verb_strength",
            not strong_attack or bool(explicit_attack),
            "安全/隐私项目没有把清理能力夸大成绕过官方机制" if not strong_attack or explicit_attack
            else "原文只支持清理/去除/检测，不能升级为绕过、破解或攻破",
        ))
    quantitative_pattern = re.compile(
        r"(?:约|近|超过|超|不到|从)?(?:\d+(?:\.\d+)?|[数几](?:[十百千万亿])?|[十百千万亿]+)\s*"
        r"(?:%|％|倍|秒|分钟|小时|天|周|月|年|万美元|亿美元|美元|美金|人民币|万元|亿元)"
    )
    evidence_text = normalize_source_text("\n".join(item.quote for item in manifest.evidence))
    visible_quantities = list(dict.fromkeys(quantitative_pattern.findall(visible_copy + "\n" + editorial_strategy_copy)))
    unsupported_quantities = [
        claim for claim in visible_quantities
        if normalize_source_text(claim) not in evidence_text and not _quantity_supported(claim, raw_evidence_text)
    ]
    checks.append(CheckResult(
        "quantified_claims",
        not unsupported_quantities,
        "可见数字和效率量化均来自证据" if not unsupported_quantities
        else "可见文案包含证据中不存在的量化说法：" + "、".join(unsupported_quantities),
    ))
    checks.append(CheckResult(
        "audio_mode",
        manifest.audio_mode == "bgm_only",
        "成片采用无旁白、仅背景音乐模式" if manifest.audio_mode == "bgm_only" else f"不允许的音频模式：{manifest.audio_mode}",
    ))
    license_status = manifest.music_license_status.strip().casefold()
    license_ok = license_status in {"verified", "licensed", "original", "royalty_free_verified"} and bool(manifest.license_records)
    checks.append(CheckResult(
        "music_license_record",
        license_ok,
        f"背景音乐许可状态：{manifest.music_license_status}" if license_ok
        else "背景音乐仍缺少已核验的授权状态和 license_records；样片可生成，但不能发布",
    ))
    safety_review = review_evidence(manifest.evidence)
    checks.append(CheckResult(
        "editorial_safety_review",
        not safety_review.requires_human_review,
        "未发现需要升级审核的敏感内容" if not safety_review.requires_human_review
        else "发布前必须人工审核：" + "；".join(safety_review.reasons),
    ))
    if manifest.topic_type:
        answered = {beat.id for beat in manifest.story_beats if beat.answer.strip() and beat.evidence_ids}
        missing = [item.id for item in requirements_for(manifest.topic_type, manifest.content_type) if item.id not in answered]
        checks.append(CheckResult(
            "narrative_contract",
            not missing,
            "已回答该内容类型的叙事问题" if not missing else f"缺少叙事问题：{', '.join(missing)}",
        ))
    if manifest.topic_type and manifest.topic_type.value == "github_project":
        brief_errors = validate_github_brief(manifest.github_brief, manifest.evidence) if manifest.github_brief else ["missing github_brief"]
        selected = select_github_focuses(manifest.github_brief) if manifest.github_brief else []
        expected_hook = compose_github_hook(manifest.github_brief) if manifest.github_brief else ""
        if manifest.fixed_hook != expected_hook:
            brief_errors.append("fixed_hook does not match structured hook_stance + hook_fact")
        hook_errors = [
            error for error in brief_errors
            if error.startswith("hook_") or "GitHub hook" in error or "fixed_hook" in error
        ]
        checks.append(CheckResult(
            "editorial_hook",
            not hook_errors,
            "冷开场由主观态度、具体事实与证据 ID 结构化生成" if not hook_errors
            else "GitHub 钩子无效：" + "; ".join(hook_errors),
        ))
        cold_open = manifest.cold_open_beats
        cold_open_errors: list[str] = []
        if len(cold_open) != 3:
            cold_open_errors.append("must contain exactly three beats")
        else:
            emphases = [beat.emphasis for beat in cold_open]
            valid_progressions = {
                ("event_hook", "capability_reveal", "editorial_verdict"),
                ("reaction", "conflict", "payoff"),  # archived manifests
            }
            if tuple(emphases) not in valid_progressions:
                cold_open_errors.append("beats must progress event → reveal → verdict")
            if not 2.5 <= sum(beat.duration for beat in cold_open) <= 4.0:
                cold_open_errors.append("total beat duration must be 2.5–4.0 seconds")
            if any(not beat.text.strip() or not set(beat.evidence_ids) <= evidence_ids for beat in cold_open):
                cold_open_errors.append("every beat needs text and valid evidence IDs")
        checks.append(CheckResult(
            "github_cold_open",
            not cold_open_errors,
            "GitHub 带读前有独立的高情绪冷开场分镜" if not cold_open_errors
            else "GitHub 冷开场无效：" + "; ".join(cold_open_errors),
        ))
        repo_url = next((url for url in manifest.source_urls if "github.com/" in url), "")
        repo_name = repo_url.rstrip("/").rsplit("/", 1)[-1] if repo_url else ""
        checks.append(CheckResult(
            "github_project_title",
            bool(repo_name and manifest.fixed_title and repo_name.casefold() in manifest.fixed_title.casefold()),
            "冷开场结束后恢复包含 repo 名称的项目标题" if repo_name and manifest.fixed_title and repo_name.casefold() in manifest.fixed_title.casefold()
            else "GitHub 带读标题必须保留 repo 名称",
        ))
        valid = not brief_errors and len(selected) == 2
        checks.append(CheckResult(
            "github_walkthrough",
            valid,
            "GitHub 成片包含完整首页身份、README 核心声明与两个证据排序后的重点" if valid else "GitHub 项目分析无效：" + "; ".join(brief_errors),
        ))
    elif manifest.topic_type and manifest.editorial_brief:
        brief_errors = validate_editorial_structure(
            manifest.editorial_brief,
            _candidate_from_manifest(manifest), manifest.evidence,
            manifest.topic_type, manifest.content_type,
        )
        strategy = manifest.editorial_brief.attention_strategy
        progression = [shot.relation_to_previous for shot in manifest.editorial_brief.evidence_shots[1:]]
        checks.extend([
            CheckResult(
                "attention_strategy", not brief_errors,
                "钩子、冲突、主体、证据和结论已结构化绑定" if not brief_errors
                else "短视频内容策略无效：" + "; ".join(brief_errors),
            ),
            CheckResult(
                "hook_payoff_loop", bool(strategy.selected_hook and strategy.payoff and strategy.selected_hook != strategy.payoff),
                "结尾回答开头而不是重复开头",
            ),
            CheckResult(
                "information_progression", bool(progression) and all(item.strip() for item in progression),
                "相邻分镜均声明了新的因果、反差、证据或结果",
            ),
        ])
    if manifest.topic_type:
        bad_beats = [beat.id for beat in manifest.story_beats if not set(beat.evidence_ids) <= evidence_ids]
        checks.append(CheckResult(
            "story_beat_evidence",
            not bad_beats,
            "叙事回答均能回溯到证据" if not bad_beats else f"叙事回答引用未知证据：{', '.join(bad_beats)}",
        ))

    for scene in manifest.scenes:
        checks.append(CheckResult(
            f"scene:{scene.id}:timing",
            scene.start >= 0 and scene.end > scene.start,
            "镜头时间必须是递增的正区间",
        ))
        checks.append(CheckResult(
            f"scene:{scene.id}:one_main_message",
            bool(scene.caption.strip()) and bool(scene.visual_action.strip()),
            "每幕需要明确的一条信息和一个视觉动作",
        ))
        checks.append(CheckResult(
            f"scene:{scene.id}:sound_off_readability",
            bool((scene.screen_fact or scene.caption).strip()),
            "每幕均包含可独立理解的观众信息；内部导演说明不参与渲染",
        ))
        highlighted = "highlight" in scene.visual_action.casefold()
        checks.append(CheckResult(
            f"scene:{scene.id}:highlight_translation",
            not highlighted or bool((scene.highlight_translation or "").strip()),
            "高亮外文时必须提供对应中文翻译" if highlighted else "本幕无外文高亮要求",
        ))
        checks.append(CheckResult(
            f"scene:{scene.id}:visual_cadence",
            scene.duration <= 5,
            "视觉变化间隔不超过 5 秒" if scene.duration <= 5 else "镜头超过 5 秒，必须拆分或添加明确视觉变化",
        ))
        if scene.material_role in {MaterialRole.PROOF, MaterialRole.EXPLANATION}:
            missing = sorted(set(scene.evidence_ids) - evidence_ids)
            checks.append(CheckResult(
                f"scene:{scene.id}:evidence",
                not missing,
                "证据镜头必须引用已记录证据" if not missing else f"缺失 evidence_ids: {', '.join(missing)}",
            ))

    if manifest.topic_type and manifest.topic_type.value == "github_project":
        bounds = {ContentType.FLASH: (12, 15), ContentType.EXPLAINER: (15, 25), ContentType.DEEP_DIVE: (25, 40)}
    else:
        # A dense technical explanation may be complete below 20 seconds.
        # Duration is a ceiling/legibility constraint, not a quota to fill.
        bounds = {ContentType.FLASH: (8, 15), ContentType.EXPLAINER: (15, 45), ContentType.DEEP_DIVE: (25, 90)}
    lower, upper = bounds[manifest.content_type]
    checks.append(CheckResult(
        "duration_band",
        lower <= manifest.duration <= upper,
        f"{manifest.content_type} 目标时长为 {lower}–{upper} 秒；当前 {manifest.duration:.2f} 秒",
    ))

    if workspace:
        for evidence in manifest.evidence:
            if evidence.captured_asset:
                checks.append(CheckResult(
                    f"evidence:{evidence.id}:asset",
                    (workspace / evidence.captured_asset).is_file(),
                    "证据资产存在" if (workspace / evidence.captured_asset).is_file() else "证据资产文件缺失",
                ))

    ordered = sorted(manifest.scenes, key=lambda scene: scene.start)
    overlaps = [f"{left.id}/{right.id}" for left, right in zip(ordered, ordered[1:]) if left.end > right.start]
    checks.append(CheckResult(
        "timeline_order",
        not overlaps,
        "镜头时间轴无重叠" if not overlaps else f"镜头时间重叠：{', '.join(overlaps)}",
    ))
    return checks


def _candidate_from_manifest(manifest: RenderManifest):
    """Build the minimum candidate view needed by editorial validation."""
    from .models import Candidate, SourceType

    source_url = manifest.source_urls[0] if manifest.source_urls else ""
    if "x.com/" in source_url or "twitter.com/" in source_url:
        source_type = SourceType.TWEET
    elif source_url.casefold().endswith(".pdf") or "arxiv.org" in source_url:
        source_type = SourceType.PAPER
    else:
        source_type = SourceType.WEB
    return Candidate(manifest.candidate_id, source_type, source_url, manifest.fixed_title or manifest.fixed_hook or "story")


def _quantity_supported(claim: str, evidence_text: str) -> bool:
    """Match translated quantities without allowing arithmetic inventions."""
    number_match = re.search(r"\d+(?:\.\d+)?", claim.replace(",", ""))
    if not number_match:
        return False
    value = float(number_match.group())
    if any(marker in claim for marker in ("美元", "美金")):
        claim_scale = 100_000_000 if "亿美元" in claim else 10_000 if "万美元" in claim else 1
        claim_dollars = value * claim_scale
        source_amounts: list[float] = []
        currency_patterns = (
            r"\$\s*([\d,.]+)\s*(billion|million|bn|m)?",
            r"([\d,.]+)\s*(billion|million|bn|m)?\s*(?:usd|u\.s\. dollars?|dollars?)",
        )
        source_scales = {"billion": 1_000_000_000, "bn": 1_000_000_000, "million": 1_000_000, "m": 1_000_000}
        for pattern in currency_patterns:
            for amount, scale in re.findall(pattern, evidence_text, re.IGNORECASE):
                source_amounts.append(float(amount.replace(",", "")) * source_scales.get(scale.casefold(), 1))
        if any(abs(amount - claim_dollars) <= max(1, claim_dollars) * 1e-9 for amount in source_amounts):
            return True
    word_numbers = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    }
    time_unit = next((
        english for chinese, english in (
            ("秒", "seconds?"), ("分钟", "minutes?"), ("小时", "hours?"),
            ("天", "days?"), ("周", "weeks?"), ("月", "months?"), ("年", "years?"),
        ) if chinese in claim
    ), None)
    if time_unit and value.is_integer():
        word = next((name for name, number in word_numbers.items() if number == int(value)), None)
        if word and re.search(rf"\b{word}\s+{time_unit}\b", evidence_text, re.IGNORECASE):
            return True
    # Calendar dates are commonly rendered as Chinese 年/月 while an official
    # English page uses `Aug 16, 2026`.  Treat only unmistakable calendar
    # forms as equivalent; duration quantities still use the strict unit path.
    if re.search(r"\b\d{4}\s*年", claim):
        year = str(int(value))
        if re.search(rf"(?<!\d){re.escape(year)}(?!\d)", evidence_text) and re.search(
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
            evidence_text, re.IGNORECASE,
        ):
            return True
    month_match = re.fullmatch(r"\s*(\d{1,2})\s*月\s*", claim)
    if month_match:
        month_names = (
            "Jan(?:uary)?", "Feb(?:ruary)?", "Mar(?:ch)?", "Apr(?:il)?", "May", "Jun(?:e)?",
            "Jul(?:y)?", "Aug(?:ust)?", "Sep(?:tember)?", "Oct(?:ober)?", "Nov(?:ember)?", "Dec(?:ember)?",
        )
        month = int(month_match.group(1))
        if 1 <= month <= 12 and re.search(rf"\b(?:{month_names[month - 1]})\b", evidence_text, re.IGNORECASE):
            return True
    unit_groups = (
        (("年",), r"years?|yrs?|年"), (("秒",), r"seconds?|secs?|秒"),
        (("分钟",), r"minutes?|mins?|分钟"), (("小时",), r"hours?|hrs?|小时"),
        (("天",), r"days?|天"), (("周",), r"weeks?|周"), (("月",), r"months?|月"),
        (("倍",), r"times?|x|倍"), (("%", "％"), r"%|percent|％"),
        (("美元", "美金"), r"usd|dollars?|\$|美元|美金"),
        (("人民币",), r"cny|rmb|人民币|元"),
    )
    units = next((pattern for markers, pattern in unit_groups if any(marker in claim for marker in markers)), None)
    if not units:
        return False
    variants = {f"{value:g}"}
    if value.is_integer():
        integer = int(value)
        variants.update({str(integer), f"{integer:,}"})
    number_pattern = "(?:" + "|".join(re.escape(item) for item in sorted(variants, key=len, reverse=True)) + ")"
    return bool(re.search(rf"(?<!\d){number_pattern}\s*(?:{units})", evidence_text, re.IGNORECASE))


def is_publishable(checks: list[CheckResult]) -> bool:
    return all(item.passed for item in checks)
