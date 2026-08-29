from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ContentType, SourceType, TopicType


@dataclass(frozen=True, slots=True)
class NarrativeRequirement:
    id: str
    question: str
    required_for_flash: bool = False


TEMPLATES: dict[TopicType, tuple[NarrativeRequirement, ...]] = {
    TopicType.PRACTICE_POST: (
        NarrativeRequirement("author_claim", "作者具体声称或做了什么", True),
        NarrativeRequirement("evidence_context", "原帖、作者身份和上下文能证实什么", True),
        NarrativeRequirement("our_observation", "我们从中观察到的工作流信号"),
        NarrativeRequirement("scope", "为什么它不是普遍规律，适用边界在哪里", True),
    ),
    TopicType.GITHUB_PROJECT: (
        NarrativeRequirement("problem", "它解决哪一个具体问题", True), NarrativeRequirement("input_output", "输入和输出是什么"),
        NarrativeRequirement("install_run", "如何安装并真实运行"), NarrativeRequirement("labor_saved", "节省了哪段重复劳动"),
        NarrativeRequirement("trial_task", "一个具体的试用任务"),
        NarrativeRequirement("workflow_fit", "它已经能为哪类工作流创造价值"),
    ),
    TopicType.TOOL_SDK_AGENT: (
        NarrativeRequirement("manual_before", "开发者原本需要手动做什么", True), NarrativeRequirement("input_output", "工具的输入和输出是什么", True),
        NarrativeRequirement("integration", "安装或接入方式是什么"), NarrativeRequirement("labor_saved", "节省了哪段重复劳动"),
        NarrativeRequirement("human_check", "仅当官方明确写出时，哪些地方仍须人工检查"), NarrativeRequirement("trial_task", "给一个具体试用任务"),
        NarrativeRequirement("workflow_fit", "是否值得纳入工作流"),
    ),
    TopicType.MODEL_OR_PRODUCT: (
        NarrativeRequirement("released", "到底发布了什么", True), NarrativeRequirement("verified_capability", "官方页面中哪项能力可验证", True),
        NarrativeRequirement("practical_change", "对使用者或开发者的实际变化", True), NarrativeRequirement("unknown", "什么没有公布或不能推断"),
        NarrativeRequirement("trial_scenario", "合适的试用场景"), NarrativeRequirement("adoption_choice", "迁移、采用还是等待"),
    ),
    TopicType.COMPANY_OR_TEAM: (
        NarrativeRequirement("identity", "公司或团队是谁", True), NarrativeRequirement("product_direction", "具体在做什么技术产品", True),
        NarrativeRequirement("primary_evidence", "官网、公告或可信原始文件"), NarrativeRequirement("technical_route", "产品方向和技术路线，而非只报融资金额"),
        NarrativeRequirement("unknown", "当前已验证的事实与未知信息"), NarrativeRequirement("impact", "可能影响谁、哪个市场或工具链", True),
    ),
    TopicType.RESEARCH_OR_BENCHMARK: (
        NarrativeRequirement("research_question", "论文试图解决什么问题", True), NarrativeRequirement("method", "核心方法是什么"),
        NarrativeRequirement("primary_artifact", "关键图表、方法或代码模块"), NarrativeRequirement("conditions", "实验条件、数据集和比较对象", True),
        NarrativeRequirement("scope", "结论适用范围", True), NarrativeRequirement("unproven", "尚未被证明的部分", True),
        NarrativeRequirement("recommendation", "收藏、复现还是等待"),
    ),
    TopicType.OFFICIAL_ANNOUNCEMENT: (
        NarrativeRequirement("event", "发生了什么", True), NarrativeRequirement("official_change", "官方原文中的关键变化", True),
        NarrativeRequirement("impact", "对用户、开发者或工作流的影响", True), NarrativeRequirement("effective_scope", "何时生效、覆盖哪些人", True),
        NarrativeRequirement("migration", "需要迁移还是可以等待"), NarrativeRequirement("action", "一个具体行动或取舍问题"),
    ),
    TopicType.LINKED_EXTERNAL_SOURCE: (
        NarrativeRequirement("post_reference", "帖子提到了哪个外部来源", True), NarrativeRequirement("primary_opened", "已经打开并采集对应的原始来源", True),
        NarrativeRequirement("routed_template", "已按外部来源类型切换到相应带读模板", True),
    ),
}


def requirements_for(topic: TopicType, content_type: ContentType) -> tuple[NarrativeRequirement, ...]:
    requirements = TEMPLATES[topic]
    return tuple(item for item in requirements if item.required_for_flash) if content_type == ContentType.FLASH else requirements


def route_external_source(source_type: SourceType) -> TopicType:
    return {
        SourceType.GITHUB: TopicType.GITHUB_PROJECT, SourceType.PAPER: TopicType.RESEARCH_OR_BENCHMARK,
        SourceType.OFFICIAL_ANNOUNCEMENT: TopicType.OFFICIAL_ANNOUNCEMENT, SourceType.TWEET: TopicType.PRACTICE_POST,
    }.get(source_type, TopicType.TOOL_SDK_AGENT)


def extract_external_urls(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"https?://[^\s)\]>]+", text)))
