from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .models import (
    Candidate, ContentType, ContextGraph, EditorialOpportunity, Evidence,
    InformationRenderProfile, TopicType,
)
from .narrative import requirements_for
from .translation import IT_TRANSLATION_CONTRACT, PLAIN_CHINESE_CONTRACT


@dataclass(frozen=True, slots=True)
class StoryWriterPacket:
    """Provider-neutral, evidence-bounded input for an LLM or human editor."""

    candidate: Candidate
    evidence: list[Evidence]
    topic_type: TopicType
    content_type: ContentType
    target_duration: float
    editorial_direction: str = ""
    visual_policy: str = (
        "This is a BGM-only video: it has no voiceover and no timed subtitles. "
        "For GitHub, the emotional hook is a dedicated three-beat cold open; after it, the stable project title stays at the top and the conclusion stays at the bottom. Each scene must be understandable by reading its own fact and interpretation. "
        "Never display source URLs in the picture."
    )
    opportunity: EditorialOpportunity | None = None
    context_graph: ContextGraph | None = None
    render_profile: str = InformationRenderProfile.CLASSIC.value

    def prompt(self) -> str:
        radar = self.render_profile == InformationRenderProfile.RADAR_V2.value
        requirements = requirements_for(self.topic_type, self.content_type)
        # Evidence remains archived in full. The writing model receives a
        # bounded, source-aware excerpt so a malformed web extraction cannot
        # turn one short-video job into a 300k-token request.
        compact_evidence = []
        remaining_chars = 72_000
        for item in self.evidence:
            per_source = 30_000 if item.source_kind.startswith(("github:", "paper:")) else 18_000
            quote = item.quote[: min(per_source, remaining_chars)]
            compact_evidence.append({
                "id": item.id, "url": item.url, "quote": quote,
                "kind": item.source_kind, "notes": item.notes or "", "metadata": item.metadata,
            })
            remaining_chars -= len(quote)
            if remaining_chars <= 0:
                break
        schema = {
            "answers": [{"beat_id": item.id, "answer": "Chinese answer grounded in evidence_ids", "evidence_ids": ["evidence-id"]} for item in requirements],
            "footer": "one specific Chinese conclusion or impact, fixed for the full video",
        }
        github_contract = ""
        if self.topic_type == TopicType.GITHUB_PROJECT:
            # GitHub has its own editorial schema. Do not expose the generic
            # scene.material_role field beside focus editorial roles: models
            # otherwise reasonably confuse trial/boundary with proof/explanation.
            schema.pop("scenes", None)
            schema.pop("hook", None)
            schema["github_brief"] = {
                "project_kind": "tool|cli_sdk|agent_framework|model_research|infrastructure|security_privacy|template_ui|other",
                "hook_strategy": "direct_fact|conflict|surprise|practical_win|warning|counterintuitive|verdict",
                "hook_opening": "10–28 Chinese characters: a complete first-screen statement of what happened; use direct_fact when the repository action itself is important, otherwise use only an evidence-backed contrast, friction, reversal, relief, or consequence; never a bare reaction such as 太快了/太狠了",
                "hook_reveal": "10–34 Chinese characters: a new, evidence-backed capability/result that answers the opening; never repeat or merely rephrase hook_opening",
                "hook_verdict": "8–30 Chinese characters: our distinctive, value-forward judgment or consequence; for a maintained/popular repository, default to what becomes possible or worth trying, never force a ritual caveat",
                "hook_evidence_ids": ["evidence id supporting hook_fact"],
                "project_title": "6–40 character stable title used after the cold open; name the repo and its job, not the emotional hook",
                "subject_name": "exact project name",
                "subject_type": "project|company|person|vendor|team|ecosystem",
                "subject_action": "what this named subject concretely does",
                "subject_consequence": "what changes for the viewer or ecosystem",
                "background_actor": "named company/person/vendor that caused the project response; empty when no evidenced external trigger exists",
                "background_action": "the exact evidenced action by background_actor; empty without background_actor",
                "background_consequence": "why that action created this project opportunity/tension; empty without background_actor",
                "background_evidence_ids": ["evidence ids supporting the external trigger; empty with no trigger"],
                "core_job": "the concrete job and result in Chinese",
                "input_output": "specific input and output in Chinese",
                "adoption_path": "shortest real way to try it",
                "unique_edge": "what is genuinely different, or unknown",
                "boundary": "an explicit README limitation/warning only; empty string when the project does not state one",
                "verdict": "who should try it and who should wait",
                "repo_description_target": "exact browser-visible repository description",
                "repo_description_translation": "decisive Chinese meaning only, preferably 18–32 and never over 40 characters",
                "readme_claim_target": "exact browser-visible README sentence that best explains the project",
                "readme_claim_translation": "decisive Chinese meaning only, preferably 18–32 and never over 40 characters",
                "file_tree_target": "exact single top-level file or directory name visible on the repository home; never a nested path and never contain / or \\",
                "selected_focus_ids": ["one proof candidate id", "one decision candidate id"],
                "focus_candidates": [{
                    "id": "stable-id", "editorial_role": "demo|input_output|trial|technical_edge|adoption|boundary",
                    "target": "smallest exact browser-visible text or code excerpt", "translation": "Chinese meaning only when needed",
                    "browser_target": "optional exact prose line to highlight when target is code, or one exact cell when target is a Markdown table row; never span cells with |",
                    "browser_translation": "Chinese meaning of browser_target only when browser_target is non-Chinese",
                    "summary": "what this proves", "why_it_matters": "why either audience should care",
                    "evidence_ids": ["evidence-id"], "source_url": "source page URL or null",
                    "viewer_value": 0, "visual_proof": 0, "distinctiveness": 0,
                    "actionability": 0, "risk_importance": 0, "redundancy_penalty": 0
                }]
            }
            schema["github_scenes"] = [
                {
                    "stage": "repo_identity|readme_claim|selected_focus",
                    "message": "one concise Chinese message shown by this evidence shot",
                    "interpretation": "why it matters or its boundary in Chinese",
                    "focus_id": "null for repo/readme; selected candidate id for selected_focus",
                    "evidence_ids": ["evidence-id"],
                    "beat_ids": ["beat-id"],
                    "duration_hint": 5
                }
            ]
            github_contract = (
                "GitHub editorial contract: treat the repo page as provenance, not as the story. The final cut carries one core value, two strongest exact proofs, and a value-forward adoption verdict. Do not search for a boundary by default. Fill boundary only when README itself explicitly labels or states a limitation/warning; otherwise return an empty string and omit it from focus_candidates and scenes. "
                "Classify the repository before choosing proof. For tools prefer a visible result and fastest trial; for CLI/SDK prefer a minimal call and actual I/O; for agent/framework projects prefer a concrete task and one mechanism that changes workflow; for model/research prefer a reproducible capability and its conditions; for security/privacy prefer supported scope and risk boundary; for template/UI prefer the rendered result and customization/deploy path. "
                "Return 2–5 focus_candidates; two excellent candidates are better than filler. editorial_role describes story meaning only and must never be used as a material type. Score each independently from 0 to 3, then put exactly one demo/input_output/trial id and exactly one technical_edge/adoption id in selected_focus_ids. A boundary candidate is archival context only and is never selected for the default final cut. "
                "Return exactly four github_scenes in this order: repo_identity, readme_claim, selected_focus, selected_focus. The two selected_focus scenes must reference the two selected_focus_ids. GitHub scenes intentionally have no narration, material_role, recording_cues, or low-level rendering fields. "
                "Every target must be the smallest exact contiguous browser-visible substring present in its cited evidence. A target must stay on one rendered line or one intact paragraph: never merge adjacent source lines, remove a shell line-continuation backslash, or reconstruct a command. Prefer a complete sentence or a self-contained command line that the browser recorder can match literally. For code targets, also provide browser_target as the exact explanatory prose line immediately above the code; GitHub copy buttons make raw-code highlights visually ambiguous. The command remains the content proof while browser_target owns the yellow box. Every repo/readme/focus browser translation must be at most 44 characters, state only the decisive Chinese meaning, and omit exhaustive extension/provider lists. Never choose a generic heading when the useful line or code snippet exists. Never choose badges, sponsor blocks, Star History, or the README footer unless they are themselves the verified story. "
                "Evidence marked github:visual_analysis or x:visual_analysis is multimodal editorial context: use it to understand visible architecture, benchmark, screenshot text, or quoted-post context, but never use it as a browser target or the sole evidence for a hook/factual claim. Cite the associated repository/readme or parent X image with it. "
                "Viewer-facing hook/message/interpretation/footer must speak directly about a named actor and its consequence. Fill subject_name/action/consequence for every repository. If linked context proves that a named company, person, or vendor action triggered the project, also fill all background_* fields: hook_opening must name that actor and action, hook_reveal must name the repository and its response, and hook_verdict may state the evidence-backed consequence. Without such background, hook_opening itself must name the repository. Never write an ownerless opening such as 只需一个主题/现在可以自动化. Never expose production labels such as 仓库描述、README声明、证据带读、关键结论、翻译, and never begin the hook with generic wording such as 一个工具、这个项目、这个仓库."
                " Preserve verb strength exactly for security/privacy tools: strip/remove/clean/detect/inspect does not mean bypass, crack, defeat, break, or evade an official detector. Never upgrade 去除/清理/检测 into 绕过、破解、攻破、干碎、失效 unless the repository's opening claim explicitly makes that stronger claim. Class-level support is not proof against a vendor's private official detector."
                " Preserve scope and novelty exactly. Chinese absolutes and superlatives such as 全面、全部、所有、任何、彻底、完全、首次、第一次、唯一 are factual claims, not harmless emotional wording. Use one only when the cited evidence proves the same actor, object, time, and coverage. Phrases such as supported models, where supported, new models, or across products do not prove 全面/全部; a newly visible repository does not prove 首次/第一次/唯一. Build energy from a verified actor-action reversal or workflow payoff instead of enlarging scope."
                " Treat a maintained, widely adopted open-source repository with constructive enthusiasm: lead with the useful work it already makes possible. Popularity is not proof that every output is perfect, but external APIs, configurable providers, local installation, an open-source license, or the need to choose materials are normal engineering choices—not an automatic negative verdict. For such a repository, hook_verdict and footer should normally be affirmative and adoption-oriented, without a ritual 但/不过/然而 clause. Never write 只/仅适合原型、而非成品交付、不适合生产、不能用于正式项目、质量仍需人工把关, or an equivalent downgrade unless the repository explicitly states that exact limitation. Put concrete source-backed limitations in the relevant evidence scene instead of forcing them into the last word."
                " For the opening, do not return a top-level hook and do not return one sentence for the program to split. Fill hook_strategy, hook_opening, hook_reveal, hook_verdict, hook_evidence_ids, and project_title. The three hook texts are three independent screens: what happened → repository proof/response → concrete payoff/opinion. Each must be complete, introduce new information, and remain understandable alone. Shock is optional. Choose direct_fact when the named repository action is already the reason to stop; use conflict/counterintuitive/practical_win only when evidence earns it. If a repository genuinely removes, counters, audits, bypasses, or reverses a named vendor capability, conflict may name both sides. project_title is calmer and remains above the real walkthrough. Length limits are Chinese-character equivalents: Han characters count 1, ASCII letters/digits about 0.55, spaces about 0.35. Keep the real repository name; shorter than the stated limit is better."
            )
        else:
            opening_contract = (
                "Treat attention as a factual layer, not decoration. Use Hook/What happened → Proof → Takeaway; do not force shock. Select exactly one opening_mode: direct_fact when the verified event itself is important, conflict only for a real comparison, counter_intuitive only when evidence violates a common expectation, or developer_roi only for an explicit cost/time/performance/workflow gain. Never fabricate opposition, an incumbent comparison, or an ROI number. "
                if radar else
                "Treat attention as a factual layer, not decoration. Find a concrete tension, surprise, record scale, or high-stakes change, but never fabricate opposition or a cost conflict. "
            )
            radar_copy_contract = (
                "Radar visible-copy contract: ordinary headlines fit at most 20 Chinese-character-equivalents; when preserving an exact subject/model/project name makes that impossible, at most 28 and never more than two rendered lines. A highlighted Chinese gloss is 16–24 equivalents. Every concise narrative sentence contains at least one concrete metric, action verb, or named entity. Avoid 项目发布了、关于…的探讨、作者表示 as ownerless openings and ban vague phrases such as 反映了…的深度、体现了生态多样性、提供了新思考. A competitor or causal association absent from direct evidence may appear only as an explicit question and must be copied into editorial_inference; proof and takeaway must not assert it. category_label is optional UI metadata, not a judgment. Use only 模型发布、价格变化、开源项目、论文结果、工具更新、行业公告 when the source clearly fits; otherwise return empty. Never invent 神器、平替、突破、必看 as a category. direct_identifier must be copied exactly from supplied evidence or remain empty. "
                if radar else ""
            )
            schema["editorial_brief"] = {
                "headline": "specific adaptive upper-rail headline naming the actor and conflict; keep the exact entity even when it needs a smaller font or extra line",
                "subheadline": "new information: consequence, technical change, or open loop",
                "fixed_conclusion": "distinctive evidence-backed stance that resolves the opening",
                "duration_target": self.target_duration,
                "opening_mode": "direct_fact|conflict|counter_intuitive|developer_roi; choose from evidence and default to direct_fact when the event itself is enough",
                "category_label": "optional neutral factual label: 模型发布|价格变化|开源项目|论文结果|工具更新|行业公告; empty when none fits exactly",
                "direct_identifier": "optional exact HF:, pip install, docker pull, or github.com/org/repo identifier copied from evidence; empty otherwise",
                "editorial_inference": "the exact question-form inference used by the hook, or empty when every hook claim is directly proved",
                "attention_strategy": {
                    "hook_fact": "the concrete event/result that earns the first second",
                    "conflict": "named actor/action versus old workflow, incumbent, expectation, price, or migration cost",
                    "surprise": "specific counterintuitive detail, or empty only when none exists",
                    "stakes": "who is affected and what changes",
                    "stance": "our brave, specific editorial view",
                    "payoff": "the answer delivered at the end; not a repeat of the hook",
                    "hook_candidates": ["three different evidence-bounded Chinese hooks"],
                    "hook_evidence_ids": ["evidence-id"],
                    "selected_hook": "exactly one string copied from hook_candidates",
                },
                "subjects": [{
                    "name": "exact person/company/vendor/product/paper name", "subject_type": "person|company|vendor|product|paper|team|tool|model",
                    "action": "what it concretely did", "consequence": "what concretely changes", "evidence_ids": ["evidence-id"],
                }],
                "context_events": [{
                    "id": "stable context event id, preferably reuse one from context_graph",
                    "actor": "named actor", "action": "past or parallel event", "occurred_at": "source date or unknown",
                    "relation": "how it changes the meaning of the current event", "evidence_ids": ["evidence-id"],
                }],
                "evidence_shots": [{
                    "id": "stable-shot-id",
                    "question": "INTERNAL director question; never rendered",
                    "fact": "one concise on-screen fact",
                    "interpretation": "INTERNAL editorial rationale; never rendered",
                    "audience_copy": "optional second on-screen sentence: a declarative fact, comparison, causal implication, or concrete impact; empty if evidence adds nothing beyond fact",
                    "evidence_ids": ["evidence-id"], "beat_ids": ["required beat id"],
                    "source_url": "an exact URL already present in candidate/evidence",
                    "target": "shortest self-contained exact source quote that proves the visible claim (normally 6–35 words), heading, code line, or PDF caption; never a two-word fragment; empty only for a complete tweet/image hold",
                    "translation": "natural technical Chinese beside the highlighted foreign text; never use 翻译/译为",
                    "relation_to_previous": "new causal step, contrast, evidence, consequence, or payoff; empty only for first shot",
                    "visual_family": "tweet|quoted_post|official_page|source_image|product_ui|chart|timeline|code|paper|quote_card|impact_card|stat_card",
                    "retention_job": "hook_proof|reveal|contrast|turn|impact|payoff",
                    "selection_reason_ids": ["selection reason id this shot advances"],
                    "context_event_ids": ["context event id this shot explains"],
                    "full_translation": "for a non-Chinese root post: 40–120 Chinese characters covering decisive actor/action/scope/numbers without handles/URL; empty otherwise",
                    "narrative_beat": "opening|proof|takeaway",
                }],
                "director_brief": {
                    "editorial_thesis": "the specific evidence-backed point of this story",
                    "viewer_tension": "the unanswered question that keeps the viewer watching",
                    "emotion": "surprise|excitement|relief|alarm|conflict|curiosity|opportunity",
                    "emotion_intensity": "1|2|3",
                    "attention_trigger": "named actor + concrete change + audience stake",
                    "selected_context_ids": ["context event id that materially changes interpretation"],
                    "story_arc": [{
                        "role": "event|background|proof|turn|impact|payoff", "claim": "one new fact or implication",
                        "why_here": "why this beat follows the previous one", "evidence_ids": ["evidence id"],
                        "selection_reason_ids": ["selection reason id"], "context_event_ids": ["context event id"],
                        "suggested_visual": "semantic visual family only",
                    }],
                    "recommended_duration": self.target_duration,
                },
            }
            if not radar:
                for name in (
                    "opening_mode", "category_label", "direct_identifier", "editorial_inference",
                ):
                    schema["editorial_brief"].pop(name, None)
                for shot in schema["editorial_brief"]["evidence_shots"]:
                    shot.pop("narrative_beat", None)
            github_contract = (
                "Non-GitHub editorial contract: return editorial_brief, never scenes, kind, material_role, visual_action, recording_cues, selectors, pointer tracks, zoom tracks, duration schedules, trial as a material, or boundary as a material. The execution layer compiles cited evidence plus visual_family into browser/render scenes and schedules flash timing deterministically. "
                + opening_contract +
                "Return exactly three materially different hook_candidates and select one verbatim. The chosen hook, headline, and subheadline must name the actor/product/company/paper and say what happened. Do not merely add emotional adjectives to a summary. The payoff must answer the opening with a distinct judgment. Optimize the selected hook aggressively for first-1.5-second retention: use the strongest source-backed lever available inside the chosen story—an exact number or contrast, a named consequential actor, concrete developer pain/ROI, or an honest open question. A neutral announcement label is not enough when the same evidence supports a sharper lever; never fabricate shock when it does not. The first EditorialOpportunity.selection_reason is the locked primary story promise. A hook may intensify curiosity, conflict, surprise, consequence, or audience relevance inside that promise, but must never promote a secondary capability, metric, or side character into a different story. Keep the primary actor/change visible in the selected hook, fixed conclusion, and final payoff. If the strongest evidence-backed hook is close to the descriptive headline, that is acceptable; semantic fidelity outranks artificial novelty. "
                + radar_copy_contract +
                "The audience includes vibe coders, not only AI researchers. When the hook or a visible metric uses a specialist term, its first evidence shot must immediately add a short plain-Chinese explanation of what the term measures or means in practice. For example, a refusal rate means the share of prompts the model declines to answer; explain the concept without assuming familiarity, while keeping the exact sourced number. Do not waste space defining common words such as API or model. "
                "Never shorten a title by deleting the exact person, company, project, model, or product name. The renderer owns title fitting and may reduce font size, add a line, and increase the fixed upper rail; preserve identity and meaning first. "
                "For short video, at least two hook candidates must exploit a real evidence-backed contradiction, surprise, consequence, or unresolved tension rather than a neutral topic label. Use direct emotional Chinese when the facts earn it. Phrases such as 引发讨论、值得关注、注意风险 or 生态竞争 cannot carry the hook or payoff by themselves. The fixed conclusion must give the viewer a memorable stance or consequence, not a ritual compliance reminder. "
                "For an X-rooted story, evidence_shots[0] must be tweet_card, show the complete original post in one shot, and cite the root post. Do not split its text into multiple cards. Then extend only when primary evidence adds a new causal step or impact. "
                "Primary-source images are first-class evidence. When evidence kind web:source_image or x:media_photo has editorial_priority=high and directly depicts a named team, product, architecture, benchmark, result, or decisive quoted-post context used by the story, cite it in the relevant evidence_shot and use visual_family=source_image. An image attached to the root or quoted X post is direct source evidence and should be shown when it explains the post's central object or conflict. A company/team story with an official founding-team photo should show the photo instead of replacing it with a generic text card. "
                "Acquisition metadata such as pixel dimensions, file size, cache path, MIME type, screenshot index, and evidence id is never audience content. Do not mention it in a headline, fact, audience_copy, hook, or conclusion unless the source event itself is explicitly about that technical property. "
                "If the root quotes an earlier post, frame that quoted post explicitly as 此前/先预告/随后落地 chronology. Start with the current root as required, then use the earlier quote to reveal the setup; never present it as a detached duplicate detail. "
                "For an official flash where the only verified facts are current rollout, an earlier quoted timetable, and affected scope, use exactly three shots: current root → earlier setup → scope/payoff. Do not create separate impact and effective-scope cards that repeat the same audience coverage. "
                "For flash, use 4–5 shots totaling no more than 15 seconds. For explainers/deep dives use only the minimum shots needed for causal clarity; 15–20 seconds is valid when the story is already complete. Adjacent shots must add new information; never repeat the same copy with a different presentation. No format may end its final changing shot on missing data, an unknown, waiting for more information, or generic caution; finish on the strongest verified capability, impact, or insight. Never invent the category of an unknown: do not mention a license, price, benchmark, deployment cost, funding amount, or limitation unless supplied evidence itself discusses that category. Source silence is not a visible story beat. For an expert technical-analysis post, prioritize its concrete observations, architecture changes, figures, and measured numbers. Preserve editorial salience: when the author presents an overall positive release with several innovations, a measured overhead attached to one component is one useful beat, not the headline, permanent rail, viewer tension, and payoff all at once. Conflict is not mandatory negativity; a record scale, counterintuitive architecture choice, or surprising capability can provide stronger tension. Do not replace source details with inferred deployment cost, generic adoption advice, or multiple scenes about absent data. An unknown may remain in the structured answers, but deserves a changing visual shot only when it materially changes the viewer's decision or the source explicitly makes it central. Context that was researched but placed in ContextGraph.discarded_context_ids is internal investigation only and must not appear in evidence_shots or the visible story. "
                "A flash must end on impact, payoff, effective scope, or a concrete action—not an unknown-information card, ritual caveat, or 值得关注. Do not spend any flash shot on missing mechanism/details; unknowns belong only in a longer explanation when the topic contract requires them. fixed_conclusion must be an assertive evidence-backed judgment, fit the persistent bottom rail in no more than 62 Chinese-character-equivalents, and must not end with 但/不过/然而/未知/尚未公开/有待验证/值得关注. "
                "Every browser target must be an exact contiguous substring from cited evidence. A translation contains only natural Chinese IT/AI meaning and appears beside that target; it never says 翻译 or 译为. Internal labels such as 证据带读、关键结论、trial、boundary must never enter visible copy."
                " Evidence-shot fields have a hard visibility boundary: question, interpretation, relation_to_previous, retention_job, and director_brief are internal production metadata and are never shown. Only fact, audience_copy, translation, and full_translation are audience-facing. audience_copy is optional and must be a complete declarative statement about the subject, evidence, comparison, causal implication, or concrete impact. Never put reading/viewing/editor instructions there: no ‘正确读法’, ‘解读时’, ‘不必被…吓退’, ‘值得跟进学习’, ‘降低预期’, or advice about how the editor or viewer should understand the material. If there is no additional supported audience information, return an empty audience_copy instead of filler."
                " A documentation or README page proves that a product/capability exists and how it works; it does not by itself prove a new launch, announcement, release date, or ‘正式上线’. Use 发布/上线/宣布 only when the cited source explicitly says released, launched, announced, introducing, or available today. Otherwise build the hook around the verified workflow change or capability."
                " A measurement, correlation, billing count, or evaluation result is evidence for a claim; it is not automatically the causal mechanism. For example, billed-token 1:1 matching does not mean the vulnerability is a billing vulnerability unless the source explicitly says so."
                " Investigate like an editor but express like a high-retention WeChat Channels video. The selection reason is the seed of the story: every final arc must explain why this item matters to this audience. Context is mandatory to investigate but optional to show; include only context that changes interpretation. ContextGraph.required_context_ids are not optional: put every one in director_brief.selected_context_ids and use them as chronological setup. For a people_change story, if pattern_context_ids contains a verified earlier move from the same incumbent, select at least one and show why the current departure is part of a sequence rather than an isolated headline. Do not call it a trend when that pattern evidence is absent. For a multi-party reply or quoted-post chain, preserve every bridge needed to understand why the root author reacted: show the initiating claim, the intervening response, and the final reaction in order. A safety or causality qualifier belongs next to the disputed claim; it must not replace the intervening response or become the whole takeaway. Use a concrete emotional mode grounded in evidence, never bureaucratic copy such as 标志着新阶段、重心转移的重要信号、值得关注、密切关注、关注后续说明. The emotional engine must be a named actor doing something consequential: relief for a familiar pain, a powerful team making a bet, a vendor/community conflict, a surprising capability, or a deadline/cost turning point. Do not replace it with vague curiosity."
                " For every duration, never render the same complete root tweet more than once: the first shot is the source card, while later implications use the attached source image, another primary source, or an evidence-backed quote_card/timeline/impact_card/stat_card. For flash, use 3–5 semantic visual changes, 1.3–2.8 seconds per shot except the complete root tweet may use 3.2 seconds, and at least three renderable visual families. tweet and quoted_post are complete archived X cards; official_page/source_image/product_ui/chart/code/paper are real captured sources; quote_card/timeline/impact_card/stat_card are evidence-backed program-rendered pacing cards. Use a derived card when the next beat is an implication, sequence, decisive quote, or payoff—not another paragraph from the same page. Never label two repeats of the same tweet as different visual families. Because the conclusion already remains visible in the fixed bottom rail, do not spend a separate final scene repeating the opening event merely to create a payoff card. The last changing shot must add the strongest audience payoff, impact, effective scope, or concrete next move."
                " Because an X-rooted video must show the complete root post first, the selected fixed hook should normally name the root author and their concrete action/reaction. Do not headline a later background event over an unrelated-looking root card; reveal that event in the next image/evidence shot instead."
                " A branded feature name plus words such as landed/available proves existence and timing, not its hidden mechanism. Never infer API coverage, quota accumulation, cost reduction, flexibility, or workflow efficiency unless cited evidence states that concept. Preserve quantity, duration, and recurrence: a one-time credit, reset, trial, exception, or temporary rollout is a finite benefit and must never become permanent freedom from recurring limits or costs. Use the verified rollout scope itself as the payoff when that is the strongest fact."
                " Preserve an unofficial lowercase product/feature identifier such as banked reset in its original form unless the evidence gives an official Chinese name. Preservation does not mean leaving it unexplained: at its first visible occurrence, state in natural Chinese exactly what the evidence says the feature gives or does. Never manufacture a dictionary-style Chinese product name such as 银行重置、银行级重置 or any other 银行…重置 variant. Never infer 无限制/没有限制 from a rollout time."
                " For a non-Chinese X root, the first tweet card keeps the complete original post together and provides a compact 40–120 Chinese-character translation covering the decisive actor, action, scope, and numbers; omit handles, URL, emoji, greetings, and ceremonial filler. It appears adjacent to the source, never as an internal label."
            )
        topic_contract = ""
        if self.topic_type == TopicType.COMPANY_OR_TEAM:
            topic_contract = (
                "Company/funding editorial contract: funding is evidence of a bet, not the whole story. "
                "Open on the concrete product or market reversal that makes the bet surprising, then use the round, "
                "valuation, users, revenue, team size, or economics only when each number is explicitly attributed in evidence. "
                "Show what the company actually does and why the named investors/customer behavior make that direction matter. "
                "When the source reports both traction and an economic tension, preserve both without turning the caveat into ritual negativity. "
                "For a news source, attribute company-supplied metrics and claims instead of presenting them as independently audited facts."
            )
        elif self.topic_type == TopicType.RESEARCH_OR_BENCHMARK:
            topic_contract = (
                "Research/evaluation editorial contract: make the research question visceral through a source-backed analogy, failure mode, "
                "or measurement tension, then reveal the method and what it changes. Distinguish a protocol, pilot, or proposed method from "
                "a completed comparative result. If the source claims first/world-first/novel, attribute that claim to the named source unless "
                "independent evidence establishes priority. For evaluation security, explain exactly who can and cannot see prompts, model weights, "
                "or outputs; never inflate confidentiality into proof that a model is safer or more capable. Prefer the real architecture diagram, "
                "benchmark table, or technical-report figure over generic AI imagery."
            )
        elif self.topic_type == TopicType.OFFICIAL_ANNOUNCEMENT:
            topic_contract = (
                "Official-news editorial contract: separate the announced event, its effective scope, and the audience action. "
                "Lead with the concrete change or deadline, preserve the official availability wording, and never convert a pilot into a rollout."
            )
        elif self.topic_type == TopicType.MODEL_OR_PRODUCT:
            topic_contract = (
                "Model/product editorial contract: the persistent title and selected hook must name the exact model or product, "
                "not only its publisher, host, benchmark, or release channel. Lead with what concretely changed and why it matters. "
                "Translate specialist metrics into one-line plain Chinese at their first visible evidence shot so a technically curious "
                "vibe coder can follow the stakes without prior research knowledge."
            )
        return "\n".join([
            "You are an evidence-bound Chinese technical-video editor for WeChat Channels.",
            "Use constructive optimism across posts, news, research, products, companies, tools, and repositories: exploration and concrete progress are valuable even before every unknown is resolved. Lead with what the work makes possible and why it is worth attention. Accuracy means separating verified facts, reasoned implications, and unknowns; it does not mean adding a ritual negative conclusion. Never turn missing evidence into claims that something is immature, prototype-only, unsuitable for production, unlikely to work, not worth adopting, or necessarily needs human checking. Risks, failures, manual-review requirements, and limitations require explicit source evidence; otherwise label the point unknown and keep the editorial judgment open.",
            "Write with a clear, brave point of view and a strong opening, but never invent facts, metrics, capabilities, users, demos, funding, market moves, or performance. Every event must say who did what and what changed: name the person/company/vendor/project instead of using an ownerless capability sentence. Do not use empty hype such as ‘颠覆’、‘改写’、‘爆了’ unless the evidence itself supports a concrete version of that claim.",
            "Any visible number, duration, price, percentage, multiplier, or before/after efficiency claim must appear in the supplied evidence. Never turn a qualitative automation claim into invented wording such as ‘from hours to minutes’.",
            "The fixed conclusion must be project-specific. Do not fall back to second-person moralizing such as ‘仍是你的责任’、‘替你决定’、‘最终还是要你’; state the verified capability, adoption cost, or limitation instead.",
            "For practice posts, explicitly keep the author claim distinct from our observation and its scope.",
            "When a post leads to an external source, begin with what a reader sees in the post, translate the decisive wording, then extend to what the primary page reveals and the evidence-backed implication. For X/post events, research may connect the named person or company to recent related events when primary evidence supports a real pattern (for example, one executive departure within a documented sequence of AI-talent departures); distinguish the current event from historical context and never turn one event into a trend by itself. GitHub is narrower: do not add free-ranging industry context. Use company/vendor background only when the README or a README-linked repository document explicitly names it and it directly explains the project's capability. Do not produce a detached webpage summary.",
            "Every answer and proof/explanation scene must cite evidence_ids. If evidence is missing, say unknown.",
            "For GitHub, return exactly four github_scenes using the dedicated schema above. For other topics, return the dedicated editorial_brief and never return generic scenes. Do not cut causal logic merely to fit the duration. Each evidence shot must make sense with sound off.",
            self.visual_policy,
            (
                "Radar V2 is opt-in for this job. Enforce the compact opening/proof/takeaway, factual-label, "
                "identifier, and visible-copy contracts above."
                if radar else
                "Classic profile is active. Do not add Radar-only metadata fields."
            ),
            github_contract,
            topic_contract,
            IT_TRANSLATION_CONTRACT,
            PLAIN_CHINESE_CONTRACT,
            self.visual_policy,
            "Editorial direction from the bounded research pass:\n" + (self.editorial_direction or "Use only the evidence to choose the strongest concrete angle."),
            "Editorial opportunity (why this was selected):\n" + json.dumps(asdict(self.opportunity), ensure_ascii=False) if self.opportunity else "Editorial opportunity: not provided; infer cautiously from evidence.",
            "Context graph (investigated background):\n" + json.dumps(asdict(self.context_graph), ensure_ascii=False) if self.context_graph else "Context graph: not provided; do not invent background.",
            f"Topic: {self.topic_type.value}; format: {self.content_type.value}; target duration: {self.target_duration}s.",
            "Required editorial questions:\n" + json.dumps([asdict(item) for item in requirements], ensure_ascii=False),
            "Candidate:\n" + json.dumps(asdict(self.candidate), ensure_ascii=False),
            "Evidence:\n" + json.dumps(compact_evidence, ensure_ascii=False),
            "Return JSON only, matching this schema:\n" + json.dumps(schema, ensure_ascii=False),
        ])
