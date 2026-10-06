# Handoff: YouTube subtitle terminology and pipeline spec

Date: 2026-10-06. Repository: `/Users/clairehou/pyProjects/video_factory`.

## Current user request

The user rejected a hard-coded `CONTEXTUAL_TERM_GUIDANCE` rule for `retrieval agent`: tomorrow's source could say `recommendation agent` or another context-sensitive term. Replace the one-off rule with a general, context-aware terminology workflow. Keep `docs/specs/youtube_pipeline.md` and code synchronized, as required by the newly created root `AGENTS.md`. The user then asked for a new-conversation handoff because this long thread feels slow. Do not assume the generic terminology change is complete.

## What is already implemented in the dirty worktree

- `src/video_factory/youtube.py`: interview subtitle fallback lets the translation LLM propose source word boundaries and Chinese text together for a locally rejected window; deterministic validation reconstructs all source words, checks order/timing/length/terminology, and an independent LLM reviewer can reject. Reviewer feedback and previous proposals feed bounded retries.
- The exact nested-term bug is partially fixed: per-card validation no longer demands `agent → 智能体` inside a longer declared `retrieval agent` phrase. The full-cue validator already had longer-term protection. Check the earlier fixed-card pre-review validation loop around `segment_interview_subtitle_cards`; it may still use `_contains_term` without longer-term masking.
- Current experimental code adds `CONTEXTUAL_CHINESE_TERMS[("retrieval agent", "retrieval agent")] = ("检索智能体", "检索代理")`, `CONTEXTUAL_TERM_GUIDANCE` with a hard-coded decision rule, `_accepted_terminology_targets`, and `_terminology_prompt_row`. Planner/coarse translation/fixed-card translation/joint repair receive some of these fields. This is precisely the one-off approach the user wants removed/replaced.
- `tests/test_youtube_collection.py` has joint-repair, rejected-draft diagnostics, nested-term, and contextual-option tests. Its 204 focused tests pass after the latest edits; the full suite has not been rerun since then.
- `docs/specs/youtube_pipeline.md` was added with the end-to-end pipeline and roles (LLM vs deterministic), known limitations, and terminology responsibilities. It currently describes the hard-coded context rule and must change when implementation changes.
- Root `AGENTS.md` was added requiring spec/code/test synchronization.
- The repository had many unrelated user changes before this task. Preserve them; do not reset or overwrite. No commit was made.

## Evidence from the live video

Video `OTQ-lFsq7zA`, title `Why AI Is Reinventing How Businesses Buy Everything`. Failed joint draft source: `Um and so you um you have like this retrieval agent / um getting like all of the information putting it into SAP.` The model wrote `检索代理` after being given that as an exact glossary target. The validator also rejected it for not containing `智能体` because it separately matched the nested word `agent`; that is a validator defect, not proof the LLM cannot understand context. In this AI-workflow passage, `检索智能体` is likely more natural, but the full audio/context has not been independently reviewed. Several live retries failed at other subtitle cards; no successful completed video from this change has been claimed.

## Recommended generic implementation

1. Make the planning LLM decide, from nearby source sentences and role/actions, whether a term is unambiguous or context-sensitive. For the latter, return a chosen Chinese rendering, a small list of acceptable alternatives when genuinely warranted, and a concise evidence/rationale note. Do not maintain per-phrase Python rules like `retrieval agent` or `recommendation agent`.
2. Extend `TerminologyEntry` (`src/video_factory/models.py`) and `_parse_terminology` to carry validated per-video alternatives/rationale. Constrain size/types and require source occurrence. Existing established translations and protected product/API/code names remain strict; ordinary context-sensitive vocabulary goes to contextual review.
3. Pass the per-video decision and evidence to each translation call and the *independent reviewer*. The reviewer should judge whether the chosen wording fits the actual source, not merely whether it is in a list. Rejections should feed the bounded repair loop; do not treat all alternatives as automatically publication-ready.
4. Deterministic validators should check source-to-card coverage, longer-phrase masking, no accidental English leakage, bounds/reading speed, and consistency with the selected per-video term decision. They cannot determine semantic meaning by exact substring matching. Avoid deterministic fallback replacing a contextual term with a planner target after the reviewer chose a natural alternative.
5. Update the pipeline spec with the implemented behavior and remaining limits; update tests using an unseen phrase such as `recommendation agent`, plus retrieval-agent regression, nesting, cache roundtrip, reviewer rejection and fallback cases. Run focused and full tests, `git diff --check`. Only then consider a bounded live retry if network/model access is available.

## Starting points

- `src/video_factory/youtube.py`: `CONTEXTUAL_CHINESE_TERMS` around line 296; `CONTEXTUAL_TERM_GUIDANCE` around 315; `terminology_contract_errors` around 1153; `_translated_term_present` and `_terminology_prompt_row` around 1240; `NaturalSubtitleTranslator.translate` around 2050; joint repair around 2680; fixed-card translation/review around 2900–3140; `_enforce_terminology_contract` around 3940; `_parse_terminology` around 4180; `YouTubeCollectionService.generate` around 8300.
- `src/video_factory/models.py`: `TerminologyEntry` around line 181.
- `tests/test_youtube_collection.py`: terminology and joint-repair cases around lines 2580–2780, plus many earlier term-contract tests.
- `docs/specs/youtube_pipeline.md`; root `AGENTS.md`.
- Test command: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q`. Last confirmed full suite before the most recent contextual-guidance edits: 688 tests passed. Focused `tests.test_youtube_collection` now passes 204 tests. `git diff --check` currently passes.

## Cautions

- This is a large dirty worktree. Run `git status --short` and inspect relevant diffs first.
- Do not claim `CONTEXTUAL_TERM_GUIDANCE` is a general solution; user explicitly objected.
- The current model prompt still contains exact-target wording in some paths. Search every terminology prompt and validator before concluding the architecture is generic.
- `targeted_whisper_caption_audit` currently runs after initial translation/segmentation once media is local; severe ASR errors can fail earlier. This is documented as a limitation, not solved by the terminology change.
