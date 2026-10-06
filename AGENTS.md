# Repository agent instructions

- Treat `docs/specs/youtube_pipeline.md` as the maintained description of the implemented YouTube processing pipeline. Before changing discovery handoff, editorial selection, terminology, translation, subtitle segmentation, reviewer/repair loops, media acquisition, rendering, or validation, read the affected spec section.
- Keep spec and code in sync in the same change. Update the relevant spec row or limitation whenever behavior, order, thresholds, LLM responsibility, deterministic gates, or failure/retry behavior changes. Do not document proposed behavior as already implemented; label remaining gaps explicitly.
- Add or update focused tests for behavior changes, especially terminology overlap/context alternatives, exact English source coverage, timing limits, and independent review. Verify the spec against the actual implementation before reporting completion.
