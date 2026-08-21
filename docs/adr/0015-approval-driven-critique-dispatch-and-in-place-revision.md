# 15. Approval-Driven Critique Dispatch and In-Place Revision Refinement

Date: 2026-08-21

## Status

Accepted

## Context

During multi-pitch runs (`top_n_pitches >= 2`), two interrelated orchestration flaws degraded pipeline efficiency:
1. When a pitch brief was flagged for revision by the Critic, `pitch_writer` returned only the single revised brief, inadvertently overwriting the `state.pitch_briefs` array and dropping all other generated briefs.
2. `reset_for_revision` reset `current_critique_index = 0`, forcing the Critic to re-evaluate already approved briefs from the start of the list.
3. Revision prompts in `pitch_writer` and `idea_generator` asked for minimal changes but never supplied the worker agent with its previous output, forcing the LLM to hallucinate or regenerate the entire brief from scratch rather than making surgical edits.

## Decision

1. **In-Place Pitch Brief Merging**: `pitch_writer` merges newly revised briefs into existing briefs matching on `idea_id`, preserving other active briefs.
2. **Approval-Driven Critique Dispatch**: Replace sequential integer array indexing (`current_critique_index`) with dynamic ledger queries (`state.revisions.approved_idea_ids` and `state.revisions.quarantined_idea_ids`). The Critic strictly evaluates unapproved, non-quarantined briefs.
3. **In-Place Revision Prompts**: Supply the worker agents (`pitch_writer`, `idea_generator`) with their previous generation alongside the Critic's failing rubric checks and feedback, instructing the LLM to retain passing sections verbatim and patch only the failing dimensions.
4. **SQLite Cross-Run Evidence Cache**: Add a lightweight TTL cache in `.cache/ventureforge.db` for raw mined evidence to prevent redundant external API/scraper calls across repeated runs in the same domain.

## Consequences

- Multi-pitch discovery runs maintain full artifact integrity without brief dropouts.
- Redundant re-critiques of approved briefs are completely eliminated.
- Revisions are truly surgical, preserving passed positioning and copy.
- Domain mining stage latency drops from ~5s to <10ms on cached runs.
