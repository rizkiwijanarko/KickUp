# 16. Identity-Preserving Revision and Computed Critique Routing

Date: 2026-08-21

## Status

Accepted

## Context

A live run (`output_adhd.json`, domain "ADHD", DeepSeek V4 Flash) exposed a structural flaw in the revision loop:

1. The Critic's rubric allowed routing a failing pitch brief to `idea_generator`, and the Pydantic validator actively forced it for positioning/scorer failures. `reset_for_revision("idea_generator")` then **deleted** the failing Idea, its Scored Idea, and its Pitch Brief, and `idea_generator` produced a **fresh Idea with a new UUID**.
2. The run ended with the surviving pitch brief referencing idea `51f278a0` while both critiques referenced orphaned IDs (`10c87b62`, `6d8d8083`). The revised pitches were silently discarded; `revision_counts` tracked the dead IDs; the pipeline never completed (`is_complete: false`, `approved_pitches: []`).
3. The Critic's `revision_feedback` demanded new mining ("Mine additional pain points focusing on willingness to pay...") — un-completable in one revision and outside the pitch writer's power.
4. The pitch invented market statistics ("$4.5B neuroinclusive workplace tools market", "120k remote ADHD developers") with no evidence link, which the Critic rightly failed — but the pitch writer had no instruction forbidding invented numbers, and no research agent exists to supply real ones.

The glossary promised "In-Place Revision Refinement" (identity-preserving), but the code performed replacement. The code and the prompt's `target_agent` priority table also fought each other: the LLM chose a target, the validator overrode it.

## Decision

1. **Revision is identity-preserving**. A Revision routes feedback to a worker that surgically modifies the *existing* artifact, keeping its ID. Idea generation in revision mode passes `existing_id=state.current_revision_idea_id` to `convert_to_idea`, so the Idea keeps its UUID and downstream Scored Idea / Pitch Brief / revision counts stay coherent. Fresh-idea generation with new UUIDs is now the distinct concept **Ideation Round** (initial generation, or "all ideas parked" recovery).
2. **`target_agent` is computed, not LLM-chosen**. The Critic model validator derives it deterministically from the rubric:
   - `idea_generator` — only when `target_is_contained_fire` and/or `competition_embraced_with_thesis` fail with no evidence/claims failures (the only idea-level dimension: `target_user`).
   - `pitch_writer` — every other failing check (`all_claims_evidence_backed`, `no_hallucinated_source_urls`, `minimum_evidence_sources`, `tagline_under_12_words`, `scorer_verdict_justified`, `validation_plan_complete`).
   - `pain_point_miner` — never from pitch critique; mining shortfalls are handled by the pre-idea retry loop in `routing.py`.
3. **No-invented-statistics constraint** in the pitch writer prompt: factual claims must trace to evidence URLs; unsupported numbers are framed qualitatively or marked as unverified estimates, until a research agent exists.
4. **`revision_feedback` must be single-revision completable** by the routed worker; it never demands new mining or new evidence from the pitch writer.

## Consequences

- The revision loop terminates: a failing pitch is reworked in place and re-critiqued under the same `idea_id`, producing approved or quarantined pitches instead of a stuck "revising" state.
- `revision_counts`, Approved/Quarantined ledgers, and the Critic's URL cross-referencing are coherent because artifact IDs are stable.
- Pitch briefs stop inventing market statistics; the Critic's `all_claims_evidence_backed` becomes satisfiable with existing evidence.
- The LLM no longer influences routing; the prompt documents the computed rule instead of a competing priority table.
