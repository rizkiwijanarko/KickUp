# 12. Bounded Revision Loop and Best-Effort Graduation

Date: 2026-08-21

## Status

Accepted

## Context

The discovery pipeline was previously executing up to 2-3 revision loops whenever ideas were parked by Scorer or quarantined by Critic. With sequential OpenRouter LLM calls taking 30-60s each, runs frequently took 6-10 minutes and risked stalling on difficult domains.

## Decision

1. Cap pipeline revisions to `MAX_REVISION_LOOPS = 1` by default.
2. If all ideas remain parked after the single revision round, the Orchestrator will graduate the highest-scoring candidate to `pitch_writer` under a quarantined flag (`quarantined_pitches`) instead of aborting the pipeline with a failure.
3. Every execution is guaranteed to terminate deterministically with completed output artifacts.

## Consequences

- Pipeline runs finish reliably in under 2 minutes.
- Users always receive an end-to-end discovery output (with clear quarantine labels if rubric thresholds were not met).
- Downstream UI and CLI callers never experience indefinite hanging or silent pipeline aborts.
