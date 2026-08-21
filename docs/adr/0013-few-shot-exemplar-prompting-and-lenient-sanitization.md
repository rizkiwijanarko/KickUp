# 13. Few-Shot Exemplar Prompting and Lenient Pre-Sanitization

Date: 2026-08-21

## Status

Accepted

## Context

Strict negative constraints in prompts (e.g. "DO NOT exceed 12 words", "NEVER use markdown") and rigid Pydantic field validation caused avoidable runtime failures and retries when LLMs returned slightly longer taglines (e.g. 13 words) or wrapped responses in markdown fences (` ```json `).

## Decision

1. **Few-Shot Exemplars**: Provide positive, concrete few-shot examples in system prompts for all agents (`pain_point_miner`, `idea_generator`, `scorer`, `pitch_writer`, `critic`) demonstrating ideal concise outputs, structures, and tone.
2. **Lenient Pre-Sanitization**: Convert strict validation errors into auto-sanitizing Pydantic `field_validator(mode="before")` (e.g. auto-trimming taglines to 12 words, stripping markdown fences, and coercing UUIDs).
3. **Resilient Multi-Strategy JSON Parser**: Clean trailing commas, strip fences, and handle structural boundaries in `extract_json`.

## Consequences

- LLMs generate consistent, well-structured outputs on the first attempt by mimicking positive exemplars.
- Edge cases in output length or formatting are silently sanitized without triggering costly retry cascades.
