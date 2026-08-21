# 14. Engagement-Ranked Ingestion and Hybrid Evidence Augmentation

Date: 2026-08-21

## Status

Accepted

## Context

Prior to this decision:
1. Data source harvesting across concurrent providers in `CompositeDataMiner` appended evidence in raw thread completion arrival order, truncating evidence to the first 20 items. This resulted in pure First-Come-First-Served (FCFS) selection where high-signal posts with hundreds of upvotes and comments were dropped in favor of whichever low-engagement comment arrived first.
2. In obscure market domains or when live APIs returned sparse evidence (< 8 comments), the miner did not trigger synthetic fallback (which previously required 0 items), starving the LLM of sufficient data density to discover multi-source clusters.
3. Reddit scraping was blocked by unconfigured OAuth credentials without clean, silent provider skipping.

## Decision

1. **Composite Engagement Ranking**: Extract `points` (upvotes) and `num_comments` from Algolia Hacker News and other active providers, computing a composite score:
   $$\text{score} = (\text{points} \times 1.0) + (\text{num\_comments} \times 2.0)$$
   Sort all ingested evidence by composite score descending before constructing the prompt context.
2. **Hybrid Evidence Augmentation ($N < 8$)**: If live providers return fewer than 8 valid evidence items within the SLA window, automatically augment the corpus with high-signal domain-grounded synthetic items to provide sufficient evidence density for clustering.
3. **Multi-Evidence Clustering Prompt & 45-Item Context**: Expand the prompt window from 20 to 45 comments, aligning prompt instructions with `PROMPTS.md` to mandate `evidence: [...]` array outputs with 2+ verified quotes per pain point.
4. **Resilient Per-Quote Code Validation**: Verify each quote in a pain point's `evidence` list against the scraped corpus. Discard ungrounded secondary quotes while preserving pain points that have at least one verified quote.
5. **Graceful Provider Degradation**: Cleanly omit unconfigured providers (such as Reddit when keys are absent) via `is_available()` and log informative mining summaries.

## Consequences

- The Pain Point Miner prioritizes high-demand, community-validated frustrations over noise.
- Obscure domains run seamlessly without stalling or generating hallucinated pain points.
- Multi-evidence clustering provides richer, better-grounded inputs to downstream idea generators and pitch writers.
