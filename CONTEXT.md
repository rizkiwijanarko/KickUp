# VentureForge Domain

VentureForge is an autonomous multi-agent pipeline that discovers market pain points, synthesizes startup ideas, evaluates them against binary rubrics, generates pitches, and validates claims against grounded evidence.

## Language

**Domain**:
The specific industry, vertical, or market niche targeted for discovery (e.g., "developer tooling", "telehealth").
_Avoid_: Topic, category, niche

**Pain Point**:
A specific, current user frustration extracted from primary sources, carrying verbatim quotes and verified URLs.
_Avoid_: Problem, bug, issue, complaint

**Idea**:
A proposed startup solution designed to resolve one or more identified pain points.
_Avoid_: Concept, proposal, project

**Scored Idea**:
An idea evaluated against strict binary feasibility, demand, and defensibility rubrics.
_Avoid_: Ranked idea, validated idea, graded idea

**Pitch Brief**:
A structured executive summary detailing the problem statement, proposed solution, business model, and competitive moat.
_Avoid_: Pitch deck, one-pager, startup spec

**Critique**:
A factual grounding assessment from the Critic verifying that every claim and pain point traces to a genuine source URL.
_Avoid_: Code review, evaluation, feedback

**Revision**:
An identity-preserving refinement cycle that routes Critic feedback to a specific worker agent, which surgically modifies the failing dimensions of the existing artifact while keeping its ID (and passing sections) intact.
_Avoid_: Retry, rerun, loop, replacement

**Ideation Round**:
A fresh-idea generation pass producing new ideas with new IDs (distinct from a Revision, which preserves identity). Used for initial generation and for "all ideas parked" recovery, never for refining a failed idea.
_Avoid_: Reroll, idea revision, regeneration

**Evidence-Backed Claims**:
The requirement that every factual claim in a Pitch Brief (market size, user count, behavior prevalence) trace to a source URL in the mined evidence. Without a research agent, numeric statistics must not be invented; claims are framed qualitatively from the evidence or marked as unverified estimates.
_Avoid_: Invented statistics, unsourced market figures, fake numbers

**Contained Fire**:
The target-market criterion requiring the Pitch Brief's `target_user` to be a specific, named, reachable community (e.g. "r/ADHD_Programmers"), where a founder could identify 50 members by name within a week. A demographic without a named community fails this check.
_Avoid_: Broad demographic, generic segment

**Data Miner**:
The unified ingestion subsystem responsible for searching and extracting grounded evidence across external sources.
_Avoid_: Scraper, crawler, harvester

**Approved Pitch**:
A Pitch Brief that has passed 100% of the binary Critic rubric checks and is certified ready for investor decks.
_Avoid_: Validated pitch, winning pitch, final pitch

**Quarantined Pitch**:
A Pitch Brief that reached max revisions without satisfying all Critic rubric checks, preserved separately with attached flaw diagnostics for manual review.
_Avoid_: Failed pitch, rejected brief, broken pitch

**Ingestion SLA Budget**:
The maximum bounded time window (default: 5.0 seconds) allocated for concurrent source providers to return evidence before mining synthesis proceeds.
_Avoid_: Scrape timeout, network wait limit

**Downstream Invalidation Cascade**:
The protocol ensuring that reflection feedback routed to an upstream worker agent (e.g. Pain Point Miner) automatically invalidates downstream artifacts (Ideas, Scored Ideas, Pitch Briefs) to prevent stale state propagation.
_Avoid_: Hard reset, wipeout, pipeline flush

**Pursue-First Filtering**:
The scoring constraint requiring pitch briefs to be authored only for ideas receiving a 'pursue' verdict (no fatal flaws and passing demand/feasibility rubrics), triggering an idea generator reflection if 0 ideas qualify.
_Avoid_: Top scoring filter, greedy selection

**Engagement-Ranked Evidence**:
Raw evidence sorted by composite engagement metrics (upvotes, comment volume, domain keyword density) before prompt ingestion.
_Avoid_: Top comments, popular posts

**Per-Source Evidence Cap**:
The serialization window is capped per source (8 items) before the global 15-item limit so one high-volume provider cannot crowd out diverse sources.
_Avoid_: Global-only truncation, source monopolies

**Live-Evidence-Only Policy**:
The pipeline only ingests grounded evidence from real sources; sparse or unavailable live data surfaces as a shortfall rather than being padded with fabricated items.
_Avoid_: Synthetic fallback, fake data mode, dummy mode, offline stub

**Multi-Evidence Clustering**:
The synthesis pattern of grouping multiple distinct verbatim quotes across sources into a single validated Pain Point with 2+ grounded evidence references.
_Avoid_: Grouping, complaint bundling

**In-Place Revision Refinement**:
The reflection protocol where worker agents receive their previous generated output alongside Critic feedback and failing checks to surgically modify flawed dimensions while preserving passing sections.
_Avoid_: Blind regeneration, full rewrite, clean-slate retry

**Approval-Driven Critique Dispatch**:
The routing mechanism that iterates through pitch briefs based on their unapproved status in the Revision Ledger rather than sequential integer array indexing, preventing redundant re-evaluations.
_Avoid_: Sequential critique index, array index iteration

**Evidence Cache**:
The SQLite-backed persistence layer for raw mined posts and comments with time-to-live (TTL) expiration to eliminate redundant scraper network calls across pipeline runs.
_Avoid_: Temporary memory cache, scraper cache, local dump

