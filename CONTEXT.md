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
A targeted reflection cycle that routes Critic feedback to a specific worker agent to fix groundedness or rubric failures.
_Avoid_: Retry, rerun, loop

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
