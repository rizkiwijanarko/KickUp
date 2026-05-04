"""Pain Point Miner — extracts structured pain points from Reddit using
public JSON endpoints, with Tavily fallback for community discovery.

Pipeline flow
=============
1. Resolve domain → category → ordered subreddit list.
2. Scrape subreddits sequentially (niche → general), early-stop when enough
   comments collected.
3. If comment count < threshold, trigger Tavily to find extra subreddits.
4. LLM extracts structured pain points from the corpus of comments.
5. Code validates that each raw_quote is an actual substring.
6. Return only pain points where all rubric checks pass.
"""
from __future__ import annotations

import json
import logging
import time
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import settings
from src.llm.client import get_llm
from src.llm.prompts import get_prompt
from src.state.schema import DataSource, PainPoint, PainPointRubric, PipelineStage, VentureForgeState
from src.tools.reddit_scraper import (
    COMMUNITY_MAP,
    ScrapedComment,
    scrape_for_domain,
    validate_quote,
)
from src.tools.tavily_fallback import search_communities

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Thresholds (derived from state.max_pain_points)
# ------------------------------------------------------------------
_TAVILY_FALLBACK_RATIO: float = 0.5
_MAX_COMMENTS_PER_SUBREDDIT: int = 50
_MAX_TOTAL_COMMENTS: int = 200
_MAX_PAIN_POINTS_DEFAULT: int = 30


def _build_system_prompt() -> str:
    return get_prompt("pain_point_miner")


def _build_user_prompt(
    state: VentureForgeState,
    comments: list[ScrapedComment],
) -> str:
    max_pp = state.max_pain_points or _MAX_PAIN_POINTS_DEFAULT
    feedback = state.revision_feedback or "None"

    # Serialize comments compactly
    comment_blobs: list[dict] = [
        {
            "text": c.text[:800],  # truncate to keep token count sane
            "url": c.url,
            "subreddit": c.subreddit,
            "post_title": c.post_title[:120],
        }
        for c in comments
    ]

    payload: dict = {
        "domain": state.domain,
        "max_pain_points": max_pp,
        "revision_feedback": feedback,
        "comments": comment_blobs,
    }

    user_text = (
        f"Extract up to {max_pp} pain points from the {len(comments)} Reddit comments below.\n"
        f"Domain: {state.domain}\n"
        f"Revision feedback (if any): {feedback}\n\n"
        f"COMMENTS:\n{json.dumps(comment_blobs, indent=2)}\n\n"
        "Return a JSON array of pain points. Each must have: "
        "id, title, description, rubric, passes_rubric, source_url, raw_quote, source.\n"
        "The raw_quote MUST be a literal substring from one of the provided comment texts.\n"
        f"Extract exactly {max_pp} pain points or fewer if not enough genuine points exist."
    )
    return user_text


def _llm_extract_pain_points(
    state: VentureForgeState,
    comments: list[ScrapedComment],
) -> list[PainPoint]:
    """Call the LLM and parse structured pain points."""
    llm = get_llm(temperature=0.2, max_tokens=4096)
    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=_build_user_prompt(state, comments)),
    ]

    start = time.monotonic()
    try:
        raw = llm.invoke(messages)
        content = raw.content if hasattr(raw, "content") else str(raw)
    except Exception as e:
        logger.error(f"[pain_point_miner] LLM invocation failed after {time.monotonic()-start:.1f}s: {e}")
        return []

    logger.info(f"[pain_point_miner] LLM responded in {time.monotonic()-start:.1f}s")

    # Strip markdown code fences if present
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        raw_list = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"[pain_point_miner] JSON parse error: {e}")
        return []

    if not isinstance(raw_list, list):
        logger.warning("[pain_point_miner] LLM did not return a JSON array")
        return []

    pain_points: list[PainPoint] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        try:
            pp = PainPoint(
                id=item.get("id") or uuid4(),
                title=item["title"],
                description=item["description"],
                rubric=PainPointRubric(**item["rubric"]),
                passes_rubric=item["passes_rubric"],
                source_url=item["source_url"],
                raw_quote=item["raw_quote"],
                source=DataSource.REDDIT,
            )
            pain_points.append(pp)
        except Exception as e:
            logger.debug(f"[pain_point_miner] skipping malformed pain point: {e}")
            continue

    return pain_points


def _validate_pain_points(
    pain_points: list[PainPoint],
    comments: list[ScrapedComment],
) -> list[PainPoint]:
    """Code-level validation: raw_quote must exist verbatim PLUS URL must match."""
    valid: list[PainPoint] = []
    for pp in pain_points:
        found, matched_url = validate_quote(pp.raw_quote, comments)
        if not found:
            logger.debug(
                f"[pain_point_miner] REJECTED — quote not found verbatim: {pp.raw_quote[:60]}..."
            )
            continue

        # Overwrite source_url with the actual URL from the matching comment
        pp.source_url = matched_url
        pp.source = DataSource.REDDIT

        # Force has_verbatim_quote to True
        if not pp.rubric.has_verbatim_quote:
            pp.rubric = PainPointRubric(
                is_genuine_current_frustration=pp.rubric.is_genuine_current_frustration,
                has_verbatim_quote=True,
                user_segment_specific=pp.rubric.user_segment_specific,
            )

        # Recompute passes_rubric
        pp.passes_rubric = (
            pp.rubric.is_genuine_current_frustration
            and pp.rubric.has_verbatim_quote
            and pp.rubric.user_segment_specific
        )

        if pp.passes_rubric:
            valid.append(pp)
        else:
            logger.debug(f"[pain_point_miner] REJECTED — rubric failed: {pp.title}")

    return valid


def _tavily_enriched_scrape(domain: str, threshold: int) -> list[ScrapedComment]:
    """Scrape static subreddits, then trigger Tavily if under threshold."""
    comments = scrape_for_domain(domain, max_total_comments=_MAX_TOTAL_COMMENTS)

    if len(comments) >= threshold:
        return comments

    logger.info(
        f"[pain_point_miner] collected {len(comments)} comments "
        f"(< threshold {threshold}), triggering Tavily fallback"
    )

    extra_subs = search_communities(domain)
    if not extra_subs:
        return comments

    from src.tools.reddit_scraper import scrape_subreddit

    for sr in extra_subs:
        if len(comments) >= threshold:
            break
        batch = scrape_subreddit(sr, cap=threshold - len(comments))
        comments.extend(batch)
        logger.info(
            f"[pain_point_miner] r/{sr}: +{len(batch)} from Tavily fallback "
            f"(running total {len(comments)})"
        )

    return comments


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def run(state: VentureForgeState) -> dict:
    """Entry point called by the LangGraph orchestrator."""
    domain = state.domain
    max_pp = state.max_pain_points or _MAX_PAIN_POINTS_DEFAULT
    threshold = int(max_pp * _TAVILY_FALLBACK_RATIO)

    # If revision feedback exists, we may already have comments cached —
    # but today we re-scrape every run (deterministic, simple).
    comments = _tavily_enriched_scrape(domain, threshold)
    logger.info(
        f"[pain_point_miner] domain='{domain}' → {len(comments)} comments "
        f"from all sources (threshold={threshold})"
    )

    if not comments:
        logger.warning("[pain_point_miner] zero comments scraped — returning empty")
        return {
            "pain_points": [],
            "current_stage": PipelineStage.MINING,
            "next_node": "orchestrator",
        }

    # --- Step 1: LLM extraction ---
    extracted = _llm_extract_pain_points(state, comments)
    logger.info(f"[pain_point_miner] LLM extracted {len(extracted)} raw pain points")

    # --- Step 2: Code validation (exact quotes) ---
    validated = _validate_pain_points(extracted, comments)
    logger.info(f"[pain_point_miner] {len(validated)}/{len(extracted)} passed code validation")

    # --- Step 3: Cap to max_pain_points ---
    final = validated[:max_pp]

    # --- Step 4: Revision support — if revision_feedback, prioritize flagged fixes ---
    if state.revision_feedback:
        # Keep only ones that explicitly address the feedback
        # ( simplistic: if feedback is in desc or title )
        feedback_lower = state.revision_feedback.lower()
        addressed = [pp for pp in final if feedback_lower in (pp.title + pp.description).lower()]
        if addressed:
            final = addressed[:max_pp]

    return {
        "pain_points": final,
        "current_stage": PipelineStage.MINING,
        "next_node": "orchestrator",
    }
