"""Pain Point Miner — extracts structured pain points from multiple sources.


Pipeline flow
=============
1. Scrape from multiple sources with fallback: HackerNews → ProductHunt → Tavily Web → Reddit (optional)
2. Reddit is optional and requires API approval; system works without it using other sources
3. LLM extracts structured pain points from the corpus of comments
4. Code validates that each raw_quote is an actual substring
5. Return only pain points where all rubric checks pass
"""
from __future__ import annotations

import json
import logging
import time
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import settings
from src.llm.client import coerce_rubric_bools, coerce_yes_no, extract_json, get_llm
from src.llm.prompts import get_prompt
from src.state.schema import DataSource, PainPoint, PainPointRubric, PipelineStage, VentureForgeState
from src.tools.reddit_scraper import (
    COMMUNITY_MAP,
    ScrapedComment,
    scrape_for_domain as reddit_scrape_for_domain,
    validate_quote,
)
from src.tools.tavily_fallback import search_communities
from src.tools.hackernews_scraper import scrape_for_domain as hn_scrape_for_domain
from src.tools.producthunt_scraper import scrape_for_domain as ph_scrape_for_domain
from src.tools.tavily_content_scraper import scrape_for_domain as tavily_content_scrape_for_domain

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

    # Debug: log what we received
    logger.info(f"[pain_point_miner] Received {len(comments)} items of type: {[type(c).__name__ for c in comments[:5]]}")

    # Defensive: ensure all items are ScrapedComment objects
    # Convert dicts to ScrapedComment if needed (fallback from broken scrapers)
    normalized_comments: list[ScrapedComment] = []
    for i, c in enumerate(comments):
        if isinstance(c, ScrapedComment):
            normalized_comments.append(c)
        elif isinstance(c, dict):
            # Try to convert dict to ScrapedComment
            try:
                normalized_comments.append(
                    ScrapedComment(
                        text=c.get("text", ""),
                        url=c.get("url", ""),
                        subreddit=c.get("subreddit", "unknown"),
                        post_title=c.get("post_title", ""),
                    )
                )
            except Exception as e:
                logger.warning(f"[pain_point_miner] Skipping malformed comment dict at index {i}: {e}")
                continue
        else:
            logger.warning(f"[pain_point_miner] Skipping non-ScrapedComment item at index {i}: type={type(c).__name__}, value={str(c)[:100]}")
            continue

    logger.info(f"[pain_point_miner] Normalized to {len(normalized_comments)} valid ScrapedComment objects")

    # Serialize comments compactly
    comment_blobs: list[dict] = [
        {
            "text": c.text[:800],  # truncate to keep token count sane
            "url": c.url,
            "subreddit": c.subreddit,
            "post_title": c.post_title[:120],
        }
        for c in normalized_comments
    ]

    payload: dict = {
        "domain": state.domain,
        "max_pain_points": max_pp,
        "revision_feedback": feedback,
        "comments": comment_blobs,
    }

    user_text = (
        f"Extract up to {max_pp} pain points from the {len(normalized_comments)} comments below.\n"
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
    llm = get_llm(temperature=0.2, max_tokens=4096, reasoning=False)
    
    # Add explicit JSON-only instruction
    system_prompt = _build_system_prompt()
    system_prompt += "\n\n**CRITICAL: Output ONLY the JSON array. No markdown code fences, no explanations, no preamble. Start with [ and end with ].**"
    
    messages = [
        SystemMessage(content=system_prompt),
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

    # Debug: log first 500 chars of response
    logger.info(f"[pain_point_miner] Response preview (first 500 chars): {content[:500]}")
    
    parsed = extract_json(content)
    if parsed is None:
        logger.error(f"[pain_point_miner] JSON extraction failed. Response length: {len(content)} chars")
        logger.error(f"[pain_point_miner] Full response (first 2000 chars): {content[:2000]}")
        return []

    # Handle both flat array and {"pain_points": [...]} wrapper
    if isinstance(parsed, dict) and "pain_points" in parsed:
        raw_list = parsed["pain_points"]
    elif isinstance(parsed, list):
        raw_list = parsed
    else:
        raw_list = []

    if not isinstance(raw_list, list):
        logger.warning("[pain_point_miner] LLM did not return a JSON array")
        return []

    pain_points: list[PainPoint] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        try:
            # Parse evidence array
            evidence_list = item.get("evidence", [])
            if not evidence_list:
                # Backward compatibility: if no evidence array, try old format
                evidence_list = [{
                    "source_url": item.get("source_url", ""),
                    "raw_quote": item.get("raw_quote", ""),
                    "source": item.get("source", "hackernews"),
                }]
            
            from src.state.schema import PainPointEvidence
            evidence_objects = []
            for ev in evidence_list:
                if isinstance(ev, dict):
                    evidence_objects.append(PainPointEvidence(
                        source_url=ev["source_url"],
                        raw_quote=ev["raw_quote"],
                        source=DataSource(ev.get("source", "hackernews")),
                    ))
            
            if not evidence_objects:
                logger.debug(f"[pain_point_miner] skipping pain point with no valid evidence: {item.get('title', 'unknown')}")
                continue
            
            pp = PainPoint(
                id=item.get("id") or uuid4(),
                title=item["title"],
                description=item["description"],
                rubric=PainPointRubric(**coerce_rubric_bools(item["rubric"])),
                passes_rubric=coerce_yes_no(item["passes_rubric"]),
                evidence=evidence_objects,
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
    """Code-level validation: ALL raw_quotes in evidence must exist verbatim.

    Also enforces that descriptions are non-trivial and de-duplicates
    near-identical pain points.
    """
    validated: list[PainPoint] = []
    for pp in pain_points:
        # Validate ALL evidence items
        all_valid = True
        for ev in pp.evidence:
            found, matched_url = validate_quote(ev.raw_quote, comments)
            if not found:
                logger.debug(
                    f"[pain_point_miner] REJECTED — quote not found verbatim: {ev.raw_quote[:60]}..."
                )
                all_valid = False
                break
            
            # Update the evidence item's source_url if it was generic
            if ev.source_url != matched_url:
                ev.source_url = matched_url
            
            # Determine source based on the matching comment
            matched_comment = next((c for c in comments if c.url == matched_url), None)
            if matched_comment:
                if matched_comment.subreddit == "hackernews":
                    ev.source = DataSource.HACKERNEWS
                elif matched_comment.subreddit == "producthunt":
                    ev.source = DataSource.PRODUCTHUNT
                elif matched_comment.subreddit in ("web", "stackoverflow", "github", "devto", "indiehackers", "lobsters"):
                    ev.source = DataSource.WEB
                else:
                    ev.source = DataSource.REDDIT
        
        if not all_valid:
            continue

        # Force has_verbatim_quote to True (all quotes were validated)
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

        if not pp.passes_rubric:
            logger.debug(f"[pain_point_miner] REJECTED — rubric failed: {pp.title}")
            continue

        # Additional quality filter: drop extremely short / vague descriptions
        if len(pp.description.strip()) < 40:
            logger.debug(
                f"[pain_point_miner] REJECTED — description too short/vague: {pp.description!r}"
            )
            continue

        validated.append(pp)

    # De-duplicate by (primary source_url, normalized description)
    deduped: list[PainPoint] = []
    seen_keys: set[tuple[str, str]] = set()
    for pp in validated:
        key = (
            pp.source_url,  # Uses @property which returns first evidence URL
            " ".join(pp.description.lower().split()),
        )
        if key in seen_keys:
            logger.debug(f"[pain_point_miner] DEDUPED — {pp.title!r} / {pp.source_url}")
            continue
        seen_keys.add(key)
        deduped.append(pp)

    return deduped


def _determine_primary_source(comments: list[ScrapedComment]) -> DataSource:
    """Determine which source provided the most comments."""
    reddit_count = sum(
        1
        for c in comments
        if c.subreddit != "hackernews"
        and c.subreddit != "producthunt"
        and c.subreddit not in ("web", "stackoverflow", "github", "devto", "indiehackers", "lobsters")
    )
    hn_count = sum(1 for c in comments if c.subreddit == "hackernews")
    ph_count = sum(1 for c in comments if c.subreddit == "producthunt")
    web_count = sum(
        1
        for c in comments
        if c.subreddit in ("web", "stackoverflow", "github", "devto", "indiehackers", "lobsters")
    )

    # Find the source with the most comments
    source_counts = {
        DataSource.REDDIT: reddit_count,
        DataSource.HACKERNEWS: hn_count,
        DataSource.PRODUCTHUNT: ph_count,
        DataSource.WEB: web_count,
    }

    return max(source_counts, key=source_counts.get)


def _tavily_enriched_scrape(domain: str, threshold: int) -> tuple[list[ScrapedComment], DataSource]:
    """Scrape with multi-source fallback: Hacker News → Product Hunt → Tavily web content → Reddit (optional).

    Reddit is now optional since it requires API approval. Other sources work without API keys.

    Returns (comments, primary_source) where primary_source indicates which
    scraper provided the bulk of the data.
    """
    all_comments: list[ScrapedComment] = []
    source_counts: dict[DataSource, int] = {
        DataSource.HACKERNEWS: 0,
        DataSource.PRODUCTHUNT: 0,
        DataSource.WEB: 0,
        DataSource.REDDIT: 0,
    }

    # --- Attempt 1: Hacker News (no API key required) ---
    logger.info("[pain_point_miner] Starting with Hacker News (no API key required)")
    try:
        hn_comments = hn_scrape_for_domain(domain, max_total_comments=_MAX_TOTAL_COMMENTS)
        if hn_comments:
            logger.info(f"[pain_point_miner] Hacker News returned {len(hn_comments)} comments")
            all_comments.extend(hn_comments)
            source_counts[DataSource.HACKERNEWS] = len(hn_comments)
    except Exception as e:
        logger.warning(f"[pain_point_miner] Hacker News scraper failed: {e}")

    if len(all_comments) >= threshold:
        primary = max(source_counts, key=source_counts.get)
        return all_comments, primary

    # --- Attempt 2: Product Hunt (API key optional, has fallback) ---
    logger.info(
        f"[pain_point_miner] Have {len(all_comments)} comments, "
        f"trying Product Hunt"
    )
    try:
        ph_comments = ph_scrape_for_domain(domain, max_total_comments=_MAX_TOTAL_COMMENTS)
        if ph_comments:
            logger.info(f"[pain_point_miner] Product Hunt returned {len(ph_comments)} comments")
            all_comments.extend(ph_comments)
            source_counts[DataSource.PRODUCTHUNT] = len(ph_comments)
    except Exception as e:
        logger.warning(f"[pain_point_miner] Product Hunt scraper failed: {e}")

    if len(all_comments) >= threshold:
        primary = max(source_counts, key=source_counts.get)
        return all_comments, primary

    # --- Attempt 3: Tavily web content search (requires TAVILY_API_KEY) ---
    logger.info(
        f"[pain_point_miner] Have {len(all_comments)} comments, "
        f"trying Tavily web content search"
    )
    try:
        web_comments = tavily_content_scrape_for_domain(domain, max_total_comments=_MAX_TOTAL_COMMENTS)
        if web_comments:
            logger.info(f"[pain_point_miner] Tavily web search returned {len(web_comments)} content chunks")
            all_comments.extend(web_comments)
            source_counts[DataSource.WEB] = len(web_comments)
    except Exception as e:
        logger.warning(f"[pain_point_miner] Tavily web content scraper failed: {e}")

    if len(all_comments) >= threshold:
        primary = max(source_counts, key=source_counts.get)
        return all_comments, primary

    # --- Attempt 4: Reddit (optional, requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET) ---
    # Only try Reddit if we have credentials and still need more comments
    from src.config import settings

    if settings.reddit_client_id and settings.reddit_client_secret:
        logger.info(
            f"[pain_point_miner] Have {len(all_comments)} comments, "
            f"trying Reddit (optional, requires API credentials)"
        )
        try:
            reddit_comments = reddit_scrape_for_domain(domain, max_total_comments=_MAX_TOTAL_COMMENTS)
            if reddit_comments:
                logger.info(f"[pain_point_miner] Reddit returned {len(reddit_comments)} comments")
                all_comments.extend(reddit_comments)
                source_counts[DataSource.REDDIT] = len(reddit_comments)
        except Exception as e:
            logger.warning(f"[pain_point_miner] Reddit scraper failed: {e}")
    else:
        logger.info("[pain_point_miner] Skipping Reddit - no API credentials configured")

    # Determine primary source
    if not all_comments:
        return [], DataSource.HACKERNEWS  # Default to HN as primary if empty

    primary = _determine_primary_source(all_comments)
    return all_comments, primary


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
    comments, primary_source = _tavily_enriched_scrape(domain, threshold)
    logger.info(
        f"[pain_point_miner] domain='{domain}' → {len(comments)} comments "
        f"from all sources (threshold={threshold}, primary={primary_source})"
    )

    if not comments:
        logger.warning("[pain_point_miner] zero comments scraped — returning empty")
        patch = {
            "pain_points": [],
            "current_stage": PipelineStage.MINING,
            "next_node": "orchestrator",
        }
        patch.update(
            state.add_event(
                agent="pain_point_miner",
                stage=PipelineStage.MINING,
                kind="warning",
                message=f"No Reddit comments scraped for domain '{domain}'.",
            )
        )
        return patch

    # --- Step 1: LLM extraction ---
    extracted = _llm_extract_pain_points(state, comments)
    logger.info(f"[pain_point_miner] LLM extracted {len(extracted)} raw pain points")

    # --- Step 2: Code validation (exact quotes + quality filters) ---
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

    patch = {
        "pain_points": final,
        "next_node": "orchestrator",
    }
    patch.update(
        state.add_event(
            agent="pain_point_miner",
            stage=PipelineStage.MINING,
            kind="info",
            message=(
                f"Scraped {len(comments)} Reddit comments "+
                f"→ {len(final)} validated pain points for domain '{domain}'."
            ),
        )
    )
    return patch
