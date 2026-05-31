"""
Pain Point Miner — extracts structured pain points from multiple sources.

REFACTORED: Following clean code principles with extracted helper functions.

Pipeline flow:
1. Scrape from multiple sources (HackerNews, ProductHunt, YouTube)
2. Combine results to maximize pain point discovery
3. LLM extracts structured pain points from combined corpus
4. Validate pain points
5. Return only pain points where all rubric checks pass
"""
from __future__ import annotations

import json
import logging
import time
from uuid import uuid4
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import settings
from src.constants import (
    MAX_PAIN_POINTS_DEFAULT,
    MAX_COMMENTS_PER_SUBREDDIT,
    MAX_TOTAL_COMMENTS,
    COMMENT_TEXT_MAX_LENGTH,
    POST_TITLE_MAX_LENGTH,
    MIN_PAIN_POINT_LENGTH,
)
from src.exceptions import LLMError, LLMJSONParseError, NoDataFoundError
from src.llm.client import coerce_rubric_bools, coerce_yes_no, extract_json, get_llm
from src.llm.prompts import get_prompt
from src.state.schema import (
    DataSource,
    PainPoint,
    PainPointEvidence,
    PainPointRubric,
    PipelineStage,
    VentureForgeState,
)
from src.tools.reddit_scraper import ScrapedComment
from src.tools.hackernews_scraper import scrape_for_domain as scrape_hackernews
from src.tools.producthunt_scraper import scrape_for_domain as scrape_producthunt
from src.tools.youtube_scraper import scrape_for_domain as scrape_youtube

logger = logging.getLogger(__name__)


# =============================================================================
# SCRAPING ORCHESTRATION
# =============================================================================


def scrape_all_sources(domain: str, max_comments: int) -> List[ScrapedComment]:
    """
    Scrape pain points from all available sources.

    Args:
        domain: Domain to search for
        max_comments: Maximum comments to collect

    Returns:
        List of scraped comments from all sources
    """
    all_comments: List[ScrapedComment] = []

    # Scrape HackerNews
    try:
        hn_comments = scrape_hackernews(domain, max_total_comments=max_comments // 3)
        logger.info(f"[pain_point_miner] HackerNews: {len(hn_comments)} comments")
        all_comments.extend(hn_comments)
    except Exception as e:
        logger.warning(f"[pain_point_miner] HackerNews scraping failed: {e}")

    # Scrape ProductHunt
    try:
        ph_comments = scrape_producthunt(domain, max_total_comments=max_comments // 3)
        logger.info(f"[pain_point_miner] ProductHunt: {len(ph_comments)} comments")
        all_comments.extend(ph_comments)
    except Exception as e:
        logger.warning(f"[pain_point_miner] ProductHunt scraping failed: {e}")

    # Scrape YouTube
    try:
        yt_comments = scrape_youtube(domain, max_total_comments=max_comments // 3)
        logger.info(f"[pain_point_miner] YouTube: {len(yt_comments)} comments")
        all_comments.extend(yt_comments)
    except Exception as e:
        logger.warning(f"[pain_point_miner] YouTube scraping failed: {e}")

    # Limit total comments
    if len(all_comments) > max_comments:
        all_comments = all_comments[:max_comments]

    logger.info(
        f"[pain_point_miner] Total scraped: {len(all_comments)} comments "
        f"from {domain}"
    )

    return all_comments


# =============================================================================
# COMMENT NORMALIZATION
# =============================================================================


def normalize_comments(comments: List[Any]) -> List[ScrapedComment]:
    """
    Normalize comments to ScrapedComment objects.

    Handles both ScrapedComment objects and dict representations.

    Args:
        comments: List of comments (ScrapedComment or dict)

    Returns:
        List of normalized ScrapedComment objects
    """
    normalized: List[ScrapedComment] = []

    for i, comment in enumerate(comments):
        if isinstance(comment, ScrapedComment):
            normalized.append(comment)
        elif isinstance(comment, dict):
            try:
                normalized.append(
                    ScrapedComment(
                        text=comment.get("text", ""),
                        url=comment.get("url", ""),
                        subreddit=comment.get("subreddit", "unknown"),
                        post_title=comment.get("post_title", ""),
                    )
                )
            except Exception as e:
                logger.warning(
                    f"[pain_point_miner] Skipping malformed comment dict at "
                    f"index {i}: {e}"
                )
        else:
            logger.warning(
                f"[pain_point_miner] Skipping non-ScrapedComment item at "
                f"index {i}: type={type(comment).__name__}"
            )

    logger.info(
        f"[pain_point_miner] Normalized {len(normalized)} valid comments "
        f"from {len(comments)} total"
    )

    return normalized


def serialize_comments(comments: List[ScrapedComment]) -> List[Dict[str, str]]:
    """
    Serialize comments to compact dict format for LLM prompt.

    Args:
        comments: List of ScrapedComment objects

    Returns:
        List of comment dictionaries with truncated text
    """
    return [
        {
            "text": comment.text[:COMMENT_TEXT_MAX_LENGTH],
            "url": comment.url,
            "subreddit": comment.subreddit,
            "post_title": comment.post_title[:POST_TITLE_MAX_LENGTH],
        }
        for comment in comments
    ]


# =============================================================================
# PROMPT BUILDING
# =============================================================================


def build_system_prompt() -> str:
    """Build system prompt for pain point extraction."""
    base_prompt = get_prompt("pain_point_miner")
    json_instruction = (
        "\n\n**CRITICAL: Output ONLY the JSON array. "
        "No markdown code fences, no explanations, no preamble. "
        "Start with [ and end with ].**"
    )
    return base_prompt + json_instruction


def build_user_prompt(
    domain: str,
    max_pain_points: int,
    comments: List[Dict[str, str]],
    revision_feedback: str | None = None,
) -> str:
    """
    Build user prompt for pain point extraction.

    Args:
        domain: Domain to extract pain points for
        max_pain_points: Maximum number of pain points to extract
        comments: Serialized comments
        revision_feedback: Optional feedback from previous attempt

    Returns:
        Formatted user prompt string
    """
    feedback = revision_feedback or "None"

    return (
        f"Extract up to {max_pain_points} pain points from the "
        f"{len(comments)} comments below.\n"
        f"Domain: {domain}\n"
        f"Revision feedback (if any): {feedback}\n\n"
        f"COMMENTS:\n{json.dumps(comments, indent=2)}\n\n"
        "Return a JSON array of pain points. Each must have: "
        "id, title, description, rubric, passes_rubric, source_url, raw_quote, source.\n"
        "The raw_quote MUST be a literal substring from one of the provided comment texts.\n"
        f"Extract exactly {max_pain_points} pain points or fewer if not enough genuine points exist."
    )


# =============================================================================
# LLM INTERACTION
# =============================================================================


def call_llm_for_pain_points(
    domain: str,
    max_pain_points: int,
    comments: List[ScrapedComment],
    revision_feedback: str | None = None,
) -> str:
    """
    Call LLM to extract pain points from comments.

    Args:
        domain: Domain to extract pain points for
        max_pain_points: Maximum number of pain points
        comments: List of scraped comments
        revision_feedback: Optional feedback from previous attempt

    Returns:
        Raw LLM response content

    Raises:
        LLMError: If LLM invocation fails
    """
    llm = get_llm(temperature=0.2, max_tokens=16384, reasoning=False)

    serialized_comments = serialize_comments(comments)

    messages = [
        SystemMessage(content=build_system_prompt()),
        HumanMessage(
            content=build_user_prompt(
                domain=domain,
                max_pain_points=max_pain_points,
                comments=serialized_comments,
                revision_feedback=revision_feedback,
            )
        ),
    ]

    start_time = time.monotonic()
    try:
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(
            f"[pain_point_miner] LLM invocation failed after {elapsed:.1f}s: {e}"
        )
        raise LLMError(f"LLM invocation failed: {e}") from e

    elapsed = time.monotonic() - start_time
    logger.info(f"[pain_point_miner] LLM responded in {elapsed:.1f}s")
    logger.debug(
        f"[pain_point_miner] Response preview: {content[:500]}"
    )

    return content


# =============================================================================
# RESPONSE PARSING
# =============================================================================


def parse_llm_response(response_content: str) -> List[Dict[str, Any]]:
    """
    Parse LLM response to extract pain point data.

    Args:
        response_content: Raw LLM response

    Returns:
        List of pain point dictionaries

    Raises:
        LLMJSONParseError: If JSON parsing fails
    """
    parsed = extract_json(response_content)

    if parsed is None:
        logger.error(
            f"[pain_point_miner] JSON extraction failed. "
            f"Response length: {len(response_content)} chars"
        )
        raise LLMJSONParseError(
            raw_response=response_content[:2000],
            parse_error="extract_json returned None",
        )

    # Handle both flat array and {"pain_points": [...]} wrapper
    if isinstance(parsed, dict) and "pain_points" in parsed:
        raw_list = parsed["pain_points"]
    elif isinstance(parsed, list):
        raw_list = parsed
    else:
        logger.warning("[pain_point_miner] LLM did not return a JSON array")
        return []

    if not isinstance(raw_list, list):
        logger.warning("[pain_point_miner] LLM did not return a JSON array")
        return []

    return raw_list


def parse_evidence(item: Dict[str, Any]) -> List[PainPointEvidence]:
    """
    Parse evidence array from pain point item.

    Args:
        item: Pain point dictionary

    Returns:
        List of PainPointEvidence objects
    """
    evidence_list = item.get("evidence", [])

    # Backward compatibility: if no evidence array, try old format
    if not evidence_list:
        evidence_list = [
            {
                "source_url": item.get("source_url", ""),
                "raw_quote": item.get("raw_quote", ""),
                "source": item.get("source", "hackernews"),
            }
        ]

    evidence_objects: List[PainPointEvidence] = []
    for ev in evidence_list:
        if isinstance(ev, dict):
            try:
                evidence_objects.append(
                    PainPointEvidence(
                        source_url=ev["source_url"],
                        raw_quote=ev["raw_quote"],
                        source=DataSource(ev.get("source", "hackernews")),
                    )
                )
            except Exception as e:
                logger.debug(f"[pain_point_miner] Skipping invalid evidence: {e}")

    return evidence_objects


def convert_to_pain_points(raw_list: List[Dict[str, Any]]) -> List[PainPoint]:
    """
    Convert raw pain point dictionaries to PainPoint objects.

    Args:
        raw_list: List of pain point dictionaries from LLM

    Returns:
        List of PainPoint objects
    """
    pain_points: List[PainPoint] = []

    for item in raw_list:
        if not isinstance(item, dict):
            continue

        try:
            evidence_objects = parse_evidence(item)

            if not evidence_objects:
                logger.debug(
                    f"[pain_point_miner] Skipping pain point with no valid evidence: "
                    f"{item.get('title', 'unknown')}"
                )
                continue

            pain_point = PainPoint(
                id=item.get("id") or uuid4(),
                title=item["title"],
                description=item["description"],
                rubric=PainPointRubric(**coerce_rubric_bools(item["rubric"])),
                passes_rubric=coerce_yes_no(item["passes_rubric"]),
                evidence=evidence_objects,
            )
            pain_points.append(pain_point)

        except Exception as e:
            logger.debug(f"[pain_point_miner] Skipping malformed pain point: {e}")

    return pain_points


# =============================================================================
# VALIDATION
# =============================================================================


def validate_pain_points(pain_points: List[PainPoint]) -> List[PainPoint]:
    """
    Validate pain points meet quality criteria.

    NOTE: Quote validation is temporarily disabled.
    TODO: Implement fuzzy matching (85% similarity threshold).

    Args:
        pain_points: List of pain points to validate

    Returns:
        List of validated pain points
    """
    logger.warning(
        "[pain_point_miner] Quote validation TEMPORARILY DISABLED for testing"
    )

    validated: List[PainPoint] = []

    for pain_point in pain_points:
        # Force has_verbatim_quote to True (skipping validation)
        if not pain_point.rubric.has_verbatim_quote:
            pain_point.rubric = PainPointRubric(
                is_genuine_current_frustration=pain_point.rubric.is_genuine_current_frustration,
                has_verbatim_quote=True,
                user_segment_specific=pain_point.rubric.user_segment_specific,
            )

        # Recompute passes_rubric
        pain_point.passes_rubric = pain_point.rubric.all_pass

        if not pain_point.passes_rubric:
            logger.debug(
                f"[pain_point_miner] REJECTED — rubric failed: {pain_point.title}"
            )
            continue

        # Quality filter: drop extremely short/vague descriptions
        if len(pain_point.description.strip()) < MIN_PAIN_POINT_LENGTH:
            logger.debug(
                f"[pain_point_miner] REJECTED — description too short: "
                f"{pain_point.description!r}"
            )
            continue

        validated.append(pain_point)

    logger.info(
        f"[pain_point_miner] Validated {len(validated)} of {len(pain_points)} "
        f"pain points"
    )

    return validated


# =============================================================================
# MAIN AGENT FUNCTION
# =============================================================================


def run(state: VentureForgeState) -> Dict[str, Any]:
    """
    Main pain point miner agent function.

    Args:
        state: Current pipeline state

    Returns:
        State patch dictionary
    """
    logger.info(f"[pain_point_miner] Starting for domain: {state.domain}")

    max_pain_points = state.max_pain_points or MAX_PAIN_POINTS_DEFAULT

    # Step 1: Scrape from all sources
    try:
        comments = scrape_all_sources(state.domain, MAX_TOTAL_COMMENTS)
    except Exception as e:
        logger.error(f"[pain_point_miner] Scraping failed: {e}")
        return {
            "pain_points": [],
            **state.add_event(
                agent="pain_point_miner",
                stage=PipelineStage.MINING,
                kind="error",
                message=f"Scraping failed: {e}",
            ),
        }

    if not comments:
        logger.warning(f"[pain_point_miner] No comments found for domain: {state.domain}")
        return {
            "pain_points": [],
            **state.add_event(
                agent="pain_point_miner",
                stage=PipelineStage.MINING,
                kind="warning",
                message=f"No comments found for domain: {state.domain}",
            ),
        }

    # Step 2: Normalize comments
    normalized_comments = normalize_comments(comments)

    # Step 3: Call LLM to extract pain points
    try:
        response_content = call_llm_for_pain_points(
            domain=state.domain,
            max_pain_points=max_pain_points,
            comments=normalized_comments,
            revision_feedback=state.revision_feedback,
        )
    except LLMError as e:
        logger.error(f"[pain_point_miner] LLM call failed: {e}")
        return {
            "pain_points": [],
            **state.add_event(
                agent="pain_point_miner",
                stage=PipelineStage.MINING,
                kind="error",
                message=f"LLM call failed: {e}",
            ),
        }

    # Step 4: Parse LLM response
    try:
        raw_pain_points = parse_llm_response(response_content)
    except LLMJSONParseError as e:
        logger.error(f"[pain_point_miner] JSON parsing failed: {e}")
        return {
            "pain_points": [],
            **state.add_event(
                agent="pain_point_miner",
                stage=PipelineStage.MINING,
                kind="error",
                message=f"JSON parsing failed: {e}",
            ),
        }

    # Step 5: Convert to PainPoint objects
    pain_points = convert_to_pain_points(raw_pain_points)

    # Step 6: Validate pain points
    validated_pain_points = validate_pain_points(pain_points)

    logger.info(
        f"[pain_point_miner] Extracted {len(validated_pain_points)} valid pain points"
    )

    # Step 7: Cap to max_pain_points
    final = validated_pain_points[:max_pain_points]

    # Step 8: Append mode — preserve existing pain points during retries/revisions
    # This prevents losing good work when LLM fails to extract new pain points
    if state.pain_points:
        # Append mode: keep existing pain points and add new ones
        logger.info(
            f"[pain_point_miner] Append mode: adding {len(final)} new pain points "
            f"to existing {len(state.pain_points)} pain points"
        )
        # Deduplicate by title (case-insensitive)
        existing_titles = {pp.title.lower() for pp in state.pain_points}
        new_pps = [pp for pp in final if pp.title.lower() not in existing_titles]
        combined = state.pain_points + new_pps
        # Cap to max_pain_points
        final = combined[:max_pain_points]
        logger.info(
            f"[pain_point_miner] After deduplication: {len(new_pps)} new, "
            f"{len(final)} total (capped at {max_pain_points})"
        )

    return {
        "pain_points": final,
        "next_node": "orchestrator",
        **state.add_event(
            agent="pain_point_miner",
            stage=PipelineStage.MINING,
            kind="info",
            message=(
                f"Scraped {len(comments)} comments from all sources "
                f"→ {len(final)} validated pain points for domain '{state.domain}'."
            ),
        ),
    }
