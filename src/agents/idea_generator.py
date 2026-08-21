"""
Idea Generator — clusters pain points into themes and generates distinct startup ideas.

REFACTORED: Following clean code principles with extracted helper functions.

Pipeline flow:
1. Receive filtered pain points from state
2. Cap to manageable number to fit context window
3. LLM brainstorms ideas grouped by themes
4. Validate that addresses_pain_point_ids reference real pain points
5. Return validated Idea objects
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import settings
from src.constants import IDEA_THEME_ANGLES, MAX_IDEAS_PER_RUN_DEFAULT, MAX_PAIN_POINTS_FOR_PROMPT
from src.exceptions import LLMError, LLMJSONParseError
from src.llm.client import extract_json, get_llm
from src.llm.prompts import get_prompt
from src.state.schema import Idea, PipelineStage, VentureForgeState

logger = logging.getLogger(__name__)


# =============================================================================
# PAIN POINT SELECTION
# =============================================================================


def select_pain_points_for_prompt(
    pain_points: List[Any],
    max_count: int = MAX_PAIN_POINTS_FOR_PROMPT,
) -> List[Any]:
    """
    Select pain points for idea generation prompt.

    Sorts by evidence count and caps to context window limit.

    Args:
        pain_points: List of pain points
        max_count: Maximum pain points to include

    Returns:
        Sorted and capped list of pain points
    """
    sorted_pps = sorted(
        pain_points,
        key=lambda pp: len(pp.evidence),
        reverse=True,
    )
    return sorted_pps[:max_count]


def serialize_pain_point_for_ideas(pain_point: Any) -> Dict[str, Any]:
    """
    Serialize pain point for idea generation prompt.

    Args:
        pain_point: Pain point object

    Returns:
        Serialized pain point dictionary
    """
    return {
        "id": str(pain_point.id),
        "title": pain_point.title,
        "description": pain_point.description,
        "evidence": [
            {
                "source_url": ev.source_url,
                "raw_quote": ev.raw_quote,
                "source": ev.source.value if hasattr(ev.source, "value") else str(ev.source),
            }
            for ev in pain_point.evidence
        ],
        "evidence_count": len(pain_point.evidence),
    }


# =============================================================================
# PROMPT BUILDING
# =============================================================================


def _build_system_prompt() -> str:
    """Build system prompt for idea generation."""
    return get_prompt("idea_generator")


def build_revision_block(revision_feedback: str | None) -> str:
    """
    Build revision instruction block if in revision mode.

    Args:
        revision_feedback: Feedback from critic

    Returns:
        Revision instruction text, or empty string
    """
    if not revision_feedback:
        return ""

    return (
        "THIS IS A REVISION ROUND. The critic flagged specific weaknesses. "
        "You MUST address ONLY the failing checks mentioned in the feedback. "
        "DO NOT change dimensions that were previously passing - preserve them exactly.\n"
        f"- Critic feedback: {revision_feedback}\n\n"
        "CRITICAL: Only fix the specific issues mentioned. "
        "If target_user was passing before, keep it unchanged. "
        "If competitive thesis was passing before, keep it unchanged. "
        "Make minimal, surgical changes to address only the failing checks.\n\n"
    )


def build_requirement_block(pain_point_count: int) -> str:
    """
    Build requirement block based on available pain points.

    Args:
        pain_point_count: Number of available pain points

    Returns:
        Requirement instruction text
    """
    min_refs = min(2, pain_point_count)

    if pain_point_count == 1:
        return (
            "**SPECIAL CASE: Only 1 pain point available.**\n"
            "Generate ideas that deeply address this single pain point. "
            "Each idea must reference this pain point UUID in 'addresses_pain_point_ids'. "
            "Focus on different solution angles, user segments, or implementation approaches for variety.\n\n"
        )
    elif pain_point_count >= 2:
        return (
            f"**CRITICAL REQUIREMENT: Each idea MUST reference AT LEAST {min_refs} pain point UUIDs in 'addresses_pain_point_ids'.**\n"
            f"Ideas with fewer than {min_refs} references will be REJECTED. "
            "Cross-pollinate pain points to create stronger, more defensible ideas that solve multiple problems.\n\n"
        )
    else:
        return "ERROR: No pain points provided. Cannot generate ideas.\n\n"


def _build_user_prompt(
    domain: str,
    pain_points: List[Dict[str, Any]],
    ideas_count: int,
    revision_feedback: str | None = None,
) -> str:
    """
    Build user prompt for idea generation.

    Args:
        domain: Domain to generate ideas for
        pain_points: Serialized pain points
        ideas_count: Number of ideas to generate
        revision_feedback: Optional feedback from critic

    Returns:
        Formatted user prompt
    """
    revision_block = build_revision_block(revision_feedback)
    requirement_block = build_requirement_block(len(pain_points))

    return (
        f"Domain: {domain}\n"
        f"Ideas to generate: {ideas_count}\n\n"
        f"PAIN POINTS ({len(pain_points)} provided):\n"
        f"{json.dumps(pain_points, indent=2)}\n\n"
        f"{revision_block}"
        f"{requirement_block}"
        "Only use UUIDs from the pain points list above — do not invent new UUIDs.\n\n"
        'Return JSON: {"ideas": [ ... ]}.'
    )


def _build_user_prompt_single(
    state: VentureForgeState,
    idea_number: int,
    total_ideas: int,
    theme_angle: str | None = None,
) -> str:
    """Build the user prompt for generating a SINGLE idea."""
    pps = state.filtered_pain_points or state.pain_points
    domain = state.domain
    feedback = state.revision_feedback or ""

    # Sort pain points by evidence count (descending) and cap to top N
    pps = sorted(pps, key=lambda pp: len(pp.evidence), reverse=True)[:MAX_PAIN_POINTS_FOR_PROMPT]

    # Serialize pain points with evidence (limit to top 2 evidence items per pain point)
    pp_blobs: List[Dict[str, Any]] = [
        {
            "id": str(pp.id),
            "title": pp.title,
            "description": pp.description,
            "evidence": [
                {
                    "source_url": ev.source_url,
                    "raw_quote": ev.raw_quote[:300],  # Truncate long quotes
                    "source": ev.source.value if hasattr(ev.source, "value") else str(ev.source),
                }
                for ev in pp.evidence[:2]  # Only top 2 evidence items
            ],
            "evidence_count": len(pp.evidence),
        }
        for pp in pps
    ]

    # Angle block if provided (initial generation diversity)
    angle_block = ""
    if theme_angle and not state.revision_feedback:
        angle_block = (
            f"**CREATIVE ANGLE / FOCUS FOR THIS IDEA:**\n"
            f"Explore solutions aligned with: {theme_angle}\n"
            f"Ground your concept in this perspective to ensure maximum market differentiation.\n\n"
        )

    # Revision block if applicable
    revision_block = ""
    if state.revision_feedback:
        prev_idea = (
            next((i for i in state.ideas if i.id == state.current_revision_idea_id), None)
            if state.current_revision_idea_id
            else None
        )
        prev_blob = ""
        if prev_idea:
            prev_idea_dict = {
                "title": prev_idea.title,
                "one_liner": prev_idea.one_liner,
                "problem": prev_idea.problem,
                "solution": prev_idea.solution,
                "target_user": prev_idea.target_user,
                "key_features": prev_idea.key_features,
                "addresses_pain_point_ids": [
                    str(pid) for pid in prev_idea.addresses_pain_point_ids
                ],
            }
            prev_blob = f"\nPREVIOUS IDEA TO REFINE:\n{json.dumps(prev_idea_dict, indent=2)}\n"

        revision_block = (
            "THIS IS A REVISION ROUND. The critic flagged weaknesses in positioning. "
            "You MUST address the following feedback:\n"
            f"- Critic feedback: {feedback}\n"
            f"{prev_blob}\n"
            "Make the target_user a specific, named, reachable community (a 'contained fire') "
            "and make the competition thesis explicit. Refine the existing concept rather than inventing an unrelated one.\n\n"
        )

    # Determine minimum pain point references
    min_refs = min(2, len(pps))

    # Build requirement block
    if len(pps) == 1:
        requirement_block = (
            "**SPECIAL CASE: Only 1 pain point available.**\n"
            "Generate an idea that deeply addresses this single pain point. "
            "The idea must reference this pain point UUID in 'addresses_pain_point_ids'.\n\n"
        )
    elif len(pps) >= 2:
        requirement_block = (
            f"**CRITICAL: The idea MUST reference AT LEAST {min_refs} pain point UUIDs in 'addresses_pain_point_ids'.**\n"
            f"Ideas with fewer than {min_refs} references will be REJECTED. "
            "Cross-pollinate pain points to create a stronger, more defensible idea.\n\n"
        )
    else:
        requirement_block = "ERROR: No pain points provided.\n\n"

    user_text = (
        f"Domain: {domain}\n"
        f"Generating idea {idea_number} of {total_ideas}\n\n"
        f"PAIN POINTS ({len(pps)} provided):\n"
        f"{json.dumps(pp_blobs, indent=2)}\n\n"
        f"{angle_block}"
        f"{revision_block}"
        f"{requirement_block}"
        "Only use UUIDs from the pain points list above — do not invent new UUIDs.\n\n"
        'Return a single JSON object (not an array): {"title": ..., "one_liner": ..., ...}'
    )
    return user_text


# =============================================================================
# LLM INTERACTION
# =============================================================================


def call_llm_for_ideas(
    domain: str,
    pain_points: List[Dict[str, Any]],
    ideas_count: int,
    revision_feedback: str | None = None,
) -> str:
    """
    Call LLM to generate ideas.

    Args:
        domain: Domain to generate ideas for
        pain_points: Serialized pain points
        ideas_count: Number of ideas to generate
        revision_feedback: Optional feedback from critic

    Returns:
        Raw LLM response content

    Raises:
        LLMError: If LLM invocation fails
    """
    llm = get_llm(temperature=0.7, max_tokens=16384, reasoning=False)

    system_prompt = _build_system_prompt()
    system_prompt += (
        "\n\n**CRITICAL: Output ONLY the JSON object. "
        "No markdown code fences, no explanations, no preamble. Start with { and end with }.**"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=_build_user_prompt(
                domain=domain,
                pain_points=pain_points,
                ideas_count=ideas_count,
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
        logger.error(f"[idea_generator] LLM invocation failed after {elapsed:.1f}s: {e}")
        raise LLMError(f"LLM invocation failed: {e}") from e

    elapsed = time.monotonic() - start_time
    logger.info(f"[idea_generator] LLM responded in {elapsed:.1f}s")

    # Debug: log response preview
    logger.info(f"[idea_generator] Response preview (first 500 chars): {content[:500]}")

    return content


def invoke_llm_single(
    state: VentureForgeState,
    idea_number: int,
    total_ideas: int,
    theme_angle: str | None = None,
    retry_count: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Invoke LLM to generate a SINGLE idea.

    Args:
        state: Current pipeline state
        idea_number: Which idea this is (1-indexed)
        total_ideas: Total number of ideas to generate
        theme_angle: Optional theme angle for parallel differentiation
        retry_count: Current retry attempt (0-indexed)

    Returns:
        Raw idea dict, or None on failure
    """
    llm = get_llm(temperature=0.7, max_tokens=16384, reasoning=False)

    system_prompt = _build_system_prompt()
    system_prompt += "\n\n**CRITICAL: Output ONLY a single JSON object. No markdown fences, no explanations. Start with { and end with }.**"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=_build_user_prompt_single(
                state, idea_number, total_ideas, theme_angle=theme_angle
            )
        ),
    ]

    start = time.monotonic()
    try:
        raw = llm.invoke(messages)
        content = raw.content if hasattr(raw, "content") else str(raw)
    except Exception as e:
        logger.error(
            f"[idea_generator] LLM invocation failed for idea {idea_number} (attempt {retry_count + 1}): {e}"
        )
        return None

    elapsed = time.monotonic() - start
    logger.info(
        f"[idea_generator] LLM responded in {elapsed:.1f}s for idea {idea_number} (attempt {retry_count + 1})"
    )

    # Warn if response looks truncated
    if content and not content.rstrip().endswith("}"):
        logger.warning(
            f"[idea_generator] Response may be truncated for idea {idea_number}. "
            f"Last 100 chars: {content[-100:]}"
        )

    parsed = extract_json(content)
    if parsed is None:
        logger.error(
            f"[idea_generator] JSON extraction failed for idea {idea_number} (attempt {retry_count + 1}). "
            f"Response length: {len(content)} chars"
        )
        logger.error(f"[idea_generator] Response preview: {content[:500]}")
        return None

    # Handle both dict and wrapped dict formats
    if isinstance(parsed, dict):
        if "ideas" in parsed and isinstance(parsed["ideas"], list):
            return parsed["ideas"][0] if parsed["ideas"] else None
        return parsed

    # Handle plain array format (backward compatibility with tests)
    if isinstance(parsed, list) and len(parsed) > 0:
        return parsed[0]

    return None


def generate_single_idea_with_retry(
    state: VentureForgeState,
    idea_number: int,
    total_ideas: int,
    theme_angle: str | None = None,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Generate a single idea with retry attempts."""
    for retry in range(max_retries):
        raw_idea = invoke_llm_single(
            state, idea_number, total_ideas, theme_angle=theme_angle, retry_count=retry
        )
        if raw_idea:
            logger.info(
                f"[idea_generator] Successfully generated idea {idea_number} on attempt {retry + 1}"
            )
            return raw_idea

        if retry < max_retries - 1:
            logger.warning(
                f"[idea_generator] Attempt {retry + 1}/{max_retries} failed for idea {idea_number}. Retrying..."
            )
        else:
            logger.error(
                f"[idea_generator] All {max_retries} attempts failed for idea {idea_number}."
            )

    return None


# =============================================================================
# RESPONSE PARSING
# =============================================================================


def parse_ideas_response(response_content: str) -> List[Dict[str, Any]]:
    """
    Parse LLM response to extract ideas.

    Args:
        response_content: Raw LLM response

    Returns:
        List of idea dictionaries

    Raises:
        LLMJSONParseError: If JSON parsing fails
    """
    parsed = extract_json(response_content)

    if parsed is None:
        logger.error(
            f"[idea_generator] JSON extraction failed. "
            f"Response length: {len(response_content)} chars"
        )
        raise LLMJSONParseError(
            raw_response=response_content[:2000],
            parse_error="extract_json returned None",
        )

    # Handle both flat array and {"ideas": [...]} wrapper
    if isinstance(parsed, dict) and "ideas" in parsed:
        raw_list = parsed["ideas"]
    elif isinstance(parsed, list):
        raw_list = parsed
    else:
        logger.warning("[idea_generator] LLM did not return a JSON array")
        return []

    if not isinstance(raw_list, list):
        logger.warning("[idea_generator] LLM did not return a JSON array")
        return []

    return raw_list


# =============================================================================
# VALIDATION
# =============================================================================


def validate_pain_point_references(
    idea_dict: Dict[str, Any],
    valid_pain_point_ids: set[UUID],
) -> bool:
    """
    Validate that idea references valid pain point IDs.

    Args:
        idea_dict: Idea dictionary
        valid_pain_point_ids: Set of valid pain point UUIDs

    Returns:
        True if at least one valid reference exists, False otherwise
    """
    addresses_ids = idea_dict.get("addresses_pain_point_ids", [])

    if not addresses_ids:
        logger.debug(
            f"[idea_generator] Idea '{idea_dict.get('title', 'unknown')}' "
            "has no pain point references"
        )
        return False

    # Convert to UUIDs and check validity
    referenced_ids: set[UUID] = set()
    for pid in addresses_ids:
        try:
            referenced_ids.add(UUID(str(pid)))
        except (ValueError, TypeError):
            continue

    valid_refs = referenced_ids & valid_pain_point_ids
    if not valid_refs:
        logger.debug(
            f"[idea_generator] Idea '{idea_dict.get('title', 'unknown')}' "
            f"references no valid pain point IDs: {referenced_ids}"
        )
        return False

    return True


def convert_to_idea(
    idea_dict: Dict[str, Any],
    valid_pain_point_ids: set[UUID],
    min_refs: int = 2,
) -> Optional[Idea]:
    """
    Convert raw idea dict to Idea object with validation.

    Args:
        idea_dict: Raw idea dictionary
        valid_pain_point_ids: Set of valid pain point UUIDs
        min_refs: Minimum number of pain point references required

    Returns:
        Idea object, or None if validation fails
    """
    # Validate pain point references
    if not validate_pain_point_references(idea_dict, valid_pain_point_ids):
        return None

    # Check minimum reference count
    addresses_ids = idea_dict.get("addresses_pain_point_ids", [])
    resolved: list[UUID] = []
    for pid in addresses_ids:
        try:
            uid = UUID(str(pid))
            if uid in valid_pain_point_ids and uid not in resolved:
                resolved.append(uid)
        except (ValueError, TypeError):
            continue

    if len(resolved) < min_refs:
        logger.debug(
            f"[idea_generator] REJECTED — idea '{idea_dict.get('title', '?')}' "
            f"references only {len(resolved)} valid pain point(s), need {min_refs}"
        )
        return None

    try:
        return Idea(
            id=uuid4(),
            title=idea_dict["title"],
            one_liner=idea_dict["one_liner"],
            problem=idea_dict["problem"],
            solution=idea_dict["solution"],
            target_user=idea_dict["target_user"],
            key_features=idea_dict.get("key_features", []),
            addresses_pain_point_ids=resolved,
        )

    except Exception as e:
        logger.debug(f"[idea_generator] REJECTED — malformed idea: {e}")
        return None


# =============================================================================
# MAIN AGENT FUNCTION
# =============================================================================


def run(state: VentureForgeState) -> Dict[str, Any]:
    """
    Main idea generator agent function.

    Args:
        state: Current pipeline state

    Returns:
        State patch dictionary
    """
    pps = state.filtered_pain_points
    if not pps:
        logger.warning("[idea_generator] no pain points available — returning empty")
        patch = {
            "ideas": [],
            "current_stage": PipelineStage.GENERATING,
            "next_node": "orchestrator",
            "idea_generation_attempts": state.idea_generation_attempts + 1,
        }
        patch.update(
            state.add_event(
                agent="idea_generator",
                stage=PipelineStage.GENERATING,
                kind="warning",
                message="No pain points available to generate ideas from.",
            )
        )
        return patch

    valid_ids = {pp.id for pp in pps}
    min_refs = min(2, len(pps))  # Adaptive: require 1 ref if only 1 pain point exists

    # Determine how many ideas to generate
    if state.current_revision_idea_id:
        count = 1
        logger.info(
            f"[idea_generator] Revision mode: generating 1 replacement idea for {state.current_revision_idea_id}"
        )
    else:
        count = state.ideas_per_run or MAX_IDEAS_PER_RUN_DEFAULT
        logger.info(f"[idea_generator] Initial generation: generating {count} ideas")

    MAX_RETRIES = 3
    raw_ideas: list[dict[str, Any]] = []

    if state.current_revision_idea_id or count <= 1:
        logger.info(f"[idea_generator] Generating {count} idea(s) sequentially")
        raw_idea = generate_single_idea_with_retry(
            state,
            1,
            1,
            max_retries=MAX_RETRIES,
        )
        if raw_idea:
            raw_ideas.append(raw_idea)
    else:
        max_workers = min(count, settings.llm_max_concurrency)
        logger.info(
            f"[idea_generator] Dispatching {count} idea generation tasks across {max_workers} concurrent threads"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_num = {
                executor.submit(
                    generate_single_idea_with_retry,
                    state,
                    i + 1,
                    count,
                    IDEA_THEME_ANGLES[i % len(IDEA_THEME_ANGLES)],
                    MAX_RETRIES,
                ): i + 1
                for i in range(count)
            }
            for future in concurrent.futures.as_completed(future_to_num):
                idea_num = future_to_num[future]
                try:
                    raw_idea = future.result()
                    if raw_idea:
                        raw_ideas.append(raw_idea)
                except Exception as exc:
                    logger.error(
                        f"[idea_generator] Worker thread failed for idea {idea_num}: {exc}"
                    )

    logger.info(f"[idea_generator] LLM produced {len(raw_ideas)} raw ideas")

    # DEBUG: Log first raw idea to diagnose validation failures
    if raw_ideas:
        logger.info(f"[idea_generator] Sample raw idea: {json.dumps(raw_ideas[0], indent=2)}")
        logger.info(
            f"[idea_generator] Valid pain point IDs: {[str(vid) for vid in list(valid_ids)[:3]]}"
        )

    validated: List[Idea] = []
    for raw in raw_ideas:
        idea = convert_to_idea(raw, valid_ids, min_refs)
        if idea:
            validated.append(idea)
            logger.debug(
                f"[idea_generator] Validated idea: {idea.title} with {len(idea.addresses_pain_point_ids)} pain point refs"
            )
        else:
            logger.debug(f"[idea_generator] Rejected idea: {raw.get('title', 'unknown')}")

    final = validated[:count]

    logger.info(
        f"[idea_generator] {len(final)}/{len(raw_ideas)} ideas validated "
        f"(required <= {count} with >={min_refs} real pain point refs each)"
    )

    # Merge with existing ideas (if not in revision mode for this specific idea)
    if state.current_revision_idea_id:
        # In revision mode: add the new idea(s) to existing ideas
        all_ideas = state.ideas + final
    else:
        # Initial generation: replace all ideas
        all_ideas = final

    patch = {
        "ideas": all_ideas,
        "idea_generation_attempts": state.idea_generation_attempts + 1,
        "current_revision_idea_id": None,  # Clear revision flag
        "next_node": "orchestrator",
    }
    patch.update(
        state.add_event(
            agent="idea_generator",
            stage=PipelineStage.GENERATING,
            kind="info",
            message=(
                f"Generated {len(final)} ideas (requested {count}) "
                f"addressing >={min_refs} validated pain points each."
            ),
        )
    )
    return patch
