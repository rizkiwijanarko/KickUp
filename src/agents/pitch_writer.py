"""
Pitch Writer — writes investor-ready one-page pitch briefs for top ideas.

REFACTORED: Following clean code principles with extracted helper functions.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, Any, List, Optional
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage

from src.constants import MAX_PITCH_GENERATION_ATTEMPTS
from src.exceptions import LLMError, LLMJSONParseError
from src.llm.client import get_llm, get_structured_llm
from src.llm.prompts import get_prompt
from src.state.schema import (
    CompetitiveLandscape,
    PipelineStage,
    PitchBrief,
    ValidationPlan,
    VentureForgeState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT BUILDING
# =============================================================================


def build_system_prompt() -> str:
    """
    Build complete system prompt.

    Returns:
        Complete system prompt
    """
    base_prompt = get_prompt("pitch_writer")
    # No JSON instruction needed - structured output handles this
    return base_prompt


def get_target_ideas(state: VentureForgeState) -> List[Any]:
    """
    Get the list of ideas to write briefs for.

    If in revision mode, returns only the idea being revised.
    Otherwise, returns all top scored ideas.

    Args:
        state: Current pipeline state

    Returns:
        List of scored ideas to write briefs for
    """
    if state.current_revision_idea_id:
        target_scored = next(
            (
                s
                for s in state.scored_ideas
                if s.idea_id == state.current_revision_idea_id
            ),
            None,
        )
        if target_scored:
            return [target_scored]

    return state.top_scored_ideas


def serialize_scored_idea(scored_idea: Any, idea: Any) -> Dict[str, Any]:
    """
    Serialize a scored idea to dict format for prompt.

    Args:
        scored_idea: Scored idea object
        idea: Original idea object

    Returns:
        Serialized scored idea dictionary
    """
    return {
        "idea_id": str(scored_idea.idea_id),
        "title": idea.title,
        "one_liner": idea.one_liner,
        "problem": idea.problem,
        "solution": idea.solution,
        "target_user": idea.target_user,
        "key_features": idea.key_features,
        "yes_count": scored_idea.yes_count,
        "core_assumption": scored_idea.core_assumption,
        "fatal_flaws": [f.model_dump() for f in scored_idea.fatal_flaws],
        "one_risk": scored_idea.one_risk,
    }


def serialize_pain_point(pain_point: Any, max_evidence: int = 2) -> Dict[str, Any]:
    """
    Serialize a pain point to dict format for prompt.

    Args:
        pain_point: Pain point object
        max_evidence: Maximum evidence items to include

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
                "raw_quote": ev.raw_quote[:300],  # Truncate long quotes
                "source": ev.source.value,
            }
            for ev in pain_point.evidence[:max_evidence]
        ],
        "evidence_count": len(pain_point.evidence),
    }


def get_relevant_pain_points(
    state: VentureForgeState,
    idea: Any,
    max_pain_points: int = 4,
) -> List[Any]:
    """
    Get pain points relevant to a specific idea.

    Args:
        state: Current pipeline state
        idea: Idea to get pain points for
        max_pain_points: Maximum pain points to return

    Returns:
        List of relevant pain points, sorted by evidence count
    """
    relevant_pp_ids = set(idea.addresses_pain_point_ids)
    relevant_pps = [
        pp
        for pp in state.filtered_pain_points
        if pp.id in relevant_pp_ids
    ]

    # Sort by evidence count (descending)
    sorted_pps = sorted(
        relevant_pps,
        key=lambda pp: len(pp.evidence),
        reverse=True,
    )

    return sorted_pps[:max_pain_points]


def build_revision_block(state: VentureForgeState) -> str:
    """
    Build revision instruction block if in revision mode.

    Args:
        state: Current pipeline state

    Returns:
        Revision instruction text, or empty string if not in revision
    """
    if not state.revision_feedback:
        return ""

    last_critique = state.critiques[-1] if state.critiques else None
    failing_checks = (
        ", ".join(last_critique.failing_checks)
        if last_critique
        else "(see feedback)"
    )

    return (
        "THIS IS A REVISION ROUND for the pitch briefs. The critic flagged specific issues. "
        "You MUST address ONLY the failing checks mentioned. "
        "DO NOT change dimensions that were previously passing - preserve them exactly.\n"
        f"- Failing checks: {failing_checks}\n"
        f"- Feedback: {state.revision_feedback}\n\n"
        "CRITICAL: Only fix the specific issues mentioned. "
        "If tagline was passing before, keep it unchanged. "
        "If target_user was passing before, keep it unchanged. "
        "If evidence_links were passing before, keep them unchanged. "
        "Make minimal, surgical changes to address only the failing checks.\n\n"
    )


def build_user_prompt_single(
    state: VentureForgeState,
    scored_idea: Any,
) -> str:
    """
    Build user prompt for a SINGLE idea.

    Generates one brief at a time for better focus and token efficiency.

    Args:
        state: Current pipeline state
        scored_idea: Scored idea to write brief for

    Returns:
        Formatted user prompt
    """
    ideas_map = {str(idea.id): idea for idea in state.ideas}
    idea = ideas_map.get(str(scored_idea.idea_id))

    if not idea:
        logger.warning(
            f"[pitch_writer] Could not find idea {scored_idea.idea_id}"
        )
        return ""

    # Serialize scored idea
    scored_blob = serialize_scored_idea(scored_idea, idea)

    # Get relevant pain points
    relevant_pps = get_relevant_pain_points(state, idea)
    pp_blobs = [serialize_pain_point(pp) for pp in relevant_pps]

    # Build revision block if applicable
    revision_block = build_revision_block(state)

    return (
        f"Domain: {state.domain}\n\n"
        f"SCORED IDEA:\n{json.dumps(scored_blob, indent=2)}\n\n"
        f"SUPPORTING PAIN POINTS:\n{json.dumps(pp_blobs, indent=2)}\n\n"
        f"{revision_block}"
        "Write a full pitch brief for this idea. "
        "Return a single JSON object (not an array)."
    )


# =============================================================================
# LLM INTERACTION
# =============================================================================


def call_llm_for_pitch(
    state: VentureForgeState,
    scored_idea: Any,
    retry_count: int = 0,
) -> Optional[PitchBrief]:
    """
    Call LLM to generate a single pitch brief using structured output.

    Args:
        state: Current pipeline state
        scored_idea: Scored idea to write brief for
        retry_count: Current retry attempt (0-indexed)

    Returns:
        PitchBrief object, or None on failure
    """
    llm = get_structured_llm(
        PitchBrief,
        temperature=0.6,
        reasoning=False,
    )

    messages = [
        SystemMessage(content=build_system_prompt()),
        HumanMessage(content=build_user_prompt_single(state, scored_idea)),
    ]

    start_time = time.monotonic()
    try:
        response = llm.invoke(messages)
        # response is already a PitchBrief object from structured output
    except Exception as e:
        logger.error(
            f"[pitch_writer] LLM invocation failed for idea {scored_idea.idea_id} "
            f"(attempt {retry_count + 1}): {e}"
        )
        return None

    elapsed = time.monotonic() - start_time
    logger.info(
        f"[pitch_writer] LLM responded in {elapsed:.1f}s for idea "
        f"{scored_idea.idea_id} (attempt {retry_count + 1})"
    )

    return response


def parse_pitch_response(
    pitch_brief: Optional[PitchBrief],
    idea_id: UUID,
    retry_count: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Validate and convert PitchBrief object to dict format.

    Args:
        pitch_brief: PitchBrief object from structured output
        idea_id: ID of the idea being pitched
        retry_count: Current retry attempt

    Returns:
        Pitch brief dict, or None on validation failure
    """
    if pitch_brief is None:
        logger.error(
            f"[pitch_writer] No pitch brief returned for idea {idea_id} "
            f"(attempt {retry_count + 1})"
        )
        return None

    # Convert to dict for downstream processing
    pitch_dict = pitch_brief.model_dump()

    # Validate that pitch_dict has meaningful content (not placeholders)
    # Check for placeholder text in markdown_content
    markdown = pitch_dict.get("markdown_content", "")
    if markdown and len(markdown) < 100:
        logger.warning(
            f"[pitch_writer] Rejected pitch with insufficient markdown_content "
            f"({len(markdown)} chars < 100 minimum) for idea {idea_id}"
        )
        return None

    # Check for placeholder text
    placeholder_patterns = ["full one-page brief", "placeholder", "tbd", "to be determined"]
    if any(pattern.lower() in markdown.lower() for pattern in placeholder_patterns):
        logger.warning(
            f"[pitch_writer] Rejected pitch with placeholder text in markdown_content for idea {idea_id}"
        )
        return None

    # Check for all empty fields (indicates LLM failure)
    required_fields = ["title", "problem", "solution", "target_user", "market_opportunity"]
    empty_count = sum(1 for field in required_fields if not pitch_dict.get(field) or len(str(pitch_dict.get(field, "")).strip()) < 3)
    if empty_count >= len(required_fields) - 1:  # Allow at most 1 field to be empty
        logger.warning(
            f"[pitch_writer] Rejected pitch with {empty_count}/{len(required_fields)} empty required fields for idea {idea_id}"
        )
        return None

    return pitch_dict


def generate_pitch_with_retry(
    state: VentureForgeState,
    scored_idea: Any,
    max_attempts: int = MAX_PITCH_GENERATION_ATTEMPTS,
) -> Optional[Dict[str, Any]]:
    """
    Generate pitch brief with retry logic.

    Args:
        state: Current pipeline state
        scored_idea: Scored idea to write brief for
        max_attempts: Maximum retry attempts

    Returns:
        Pitch brief dict, or None if all attempts fail
    """
    for attempt in range(max_attempts):
        # Call LLM
        pitch_brief = call_llm_for_pitch(state, scored_idea, retry_count=attempt)
        if pitch_brief is None:
            continue

        # Validate and convert to dict
        pitch_dict = parse_pitch_response(
            pitch_brief,
            scored_idea.idea_id,
            retry_count=attempt,
        )

        if pitch_dict is not None:
            return pitch_dict

        logger.warning(
            f"[pitch_writer] Attempt {attempt + 1}/{max_attempts} failed "
            f"for idea {scored_idea.idea_id}"
        )

    logger.error(
        f"[pitch_writer] All {max_attempts} attempts failed for idea "
        f"{scored_idea.idea_id}"
    )
    return None


# =============================================================================
# PITCH BRIEF CONVERSION
# =============================================================================


def collect_evidence_urls(idea_id: UUID, state: VentureForgeState) -> List[str]:
    """
    Collect all evidence URLs from pain points referenced by this idea.
    Fallback for when LLM fails to provide evidence_links.

    Args:
        idea_id: ID of the idea
        state: Current pipeline state

    Returns:
        List of evidence URLs from relevant pain points
    """
    urls = []
    idea = next((i for i in state.ideas if str(i.id) == str(idea_id)), None)
    if not idea:
        return urls

    # Use filtered_pain_points instead of pain_points
    for pp_id in idea.addresses_pain_point_ids:
        pp = next((p for p in state.filtered_pain_points if str(p.id) == str(pp_id)), None)
        if pp and hasattr(pp, 'evidence') and pp.evidence:
            for ev in pp.evidence:
                if ev.source_url and ev.source_url not in urls:
                    urls.append(ev.source_url)

    logger.info(
        f"[pitch_writer] Collected {len(urls)} evidence URLs from pain points for idea {idea_id}: {urls[:5]}..."
    )
    return urls


def validate_evidence_links(
    evidence_links: List[str],
    idea_id: UUID,
    state: VentureForgeState,
) -> List[str]:
    """
    Validate evidence links against pain point evidence URLs.
    
    Filters out any URLs that don't exist in the pain points' evidence arrays.
    This prevents the LLM from hallucinating or modifying URLs.
    
    If no valid URLs remain after filtering, falls back to collecting URLs
    from pain points directly.
    
    Args:
        evidence_links: List of URLs from LLM response
        idea_id: ID of the idea
        state: Current pipeline state
        
    Returns:
        Filtered list of valid URLs that exist in pain points
    """
    # Get the idea to find its pain point IDs
    idea = next((i for i in state.ideas if i.id == idea_id), None)
    if not idea:
        logger.warning(f"[pitch_writer] Could not find idea {idea_id} for evidence validation")
        return evidence_links
    
    # Collect all valid URLs from relevant pain points
    valid_urls = set()
    for pp in state.filtered_pain_points:
        if pp.id in idea.addresses_pain_point_ids:
            for ev in pp.evidence:
                valid_urls.add(ev.source_url)
    
    # Filter evidence_links to only include valid URLs
    validated_links = [url for url in evidence_links if url in valid_urls]
    
    # Log if we filtered out any URLs
    if len(validated_links) < len(evidence_links):
        invalid_urls = set(evidence_links) - set(validated_links)
        logger.warning(
            f"[pitch_writer] Filtered out {len(invalid_urls)} hallucinated URLs for idea {idea_id}: "
            f"{invalid_urls}"
        )
        logger.info(
            f"[pitch_writer] Kept {len(validated_links)} valid URLs from pain points"
        )
    
    # Fallback: if no valid URLs after filtering, collect from pain points directly
    if not validated_links:
        logger.warning(
            f"[pitch_writer] No valid evidence links from LLM for idea {idea_id}. "
            f"Falling back to collecting URLs from pain points."
        )
        validated_links = collect_evidence_urls(idea_id, state)
        logger.info(
            f"[pitch_writer] Collected {len(validated_links)} URLs from pain points for idea {idea_id}"
        )
    
    return validated_links


def convert_to_pitch_brief(
    pitch_dict: Dict[str, Any],
    idea_id: UUID,
    state: VentureForgeState,
) -> Optional[PitchBrief]:
    """
    Convert raw pitch dict to PitchBrief object.

    Args:
        pitch_dict: Raw pitch brief dictionary
        idea_id: ID of the idea
        state: Current pipeline state (for revision_count lookup)

    Returns:
        PitchBrief object, or None on conversion failure
    """
    try:
        # Parse nested objects
        competitive_landscape = None
        if "competitive_landscape" in pitch_dict:
            comp_data = pitch_dict["competitive_landscape"]
            if isinstance(comp_data, dict):
                competitive_landscape = CompetitiveLandscape(**comp_data)

        validation_plan = None
        if "validation_plan" in pitch_dict:
            val_data = pitch_dict["validation_plan"]
            if isinstance(val_data, dict):
                validation_plan = ValidationPlan(**val_data)

        # Coerce list fields to strings (LLM sometimes returns lists)
        def coerce_to_string(value: Any) -> str:
            if isinstance(value, list):
                return " ".join(str(item) for item in value)
            return str(value) if value else ""

        # Get revision count from state
        revision_count = state.revision_counts.get(str(idea_id), 0)
        
        # Validate evidence links against pain point URLs
        raw_evidence_links = pitch_dict.get("evidence_links", [])
        validated_evidence_links = validate_evidence_links(raw_evidence_links, idea_id, state)

        # Get idea title as fallback for empty pitch title
        idea_title = ""
        idea = next((i for i in state.ideas if i.id == idea_id), None)
        if idea:
            idea_title = idea.title

        # Create PitchBrief with correct field names matching schema
        return PitchBrief(
            idea_id=idea_id,
            title=pitch_dict.get("title", "") or idea_title,
            tagline=pitch_dict.get("tagline", ""),
            problem=pitch_dict.get("problem", ""),
            solution=pitch_dict.get("solution", ""),
            target_user=pitch_dict.get("target_user", ""),
            market_opportunity=pitch_dict.get("market_opportunity", ""),
            competitive_landscape=competitive_landscape,
            differentiation=pitch_dict.get("differentiation", ""),
            validation_plan=validation_plan,
            business_model=pitch_dict.get("business_model", ""),
            go_to_market=pitch_dict.get("go_to_market", ""),
            key_risk=pitch_dict.get("key_risk", ""),
            next_steps=coerce_to_string(pitch_dict.get("next_steps", "")),
            evidence_links=validated_evidence_links,
            markdown_content=pitch_dict.get("markdown_content", ""),
            revision_count=revision_count,
        )

    except Exception as e:
        logger.error(
            f"[pitch_writer] Failed to convert pitch dict to PitchBrief "
            f"for idea {idea_id}: {e}"
        )
        return None


# =============================================================================
# MAIN AGENT FUNCTION
# =============================================================================


def run(state: VentureForgeState) -> Dict[str, Any]:
    """
    Main pitch writer agent function.

    Args:
        state: Current pipeline state

    Returns:
        State patch dictionary
    """
    logger.info("[pitch_writer] Starting pitch brief generation")

    # Get target ideas
    target_ideas = get_target_ideas(state)

    if not target_ideas:
        logger.warning("[pitch_writer] No top scored ideas available")
        return {
            "pitch_briefs": [],
            "pitch_writer_attempts": state.pitch_writer_attempts + 1,
            **state.add_event(
                agent="pitch_writer",
                stage=PipelineStage.WRITING,
                kind="warning",
                message="No top scored ideas available for pitch writing",
            ),
        }

    # Generate pitch briefs one at a time
    pitch_briefs: List[PitchBrief] = []
    failed_ideas: List[UUID] = []

    for scored_idea in target_ideas:
        logger.info(
            f"[pitch_writer] Generating brief for idea {scored_idea.idea_id}"
        )

        # Generate pitch with retry
        pitch_dict = generate_pitch_with_retry(state, scored_idea)

        if pitch_dict is None:
            failed_ideas.append(scored_idea.idea_id)
            
            # If in revision mode and failed, remove the brief to prevent infinite loop
            if state.current_revision_idea_id:
                logger.warning(
                    f"[pitch_writer] Revision failed for idea {state.current_revision_idea_id} after {MAX_PITCH_GENERATION_ATTEMPTS} attempts. "
                    f"Removing brief to prevent infinite revision loop."
                )
                # Remove the failed brief from pitch_briefs
                filtered_briefs = [b for b in state.pitch_briefs if b.idea_id != state.current_revision_idea_id]
                patch = {
                    "pitch_briefs": filtered_briefs,
                    "current_revision_idea_id": None,
                    "next_node": "orchestrator",
                    "pitch_writer_attempts": state.pitch_writer_attempts + 1,
                }
                patch.update(
                    state.add_event(
                        agent="pitch_writer",
                        stage=PipelineStage.WRITING,
                        kind="error",
                        message=f"Failed to revise pitch brief for idea {state.current_revision_idea_id} after {MAX_PITCH_GENERATION_ATTEMPTS} attempts. Removed brief to prevent infinite loop.",
                        idea_id=str(state.current_revision_idea_id),
                    )
                )
                return patch
            
            continue

        # Convert to PitchBrief object
        pitch_brief = convert_to_pitch_brief(pitch_dict, scored_idea.idea_id, state)

        if pitch_brief is None:
            failed_ideas.append(scored_idea.idea_id)
            
            # If in revision mode and conversion failed, remove the brief to prevent infinite loop
            if state.current_revision_idea_id:
                logger.warning(
                    f"[pitch_writer] Conversion failed for idea {state.current_revision_idea_id} in revision mode. "
                    f"Removing brief to prevent infinite revision loop."
                )
                # Remove the failed brief from pitch_briefs
                filtered_briefs = [b for b in state.pitch_briefs if b.idea_id != state.current_revision_idea_id]
                patch = {
                    "pitch_briefs": filtered_briefs,
                    "current_revision_idea_id": None,
                    "next_node": "orchestrator",
                    "pitch_writer_attempts": state.pitch_writer_attempts + 1,
                }
                patch.update(
                    state.add_event(
                        agent="pitch_writer",
                        stage=PipelineStage.WRITING,
                        kind="error",
                        message=f"Failed to convert pitch brief for idea {state.current_revision_idea_id}. Removed brief to prevent infinite loop.",
                        idea_id=str(state.current_revision_idea_id),
                    )
                )
                return patch
            
            continue

        pitch_briefs.append(pitch_brief)

    # Log results
    logger.info(
        f"[pitch_writer] Generated {len(pitch_briefs)} pitch briefs "
        f"({len(failed_ideas)} failed)"
    )

    if failed_ideas:
        logger.warning(
            f"[pitch_writer] Failed to generate briefs for ideas: {failed_ideas}"
        )

    return {
        "pitch_briefs": pitch_briefs,
        "pitch_writer_attempts": state.pitch_writer_attempts + 1,
        "current_revision_idea_id": None,
        "next_node": "orchestrator",
        **state.add_event(
            agent="pitch_writer",
            stage=PipelineStage.WRITING,
            kind="info" if pitch_briefs else "error",
            message=(
                f"Generated {len(pitch_briefs)} pitch briefs "
                f"({len(failed_ideas)} failed)"
            ),
        ),
    }
