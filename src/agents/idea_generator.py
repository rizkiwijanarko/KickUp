"""Idea Generator — clusters pain points into themes and generates distinct
startup ideas.

Pipeline flow
=============
1. Receive filtered pain points from state.
2. Cap to a manageable number (~50) to fit context window.
3. LLM brainstorms ideas grouped by themes, returns structured JSON.
4. Code validates that every ``addresses_pain_point_ids`` references a real
   pain point UUID that was present in the input.
5. Return validated Idea objects.
"""
from __future__ import annotations

import json
import logging
import time
from uuid import UUID, uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.client import extract_json, get_llm
from src.llm.prompts import get_prompt
from src.state.schema import Idea, PipelineStage, VentureForgeState

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Tunables
# ------------------------------------------------------------------
_MAX_PAIN_POINTS_CONTEXT: int = 50  # avoid blowing context window
_IDEAS_PER_RUN_DEFAULT: int = 5


def _build_system_prompt() -> str:
    return get_prompt("idea_generator")


def _build_user_prompt(state: VentureForgeState) -> str:
    # Sort pain points by evidence count (descending) to prioritize well-validated pain points
    # Then cap to context window limit
    sorted_pps = sorted(
        state.filtered_pain_points,
        key=lambda pp: len(pp.evidence),
        reverse=True
    )
    pps = sorted_pps[:_MAX_PAIN_POINTS_CONTEXT]
    domain = state.domain
    count = state.ideas_per_run or _IDEAS_PER_RUN_DEFAULT
    feedback = state.revision_feedback or "None"

    # Serialize pain points with full evidence array
    pp_blobs: list[dict] = [
        {
            "id": str(pp.id),
            "title": pp.title,
            "description": pp.description,
            "evidence": [
                {
                    "source_url": ev.source_url,
                    "raw_quote": ev.raw_quote,
                    "source": ev.source.value,
                }
                for ev in pp.evidence
            ],
            "evidence_count": len(pp.evidence),
        }
        for pp in pps
    ]

    # If revision feedback exists, this run is part of the reflection
    # loop (typically triggered by the Critic for positioning issues
    # such as target_is_contained_fire or competition_embraced_with_thesis).
    # Make that explicit in the prompt so the LLM focuses on fixing
    # those weaknesses first.
    revision_block = ""
    if state.revision_feedback:
        revision_block = (
            "THIS IS A REVISION ROUND. The critic flagged weaknesses in "
            "positioning (e.g., target user not a contained community, "
            "or weak competitive thesis). You MUST address the following "
            "feedback before generating ideas:\n"  # noqa: E501
            f"- Critic feedback: {feedback}\n\n"
            "In your new ideas, make the target_user a specific, named, "
            "reachable community (a 'contained fire') and make the "
            "competition thesis explicit: what are users doing today and "
            "what incumbents are afraid to do.\n\n"
        )

    # Determine minimum pain point references based on availability
    # If only 1 pain point exists, require 1; otherwise require 2
    min_refs = min(2, len(pps))
    
    # Build adaptive requirement block based on available pain points
    if len(pps) == 1:
        requirement_block = (
            "**SPECIAL CASE: Only 1 pain point available.**\\n"
            "Generate ideas that deeply address this single pain point. "
            "Each idea must reference this pain point UUID in 'addresses_pain_point_ids'. "
            "Focus on different solution angles, user segments, or implementation approaches for variety.\\n\\n"
        )
    elif len(pps) >= 2:
        requirement_block = (
            f"**CRITICAL REQUIREMENT: Each idea MUST reference AT LEAST {min_refs} pain point UUIDs in 'addresses_pain_point_ids'.**\\n"
            f"Ideas with fewer than {min_refs} references will be REJECTED. "
            "Cross-pollinate pain points to create stronger, more defensible ideas that solve multiple problems.\\n\\n"
        )
    else:
        requirement_block = "ERROR: No pain points provided. Cannot generate ideas.\\n\\n"
    
    user_text = (
        f"Domain: {domain}\\n"
        f"Ideas to generate: {count}\\n\\n"
        f"PAIN POINTS ({len(pps)} provided):\\n"
        f"{json.dumps(pp_blobs, indent=2)}\\n\\n"
        f"{revision_block}"
        f"{requirement_block}"
        "Only use UUIDs from the pain points list above — do not invent new UUIDs.\\n\\n"
        "Return JSON: {\\\"ideas\\\": [ ... ]}."
    )
    return user_text


def _build_user_prompt_single(state: VentureForgeState, idea_number: int, total_ideas: int) -> str:
    """Build prompt for generating a SINGLE idea to reduce token usage.
    
    Generates one idea at a time for:
    - Better fit within vLLM 2048 token limit
    - More focused, comprehensive ideas
    - LLM can concentrate on each idea individually
    
    Args:
        state: Current pipeline state
        idea_number: Which idea this is (1-indexed for display)
        total_ideas: Total number of ideas to generate
    """
    # Sort pain points by evidence count
    sorted_pps = sorted(
        state.filtered_pain_points,
        key=lambda pp: len(pp.evidence),
        reverse=True
    )
    pps = sorted_pps[:_MAX_PAIN_POINTS_CONTEXT]
    domain = state.domain
    feedback = state.revision_feedback or "None"
    
    # Serialize pain points with evidence (limit to top 2 evidence items per pain point)
    pp_blobs: list[dict] = [
        {
            "id": str(pp.id),
            "title": pp.title,
            "description": pp.description,
            "evidence": [
                {
                    "source_url": ev.source_url,
                    "raw_quote": ev.raw_quote[:300],  # Truncate long quotes
                    "source": ev.source.value,
                }
                for ev in pp.evidence[:2]  # Only top 2 evidence items
            ],
            "evidence_count": len(pp.evidence),
        }
        for pp in pps
    ]
    
    # Revision block if applicable
    revision_block = ""
    if state.revision_feedback:
        revision_block = (
            "THIS IS A REVISION ROUND. The critic flagged weaknesses in positioning. "
            "You MUST address the following feedback:\n"
            f"- Critic feedback: {feedback}\n\n"
            "Make the target_user a specific, named, reachable community (a 'contained fire') "
            "and make the competition thesis explicit.\n\n"
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
        f"{revision_block}"
        f"{requirement_block}"
        "Only use UUIDs from the pain points list above — do not invent new UUIDs.\n\n"
        "Return a single JSON object (not an array): {\"title\": ..., \"one_liner\": ..., ...}"
    )
    return user_text


def _invoke_llm_single(state: VentureForgeState, idea_number: int, total_ideas: int, retry_count: int = 0) -> dict | None:
    """Invoke LLM to generate a SINGLE idea.
    
    Args:
        state: Current pipeline state
        idea_number: Which idea this is (1-indexed)
        total_ideas: Total number of ideas to generate
        retry_count: Current retry attempt (0-indexed)
    
    Returns:
        Raw idea dict, or None on failure
    """
    llm = get_llm(temperature=0.7, max_tokens=16384, reasoning=False)
    
    system_prompt = _build_system_prompt()
    system_prompt += "\n\n**CRITICAL: Output ONLY a single JSON object. No markdown fences, no explanations. Start with { and end with }.**"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_build_user_prompt_single(state, idea_number, total_ideas)),
    ]
    
    start = time.monotonic()
    try:
        raw = llm.invoke(messages)
        content = raw.content if hasattr(raw, "content") else str(raw)
    except Exception as e:
        logger.error(f"[idea_generator] LLM invocation failed for idea {idea_number} (attempt {retry_count + 1}): {e}")
        return None
    
    elapsed = time.monotonic() - start
    logger.info(f"[idea_generator] LLM responded in {elapsed:.1f}s for idea {idea_number} (attempt {retry_count + 1})")
    
    # Warn if response looks truncated
    if content and not content.rstrip().endswith('}'):
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
    
    return None


def _invoke_llm(state: VentureForgeState) -> list[dict]:
    """Call LLM, parse JSON, return raw idea dicts."""
    llm = get_llm(temperature=0.7, max_tokens=16384, reasoning=False)
    
    # Add explicit JSON-only instruction
    system_prompt = _build_system_prompt()
    system_prompt += "\n\n**CRITICAL: Output ONLY the JSON object. No markdown code fences, no explanations, no preamble. Start with { and end with }.**"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_build_user_prompt(state)),
    ]

    start = time.monotonic()
    try:
        raw = llm.invoke(messages)
        content = raw.content if hasattr(raw, "content") else str(raw)
    except Exception as e:
        logger.error(f"[idea_generator] LLM invocation failed after {time.monotonic()-start:.1f}s: {e}")
        return []

    logger.info(f"[idea_generator] LLM responded in {time.monotonic()-start:.1f}s")
    
    # Debug: log response preview
    logger.info(f"[idea_generator] Response preview (first 500 chars): {content[:500]}")

    parsed = extract_json(content)
    if parsed is None:
        logger.error(f"[idea_generator] JSON extraction failed. Response length: {len(content)} chars")
        logger.error(f"[idea_generator] Full response (first 2000 chars): {content[:2000]}")
        return []

    ideas = parsed.get("ideas") if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else []
    if not isinstance(ideas, list):
        logger.warning("[idea_generator] LLM did not return an array of ideas")
        return []
    return ideas


def _validate_idea(raw: dict, valid_ids: set[UUID], min_refs: int = 2) -> Idea | None:
    """Return an Idea if it references real pain points, else None.
    
    Args:
        raw: Raw idea dict from LLM
        valid_ids: Set of valid pain point UUIDs
        min_refs: Minimum number of pain point references required (default 2)
    """
    # Parse UUID references
    raw_ids = raw.get("addresses_pain_point_ids", [])
    resolved: list[UUID] = []
    for rid in raw_ids:
        try:
            uid = UUID(str(rid))
            if uid in valid_ids:
                resolved.append(uid)
        except (ValueError, TypeError):
            continue

    if len(resolved) < min_refs:
        logger.debug(
            f"[idea_generator] REJECTED — idea '{raw.get('title', '?')}' "
            f"references only {len(resolved)} valid pain point(s), need {min_refs}"
        )
        return None

    try:
        idea = Idea(
            id=uuid4(),
            title=raw["title"],
            one_liner=raw["one_liner"],
            problem=raw["problem"],
            solution=raw["solution"],
            target_user=raw["target_user"],
            key_features=raw.get("key_features", []),
            addresses_pain_point_ids=resolved,
        )
        return idea
    except Exception as e:
        logger.debug(f"[idea_generator] REJECTED — malformed idea: {e}")
        return None


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def run(state: VentureForgeState) -> dict:
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
    
    # ONE-IDEA-AT-A-TIME GENERATION
    # Determine how many ideas to generate
    if state.current_revision_idea_id:
        count = 1
        logger.info(f"[idea_generator] Revision mode: generating 1 replacement idea for {state.current_revision_idea_id}")
    else:
        count = state.ideas_per_run or _IDEAS_PER_RUN_DEFAULT
        logger.info(f"[idea_generator] Initial generation: generating {count} ideas one at a time")
    
    MAX_RETRIES = 3
    raw_ideas = []
    
    # Generate one idea at a time
    for i in range(count):
        idea_number = i + 1
        logger.info(f"[idea_generator] Generating idea {idea_number} of {count}")
        
        raw_idea = None
        for retry in range(MAX_RETRIES):
            raw_idea = _invoke_llm_single(state, idea_number, count, retry_count=retry)
            
            if raw_idea:
                logger.info(f"[idea_generator] Successfully generated idea {idea_number} on attempt {retry + 1}")
                raw_ideas.append(raw_idea)
                break
            
            if retry < MAX_RETRIES - 1:
                logger.warning(
                    f"[idea_generator] Attempt {retry + 1}/{MAX_RETRIES} failed for idea {idea_number}. Retrying..."
                )
            else:
                logger.error(
                    f"[idea_generator] All {MAX_RETRIES} attempts failed for idea {idea_number}."
                )
    
    logger.info(f"[idea_generator] LLM produced {len(raw_ideas)} raw ideas")
    
    # DEBUG: Log first raw idea to diagnose validation failures
    if raw_ideas:
        logger.info(f"[idea_generator] Sample raw idea: {json.dumps(raw_ideas[0], indent=2)}")
        logger.info(f"[idea_generator] Valid pain point IDs: {[str(vid) for vid in list(valid_ids)[:3]]}")

    validated: list[Idea] = []
    for raw in raw_ideas:
        idea = _validate_idea(raw, valid_ids, min_refs)
        if idea:
            validated.append(idea)

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
                f"addressing ≥{min_refs} validated pain points each."
            ),
        )
    )
    return patch
