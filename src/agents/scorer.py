"""Scorer — evaluates ideas with a binary yes/no rubric."""
from __future__ import annotations

import json
import logging
import time
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage

from src.constants import (
    SCORER_LLM_TEMPERATURE,
    SCORER_LLM_MAX_TOKENS,
    ERROR_SCORER_LLM_INVOCATION_FAILED,
    ERROR_SCORER_JSON_EXTRACTION_FAILED,
    WARNING_SCORER_NO_IDEAS,
)
from src.exceptions import LLMError, ValidationError
from src.llm.client import coerce_rubric_bools, extract_json, get_llm
from src.llm.prompts import get_prompt
from src.state.schema import (
    DemandRubric,
    FatalFlaw,
    FeasibilityRubric,
    NoveltyRubric,
    PipelineStage,
    ScoredIdea,
    VentureForgeState,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Prompt Building
# ============================================================================


def _build_system_prompt() -> str:
    """Build the system prompt for the Scorer agent.
    
    Returns:
        System prompt with JSON-only instruction
    """
    base_prompt = get_prompt("scorer")
    json_instruction = (
        "\n\n**CRITICAL: Output ONLY the JSON array. "
        "No markdown code fences, no explanations, no preamble. "
        "Start with [ and end with ].**"
    )
    return base_prompt + json_instruction


def _get_ideas_to_score(state: VentureForgeState) -> list:
    """Get the list of ideas to score based on revision mode.
    
    In revision mode, only score the specific idea being revised.
    In initial mode, score all ideas.
    
    Args:
        state: Current pipeline state
        
    Returns:
        List of ideas to score
    """
    if state.current_revision_idea_id:
        # Revision mode: only score the revised idea
        ideas = [idea for idea in state.ideas if idea.id == state.current_revision_idea_id]
        if not ideas:
            raise ValidationError(f"Idea {state.current_revision_idea_id} not found for revision scoring")
        return ideas
    else:
        # Initial mode: score all ideas
        return state.ideas


def _serialize_ideas(ideas: list) -> list[dict]:
    """Serialize ideas for JSON output.
    
    Args:
        ideas: List of ideas to serialize
        
    Returns:
        List of serialized idea dicts
    """
    return [
        {
            "id": str(idea.id),
            "title": idea.title,
            "one_liner": idea.one_liner,
            "problem": idea.problem,
            "solution": idea.solution,
            "target_user": idea.target_user,
        }
        for idea in ideas
    ]


def _serialize_pain_points(pain_points: list) -> list[dict]:
    """Serialize pain points for JSON output, sorted by evidence count.
    
    Args:
        pain_points: List of pain points to serialize
        
    Returns:
        List of serialized pain point dicts, sorted by evidence count (descending)
    """
    sorted_pps = sorted(
        pain_points,
        key=lambda pp: len(pp.evidence),
        reverse=True
    )
    return [
        {
            "id": str(pp.id),
            "title": pp.title,
            "description": pp.description,
        }
        for pp in sorted_pps
    ]


def _build_user_prompt(state: VentureForgeState) -> str:
    """Build the user prompt for the Scorer agent.
    
    Args:
        state: Current pipeline state
        
    Returns:
        User prompt with ideas and pain points
    """
    ideas_to_score = _get_ideas_to_score(state)
    ideas_blobs = _serialize_ideas(ideas_to_score)
    pp_blobs = _serialize_pain_points(state.filtered_pain_points)
    
    user_text = (
        f"Domain: {state.domain}\n\n"
        f"IDEAS TO SCORE:\n{json.dumps(ideas_blobs, indent=2)}\n\n"
        f"SUPPORTING PAIN POINTS:\n{json.dumps(pp_blobs, indent=2)}\n\n"
        "Evaluate each idea according to the binary rubric. "
        "Return a JSON array of scored ideas."
    )
    return user_text


# ============================================================================
# LLM Interaction
# ============================================================================


def _invoke_llm(state: VentureForgeState) -> list[dict]:
    """Invoke the LLM to score ideas.
    
    Args:
        state: Current pipeline state
        
    Returns:
        List of raw scored idea dicts from LLM
        
    Raises:
        LLMError: If LLM invocation fails
        ValidationError: If JSON extraction fails
    """
    llm = get_llm(
        temperature=SCORER_LLM_TEMPERATURE,
        max_tokens=SCORER_LLM_MAX_TOKENS,
        reasoning=False  # Disable thinking mode for structured JSON
    )
    
    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=_build_user_prompt(state)),
    ]
    
    start = time.monotonic()
    try:
        raw = llm.invoke(messages)
        content = raw.content if hasattr(raw, "content") else str(raw)
    except Exception as e:
        logger.error(f"[scorer] LLM invocation failed: {e}")
        raise LLMError(ERROR_SCORER_LLM_INVOCATION_FAILED.format(error=str(e)))
    
    elapsed = time.monotonic() - start
    logger.info(f"[scorer] LLM responded in {elapsed:.1f}s")
    logger.info(f"[scorer] Response preview (first 500 chars): {content[:500]}")
    
    parsed = extract_json(content)
    if parsed is None:
        logger.error(f"[scorer] JSON extraction failed. Response length: {len(content)} chars")
        logger.error(f"[scorer] Full response (first 2000 chars): {content[:2000]}")
        raise ValidationError(ERROR_SCORER_JSON_EXTRACTION_FAILED)
    
    # Unwrap if LLM returned {"scored_ideas": [...]}
    if isinstance(parsed, dict) and "scored_ideas" in parsed:
        return parsed["scored_ideas"]
    
    return parsed if isinstance(parsed, list) else []


# ============================================================================
# Score Parsing
# ============================================================================


def _parse_rubrics(raw: dict) -> tuple[FeasibilityRubric, DemandRubric, NoveltyRubric]:
    """Parse and coerce rubric fields from raw LLM output.
    
    Args:
        raw: Raw scored idea dict
        
    Returns:
        Tuple of (feasibility_rubric, demand_rubric, novelty_rubric)
    """
    f_rubric = FeasibilityRubric(**coerce_rubric_bools(raw["feasibility_rubric"]))
    d_rubric = DemandRubric(**coerce_rubric_bools(raw["demand_rubric"]))
    n_rubric = NoveltyRubric(**coerce_rubric_bools(raw["novelty_rubric"]))
    return f_rubric, d_rubric, n_rubric


def _calculate_yes_count(
    f_rubric: FeasibilityRubric,
    d_rubric: DemandRubric,
    n_rubric: NoveltyRubric
) -> int:
    """Calculate the total number of 'yes' answers across all rubrics.
    
    Args:
        f_rubric: Feasibility rubric
        d_rubric: Demand rubric
        n_rubric: Novelty rubric
        
    Returns:
        Total yes count (0-8)
    """
    return sum([
        f_rubric.can_be_solved_manually_first,
        f_rubric.has_schlep_or_unsexy_advantage,
        f_rubric.can_2_3_person_team_build_mvp_in_6_months,
        d_rubric.addresses_at_least_2_pain_points,
        d_rubric.is_painkiller_not_vitamin,
        d_rubric.has_clear_vein_of_early_adopters,
        n_rubric.differentiated_from_current_behavior,
        n_rubric.has_path_out_of_niche,
    ])


def _parse_fatal_flaws(raw: dict) -> list[FatalFlaw]:
    """Parse fatal flaws from raw LLM output.
    
    Args:
        raw: Raw scored idea dict
        
    Returns:
        List of FatalFlaw objects
    """
    raw_flaws = raw.get("fatal_flaws", [])
    return [FatalFlaw(**f) for f in raw_flaws if isinstance(f, dict)]


def _extract_idea_id(raw: dict) -> UUID | None:
    """Extract idea_id from raw LLM output.
    
    LLM may return either 'id' (echoing input) or 'idea_id'.
    
    Args:
        raw: Raw scored idea dict
        
    Returns:
        UUID of the idea, or None if not found
    """
    idea_id = raw.get("idea_id") or raw.get("id")
    return idea_id if idea_id else None


def _parse_scored_idea(raw: dict) -> ScoredIdea | None:
    """Parse a single scored idea from raw LLM output.
    
    Args:
        raw: Raw scored idea dict
        
    Returns:
        ScoredIdea object, or None if parsing fails
    """
    try:
        idea_id = _extract_idea_id(raw)
        if not idea_id:
            logger.debug("[scorer] Skipping scored idea: missing idea_id")
            return None
        
        f_rubric, d_rubric, n_rubric = _parse_rubrics(raw)
        yes_count = _calculate_yes_count(f_rubric, d_rubric, n_rubric)
        fatal_flaws = _parse_fatal_flaws(raw)
        
        scored = ScoredIdea(
            idea_id=idea_id,
            reasoning_trace=raw.get("reasoning_trace", ""),
            feasibility_rubric=f_rubric,
            demand_rubric=d_rubric,
            novelty_rubric=n_rubric,
            core_assumption=raw["core_assumption"],
            fatal_flaws=fatal_flaws,
            yes_count=yes_count,
            verdict=raw["verdict"],
            one_risk=raw["one_risk"],
        )
        return scored
        
    except Exception as e:
        logger.debug(f"[scorer] Skipping malformed scored idea: {e}")
        return None


def _parse_all_scores(raw_scores: list[dict]) -> list[ScoredIdea]:
    """Parse all scored ideas from raw LLM output.
    
    Args:
        raw_scores: List of raw scored idea dicts
        
    Returns:
        List of ScoredIdea objects (skipping malformed entries)
    """
    scored_ideas = []
    for raw in raw_scores:
        scored = _parse_scored_idea(raw)
        if scored:
            scored_ideas.append(scored)
    return scored_ideas


# ============================================================================
# Ranking & Merging
# ============================================================================


def _rank_ideas(scored_ideas: list[ScoredIdea]) -> list[ScoredIdea]:
    """Rank scored ideas by yes_count (descending).
    
    Args:
        scored_ideas: List of scored ideas
        
    Returns:
        Sorted list with rank field set
    """
    scored_ideas.sort(key=lambda s: s.yes_count, reverse=True)
    for i, s in enumerate(scored_ideas):
        s.rank = i + 1
    return scored_ideas


def _merge_revision_scores(
    state: VentureForgeState,
    new_scores: list[ScoredIdea]
) -> list[ScoredIdea]:
    """Merge new scores with existing scores in revision mode.
    
    Removes the old score for the revised idea and adds the new score,
    then re-ranks all scores.
    
    Args:
        state: Current pipeline state
        new_scores: New scores from LLM
        
    Returns:
        Merged and re-ranked list of all scores
    """
    # Remove old score for the revised idea
    existing_scores = [
        s for s in state.scored_ideas
        if s.idea_id != state.current_revision_idea_id
    ]
    
    # Add new score
    all_scores = existing_scores + new_scores
    
    # Re-rank all scores
    all_scores = _rank_ideas(all_scores)
    
    logger.info(
        f"[scorer] Revision mode: re-scored idea {state.current_revision_idea_id}. "
        f"Total scores: {len(all_scores)}"
    )
    
    return all_scores


def _get_verdict_counts(scored_ideas: list[ScoredIdea]) -> tuple[int, int, int]:
    """Count verdicts for logging.
    
    Args:
        scored_ideas: List of scored ideas
        
    Returns:
        Tuple of (pursue_count, explore_count, park_count)
    """
    pursue = sum(1 for s in scored_ideas if s.verdict == "pursue")
    explore = sum(1 for s in scored_ideas if s.verdict == "explore")
    park = sum(1 for s in scored_ideas if s.verdict == "park")
    return pursue, explore, park


# ============================================================================
# Patch Building
# ============================================================================


def _build_success_patch(
    state: VentureForgeState,
    all_scores: list[ScoredIdea],
    new_scores_count: int
) -> dict:
    """Build the patch dict for successful scoring.
    
    Args:
        state: Current pipeline state
        all_scores: All scored ideas (merged if in revision mode)
        new_scores_count: Number of newly scored ideas
        
    Returns:
        Patch dict for state update
    """
    pursue, explore, park = _get_verdict_counts(all_scores)
    
    patch = {
        "scored_ideas": all_scores,
        "scorer_attempts": state.scorer_attempts + 1,
        "current_revision_idea_id": None,  # Clear revision flag
        "next_node": "orchestrator",
    }
    
    patch.update(
        state.add_event(
            agent="scorer",
            stage=PipelineStage.SCORING,
            kind="info",
            message=(
                f"Scored {new_scores_count} ideas → "
                f"{pursue} pursue / {explore} explore / {park} park."
            ),
        )
    )
    
    return patch


def _build_no_ideas_patch(state: VentureForgeState) -> dict:
    """Build the patch dict when there are no ideas to score.
    
    Args:
        state: Current pipeline state
        
    Returns:
        Patch dict for state update
    """
    patch = {
        "scored_ideas": [],
        "current_stage": PipelineStage.SCORING,
        "next_node": "orchestrator",
    }
    
    patch.update(
        state.add_event(
            agent="scorer",
            stage=PipelineStage.SCORING,
            kind="warning",
            message=WARNING_SCORER_NO_IDEAS,
        )
    )
    
    return patch


def _build_error_patch(state: VentureForgeState, error_message: str) -> dict:
    """Build the patch dict for an error case.
    
    Args:
        state: Current pipeline state
        error_message: Error message to log
        
    Returns:
        Patch dict for state update
    """
    patch = {
        "scored_ideas": state.scored_ideas,  # Keep existing scores
        "current_stage": PipelineStage.SCORING,
        "next_node": "orchestrator",
    }
    
    patch.update(
        state.add_event(
            agent="scorer",
            stage=PipelineStage.SCORING,
            kind="error",
            message=error_message,
        )
    )
    
    return patch


# ============================================================================
# Main Entry Point
# ============================================================================


def run(state: VentureForgeState) -> dict:
    """Run the Scorer agent to evaluate ideas.
    
    The Scorer evaluates ideas using a binary yes/no rubric across three
    dimensions: feasibility, demand, and novelty. Each idea receives a
    verdict (pursue/explore/park) based on the number of 'yes' answers.
    
    In revision mode, only the revised idea is re-scored, and the new score
    is merged with existing scores.
    
    Args:
        state: Current pipeline state
        
    Returns:
        Patch dict to update state with scored ideas
    """
    # Validate we have ideas to score
    if not state.ideas:
        logger.warning("[scorer] No ideas to score")
        return _build_no_ideas_patch(state)
    
    # Invoke LLM
    try:
        raw_scores = _invoke_llm(state)
    except (LLMError, ValidationError) as e:
        logger.error(f"[scorer] {e}")
        return _build_error_patch(state, str(e))
    
    # Parse scores
    scored_ideas = _parse_all_scores(raw_scores)
    
    if not scored_ideas:
        logger.warning("[scorer] No valid scores parsed from LLM response")
        return _build_error_patch(state, "Failed to parse any valid scores from LLM response")
    
    # Rank and merge scores
    if state.current_revision_idea_id:
        # Revision mode: merge with existing scores
        all_scores = _merge_revision_scores(state, scored_ideas)
    else:
        # Initial mode: rank new scores
        all_scores = _rank_ideas(scored_ideas)
    
    return _build_success_patch(state, all_scores, len(scored_ideas))
