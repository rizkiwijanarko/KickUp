"""
Orchestrator — routes tasks, manages state, handles reflection loop.
Never generates content.

REFACTORED: Following clean code principles with extracted helper functions.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any

from src.constants import (
    MAX_INITIAL_MINING_ATTEMPTS,
    MIN_PAIN_POINTS_FOR_IDEAS,
    MAX_MINING_RETRIES,
    ERROR_NO_PAIN_POINTS,
    ERROR_INSUFFICIENT_PAIN_POINTS,
    ERROR_MAX_IDEA_ATTEMPTS,
    ERROR_NO_SCORED_IDEAS,
)
from src.state.schema import PipelineStage, VentureForgeState

# Import worker agents
from src.agents.pain_point_miner import run as run_pain_point_miner
from src.agents.idea_generator import run as run_idea_generator
from src.agents.scorer import run as run_scorer
from src.agents.pitch_writer import run as run_pitch_writer
from src.agents.critic import run as run_critic

logger = logging.getLogger(__name__)


# =============================================================================
# ROUTING DECISION HELPERS
# =============================================================================


def should_mine_pain_points(state: VentureForgeState) -> bool:
    """Check if we need to mine more pain points."""
    return not state.pain_points


def should_retry_mining(state: VentureForgeState) -> bool:
    """Check if we should retry mining for more pain points."""
    return (
        not state.ideas
        and len(state.filtered_pain_points) < MIN_PAIN_POINTS_FOR_IDEAS
        and state.pain_point_miner_revision_count < MAX_MINING_RETRIES
    )


def should_generate_ideas(state: VentureForgeState) -> bool:
    """Check if we need to generate ideas."""
    return not state.ideas


def should_score_ideas(state: VentureForgeState) -> bool:
    """Check if we need to score ideas."""
    return not state.scored_ideas


def should_write_pitches(state: VentureForgeState) -> bool:
    """Check if we need to write pitch briefs."""
    return not state.pitch_briefs


def should_critique_pitches(state: VentureForgeState) -> bool:
    """Check if we need to critique pitches."""
    return state.pitch_briefs and state.critique is None


def should_revise(state: VentureForgeState) -> bool:
    """Check if revision is needed based on critique."""
    return (
        state.critique is not None
        and not state.critique.all_pass
        and state.can_revise
    )


def has_unscored_ideas(state: VentureForgeState) -> bool:
    """Check if there are ideas that haven't been scored yet."""
    scored_idea_ids = {s.idea_id for s in state.scored_ideas}
    unscored_ideas = [idea for idea in state.ideas if idea.id not in scored_idea_ids]
    return len(unscored_ideas) > 0


def has_unpitched_ideas(state: VentureForgeState) -> bool:
    """Check if there are scored ideas without pitch briefs."""
    if not state.top_scored_ideas:
        return False
    top_ids = {s.idea_id for s in state.top_scored_ideas}
    brief_ids = {b.idea_id for b in state.pitch_briefs}
    return len(top_ids - brief_ids) > 0


def has_more_briefs_to_critique(state: VentureForgeState) -> bool:
    """Check if there are more briefs to critique."""
    return state.current_critique_index + 1 < len(state.pitch_briefs)


def has_reached_max_mining_attempts(state: VentureForgeState) -> bool:
    """Check if we've exceeded max mining attempts."""
    return state.pain_point_miner_revision_count >= MAX_INITIAL_MINING_ATTEMPTS


def has_reached_max_idea_attempts(state: VentureForgeState) -> bool:
    """Check if we've exceeded max idea generation attempts."""
    return state.idea_generation_attempts >= state.max_idea_generation_attempts


def has_reached_global_llm_limit(state: VentureForgeState, agent: str) -> bool:
    """Check if an agent has reached its global LLM call limit."""
    if agent == "idea_generator":
        return state.idea_generation_attempts >= state.max_total_llm_calls_per_agent
    elif agent == "scorer":
        return state.scorer_attempts >= state.max_total_llm_calls_per_agent
    return False


# =============================================================================
# PATCH BUILDERS
# =============================================================================


def build_route_patch(
    stage: PipelineStage,
    next_node: str,
    additional_updates: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build a routing patch for state update.

    Args:
        stage: Target pipeline stage
        next_node: Next agent to execute
        additional_updates: Optional additional fields to update

    Returns:
        Dictionary patch for state update
    """
    patch = {
        "current_stage": stage,
        "next_node": next_node,
    }
    if additional_updates:
        patch.update(additional_updates)
    return patch


def build_failure_patch(
    state: VentureForgeState,
    error_message: str,
    stage: PipelineStage = PipelineStage.FAILED,
) -> Dict[str, Any]:
    """
    Build a failure patch with error event.

    Args:
        state: Current state
        error_message: Error description
        stage: Pipeline stage (default: FAILED)

    Returns:
        Dictionary patch for state update
    """
    patch = state.mark_failed(error_message)
    patch.update(
        state.add_event(
            agent="orchestrator",
            stage=stage,
            kind="error",
            message=error_message,
        )
    )
    return patch


def build_event_patch(
    state: VentureForgeState,
    stage: PipelineStage,
    kind: str,
    message: str,
    additional_updates: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build a patch with an event log entry.

    Args:
        state: Current state
        stage: Pipeline stage
        kind: Event kind (info, warning, error)
        message: Event message
        additional_updates: Optional additional fields to update

    Returns:
        Dictionary patch for state update
    """
    patch = additional_updates or {}
    patch.update(
        state.add_event(
            agent="orchestrator",
            stage=stage,
            kind=kind,
            message=message,
        )
    )
    return patch


# =============================================================================
# STAGE HANDLERS
# =============================================================================


def handle_pain_point_mining(state: VentureForgeState) -> Dict[str, Any]:
    """
    Handle pain point mining stage routing.

    Returns:
        State patch dictionary
    """
    # Circuit breaker: prevent infinite loops
    if has_reached_max_mining_attempts(state):
        error_msg = ERROR_NO_PAIN_POINTS.format(
            attempts=MAX_INITIAL_MINING_ATTEMPTS,
            domain=state.domain,
        )
        logger.error(f"[orchestrator] {error_msg}")
        return build_failure_patch(state, error_msg)

    # Route to pain point miner
    patch = build_route_patch(
        stage=PipelineStage.MINING,
        next_node="pain_point_miner",
        additional_updates={
            "pain_point_miner_revision_count": state.pain_point_miner_revision_count + 1,
        },
    )

    attempt_num = state.pain_point_miner_revision_count + 1
    message = (
        f"Routing to pain_point_miner (no pain points yet, "
        f"attempt {attempt_num}/{MAX_INITIAL_MINING_ATTEMPTS})"
    )

    patch.update(
        build_event_patch(
            state=state,
            stage=PipelineStage.MINING,
            kind="info",
            message=message,
        )
    )

    return patch


def handle_mining_retry(state: VentureForgeState) -> Dict[str, Any]:
    """
    Handle retry logic for insufficient pain points.

    Returns:
        State patch dictionary
    """
    patch = build_route_patch(
        stage=PipelineStage.MINING,
        next_node="pain_point_miner",
        additional_updates={
            "pain_point_miner_revision_count": state.pain_point_miner_revision_count + 1,
        },
    )

    attempt_num = state.pain_point_miner_revision_count + 1
    message = ERROR_INSUFFICIENT_PAIN_POINTS.format(
        count=len(state.filtered_pain_points),
        target=MIN_PAIN_POINTS_FOR_IDEAS,
        attempt=attempt_num,
        max_attempts=MAX_MINING_RETRIES,
    )

    patch.update(
        build_event_patch(
            state=state,
            stage=PipelineStage.MINING,
            kind="warning",
            message=message,
        )
    )

    return patch


def handle_idea_generation(state: VentureForgeState) -> Dict[str, Any]:
    """
    Handle idea generation stage routing.

    Returns:
        State patch dictionary
    """
    # Check global LLM call limit
    if has_reached_global_llm_limit(state, "idea_generator"):
        error_msg = (
            f"Reached global LLM call limit ({state.max_total_llm_calls_per_agent}) "
            f"for idea_generator. Check logs for root cause."
        )
        logger.error(f"[orchestrator] {error_msg}")
        return build_failure_patch(state, error_msg)

    # Check per-run validation retry limit
    if has_reached_max_idea_attempts(state):
        error_msg = ERROR_MAX_IDEA_ATTEMPTS.format(
            attempts=state.idea_generation_attempts,
            count=len(state.ideas),
        )
        logger.error(f"[orchestrator] {error_msg}")
        return build_failure_patch(state, error_msg)

    # Route to idea generator
    patch = build_route_patch(
        stage=PipelineStage.GENERATING,
        next_node="idea_generator",
    )

    attempt_num = state.idea_generation_attempts + 1
    message = (
        f"Routing to idea_generator (no ideas yet, "
        f"attempt {attempt_num}/{state.max_idea_generation_attempts}, "
        f"global {attempt_num}/{state.max_total_llm_calls_per_agent})"
    )

    patch.update(
        build_event_patch(
            state=state,
            stage=PipelineStage.GENERATING,
            kind="info",
            message=message,
        )
    )

    return patch


def handle_scoring(state: VentureForgeState) -> Dict[str, Any]:
    """
    Handle scoring stage routing.

    Returns:
        State patch dictionary
    """
    # Circuit breaker for scorer failures
    if has_reached_global_llm_limit(state, "scorer"):
        error_msg = (
            f"Reached global LLM call limit ({state.max_total_llm_calls_per_agent}) "
            f"for scorer. Check logs for JSON extraction failures."
        )
        logger.error(f"[orchestrator] {error_msg}")
        return build_failure_patch(state, error_msg)

    # Route to scorer
    patch = build_route_patch(
        stage=PipelineStage.SCORING,
        next_node="scorer",
    )

    attempt_num = state.scorer_attempts + 1
    message = (
        f"Routing to scorer (no scored ideas yet, "
        f"attempt {attempt_num}/{state.max_total_llm_calls_per_agent})"
    )

    patch.update(
        build_event_patch(
            state=state,
            stage=PipelineStage.SCORING,
            kind="info",
            message=message,
        )
    )

    return patch


def handle_pitch_writing(state: VentureForgeState) -> Dict[str, Any]:
    """
    Handle pitch writing stage routing.

    Returns:
        State patch dictionary
    """
    # Quality gate: check if we generated enough ideas before checking verdicts
    MIN_IDEAS_THRESHOLD = max(2, state.ideas_per_run // 2)  # At least half of requested ideas
    if len(state.ideas) < MIN_IDEAS_THRESHOLD:
        # Check circuit breaker: prevent infinite loops when idea generator consistently fails
        logger.info(
            f"[orchestrator] Insufficient ideas check: {len(state.ideas)} < {MIN_IDEAS_THRESHOLD}. "
            f"Attempts: {state.idea_generation_attempts}/{state.max_idea_generation_attempts}"
        )
        
        if state.idea_generation_attempts >= state.max_idea_generation_attempts:
            # Fail if we've exhausted retries and still have insufficient ideas
            error_msg = (
                f"Failed to generate sufficient ideas after {state.idea_generation_attempts} attempts. "
                f"Only {len(state.ideas)} ideas generated (minimum: {MIN_IDEAS_THRESHOLD}). "
                "This usually means the LLM is not producing ideas with valid pain_point_ids. "
                "Check logs for validation failures."
            )
            logger.error(f"[orchestrator] Circuit breaker triggered: {error_msg}")
            return build_failure_patch(state, error_msg)
        
        if state.idea_generation_attempts < state.max_idea_generation_attempts:
            # Retry idea generation to get more candidates
            logger.info(
                f"[orchestrator] Retrying idea generation (attempt {state.idea_generation_attempts + 1}/{state.max_idea_generation_attempts})"
            )
            patch = build_route_patch(
                stage=PipelineStage.GENERATING,
                next_node="idea_generator",
            )
            patch.update(
                build_event_patch(
                    state=state,
                    stage=PipelineStage.GENERATING,
                    kind="warning",
                    message=(
                        f"Only {len(state.ideas)} ideas generated (target: {state.ideas_per_run}, "
                        f"minimum: {MIN_IDEAS_THRESHOLD}). Retrying idea generation "
                        f"(attempt {state.idea_generation_attempts + 1}/{state.max_idea_generation_attempts})."
                    ),
                )
            )
            return patch
    
    # Check if there are unscored ideas (e.g., after retry)
    scored_idea_ids = {s.idea_id for s in state.scored_ideas}
    unscored_ideas = [idea for idea in state.ideas if idea.id not in scored_idea_ids]
    if unscored_ideas:
        # Route back to scorer for new ideas
        patch = build_route_patch(
            stage=PipelineStage.SCORING,
            next_node="scorer",
        )
        patch.update(
            build_event_patch(
                state=state,
                stage=PipelineStage.SCORING,
                kind="info",
                message=f"Found {len(unscored_ideas)} unscored ideas (after retry). Routing to scorer.",
            )
        )
        return patch
    
    # Log verdict distribution
    if state.top_scored_ideas:
        verdict_counts = {
            "pursue": sum(1 for s in state.top_scored_ideas if s.verdict == "pursue"),
            "explore": sum(1 for s in state.top_scored_ideas if s.verdict == "explore"),
            "park": sum(1 for s in state.top_scored_ideas if s.verdict == "park"),
        }
        logger.info(
            f"[orchestrator] Top {len(state.top_scored_ideas)} ideas verdict distribution: "
            f"pursue={verdict_counts['pursue']}, explore={verdict_counts['explore']}, "
            f"park={verdict_counts['park']}"
        )

        # Warn if all ideas are "park"
        if all(s.verdict == "park" for s in state.top_scored_ideas):
            warning_msg = (
                f"All {len(state.top_scored_ideas)} top-scored ideas received 'park' verdict. "
                f"Generating pitch briefs for documentation."
            )
            logger.warning(f"[orchestrator] {warning_msg}")

    # Circuit breaker for pitch writer failures
    if state.pitch_writer_attempts >= state.max_total_llm_calls_per_agent:
        error_msg = (
            f"Reached global LLM call limit ({state.max_total_llm_calls_per_agent}) "
            f"for pitch_writer. Check logs for JSON extraction failures."
        )
        logger.error(f"[orchestrator] {error_msg}")
        return build_failure_patch(state, error_msg)

    # Route to pitch writer
    patch = build_route_patch(
        stage=PipelineStage.WRITING,
        next_node="pitch_writer",
    )

    attempt_num = state.pitch_writer_attempts + 1
    message = (
        f"Routing to pitch_writer (no pitch briefs yet, "
        f"attempt {attempt_num}/{state.max_total_llm_calls_per_agent})"
    )

    patch.update(
        build_event_patch(
            state=state,
            stage=PipelineStage.WRITING,
            kind="info",
            message=message,
        )
    )

    return patch


def handle_critique(state: VentureForgeState) -> Dict[str, Any]:
    """
    Handle critique stage routing.

    Returns:
        State patch dictionary
    """
    # Validate pitch briefs match top scored ideas
    top_ids = {s.idea_id for s in state.top_scored_ideas}
    brief_ids = {b.idea_id for b in state.pitch_briefs}

    if not brief_ids.issubset(top_ids):
        error_msg = (
            f"Pitch briefs contain ideas not in top_scored_ideas. "
            f"This indicates a bug in pitch_writer or scorer."
        )
        logger.error(f"[orchestrator] {error_msg}")
        return build_failure_patch(state, error_msg)

    # Route to critic
    patch = build_route_patch(
        stage=PipelineStage.CRITIQUING,
        next_node="critic",
    )

    brief_num = state.current_critique_index + 1
    total_briefs = len(state.pitch_briefs)
    message = f"Routing to critic (reviewing brief {brief_num}/{total_briefs})"

    patch.update(
        build_event_patch(
            state=state,
            stage=PipelineStage.CRITIQUING,
            kind="info",
            message=message,
        )
    )

    return patch


def handle_revision(state: VentureForgeState) -> Dict[str, Any]:
    """
    Handle revision loop routing.

    Returns:
        State patch dictionary
    """
    target = state.critique.target_agent

    # Check global LLM limits for target agent
    if target == "idea_generator":
        if state.idea_generation_attempts >= state.max_total_llm_calls_per_agent:
            error_msg = (
                f"Reached global LLM call limit ({state.max_total_llm_calls_per_agent}) "
                f"for idea_generator during revision loop."
            )
            logger.error(f"[orchestrator] {error_msg}")
            return build_failure_patch(state, error_msg)

    elif target == "pitch_writer":
        if state.pitch_writer_attempts >= state.max_total_llm_calls_per_agent:
            error_msg = (
                f"Reached global LLM call limit ({state.max_total_llm_calls_per_agent}) "
                f"for pitch_writer during revision loop."
            )
            logger.error(f"[orchestrator] {error_msg}")
            return build_failure_patch(state, error_msg)

    # Build revision patch
    patch = state.bump_revision(state.critique)
    patch.update(state.reset_for_revision(target, state.critique.idea_id))

    message = (
        f"Revision requested by critic for idea {state.critique.idea_id} "
        f"→ target_agent={target}"
    )

    patch.update(
        state.add_event(
            agent="orchestrator",
            stage=PipelineStage.REVISING,
            kind="info",
            message=message,
            idea_id=state.critique.idea_id,
        )
    )

    return patch


def handle_next_brief(state: VentureForgeState) -> Dict[str, Any]:
    """
    Handle moving to the next brief for critique.

    Returns:
        State patch dictionary
    """
    patch = {
        "current_critique_index": state.current_critique_index + 1,
        "critique": None,
        "revision_feedback": None,
        "current_stage": PipelineStage.CRITIQUING,
        "next_node": "critic",
    }

    # Determine message based on approval status
    if state.critique.approval_status == "max_revisions_reached":
        message = (
            f"Brief {state.current_critique_index + 1} reached max revisions "
            f"(still has {len(state.critique.failing_checks)} failing checks). "
            f"Moving to brief {state.current_critique_index + 2}/{len(state.pitch_briefs)}"
        )
        kind = "warning"
    else:
        message = (
            f"Brief {state.current_critique_index + 1} approved. "
            f"Moving to brief {state.current_critique_index + 2}/{len(state.pitch_briefs)}"
        )
        kind = "info"

    patch.update(
        build_event_patch(
            state=state,
            stage=PipelineStage.CRITIQUING,
            kind=kind,
            message=message,
        )
    )

    return patch


def handle_completion(state: VentureForgeState) -> Dict[str, Any]:
    """
    Handle pipeline completion.

    Returns:
        State patch dictionary
    """
    # Check for briefs that reached max revisions
    max_revision_briefs = [
        c for c in state.critiques
        if c.approval_status == "max_revisions_reached"
    ]

    if max_revision_briefs:
        summary = (
            f"Pipeline completed with {len(state.pain_points)} pain points, "
            f"{len(state.ideas)} ideas, {len(state.scored_ideas)} scored ideas, "
            f"and {len(state.pitch_briefs)} pitch briefs. "
            f"⚠️ WARNING: {len(max_revision_briefs)} brief(s) reached max revisions "
            f"with unresolved quality issues."
        )
        kind = "warning"
    else:
        summary = (
            f"Pipeline completed with {len(state.pain_points)} pain points, "
            f"{len(state.ideas)} ideas, {len(state.scored_ideas)} scored ideas, "
            f"and {len(state.pitch_briefs)} pitch briefs (all approved)."
        )
        kind = "info"

    patch = state.mark_completed()
    patch["revision_feedback"] = None

    patch.update(
        state.add_event(
            agent="orchestrator",
            stage=PipelineStage.COMPLETED,
            kind=kind,
            message=summary,
        )
    )

    return patch


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================


def orchestrator(state: VentureForgeState) -> Dict[str, Any]:
    """
    Supervisor node. Based on pipeline progress, decides which worker to run next.

    Args:
        state: Current pipeline state

    Returns:
        Dictionary patch for state update
    """
    logger.info(
        f"[orchestrator] Called with: ideas={len(state.ideas)}, "
        f"scored={len(state.scored_ideas)}, briefs={len(state.pitch_briefs)}, "
        f"attempts={state.idea_generation_attempts}/{state.max_idea_generation_attempts}"
    )

    # Stage 1: Pain Point Mining
    if should_mine_pain_points(state):
        return handle_pain_point_mining(state)

    # Stage 2: Mining Retry (quality gate)
    if should_retry_mining(state):
        return handle_mining_retry(state)

    # Stage 3: Idea Generation
    if should_generate_ideas(state):
        return handle_idea_generation(state)

    # Stage 4: Scoring
    if should_score_ideas(state):
        return handle_scoring(state)

    # Stage 5: Pitch Writing
    if should_write_pitches(state):
        return handle_pitch_writing(state)

    # Stage 6: Critique
    if should_critique_pitches(state):
        return handle_critique(state)

    # Stage 7: Revision Loop
    if should_revise(state):
        return handle_revision(state)

    # Post-revision: Check for unscored ideas
    if has_unscored_ideas(state):
        logger.info(
            f"[orchestrator] Found unscored ideas after revision. Routing to scorer."
        )
        return handle_scoring(state)

    # Post-revision: Check for unpitched ideas
    if has_unpitched_ideas(state):
        logger.info(
            f"[orchestrator] Found unpitched ideas after revision. Routing to pitch_writer."
        )
        return handle_pitch_writing(state)

    # Stage 8: Move to Next Brief
    if has_more_briefs_to_critique(state):
        return handle_next_brief(state)

    # Stage 9: Completion
    logger.info("[orchestrator] All stages completed")
    return handle_completion(state)


# ============================================================================
# Worker Wrapper Nodes (LangGraph Entry Points)
# ============================================================================


def pain_point_miner(state: VentureForgeState) -> dict:
    """Wrapper for pain_point_miner agent with timing."""
    t0 = time.monotonic()
    result = run_pain_point_miner(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("pain_point_miner", elapsed)}


def idea_generator(state: VentureForgeState) -> dict:
    """Wrapper for idea_generator agent with timing."""
    t0 = time.monotonic()
    result = run_idea_generator(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("idea_generator", elapsed)}


def scorer(state: VentureForgeState) -> dict:
    """Wrapper for scorer agent with timing."""
    t0 = time.monotonic()
    result = run_scorer(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("scorer", elapsed)}


def pitch_writer(state: VentureForgeState) -> dict:
    """Wrapper for pitch_writer agent with timing."""
    t0 = time.monotonic()
    result = run_pitch_writer(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("pitch_writer", elapsed)}


def critic(state: VentureForgeState) -> dict:
    """Wrapper for critic agent with timing."""
    t0 = time.monotonic()
    result = run_critic(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("critic", elapsed)}
