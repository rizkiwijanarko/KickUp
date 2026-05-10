"""Orchestrator — routes tasks, manages state, handles reflection loop. Never generates content."""
from __future__ import annotations

import logging
import time

from src.state.schema import PipelineStage, VentureForgeState

# Import worker agents (stubs; real implementations in agent files)
from src.agents.pain_point_miner import run as run_pain_point_miner
from src.agents.idea_generator import run as run_idea_generator
from src.agents.scorer import run as run_scorer
from src.agents.pitch_writer import run as run_pitch_writer
from src.agents.critic import run as run_critic

logger = logging.getLogger(__name__)


def orchestrator(state: VentureForgeState) -> dict:
    """
    Supervisor node. Based on pipeline progress, decides which worker to run next.
    Returns a dict patch for state update.
    """
    logger.info(
        f"[orchestrator] Called with: ideas={len(state.ideas)}, scored={len(state.scored_ideas)}, "
        f"briefs={len(state.pitch_briefs)}, attempts={state.idea_generation_attempts}/{state.max_idea_generation_attempts}"
    )
    stage = state.current_stage

    # --- Determine next stage ---
    if not state.pain_points:
        # Circuit breaker: prevent infinite loops when pain_point_miner keeps returning 0 pain points
        MAX_INITIAL_MINING_ATTEMPTS = 5
        if state.pain_point_miner_revision_count >= MAX_INITIAL_MINING_ATTEMPTS:
            error_msg = (
                f"Reached max initial mining attempts ({MAX_INITIAL_MINING_ATTEMPTS}) with 0 pain points. "
                f"This usually means: (1) LLM is failing to extract pain points from scraped content, "
                f"(2) All extracted pain points are failing validation (no verbatim quotes), or "
                f"(3) Domain '{state.domain}' has insufficient community discussion. "
                f"Try a different domain or check LLM logs for extraction failures."
            )
            patch = state.mark_failed(error_msg)
            patch.update(
                state.add_event(
                    agent="orchestrator",
                    stage=PipelineStage.FAILED,
                    kind="error",
                    message=error_msg,
                )
            )
            return patch
        
        patch = {
            "current_stage": PipelineStage.MINING,
            "next_node": "pain_point_miner",
            "pain_point_miner_revision_count": state.pain_point_miner_revision_count + 1,
        }
        patch.update(
            state.add_event(
                agent="orchestrator",
                stage=PipelineStage.MINING,
                kind="info",
                message=f"Routing to pain_point_miner (no pain points yet, attempt {state.pain_point_miner_revision_count + 1}/{MAX_INITIAL_MINING_ATTEMPTS}).",
            )
        )
        return patch

    # --- Quality gate: ensure sufficient pain points before idea generation ---
    MIN_PAIN_POINTS_FOR_IDEAS = 2
    MAX_MINING_RETRIES = 2
    
    if not state.ideas:
        # Quality gate: check if we have enough pain points for robust idea generation
        if len(state.filtered_pain_points) < MIN_PAIN_POINTS_FOR_IDEAS:
            if state.pain_point_miner_revision_count < MAX_MINING_RETRIES:
                # Retry mining to collect more pain points
                patch = {
                    "current_stage": PipelineStage.MINING,
                    "next_node": "pain_point_miner",
                    "pain_point_miner_revision_count": state.pain_point_miner_revision_count + 1,
                }
                patch.update(
                    state.add_event(
                        agent="orchestrator",
                        stage=PipelineStage.MINING,
                        kind="warning",
                        message=(
                            f"Only {len(state.filtered_pain_points)} pain points found "
                            f"(target: {MIN_PAIN_POINTS_FOR_IDEAS}). Retrying mining "
                            f"(attempt {state.pain_point_miner_revision_count + 1}/{MAX_MINING_RETRIES})."
                        ),
                    )
                )
                return patch
            else:
                # Proceed with degraded quality after max retries
                patch = {
                    "current_stage": PipelineStage.GENERATING,
                    "next_node": "idea_generator",
                }
                patch.update(
                    state.add_event(
                        agent="orchestrator",
                        stage=PipelineStage.GENERATING,
                        kind="warning",
                        message=(
                            f"Proceeding with only {len(state.filtered_pain_points)} pain points "
                            f"after {MAX_MINING_RETRIES} mining attempts. Idea quality may be lower."
                        ),
                    )
                )
                return patch
        
        # Check global cap first (prevents compounding validation + revision retries)
        if state.idea_generation_attempts >= state.max_total_llm_calls_per_agent:
            error_msg = (
                f"Reached global LLM call limit ({state.max_total_llm_calls_per_agent}) for idea_generator. "
                f"This prevents excessive retries from validation failures + Critic revisions. "
                f"Check logs for root cause (invalid pain_point_ids, schema mismatches, etc.)."
            )
            patch = state.mark_failed(error_msg)
            patch.update(
                state.add_event(
                    agent="orchestrator",
                    stage=PipelineStage.FAILED,
                    kind="error",
                    message=error_msg,
                )
            )
            return patch
        
        # Check per-run validation retry limit
        if state.idea_generation_attempts >= state.max_idea_generation_attempts:
            error_msg = (
                f"Failed to generate valid ideas after {state.idea_generation_attempts} attempts. "
                "This usually means the LLM is not producing ideas with valid pain_point_ids. "
                "Check logs for validation failures."
            )
            patch = state.mark_failed(error_msg)
            patch.update(
                state.add_event(
                    agent="orchestrator",
                    stage=PipelineStage.FAILED,
                    kind="error",
                    message=error_msg,
                )
            )
            return patch
        
        patch = {
            "current_stage": PipelineStage.GENERATING,
            "next_node": "idea_generator",
        }
        patch.update(
            state.add_event(
                agent="orchestrator",
                stage=PipelineStage.GENERATING,
                kind="info",
                message=f"Routing to idea_generator (no ideas yet, attempt {state.idea_generation_attempts + 1}/{state.max_idea_generation_attempts}, global {state.idea_generation_attempts + 1}/{state.max_total_llm_calls_per_agent}).",
            )
        )
        return patch

    if not state.scored_ideas:
        # Circuit breaker: prevent infinite loops when scorer fails to generate valid output
        if state.scorer_attempts >= state.max_total_llm_calls_per_agent:
            error_msg = (
                f"Reached global LLM call limit ({state.max_total_llm_calls_per_agent}) for scorer. "
                f"This usually means the LLM is failing to generate valid JSON (truncation or parsing errors). "
                f"Check logs for 'JSON extraction failed' or 'Response may be truncated' warnings."
            )
            patch = state.mark_failed(error_msg)
            patch.update(
                state.add_event(
                    agent="orchestrator",
                    stage=PipelineStage.FAILED,
                    kind="error",
                    message=error_msg,
                )
            )
            return patch
        
        patch = {"current_stage": PipelineStage.SCORING, "next_node": "scorer"}
        patch.update(
            state.add_event(
                agent="orchestrator",
                stage=PipelineStage.SCORING,
                kind="info",
                message=f"Routing to scorer (no scored ideas yet, attempt {state.scorer_attempts + 1}/{state.max_total_llm_calls_per_agent}).",
            )
        )
        return patch

    if not state.pitch_briefs:
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
                patch = state.mark_failed(error_msg)
                patch.update(
                    state.add_event(
                        agent="orchestrator",
                        stage=PipelineStage.FAILED,
                        kind="error",
                        message=error_msg,
                    )
                )
                return patch
            
            if state.idea_generation_attempts < state.max_idea_generation_attempts:
                # Retry idea generation to get more candidates
                logger.info(
                    f"[orchestrator] Retrying idea generation (attempt {state.idea_generation_attempts + 1}/{state.max_idea_generation_attempts})"
                )
                patch = {
                    "current_stage": PipelineStage.GENERATING,
                    "next_node": "idea_generator",
                }
                patch.update(
                    state.add_event(
                        agent="orchestrator",
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
            patch = {"current_stage": PipelineStage.SCORING, "next_node": "scorer"}
            patch.update(
                state.add_event(
                    agent="orchestrator",
                    stage=PipelineStage.SCORING,
                    kind="info",
                    message=f"Found {len(unscored_ideas)} unscored ideas (after retry). Routing to scorer.",
                )
            )
            return patch
        
        # Log verdict distribution for visibility
        if state.top_scored_ideas:
            verdict_counts = {
                "pursue": sum(1 for s in state.top_scored_ideas if s.verdict == "pursue"),
                "explore": sum(1 for s in state.top_scored_ideas if s.verdict == "explore"),
                "park": sum(1 for s in state.top_scored_ideas if s.verdict == "park"),
            }
            logger.info(
                f"[orchestrator] Top {len(state.top_scored_ideas)} ideas verdict distribution: "
                f"pursue={verdict_counts['pursue']}, explore={verdict_counts['explore']}, park={verdict_counts['park']}"
            )
            
            # If all ideas are "park", log a warning but continue to generate pitch briefs
            # "Park" means "interesting but has concerns" - still worth documenting
            if all(s.verdict == "park" for s in state.top_scored_ideas):
                patch = {}
                patch.update(
                    state.add_event(
                        agent="orchestrator",
                        stage=PipelineStage.WRITING,
                        kind="warning",
                        message=(
                            f"All {len(state.top_scored_ideas)} top-scored ideas received 'park' verdict "
                            f"(interesting but has concerns). Generating pitch briefs for documentation. "
                            f"Consider: (1) adjusting domain for better pain points, "
                            f"(2) increasing ideas_per_run for more candidates, or "
                            f"(3) reviewing scorer rubric if verdicts seem too harsh."
                        ),
                    )
                )
                # Don't return here - continue to pitch_writer
        
        # Circuit breaker: prevent infinite loops when pitch_writer fails to generate briefs
        if state.pitch_writer_attempts >= state.max_total_llm_calls_per_agent:
            error_msg = (
                f"Reached global LLM call limit ({state.max_total_llm_calls_per_agent}) for pitch_writer. "
                f"This usually means the LLM is failing to generate valid JSON (truncation or parsing errors). "
                f"Check logs for 'JSON extraction failed' or 'Response may be truncated' warnings."
            )
            patch = state.mark_failed(error_msg)
            patch.update(
                state.add_event(
                    agent="orchestrator",
                    stage=PipelineStage.FAILED,
                    kind="error",
                    message=error_msg,
                )
            )
            return patch
        
        patch = {"current_stage": PipelineStage.WRITING, "next_node": "pitch_writer"}
        patch.update(
            state.add_event(
                agent="orchestrator",
                stage=PipelineStage.WRITING,
                kind="info",
                message=f"Routing to pitch_writer (no pitch briefs yet, attempt {state.pitch_writer_attempts + 1}/{state.max_total_llm_calls_per_agent}).",
            )
        )
        return patch

    # We have pitch_briefs. Now check if we need to critique them.
    if state.critique is None:
        # ✅ MAJOR FIX #6: Validate pitch_briefs match top_scored_ideas
        top_ids = {s.idea_id for s in state.top_scored_ideas}
        brief_ids = {b.idea_id for b in state.pitch_briefs}
        
        if not brief_ids.issubset(top_ids):
            error_msg = (
                f"Pitch briefs contain ideas not in top_scored_ideas. "
                f"Brief IDs: {brief_ids}, Top IDs: {top_ids}. "
                f"This indicates a bug in pitch_writer or scorer."
            )
            patch = state.mark_failed(error_msg)
            patch.update(
                state.add_event(
                    agent="orchestrator",
                    stage=PipelineStage.FAILED,
                    kind="error",
                    message=error_msg,
                )
            )
            return patch
        
        # First critique - start with index 0
        patch = {"current_stage": PipelineStage.CRITIQUING, "next_node": "critic"}
        patch.update(
            state.add_event(
                agent="orchestrator",
                stage=PipelineStage.CRITIQUING,
                kind="info",
                message=f"Routing to critic (reviewing brief {state.current_critique_index + 1}/{len(state.pitch_briefs)}).",
            )
        )
        return patch

    # --- Reflection loop: we have a critique ---
    if not state.critique.all_pass and state.can_revise:
        # Loop back to target worker for revision
        target = state.critique.target_agent
        
        # ✅ CRITICAL FIX #3: Enforce global LLM call limits during revision
        if target == "idea_generator":
            if state.idea_generation_attempts >= state.max_total_llm_calls_per_agent:
                error_msg = (
                    f"Reached global LLM call limit ({state.max_total_llm_calls_per_agent}) "
                    f"for idea_generator during revision loop. "
                    f"Check logs for validation failures or infinite revision loops."
                )
                patch = state.mark_failed(error_msg)
                patch.update(
                    state.add_event(
                        agent="orchestrator",
                        stage=PipelineStage.FAILED,
                        kind="error",
                        message=error_msg,
                    )
                )
                return patch
        
        elif target == "pitch_writer":
            if state.pitch_writer_attempts >= state.max_total_llm_calls_per_agent:
                error_msg = (
                    f"Reached global LLM call limit ({state.max_total_llm_calls_per_agent}) "
                    f"for pitch_writer during revision loop. "
                    f"Check logs for JSON parsing failures or infinite revision loops."
                )
                patch = state.mark_failed(error_msg)
                patch.update(
                    state.add_event(
                        agent="orchestrator",
                        stage=PipelineStage.FAILED,
                        kind="error",
                        message=error_msg,
                    )
                )
                return patch
        
        # Continue with revision
        patch = state.bump_revision(state.critique)
        patch.update(state.reset_for_revision(target, state.critique.idea_id))
        patch.update(
            state.add_event(
                agent="orchestrator",
                stage=PipelineStage.REVISING,
                kind="info",
                message=(
                    f"Revision requested by critic for idea {state.critique.idea_id} "
                    f"→ target_agent={target}."
                ),
                idea_id=state.critique.idea_id,
            )
        )
        return patch
    
    # ✅ POST-REVISION PIPELINE COMPLETION CHECK
    # After a revision, we may have new ideas that need scoring/pitching before returning to Critic
    
    # Check for unscored ideas (after idea_generator revision)
    scored_idea_ids = {s.idea_id for s in state.scored_ideas}
    unscored_ideas = [idea for idea in state.ideas if idea.id not in scored_idea_ids]
    
    if unscored_ideas:
        logger.info(
            f"[orchestrator] Found {len(unscored_ideas)} unscored ideas after revision. "
            f"Routing to scorer before returning to critic."
        )
        patch = {"current_stage": PipelineStage.SCORING, "next_node": "scorer"}
        patch.update(
            state.add_event(
                agent="orchestrator",
                stage=PipelineStage.SCORING,
                kind="info",
                message=f"Found {len(unscored_ideas)} unscored ideas after revision. Routing to scorer.",
            )
        )
        return patch
    
    # Check for unpitched scored ideas (after scorer revision or idea_generator → scorer)
    if state.top_scored_ideas:
        top_ids = {s.idea_id for s in state.top_scored_ideas}
        brief_ids = {b.idea_id for b in state.pitch_briefs}
        missing_brief_ids = top_ids - brief_ids
        
        if missing_brief_ids:
            logger.info(
                f"[orchestrator] Found {len(missing_brief_ids)} ideas without pitch briefs after revision. "
                f"Routing to pitch_writer before returning to critic."
            )
            patch = {"current_stage": PipelineStage.WRITING, "next_node": "pitch_writer"}
            patch.update(
                state.add_event(
                    agent="orchestrator",
                    stage=PipelineStage.WRITING,
                    kind="info",
                    message=f"Found {len(missing_brief_ids)} ideas without pitch briefs after revision. Routing to pitch_writer.",
                )
            )
            return patch
    
    # All ideas are scored and pitched - continue with critique workflow
    
    # --- Critique passed or max revisions reached ---
    # Check if there are more briefs to critique
    if state.current_critique_index + 1 < len(state.pitch_briefs):
        # Move to next brief
        patch = {
            "current_critique_index": state.current_critique_index + 1,
            "critique": None,  # Clear critique for next brief
            "revision_feedback": None,  # Clear stale feedback
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "critic",
        }
        
        # Log different messages based on approval status
        if state.critique.approval_status == "max_revisions_reached":
            message = (
                f"Brief {state.current_critique_index + 1} reached max revisions "
                f"(still has {len(state.critique.failing_checks)} failing checks). "
                f"Moving to brief {state.current_critique_index + 2}/{len(state.pitch_briefs)}."
            )
        else:
            message = f"Brief {state.current_critique_index + 1} approved. Moving to brief {state.current_critique_index + 2}/{len(state.pitch_briefs)}."
        
        patch.update(
            state.add_event(
                agent="orchestrator",
                stage=PipelineStage.CRITIQUING,
                kind="warning" if state.critique.approval_status == "max_revisions_reached" else "info",
                message=message,
            )
        )
        return patch

    # --- All briefs critiqued ---
    # Check if any briefs were forced through at max revisions
    max_revision_briefs = [
        c for c in state.critiques 
        if c.approval_status == "max_revisions_reached"
    ]
    
    if max_revision_briefs:
        summary = (
            f"Pipeline completed with {len(state.pain_points)} pain points, "
            f"{len(state.ideas)} ideas, {len(state.scored_ideas)} scored ideas, "
            f"and {len(state.pitch_briefs)} pitch briefs. "
            f"⚠️ WARNING: {len(max_revision_briefs)} brief(s) reached max revisions with unresolved quality issues."
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
    patch["revision_feedback"] = None  # Clear stale feedback
    patch.update(
        state.add_event(
            agent="orchestrator",
            stage=PipelineStage.COMPLETED,
            kind=kind,
            message=summary,
        )
    )
    return patch


# Worker wrapper nodes (LangGraph calls these, they call the agent logic)

def pain_point_miner(state: VentureForgeState) -> dict:
    t0 = time.monotonic()
    result = run_pain_point_miner(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("pain_point_miner", elapsed)}


def idea_generator(state: VentureForgeState) -> dict:
    t0 = time.monotonic()
    result = run_idea_generator(state)
    elapsed = time.monotonic() - t0
    # After generating, always clear scored_ideas/pitch_briefs in case of revision
    patch = {**result, **state.record_timing("idea_generator", elapsed)}
    return patch


def scorer(state: VentureForgeState) -> dict:
    t0 = time.monotonic()
    result = run_scorer(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("scorer", elapsed)}


def pitch_writer(state: VentureForgeState) -> dict:
    t0 = time.monotonic()
    result = run_pitch_writer(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("pitch_writer", elapsed)}


def critic(state: VentureForgeState) -> dict:
    t0 = time.monotonic()
    result = run_critic(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("critic", elapsed)}
