"""
VentureForge Routing Policy
============================
All pipeline routing decisions live here.

Interface (the seam callers cross):
    route(state) -> dict[str, Any]

Implementation (private to this module):
    seven _should_* predicate functions + the ordered decision tree
    inside route().

The orchestrator node in orchestrator.py is a thin adapter that
calls route() and merges the timing patch.  Nothing else in the
codebase needs to know which predicate fired.

Glossary:
    module   – this file; interface is route(), implementation is everything else
    seam     – the boundary at route(); the orchestrator crosses it
    depth    – all routing decisions are buried here; one name for callers
    locality – routing bugs concentrate here, not spread across orchestrator
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.constants import (
    ERROR_NO_PAIN_POINTS,
    ERROR_NO_SCORED_IDEAS,
    MAX_INITIAL_MINING_ATTEMPTS,
    MAX_MINING_RETRIES,
    MIN_PAIN_POINTS_FOR_IDEAS,
)
from src.models import PipelineStage
from src.state.graph_state import VentureForgeState

logger = logging.getLogger(__name__)


# =============================================================================
# Predicates — private implementation, never called by external code
# =============================================================================


def _should_mine(state: VentureForgeState) -> bool:
    """Initial mining run — before any retries or ideas exist."""
    return not state.pain_points and state.pain_point_miner_revision_count == 0


def _should_retry_mining(state: VentureForgeState) -> bool:
    return (
        not state.ideas
        and len(state.filtered_pain_points) < MIN_PAIN_POINTS_FOR_IDEAS
        and state.pain_point_miner_revision_count < min(
            MAX_MINING_RETRIES, MAX_INITIAL_MINING_ATTEMPTS - 1
        )
    )


def _should_generate_ideas(state: VentureForgeState) -> bool:
    return not state.ideas


def _should_score_ideas(state: VentureForgeState) -> bool:
    return not state.scored_ideas or len({s.idea_id for s in state.scored_ideas}) < len(state.ideas)


def _should_write_pitches(state: VentureForgeState) -> bool:
    if not state.top_scored_ideas:
        return False
    top_ids = {s.idea_id for s in state.top_scored_ideas}
    brief_ids = {b.idea_id for b in state.pitch_briefs}
    return bool(top_ids - brief_ids)


def _should_critique(state: VentureForgeState) -> bool:
    if not state.pitch_briefs:
        return False
    pending = state.revisions.pending_briefs(state.pitch_briefs)
    if not pending:
        return False
    if state.critique is None:
        return True
    return state.critique.idea_id not in {b.idea_id for b in pending}


def _should_revise(state: VentureForgeState) -> bool:
    return bool(state.critique and not state.critique.all_pass and state.can_revise)


# =============================================================================
# Public interface
# =============================================================================


def route(state: VentureForgeState) -> dict[str, Any]:
    """Return the full state patch for the next pipeline step.

    This is the only name external code calls.  All predicate logic
    is implementation; callers never need to know which predicate fired
    or in what order they are evaluated.
    """
    logger.info(
        "[routing] pain_points=%d, ideas=%d, scored=%d, briefs=%d",
        len(state.pain_points),
        len(state.ideas),
        len(state.scored_ideas),
        len(state.pitch_briefs),
    )

    # 1. Pain point mining -----------------------------------------------
    if _should_mine(state):
        return {
            "current_stage": PipelineStage.MINING,
            "next_node": "pain_point_miner",
        }

    if _should_retry_mining(state):
        return {
            "current_stage": PipelineStage.MINING,
            "next_node": "pain_point_miner",
            "pain_point_miner_revision_count": state.pain_point_miner_revision_count + 1,
            **state.add_event(
                agent="orchestrator",
                stage=PipelineStage.MINING,
                kind="warning",
                message=f"Only {len(state.filtered_pain_points)} pain points found; retrying mining.",
            ),
        }

    # 2. Idea generation -------------------------------------------------
    if _should_generate_ideas(state):
        if not state.filtered_pain_points:
            return state.mark_failed(ERROR_NO_PAIN_POINTS)
        return {
            "current_stage": PipelineStage.GENERATING,
            "next_node": "idea_generator",
        }

    # 3. Scoring ---------------------------------------------------------
    if _should_score_ideas(state):
        return {
            "current_stage": PipelineStage.SCORING,
            "next_node": "scorer",
        }

    # 4. Scorer reflection or pitch writing ------------------------------
    if state.scored_ideas and not state.top_scored_ideas and not state.pitch_briefs:
        if state.can_revise:
            logger.warning(
                "[routing] All ideas parked by Scorer. Requesting revision from idea_generator."
            )
            return {
                "ideas": [],
                "scored_ideas": [],
                "pitch_briefs": [],
                "current_stage": PipelineStage.REVISING,
                "next_node": "idea_generator",
                "revision_feedback": (
                    "All previously generated ideas were parked due to fatal flaws or failing rubrics. "
                    "Generate fresh ideas exploring alternative angles."
                ),
                **state.add_event(
                    agent="orchestrator",
                    stage=PipelineStage.REVISING,
                    kind="warning",
                    message="All ideas parked by Scorer. Requesting new ideas from idea_generator.",
                ),
            }
        logger.warning(
            "[routing] All ideas parked and max revisions reached. Graduating best candidate."
        )
        return {
            "current_stage": PipelineStage.WRITING,
            "next_node": "pitch_writer",
            **state.add_event(
                agent="orchestrator",
                stage=PipelineStage.WRITING,
                kind="info",
                message="Max revisions reached. Graduating best-effort candidate to pitch_writer.",
            ),
        }

    if _should_write_pitches(state):
        if not state.scored_ideas:
            return state.mark_failed(ERROR_NO_SCORED_IDEAS)
        return {
            "current_stage": PipelineStage.WRITING,
            "next_node": "pitch_writer",
        }

    # 5. Critique & revision ---------------------------------------------
    if _should_critique(state):
        return {
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "critic",
        }

    if _should_revise(state):
        assert state.critique is not None
        target = state.critique.target_agent
        idea_id = state.critique.idea_id
        logger.info("[routing] Revision for idea=%s → target=%s", idea_id, target)
        reset_patch = state.reset_for_revision(target, idea_id)
        bump_patch = state.bump_revision(state.critique)
        return {
            **reset_patch,
            **bump_patch,
            **state.add_event(
                agent="orchestrator",
                stage=PipelineStage.REVISING,
                kind="warning",
                message=f"Critique flagged issues. Routing revision to {target}: {state.critique.revision_feedback}",
                idea_id=idea_id,
            ),
        }

    # 6. More pending briefs to critique? --------------------------------
    pending_briefs = state.revisions.pending_briefs(state.pitch_briefs)
    if pending_briefs:
        return {
            "critique": None,
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "critic",
            **state.add_event(
                agent="orchestrator",
                stage=PipelineStage.CRITIQUING,
                kind="info",
                message=f"Moving to critique next pending brief ({len(state.pitch_briefs) - len(pending_briefs) + 1} of {len(state.pitch_briefs)}).",
            ),
        }

    # 7. Complete --------------------------------------------------------
    logger.info("[routing] Pipeline finished successfully.")
    return {
        "current_stage": PipelineStage.COMPLETED,
        "next_node": "__end__",
        "completed_at": datetime.now(timezone.utc),
        **state.add_event(
            agent="orchestrator",
            stage=PipelineStage.COMPLETED,
            kind="info",
            message="Pipeline run completed successfully.",
        ),
    }
