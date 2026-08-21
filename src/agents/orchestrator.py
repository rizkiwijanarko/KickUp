"""
Orchestrator — coordinates execution stages and handles the reflection loop.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from src.agents.critic import run as run_critic
from src.agents.idea_generator import run as run_idea_generator
from src.agents.pain_point_miner import run as run_pain_point_miner
from src.agents.pitch_writer import run as run_pitch_writer
from src.agents.scorer import run as run_scorer
from src.constants import (
    ERROR_INSUFFICIENT_PAIN_POINTS,
    ERROR_NO_PAIN_POINTS,
    ERROR_NO_SCORED_IDEAS,
    MAX_MINING_RETRIES,
    MIN_PAIN_POINTS_FOR_IDEAS,
)
from src.models import PipelineStage, TargetAgent
from src.state.graph_state import VentureForgeState

logger = logging.getLogger(__name__)


# =============================================================================
# Stage Decision Predicates
# =============================================================================


def should_mine(state: VentureForgeState) -> bool:
    return not state.pain_points


def should_retry_mining(state: VentureForgeState) -> bool:
    return (
        not state.ideas
        and len(state.filtered_pain_points) < MIN_PAIN_POINTS_FOR_IDEAS
        and state.pain_point_miner_revision_count < MAX_MINING_RETRIES
    )


def should_generate_ideas(state: VentureForgeState) -> bool:
    return not state.ideas


def should_score_ideas(state: VentureForgeState) -> bool:
    return not state.scored_ideas or len({s.idea_id for s in state.scored_ideas}) < len(state.ideas)


def should_write_pitches(state: VentureForgeState) -> bool:
    if not state.top_scored_ideas:
        return False
    top_ids = {s.idea_id for s in state.top_scored_ideas}
    brief_ids = {b.idea_id for b in state.pitch_briefs}
    return len(top_ids - brief_ids) > 0


def should_critique(state: VentureForgeState) -> bool:
    if not state.pitch_briefs:
        return False
    if state.critique is None:
        return True
    current_brief = state.pitch_briefs[min(state.current_critique_index, len(state.pitch_briefs) - 1)]
    return state.critique.idea_id != current_brief.idea_id


def should_revise(state: VentureForgeState) -> bool:
    return bool(state.critique and not state.critique.all_pass and state.can_revise)


# =============================================================================
# Main Orchestrator Node
# =============================================================================


def orchestrator(state: VentureForgeState) -> dict[str, Any]:
    """Decides the next pipeline stage based on current state."""
    logger.info(
        f"[orchestrator] State: pain_points={len(state.pain_points)}, "
        f"ideas={len(state.ideas)}, scored={len(state.scored_ideas)}, "
        f"briefs={len(state.pitch_briefs)}"
    )

    # 1. Pain point mining
    if should_mine(state):
        return {
            "current_stage": PipelineStage.MINING,
            "next_node": "pain_point_miner",
        }

    if should_retry_mining(state):
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

    # 2. Idea generation
    if should_generate_ideas(state):
        if not state.filtered_pain_points:
            return state.mark_failed(ERROR_NO_PAIN_POINTS)
        return {
            "current_stage": PipelineStage.GENERATING,
            "next_node": "idea_generator",
        }

    # 3. Scoring
    if should_score_ideas(state):
        return {
            "current_stage": PipelineStage.SCORING,
            "next_node": "scorer",
        }

    # 4. Scorer Reflection or Pitch writing
    if state.scored_ideas and not state.top_scored_ideas and not state.pitch_briefs:
        if state.can_revise:
            logger.warning("[orchestrator] All generated ideas were parked by Scorer. Routing revision to idea_generator.")
            return {
                "ideas": [],
                "scored_ideas": [],
                "pitch_briefs": [],
                "current_stage": PipelineStage.REVISING,
                "next_node": "idea_generator",
                "revision_feedback": "All previously generated ideas were parked due to fatal flaws or failing rubrics. Generate fresh ideas exploring alternative angles.",
                **state.add_event(
                    agent="orchestrator",
                    stage=PipelineStage.REVISING,
                    kind="warning",
                    message="All ideas parked by Scorer. Requesting new ideas from idea_generator.",
                ),
            }
        else:
            logger.warning("[orchestrator] All ideas parked by Scorer and max revisions reached. Graduating best candidate to pitch_writer.")
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

    if should_write_pitches(state):
        if not state.scored_ideas:
            return state.mark_failed(ERROR_NO_SCORED_IDEAS)
        return {
            "current_stage": PipelineStage.WRITING,
            "next_node": "pitch_writer",
        }

    # 5. Critique & Revision
    if should_critique(state):
        return {
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "critic",
        }

    if should_revise(state):
        assert state.critique is not None
        target = state.critique.target_agent
        idea_id = state.critique.idea_id
        logger.info(f"[orchestrator] Routing revision for idea={idea_id} to target={target}")

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

    # 6. More briefs to critique?
    if state.current_critique_index + 1 < len(state.pitch_briefs):
        return {
            "current_critique_index": state.current_critique_index + 1,
            "critique": None,
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "critic",
            **state.add_event(
                agent="orchestrator",
                stage=PipelineStage.CRITIQUING,
                kind="info",
                message=f"Moving to critique brief {state.current_critique_index + 2} of {len(state.pitch_briefs)}.",
            ),
        }

    # 7. Complete
    logger.info("[orchestrator] Pipeline finished successfully")
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


# =============================================================================
# Worker Wrapper Nodes (with timing)
# =============================================================================


def pain_point_miner(state: VentureForgeState) -> dict[str, Any]:
    t0 = time.monotonic()
    result = run_pain_point_miner(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("pain_point_miner", elapsed)}


def idea_generator(state: VentureForgeState) -> dict[str, Any]:
    t0 = time.monotonic()
    result = run_idea_generator(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("idea_generator", elapsed)}


def scorer(state: VentureForgeState) -> dict[str, Any]:
    t0 = time.monotonic()
    result = run_scorer(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("scorer", elapsed)}


def pitch_writer(state: VentureForgeState) -> dict[str, Any]:
    t0 = time.monotonic()
    result = run_pitch_writer(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("pitch_writer", elapsed)}


def critic(state: VentureForgeState) -> dict[str, Any]:
    t0 = time.monotonic()
    result = run_critic(state)
    elapsed = time.monotonic() - t0
    return {**result, **state.record_timing("critic", elapsed)}
