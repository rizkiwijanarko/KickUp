"""Orchestrator — routes tasks, manages state, handles reflection loop. Never generates content."""
from __future__ import annotations

import time

from src.state.schema import PipelineStage, VentureForgeState

# Import worker agents (stubs; real implementations in agent files)
from src.agents.pain_point_miner import run as run_pain_point_miner
from src.agents.idea_generator import run as run_idea_generator
from src.agents.scorer import run as run_scorer
from src.agents.pitch_writer import run as run_pitch_writer
from src.agents.critic import run as run_critic


def orchestrator(state: VentureForgeState) -> dict:
    """
    Supervisor node. Based on pipeline progress, decides which worker to run next.
    Returns a dict patch for state update.
    """
    stage = state.current_stage
    revision = state.revision_count > 0

    # --- Determine next stage ---
    if not state.pain_points:
        return {"current_stage": PipelineStage.MINING, "next_node": "pain_point_miner"}

    if not state.ideas:
        return {"current_stage": PipelineStage.GENERATING, "next_node": "idea_generator"}

    if not state.scored_ideas:
        return {"current_stage": PipelineStage.SCORING, "next_node": "scorer"}

    if not state.pitch_briefs:
        return {"current_stage": PipelineStage.WRITING, "next_node": "pitch_writer"}

    if state.critique is None:
        return {"current_stage": PipelineStage.CRITIQUING, "next_node": "critic"}

    # --- Reflection loop: we have a critique ---
    if not state.critique.all_pass and state.can_revise:
        # Loop back to target worker for revision
        target = state.critique.target_agent.value
        patch = state.bump_revision(state.critique)
        patch.update(state.reset_for_revision(target))
        return patch

    # --- Done or max revisions reached ---
    return state.mark_completed()


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
