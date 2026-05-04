"""Scorer — evaluates ideas with a binary yes/no rubric."""
from __future__ import annotations

from src.state.schema import PipelineStage, VentureForgeState


def run(state: VentureForgeState) -> dict:
    """Stub: score ideas."""
    return {
        "scored_ideas": [],
        "current_stage": PipelineStage.SCORING,
        "next_node": "orchestrator",
    }
