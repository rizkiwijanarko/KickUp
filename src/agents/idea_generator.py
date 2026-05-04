"""Idea Generator — groups pain points into themes and generates startup ideas."""
from __future__ import annotations

from src.state.schema import PipelineStage, VentureForgeState


def run(state: VentureForgeState) -> dict:
    """Stub: generate ideas from pain points."""
    return {
        "ideas": [],
        "current_stage": PipelineStage.GENERATING,
        "next_node": "orchestrator",
    }
