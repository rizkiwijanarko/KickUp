"""Pitch Writer — writes investor-ready one-page pitch briefs for top ideas."""
from __future__ import annotations

from src.state.schema import PipelineStage, VentureForgeState


def run(state: VentureForgeState) -> dict:
    """Stub: write pitch briefs for top N scored ideas."""
    return {
        "pitch_briefs": [],
        "current_stage": PipelineStage.WRITING,
        "next_node": "orchestrator",
    }
