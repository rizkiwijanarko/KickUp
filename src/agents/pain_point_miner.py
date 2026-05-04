"""Pain Point Miner — extracts structured pain points from Reddit and Hacker News."""
from __future__ import annotations

from src.state.schema import PainPoint, PipelineStage, VentureForgeState


def run(state: VentureForgeState) -> dict:
    """
    Stub: extract pain points.
    TODO: integrate Reddit PRAW + HN Algolia API.
    """
    # Placeholder: return empty list, pipeline will log and proceed.
    return {
        "pain_points": [],
        "current_stage": PipelineStage.MINING,
        "next_node": "orchestrator",
    }
