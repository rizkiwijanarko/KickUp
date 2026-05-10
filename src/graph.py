"""
VentureForge LangGraph
======================
Assembles the hierarchical multi-agent graph with reflection loop.

Usage:
    from src.graph import build_graph
    graph = build_graph()
    result = graph.invoke(state)
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.orchestrator import (
    critic,
    idea_generator,
    orchestrator,
    pain_point_miner,
    pitch_writer,
    scorer,
)
from src.config import settings
from src.state.schema import PipelineStage, VentureForgeState


def route_after_orchestrator(state: VentureForgeState) -> str:
    """Return the next node name after the orchestrator runs."""
    return state.next_node


def route_after_critic(state: VentureForgeState) -> str:
    """
    After critic, always return to orchestrator for routing decisions.
    
    The orchestrator is the single source of truth for:
    - Moving to next brief
    - Marking pipeline as completed
    - Handling revision loops
    
    This prevents the critic from bypassing orchestrator logic and ending
    the pipeline prematurely (e.g., after first brief approval).
    """
    return "orchestrator"


def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph StateGraph.

    The compiled graph is configured with an in-memory checkpointer so
    that runs can be inspected via LangGraph's persistence layer during
    a single process lifetime.
    """
    workflow = StateGraph(VentureForgeState)

    # Register nodes
    workflow.add_node("orchestrator", orchestrator)
    workflow.add_node("pain_point_miner", pain_point_miner)
    workflow.add_node("idea_generator", idea_generator)
    workflow.add_node("scorer", scorer)
    workflow.add_node("pitch_writer", pitch_writer)
    workflow.add_node("critic", critic)

    # Entry point
    workflow.set_entry_point("orchestrator")

    # Orchestrator routes to the appropriate worker (or end)
    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "pain_point_miner": "pain_point_miner",
            "idea_generator": "idea_generator",
            "scorer": "scorer",
            "pitch_writer": "pitch_writer",
            "critic": "critic",
            "__end__": END,
        },
    )

    # Workers always return to orchestrator
    for worker in ("pain_point_miner", "idea_generator", "scorer", "pitch_writer"):
        workflow.add_edge(worker, "orchestrator")

    # Critic either loops back (revision) or ends
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "orchestrator": "orchestrator",
            END: END,
        },
    )

    # Configure in-memory checkpointer with custom type support
    # This allows PipelineStage and other custom types to be serialized
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# Convenience: pre-compiled graph instance
GRAPH = build_graph()
