"""
VentureForge LangGraph
======================
Assembles the hierarchical multi-agent graph with reflection loop and SQLite checkpoint persistence.

Usage:
    from src.graph import build_graph, GRAPH
    graph = build_graph()
    result = graph.invoke(state)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agents.orchestrator import (
    critic,
    idea_generator,
    orchestrator,
    pain_point_miner,
    pitch_writer,
    scorer,
)
from src.state.graph_state import VentureForgeState

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DB_PATH = ".cache/ventureforge.db"

ALLOWED_MSGPACK_MODULES = [
    ("src.models.common", "PipelineStage"),
    ("src.models.common", "DataSource"),
    ("src.models.common", "Verdict"),
    ("src.models.common", "TargetAgent"),
    ("src.models.common", "RunEvent"),
    ("src.models.common", "ErrorEntry"),
    ("src.models.pain_point", "PainPoint"),
    ("src.models.pain_point", "PainPointEvidence"),
    ("src.models.pain_point", "PainPointRubric"),
    ("src.models.idea", "Idea"),
    ("src.models.idea", "ScoredIdea"),
    ("src.models.pitch", "PitchBrief"),
    ("src.models.critique", "Critique"),
    ("src.models.critique", "CritiqueRubric"),
]


def get_checkpointer(db_path: str | None = DEFAULT_CHECKPOINT_DB_PATH) -> BaseCheckpointSaver:
    """Create a persistent SQLite checkpointer, falling back to in-memory if unavailable."""
    serde = JsonPlusSerializer().with_msgpack_allowlist(ALLOWED_MSGPACK_MODULES)
    if db_path:
        try:
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_file), check_same_thread=False)
            logger.info(f"[graph] Initialized SqliteSaver checkpoint persistence at '{db_path}'.")
            return SqliteSaver(conn, serde=serde)
        except Exception as e:
            logger.warning(f"[graph] Failed to initialize SQLite checkpointer at '{db_path}': {e}. Using MemorySaver.")
    return MemorySaver(serde=serde)


def route_after_orchestrator(state: VentureForgeState) -> str:
    """Return the next node name after the orchestrator runs."""
    return state.next_node


def route_after_critic(state: VentureForgeState) -> str:
    """After critic, always return to orchestrator for routing decisions."""
    return "orchestrator"


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """Build and return the compiled LangGraph StateGraph."""
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

    # Critic returns to orchestrator
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "orchestrator": "orchestrator",
            END: END,
        },
    )

    saver = checkpointer if checkpointer is not None else get_checkpointer()
    return workflow.compile(checkpointer=saver)


# Convenience: pre-compiled graph instance
GRAPH = build_graph()
