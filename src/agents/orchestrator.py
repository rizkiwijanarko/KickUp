"""
Orchestrator — graph node adapter.

Routing decisions: src/agents/routing.py
Timing:           src/graph.py (timed() decorator applied at wiring site)
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents import routing
from src.state.graph_state import VentureForgeState

logger = logging.getLogger(__name__)


def orchestrator(state: VentureForgeState) -> dict[str, Any]:
    """Graph node: delegates all routing decisions to routing.route()."""
    return routing.route(state)
