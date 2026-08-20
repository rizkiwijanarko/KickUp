# Architecture & Orchestration

VentureForge is built around a single LangGraph state container: `VentureForgeState` (`src/state/schema.py`).

## Graph Nodes (`src/graph.py`)

- **`orchestrator`**: Coordinates execution order, checks evaluation gates, and handles reflection routing.
- **`pain_point_miner`**: Extracts grounded user pain points from sources (Hacker News, Product Hunt, Tavily, Reddit).
- **`idea_generator`**: Produces candidate startup solutions addressing extracted pain points.
- **`scorer`**: Evaluates ideas using binary yes/no rubric criteria.
- **`pitch_writer`**: Drafts investor-ready one-pagers and pitch decks for approved ideas.
- **`critic`**: Enforces strict hallucination-free verification. Verifies all pain points and claims map to real source URLs.

## Orchestration Contract

- **Pure State Updates**: Agents receive a state snapshot and return a dictionary patch. The graph merges this via `model_copy(update=...)`.
- **Revision & Reflection**: Governed by `max_revisions`. When Critic returns `revise`, the orchestrator routes to the target agent and calls `reset_for_revision` to invalidate downstream fields.
- **Specification Parity**: `orchestration.json` is the machine-readable spec for the graph; update it whenever changing node behavior or routing.
- **Checkpoints**: LangGraph execution checkpoints are persisted under `.cache/`. Deleting `.cache/` clears run history without affecting pipeline behavior.
