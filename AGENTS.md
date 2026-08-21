# VentureForge

VentureForge is a hierarchical LangGraph multi-agent pipeline that mines market pain points, generates startup ideas, scores them against binary rubrics, drafts pitches, and critiques outputs with grounded source verification.

## Core Invariants

- **Ignored Directories**: Treat as non-existent: `.venv/`, `__pycache__/`, `.cache/`, `.mypy_cache/`, `.ruff_cache/`, `dist/`, `build/`, `*.egg-info/`.
- **Pure State Transformers**: Never mutate `VentureForgeState` in-place. Agents must return patch dicts for `model_copy(update=...)`.
- **Graph Specification**: Always keep `src/graph.py` synchronized with `orchestration.json`.
- **Token Savior MCP Enforcement**: Use the project-scoped `token-savior` MCP server from `.mcp.json`. It is provided by `token-savior-recall[mcp]` and exposes the `token-savior` executable. Prefer its structural navigation, precision editing, and impact-analysis tools over whole-file source reads or blind grep to minimize context token usage.

## Development Commands

Managed with Python 3.11+ and `uv`:

- **Gradio UI** (Primary): `uv run app.py`
- **CLI Pipeline**: `uv run python -m src.main --domain "<domain>" --output output.json`
- **Test Suite**: `uv run pytest`
- **Type Checking**: `uv run mypy src/`
- **Linting & Formatting**: `uv run ruff check .` && `uv run ruff format .`

## Context Pointers

Read these specialized documents on demand when working in their area:

- [Architecture & Graph](file:///docs/agents/architecture.md): LangGraph nodes, `VentureForgeState`, orchestrator routing, reflection loop.
- [Environment & LLM Configuration](file:///docs/agents/environment.md): LLM provider switching, API keys, Hacker News/Reddit/Tavily data sources.
- [Token Savior MCP](file:///docs/agents/token-savior.md): Symbol lookups, token-efficient navigation, and precision code modifications.
- [Code Style & Types](file:///docs/agents/code-style.md): Python 3.11+, strict mypy types, Pydantic v2 models, prompt storage in `PROMPTS.md`.
- [Testing Guidelines](file:///docs/agents/testing.md): Component vs E2E testing, pytest invocation patterns.
- [Issue Tracker](file:///docs/agents/issue-tracker.md): GitHub Issues via `gh` CLI.
- [Triage Labels](file:///docs/agents/triage-labels.md): Canonical 5-role triage vocabulary.
- [Domain Docs](file:///docs/agents/domain.md): Single-context documentation conventions (`CONTEXT.md` + `docs/adr/`).
