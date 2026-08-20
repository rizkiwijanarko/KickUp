# Testing Guidelines

Tests live under `test/` and are executed via `pytest`.

## Running Tests

- Run full suite: `uv run pytest`
- Run single test file: `uv run pytest test/test_critic_component.py`
- Run specific test case: `uv run pytest test/test_critic_component.py::test_auto_approve_at_max_revisions`

## Test Categories

- **Component Tests** (`test/test_*_component.py`): Test individual agents in isolation with mock states.
- **Flow & Revision Tests** (`test/test_revision_feedback_flow.py`): Validate orchestration loops, routing logic, and state patching.
- **End-to-End Tests** (`test/test_e2e.py`): Full pipeline execution hitting configured LLM providers and data sources. Requires active LLM API credentials (`LLM_API_KEY` or `OPENAI_API_KEY`). Falls back to Hacker News / Tavily if Reddit credentials are absent.
