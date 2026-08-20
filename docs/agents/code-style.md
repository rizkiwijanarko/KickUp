# Code Style & Quality Standards

- **Target Runtime**: Python 3.11+.
- **Formatting**: 100-character line length enforced by Black and Ruff (`uv run ruff format .` and `uv run ruff check .`).
- **Typing**: Strict mypy (`disallow_untyped_defs = true`, `warn_return_any = true`). Public APIs and functions must have explicit type annotations.
- **Data Models**: Use Pydantic v2 `BaseModel` classes (`src/state/schema.py`) instead of unstructured dictionaries for domain state.
- **Commit Messages**: Single-sentence, plain-English, action-oriented messages without Conventional Commits prefixes (e.g., `Update idea evaluation rubric weights in scorer node`).
