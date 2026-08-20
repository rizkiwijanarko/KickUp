# Token Savior MCP Guide

Token Savior is an MCP server providing structural indexation, symbol resolution, and precision code editing to minimize context window consumption and avoid wasteful whole-file reads.

## Core Directives

1. **Symbol Inspection over Full-File Reads**: When investigating functions, models, or classes, prefer `find_symbol`, `get_function_source`, or `get_class_source` over viewing the full file.
2. **Context-Preserving Edits**: Use `get_edit_context` and `replace_symbol_source` for surgical modifications to existing symbols without perturbing adjacent code.
3. **Targeted Codebase Search**: Use `search_codebase` or `search_in_symbols` to locate patterns across the codebase efficiently.
4. **Impact Analysis**: Use `find_impacted_test_files` when refactoring core models or agents to pinpoint affected tests.

## Primary Tool Workflows

### 1. Code Exploration & Navigation
- `get_project_summary`: High-level summary of modules, routes, and statistics.
- `find_symbol(name="...")`: Locate where a class, function, or model is defined.
- `get_function_source(file_path="...", function_name="...")`: Extract only the function definition.
- `get_class_source(file_path="...", class_name="...")`: Extract only the class definition.
- `get_call_chain(symbol_name="...")`: Trace upstream callers and downstream dependencies.

### 2. Surgical Modifications
- `get_edit_context(file_path="...", line_number=...)`: View minimal surrounding lines for accurate edits.
- `replace_symbol_source(file_path="...", symbol_name="...", new_source="...")`: Replace a symbol's entire body cleanly.
- `insert_near_symbol(file_path="...", symbol_name="...", content="...", position="after")`: Insert helper functions or methods adjacent to a symbol.

### 3. Impact & Testing
- `find_impacted_test_files(changed_files=[...])`: Discover which test suites need to run for a specific change.
