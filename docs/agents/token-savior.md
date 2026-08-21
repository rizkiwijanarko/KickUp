# Token Savior MCP Guide

Token Savior is the MCP server provided by the `token-savior-recall` package. Install it with `pip install "token-savior-recall[mcp]"`; the package exposes the `token-savior` executable. VentureForge registers it in the project-scoped `.mcp.json` with `WORKSPACE_ROOTS` set to this repository and `TOKEN_SAVIOR_PROFILE=optimized`.

## Core Directives

1. **Structural Inspection over Full-File Reads**: Prefer `find_symbol`, `get_full_context`, `read_lines`, and `search_codebase` over viewing entire source files or using blind grep.
2. **Context-Preserving Edits**: Use `get_edit_context`, `replace_symbol_source`, and `insert_near_symbol` for surgical code changes that preserve surrounding context and index consistency.
3. **Targeted Tool Discovery**: Use `ts_search` when the required Token Savior tool is not already known.
4. **Impact Analysis**: Use `find_impacted_test_files`, `find_import_cycles`, `detect_breaking_changes`, `find_dead_code`, or `analyze_config` when the change warrants it.
5. **Project Scope**: Keep `WORKSPACE_ROOTS` pointed at the active VentureForge checkout; do not index ignored directories or unrelated workspaces.

## Primary Tool Workflows

### 1. Code Exploration & Navigation
- `find_symbol(name="...")`: Locate a class, function, model, or other symbol definition.
- `get_full_context(name="...", depth=2)`: Retrieve a symbol with callers, dependencies, neighbors, and related tests in one request.
- `read_lines(file_path="...", start=..., end=...)`: Read a narrow line range when an error or trace provides line numbers.
- `search_codebase(pattern="...")`: Search the indexed project for a targeted pattern.
- `ts_search(query="...")`: Discover available Token Savior tools and their capabilities.

### 2. Surgical Modifications
- `get_edit_context(name="...")`: Retrieve source, callers, dependencies, neighbors, and tests before editing.
- `replace_symbol_source(file_path="...", symbol_name="...", new_source="...")`: Replace a symbol cleanly.
- `insert_near_symbol(file_path="...", symbol_name="...", content="...", position="after")`: Insert content adjacent to an existing symbol.

### 3. Impact & Testing
- `find_impacted_test_files(changed_files=[...])`: Identify tests affected by core model or agent changes.
- `find_import_cycles`: Detect import-cycle risk after module changes.
- `detect_breaking_changes(ref="...")`: Audit API changes against a reference revision.
- `find_dead_code`: Locate unused symbols when removing or restructuring code.
- `analyze_config(checks=["orphans"])`: Audit configuration references after config changes.

## Configuration Reference

The project registration is maintained by Command Code in `.mcp.json`. The equivalent setup command is:

```text
cmdc mcp add --scope project --transport stdio --env WORKSPACE_ROOTS=D:\\Project\\KickUp --env TOKEN_SAVIOR_CLIENT=command-code --env TOKEN_SAVIOR_PROFILE=optimized token-savior -- token-savior
```

Do not add `token-savior-recall` to VentureForge runtime dependencies unless application code directly imports it; this MCP integration is an agent development tool.
