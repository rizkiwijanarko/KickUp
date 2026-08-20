# Decouple Domain Agents from LangGraph State Container

The monolithic `VentureForgeState` mixed domain models, workflow routing variables, timing metrics, and custom patch merge mechanics into a single 900-line God Object. We decided to decouple worker agents into pure domain modules with explicit parameter signatures, converting the LangGraph layer into a thin declarative orchestration graph using native reducers. This establishes clear interface seams, improves testability without mocking graph checkpoints, and isolates external data miners behind a unified strategy adapter.
