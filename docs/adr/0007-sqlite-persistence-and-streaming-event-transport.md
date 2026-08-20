# SQLite Checkpointer Persistence and Streaming Event Transport

Relying solely on ephemeral `MemorySaver` checkpointers prevents process resumption, multi-turn UI persistence, and crash recovery. Additionally, polling snapshots via `get_state` introduces polling latency in the UI. We decided to adopt `SqliteSaver` pointing to `.cache/ventureforge.db` for thread-isolated state persistence, enabling the `--resume <run_id>` CLI workflow and reliable state recovery. Concurrently, `run_controller.py` transitions to yielding real-time streaming events and tokens directly from LangGraph execution.
