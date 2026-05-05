"""Run controller for VentureForge UI.

Provides a thin abstraction over the LangGraph pipeline so that the
Gradio UI can:

* start a new run (returns run_id)
* poll the latest state for a run_id
* request cancellation (best-effort; currently UI-level only)

NOTE: The actual persistence is handled by the LangGraph SQLite
checkpointer configured in ``src.graph``. ``GRAPH.get_state`` is used to
recover the latest VentureForgeState snapshot for a given run_id.
"""
from __future__ import annotations

import threading
from typing import Optional

from src.graph import GRAPH
from src.main import make_initial_state
from src.state.schema import VentureForgeState

# Single active run tracking (hackathon scope: one run at a time)
_active_thread: Optional[threading.Thread] = None
_active_run_id: Optional[str] = None
_cancel_requested: set[str] = set()


class RunInProgressError(RuntimeError):
    """Raised when attempting to start a new run while one is active."""


def start_run(
    domain: str,
    *,
    max_pain_points: int | None = None,
    ideas_per_run: int | None = None,
) -> str:
    """Start a new pipeline run in a background thread.

    Returns the newly created ``run_id``.  Raises ``RunInProgressError``
    if a previous run is still active.
    """
    global _active_thread, _active_run_id

    if _active_thread is not None and _active_thread.is_alive():
        raise RunInProgressError("A run is already in progress")

    state = make_initial_state(
        domain,
        max_pain_points=max_pain_points,
        ideas_per_run=ideas_per_run,
    )
    run_id = state.run_id

    def _worker() -> None:
        global _active_thread, _active_run_id
        try:
            # Drive the pipeline via LangGraph streaming so we can
            # respond to cancellation requests between steps. The
            # SQLite checkpointer persists intermediate states;
            # ``poll_state`` reads them for the UI.
            for _ in GRAPH.stream(
                state,
                config={
                    "recursion_limit": 80,
                    "configurable": {"thread_id": run_id},
                },
            ):
                if is_cancel_requested(run_id):
                    break
        finally:
            # Mark thread as finished; the last checkpoint remains
            # available for inspection.
            _active_thread = None
            _active_run_id = None

    t = threading.Thread(target=_worker, daemon=True)
    _active_thread = t
    _active_run_id = run_id
    t.start()

    return run_id


def poll_state(run_id: str) -> VentureForgeState | None:
    """Return the latest VentureForgeState for ``run_id`` from checkpoints.

    Returns ``None`` if no state has been persisted yet.
    """
    stored = GRAPH.get_state(config={"configurable": {"thread_id": run_id}})
    # ``stored`` is a LangGraph StoredState; ``values`` holds the latest
    # state dict that was checkpointed.
    if stored is None or stored.values is None:
        return None
    return VentureForgeState.model_validate(stored.values)


def request_cancel(run_id: str) -> None:
    """Record that the user requested cancellation for ``run_id``.

    Current implementation is UI-level only: it records the request so
    the UI can reflect that the user pressed Stop. The underlying
    LangGraph run is not yet interrupted mid-flight; it will naturally
    complete and persist its final state.
    """
    _cancel_requested.add(run_id)


def is_cancel_requested(run_id: str) -> bool:
    """Return True if cancellation was requested for this run_id."""
    return run_id in _cancel_requested


def is_run_active() -> bool:
    """Return True if a pipeline run is currently active."""
    return _active_thread is not None and _active_thread.is_alive()
