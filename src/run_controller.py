"""Run controller for VentureForge UI.

Provides a thin abstraction over the LangGraph pipeline so that the
Gradio UI can:

* start a new run (returns run_id)
* poll the latest state for a run_id
* request cancellation (interrupts the LangGraph execution)

NOTE: The actual persistence is handled by the LangGraph SQLite
checkpointer configured in ``src.graph``. ``GRAPH.get_state`` is used to
recover the latest VentureForgeState snapshot for a given run_id.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from pydantic import ValidationError

from src.graph import GRAPH
from src.main import make_initial_state
from src.state.schema import VentureForgeState

logger = logging.getLogger(__name__)

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
        import time
        last_step_time = time.time()
        timeout_seconds = 300  # 5 minutes per step
        
        try:
            logger.info(f"[run_controller] Starting pipeline run {run_id}")
            # Drive the pipeline via LangGraph streaming so we can
            # respond to cancellation requests between steps. The
            # checkpointer persists intermediate states; ``poll_state``
            # reads them for the UI.
            step_count = 0
            for chunk in GRAPH.stream(
                state,
                config={
                    "recursion_limit": 150,  # Increased from 80 to handle retry loops
                    "configurable": {"thread_id": run_id},
                },
            ):
                step_count += 1
                current_time = time.time()
                elapsed = current_time - last_step_time
                
                # Log each step for debugging
                if chunk:
                    node_name = list(chunk.keys())[0] if chunk else "unknown"
                    logger.info(f"[run_controller] Step {step_count}: {node_name} (took {elapsed:.1f}s)")
                    
                    # Warn if step took too long
                    if elapsed > timeout_seconds:
                        logger.warning(
                            f"[run_controller] Step {step_count} ({node_name}) took {elapsed:.1f}s "
                            f"(timeout threshold: {timeout_seconds}s)"
                        )
                
                last_step_time = current_time
                
                # Check for cancellation between steps
                if is_cancel_requested(run_id):
                    logger.info(f"[run_controller] Cancellation requested for {run_id}, stopping")
                    # Update state to mark as cancelled
                    current_state = poll_state(run_id)
                    if current_state:
                        GRAPH.update_state(
                            config={"configurable": {"thread_id": run_id}},
                            values=current_state.mark_cancelled("User requested stop"),
                        )
                    break
            
            logger.info(f"[run_controller] Pipeline run {run_id} completed after {step_count} steps")
        except Exception as exc:  # pragma: no cover - surfaced via UI logs
            # Surface worker errors to stdout so they are visible when
            # running the Gradio UI. Without this, exceptions inside
            # the background thread can make the UI hang without any
            # clear error message.
            logger.error(f"[run_controller] Pipeline worker crashed: {exc!r}", exc_info=True)
            print(f"[run_controller] Pipeline worker crashed: {exc!r}", flush=True)
        finally:
            # Mark thread as finished; the last checkpoint remains
            # available for inspection.
            logger.info(f"[run_controller] Worker thread for {run_id} finished")
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
    if stored is None or not stored.values:
        # No meaningful state yet (LangGraph may persist an empty dict
        # before the first node runs). Treat this the same as "no
        # checkpoints" so the UI keeps waiting instead of raising a
        # validation error about missing required fields like ``domain``.
        return None

    try:
        return VentureForgeState.model_validate(stored.values)
    except ValidationError:
        # If the stored values are incomplete or from an older schema,
        # fail gracefully and let the UI keep polling.
        return None


def request_cancel(run_id: str) -> None:
    """Record that the user requested cancellation for ``run_id``.

    The cancellation will take effect between LangGraph steps. If an agent
    is currently executing (e.g., waiting for an LLM response), the
    cancellation will be processed after that agent completes.
    """
    logger.info(f"[run_controller] Cancellation requested for run {run_id}")
    _cancel_requested.add(run_id)


def is_cancel_requested(run_id: str) -> bool:
    """Return True if cancellation was requested for this run_id."""
    return run_id in _cancel_requested


def is_run_active() -> bool:
    """Return True if a pipeline run is currently active."""
    return _active_thread is not None and _active_thread.is_alive()
