"""
VentureForge CLI
================
Run the full multi-agent pipeline from the command line.

Usage:
    python -m src.main --domain "developer tools"
"""

from __future__ import annotations

import argparse
import json

from src.config import settings
from src.graph import GRAPH
from src.state.schema import VentureForgeState


def make_initial_state(
    domain: str,
    max_pain_points: int | None = None,
    ideas_per_run: int | None = None,
    top_n_pitches: int | None = None,
    max_revisions: int | None = None,
) -> VentureForgeState:
    """Construct the initial VentureForgeState for a new run.

    Shared between the CLI entrypoint and the Gradio UI controller so
    that both use the same defaults from ``src.config.settings``.
    """
    return VentureForgeState(
        domain=domain,
        max_pain_points=max_pain_points or settings.max_pain_points,
        ideas_per_run=ideas_per_run or settings.ideas_per_run,
        top_n_pitches=top_n_pitches or settings.top_n_pitches,
        max_revisions=max_revisions or settings.max_revisions,
    )


def run_pipeline(
    domain: str | None,
    max_pain_points: int | None = None,
    *,
    recursion_limit: int = 80,
    resume_run_id: str | None = None,
) -> VentureForgeState:
    """Execute the full end-to-end pipeline and return final state.

    If ``resume_run_id`` is provided, the pipeline resumes from the latest
    checkpoint for that ``run_id`` (thread_id) using the LangGraph SQLite
    checkpointer and ignores ``domain``/``max_pain_points``.
    """

    # Resume mode: load state from existing checkpoints and continue.
    if resume_run_id is not None:
        return GRAPH.invoke(
            None,
            config={
                "recursion_limit": recursion_limit,
                "configurable": {"thread_id": resume_run_id},
            },
        )

    if domain is None:
        raise ValueError("domain is required when not resuming from a previous run")

    state = make_initial_state(domain, max_pain_points=max_pain_points)
    # LangGraph invoke returns updated state. Use the state's run_id as
    # the checkpoint "thread_id" so runs can be resumed/inspected.
    return GRAPH.invoke(
        state,
        config={
            "recursion_limit": recursion_limit,
            "configurable": {"thread_id": state.run_id},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VentureForge — AI Startup Discovery")
    parser.add_argument(
        "--domain",
        type=str,
        required=False,
        help="Target domain, e.g. 'developer tools' (ignored when using --resume)",
    )
    parser.add_argument(
        "--max-pain-points",
        type=int,
        default=None,
        help="Override max pain points to extract (new runs only)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Existing run_id to resume from LangGraph checkpoints",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    if args.resume:
        print(f"Resuming VentureForge run: run_id='{args.resume}'")
        result = run_pipeline(
            domain=None,
            max_pain_points=None,
            resume_run_id=args.resume,
        )
    else:
        if not args.domain:
            parser.error("--domain is required for new runs (omit it when using --resume)")
        print(f"VentureForge starting: domain='{args.domain}'")
        result = run_pipeline(args.domain, args.max_pain_points)

    # Serialize final state
    # LangGraph returns a dict that may contain Pydantic models
    # We need to serialize them properly to avoid "Object of type X is not JSON serializable" errors
    if isinstance(result, dict):
        # Convert dict to VentureForgeState to ensure proper serialization
        from src.state.schema import VentureForgeState
        try:
            state = VentureForgeState(**result)
            output = state.model_dump(mode="json", exclude_none=True)
        except Exception as e:
            # Fallback: try direct serialization (may fail if dict contains Pydantic models)
            print(f"Warning: Could not convert result to VentureForgeState: {e}")
            print("Attempting direct serialization...")
            output = result
    else:
        output = result.model_dump(mode="json", exclude_none=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nPipeline finished in stage: {output.get('current_stage', 'unknown')}")
    print(f"   Run ID     : {output.get('run_id', 'unknown')}")
    print(f"   Duration   : {output.get('agent_timings', {})}")
    print(f"   Pain points: {len(output.get('pain_points', []))}")
    print(f"   Ideas      : {len(output.get('ideas', []))}")
    print(f"   Pitches    : {len(output.get('pitch_briefs', []))}")
    revision_counts = output.get('revision_counts', {})
    total_revisions = sum(revision_counts.values())
    print(f"   Revisions  : {total_revisions} (across {len(revision_counts)} pitches)")
    print(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()
