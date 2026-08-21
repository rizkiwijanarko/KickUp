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
import logging
from typing import Any

from src.config import settings
from src.graph import GRAPH
from src.state.schema import VentureForgeState

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
)


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
    force_refresh: bool = False,
) -> VentureForgeState | dict[str, Any]:
    """Execute the full end-to-end pipeline with real-time stage progress reporting.

    If ``resume_run_id`` is provided, the pipeline resumes from the latest
    checkpoint for that ``run_id`` (thread_id) using the LangGraph SQLite
    checkpointer and ignores ``domain``/``max_pain_points``.
    """
    if resume_run_id is not None:
        thread_id = resume_run_id
        initial_input = None
    else:
        if domain is None:
            raise ValueError("domain is required when not resuming from a previous run")
        state = make_initial_state(domain, max_pain_points=max_pain_points)
        thread_id = state.run_id
        initial_input = state

    config = {
        "recursion_limit": recursion_limit,
        "configurable": {"thread_id": thread_id},
    }

    print(f"\n{'='*70}\n[VentureForge] Pipeline Execution (Thread ID: {thread_id})\n{'='*70}\n", flush=True)

    for event in GRAPH.stream(initial_input, config=config):
        for node_name, patch in event.items():
            if node_name == "orchestrator":
                stage = patch.get("current_stage", "")
                next_node = patch.get("next_node", "")
                print(f"[Stage] Orchestrator: {stage} -> Next Worker: '{next_node}'", flush=True)
            elif node_name == "pain_point_miner":
                pps = patch.get("pain_points", [])
                print(f"\n[Stage] Pain Point Miner: Extracted {len(pps)} validated pain points:", flush=True)
                for i, p in enumerate(pps):
                    title = getattr(p, "title", p.get("title", "")) if isinstance(p, dict) else p.title
                    ev_count = len(getattr(p, "evidence", p.get("evidence", []))) if isinstance(p, dict) else len(p.evidence)
                    print(f"   {i+1}. {title} ({ev_count} verified quotes)", flush=True)
                print("", flush=True)
            elif node_name == "idea_generator":
                ideas = patch.get("ideas", [])
                print(f"\n[Stage] Idea Generator: Generated {len(ideas)} startup concepts in parallel:", flush=True)
                for i, idea in enumerate(ideas):
                    title = getattr(idea, "title", idea.get("title", "")) if isinstance(idea, dict) else idea.title
                    target = getattr(idea, "target_user", idea.get("target_user", "")) if isinstance(idea, dict) else idea.target_user
                    print(f"   {i+1}. {title} (Target: {target})", flush=True)
                print("", flush=True)
            elif node_name == "scorer":
                scored = patch.get("scored_ideas", [])
                print(f"\n[Stage] Scorer: Evaluated {len(scored)} ideas against binary rubric:", flush=True)
                for si in scored:
                    rank = getattr(si, "rank", si.get("rank", 0) if isinstance(si, dict) else 0)
                    yes_count = getattr(si, "yes_count", si.get("yes_count", 0) if isinstance(si, dict) else 0)
                    total = getattr(si, "total_checks", si.get("total_checks", 8) if isinstance(si, dict) else 8)
                    verdict = getattr(si, "verdict", si.get("verdict", "") if isinstance(si, dict) else "")
                    idea_id = getattr(si, "idea_id", si.get("idea_id", "") if isinstance(si, dict) else "")
                    print(f"   * Idea {idea_id} -> Rank {rank} [{verdict.upper()}]: {yes_count}/{total} binary checks passed", flush=True)
                print("", flush=True)
            elif node_name == "pitch_writer":
                briefs = patch.get("pitch_briefs", [])
                print(f"\n[Stage] Pitch Writer: Drafted {len(briefs)} pitch briefs in parallel:", flush=True)
                for i, pb in enumerate(briefs):
                    title = getattr(pb, "title", pb.get("title", "")) if isinstance(pb, dict) else pb.title
                    opp = getattr(pb, "market_opportunity", pb.get("market_opportunity", "")) if isinstance(pb, dict) else pb.market_opportunity
                    print(f"   * Brief #{i+1}: {title} | Opp: {opp[:80]}...", flush=True)
                print("", flush=True)
            elif node_name == "critic":
                critique = patch.get("critique")
                if critique:
                    all_pass = getattr(critique, "all_pass", critique.get("all_pass", False) if isinstance(critique, dict) else False)
                    status = getattr(critique, "approval_status", critique.get("approval_status", "revise") if isinstance(critique, dict) else "revise")
                    failing = getattr(critique, "failing_checks", critique.get("failing_checks", []) if isinstance(critique, dict) else [])
                    print(f"\n[Stage] Critic Evaluation: {status.upper()} (All pass: {all_pass}, Failing checks: {len(failing)})\n", flush=True)

    checkpoint = GRAPH.get_state(config)
    return checkpoint.values


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
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass the evidence cache and re-mine live sources (new runs only)",
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
        if args.force_refresh:
            import src.agents.pain_point_miner as ppm

            ppm.force_refresh = True
            print("Force-refresh enabled: bypassing evidence cache.")
        print(f"VentureForge starting: domain='{args.domain}'")
        result = run_pipeline(args.domain, args.max_pain_points)

    # Serialize final state
    if isinstance(result, dict):
        from src.state.schema import VentureForgeState

        try:
            state = VentureForgeState(**result)
            output = state.model_dump(mode="json", exclude_none=True)
        except Exception as e:
            print(f"Warning: Could not convert result to VentureForgeState: {e}")
            output = result
    else:
        output = result.model_dump(mode="json", exclude_none=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nPipeline finished in stage: {output.get('current_stage', 'unknown')}")
    print(f"   Run ID          : {output.get('run_id', 'unknown')}")
    print(f"   Duration        : {output.get('agent_timings', {})}")
    print(f"   Pain points     : {len(output.get('pain_points', []))}")
    print(f"   Ideas           : {len(output.get('ideas', []))}")
    print(f"   Total Pitches   : {len(output.get('pitch_briefs', []))}")
    print(f"   Approved Pitches: {len(output.get('approved_pitches', []))}")
    if output.get("quarantined_pitches"):
        print(
            f"   Quarantined     : {len(output.get('quarantined_pitches', []))} (failed rubric at max revisions)"
        )
    revision_counts = output.get("revision_counts", {})
    total_revisions = sum(revision_counts.values())
    print(f"   Revisions       : {total_revisions} (across {len(revision_counts)} pitches)")
    print(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()
