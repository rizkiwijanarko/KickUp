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


def run_pipeline(domain: str, max_pain_points: int | None = None) -> VentureForgeState:
    """Execute the full end-to-end pipeline and return final state."""
    state = VentureForgeState(
        domain=domain,
        max_pain_points=max_pain_points or settings.max_pain_points,
        ideas_per_run=settings.ideas_per_run,
        top_n_pitches=settings.top_n_pitches,
        max_revisions=settings.max_revisions,
    )
    # LangGraph invoke returns updated state
    return GRAPH.invoke(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="VentureForge — AI Startup Discovery")
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Target domain, e.g. 'developer tools'",
    )
    parser.add_argument(
        "--max-pain-points",
        type=int,
        default=None,
        help="Override max pain points to extract",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    print(f"🚀 VentureForge starting: domain='{args.domain}'")
    result = run_pipeline(args.domain, args.max_pain_points)

    # Serialize final state
    output = result.model_dump(mode="json", exclude_none=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Pipeline finished in stage: {result.current_stage}")
    print(f"   Run ID     : {result.run_id}")
    print(f"   Duration   : {result.agent_timings}")
    print(f"   Pain points: {len(result.pain_points)}")
    print(f"   Ideas      : {len(result.ideas)}")
    print(f"   Pitches    : {len(result.pitch_briefs)}")
    print(f"   Revisions  : {result.revision_count}")
    print(f"\n💾 Output written to: {args.output}")


if __name__ == "__main__":
    main()
