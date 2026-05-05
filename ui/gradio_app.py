"""
VentureForge Gradio UI - Real-time multi-agent startup discovery dashboard.
"""

import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator

import gradio as gr

from src.config import settings
from src.run_controller import (
    RunInProgressError,
    start_run,
    poll_state,
    request_cancel,
    is_cancel_requested,
)
from src.state.schema import PipelineStage, VentureForgeState
from src.tools.reddit_scraper import resolve_domain


CURRENT_RUN_ID: str | None = None


def format_agent_log(agent_name: str, status: str, message: str) -> str:
    """Format agent execution log entry."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji = {
        "running": "🔄",
        "completed": "✅",
        "error": "❌",
        "idle": "⏳",
    }.get(status, "⏳")
    return f"[{timestamp}] {emoji} **{agent_name}**: {message}"


async def run_venture_pipeline(
    domain: str,
    max_pain_points_val: int,
    ideas_per_run_val: int,
    progress: gr.Progress,
) -> AsyncGenerator:
    """Run the real VentureForge pipeline and yield live UI updates.

    Uses ``src.run_controller`` to start a background run and polls the
    latest VentureForgeState from the LangGraph SQLite checkpointer.
    """
    logs: list[str] = []

    # Start a new run (one at a time)
    global CURRENT_RUN_ID

    try:
        run_id = start_run(
            domain,
            max_pain_points=int(max_pain_points_val),
            ideas_per_run=int(ideas_per_run_val),
        )
        CURRENT_RUN_ID = run_id
    except RunInProgressError as e:
        logs.append(format_agent_log("Orchestrator", "error", str(e)))
        yield {
            logs_display: "\n\n".join(logs[::-1]),
            pain_points_stat: 0,
            clusters_stat: 0,
            ideas_stat: 0,
            validated_stat: 0,
            pitch_briefs_stat: 0,
            results_output: "{}",
        }
        return

    logs.append(
        format_agent_log(
            "Orchestrator",
            "running",
            f"Started run_id={run_id!r} for domain '{domain}'.",
        )
    )

    state: VentureForgeState | None = None
    while True:
        progress(0.0, desc="Polling pipeline state...")
        # If user requested cancellation, stop updating UI after
        # showing the latest available state.
        if is_cancel_requested(run_id):
            break

        state = poll_state(run_id)
        if state is None:
            # No checkpoints yet — keep waiting
            yield {
                logs_display: "\n\n".join(logs[::-1]),
                pain_points_stat: 0,
                clusters_stat: 0,
                ideas_stat: 0,
                validated_stat: 0,
                pitch_briefs_stat: 0,
                results_output: "{}",
                pitch_display: "_Initializing pipeline..._",
                under_the_hood: "_Awaiting first checkpoint..._",
            }
            await asyncio.sleep(0.5)
            continue

        # Live stats
        pain_points_count = len(state.pain_points)
        ideas_count = len(state.ideas)
        scored_count = len(state.scored_ideas)
        briefs_count = len(state.pitch_briefs)
        revisions_count = sum(state.revision_counts.values())

        # Render events into a log (last 40 events)
        rendered_events: list[str] = []
        for ev in reversed(state.events[-40:]):
            ts = ev.timestamp.astimezone().strftime("%H:%M:%S")
            rendered_events.append(
                f"[{ts}] **{ev.agent}** ({ev.stage.value}): {ev.message}"
            )
        logs_markdown = "\n\n".join(rendered_events) if rendered_events else "_Waiting for agent events..._"

        # Approximate progress from stage order
        stage_order = [
            PipelineStage.MINING,
            PipelineStage.GENERATING,
            PipelineStage.SCORING,
            PipelineStage.WRITING,
            PipelineStage.CRITIQUING,
            PipelineStage.REVISING,
            PipelineStage.COMPLETED,
            PipelineStage.FAILED,
            PipelineStage.CANCELLED,
        ]
        try:
            idx = stage_order.index(state.current_stage)
            pct = idx / max(1, len(stage_order) - 1)
        except ValueError:
            pct = 0.0
        progress(pct, desc=f"Stage: {state.current_stage.value}")

        # Final summary JSON on terminal
        is_terminal = state.current_stage in (
            PipelineStage.COMPLETED,
            PipelineStage.FAILED,
            PipelineStage.CANCELLED,
        )
        # Build evidence & rubrics markdown for top idea and one parked idea
        evidence_lines: list[str] = []
        if state.filtered_pain_points:
            evidence_lines.append("### 🔍 Evidence from Reddit (sample)")
            for pp in state.filtered_pain_points[:3]:
                excerpt = pp.raw_quote.strip()
                if len(excerpt) > 160:
                    excerpt = excerpt[:157] + "..."
                evidence_lines.append(
                    f"- **{pp.title}** – {excerpt} [Source]({pp.source_url})"
                )
            evidence_lines.append("")

        # Helper to render a rubric block
        def _rubric_block(label: str, scored) -> list[str]:
            if scored is None:
                return []
            lines: list[str] = [f"### {label}: {scored.verdict.upper()} (yes_count={scored.yes_count})"]
            lines.append("**Feasibility**")
            fr = scored.feasibility_rubric
            lines.append(f"- can_be_solved_manually_first: {'✅' if fr.can_be_solved_manually_first else '❌'}")
            lines.append(f"- has_schlep_or_unsexy_advantage: {'✅' if fr.has_schlep_or_unsexy_advantage else '❌'}")
            lines.append(f"- can_2_3_person_team_build_mvp_in_6_months: {'✅' if fr.can_2_3_person_team_build_mvp_in_6_months else '❌'}")
            lines.append("**Demand**")
            dr = scored.demand_rubric
            lines.append(f"- addresses_at_least_2_pain_points: {'✅' if dr.addresses_at_least_2_pain_points else '❌'}")
            lines.append(f"- is_painkiller_not_vitamin: {'✅' if dr.is_painkiller_not_vitamin else '❌'}")
            lines.append(f"- has_clear_vein_of_early_adopters: {'✅' if dr.has_clear_vein_of_early_adopters else '❌'}")
            lines.append("**Novelty**")
            nr = scored.novelty_rubric
            lines.append(f"- differentiated_from_current_behavior: {'✅' if nr.differentiated_from_current_behavior else '❌'}")
            lines.append(f"- has_path_out_of_niche: {'✅' if nr.has_path_out_of_niche else '❌'}")
            if scored.fatal_flaws:
                lines.append("**Fatal / major flaws**")
                for flaw in scored.fatal_flaws:
                    lines.append(f"- ({flaw.severity}) {flaw.flaw}")
            lines.append("")
            return lines

        top_scored = None
        parked_scored = None
        if state.scored_ideas:
            # Top idea by yes_count then rank
            sorted_scored = sorted(
                state.scored_ideas,
                key=lambda s: (s.yes_count, -(s.rank or 0)),
                reverse=True,
            )
            top_scored = sorted_scored[0]
            parked = [s for s in state.scored_ideas if s.verdict == "park"]
            if parked:
                parked_scored = sorted(parked, key=lambda s: s.yes_count)[0]

        rubric_lines: list[str] = []
        rubric_lines.extend(_rubric_block("Top Idea", top_scored))
        rubric_lines.extend(_rubric_block("Parked Idea", parked_scored))

        pitch_md = "\n".join(evidence_lines + rubric_lines) if (evidence_lines or rubric_lines) else "_No evidence or scored ideas yet._"

        if is_terminal:
            summary = {
                "run_id": state.run_id,
                "current_stage": state.current_stage.value,
                "pain_points": pain_points_count,
                "ideas": ideas_count,
                "scored_ideas": scored_count,
                "pitch_briefs": briefs_count,
                "revisions": revisions_count,
            }
            results_json = json.dumps(summary, indent=2)
        else:
            results_json = "{}"

        # Build "Under the hood" markdown
        category, subreddits = resolve_domain(state.domain)
        uth_lines: list[str] = [
            f"**Run ID:** `{state.run_id}`",
            f"**Domain:** {state.domain}",
            f"**Current stage:** `{state.current_stage.value}`",
            f"**Resolved category:** `{category}`",
            "**Subreddits (static map):** "
            + ", ".join(f"r/{s}" for s in subreddits),
            "",
            "**LLM configuration:**",
            f"- Reasoning base URL: `{settings.llm_base_url}`",
            f"- Reasoning model: `{settings.llm_model}`",
            f"- Fast base URL: `{settings.fast_llm_base_url or settings.llm_base_url}`",
            f"- Fast model: `{settings.fast_llm_model or settings.llm_model}`",
        ]
        if state.agent_timings:
            uth_lines.append("")
            uth_lines.append("**Agent timings (seconds):**")
            for agent_id, t in state.agent_timings.items():
                uth_lines.append(f"- {agent_id}: {t:.2f}")

        under_the_hood_md = "\n".join(uth_lines)

        yield {
            logs_display: logs_markdown,
            pain_points_stat: pain_points_count,
            clusters_stat: scored_count,
            ideas_stat: ideas_count,
            validated_stat: revisions_count,
            pitch_briefs_stat: briefs_count,
            results_output: results_json,
            pitch_display: pitch_md,
            under_the_hood: under_the_hood_md,
        }

        if is_terminal or is_cancel_requested(run_id):
            break

        await asyncio.sleep(0.5)


def _generate_sample_results() -> str:
    """Generate sample results output for demo purposes."""
    sample = {
        "top_ideas": [
            {
                "rank": 1,
                "name": "DevFlow AI",
                "score": 92,
                "description": "AI-powered developer workflow optimization",
                "market_size": "$2.4B",
                "feasibility": "High",
            },
            {
                "rank": 2,
                "name": "CodeReview Bot",
                "score": 88,
                "description": "Automated code review with context awareness",
                "market_size": "$1.8B",
                "feasibility": "High",
            },
            {
                "rank": 3,
                "name": "API Guardian",
                "score": 85,
                "description": "AI-driven API security monitoring",
                "market_size": "$3.1B",
                "feasibility": "Medium",
            },
        ]
    }
    return json.dumps(sample, indent=2)


def create_ui() -> gr.Blocks:
    """Create and configure the Gradio UI."""

    with gr.Blocks(
        title="VentureForge - AI Startup Discovery",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        css="""
        .stat-box { text-align: center; padding: 10px; border-radius: 8px; background: #f8fafc; }
        .stat-number { font-size: 2rem; font-weight: bold; color: #4f46e5; }
        .stat-label { font-size: 0.875rem; color: #64748b; }
        .agent-log { font-family: monospace; font-size: 0.875rem; }
        """,
    ) as app:

        gr.Markdown("""
        # 🚀 VentureForge
        ## Autonomous Multi-Agent Startup Discovery System

        Discover validated startup ideas from real user pain points.
        Powered by AMD Instinct MI300X + LangGraph + Open Source LLMs.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 🎯 Configuration")
                domain_input = gr.Textbox(
                    label="Target Domain",
                    placeholder="e.g., developer tools, remote work, fintech",
                    value="developer tools",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    max_pain_points = gr.Slider(
                        10, 100, value=30, step=5, label="Max Pain Points to Discover"
                    )
                    ideas_per_run = gr.Slider(
                        1, 20, value=5, step=1, label="Ideas per Run"
                    )

                run_btn = gr.Button("🚀 Run Discovery Pipeline", variant="primary", size="lg")
                stop_btn = gr.Button("⏹ Stop", variant="stop")

                # Stats section
                gr.Markdown("### 📊 Live Statistics")
                with gr.Row():
                    with gr.Column():
                        pain_points_stat = gr.Number(
                            value=0, label="Pain Points", interactive=False
                        )
                    with gr.Column():
                        clusters_stat = gr.Number(
                            value=0, label="Scored Ideas", interactive=False
                        )
                    with gr.Column():
                        ideas_stat = gr.Number(
                            value=0, label="Ideas Generated", interactive=False
                        )
                with gr.Row():
                    with gr.Column():
                        validated_stat = gr.Number(
                            value=0, label="Revisions", interactive=False
                        )
                    with gr.Column():
                        pitch_briefs_stat = gr.Number(
                            value=0, label="Pitch Briefs", interactive=False
                        )

            with gr.Column(scale=2):
                # Agent execution logs
                gr.Markdown("### 🔄 Agent Execution Logs")
                logs_display = gr.Markdown(
                    value="_Click 'Run Discovery Pipeline' to start..._",
                    elem_classes=["agent-log"],
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📋 Results (Summary)")
                results_output = gr.Code(
                    language="json",
                    label="Pipeline Summary",
                    value='{}',
                )

            with gr.Column():
                gr.Markdown("### 📄 Evidence & Rubrics")
                pitch_display = gr.Markdown(
                    value="_Results will appear here after pipeline completion..._",
                )

        with gr.Accordion("🔧 Under the hood", open=False):
            under_the_hood = gr.Markdown(
                value="_Run the pipeline to see internal routing, subreddits, and LLM config..._",
            )

        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #64748b; font-size: 0.875rem;">
        Built with ❤️ for AMD AI Hackathon 2025 | 
        <a href="https://github.com/yourusername/ventureforge">GitHub</a>
        </div>
        """)

        # Event handlers
        run_btn.click(
            fn=run_venture_pipeline,
            inputs=[domain_input, max_pain_points, ideas_per_run],
            outputs=[
                logs_display,
                pain_points_stat,
                clusters_stat,
                ideas_stat,
                validated_stat,
                pitch_briefs_stat,
                results_output,
                pitch_display,
                under_the_hood,
            ],
        )

        def _on_stop() -> str:
            global CURRENT_RUN_ID
            if CURRENT_RUN_ID is not None:
                request_cancel(CURRENT_RUN_ID)
                return "_Stop requested. The current run will halt after the current step; partial results remain in history._"
            return "_No active run to stop._"

        stop_btn.click(fn=_on_stop, inputs=None, outputs=[logs_display])

    return app


def main():
    """Entry point for the Gradio UI."""
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
