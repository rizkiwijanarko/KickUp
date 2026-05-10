"""
VentureForge Gradio UI for HuggingFace Spaces
================================================
Web interface for the autonomous multi-agent startup discovery system.
"""

import json
import time
from typing import Literal

import gradio as gr
from src.config import settings
from src.main import make_initial_state
from src.run_controller import (
    is_cancel_requested,
    is_run_active,
    poll_state,
    request_cancel,
    start_run,
)
from src.state.schema import PipelineStage


# =============================================================================
# DOMAIN RECOMMENDATIONS
# =============================================================================

DOMAIN_RECOMMENDATIONS = [
    "developer tools",
    "healthcare",
    "finance & fintech",
    "education",
    "food service",
    "e-commerce",
    "marketing & social media",
    "real estate",
    "transportation",
    "AI & machine learning",
    "productivity tools",
    "fashion & retail",
    "sports & fitness",
    "agriculture",
    "content creation",
]


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================


def format_state_summary(state) -> str:
    """Format a summary of the current pipeline state."""
    if state is None:
        return "No active run"

    status_emoji = {
        PipelineStage.IDLE: "💤",
        PipelineStage.MINING: "⛏️",
        PipelineStage.GENERATING: "💡",
        PipelineStage.SCORING: "📊",
        PipelineStage.WRITING: "✍️",
        PipelineStage.CRITIQUING: "🔍",
        PipelineStage.REVISING: "🔄",
        PipelineStage.COMPLETED: "✅",
        PipelineStage.FAILED: "❌",
        PipelineStage.CANCELLED: "🛑",
    }

    emoji = status_emoji.get(state.current_stage, "❓")
    return (
        f"{emoji} **Stage:** {state.current_stage}\n"
        f"🆔 **Run ID:** {state.run_id}\n"
        f"📌 **Domain:** {state.domain}\n"
        f"⏱️ **Duration:** {state.agent_timings}\n"
        f"😫 **Pain Points:** {len(state.pain_points)}\n"
        f"💡 **Ideas:** {len(state.ideas)}\n"
        f"📝 **Pitches:** {len(state.pitch_briefs)}\n"
    )


def format_pain_points(state) -> str:
    """Format pain points for display."""
    if not state or not state.pain_points:
        return "No pain points yet"

    output = []
    for i, pp in enumerate(state.pain_points, 1):
        evidence_count = len(pp.evidence) if hasattr(pp, 'evidence') else 1
        output.append(
            f"### {i}. {pp.title} ({evidence_count} source{'s' if evidence_count > 1 else ''})\n"
            f"**Description:** {pp.description}\n"
            f"**Passes Rubric:** {pp.passes_rubric}\n\n"
        )
        
        # Show all evidence sources
        if hasattr(pp, 'evidence') and pp.evidence:
            output.append("**Evidence:**\n")
            for j, ev in enumerate(pp.evidence, 1):
                output.append(
                    f"{j}. [{ev.source.value}]({ev.source_url})\n"
                    f"   > \"{ev.raw_quote}\"\n\n"
                )
        else:
            # Backward compatibility
            output.append(
                f"**Source:** {pp.source.value} | [Link]({pp.source_url})\n"
                f"**Quote:** \"{pp.raw_quote}\"\n\n"
            )
    return "".join(output)


def format_ideas(state) -> str:
    """Format ideas for display."""
    if not state or not state.ideas:
        return "No ideas yet"

    output = []
    for i, idea in enumerate(state.ideas, 1):
        output.append(
            f"### {i}. {idea.title}\n"
            f"**One-liner:** {idea.one_liner}\n"
            f"**Target User:** {idea.target_user}\n"
            f"**Problem:** {idea.problem}\n"
            f"**Solution:** {idea.solution}\n"
            f"**Key Features:**\n"
        )
        for feature in idea.key_features:
            output.append(f"- {feature}\n")
        output.append("\n")
    return "".join(output)


def format_scored_ideas(state) -> str:
    """Format scored ideas with rubrics."""
    if not state or not state.scored_ideas:
        return "No scored ideas yet"

    output = []
    for i, scored in enumerate(state.scored_ideas, 1):
        idea = next((idea for idea in state.ideas if idea.id == scored.idea_id), None)
        idea_title = idea.title if idea else "Unknown"

        output.append(
            f"### {i}. {idea_title}\n"
            f"**Verdict:** {scored.verdict.upper()} | **Yes Count:** {scored.yes_count}/8\n"
            f"**Core Assumption:** {scored.core_assumption}\n"
            f"**One Risk:** {scored.one_risk}\n\n"
        )

        if scored.fatal_flaws:
            output.append("**Fatal Flaws:**\n")
            for flaw in scored.fatal_flaws:
                output.append(f"- {flaw.severity.upper()}: {flaw.flaw}\n")
            output.append("\n")

        output.append(
            "**Feasibility:**\n"
            f"- Manual-first: {scored.feasibility_rubric.can_be_solved_manually_first}\n"
            f"- Schlep advantage: {scored.feasibility_rubric.has_schlep_or_unsexy_advantage}\n"
            f"- 2-3 person MVP in 6mo: {scored.feasibility_rubric.can_2_3_person_team_build_mvp_in_6_months}\n\n"
        )

        output.append(
            "**Demand:**\n"
            f"- Addresses 2+ pain points: {scored.demand_rubric.addresses_at_least_2_pain_points}\n"
            f"- Painkiller not vitamin: {scored.demand_rubric.is_painkiller_not_vitamin}\n"
            f"- Clear early adopters: {scored.demand_rubric.has_clear_vein_of_early_adopters}\n\n"
        )

        output.append(
            "**Novelty:**\n"
            f"- Differentiated: {scored.novelty_rubric.differentiated_from_current_behavior}\n"
            f"- Path out of niche: {scored.novelty_rubric.has_path_out_of_niche}\n\n"
        )

        output.append("---\n\n")
    return "".join(output)


def format_pitches(state) -> str:
    """Format pitch briefs for display."""
    if not state or not state.pitch_briefs:
        return "No pitches yet"

    output = []
    for i, pitch in enumerate(state.pitch_briefs, 1):
        output.append(
            f"### {i}. {pitch.title}\n"
            f"**Tagline:** {pitch.tagline}\n"
            f"**Target User:** {pitch.target_user}\n"
            f"**Revision Count:** {pitch.revision_count}\n\n"
        )

        if pitch.evidence_links:
            output.append("**Evidence Links:**\n")
            for link in pitch.evidence_links:
                output.append(f"- [{link}]({link})\n")
            output.append("\n")

        output.append(f"## Full Pitch\n\n{pitch.markdown_content}\n\n---\n\n")
    return "".join(output)


def format_critiques(state) -> str:
    """Format critiques for display."""
    if not state or not state.critiques:
        return "No critiques yet"

    output = []
    for i, critique in enumerate(state.critiques, 1):
        idea = next((idea for idea in state.ideas if idea.id == critique.idea_id), None)
        idea_title = idea.title if idea else "Unknown"

        status_emoji = "✅" if critique.approval_status == "approved" else "🔄"
        output.append(
            f"### {i}. {idea_title}\n"
            f"{status_emoji} **Status:** {critique.approval_status.upper()}\n"
            f"**Target Agent:** {critique.target_agent}\n"
            f"**All Checks Pass:** {critique.all_pass}\n\n"
        )

        if critique.failing_checks:
            output.append("**Failing Checks:**\n")
            for check in critique.failing_checks:
                output.append(f"- {check}\n")
            output.append("\n")

        output.append(f"**Feedback:** {critique.revision_feedback}\n\n---\n\n")
    return "".join(output)


def export_json(state) -> str:
    """Export state as JSON."""
    if state is None:
        return ""
    return json.dumps(state.model_dump(mode="json", exclude_none=True), indent=2)


def export_markdown(state) -> str:
    """Export state as Markdown report."""
    if state is None:
        return ""

    md = f"# VentureForge Report\n\n"
    md += f"**Domain:** {state.domain}\n"
    md += f"**Run ID:** {state.run_id}\n"
    md += f"**Final Stage:** {state.current_stage}\n\n"

    md += "## Summary\n\n"
    md += f"- Pain Points: {len(state.pain_points)}\n"
    md += f"- Ideas: {len(state.ideas)}\n"
    md += f"- Scored Ideas: {len(state.scored_ideas)}\n"
    md += f"- Pitches: {len(state.pitch_briefs)}\n"
    md += f"- Critiques: {len(state.critiques)}\n\n"

    if state.pain_points:
        md += "## Pain Points\n\n"
        md += format_pain_points(state)

    if state.ideas:
        md += "## Ideas\n\n"
        md += format_ideas(state)

    if state.scored_ideas:
        md += "## Scored Ideas\n\n"
        md += format_scored_ideas(state)

    if state.pitch_briefs:
        md += "## Pitch Briefs\n\n"
        md += format_pitches(state)

    if state.critiques:
        md += "## Critiques\n\n"
        md += format_critiques(state)

    return md


# =============================================================================
# PIPELINE CONTROL
# =============================================================================


def start_pipeline(
    domain: str,
    max_pain_points: int,
    ideas_per_run: int,
    top_n_pitches: int,
    max_revisions: int,
) -> tuple[str, str, str, str, str, str, str, str]:
    """Start the pipeline and return initial UI state."""
    try:
        run_id = start_run(
            domain=domain,
            max_pain_points=max_pain_points,
            ideas_per_run=ideas_per_run,
        )

        # Poll for initial state
        for _ in range(10):  # Wait up to 5 seconds for initial state
            time.sleep(0.5)
            state = poll_state(run_id)
            if state is not None:
                break

        if state is None:
            return (
                run_id,
                "Starting...",
                "Initializing...",
                "Initializing...",
                "Initializing...",
                "Initializing...",
                "Initializing...",
                "",
            )

        return (
            run_id,
            format_state_summary(state),
            format_pain_points(state),
            format_ideas(state),
            format_scored_ideas(state),
            format_pitches(state),
            format_critiques(state),
            "",
        )
    except Exception as e:
        error_msg = f"Error starting pipeline: {str(e)}"
        return (
            "",
            error_msg,
            error_msg,
            error_msg,
            error_msg,
            error_msg,
            error_msg,
            "",
        )


def update_progress(run_id: str) -> tuple[str, str, str, str, str, str]:
    """Poll for state updates and return new UI state."""
    if not run_id or not is_run_active():
        return gr.skip()

    state = poll_state(run_id)
    if state is None:
        return gr.skip()

    return (
        format_state_summary(state),
        format_pain_points(state),
        format_ideas(state),
        format_scored_ideas(state),
        format_pitches(state),
        format_critiques(state),
    )


def stop_pipeline(run_id: str) -> str:
    """Request pipeline cancellation."""
    if run_id:
        request_cancel(run_id)
        return "Stop requested. The pipeline will complete the current step and stop."
    return "No active run to stop."


def clear_cache() -> str:
    """Clear the LangGraph checkpoint cache and scraper caches."""
    import shutil
    from pathlib import Path
    
    # Check if any runs are active
    from src.run_controller import _active_runs
    if _active_runs:
        return "❌ Cannot clear cache while runs are active. Stop all runs first."
    
    cache_dir = Path(settings.cache_dir)
    if cache_dir.exists():
        try:
            # Remove all cache contents
            shutil.rmtree(cache_dir)
            # Recreate empty cache directory
            cache_dir.mkdir(parents=True, exist_ok=True)
            return "✅ Cache cleared successfully. All previous run data has been deleted."
        except PermissionError as e:
            return f"❌ Cache is locked (database in use). Stop the app and clear manually: rm -rf {cache_dir}"
        except Exception as e:
            return f"❌ Error clearing cache: {str(e)}"
    return "ℹ️ Cache directory does not exist."


# =============================================================================
# UI CONSTRUCTION
# =============================================================================


def create_ui() -> gr.Blocks:
    """Create and return the Gradio UI."""
    with gr.Blocks(title="VentureForge - AI Startup Discovery") as app:
        gr.Markdown(
            """
            # 🚀 VentureForge
            ### AI-Powered Startup Discovery System

            Enter a domain to discover startup ideas using autonomous multi-agent analysis.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                domain_recommendation = gr.Dropdown(
                    choices=DOMAIN_RECOMMENDATIONS,
                    label="Domain Recommendations",
                    info="Select a recommended domain or type your own below",
                )
                
                domain_input = gr.Textbox(
                    label="Domain",
                    placeholder="e.g., developer tools, healthcare, fintech",
                    value="developer tools",
                )

            with gr.Column(scale=1):
                start_btn = gr.Button("🚀 Start Discovery", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Stop", variant="stop")
                clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary")

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                max_pain_points = gr.Slider(
                    minimum=5,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Max Pain Points",
                )
                ideas_per_run = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Ideas per Run",
                )
                top_n_pitches = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Top N Pitches",
                )
                max_revisions = gr.Slider(
                    minimum=0,
                    maximum=5,
                    value=settings.max_revisions,
                    step=1,
                    label="Max Revisions",
                )

        run_id_display = gr.Textbox(label="Run ID", interactive=False, visible=True)

        with gr.Tabs() as tabs:
            with gr.TabItem("📊 Progress"):
                progress_display = gr.Markdown("No active run")

            with gr.TabItem("😫 Pain Points"):
                pain_points_display = gr.Markdown("No pain points yet")

            with gr.TabItem("💡 Ideas"):
                ideas_display = gr.Markdown("No ideas yet")

            with gr.TabItem("📈 Scored Ideas"):
                scored_ideas_display = gr.Markdown("No scored ideas yet")

            with gr.TabItem("📝 Pitches"):
                pitches_display = gr.Markdown("No pitches yet")

            with gr.TabItem("🔍 Critiques"):
                critiques_display = gr.Markdown("No critiques yet")

            with gr.TabItem("📥 Export"):
                with gr.Row():
                    export_json_btn = gr.Button("Export JSON", variant="secondary")
                    export_md_btn = gr.Button("Export Markdown", variant="secondary")

                export_output = gr.Code(label="Exported Data", language="json", interactive=False)

        status_message = gr.Textbox(label="Status", interactive=False, visible=True)

        # Event handlers
        domain_recommendation.change(
            fn=lambda x: x if x else "",
            inputs=[domain_recommendation],
            outputs=[domain_input],
        )
        
        start_btn.click(
            fn=start_pipeline,
            inputs=[domain_input, max_pain_points, ideas_per_run, top_n_pitches, max_revisions],
            outputs=[
                run_id_display,
                progress_display,
                pain_points_display,
                ideas_display,
                scored_ideas_display,
                pitches_display,
                critiques_display,
                status_message,
            ],
        )

        stop_btn.click(
            fn=stop_pipeline,
            inputs=[run_id_display],
            outputs=[status_message],
        )

        clear_cache_btn.click(
            fn=clear_cache,
            inputs=[],
            outputs=[status_message],
        )

        # Auto-refresh progress every 2 seconds when run is active
        timer = gr.Timer(2.0)
        timer.tick(
            fn=update_progress,
            inputs=[run_id_display],
            outputs=[
                progress_display,
                pain_points_display,
                ideas_display,
                scored_ideas_display,
                pitches_display,
                critiques_display,
            ],
        )

        export_json_btn.click(
            fn=lambda rid: export_json(poll_state(rid)),
            inputs=[run_id_display],
            outputs=[export_output],
        )

        export_md_btn.click(
            fn=lambda rid: export_markdown(poll_state(rid)),
            inputs=[run_id_display],
            outputs=[export_output],
        )

    return app


def main() -> None:
    """Launch the Gradio UI."""
    ui = create_ui()
    # Use 127.0.0.1 for local development on Windows for better compatibility
    host = "127.0.0.1" if settings.gradio_host == "0.0.0.0" else settings.gradio_host
    ui.launch(
        server_name=host,
        server_port=settings.gradio_port,
        share=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
