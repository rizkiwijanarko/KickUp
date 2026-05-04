"""
VentureForge Gradio UI - Real-time multi-agent startup discovery dashboard.
"""

import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator

import gradio as gr

# Placeholder for actual imports when agents are implemented
# from src.main import run_pipeline
# from src.state.schema import VentureState


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


async def run_venture_pipeline(domain: str, progress: gr.Progress) -> AsyncGenerator:
    """
    Run the VentureForge pipeline and yield updates for the Gradio UI.
    This is a placeholder implementation - will be replaced with actual agent calls.
    """
    logs = []
    stats = {
        "pain_points": 0,
        "clusters": 0,
        "ideas": 0,
        "validated": 0,
        "pitch_briefs": 0,
    }

    # Simulate pipeline stages
    stages = [
        ("Orchestrator", "Initializing pipeline...", 0.05),
        ("Trend Scanner", "Scanning Google Trends and HN...", 0.15),
        ("Pain Point Miner", "Mining Reddit for pain points...", 0.25),
        ("Market Research", "Gathering market data...", 0.35),
        ("Competitor Intel", "Analyzing competitors...", 0.45),
        ("Clustering Agent", "Clustering pain points...", 0.60),
        ("Idea Generator", "Generating startup ideas...", 0.70),
        ("Feasibility Agent", "Checking technical feasibility...", 0.75),
        ("Market Validator", "Validating market demand...", 0.80),
        ("Business Model Agent", "Analyzing business models...", 0.85),
        ("Risk Scoring Agent", "Calculating risk scores...", 0.90),
        ("Pitch Writer", "Writing pitch briefs...", 0.95),
        ("Critic Agent", "Reviewing and refining...", 0.98),
        ("Orchestrator", "Pipeline complete!", 1.0),
    ]

    for agent_name, message, pct in stages:
        logs.append(format_agent_log(agent_name, "running", message))
        progress(pct, desc=f"Running {agent_name}...")

        # Update stats based on progress (simulated)
        if pct >= 0.25:
            stats["pain_points"] = int(47 * (pct / 0.45))
        if pct >= 0.60:
            stats["clusters"] = 8
        if pct >= 0.70:
            stats["ideas"] = int(24 * ((pct - 0.70) / 0.20))
        if pct >= 0.90:
            stats["validated"] = stats["ideas"]
        if pct >= 0.98:
            stats["pitch_briefs"] = 5

        # Update previous agent to completed
        if len(logs) > 1:
            prev_log = logs[-2]
            if "🔄" in prev_log:
                logs[-2] = prev_log.replace("🔄", "✅").replace("running", "completed")

        yield {
            logs_display: "\n\n".join(logs[::-1]),
            pain_points_stat: stats["pain_points"],
            clusters_stat: stats["clusters"],
            ideas_stat: stats["ideas"],
            validated_stat: stats["validated"],
            pitch_briefs_stat: stats["pitch_briefs"],
        }

        await asyncio.sleep(0.5)  # Simulate work

    # Final update
    logs[-1] = format_agent_log("Orchestrator", "completed", "Pipeline complete! ✨")
    yield {
        logs_display: "\n\n".join(logs[::-1]),
        pain_points_stat: 47,
        clusters_stat: 8,
        ideas_stat: 24,
        validated_stat: 24,
        pitch_briefs_stat: 5,
        results_output: _generate_sample_results(),
    }


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
                        10, 100, value=50, step=5, label="Max Pain Points to Discover"
                    )
                    max_ideas = gr.Slider(
                        5, 50, value=24, step=1, label="Max Ideas to Generate"
                    )
                    temperature = gr.Slider(
                        0.0, 1.0, value=0.7, step=0.1, label="LLM Temperature"
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
                            value=0, label="Clusters", interactive=False
                        )
                    with gr.Column():
                        ideas_stat = gr.Number(
                            value=0, label="Ideas Generated", interactive=False
                        )
                with gr.Row():
                    with gr.Column():
                        validated_stat = gr.Number(
                            value=0, label="Validated", interactive=False
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
                gr.Markdown("### 📋 Results")
                results_output = gr.Code(
                    language="json",
                    label="Pipeline Output",
                    value='{}',
                )

            with gr.Column():
                gr.Markdown("### 📄 Top Pitch Brief")
                pitch_display = gr.Markdown(
                    value="_Results will appear here after pipeline completion..._"
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
            inputs=[domain_input],
            outputs=[
                logs_display,
                pain_points_stat,
                clusters_stat,
                ideas_stat,
                validated_stat,
                pitch_briefs_stat,
                results_output,
            ],
        )

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
