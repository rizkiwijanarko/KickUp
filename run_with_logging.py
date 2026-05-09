#!/usr/bin/env python3
"""
Run VentureForge Gradio UI with detailed logging enabled.

This wrapper enables INFO-level logging so you can see:
- Which agent is running at each step
- How long each step takes
- When cancellation is requested
- Where the pipeline gets stuck (if it does)

Usage:
    python run_with_logging.py
    # or
    uv run run_with_logging.py
"""
import logging
import sys

# Configure logging before importing the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)

# Reduce noise from some verbose libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Now import and run the app
from app import main

if __name__ == "__main__":
    print("=" * 80)
    print("VentureForge with detailed logging enabled")
    print("=" * 80)
    print()
    print("Watch for log messages like:")
    print("  - 'Starting pipeline run <run_id>'")
    print("  - 'Step N: <agent_name> (took X.Xs)'")
    print("  - 'Cancellation requested for run <run_id>'")
    print()
    print("If the pipeline hangs, the last logged step will show which agent is stuck.")
    print("=" * 80)
    print()
    
    main()
