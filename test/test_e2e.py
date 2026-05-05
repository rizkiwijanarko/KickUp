"""Quick end-to-end smoke test.

This script is intentionally a *real* E2E run (Reddit scraping + LLM calls).
It includes:
- flush=True prints so progress appears immediately on Windows runners
- recursion_limit so "no data" loops fail fast instead of hanging forever
"""
import json
import os
import time
from src.main import run_pipeline
from src.config import settings

DOMAIN = "developer tools"
MAX_PAIN_POINTS = 10
RECURSION_LIMIT = 30
# Keep this short by default so failures surface quickly.
# If you want a longer run (slow Reddit + LLM), increase this locally.
TIMEOUT_S = 3 * 60

print(f"Starting: domain={DOMAIN!r}", flush=True)
t0 = time.monotonic()

# Fail fast if LLM isn't configured the way this repo expects
if not (settings.llm_api_key or settings.fast_llm_api_key or os.getenv("OPENAI_API_KEY")):
    raise RuntimeError(
        "LLM key not set. Expected LLM_API_KEY (preferred) or FAST_LLM_API_KEY or OPENAI_API_KEY."
    )

# Simple timeout wrapper (Windows-safe)
result = None
err: Exception | None = None

import threading


def _runner() -> None:
    global result, err
    try:
        result = run_pipeline(DOMAIN, max_pain_points=MAX_PAIN_POINTS, recursion_limit=RECURSION_LIMIT)
    except Exception as e:  # bubble up after join
        err = e


th = threading.Thread(target=_runner, daemon=True)
th.start()
th.join(TIMEOUT_S)
if th.is_alive():
    raise TimeoutError(f"E2E timed out after {TIMEOUT_S}s (likely stuck scraping or looping).")
if err:
    raise err
assert result is not None

output = result.model_dump(mode="json", exclude_none=True)
with open("test_run.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

elapsed = time.monotonic() - t0
print(f"Pipeline finished in stage: {result.current_stage} (elapsed {elapsed:.1f}s)", flush=True)
print(f"   Run ID     : {result.run_id}", flush=True)
print(f"   Duration   : {result.agent_timings}", flush=True)
print(f"   Pain points: {len(result.pain_points)}")
print(f"   Ideas      : {len(result.ideas)}")
print(f"   Pitches    : {len(result.pitch_briefs)}")
print(f"   Revisions  : {sum(result.revision_counts.values())} (across {len(result.revision_counts)} pitches)")

if result.scored_ideas:
    for si in result.scored_ideas:
        print(f"   Idea {si.idea_id}: verdict={si.verdict}, yes_count={si.yes_count}, fatal={len([f for f in si.fatal_flaws if f.severity=='fatal'])}")
        if si.reasoning_trace:
            print(f"      reasoning: {si.reasoning_trace[:100]}...")

if result.critique:
    print(f"   Critique: all_pass={result.critique.all_pass}, status={result.critique.approval_status}, target={result.critique.target_agent}")
    if result.critique.reasoning_trace:
        print(f"      reasoning: {result.critique.reasoning_trace[:100]}...")
    print(f"      failing: {result.critique.failing_checks}")

print("Output written to: test_run.json", flush=True)

# Minimal invariants for a "real E2E" run:
assert result.current_stage in ("completed", "failed"), f"Unexpected stage: {result.current_stage}"
