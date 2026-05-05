"""Synthetic end-to-end test: bypass Reddit, feed fake data, test new prompts/schemas."""
import json
import logging
from uuid import uuid4

from src.state.schema import (
    DataSource, PainPoint, PainPointRubric, VentureForgeState,
)
from src.agents.idea_generator import run as run_idea_generator
from src.agents.scorer import run as run_scorer
from src.agents.pitch_writer import run as run_pitch_writer
from src.agents.critic import run as run_critic
# from src.agents.orchestrator import orchestrator  # unused import causes exit 127 on Windows

logging.basicConfig(level=logging.INFO)

# Create fake pain points
pp1 = PainPoint(
    id=uuid4(),
    title="Managing Docker Compose files is a nightmare",
    description="Developers struggle with complex multi-service local development setups.",
    rubric=PainPointRubric(is_genuine_current_frustration=True, has_verbatim_quote=True, user_segment_specific=True),
    passes_rubric=True,
    source_url="https://reddit.com/r/programming/comments/abc123",
    raw_quote="I spend more time debugging my docker-compose.yml than writing actual code",
    source=DataSource.REDDIT,
)
pp2 = PainPoint(
    id=uuid4(),
    title="No good tool for local AI model management",
    description="Developers frustrated by lack of simple local LLM runners.",
    rubric=PainPointRubric(is_genuine_current_frustration=True, has_verbatim_quote=True, user_segment_specific=True),
    passes_rubric=True,
    source_url="https://reddit.com/r/LocalLLaMA/comments/def456",
    raw_quote="I wish there was a simple local LLM runner that just works out of the box",
    source=DataSource.REDDIT,
)
pp3 = PainPoint(
    id=uuid4(),
    title="API documentation is always out of date",
    description="Teams struggle to keep OpenAPI specs in sync with actual code.",
    rubric=PainPointRubric(is_genuine_current_frustration=True, has_verbatim_quote=True, user_segment_specific=True),
    passes_rubric=True,
    source_url="https://news.ycombinator.com/item?id=789012",
    raw_quote="Every time we ship a feature the API docs are already wrong",
    source=DataSource.HACKERNEWS,
)
pp4 = PainPoint(
    id=uuid4(),
    title="CI pipelines take forever to debug",
    description="Developers waste hours trying to reproduce CI failures locally.",
    rubric=PainPointRubric(is_genuine_current_frustration=True, has_verbatim_quote=True, user_segment_specific=True),
    passes_rubric=True,
    source_url="https://reddit.com/r/devops/comments/ghi789",
    raw_quote="Why does my test pass locally but fail in CI with the exact same Dockerfile",
    source=DataSource.REDDIT,
)

state = VentureForgeState(
    domain="developer tools",
    max_pain_points=10,
    ideas_per_run=3,
    top_n_pitches=2,
    pain_points=[pp1, pp2, pp3, pp4],
)

print("=" * 60)
print("STEP 1: Idea Generator")
print("=" * 60)
result = run_idea_generator(state)
state = state.model_copy(update=result)
print(f"Ideas generated: {len(state.ideas)}")
for idea in state.ideas:
    print(f"  - {idea.title}: {idea.one_liner}")

if not state.ideas:
    print("No ideas generated — stopping.")
    exit(1)

print("\n" + "=" * 60)
print("STEP 2: Scorer")
print("=" * 60)
result = run_scorer(state)
state = state.model_copy(update=result)
print(f"Scored ideas: {len(state.scored_ideas)}")
for si in state.scored_ideas:
    print(f"  - verdict={si.verdict}, yes_count={si.yes_count}, rank={si.rank}")
    print(f"    reasoning: {si.reasoning_trace[:120]}...")
    print(f"    fatal_flaws: {[(f.flaw[:40], f.severity) for f in si.fatal_flaws]}")
    print(f"    core_assumption: {si.core_assumption}")

if not state.scored_ideas:
    print("No scored ideas — stopping.")
    exit(1)

print("\n" + "=" * 60)
print("STEP 3: Pitch Writer")
print("=" * 60)
result = run_pitch_writer(state)
state = state.model_copy(update=result)
print(f"Pitch briefs: {len(state.pitch_briefs)}")
for brief in state.pitch_briefs:
    print(f"  - {brief.title}: {brief.tagline}")

if not state.pitch_briefs:
    print("No pitch briefs — stopping.")
    exit(1)

print("\n" + "=" * 60)
print("STEP 4: Critic (pass 1)")
print("=" * 60)
try:
    result = run_critic(state)
    state = state.model_copy(update=result)
    if state.critique:
        print(f"  all_pass={state.critique.all_pass}, status={state.critique.approval_status}")
        print(f"  target_agent={state.critique.target_agent}")
        print(f"  failing_checks={state.critique.failing_checks}")
        print(f"  reasoning: {state.critique.reasoning_trace[:120]}...")
    else:
        print("  No critique produced.")

    # If revision needed, simulate one loop
    if state.critique and not state.critique.all_pass and state.can_revise:
        print("\n" + "=" * 60)
        print("STEP 5: Reflection loop (revision)")
        print("=" * 60)
        critique = state.critique
        patch = state.bump_revision(critique)
        patch.update(state.reset_for_revision(critique.target_agent))
        state = state.model_copy(update=patch)
        print(f"  Revised target: {state.next_node}")
        print(f"  Revision count for this idea: {state.get_revision_count(critique.idea_id)}")
except Exception as e:
    import traceback
    print(f"ERROR in critic/revision step: {e}")
    traceback.print_exc()

# Save output
print("Saving output...", flush=True)
output = state.model_dump(exclude_none=True, exclude_computed_fields=True)
with open("test_run_synthetic.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print("\nOutput written to: test_run_synthetic.json", flush=True)
