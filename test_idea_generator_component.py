"""Component-level test for the Idea Generator agent.

Fast, deterministic, offline by mocking the LLM.
Run with:
    uv run test_idea_generator_component.py
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from src.agents.idea_generator import run as run_idea_generator
from src.state.schema import DataSource, Idea, PainPoint, PainPointRubric, PipelineStage, VentureForgeState

logging.basicConfig(level=logging.INFO)


def _make_minimal_state() -> VentureForgeState:
    pp1 = PainPoint(
        id=uuid4(),
        title="Docker Compose is hard",
        description="Developers struggle with complex multi-service local development setups.",
        rubric=PainPointRubric(
            is_genuine_current_frustration=True,
            has_verbatim_quote=True,
            user_segment_specific=True,
        ),
        passes_rubric=True,
        source_url="https://reddit.com/r/docker/comments/abc123",
        raw_quote="I spend more time debugging docker-compose.yml than writing code.",
        source=DataSource.REDDIT,
    )
    pp2 = PainPoint(
        id=uuid4(),
        title="CI debugging is painful",
        description="Developers waste hours reproducing CI failures locally.",
        rubric=PainPointRubric(
            is_genuine_current_frustration=True,
            has_verbatim_quote=True,
            user_segment_specific=True,
        ),
        passes_rubric=True,
        source_url="https://reddit.com/r/devops/comments/def456",
        raw_quote="Why does my test pass locally but fail in CI?",
        source=DataSource.REDDIT,
    )
    return VentureForgeState(domain="developer tools", pain_points=[pp1, pp2], ideas_per_run=2)


def _make_idea_dict(*, title: str, pp_ids: list[Any]) -> dict[str, Any]:
    return {
        "title": title,
        "one_liner": "One-liner under 120 chars.",
        "problem": "A sufficiently long problem statement that passes schema constraints.",
        "solution": "A sufficiently long solution statement that passes schema constraints.",
        "target_user": "Solo developers",
        "key_features": ["A", "B", "C"],
        "addresses_pain_point_ids": pp_ids,
    }


def test_no_pain_points_returns_empty() -> None:
    state = VentureForgeState(domain="test", pain_points=[])
    result = run_idea_generator(state)
    assert result["ideas"] == []
    assert result["current_stage"] == PipelineStage.GENERATING
    assert result["next_node"] == "orchestrator"
    print("  PASS")


def test_wellformed_response_validates_ideas() -> None:
    state = _make_minimal_state()
    pp_ids = [pp.id for pp in state.pain_points]

    good = _make_idea_dict(title="Compose Debugger", pp_ids=[str(pp_ids[0]), str(pp_ids[1])])
    bad_one_ref = _make_idea_dict(title="Bad One Ref", pp_ids=[str(pp_ids[0])])
    bad_invalid_uuid = _make_idea_dict(title="Bad UUID", pp_ids=["not-a-uuid", str(pp_ids[0])])

    with patch("src.agents.idea_generator.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = json.dumps({"ideas": [good, bad_one_ref, bad_invalid_uuid]})
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        result = run_idea_generator(state)

    ideas: list[Idea] = result["ideas"]
    assert len(ideas) == 1, f"Expected only 1 validated idea, got {len(ideas)}"
    assert ideas[0].title == "Compose Debugger"
    assert len(ideas[0].addresses_pain_point_ids) >= 2
    assert all(isinstance(x, UUID) for x in ideas[0].addresses_pain_point_ids)
    print("  PASS")


def test_unwrapped_list_response_supported() -> None:
    state = _make_minimal_state()
    pp_ids = [pp.id for pp in state.pain_points]
    payload = [_make_idea_dict(title="Unwrapped", pp_ids=[str(pp_ids[0]), str(pp_ids[1])])]

    with patch("src.agents.idea_generator.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = json.dumps(payload)
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        result = run_idea_generator(state)

    ideas: list[Idea] = result["ideas"]
    assert len(ideas) == 1
    assert ideas[0].title == "Unwrapped"
    print("  PASS")


def test_live_llm_produces_ideas() -> None:
    """Uses real API call. Requires OPENAI_API_KEY in environment."""
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("  SKIP (no OPENAI_API_KEY)")
        return

    state = _make_minimal_state()
    result = run_idea_generator(state)

    ideas: list[Idea] = result["ideas"]
    assert len(ideas) > 0
    for idea in ideas:
        assert len(idea.addresses_pain_point_ids) >= 2
    print("  PASS")


_TESTS = [
    ("No pain points returns empty", test_no_pain_points_returns_empty),
    ("Well-formed response validates ideas", test_wellformed_response_validates_ideas),
    ("Unwrapped list response supported", test_unwrapped_list_response_supported),
    ("Live LLM (slow)", test_live_llm_produces_ideas),
]


if __name__ == "__main__":
    print("=" * 60)
    print("Idea Generator Component Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, fn in _TESTS:
        print(f"\n[{passed + failed + 1}] {name}...")
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback

            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed:
        import sys

        sys.exit(1)

