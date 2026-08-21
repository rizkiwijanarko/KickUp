"""
Pain Point Miner — extracts structured pain points from grounded source evidence.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from uuid import UUID, uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from src.constants import (
    COMMENT_TEXT_MAX_LENGTH,
    MAX_PAIN_POINTS_DEFAULT,
    MIN_PAIN_POINT_LENGTH,
    POST_TITLE_MAX_LENGTH,
)
from src.exceptions import LLMError, LLMJSONParseError
from src.llm.client import coerce_rubric_bools, coerce_yes_no, extract_json, get_llm
from src.llm.prompts import get_prompt
from src.mining import CompositeDataMiner, RawEvidence
from src.models import (
    DataSource,
    PainPoint,
    PainPointEvidence,
    PainPointRubric,
    PipelineStage,
)
from src.state.graph_state import VentureForgeState

logger = logging.getLogger(__name__)


def serialize_evidence(evidence: list[RawEvidence], limit: int = 20) -> list[dict[str, str]]:
    """Serialize evidence items to compact dict format for LLM prompt."""
    return [
        {
            "text": item.text[:COMMENT_TEXT_MAX_LENGTH],
            "url": item.url,
            "source": item.source.value if hasattr(item.source, "value") else str(item.source),
            "post_title": item.title[:POST_TITLE_MAX_LENGTH] if item.title else "",
        }
        for item in evidence[:limit]
    ]


def build_system_prompt() -> str:
    """Build system prompt for pain point extraction."""
    base_prompt = get_prompt("pain_point_miner")
    json_instruction = (
        "\n\n**CRITICAL: Output ONLY the JSON array. "
        "No markdown code fences, no explanations, no preamble. "
        "Start with [ and end with ].**"
    )
    return base_prompt + json_instruction


def build_user_prompt(
    domain: str,
    max_pain_points: int,
    evidence: list[dict[str, str]],
    revision_feedback: str | None = None,
) -> str:
    """Build user prompt for pain point extraction."""
    feedback = revision_feedback or "None"

    return (
        f"Extract up to {max_pain_points} pain points from the "
        f"{len(evidence)} comments below.\n"
        f"Domain: {domain}\n"
        f"Revision feedback (if any): {feedback}\n\n"
        f"COMMENTS:\n{json.dumps(evidence, indent=2)}\n\n"
        "Return a JSON array of pain points. Each must have: "
        "id, title, description, rubric, passes_rubric, source_url, raw_quote, source.\n"
        "The raw_quote MUST be a literal substring from one of the provided comment texts.\n"
        f"Extract exactly {max_pain_points} pain points or fewer if not enough genuine points exist."
    )


def call_llm_for_pain_points(
    domain: str,
    max_pain_points: int,
    evidence: list[RawEvidence],
    revision_feedback: str | None = None,
) -> str:
    """Invoke LLM to extract structured pain points from evidence."""
    llm = get_llm(temperature=0.2, max_tokens=16384, reasoning=False)
    serialized = serialize_evidence(evidence)

    messages = [
        SystemMessage(content=build_system_prompt()),
        HumanMessage(
            content=build_user_prompt(
                domain=domain,
                max_pain_points=max_pain_points,
                evidence=serialized,
                revision_feedback=revision_feedback,
            )
        ),
    ]

    start_time = time.monotonic()
    try:
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"[pain_point_miner] LLM invocation failed after {elapsed:.1f}s: {e}")
        raise LLMError(f"LLM invocation failed: {e}") from e

    elapsed = time.monotonic() - start_time
    logger.info(f"[pain_point_miner] LLM responded in {elapsed:.1f}s")
    return content


def parse_llm_response(response_content: str) -> list[dict[str, Any]]:
    """Parse JSON array from LLM response."""
    try:
        data = extract_json(response_content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    return val
            return [data]
        return []
    except Exception as e:
        logger.warning(f"[pain_point_miner] JSON parse error: {e}")
        raise LLMJSONParseError(f"Failed to parse pain points JSON: {e}") from e


def convert_to_pain_points(raw_items: list[dict[str, Any]]) -> list[PainPoint]:
    """Convert raw parsed dictionaries to validated PainPoint instances."""
    pain_points: list[PainPoint] = []

    for item in raw_items:
        try:
            source_str = item.get("source", "web").lower()
            try:
                source_enum = DataSource(source_str)
            except ValueError:
                source_enum = DataSource.WEB

            evidence_objects: list[PainPointEvidence] = []
            if "evidence" in item and isinstance(item["evidence"], list) and item["evidence"]:
                for ev in item["evidence"]:
                    ev_src = ev.get("source", "web").lower()
                    try:
                        ev_src_enum = DataSource(ev_src)
                    except ValueError:
                        ev_src_enum = DataSource.WEB
                    evidence_objects.append(
                        PainPointEvidence(
                            source_url=ev.get("source_url", "https://news.ycombinator.com"),
                            raw_quote=ev.get("raw_quote", "Quote unavailable"),
                            source=ev_src_enum,
                        )
                    )
            else:
                evidence_objects.append(
                    PainPointEvidence(
                        source_url=item.get("source_url", "https://news.ycombinator.com"),
                        raw_quote=item.get("raw_quote", item.get("description", "")),
                        source=source_enum,
                    )
                )

            raw_id = item.get("id")
            if isinstance(raw_id, UUID):
                pp_id = raw_id
            elif isinstance(raw_id, str) and len(raw_id) == 36:
                try:
                    pp_id = UUID(raw_id)
                except ValueError:
                    pp_id = uuid4()
            else:
                pp_id = uuid4()

            rubric_dict = coerce_rubric_bools(item.get("rubric", {}))
            rubric = PainPointRubric(
                is_genuine_current_frustration=rubric_dict.get("is_genuine_current_frustration", True),
                has_verbatim_quote=rubric_dict.get("has_verbatim_quote", True),
                user_segment_specific=rubric_dict.get("user_segment_specific", True),
            )

            pp = PainPoint(
                id=pp_id,
                title=item["title"],
                description=item["description"],
                rubric=rubric,
                passes_rubric=rubric.all_pass,
                evidence=evidence_objects,
            )
            pain_points.append(pp)
        except Exception as e:
            logger.warning(f"[pain_point_miner] Skipping malformed item: {e}")

    return pain_points


def validate_pain_points(pain_points: list[PainPoint]) -> list[PainPoint]:
    """Filter for pain points meeting rubric and length requirements."""
    validated: list[PainPoint] = []
    for pp in pain_points:
        if not pp.passes_rubric:
            continue
        if len(pp.description.strip()) < MIN_PAIN_POINT_LENGTH:
            continue
        validated.append(pp)
    return validated


# =============================================================================
# PURE DOMAIN FUNCTION
# =============================================================================


def mine_pain_points(
    domain: str,
    max_pain_points: int = MAX_PAIN_POINTS_DEFAULT,
    miner: CompositeDataMiner | None = None,
    revision_feedback: str | None = None,
) -> list[PainPoint]:
    """
    Pure domain function: Ingests evidence via DataMiner and extracts PainPoints with LLM.
    """
    data_miner = miner or CompositeDataMiner()
    evidence = data_miner.mine(domain, limit_per_source=50)

    if not evidence:
        logger.warning(f"[pain_point_miner] No evidence found for domain: {domain}")
        return []

    content = call_llm_for_pain_points(
        domain=domain,
        max_pain_points=max_pain_points,
        evidence=evidence,
        revision_feedback=revision_feedback,
    )
    raw_data = parse_llm_response(content)
    pain_points = convert_to_pain_points(raw_data)
    validated = validate_pain_points(pain_points)
    return validated[:max_pain_points]


# =============================================================================
# GRAPH NODE ADAPTER
# =============================================================================


def run(state: VentureForgeState) -> dict[str, Any]:
    """Graph node adapter: runs pain point miner and returns state patch."""
    logger.info(f"[pain_point_miner] Starting for domain: {state.domain}")
    max_pain_points = state.max_pain_points or MAX_PAIN_POINTS_DEFAULT

    try:
        extracted = mine_pain_points(
            domain=state.domain,
            max_pain_points=max_pain_points,
            revision_feedback=state.revision_feedback,
        )
    except Exception as e:
        logger.error(f"[pain_point_miner] Mining failed: {e}")
        return {
            "pain_points": state.pain_points,
            **state.add_event(
                agent="pain_point_miner",
                stage=PipelineStage.MINING,
                kind="error",
                message=f"Mining failed: {e}",
            ),
        }

    # Append mode with deduplication
    existing_titles = {p.title.lower() for p in state.pain_points}
    new_points = [p for p in extracted if p.title.lower() not in existing_titles]
    final = (state.pain_points + new_points)[:max_pain_points]

    return {
        "pain_points": final,
        "next_node": "idea_generator",
        **state.add_event(
            agent="pain_point_miner",
            stage=PipelineStage.MINING,
            kind="info",
            message=f"Extracted {len(final)} validated pain points for domain '{state.domain}'.",
        ),
    }
