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
from src.llm.client import coerce_rubric_bools, extract_json, get_llm
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


def _serialize_evidence(evidence: list[RawEvidence], limit: int = 45) -> list[dict[str, Any]]:
    """Serialize evidence items to compact dict format for LLM prompt."""
    return [
        {
            "text": item.text[:COMMENT_TEXT_MAX_LENGTH],
            "url": item.url,
            "source": item.source.value if hasattr(item.source, "value") else str(item.source),
            "post_title": item.title[:POST_TITLE_MAX_LENGTH] if item.title else "",
            "score": item.score,
        }
        for item in evidence[:limit]
    ]


def _build_system_prompt() -> str:
    """Build system prompt for pain point extraction."""
    base_prompt = get_prompt("pain_point_miner")
    json_instruction = (
        "\n\n**CRITICAL: Output ONLY the JSON array. "
        "No markdown code fences, no explanations, no preamble. "
        "Start with [ and end with ].**"
    )
    return base_prompt + json_instruction


def _build_user_prompt(
    domain: str,
    max_pain_points: int,
    evidence: list[dict[str, Any]],
    revision_feedback: str | None = None,
) -> str:
    """Build user prompt for pain point extraction."""
    feedback = revision_feedback or "None"

    return (
        f"Extract up to {max_pain_points} high-signal, clustered pain points from the "
        f"{len(evidence)} comments below.\n"
        f"Domain: {domain}\n"
        f"Revision feedback (if any): {feedback}\n\n"
        f"COMMENTS (ranked by engagement):\n{json.dumps(evidence, indent=2)}\n\n"
        "Return a JSON array of clustered pain points. Each pain point MUST include:\n"
        "- id: (UUID string)\n"
        "- title: (5-10 words summarizing the pain point)\n"
        "- description: (1-2 sentences explaining the problem)\n"
        "- rubric: {\"is_genuine_current_frustration\": true, \"has_verbatim_quote\": true, \"user_segment_specific\": true}\n"
        "- passes_rubric: true\n"
        "- evidence: array of 1-10 evidence objects, where each object has:\n"
        "    * source_url: exact URL from the comments above\n"
        "    * raw_quote: EXACT verbatim substring from that comment's text\n"
        "    * source: data source enum (\"hackernews\", \"producthunt\", \"web\", \"reddit\", or \"youtube\")\n\n"
        "Rules:\n"
        "1. Cluster similar complaints together — prefer pain points with 2+ evidence items.\n"
        "2. Every raw_quote MUST be an exact literal substring of the provided comments.\n"
        f"Extract up to {max_pain_points} distinct pain points."
    )


def _call_llm_for_pain_points(
    domain: str,
    max_pain_points: int,
    evidence: list[RawEvidence],
    revision_feedback: str | None = None,
) -> str:
    """Invoke LLM to extract structured pain points from evidence."""
    llm = get_llm(temperature=0.2, max_tokens=16384, reasoning=False)
    serialized = _serialize_evidence(evidence)

    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(
            content=_build_user_prompt(
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
        content = str(response.content) if hasattr(response, "content") else str(response)
    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(f"[pain_point_miner] LLM invocation failed after {elapsed:.1f}s: {e}")
        raise LLMError(f"LLM invocation failed: {e}") from e

    elapsed = time.monotonic() - start_time
    logger.info(f"[pain_point_miner] LLM responded in {elapsed:.1f}s")
    return content


def _parse_llm_response(response_content: str) -> list[dict[str, Any]]:
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
        raise LLMJSONParseError(raw_response=response_content, parse_error=str(e)) from e



def _verify_quote_against_corpus(quote: str, corpus: list[RawEvidence]) -> bool:
    """Check if quote or cleaned quote exists in raw evidence texts."""
    clean_quote = " ".join(quote.replace("*", "").replace(">", "").replace('"', "").split()).lower()
    if not clean_quote or len(clean_quote) < 5:
        return False
    for item in corpus:
        clean_text = " ".join(item.text.replace("*", "").replace(">", "").replace('"', "").split()).lower()
        if clean_quote in clean_text or clean_quote[:30] in clean_text:
            return True
    return False


def _convert_to_pain_points(
    raw_items: list[dict[str, Any]],
    evidence_corpus: list[RawEvidence] | None = None,
) -> list[PainPoint]:
    """Convert raw parsed dictionaries to validated PainPoint instances with quote verification."""
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
                    
                    raw_quote = ev.get("raw_quote", "Quote unavailable")
                    # Validate quote against corpus if corpus is provided
                    if evidence_corpus:
                        if not _verify_quote_against_corpus(raw_quote, evidence_corpus):
                            logger.info(f"[pain_point_miner] Dropping ungrounded secondary quote: {raw_quote[:40]}...")
                            continue

                    evidence_objects.append(
                        PainPointEvidence(
                            source_url=ev.get("source_url", "https://news.ycombinator.com"),
                            raw_quote=raw_quote,
                            source=ev_src_enum,
                        )
                    )
            else:
                raw_quote = item.get("raw_quote", item.get("description", ""))
                if evidence_corpus:
                    if _verify_quote_against_corpus(raw_quote, evidence_corpus):
                        evidence_objects.append(
                            PainPointEvidence(
                                source_url=item.get("source_url", "https://news.ycombinator.com"),
                                raw_quote=raw_quote,
                                source=source_enum,
                            )
                        )
                else:
                    evidence_objects.append(
                        PainPointEvidence(
                            source_url=item.get("source_url", "https://news.ycombinator.com"),
                            raw_quote=raw_quote,
                            source=source_enum,
                        )
                    )

            if not evidence_objects:
                logger.warning(f"[pain_point_miner] Skipping item '{item.get('title', '')}' — no verified quotes found.")
                continue

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


def _validate_pain_points(pain_points: list[PainPoint]) -> list[PainPoint]:
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

    content = _call_llm_for_pain_points(
        domain=domain,
        max_pain_points=max_pain_points,
        evidence=evidence,
        revision_feedback=revision_feedback,
    )
    raw_data = _parse_llm_response(content)
    pain_points = _convert_to_pain_points(raw_data, evidence_corpus=evidence)
    validated = _validate_pain_points(pain_points)
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

    # Handle reflection revision vs standard additive mode
    if state.revision_feedback:
        logger.info("[pain_point_miner] Revision mode activated. Updating pain points with fresh extractions.")
        final = extracted[:max_pain_points] if extracted else state.pain_points[:max_pain_points]
    else:
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

