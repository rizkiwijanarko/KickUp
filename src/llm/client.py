"""
VentureForge LLM Client
========================
Provider-agnostic OpenAI-compatible factory.
Switch LLM provider by changing LLM_BASE_URL / LLM_API_KEY in env.

Usage:
    from src.llm.client import get_llm
    llm = get_llm(temperature=0.1)
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config import settings


def get_llm(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    reasoning: bool = False,
) -> BaseChatModel:
    """
    Return a configured LLM instance.

    Args:
        temperature: Override default temperature (0.0–2.0).
        max_tokens: Override default max_tokens.
        model: Override default model name.
        reasoning: True for heavy reasoning tasks (scorer, critic) → uses
                   the large model. False for fast generative tasks.
    """
    config = settings.get_llm_config(reasoning=reasoning)
    return ChatOpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"] or "sk-dummy",  # prevent KeyError if empty
        model=model or config["model"],
        temperature=temperature if temperature is not None else settings.default_temperature,
        max_tokens=max_tokens or settings.max_tokens,
        timeout=config["timeout"],
    )


@lru_cache(maxsize=32)
def get_structured_llm(
    output_schema: type,
    *,
    temperature: float | None = None,
    model: str | None = None,
    reasoning: bool = False,
) -> BaseChatModel:
    """
    Return an LLM configured with a Pydantic output schema for structured generation.

    Args:
        output_schema: A Pydantic v2 BaseModel subclass describing the desired output.
        temperature: Override default temperature.
        model: Override default model name.
        reasoning: True for heavy reasoning tasks.
    """
    base = get_llm(temperature=temperature, model=model, reasoning=reasoning)
    return base.with_structured_output(output_schema)


# ------------------------------------------------------------------
# JSON extraction helper — robust against LLM formatting quirks
# ------------------------------------------------------------------

def extract_json(text: str) -> dict | list | None:
    """Extract the first JSON object or array from raw LLM text.

    Handles markdown fences, trailing prose, and control characters.
    Returns None if no valid JSON found.
    """
    if not text:
        return None
    
    # Pre-clean: strip whitespace
    text = text.strip()

    # Find first structural char
    start_idx = -1
    for ch in ("[", "{"):
        idx = text.find(ch)
        if idx != -1 and (start_idx == -1 or idx < start_idx):
            start_idx = idx

    # Find last matching char
    end_idx = -1
    for ch in ("]", "}"):
        idx = text.rfind(ch)
        if idx != -1 and idx > end_idx:
            end_idx = idx

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        # Fallback to direct load if no markers found but text looks like JSON
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError:
            return None

    candidate = text[start_idx : end_idx + 1]

    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError:
        # If widest boundary fails, maybe there's garbage between blocks?
        # For now, we stick to widest as per instructions.
        return None


def coerce_yes_no(value: str | bool) -> bool:
    """Convert 'yes'/'no' strings to bool, passing through bool values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "yes"
    return bool(value)


def coerce_rubric_bools(rubric_dict: dict) -> dict:
    """Convert all 'yes'/'no' string values in a rubric dict to booleans."""
    return {k: coerce_yes_no(v) for k, v in rubric_dict.items()}
