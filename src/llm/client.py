"""
VentureForge LLM Client
========================
Provider-agnostic OpenAI-compatible factory.
Switch LLM provider by changing LLM_BASE_URL / LLM_API_KEY in .env.

Provider-specific configuration (headers, extra body params, etc.) is
handled by the adapters in ``src/llm/adapters.py``.  The factory here
stays free of provider-identity checks.

Usage:
    from src.llm.client import get_llm
    llm = get_llm(temperature=0.1)
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import json
from typing import cast
import logging
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config import settings
from src.llm.adapters import get_adapter

logger = logging.getLogger(__name__)


def get_llm(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    reasoning: bool = False,
) -> BaseChatModel:
    """Return a configured LLM instance.

    Provider-specific extras (e.g. OpenRouter headers) are applied
    automatically via the registered adapters — no provider-name checks
    live in this function.

    Args:
        temperature: Override default temperature (0.0–2.0).
        max_tokens:  Override default max_tokens.
        model:       Override default model name.
        reasoning:   True for heavy reasoning tasks (scorer, critic) →
                     uses the large model.  False for fast generative tasks.
    """
    config = settings.get_llm_config(reasoning=reasoning)
    base_url: str = config["base_url"]

    base_params: dict[str, Any] = {
        "base_url": base_url,
        "api_key": config["api_key"] or "sk-dummy",
        "model": model or config["model"],
        "temperature": temperature if temperature is not None else settings.default_temperature,
        "max_tokens": max_tokens or settings.max_tokens,
        "timeout": float(config.get("timeout") or 120),
        "max_retries": 3,
    }

    adapter = get_adapter(base_url)
    if adapter:
        extras = adapter.extra_params(base_params, reasoning)
        base_params = {**base_params, **extras}
        logger.debug(f"[llm_client] Using {adapter.name} adapter for base_url={base_url!r}")
    else:
        logger.debug(f"[llm_client] No adapter matched for base_url={base_url!r}; using base params.")

    return ChatOpenAI(**base_params)


@lru_cache(maxsize=32)
def get_structured_llm(
    output_schema: type,
    *,
    temperature: float | None = None,
    model: str | None = None,
    max_tokens: int | None = 16384,
    reasoning: bool = False,
) -> BaseChatModel:
    """Return an LLM configured with a Pydantic output schema for structured generation.

    Args:
        output_schema: A Pydantic v2 BaseModel subclass describing the desired output.
        temperature:   Override default temperature.
        model:         Override default model name.
        max_tokens:    Override default max tokens (defaults to 16384 for structured outputs).
        reasoning:     True for heavy reasoning tasks.
    """
    base = get_llm(temperature=temperature, model=model, max_tokens=max_tokens, reasoning=reasoning)
    return cast(BaseChatModel, base.with_structured_output(output_schema))


# ---------------------------------------------------------------------------
# JSON extraction helper — robust against LLM formatting quirks
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict | list | None:
    """Extract the first JSON object or array from raw LLM text.

    Handles markdown fences, trailing prose, control characters, and
    trailing commas.  Returns None if no valid JSON is found.
    """
    if not text:
        return None

    import re

    text = text.strip()

    # Strategy 1: markdown code blocks (```json ... ``` or ``` ... ```)
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if code_block_match:
        block_text = code_block_match.group(1).strip()
        try:
            return cast(dict | list, json.loads(block_text, strict=False))
        except json.JSONDecodeError:
            cleaned_block = re.sub(r",\s*([\]}])", r"\1", block_text)
            try:
                return cast(dict | list, json.loads(cleaned_block, strict=False))
            except json.JSONDecodeError:
                pass

    # Strategy 2: outermost structural boundaries
    start_idx = -1
    for ch in ("[", "{"):
        idx = text.find(ch)
        if idx != -1 and (start_idx == -1 or idx < start_idx):
            start_idx = idx

    end_idx = -1
    for ch in ("]", "}"):
        idx = text.rfind(ch)
        if idx != -1 and idx > end_idx:
            end_idx = idx

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = text[start_idx : end_idx + 1]
        try:
            return cast(dict | list, json.loads(candidate, strict=False))
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([\]}])", r"\1", candidate)
            try:
                return cast(dict | list, json.loads(cleaned, strict=False))
            except json.JSONDecodeError:
                pass

    # Strategy 3: direct fallback
    try:
        return cast(dict | list, json.loads(text, strict=False))
    except json.JSONDecodeError:
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
