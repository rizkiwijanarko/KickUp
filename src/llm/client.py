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
    model_name = model or config["model"]
    
    # Detect if this is a Qwen3.6 model
    is_qwen36 = "qwen3.6" in model_name.lower() or "qwen/qwen3.6" in model_name.lower()
    
    # Cap max_tokens to avoid vLLM validation errors
    # Qwen3.6-35B-A3B typically has max_model_len around 32k-128k depending on deployment
    # Use a safe default that works with most vLLM deployments
    requested_max_tokens = max_tokens or settings.max_tokens
    safe_max_tokens = min(requested_max_tokens, 32768)  # Safe limit for most deployments
    
    if requested_max_tokens > safe_max_tokens:
        logging.getLogger(__name__).warning(
            f"[llm_client] Requested max_tokens={requested_max_tokens} exceeds safe limit. "
            f"Capping to {safe_max_tokens} to avoid vLLM validation errors."
        )
    
    # Base parameters
    base_params = {
        "base_url": config["base_url"],
        "api_key": config["api_key"] or "sk-dummy",
        "model": model_name,
        "temperature": temperature if temperature is not None else settings.default_temperature,
        "max_tokens": safe_max_tokens,
        "timeout": config["timeout"],
    }
    
    # Add Qwen3.6-specific parameters if detected
    if is_qwen36:
        # Determine if this is a coding task (precise) or general task
        # Reasoning tasks (scorer, critic) use precise coding parameters
        # Non-reasoning tasks use general thinking parameters
        if reasoning:
            # Precise coding tasks: temperature=0.6, top_p=0.95, presence_penalty=0.0
            qwen_params = {
                "temperature": 0.6,
                "top_p": 0.95,
                "extra_body": {
                    "top_k": 20,
                    "repetition_penalty": 1.0,
                    "presence_penalty": 0.0,
                },
            }
        else:
            # General thinking tasks: temperature=1.0, top_p=0.95, presence_penalty=1.5
            # For structured output (JSON), disable thinking mode
            qwen_params = {
                "temperature": 1.0,
                "top_p": 0.95,
                "extra_body": {
                    "top_k": 20,
                    "repetition_penalty": 1.0,
                    "presence_penalty": 1.5,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            }
        
        # Override with user-provided temperature if specified
        if temperature is not None:
            qwen_params["temperature"] = temperature
        
        # Merge Qwen parameters into base parameters
        base_params.update(qwen_params)
        
        logging.getLogger(__name__).info(
            f"[llm_client] Qwen3.6 detected, using {'thinking' if reasoning else 'instruct'} mode with "
            f"temp={qwen_params['temperature']}, top_p={qwen_params.get('top_p', 'default')}, "
            f"max_tokens={safe_max_tokens}"
        )
    
    return ChatOpenAI(**base_params)


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

def strip_thinking_tags(text: str) -> str:
    """Remove Qwen3.6 thinking tags from response text.
    
    Qwen3.6 outputs thinking process wrapped in <think>...</think> tags.
    This function strips those tags to get the actual response.
    
    Example input:
        <think>
        Here's a thinking process:
        1. Analyze...
        </think>
        
        {"result": "actual response"}
    
    Returns: '{"result": "actual response"}'
    """
    if not text:
        return text
    
    # Find and remove <think>...</think> blocks
    import re
    # Use DOTALL flag to match across newlines
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_json(text: str) -> dict | list | None:
    """Extract the first JSON object or array from raw LLM text.

    Handles markdown fences, trailing prose, control characters, and Qwen3.6 thinking tags.
    Returns None if no valid JSON found.
    """
    if not text:
        return None
    
    # Strip Qwen3.6 thinking tags first
    text = strip_thinking_tags(text)
    
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
