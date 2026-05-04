"""
VentureForge Configuration
==========================
Pydantic v2 settings loaded from environment variables.
Single LLM_BASE_URL switches between OpenAI/OpenRouter/AMD vLLM.

Usage:
    from src.config import settings
    print(settings.llm_base_url)
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings loaded from .env or environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM Provider (agnostic — swap LLM_BASE_URL to switch)
    # ------------------------------------------------------------------
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API base URL",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for the LLM provider",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name served at the endpoint",
    )

    # Optional: separate OpenRouter config (falls back to llm_* if unset)
    openrouter_api_key: str | None = Field(default=None)
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")

    # ------------------------------------------------------------------
    # LLM Generation Parameters (per-agent overrides possible)
    # ------------------------------------------------------------------
    default_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    request_timeout: int = Field(default=120, ge=1)

    # ------------------------------------------------------------------
    # Reddit — NOT required.  We use public `.json` endpoints (no PRAW).
    # Only set these if you later want PRAW features.
    # ------------------------------------------------------------------
    reddit_client_id: str | None = Field(default=None)
    reddit_client_secret: str | None = Field(default=None)
    reddit_user_agent: str = Field(
        default="ventureforge:v1.0 by u/username",
        description="Optional PRAW user agent string",
    )

    # ------------------------------------------------------------------
    # Tavily — used for community-discovery fallback
    # ------------------------------------------------------------------
    tavily_api_key: str | None = Field(default=None)

    # ------------------------------------------------------------------
    # HuggingFace (for AMD vLLM model download)
    # ------------------------------------------------------------------
    hf_token: str | None = Field(default=None)

    # ------------------------------------------------------------------
    # Pipeline Defaults
    # ------------------------------------------------------------------
    max_pain_points: int = Field(default=30, ge=5, le=100)
    ideas_per_run: int = Field(default=5, ge=1, le=20)
    top_n_pitches: int = Field(default=3, ge=1, le=10)
    max_revisions: int = Field(default=2, ge=0, le=5)
    lookback_days: int = Field(default=90, ge=7, le=365)

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------
    cache_dir: str = Field(default=".cache")
    cache_ttl_hours: int = Field(default=24, ge=1)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    gradio_port: int = Field(default=7860, ge=1024, le=65535)
    gradio_host: str = Field(default="0.0.0.0")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("reddit_user_agent", mode="after")
    @classmethod
    def _warn_default_agent(cls, v: str) -> str:
        if "username" in v:
            # Allow it but it's clearly a placeholder
            pass
        return v

    @property
    def tavily_enabled(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def effective_llm_config(self) -> dict:
        """Return the active LLM configuration as a dict."""
        return {
            "base_url": self.llm_base_url,
            "api_key": self.llm_api_key,
            "model": self.llm_model,
            "timeout": self.request_timeout,
        }


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


settings = get_settings()
