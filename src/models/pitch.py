"""
Domain models for pitch briefs, competitive landscape, and validation plans.
"""

from __future__ import annotations

from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class CompetitiveLandscape(BaseModel):
    """Competitive analysis for a startup idea."""

    current_behavior: str = Field(
        ...,
        min_length=20,
        description="What customers do today instead of using this product (the real competitor)",
    )
    direct_competitors: str = Field(
        ...,
        min_length=10,
        description="Companies solving the same problem, if any",
    )
    real_enemy: str = Field(
        ...,
        min_length=10,
        description="The specific habit or behavior this product must replace",
    )

    @field_validator("direct_competitors", mode="before")
    @classmethod
    def convert_list_to_string(cls, v: object) -> object:
        """Convert list of competitors to comma-separated string if passed as list."""
        if isinstance(v, list):
            return ", ".join(str(item) for item in v)
        return v


class ValidationPlan(BaseModel):
    """Customer discovery and validation strategy."""

    discovery_questions: list[str] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="5 open-ended questions for customer discovery (no yes/no questions)",
    )
    validation_criteria: str = Field(
        ...,
        min_length=20,
        description="Specific signals that prove the problem is real and worth solving",
    )


class PitchBrief(BaseModel):
    """A one-page investor pitch brief written for a single idea."""

    idea_id: UUID
    title: str = Field(..., min_length=3, max_length=120)
    tagline: str = Field(..., max_length=120)
    problem: str = Field(..., min_length=20)
    solution: str = Field(..., min_length=20)
    target_user: str = Field(..., min_length=5)
    market_opportunity: str = Field(..., min_length=20)
    competitive_landscape: CompetitiveLandscape
    differentiation: str = Field(
        ...,
        min_length=20,
        description="Why someone would switch from current behavior to this product",
    )
    validation_plan: ValidationPlan
    business_model: str = Field(..., min_length=20)
    go_to_market: str = Field(..., min_length=20)
    key_risk: str = Field(..., min_length=10)
    next_steps: str = Field(..., min_length=10)
    evidence_links: list[str] = Field(..., min_length=1)
    markdown_content: str = Field(..., min_length=100)
    revision_count: int = Field(default=0, ge=0, le=2)

    @field_validator("tagline")
    @classmethod
    def validate_tagline_word_count(cls, v: str) -> str:
        word_count = len(v.split())
        if word_count > 12:
            raise ValueError(f"Tagline must be under 12 words (got {word_count})")
        return v
