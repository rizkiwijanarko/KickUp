"""
Domain models for startup ideas and rubric scoring.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator

from src.models.common import Verdict


class FeasibilityRubric(BaseModel):
    """Binary checks for feasibility (Scorer)."""

    can_be_solved_manually_first: bool
    has_schlep_or_unsexy_advantage: bool
    can_2_3_person_team_build_mvp_in_6_months: bool


class DemandRubric(BaseModel):
    """Binary checks for demand (Scorer)."""

    addresses_at_least_2_pain_points: bool
    is_painkiller_not_vitamin: bool
    has_clear_vein_of_early_adopters: bool


class NoveltyRubric(BaseModel):
    """Binary checks for novelty (Scorer)."""

    differentiated_from_current_behavior: bool
    has_path_out_of_niche: bool


class FatalFlaw(BaseModel):
    """A specific, falsifiable reason an idea might fail."""

    flaw: str
    severity: Literal["fatal", "major", "minor"]


class Idea(BaseModel):
    """A startup idea generated from clustered pain points."""

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., min_length=3, max_length=100)
    one_liner: str = Field(..., max_length=120)
    problem: str = Field(..., min_length=20, max_length=800)
    solution: str = Field(..., min_length=20, max_length=800)
    target_user: str = Field(..., min_length=5, max_length=200)
    key_features: list[str] = Field(default_factory=list, min_length=3, max_length=5)
    addresses_pain_point_ids: list[UUID] = Field(default_factory=list, min_length=2)


class ScoredIdea(BaseModel):
    """An idea with binary rubric evaluation applied by the Scorer."""

    idea_id: UUID
    reasoning_trace: str
    feasibility_rubric: FeasibilityRubric
    demand_rubric: DemandRubric
    novelty_rubric: NoveltyRubric
    core_assumption: str
    fatal_flaws: list[FatalFlaw] = Field(default_factory=list)
    yes_count: int = Field(..., ge=0, le=8)
    total_checks: int = 8
    verdict: Literal["pursue", "explore", "park"]
    one_risk: str = Field(..., max_length=300)
    rank: int | None = None

    @model_validator(mode="after")
    def _derive_verdict(self) -> "ScoredIdea":
        """Derive verdict from yes_count AND fatal flaw severity."""
        has_fatal = any(f.severity == "fatal" for f in self.fatal_flaws)
        if self.yes_count <= 2 or has_fatal:
            self.verdict = "park"
        elif 3 <= self.yes_count <= 5 and not has_fatal:
            self.verdict = "explore"
        elif self.yes_count >= 6 and not has_fatal:
            self.verdict = "pursue"
        return self
