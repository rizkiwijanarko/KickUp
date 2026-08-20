"""
Domain models for critic verification and rubric checks.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field, model_validator

from src.models.common import TargetAgent


class CritiqueRubric(BaseModel):
    """Binary checks applied by the Critic to pitch briefs."""

    all_claims_evidence_backed: bool
    no_hallucinated_source_urls: bool
    tagline_under_12_words: bool
    target_is_contained_fire: bool
    competition_embraced_with_thesis: bool
    minimum_evidence_sources: bool
    scorer_verdict_justified: bool
    validation_plan_complete: bool


class Critique(BaseModel):
    """Output of the Critic agent after reviewing a pitch brief."""

    idea_id: UUID
    reasoning_trace: str
    rubric: CritiqueRubric
    all_pass: bool
    approval_status: Literal["approved", "revise", "max_revisions_reached"]
    failing_checks: list[str] = Field(default_factory=list)
    target_agent: Literal["pain_point_miner", "idea_generator", "pitch_writer"]
    revision_feedback: str = Field(..., min_length=10)

    @model_validator(mode="after")
    def _sync_from_rubric(self) -> "Critique":
        """Ensure all_pass/failing_checks/approval_status/target_agent match rubric."""
        rubric_dict = self.rubric.model_dump()
        self.failing_checks = [k for k, v in rubric_dict.items() if not v]
        self.all_pass = len(self.failing_checks) == 0

        # Approval status derived solely from rubric
        self.approval_status = "approved" if self.all_pass else "revise"

        # Enforce target_agent priority only when revision is required
        if not self.all_pass:
            r = self.rubric
            hallucinated_urls = not r.no_hallucinated_source_urls
            weak_claims = not r.all_claims_evidence_backed
            insufficient_sources = not r.minimum_evidence_sources
            scorer_issue = not r.scorer_verdict_justified
            positioning_failed = (not r.target_is_contained_fire) or (not r.competition_embraced_with_thesis)
            writing_failed = not r.tagline_under_12_words
            validation_plan_failed = not r.validation_plan_complete

            if hallucinated_urls:
                self.target_agent = "pitch_writer"
            elif positioning_failed or scorer_issue:
                self.target_agent = "idea_generator"
            elif insufficient_sources:
                self.target_agent = "pitch_writer"
            elif validation_plan_failed:
                self.target_agent = "pitch_writer"
            elif weak_claims and not hallucinated_urls:
                self.target_agent = "pitch_writer"
            else:
                self.target_agent = "pitch_writer"

        return self
