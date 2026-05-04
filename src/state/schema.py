"""
VentureForge State Schema
=========================
Single source of truth for all data passed between agents in the
LangGraph orchestration layer.

All agents read from and write to VentureForgeState instances.
Updates are immutable: agents never mutate state in place. Instead,
they return dict patches that the graph layer merges into a new
copy of the state via model_copy(update=...).

Pydantic v2 is required:
    pip install pydantic>=2.0
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field, model_validator

# =============================================================================
# ENUMS
# =============================================================================


class DataSource(str, Enum):
    """Where a pain point or piece of evidence originated."""

    REDDIT = "reddit"
    HACKERNEWS = "hackernews"


class Verdict(str, Enum):
    """Final recommendation from the Scorer."""

    PURSUE = "pursue"
    EXPLORE = "explore"
    PARK = "park"


class CriticStatus(str, Enum):
    """Outcome of a Critic review."""

    APPROVED = "approved"
    REVISE = "revise"
    REJECT = "reject"


class TargetAgent(str, Enum):
    """Which worker the reflection loop should send the revision to."""

    PAIN_POINT_MINER = "pain_point_miner"
    IDEA_GENERATOR = "idea_generator"
    PITCH_WRITER = "pitch_writer"


class PipelineStage(str, Enum):
    """Current stage of the pipeline."""

    IDLE = "idle"
    MINING = "mining"
    GENERATING = "generating"
    SCORING = "scoring"
    WRITING = "writing"
    CRITIQUING = "critiquing"
    REVISING = "revising"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# RUBRIC MODELS (binary booleans — all checks are True/False)
# =============================================================================


class PainPointRubric(BaseModel):
    """Binary rubric applied by the Pain Point Miner to self-filter output."""

    is_genuine_current_frustration: bool
    has_verbatim_quote: bool
    user_segment_specific: bool

    @computed_field
    @property
    def all_pass(self) -> bool:
        return all(self.model_dump().values())


class FeasibilityRubric(BaseModel):
    """Binary checks for feasibility (Scorer)."""

    can_2_3_person_team_build_mvp_in_6_months: bool
    uses_only_existing_proven_tech: bool
    no_special_regulatory_requirements: bool


class DemandRubric(BaseModel):
    """Binary checks for demand (Scorer)."""

    addresses_at_least_2_pain_points: bool
    pain_points_show_high_severity: bool
    target_user_clearly_defined: bool


class NoveltyRubric(BaseModel):
    """Binary checks for novelty (Scorer)."""

    differentiated_from_obvious_solutions: bool
    leverages_unique_insight: bool


class CriticRubric(BaseModel):
    """Binary checks applied by the Critic to pitch briefs."""

    all_claims_evidence_backed: bool
    no_hallucinated_source_urls: bool
    target_user_specific_not_generic: bool
    honest_risk_disclosure: bool
    no_buzzword_filler: bool
    tagline_under_12_words: bool
    go_to_market_concrete: bool

    @computed_field
    @property
    def all_pass(self) -> bool:
        return all(self.model_dump().values())

    @computed_field
    @property
    def failing_checks(self) -> list[str]:
        return [k for k, v in self.model_dump().items() if not v]


# =============================================================================
# AGENT OUTPUT MODELS
# =============================================================================


class PainPoint(BaseModel):
    """A structured user pain point extracted from Reddit or Hacker News."""

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=10, max_length=500)
    rubric: PainPointRubric
    passes_rubric: bool
    source_url: str = Field(..., min_length=5)
    raw_quote: str = Field(..., min_length=5)
    source: DataSource


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
    feasibility_rubric: FeasibilityRubric
    demand_rubric: DemandRubric
    novelty_rubric: NoveltyRubric
    yes_count: int = Field(..., ge=0, le=8)
    total_checks: int = 8
    verdict: Verdict
    verdict_logic: str = Field(
        default="pursue if yes_count >= 6, explore if 3-5, park if <= 2"
    )
    one_risk: str = Field(..., max_length=300)
    rank: int | None = None

    @model_validator(mode="after")
    def _derive_verdict(self) -> "ScoredIdea":
        """Ensure verdict matches yes_count."""
        if self.yes_count >= 6 and self.verdict != Verdict.PURSUE:
            self.verdict = Verdict.PURSUE
        elif 3 <= self.yes_count <= 5 and self.verdict != Verdict.EXPLORE:
            self.verdict = Verdict.EXPLORE
        elif self.yes_count <= 2 and self.verdict != Verdict.PARK:
            self.verdict = Verdict.PARK
        return self


class PitchBrief(BaseModel):
    """A one-page investor pitch brief written for a single idea."""

    idea_id: UUID
    title: str = Field(..., min_length=3, max_length=120)
    tagline: str = Field(..., max_length=120)
    problem: str = Field(..., min_length=20)
    solution: str = Field(..., min_length=20)
    target_user: str = Field(..., min_length=5)
    market_opportunity: str = Field(..., min_length=20)
    business_model: str = Field(..., min_length=20)
    go_to_market: str = Field(..., min_length=20)
    key_risk: str = Field(..., min_length=10)
    next_steps: str = Field(..., min_length=10)
    evidence_links: list[str] = Field(default_factory=list)
    markdown_content: str = Field(..., min_length=100)
    revision_count: int = Field(default=0, ge=0, le=2)


class Critique(BaseModel):
    """Output of the Critic agent after reviewing pitch brief(s)."""

    idea_id: UUID | None = None
    rubric: CriticRubric
    all_pass: bool
    approval_status: CriticStatus
    failing_checks: list[str] = Field(default_factory=list)
    target_agent: TargetAgent
    target_agent_logic: str = Field(
        default="pain_point_miner if evidence issues, "
                "idea_generator if fundamental problems, "
                "pitch_writer if writing/tone issues"
    )
    revision_feedback: str = Field(..., min_length=10)

    @model_validator(mode="after")
    def _sync_all_pass(self) -> "Critique":
        """Ensure all_pass matches rubric and approval_status."""
        self.all_pass = self.rubric.all_pass
        if self.all_pass:
            self.approval_status = CriticStatus.APPROVED
        elif self.approval_status == CriticStatus.APPROVED:
            self.approval_status = CriticStatus.REVISE
        return self


# =============================================================================
# SHARED STATE
# =============================================================================


class VentureForgeState(BaseModel):
    """
    The single shared state object passed between all LangGraph nodes.

    Nodes should never mutate this object in place. Instead, each node
    returns a ``dict`` of field updates; LangGraph merges them into a
    new copy via ``model_copy(update=...)``.

    Immutable update example inside a node:
        return {
            "pain_points": new_pain_points,
            "current_stage": PipelineStage.GENERATING,
            "next_node": "idea_generator",
        }
    """

    # -----------------------------------------------------------------
    # Input / Run configuration
    # -----------------------------------------------------------------
    domain: str = Field(..., min_length=2, max_length=100)
    max_pain_points: int = Field(default=30, ge=5, le=100)
    ideas_per_run: int = Field(default=5, ge=1, le=20)
    top_n_pitches: int = Field(default=3, ge=1, le=10)
    max_revisions: int = Field(default=2, ge=0, le=5)

    # -----------------------------------------------------------------
    # Pipeline data (populated by worker agents)
    # -----------------------------------------------------------------
    pain_points: list[PainPoint] = Field(default_factory=list)
    ideas: list[Idea] = Field(default_factory=list)
    scored_ideas: list[ScoredIdea] = Field(default_factory=list)
    pitch_briefs: list[PitchBrief] = Field(default_factory=list)

    # -----------------------------------------------------------------
    # Reflection loop state
    # -----------------------------------------------------------------
    critique: Critique | None = None
    revision_count: int = Field(default=0, ge=0)
    revision_feedback: str | None = None

    # -----------------------------------------------------------------
    # Orchestration control (set by orchestrator node)
    # -----------------------------------------------------------------
    next_node: str = Field(default="orchestrator")
    current_stage: PipelineStage = Field(default=PipelineStage.IDLE)
    previous_stage: PipelineStage | None = None

    # -----------------------------------------------------------------
    # Metadata & diagnostics
    # -----------------------------------------------------------------
    run_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error_log: list[str] = Field(default_factory=list)
    agent_timings: dict[str, float] = Field(
        default_factory=dict,
        description="agent_id -> elapsed seconds",
    )

    # -----------------------------------------------------------------
    # Derived properties (convenience for agent logic)
    # -----------------------------------------------------------------
    @computed_field
    @property
    def filtered_pain_points(self) -> list[PainPoint]:
        """Pain points that passed their own internal rubric."""
        return [pp for pp in self.pain_points if pp.passes_rubric]

    @computed_field
    @property
    def top_scored_ideas(self) -> list[ScoredIdea]:
        """Ideas sorted by yes_count desc, limited to top_n_pitches."""
        ranked = sorted(
            self.scored_ideas,
            key=lambda s: (s.yes_count, s.rank or 0),
            reverse=True,
        )
        return ranked[: self.top_n_pitches]

    @computed_field
    @property
    def can_revise(self) -> bool:
        """True if the reflection loop is still allowed to revise."""
        return self.revision_count < self.max_revisions

    @computed_field
    @property
    def is_complete(self) -> bool:
        """All expected outputs are present."""
        return all(
            [
                self.pain_points,
                self.ideas,
                self.scored_ideas,
                self.pitch_briefs,
                self.critique is not None,
            ]
        )

    # -----------------------------------------------------------------
    # Immutable helpers
    # -----------------------------------------------------------------
    def log_error(self, agent_id: str, message: str) -> dict[str, Any]:
        """Return a state patch that appends an error."""
        entry = f"[{agent_id}] {message}"
        return {"error_log": self.error_log + [entry]}

    def record_timing(self, agent_id: str, elapsed_s: float) -> dict[str, Any]:
        """Return a state patch recording agent timing."""
        timing = {**self.agent_timings, agent_id: elapsed_s}
        return {"agent_timings": timing}

    def bump_revision(self, critique: Critique) -> dict[str, Any]:
        """
        Return a state patch that increments the revision counter,
        stores the critique, and prepares state for the target agent.
        """
        return {
            "revision_count": self.revision_count + 1,
            "critique": critique,
            "revision_feedback": critique.revision_feedback,
            "previous_stage": self.current_stage,
            "current_stage": PipelineStage.REVISING,
            "next_node": critique.target_agent,
        }

    def mark_completed(self) -> dict[str, Any]:
        """Return a state patch marking the pipeline as complete."""
        return {
            "current_stage": PipelineStage.COMPLETED,
            "next_node": "__end__",
            "completed_at": datetime.now(timezone.utc),
        }

    def mark_failed(self, reason: str) -> dict[str, Any]:
        """Return a state patch marking the pipeline as failed."""
        patch = {
            "current_stage": PipelineStage.FAILED,
            "next_node": "__end__",
            "completed_at": datetime.now(timezone.utc),
        }
        patch.update(self.log_error("orchestrator", reason))
        return patch

    def reset_for_revision(self, target_agent: TargetAgent | str) -> dict[str, Any]:
        """
        Clear downstream data so the target worker can re-run cleanly.
        E.g. if idea_generator is revised, we clear ideas, scored_ideas,
        pitch_briefs, and critique.
        """
        target = target_agent if isinstance(target_agent, str) else target_agent.value
        updates: dict[str, Any] = {}
        if target == "pain_point_miner":
            updates = {
                "pain_points": [],
                "ideas": [],
                "scored_ideas": [],
                "pitch_briefs": [],
                "critique": None,
            }
        elif target == "idea_generator":
            updates = {
                "ideas": [],
                "scored_ideas": [],
                "pitch_briefs": [],
                "critique": None,
            }
        elif target == "pitch_writer":
            updates = {
                "pitch_briefs": [],
                "critique": None,
            }
        return updates


# =============================================================================
# QUICK VALIDATION
# =============================================================================

if __name__ == "__main__":
    # Smoke test: construct a minimal state end-to-end
    state = VentureForgeState(domain="developer tools")

    pp = PainPoint(
        title="No good local LLM tooling",
        description="Developers frustrated by lack of simple local LLM runners.",
        rubric=PainPointRubric(
            is_genuine_current_frustration=True,
            has_verbatim_quote=True,
            user_segment_specific=True,
        ),
        passes_rubric=True,
        source_url="https://reddit.com/r/programming/comments/abc123",
        raw_quote="I wish there was a simple local LLM runner that just works",
        source=DataSource.REDDIT,
    )

    pp2 = PainPoint(
        title="Managing multiple local models is painful",
        description="Developers struggle to keep track of which local models are installed and configured.",
        rubric=PainPointRubric(
            is_genuine_current_frustration=True,
            has_verbatim_quote=True,
            user_segment_specific=True,
        ),
        passes_rubric=True,
        source_url="https://news.ycombinator.com/item?id=123456",
        raw_quote="I have 5 different local LLM tools and I can never remember which one I configured for what.",
        source=DataSource.HACKERNEWS,
    )

    idea = Idea(
        title="DevFlow LLM",
        one_liner="One-click local LLM workspace for developers.",
        problem="Setting up local LLM environments is complicated.",
        solution="A CLI tool that downloads, configures, and runs any open-source LLM with sensible defaults.",
        target_user="solo developers wanting local AI without DevOps",
        key_features=["one-command setup", "auto GPU detection", "model registry"],
        addresses_pain_point_ids=[pp.id, pp2.id],
    )

    scored = ScoredIdea(
        idea_id=idea.id,
        feasibility_rubric=FeasibilityRubric(
            can_2_3_person_team_build_mvp_in_6_months=True,
            uses_only_existing_proven_tech=True,
            no_special_regulatory_requirements=True,
        ),
        demand_rubric=DemandRubric(
            addresses_at_least_2_pain_points=True,
            pain_points_show_high_severity=True,
            target_user_clearly_defined=True,
        ),
        novelty_rubric=NoveltyRubric(
            differentiated_from_obvious_solutions=True,
            leverages_unique_insight=True,
        ),
        yes_count=8,
        verdict=Verdict.PURSUE,
        one_risk="Established competitors could copy the UX quickly.",
        rank=1,
    )

    patch = {
        "pain_points": [pp, pp2],
        "ideas": [idea],
        "scored_ideas": [scored],
        "current_stage": PipelineStage.COMPLETED,
    }
    state = state.model_copy(update=patch)

    print("[OK] Schema validation passed")
    print(f"   Run ID      : {state.run_id}")
    print(f"   Domain      : {state.domain}")
    print(f"   Pain points : {len(state.filtered_pain_points)} (passed rubric)")
    print(f"   Ideas       : {len(state.ideas)}")
    print(f"   Pursue      : {sum(1 for s in state.scored_ideas if s.verdict == Verdict.PURSUE)}")
    print(f"   Is complete : {state.is_complete}")
    print(f"   Can revise  : {state.can_revise}")
