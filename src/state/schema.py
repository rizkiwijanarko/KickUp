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
from typing import Any, Literal
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
    CANCELLED = "cancelled"


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
    """Binary checks for feasibility (Scorer).

    PG-style: manual-first and schlep are POSITIVE signals, not penalties.
    """

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


class CritiqueRubric(BaseModel):
    """Binary checks applied by the Critic to pitch briefs.

    PG pressure-test criteria with concrete, non-vague definitions.
    """

    all_claims_evidence_backed: bool
    no_hallucinated_source_urls: bool
    tagline_under_12_words: bool
    target_is_contained_fire: bool
    competition_embraced_with_thesis: bool
    unscalable_acquisition_concrete: bool
    gtm_leads_with_manual_recruitment: bool


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
        """Derive verdict from yes_count AND fatal flaw severity.

        Per the Scorer prompt: a fatal severity flaw ALWAYS parks,
        regardless of yes_count.
        """
        has_fatal = any(f.severity == "fatal" for f in self.fatal_flaws)
        if self.yes_count <= 2 or has_fatal:
            self.verdict = "park"
        elif 3 <= self.yes_count <= 5 and not has_fatal:
            self.verdict = "explore"
        elif self.yes_count >= 6 and not has_fatal:
            self.verdict = "pursue"
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
    """Output of the Critic agent after reviewing a pitch brief.

    The model validator enforces that ``all_pass``, ``failing_checks``,
    ``approval_status`` and ``target_agent`` are consistent with the
    rubric and the documented target-agent priority rules.
    """

    idea_id: UUID
    reasoning_trace: str
    rubric: CritiqueRubric
    all_pass: bool
    approval_status: Literal["approved", "revise"]
    failing_checks: list[str] = Field(default_factory=list)
    target_agent: Literal["pain_point_miner", "idea_generator", "pitch_writer"]
    revision_feedback: str = Field(..., min_length=10)

    @model_validator(mode="after")
    def _sync_from_rubric(self) -> "Critique":
        """Ensure all_pass/failing_checks/approval_status/target_agent match rubric.

        Target-agent priority:
        1. pain_point_miner — if any evidence check fails
           (all_claims_evidence_backed or no_hallucinated_source_urls).
        2. idea_generator — if positioning checks fail
           (target_is_contained_fire or competition_embraced_with_thesis)
           and evidence checks pass.
        3. pitch_writer — otherwise (writing/tone only).
        """
        rubric_dict = self.rubric.model_dump()
        self.failing_checks = [k for k, v in rubric_dict.items() if not v]
        self.all_pass = len(self.failing_checks) == 0

        # Approval status derived solely from rubric
        self.approval_status = "approved" if self.all_pass else "revise"

        # Enforce target_agent priority only when revision is required
        if not self.all_pass:
            r = self.rubric
            evidence_failed = (not r.all_claims_evidence_backed) or (not r.no_hallucinated_source_urls)
            positioning_failed = (not r.target_is_contained_fire) or (not r.competition_embraced_with_thesis)

            if evidence_failed:
                self.target_agent = "pain_point_miner"
            elif positioning_failed:
                self.target_agent = "idea_generator"
            else:
                self.target_agent = "pitch_writer"

        return self


class RunEvent(BaseModel):
    """A single high-level event emitted during a pipeline run.

    Used by the UI to render an agent execution log.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str
    stage: PipelineStage
    kind: Literal["info", "warning", "error"] = "info"
    message: str
    idea_id: UUID | None = None


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
    # Reflection loop state (per-pitch revision tracking)
    # -----------------------------------------------------------------
    critique: Critique | None = None
    critiques: list[Critique] = Field(default_factory=list)
    revision_counts: dict[str, int] = Field(default_factory=dict)
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
    events: list[RunEvent] = Field(
        default_factory=list,
        description="High-level run events emitted by agents/orchestrator for UI logs.",
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
        """True if the most recently critiqued pitch can still be revised."""
        if self.critique is None:
            return True
        return self.get_revision_count(self.critique.idea_id) < self.max_revisions

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
    # Per-pitch revision helpers
    # -----------------------------------------------------------------
    def get_revision_count(self, idea_id: UUID) -> int:
        """Return the number of revisions already done for a specific idea."""
        return self.revision_counts.get(str(idea_id), 0)

    def increment_revision_count(self, idea_id: UUID) -> "VentureForgeState":
        """Return a new state with the revision count bumped for this idea."""
        updated = dict(self.revision_counts)
        updated[str(idea_id)] = updated.get(str(idea_id), 0) + 1
        return self.model_copy(update={"revision_counts": updated})

    # -----------------------------------------------------------------
    # Immutable helpers
    # -----------------------------------------------------------------
    def log_error(self, agent_id: str, message: str) -> dict[str, Any]:
        """Return a state patch that appends an error and emits an error event."""
        entry = f"[{agent_id}] {message}"
        events = self.events + [
            RunEvent(
                agent=agent_id,
                stage=self.current_stage,
                kind="error",
                message=message,
            )
        ]
        return {"error_log": self.error_log + [entry], "events": events}

    def add_event(
        self,
        *,
        agent: str,
        stage: PipelineStage,
        kind: Literal["info", "warning", "error"] = "info",
        message: str,
        idea_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Return a state patch that appends a RunEvent to ``events``.

        Agents and the orchestrator should call this to record high-level
        progress suitable for a UI log.
        """
        ev = RunEvent(agent=agent, stage=stage, kind=kind, message=message, idea_id=idea_id)
        return {"events": self.events + [ev]}

    def record_timing(self, agent_id: str, elapsed_s: float) -> dict[str, Any]:
        """Return a state patch recording agent timing."""
        timing = {**self.agent_timings, agent_id: elapsed_s}
        return {"agent_timings": timing}

    def bump_revision(self, critique: Critique) -> dict[str, Any]:
        """
        Return a state patch that increments the per-pitch revision counter,
        archives the critique, stores the critique, and prepares state for
        the target agent.
        """
        idea_id = str(critique.idea_id)
        updated_counts = {**self.revision_counts, idea_id: self.revision_counts.get(idea_id, 0) + 1}
        return {
            "revision_counts": updated_counts,
            "critiques": self.critiques + [critique],
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

    def mark_cancelled(self, reason: str = "Cancelled by user") -> dict[str, Any]:
        """Return a state patch marking the pipeline as cancelled."""
        patch = {
            "current_stage": PipelineStage.CANCELLED,
            "next_node": "__end__",
            "completed_at": datetime.now(timezone.utc),
        }
        patch.update(
            self.add_event(
                agent="orchestrator",
                stage=self.current_stage,
                kind="warning",
                message=reason,
            )
        )
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
        reasoning_trace="The manual version is a bash script that downloads models. Demand is a well — deep but narrow. The schlep is ROCm compatibility. First 10 users are in r/LocalLLaMA.",
        feasibility_rubric=FeasibilityRubric(
            can_be_solved_manually_first=True,
            has_schlep_or_unsexy_advantage=True,
            can_2_3_person_team_build_mvp_in_6_months=True,
        ),
        demand_rubric=DemandRubric(
            addresses_at_least_2_pain_points=True,
            is_painkiller_not_vitamin=True,
            has_clear_vein_of_early_adopters=True,
        ),
        novelty_rubric=NoveltyRubric(
            differentiated_from_current_behavior=True,
            has_path_out_of_niche=True,
        ),
        core_assumption="Developers will switch if setup is < 1 minute.",
        fatal_flaws=[
            FatalFlaw(flaw="Ollama already solves this for most users", severity="major"),
            FatalFlaw(flaw="Cloud providers could release a simpler free tier", severity="minor"),
        ],
        yes_count=8,
        verdict="pursue",
        one_risk="Established competitors could copy the UX quickly.",
        rank=1,
    )

    brief = PitchBrief(
        idea_id=idea.id,
        title=idea.title,
        tagline="One-click local LLM workspace.",
        problem="Setting up local LLM environments is complicated.",
        solution="A CLI tool with sensible defaults.",
        target_user="Solo developers",
        market_opportunity="Growing demand for local AI among privacy-conscious solo developers who want LLM access without cloud dependencies.",
        business_model="Freemium CLI with paid enterprise features for teams.",
        go_to_market="ProductHunt launch followed by targeted outreach to solo devs.",
        key_risk="Low barrier to entry in the local tooling space.",
        next_steps="Interview 20 r/LocalLLaMA users about their setup pain, then run a concierge pilot with 5 early adopters.",
        evidence_links=[pp.source_url],
        markdown_content="# DevFlow LLM\n\nFull pitch brief here that is definitely more than one hundred characters long to satisfy the Pydantic validation rules. We need enough text to describe the product, the problem it solves, and why it is better than the competition.",
    )

    critique = Critique(
        idea_id=idea.id,
        reasoning_trace="Tagline is 5 words. URLs check out. Target user is a demographic, not a contained fire.",
        rubric=CritiqueRubric(
            all_claims_evidence_backed=True,
            no_hallucinated_source_urls=True,
            tagline_under_12_words=True,
            target_is_contained_fire=False,
            competition_embraced_with_thesis=True,
            unscalable_acquisition_concrete=True,
            gtm_leads_with_manual_recruitment=True,
        ),
        all_pass=False,
        approval_status="revise",
        failing_checks=["target_is_contained_fire"],
        target_agent="idea_generator",
        revision_feedback="Target user is a demographic, not a contained community. Redefine as a specific reachable group.",
    )

    patch = {
        "pain_points": [pp, pp2],
        "ideas": [idea],
        "scored_ideas": [scored],
        "pitch_briefs": [brief],
        "critique": critique,
        "critiques": [critique],
        "current_stage": PipelineStage.COMPLETED,
    }
    state = state.model_copy(update=patch)

    print("[OK] Schema validation passed")
    print(f"   Run ID      : {state.run_id}")
    print(f"   Domain      : {state.domain}")
    print(f"   Pain points : {len(state.filtered_pain_points)} (passed rubric)")
    print(f"   Ideas       : {len(state.ideas)}")
    print(f"   Pursue      : {sum(1 for s in state.scored_ideas if s.verdict == 'pursue')}")
    print(f"   Is complete : {state.is_complete}")
    print(f"   Can revise  : {state.can_revise}")
    print(f"   Revision cnt: {state.get_revision_count(idea.id)}")
