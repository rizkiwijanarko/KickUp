# src/state/schema.py
"""
VentureForge Shared State Schema
=================================
Single source of truth for all data passed between agents.
All agents read from and write to an instance of VentureForgeState.

Usage:
    from src.state.schema import VentureForgeState, PainPoint, CandidateIdea

Pydantic v2 is required:
    pip install pydantic>=2.0
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ==============================================================================
# ENUMS
# ==============================================================================

class DataSource(str, Enum):
    GOOGLE_TRENDS   = "google_trends"
    HACKERNEWS      = "hackernews"
    REDDIT          = "reddit"
    WEB_SEARCH      = "web_search"
    GITHUB          = "github"
    PRODUCTHUNT     = "producthunt"
    YC              = "yc"
    SEC_EDGAR       = "sec_edgar"
    WIKIPEDIA       = "wikipedia"
    CENSUS          = "census"
    CRAWL4AI        = "crawl4ai"
    OPENVC          = "openvc"


class CompanyStage(str, Enum):
    OPEN_SOURCE     = "open_source"
    INDIE           = "indie"
    SEED            = "seed"
    SERIES_A_PLUS   = "series_a_plus"
    PUBLIC          = "public"


class MarketSizeFit(str, Enum):
    TOO_SMALL       = "too_small"
    NICHE           = "niche"
    SWEET_SPOT      = "sweet_spot"
    TOO_BROAD       = "too_broad"


class RevenueModel(str, Enum):
    SUBSCRIPTION    = "subscription"
    USAGE_BASED     = "usage_based"
    MARKETPLACE     = "marketplace"
    FREEMIUM        = "freemium"
    ONE_TIME        = "one_time"
    ADS             = "ads"


class RiskType(str, Enum):
    LEGAL           = "legal"
    FINANCIAL       = "financial"
    OPERATIONAL     = "operational"
    COMPETITIVE     = "competitive"
    TECHNICAL       = "technical"


class Severity(str, Enum):
    LOW             = "low"
    MEDIUM          = "medium"
    HIGH            = "high"


class IdeaVerdict(str, Enum):
    PURSUE          = "pursue"
    EXPLORE         = "explore"
    PARK            = "park"
    KILL            = "kill"


class CriticStatus(str, Enum):
    APPROVED        = "approved"
    REVISE          = "revise"
    REJECT          = "reject"


class IssueType(str, Enum):
    HALLUCINATION   = "hallucination"
    WEAK_EVIDENCE   = "weak_evidence"
    LOGICAL_GAP     = "logical_gap"
    MARKET_MISMATCH = "market_mismatch"
    TONE            = "tone"


class MarketConfidence(str, Enum):
    LOW             = "low"
    MEDIUM          = "medium"
    HIGH            = "high"


class PipelineStatus(str, Enum):
    SUCCESS         = "success"
    PARTIAL         = "partial"
    FAILED          = "failed"


# ==============================================================================
# USER INPUT
# ==============================================================================

class UserConstraints(BaseModel):
    """Optional constraints provided by the user at runtime."""
    budget_usd:         Optional[int]   = Field(None, description="Max startup budget in USD")
    team_size:          Optional[int]   = Field(None, description="Available team size")
    geography:          Optional[str]   = Field(None, description="Target geography e.g. 'US', 'global'")
    exclude_domains:    list[str]       = Field(default_factory=list, description="Domains to exclude")
    preferred_revenue:  Optional[RevenueModel] = Field(None, description="Preferred revenue model")


class UserInput(BaseModel):
    """Entry point — provided by the user via CLI or Streamlit UI."""
    domain:             str             = Field(..., description="Target domain e.g. 'developer tools'")
    constraints:        UserConstraints = Field(default_factory=UserConstraints)
    ideas_per_cluster:  int             = Field(default=3, ge=1, le=10)
    top_n_ideas:        int             = Field(default=5, ge=1, le=20)
    lookback_days:      int             = Field(default=90, ge=7, le=365)
    max_posts_per_source: int           = Field(default=100, ge=10, le=500)


# ==============================================================================
# DISCOVERY TEAM OUTPUTS
# ==============================================================================

# --- Trend Scanner ---

class Trend(BaseModel):
    """A single emerging trend identified by the Trend Scanner."""
    id:             UUID            = Field(default_factory=uuid4)
    title:          str
    description:    str
    momentum_score: float           = Field(ge=0.0, le=1.0)
    evidence_urls:  list[str]       = Field(default_factory=list)
    source:         DataSource
    first_seen:     datetime        = Field(default_factory=datetime.utcnow)


# --- Pain Point Miner ---

class PainPoint(BaseModel):
    """A single structured user pain point extracted from public discussions."""
    id:             UUID            = Field(default_factory=uuid4)
    title:          str
    description:    str
    severity:       float           = Field(ge=0.0, le=1.0, description="Emotional intensity 0-1")
    frequency:      int             = Field(ge=1, description="Number of times mentioned")
    user_segment:   str             = Field(description="e.g. 'solo developers', 'small business owners'")
    source_url:     str
    raw_quote:      str             = Field(description="Exact phrase proving the pain point")
    source:         DataSource
    extracted_at:   datetime        = Field(default_factory=datetime.utcnow)


# --- Market Research Agent ---

class MarketData(BaseModel):
    """Market sizing and growth data for the target domain."""
    tam_usd:            Optional[float] = Field(None, description="Total Addressable Market in USD")
    sam_usd:            Optional[float] = Field(None, description="Serviceable Addressable Market in USD")
    som_usd:            Optional[float] = Field(None, description="Serviceable Obtainable Market in USD")
    growth_rate_pct:    Optional[float] = Field(None, description="Annual growth rate %")
    key_segments:       list[str]       = Field(default_factory=list)
    key_drivers:        list[str]       = Field(default_factory=list)
    sources:            list[str]       = Field(default_factory=list)
    confidence:         MarketConfidence = Field(default=MarketConfidence.LOW)
    summary:            Optional[str]   = None


# --- Competitor Intel Agent ---

class Competitor(BaseModel):
    """A single competitor or existing solution in the target space."""
    id:                 UUID            = Field(default_factory=uuid4)
    name:               str
    url:                str
    description:        str
    category:           str
    stage:              CompanyStage
    founded_year:       Optional[int]   = None
    pricing_model:      Optional[str]   = None
    differentiators:    list[str]       = Field(default_factory=list)
    weaknesses:         list[str]       = Field(default_factory=list)
    source:             DataSource


# --- Combined Discovery Results ---

class DiscoveryResults(BaseModel):
    """Aggregated output of all 4 Discovery Team agents."""
    trends:         list[Trend]         = Field(default_factory=list)
    pain_points:    list[PainPoint]     = Field(default_factory=list)
    market_data:    Optional[MarketData] = None
    competitors:    list[Competitor]    = Field(default_factory=list)
    completed_at:   Optional[datetime]  = None


# ==============================================================================
# SYNTHESIS TEAM OUTPUTS
# ==============================================================================

# --- Clustering Agent ---

class Cluster(BaseModel):
    """A thematic cluster of related pain points."""
    cluster_id:     str
    theme:          str             = Field(description="LLM-generated human-readable label")
    summary:        str
    pain_point_ids: list[UUID]      = Field(default_factory=list)
    size:           int             = Field(ge=1)
    avg_severity:   float           = Field(ge=0.0, le=1.0)
    embedding_ids:  list[str]       = Field(default_factory=list, description="ChromaDB doc IDs")


# --- Idea Generator ---

class CandidateIdea(BaseModel):
    """A single startup idea generated from a cluster of pain points."""
    id:                     UUID        = Field(default_factory=uuid4)
    title:                  str
    one_liner:              str
    problem:                str
    solution:               str
    target_user:            str
    key_features:           list[str]   = Field(default_factory=list)
    underlying_cluster_id:  str
    inspiration_trend_ids:  list[UUID]  = Field(default_factory=list)
    novelty_hypothesis:     str         = Field(description="Why this is different from competitors")
    generated_at:           datetime    = Field(default_factory=datetime.utcnow)
    revision_count:         int         = Field(default=0)


# --- Combined Synthesis Results ---

class SynthesisResults(BaseModel):
    """Aggregated output of Clustering Agent + Idea Generator."""
    clusters:           list[Cluster]       = Field(default_factory=list)
    candidate_ideas:    list[CandidateIdea] = Field(default_factory=list)
    completed_at:       Optional[datetime]  = None


# ==============================================================================
# VALIDATION TEAM OUTPUTS
# ==============================================================================

# --- Feasibility Agent ---

class FeasibilityScore(BaseModel):
    """Technical, regulatory, and team feasibility assessment for one idea."""
    idea_id:                    UUID
    technical_score:            float       = Field(ge=0.0, le=1.0)
    regulatory_score:           float       = Field(ge=0.0, le=1.0)
    team_score:                 float       = Field(ge=0.0, le=1.0)
    overall_score:              float       = Field(ge=0.0, le=1.0)
    blockers:                   list[str]   = Field(default_factory=list)
    open_source_components:     list[str]   = Field(default_factory=list)
    estimated_mvp_months:       int         = Field(ge=1)
    rationale:                  str


# --- Market Validator ---

class ValidationEvidence(BaseModel):
    """A single piece of evidence supporting market validation."""
    type:       str     = Field(description="reddit_quote, ph_traction, trend_data, competitor_revenue")
    snippet:    str
    source_url: str


class MarketValidation(BaseModel):
    """Demand signals and willingness-to-pay assessment for one idea."""
    idea_id:                UUID
    demand_score:           float               = Field(ge=0.0, le=1.0)
    willingness_to_pay:     float               = Field(ge=0.0, le=1.0)
    market_size_fit:        MarketSizeFit
    supporting_evidence:    list[ValidationEvidence] = Field(default_factory=list)
    rationale:              str


# --- Business Model Agent ---

class PricingTier(BaseModel):
    """A single pricing tier for the product."""
    name:           str
    price_usd:      float           = Field(ge=0.0)
    billing_period: str             = Field(description="monthly, yearly, one-time, per-seat")
    features:       list[str]       = Field(default_factory=list)


class LeanCanvas(BaseModel):
    """Ash Maurya Lean Canvas for one startup idea."""
    problem:            str
    solution:           str
    key_metrics:        list[str]   = Field(default_factory=list)
    value_proposition:  str
    unfair_advantage:   str
    channels:           list[str]   = Field(default_factory=list)
    customer_segments:  list[str]   = Field(default_factory=list)
    cost_structure:     list[str]   = Field(default_factory=list)
    revenue_streams:    list[str]   = Field(default_factory=list)


class BusinessModel(BaseModel):
    """Revenue model and go-to-market design for one idea."""
    idea_id:            UUID
    revenue_model:      RevenueModel
    pricing_tiers:      list[PricingTier]   = Field(default_factory=list)
    primary_channels:   list[str]           = Field(default_factory=list)
    estimated_cac_usd:  Optional[float]     = None
    estimated_ltv_usd:  Optional[float]     = None
    lean_canvas:        Optional[LeanCanvas] = None


# --- Risk & Scoring Agent ---

class Risk(BaseModel):
    """A single identified risk for a startup idea."""
    type:           RiskType
    description:    str
    severity:       Severity
    mitigation:     str


class ScoreBreakdown(BaseModel):
    """Dimension-level scores feeding into the final score."""
    feasibility:        float   = Field(ge=0.0, le=1.0)
    market_demand:      float   = Field(ge=0.0, le=1.0)
    business_viability: float   = Field(ge=0.0, le=1.0)
    differentiation:    float   = Field(ge=0.0, le=1.0)
    timing:             float   = Field(ge=0.0, le=1.0)

    @property
    def weighted_score(self) -> float:
        """
        Default weighting:
            feasibility         20%
            market_demand       30%
            business_viability  20%
            differentiation     20%
            timing              10%
        """
        return round(
            self.feasibility        * 0.20 +
            self.market_demand      * 0.30 +
            self.business_viability * 0.20 +
            self.differentiation    * 0.20 +
            self.timing             * 0.10,
            4
        )


class ScoredIdea(BaseModel):
    """Fully validated and ranked startup idea — output of Risk & Scoring Agent."""
    idea_id:            UUID
    idea:               CandidateIdea
    feasibility:        Optional[FeasibilityScore]  = None
    validation:         Optional[MarketValidation]  = None
    business_model:     Optional[BusinessModel]     = None
    score_breakdown:    Optional[ScoreBreakdown]    = None
    final_score:        float                       = Field(ge=0.0, le=1.0)
    risks:              list[Risk]                  = Field(default_factory=list)
    rank:               Optional[int]               = None
    verdict:            IdeaVerdict
    rationale:          str


# --- Combined Validation Results ---

class ValidationResults(BaseModel):
    """Aggregated output of all 4 Validation Team agents."""
    scored_ideas:   list[ScoredIdea]    = Field(default_factory=list)
    completed_at:   Optional[datetime]  = None

    @property
    def ranked(self) -> list[ScoredIdea]:
        """Return ideas sorted by final_score descending."""
        return sorted(self.scored_ideas, key=lambda x: x.final_score, reverse=True)

    @property
    def top_n(self, n: int = 5) -> list[ScoredIdea]:
        """Return the top N ideas by score."""
        return self.ranked[:n]


# ==============================================================================
# OUTPUT LAYER
# ==============================================================================

# --- Pitch Writer ---

class PitchSections(BaseModel):
    """Individual sections of a pitch brief."""
    executive_summary:      str
    problem:                str
    solution:               str
    market_opportunity:     str
    business_model:         str
    competitive_landscape:  str
    go_to_market:           str
    risks_and_mitigations:  str
    next_steps:             str


class PitchBrief(BaseModel):
    """Investor-ready pitch brief for one startup idea."""
    idea_id:            UUID
    title:              str
    tagline:            str
    sections:           PitchSections
    markdown_content:   str             = Field(description="Full brief as rendered markdown")
    evidence_links:     list[str]       = Field(default_factory=list)
    generated_at:       datetime        = Field(default_factory=datetime.utcnow)
    revision_count:     int             = Field(default=0)


# --- Critic Agent ---

class CriticIssue(BaseModel):
    """A single issue identified by the Critic Agent."""
    type:           IssueType
    description:    str
    severity:       Severity
    suggested_fix:  str
    target_agent:   str     = Field(description="Which agent should be re-invoked to fix this")


class CriticFeedback(BaseModel):
    """Full critique of one pitch brief."""
    idea_id:            UUID
    approval_status:    CriticStatus
    quality_score:      float           = Field(ge=0.0, le=1.0)
    issues:             list[CriticIssue] = Field(default_factory=list)
    revision_count:     int             = Field(default=0)
    reviewed_at:        datetime        = Field(default_factory=datetime.utcnow)


# --- Combined Output ---

class OutputResults(BaseModel):
    """Final deliverables produced by the Output Layer."""
    pitch_briefs:   list[PitchBrief]    = Field(default_factory=list)
    ranked_ideas:   list[ScoredIdea]    = Field(default_factory=list)
    completed_at:   Optional[datetime]  = None


# ==============================================================================
# PIPELINE METADATA
# ==============================================================================

class TokenUsage(BaseModel):
    """LLM token tracking per agent run."""
    agent_id:           str
    model:              str
    prompt_tokens:      int = 0
    completion_tokens:  int = 0
    total_tokens:       int = 0


class AgentTimestamp(BaseModel):
    """Execution timing for one agent."""
    agent_id:       str
    started_at:     datetime
    completed_at:   Optional[datetime]  = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class PipelineMetadata(BaseModel):
    """Run-level metadata: timing, token usage, status."""
    run_id:             str             = Field(default_factory=lambda: str(uuid4()))
    started_at:         datetime        = Field(default_factory=datetime.utcnow)
    completed_at:       Optional[datetime] = None
    status:             PipelineStatus  = PipelineStatus.SUCCESS
    token_usage:        list[TokenUsage]    = Field(default_factory=list)
    agent_timestamps:   list[AgentTimestamp] = Field(default_factory=list)
    error_log:          list[str]           = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(t.total_tokens for t in self.token_usage)

    @property
    def total_duration_seconds(self) -> Optional[float]:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# ==============================================================================
# MASTER SHARED STATE
# ==============================================================================

class VentureForgeState(BaseModel):
    """
    The single shared state object passed between all agents
    via LangGraph StateGraph.

    Every agent receives this as input and returns a modified
    copy as output. Never mutate in place — always return
    a new instance with updated fields.

    Example:
        state = VentureForgeState(user_input=UserInput(domain="developer tools"))
        updated = state.model_copy(update={"discovery_results": results})
    """

    # Input
    user_input:             UserInput

    # Discovery Team outputs (populated after Phase 1)
    discovery_results:      Optional[DiscoveryResults]  = None

    # Synthesis Team outputs (populated after Phase 2)
    synthesis_results:      Optional[SynthesisResults]  = None

    # Validation Team outputs (populated after Phase 3)
    validation_results:     Optional[ValidationResults] = None

    # Output Layer outputs (populated after Phase 4)
    output:                 Optional[OutputResults]     = None

    # Critic Agent feedback (grows with each reflection loop)
    critic_feedback:        list[CriticFeedback]        = Field(default_factory=list)

    # Run metadata
    metadata:               PipelineMetadata            = Field(default_factory=PipelineMetadata)

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------

    @property
    def domain(self) -> str:
        return self.user_input.domain

    @property
    def pain_points(self) -> list[PainPoint]:
        if self.discovery_results:
            return self.discovery_results.pain_points
        return []

    @property
    def candidate_ideas(self) -> list[CandidateIdea]:
        if self.synthesis_results:
            return self.synthesis_results.candidate_ideas
        return []

    @property
    def top_ideas(self) -> list[ScoredIdea]:
        if self.validation_results:
            return self.validation_results.ranked[:self.user_input.top_n_ideas]
        return []

    @property
    def is_complete(self) -> bool:
        return all([
            self.discovery_results is not None,
            self.synthesis_results is not None,
            self.validation_results is not None,
            self.output is not None,
        ])

    @property
    def needs_revision(self) -> bool:
        """True if Critic Agent has flagged any ideas for revision."""
        return any(
            fb.approval_status == CriticStatus.REVISE
            for fb in self.critic_feedback
        )

    def log_agent_start(self, agent_id: str) -> "VentureForgeState":
        """Record agent start time in metadata."""
        timestamps = self.metadata.agent_timestamps + [
            AgentTimestamp(agent_id=agent_id, started_at=datetime.utcnow())
        ]
        updated_meta = self.metadata.model_copy(
            update={"agent_timestamps": timestamps}
        )
        return self.model_copy(update={"metadata": updated_meta})

    def log_agent_end(self, agent_id: str) -> "VentureForgeState":
        """Record agent completion time in metadata."""
        timestamps = [
            t.model_copy(update={"completed_at": datetime.utcnow()})
            if t.agent_id == agent_id and t.completed_at is None
            else t
            for t in self.metadata.agent_timestamps
        ]
        updated_meta = self.metadata.model_copy(
            update={"agent_timestamps": timestamps}
        )
        return self.model_copy(update={"metadata": updated_meta})

    def log_error(self, agent_id: str, error: str) -> "VentureForgeState":
        """Append an error without crashing the pipeline."""
        errors = self.metadata.error_log + [f"[{agent_id}] {error}"]
        updated_meta = self.metadata.model_copy(
            update={
                "error_log": errors,
                "status": PipelineStatus.PARTIAL
            }
        )
        return self.model_copy(update={"metadata": updated_meta})


# ==============================================================================
# QUICK VALIDATION (run this file directly to sanity check)
# ==============================================================================

if __name__ == "__main__":
    # Smoke test: build a minimal state and verify it serializes correctly
    state = VentureForgeState(
        user_input=UserInput(domain="developer tools")
    )

    # Simulate logging an agent
    state = state.log_agent_start("pain_point_miner")

    # Simulate a pain point
    pp = PainPoint(
        title="No good local LLM tooling",
        description="Developers frustrated by lack of lightweight local LLM tools",
        severity=0.8,
        frequency=42,
        user_segment="solo developers",
        source_url="https://reddit.com/r/programming/example",
        raw_quote="I wish there was a simple local LLM runner that just works",
        source=DataSource.REDDIT,
    )

    discovery = DiscoveryResults(pain_points=[pp])
    state = state.model_copy(update={"discovery_results": discovery})
    state = state.log_agent_end("pain_point_miner")

    print("✅ Schema validation passed")
    print(f"   Domain      : {state.domain}")
    print(f"   Pain points : {len(state.pain_points)}")
    print(f"   Is complete : {state.is_complete}")
    print(f"   Run ID      : {state.metadata.run_id}")
    print(f"   Duration    : {state.metadata.agent_timestamps[0].duration_seconds:.4f}s")