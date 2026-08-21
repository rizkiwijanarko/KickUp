"""
VentureForge Constants
======================
Centralized configuration constants for the entire application.
All magic numbers, thresholds, and limits are defined here.
"""

# =============================================================================
# PIPELINE LIMITS
# =============================================================================

# Maximum number of pain points to extract per domain
MAX_PAIN_POINTS_DEFAULT = 5

# Maximum number of ideas to generate per run
MAX_IDEAS_PER_RUN_DEFAULT = 5

# Maximum number of revision attempts per idea
MAX_REVISIONS_PER_IDEA = 2

# Maximum number of idea generation attempts before giving up
MAX_IDEA_GENERATION_ATTEMPTS = 3

# =============================================================================
# PAIN POINT MINING
# =============================================================================

# Circuit breaker: max attempts to mine pain points before failing
MAX_INITIAL_MINING_ATTEMPTS = 5

# Quality gate: minimum pain points required before generating ideas
MIN_PAIN_POINTS_FOR_IDEAS = 2

# Maximum retries for mining when below minimum threshold
MAX_MINING_RETRIES = 2

# Scraping limits
MAX_COMMENTS_PER_SUBREDDIT = 50
MAX_TOTAL_COMMENTS = 200

# Tavily fallback ratio (fraction of max_pain_points)
TAVILY_FALLBACK_RATIO = 0.5

# Text truncation limits (to manage token usage)
COMMENT_TEXT_MAX_LENGTH = 800
POST_TITLE_MAX_LENGTH = 120

# =============================================================================
# IDEA GENERATION
# =============================================================================

# Minimum pain points an idea must address
MIN_PAIN_POINTS_PER_IDEA = 2

# Maximum pain points to include in idea generation prompt
MAX_PAIN_POINTS_FOR_PROMPT = 15

# =============================================================================
# SCORING
# =============================================================================

# Minimum score threshold for "pursue" verdict
PURSUE_SCORE_THRESHOLD = 6  # out of 8 binary checks

# Minimum score threshold for "explore" verdict
EXPLORE_SCORE_THRESHOLD = 4  # out of 8 binary checks

# Below EXPLORE_SCORE_THRESHOLD = "park"

# LLM configuration for Scorer
SCORER_LLM_TEMPERATURE = 0.1
SCORER_LLM_MAX_TOKENS = 16384

# =============================================================================
# PITCH WRITING
# =============================================================================

# Maximum retry attempts for pitch brief generation
MAX_PITCH_GENERATION_ATTEMPTS = 3

# Minimum required sections in a pitch brief
REQUIRED_PITCH_SECTIONS = [
    "problem",
    "solution",
    "target_users",
    "value_proposition",
    "competitive_advantage",
]

# =============================================================================
# CRITIC & REVISION
# =============================================================================

# Number of binary checks in critic rubric
CRITIC_RUBRIC_CHECKS = 7

# Minimum passing checks for auto-approval
CRITIC_AUTO_APPROVE_THRESHOLD = 6  # out of 7

# LLM configuration for Critic
CRITIC_LLM_TEMPERATURE = 0.2
CRITIC_LLM_MAX_TOKENS = 2048

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# Default temperature for reasoning tasks (scorer, critic)
DEFAULT_REASONING_TEMPERATURE = 0.2

# Default temperature for creative tasks (idea generation, pitch writing)
DEFAULT_CREATIVE_TEMPERATURE = 0.7

# Default max tokens for LLM responses
DEFAULT_MAX_TOKENS = 4096

# Request timeout in seconds
DEFAULT_REQUEST_TIMEOUT = 120

# Default maximum concurrency for parallel generation tasks
DEFAULT_LLM_MAX_CONCURRENCY = 5

# Diverse creative theme angles for parallel idea generation
IDEA_THEME_ANGLES: list[str] = [
    "Direct Workflow Automation & B2B SaaS (eliminating repetitive manual toil, streamlining pipelines)",
    "Vertical Niche Platform & Specialized Tooling (tailored deeply to a single industry/persona's quirks)",
    "Developer Infrastructure & API-First Services (headless primitives, observability, composable architecture)",
    "Consumer / Community-Driven Experience (habit-forming UX, social proof, decentralized collaboration)",
    "Unsexy Schlep & Operational Technology (heavy lifting, messy real-world integration, compliance/governance)",
]

# =============================================================================
# RETRY & BACKOFF
# =============================================================================

# Maximum retries for LLM API calls
MAX_LLM_RETRIES = 3

# Exponential backoff base (seconds)
BACKOFF_BASE_SECONDS = 2

# Maximum backoff time (seconds)
MAX_BACKOFF_SECONDS = 60

# =============================================================================
# VALIDATION
# =============================================================================

# Minimum length for a valid pain point description
MIN_PAIN_POINT_LENGTH = 20

# Minimum length for a valid idea description
MIN_IDEA_DESCRIPTION_LENGTH = 50

# Minimum length for a valid pitch brief
MIN_PITCH_BRIEF_LENGTH = 200

# Maximum length for URLs (validation)
MAX_URL_LENGTH = 2048

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_NO_PAIN_POINTS = (
    "Reached max initial mining attempts ({attempts}) with 0 pain points. "
    "This usually means: (1) LLM is failing to extract pain points from scraped content, "
    "(2) All extracted pain points are failing validation (no verbatim quotes), or "
    "(3) Domain '{domain}' has insufficient community discussion. "
    "Try a different domain or check LLM logs for extraction failures."
)

ERROR_INSUFFICIENT_PAIN_POINTS = (
    "Only {count} pain points found (target: {target}). "
    "Retrying mining (attempt {attempt}/{max_attempts})."
)

ERROR_MAX_IDEA_ATTEMPTS = (
    "Reached max idea generation attempts ({attempts}) with {count} ideas. "
    "Generated ideas may not meet quality thresholds."
)

ERROR_NO_SCORED_IDEAS = (
    "No ideas passed scoring threshold. All {count} ideas were marked as 'park'. "
    "Consider: (1) lowering quality thresholds, (2) mining more diverse pain points, "
    "or (3) trying a different domain."
)

ERROR_PITCH_GENERATION_FAILED = (
    "Failed to generate pitch brief after {attempts} attempts. Last error: {error}"
)

ERROR_LLM_TIMEOUT = (
    "LLM request timed out after {timeout} seconds. "
    "Consider increasing request_timeout or using a faster model."
)

ERROR_INVALID_JSON_RESPONSE = (
    "LLM returned invalid JSON. Expected format: {expected_format}. Received: {received}"
)

# Critic-specific errors
ERROR_CRITIC_LLM_INVOCATION_FAILED = "Critic LLM invocation failed: {error}"

ERROR_CRITIC_JSON_EXTRACTION_FAILED = "Failed to extract valid JSON from Critic LLM response"

ERROR_CRITIC_PARSE_FAILED = "Failed to parse critique: {error}"

WARNING_CRITIC_NO_BRIEFS = "No pitch briefs available for critique"

# Scorer-specific errors
ERROR_SCORER_LLM_INVOCATION_FAILED = "Scorer LLM invocation failed: {error}"

ERROR_SCORER_JSON_EXTRACTION_FAILED = "Failed to extract valid JSON from Scorer LLM response"

WARNING_SCORER_NO_IDEAS = "No ideas available for scoring"

# =============================================================================
# LOGGING
# =============================================================================

# Log format
LOG_FORMAT = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"

# Date format for logs
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log level
DEFAULT_LOG_LEVEL = "INFO"
