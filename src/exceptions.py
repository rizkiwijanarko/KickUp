"""
VentureForge Custom Exceptions
===============================
Domain-specific exceptions for better error handling and debugging.
"""
from __future__ import annotations

from typing import Any



class VentureForgeError(Exception):
    """Base exception for all VentureForge errors."""

    pass


# =============================================================================
# PIPELINE ERRORS
# =============================================================================


class PipelineError(VentureForgeError):
    """Base class for pipeline execution errors."""

    pass


class PipelineStageError(PipelineError):
    """Error during a specific pipeline stage."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


class MaxAttemptsExceededError(PipelineError):
    """Maximum retry attempts exceeded."""

    def __init__(self, operation: str, max_attempts: int):
        self.operation = operation
        self.max_attempts = max_attempts
        super().__init__(
            f"{operation} failed after {max_attempts} attempts"
        )


# =============================================================================
# DATA VALIDATION ERRORS
# =============================================================================


class ValidationError(VentureForgeError):
    """Base class for data validation errors."""

    pass


class InvalidPainPointError(ValidationError):
    """Pain point failed validation checks."""

    def __init__(self, reason: str, pain_point_id: str | None = None):
        self.reason = reason
        self.pain_point_id = pain_point_id
        msg = f"Invalid pain point"
        if pain_point_id:
            msg += f" (ID: {pain_point_id})"
        msg += f": {reason}"
        super().__init__(msg)


class InvalidIdeaError(ValidationError):
    """Idea failed validation checks."""

    def __init__(self, reason: str, idea_id: str | None = None):
        self.reason = reason
        self.idea_id = idea_id
        msg = f"Invalid idea"
        if idea_id:
            msg += f" (ID: {idea_id})"
        msg += f": {reason}"
        super().__init__(msg)


class InvalidPitchError(ValidationError):
    """Pitch brief failed validation checks."""

    def __init__(self, reason: str, pitch_id: str | None = None):
        self.reason = reason
        self.pitch_id = pitch_id
        msg = f"Invalid pitch"
        if pitch_id:
            msg += f" (ID: {pitch_id})"
        msg += f": {reason}"
        super().__init__(msg)


class MissingRequiredFieldError(ValidationError):
    """Required field is missing from data structure."""

    def __init__(self, field_name: str, model_name: str):
        self.field_name = field_name
        self.model_name = model_name
        super().__init__(
            f"Required field '{field_name}' missing from {model_name}"
        )


# =============================================================================
# LLM ERRORS
# =============================================================================


class LLMError(VentureForgeError):
    """Base class for LLM-related errors."""

    pass


class LLMTimeoutError(LLMError):
    """LLM request timed out."""

    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"LLM request timed out after {timeout_seconds} seconds"
        )


class LLMInvalidResponseError(LLMError):
    """LLM returned invalid or unparseable response."""

    def __init__(self, expected_format: str, received: str):
        self.expected_format = expected_format
        self.received = received
        super().__init__(
            f"Expected {expected_format}, received: {received[:200]}"
        )


class LLMJSONParseError(LLMError):
    """Failed to parse JSON from LLM response."""

    def __init__(self, raw_response: str, parse_error: str):
        self.raw_response = raw_response
        self.parse_error = parse_error
        super().__init__(
            f"JSON parse error: {parse_error}. "
            f"Response: {raw_response[:200]}"
        )


class LLMRateLimitError(LLMError):
    """LLM API rate limit exceeded."""

    def __init__(self, retry_after: int | None = None):
        self.retry_after = retry_after
        msg = "LLM API rate limit exceeded"
        if retry_after:
            msg += f". Retry after {retry_after} seconds"
        super().__init__(msg)


# =============================================================================
# SCRAPING ERRORS
# =============================================================================


class ScrapingError(VentureForgeError):
    """Base class for web scraping errors."""

    pass


class NoDataFoundError(ScrapingError):
    """No data found from scraping source."""

    def __init__(self, source: str, domain: str):
        self.source = source
        self.domain = domain
        super().__init__(
            f"No data found from {source} for domain '{domain}'"
        )


class ScraperAPIError(ScrapingError):
    """External scraping API returned an error."""

    def __init__(self, api_name: str, status_code: int, message: str):
        self.api_name = api_name
        self.status_code = status_code
        self.message = message
        super().__init__(
            f"{api_name} API error (HTTP {status_code}): {message}"
        )


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================


class ConfigurationError(VentureForgeError):
    """Base class for configuration errors."""

    pass


class MissingAPIKeyError(ConfigurationError):
    """Required API key is missing."""

    def __init__(self, key_name: str):
        self.key_name = key_name
        super().__init__(
            f"Missing required API key: {key_name}. "
            f"Set it in .env or environment variables."
        )


class InvalidConfigurationError(ConfigurationError):
    """Configuration value is invalid."""

    def __init__(self, setting_name: str, value: Any, reason: str):
        self.setting_name = setting_name
        self.value = value
        self.reason = reason
        super().__init__(
            f"Invalid configuration for '{setting_name}' = {value}: {reason}"
        )



# =============================================================================
# STATE MANAGEMENT ERRORS
# =============================================================================


class StateError(VentureForgeError):
    """Base class for state management errors."""

    pass


class InvalidStateTransitionError(StateError):
    """Attempted invalid state transition."""

    def __init__(self, from_stage: str, to_stage: str, reason: str):
        self.from_stage = from_stage
        self.to_stage = to_stage
        self.reason = reason
        super().__init__(
            f"Invalid transition from {from_stage} to {to_stage}: {reason}"
        )


class StateCorruptionError(StateError):
    """State data is corrupted or inconsistent."""

    def __init__(self, field_name: str, reason: str):
        self.field_name = field_name
        self.reason = reason
        super().__init__(
            f"State corruption detected in '{field_name}': {reason}"
        )
