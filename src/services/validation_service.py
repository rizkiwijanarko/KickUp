"""
Validation Service
==================
Centralized validation logic for all domain models.
"""

from typing import Any

from src.constants import (
    MIN_IDEA_DESCRIPTION_LENGTH,
    MIN_PAIN_POINT_LENGTH,
    MIN_PITCH_BRIEF_LENGTH,
    MIN_PAIN_POINTS_PER_IDEA,
    MAX_URL_LENGTH,
    REQUIRED_PITCH_SECTIONS,
)
from src.exceptions import (
    InvalidIdeaError,
    InvalidPainPointError,
    InvalidPitchError,
    MissingRequiredFieldError,
)


def validate_pain_point(
    description: str,
    source_url: str,
    raw_quote: str | None = None,
) -> None:
    """
    Validate a pain point meets minimum requirements.

    Args:
        description: Pain point description
        source_url: Source URL for the pain point
        raw_quote: Optional verbatim quote from source

    Raises:
        InvalidPainPointError: If validation fails
    """
    if not description or len(description.strip()) < MIN_PAIN_POINT_LENGTH:
        raise InvalidPainPointError(f"Description too short (min {MIN_PAIN_POINT_LENGTH} chars)")

    if not source_url or not source_url.startswith(("http://", "https://")):
        raise InvalidPainPointError(f"Invalid source URL: {source_url}")

    if len(source_url) > MAX_URL_LENGTH:
        raise InvalidPainPointError(f"Source URL too long (max {MAX_URL_LENGTH} chars)")

    if raw_quote is not None and not raw_quote.strip():
        raise InvalidPainPointError("Raw quote provided but empty")


def validate_idea(
    description: str,
    pain_point_ids: list[str],
    one_liner: str | None = None,
) -> None:
    """
    Validate an idea meets minimum requirements.

    Args:
        description: Idea description
        pain_point_ids: List of pain point IDs this idea addresses
        one_liner: Optional one-line summary

    Raises:
        InvalidIdeaError: If validation fails
    """
    if not description or len(description.strip()) < MIN_IDEA_DESCRIPTION_LENGTH:
        raise InvalidIdeaError(f"Description too short (min {MIN_IDEA_DESCRIPTION_LENGTH} chars)")

    if not pain_point_ids or len(pain_point_ids) < MIN_PAIN_POINTS_PER_IDEA:
        raise InvalidIdeaError(f"Must address at least {MIN_PAIN_POINTS_PER_IDEA} pain points")

    if one_liner is not None and not one_liner.strip():
        raise InvalidIdeaError("One-liner provided but empty")


def validate_pitch_brief(
    content: str,
    sections: dict[str, str] | None = None,
) -> None:
    """
    Validate a pitch brief meets minimum requirements.

    Args:
        content: Full pitch brief content
        sections: Optional dict of section_name -> section_content

    Raises:
        InvalidPitchError: If validation fails
    """
    if not content or len(content.strip()) < MIN_PITCH_BRIEF_LENGTH:
        raise InvalidPitchError(f"Content too short (min {MIN_PITCH_BRIEF_LENGTH} chars)")

    if sections:
        missing_sections = [
            section
            for section in REQUIRED_PITCH_SECTIONS
            if section not in sections or not sections[section].strip()
        ]
        if missing_sections:
            raise InvalidPitchError(f"Missing required sections: {', '.join(missing_sections)}")


def validate_url(url: str, field_name: str = "URL") -> None:
    """
    Validate a URL format.

    Args:
        url: URL to validate
        field_name: Name of the field (for error messages)

    Raises:
        InvalidPainPointError: If URL is invalid
    """
    if not url:
        raise InvalidPainPointError(f"{field_name} is required")

    if not url.startswith(("http://", "https://")):
        raise InvalidPainPointError(f"{field_name} must start with http:// or https://")

    if len(url) > MAX_URL_LENGTH:
        raise InvalidPainPointError(f"{field_name} too long (max {MAX_URL_LENGTH} chars)")


def validate_required_field(
    value: Any,
    field_name: str,
    model_name: str,
) -> None:
    """
    Validate that a required field is present and non-empty.

    Args:
        value: Field value to check
        field_name: Name of the field
        model_name: Name of the model/class

    Raises:
        MissingRequiredFieldError: If field is missing or empty
    """
    if value is None:
        raise MissingRequiredFieldError(field_name, model_name)

    if isinstance(value, str) and not value.strip():
        raise MissingRequiredFieldError(field_name, model_name)

    if isinstance(value, (list, dict)) and not value:
        raise MissingRequiredFieldError(field_name, model_name)


def validate_rubric_all_pass(rubric_dict: dict[str, bool]) -> bool:
    """
    Check if all rubric checks pass.

    Args:
        rubric_dict: Dictionary of check_name -> bool

    Returns:
        True if all checks pass, False otherwise
    """
    return all(rubric_dict.values())


def validate_score_threshold(
    score: int,
    threshold: int,
    max_score: int,
) -> bool:
    """
    Check if a score meets a threshold.

    Args:
        score: Actual score
        threshold: Minimum required score
        max_score: Maximum possible score

    Returns:
        True if score >= threshold, False otherwise
    """
    if score < 0 or score > max_score:
        raise ValueError(f"Score {score} out of range [0, {max_score}]")

    if threshold < 0 or threshold > max_score:
        raise ValueError(f"Threshold {threshold} out of range [0, {max_score}]")

    return score >= threshold
