"""
Test helper to create pain points with the new evidence structure.
Use this in test files instead of manually creating PainPoint objects.
"""
from uuid import uuid4
from src.state.schema import PainPoint, PainPointEvidence, PainPointRubric, DataSource


def make_test_pain_point(
    title: str = "Test Pain Point",
    description: str = "Test description for pain point",
    source_url: str = "https://news.ycombinator.com/item?id=12345",
    raw_quote: str = "Test quote from user",
    source: DataSource = DataSource.HACKERNEWS,
    passes_rubric: bool = True,
    additional_evidence: list[tuple[str, str, DataSource]] = None,
) -> PainPoint:
    """Create a test pain point with evidence structure.
    
    Args:
        title: Pain point title
        description: Pain point description
        source_url: Primary evidence URL
        raw_quote: Primary evidence quote
        source: Primary evidence source
        passes_rubric: Whether rubric passes
        additional_evidence: List of (url, quote, source) tuples for extra evidence
    
    Returns:
        PainPoint with evidence array
    """
    # Primary evidence
    evidence = [
        PainPointEvidence(
            source_url=source_url,
            raw_quote=raw_quote,
            source=source,
        )
    ]
    
    # Add additional evidence if provided
    if additional_evidence:
        for url, quote, src in additional_evidence:
            evidence.append(
                PainPointEvidence(
                    source_url=url,
                    raw_quote=quote,
                    source=src,
                )
            )
    
    return PainPoint(
        id=uuid4(),
        title=title,
        description=description,
        rubric=PainPointRubric(
            is_genuine_current_frustration=True,
            has_verbatim_quote=True,
            user_segment_specific=True,
        ),
        passes_rubric=passes_rubric,
        evidence=evidence,
    )


def make_test_pain_point_dict(
    title: str = "Test Pain Point",
    description: str = "Test description",
    source_url: str = "https://news.ycombinator.com/item?id=12345",
    raw_quote: str = "Test quote",
    source: str = "hackernews",
    additional_evidence: list[dict] = None,
) -> dict:
    """Create a test pain point dict (for LLM response mocking).
    
    Returns dict in the format the LLM would return.
    """
    evidence = [
        {
            "source_url": source_url,
            "raw_quote": raw_quote,
            "source": source,
        }
    ]
    
    if additional_evidence:
        evidence.extend(additional_evidence)
    
    return {
        "id": str(uuid4()),
        "title": title,
        "description": description,
        "rubric": {
            "is_genuine_current_frustration": True,
            "has_verbatim_quote": True,
            "user_segment_specific": True,
        },
        "passes_rubric": True,
        "evidence": evidence,
    }
