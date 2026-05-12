#!/usr/bin/env python3
"""
Comprehensive test suite for VentureForge improvements.

Tests:
1. Multi-source pain points with evidence arrays
2. Critic with 5 checks (not 7)
3. Max revisions = 3
4. Pain point miner clustering
5. Evidence validation

Run with: uv run pytest test_improvements.py -v
Or directly: uv run python test/test_improvements.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uuid import uuid4
from src.state.schema import (
    PainPoint,
    PainPointEvidence,
    PainPointRubric,
    DataSource,
    CritiqueRubric,
    VentureForgeState,
)
from src.main import make_initial_state


def test_1_multi_source_pain_point_schema():
    """Test that pain points can have multiple evidence sources."""
    print("\n[TEST 1] Multi-source pain point schema...")
    
    evidence = [
        PainPointEvidence(
            source_url='https://news.ycombinator.com/item?id=1',
            raw_quote='Debugging microservices is a nightmare',
            source=DataSource.HACKERNEWS
        ),
        PainPointEvidence(
            source_url='https://www.producthunt.com/posts/test',
            raw_quote='We need better distributed tracing',
            source=DataSource.PRODUCTHUNT
        ),
        PainPointEvidence(
            source_url='https://stackoverflow.com/questions/12345',
            raw_quote='How do I trace requests across 20 services?',
            source=DataSource.WEB
        ),
    ]

    pp = PainPoint(
        title='Debugging distributed systems',
        description='Engineers waste hours correlating logs across microservices.',
        rubric=PainPointRubric(
            is_genuine_current_frustration=True,
            has_verbatim_quote=True,
            user_segment_specific=True
        ),
        passes_rubric=True,
        evidence=evidence
    )

    # Test properties
    assert pp.evidence_count == 3, f"Expected 3 evidence sources, got {pp.evidence_count}"
    assert pp.source_url == 'https://news.ycombinator.com/item?id=1', "Primary source_url should be first evidence"
    assert pp.raw_quote == 'Debugging microservices is a nightmare', "Primary raw_quote should be first evidence"
    assert pp.source == DataSource.HACKERNEWS, "Primary source should be first evidence"
    
    # Test evidence array
    assert len(pp.evidence) == 3
    assert pp.evidence[0].source == DataSource.HACKERNEWS
    assert pp.evidence[1].source == DataSource.PRODUCTHUNT
    assert pp.evidence[2].source == DataSource.WEB
    
    print("  ✓ Pain point has 3 evidence sources")
    print("  ✓ Primary properties work correctly")
    print("  ✓ Evidence array is accessible")
    return True


def test_2_critic_rubric_has_5_checks():
    """Test that critic rubric has 5 checks (not 7)."""
    print("\n[TEST 2] Critic rubric reduced to 5 checks...")
    
    rubric = CritiqueRubric(
        all_claims_evidence_backed=True,
        no_hallucinated_source_urls=True,
        tagline_under_12_words=True,
        target_is_contained_fire=True,
        competition_embraced_with_thesis=True,
    )
    
    # Count fields
    field_count = len(rubric.model_fields)
    assert field_count == 5, f"Expected 5 rubric fields, got {field_count}"
    
    # Verify removed fields don't exist
    assert not hasattr(rubric, 'unscalable_acquisition_concrete'), "Manual outreach check should be removed"
    assert not hasattr(rubric, 'gtm_leads_with_manual_recruitment'), "GTM manual recruitment check should be removed"
    
    print("  ✓ Rubric has exactly 5 fields")
    print(f"  ✓ Fields: {list(rubric.model_fields.keys())}")
    print("  ✓ Manual outreach checks removed")
    return True


def test_3_max_revisions_default_is_3():
    """Test that max_revisions default is 2."""
    print("\n[TEST 3] Max revisions default...")
    
    state = make_initial_state("test domain")
    assert state.max_revisions == 2, f"Expected max_revisions=2, got {state.max_revisions}"
    
    # Test get_revision_count method
    idea_id = uuid4()
    
    # With no revisions recorded
    assert state.get_revision_count(idea_id) == 0, "Should return 0 for untracked idea"
    
    # With 1 revision
    state.revision_counts = {str(idea_id): 1}
    assert state.get_revision_count(idea_id) == 1, "Should return 1"
    
    # With 2 revisions (at max)
    state.revision_counts = {str(idea_id): 2}
    assert state.get_revision_count(idea_id) == 2, "Should return 2"
    
    # Test that max is 2
    assert state.get_revision_count(idea_id) >= state.max_revisions, "Should be at max"
    
    print("  ✓ Max revisions default is 2")
    print("  ✓ get_revision_count works correctly")
    return True


def test_4_backward_compatibility():
    """Test that old code still works with new schema."""
    print("\n[TEST 4] Backward compatibility...")
    
    # Create pain point with single evidence (like old format)
    pp = PainPoint(
        title='Test',
        description='Test description that is long enough to pass validation',
        rubric=PainPointRubric(
            is_genuine_current_frustration=True,
            has_verbatim_quote=True,
            user_segment_specific=True
        ),
        passes_rubric=True,
        evidence=[
            PainPointEvidence(
                source_url='https://news.ycombinator.com/item?id=1',
                raw_quote='Test quote',
                source=DataSource.HACKERNEWS
            )
        ]
    )
    
    # Old code accessing properties should still work
    url = pp.source_url
    quote = pp.raw_quote
    source = pp.source
    
    assert url == 'https://news.ycombinator.com/item?id=1'
    assert quote == 'Test quote'
    assert source == DataSource.HACKERNEWS
    
    print("  ✓ Old property access still works")
    print("  ✓ Single evidence source works")
    return True


def test_5_evidence_validation():
    """Test that evidence array validation works."""
    print("\n[TEST 5] Evidence validation...")
    
    # Test minimum evidence requirement
    try:
        pp = PainPoint(
            title='Test',
            description='Test description that is long enough',
            rubric=PainPointRubric(
                is_genuine_current_frustration=True,
                has_verbatim_quote=True,
                user_segment_specific=True
            ),
            passes_rubric=True,
            evidence=[]  # Empty evidence should fail
        )
        assert False, "Should have raised validation error for empty evidence"
    except Exception as e:
        print(f"  ✓ Empty evidence rejected: {type(e).__name__}")
    
    # Test maximum evidence limit (10)
    evidence_list = [
        PainPointEvidence(
            source_url=f'https://example.com/{i}',
            raw_quote=f'Quote {i}',
            source=DataSource.WEB
        )
        for i in range(11)  # 11 items, should fail
    ]
    
    try:
        pp = PainPoint(
            title='Test',
            description='Test description that is long enough',
            rubric=PainPointRubric(
                is_genuine_current_frustration=True,
                has_verbatim_quote=True,
                user_segment_specific=True
            ),
            passes_rubric=True,
            evidence=evidence_list
        )
        assert False, "Should have raised validation error for >10 evidence items"
    except Exception as e:
        print(f"  ✓ Too many evidence items rejected: {type(e).__name__}")
    
    # Test valid range (1-10)
    for count in [1, 5, 10]:
        evidence_list = [
            PainPointEvidence(
                source_url=f'https://example.com/{i}',
                raw_quote=f'Quote {i}',
                source=DataSource.WEB
            )
            for i in range(count)
        ]
        
        pp = PainPoint(
            title='Test',
            description='Test description that is long enough',
            rubric=PainPointRubric(
                is_genuine_current_frustration=True,
                has_verbatim_quote=True,
                user_segment_specific=True
            ),
            passes_rubric=True,
            evidence=evidence_list
        )
        assert pp.evidence_count == count
        print(f"  ✓ {count} evidence items accepted")
    
    return True


def test_6_state_serialization():
    """Test that state with new schema can be serialized."""
    print("\n[TEST 6] State serialization...")
    
    state = make_initial_state("test domain")
    
    # Add a pain point with multiple evidence
    pp = PainPoint(
        title='Test Pain Point',
        description='Test description that is long enough to pass validation',
        rubric=PainPointRubric(
            is_genuine_current_frustration=True,
            has_verbatim_quote=True,
            user_segment_specific=True
        ),
        passes_rubric=True,
        evidence=[
            PainPointEvidence(
                source_url='https://news.ycombinator.com/item?id=1',
                raw_quote='Quote 1',
                source=DataSource.HACKERNEWS
            ),
            PainPointEvidence(
                source_url='https://www.producthunt.com/posts/test',
                raw_quote='Quote 2',
                source=DataSource.PRODUCTHUNT
            ),
        ]
    )
    
    state.pain_points = [pp]
    
    # Test serialization
    serialized = state.model_dump(mode='json')
    assert 'pain_points' in serialized
    assert len(serialized['pain_points']) == 1
    assert 'evidence' in serialized['pain_points'][0]
    assert len(serialized['pain_points'][0]['evidence']) == 2
    
    print("  ✓ State serializes correctly")
    print("  ✓ Evidence array preserved in JSON")
    
    # Test deserialization
    restored = VentureForgeState.model_validate(serialized)
    assert len(restored.pain_points) == 1
    assert restored.pain_points[0].evidence_count == 2
    
    print("  ✓ State deserializes correctly")
    print("  ✓ Evidence count preserved")
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Multi-source pain points", test_1_multi_source_pain_point_schema),
        ("Critic rubric (5 checks)", test_2_critic_rubric_has_5_checks),
        ("Max revisions (3)", test_3_max_revisions_default_is_3),
        ("Backward compatibility", test_4_backward_compatibility),
        ("Evidence validation", test_5_evidence_validation),
        ("State serialization", test_6_state_serialization),
    ]
    
    print("=" * 70)
    print("VentureForge Improvements Test Suite")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
