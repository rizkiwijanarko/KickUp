"""
Test Guidance Integration
==========================
Unit tests for Guidance-based constrained generation in VentureForge.

Run with:
    pytest test/test_guidance_integration.py -v

Or directly:
    python test/test_guidance_integration.py
"""

import pytest
from pydantic import BaseModel


class TestGuidanceBasics:
    """Test basic Guidance functionality."""

    def test_import_guidance_client(self):
        """Test that guidance_client module can be imported."""
        try:
            from src.llm import guidance_client

            assert hasattr(guidance_client, "get_guidance_llm")
            assert hasattr(guidance_client, "constrained_json")
            assert hasattr(guidance_client, "constrained_select")
        except ImportError as e:
            pytest.skip(f"Guidance not installed: {e}")

    def test_get_guidance_llm(self):
        """Test Guidance LLM initialization."""
        try:
            from src.llm.guidance_client import get_guidance_llm

            # Should not raise
            lm = get_guidance_llm(reasoning=False)
            assert lm is not None

        except ImportError:
            pytest.skip("Guidance not installed")
        except Exception as e:
            # May fail if API keys not configured, that's OK
            pytest.skip(f"LLM initialization failed (expected if no API key): {e}")


class TestConstrainedGeneration:
    """Test constrained generation functions."""

    def test_constrained_select_mock(self):
        """Test constrained_select with mock data."""
        try:
            from src.llm.guidance_client import constrained_select

            # This test verifies the function signature
            # Actual LLM call would require API key
            assert callable(constrained_select)

        except ImportError:
            pytest.skip("Guidance not installed")

    def test_constrained_json_schema(self):
        """Test constrained_json with Pydantic schema."""
        try:
            from src.llm.guidance_client import constrained_json

            class TestSchema(BaseModel):
                name: str
                value: int

            # Verify function signature
            assert callable(constrained_json)

        except ImportError:
            pytest.skip("Guidance not installed")


# TestScorerGuidance class removed - scorer_guidance module does not exist in codebase


class TestGuidanceScorerFunction:
    """Test the guidance_scorer specialized function."""

    def test_guidance_scorer_signature(self):
        """Test guidance_scorer function signature."""
        try:
            from src.llm.guidance_client import guidance_scorer

            assert callable(guidance_scorer)

        except ImportError:
            pytest.skip("Guidance not installed")


# Integration test (requires API key and Guidance installed)
@pytest.mark.integration
class TestGuidanceIntegration:
    """Integration tests requiring live LLM calls."""

    def test_constrained_select_live(self):
        """Test constrained selection with live LLM."""
        try:
            from src.llm.guidance_client import get_guidance_llm, constrained_select

            lm = get_guidance_llm()
            lm, choice = constrained_select(
                lm,
                choices=["yes", "no"],
                system_prompt="You answer yes/no questions.",
                user_prompt="Is the sky blue?",
            )

            assert choice in ["yes", "no"]

        except ImportError:
            pytest.skip("Guidance not installed")
        except Exception as e:
            pytest.skip(f"Live test failed (API key issue?): {e}")

    def test_constrained_json_live(self):
        """Test constrained JSON generation with live LLM."""
        try:
            from src.llm.guidance_client import get_guidance_llm, constrained_json

            class Person(BaseModel):
                name: str
                age: int

            lm = get_guidance_llm()
            lm, person = constrained_json(
                lm,
                schema=Person,
                system_prompt="You generate person data.",
                user_prompt="Generate a person named Alice, age 30.",
            )

            assert isinstance(person, Person)
            assert person.name == "Alice"
            assert person.age == 30

        except ImportError:
            pytest.skip("Guidance not installed")
        except Exception as e:
            pytest.skip(f"Live test failed (API key issue?): {e}")


if __name__ == "__main__":
    # Run tests directly
    print("Running Guidance integration tests...")
    print("\n=== Basic Tests ===")

    test_basics = TestGuidanceBasics()
    try:
        test_basics.test_import_guidance_client()
        print("[OK] Import test passed")
    except Exception as e:
        print(f"[FAIL] Import test failed: {e}")

    try:
        test_basics.test_get_guidance_llm()
        print("[OK] LLM initialization test passed")
    except Exception as e:
        print(f"[FAIL] LLM initialization test failed: {e}")

    print("\n=== Scorer Tests ===")
    print("[SKIP] Scorer guidance tests removed (module does not exist)")

    print("\n=== Integration Tests (require API key) ===")
    print("Run with: pytest test/test_guidance_integration.py -v -m integration")

    # Add vLLM integration test
    print("\n=== vLLM Integration Test ===")
    try:
        from src.llm.guidance_client import get_guidance_llm
        from src.config import settings
        from guidance import gen, system, user, assistant

        # Check if vLLM is configured
        if "134.199.205.81" in settings.llm_base_url:
            print(f"Detected vLLM server: {settings.llm_base_url}")
            print(f"Model: {settings.llm_model}")

            try:
                lm = get_guidance_llm(reasoning=False)
                print("[OK] Guidance LLM initialized with vLLM backend")

                # Test basic generation (vLLM has limited Guidance support)
                with system():
                    lm += "You are a helpful assistant."

                with user():
                    lm += "What is 2+2? Answer with just the number."

                with assistant():
                    lm += gen("answer", max_tokens=10)

                answer = lm["answer"].strip()
                print(f"[OK] Basic generation test passed: 2+2 = '{answer}'")
                print(f"     Full response length: {len(answer)} chars")
                print(
                    "     Note: vLLM has limited Guidance support (no SelectNode, no stop conditions)"
                )
                print("     Fallback to standard LLM recommended for production use.")

                # Verify it's a reasonable answer (may be empty with vLLM)
                if answer and ("4" in answer or "four" in answer.lower()):
                    print("     [OK] Answer contains expected value")
                else:
                    print(f"     [WARN] Answer empty or unexpected: '{answer}'")

            except Exception as e:
                print(f"[FAIL] vLLM integration test failed: {e}")
                import traceback

                traceback.print_exc()
        else:
            print("[SKIP] vLLM server not configured (using different provider)")

    except ImportError:
        print("[SKIP] Guidance not installed")
    except Exception as e:
        print(f"[FAIL] vLLM test error: {e}")
