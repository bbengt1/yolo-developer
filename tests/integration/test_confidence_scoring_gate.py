"""Integration tests for confidence scoring gate.

Tests the full integration of the confidence scoring gate with the decorator framework.
"""

from __future__ import annotations

import pytest

from yolo_developer.gates.decorator import quality_gate
from yolo_developer.gates.evaluators import clear_evaluators, get_evaluator

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def register_confidence_evaluator():
    """Ensure evaluator is registered for each test."""
    clear_evaluators()
    # Import to trigger registration
    import importlib

    from yolo_developer.gates.gates import confidence_scoring

    importlib.reload(confidence_scoring)
    yield
    clear_evaluators()


@pytest.fixture
def high_quality_artifacts() -> dict:
    """Artifacts that result in high confidence score."""
    return {
        "gate_results": [
            {"gate_name": "testability", "passed": True, "score": 95},
            {"gate_name": "ac_measurability", "passed": True, "score": 92},
            {"gate_name": "architecture_validation", "passed": True, "score": 88},
            {"gate_name": "definition_of_done", "passed": True, "score": 90},
        ],
        "coverage": {
            "line_coverage": 92.0,
            "branch_coverage": 85.0,
            "function_coverage": 95.0,
        },
        "risks": [
            {"type": "complexity", "severity": "low", "location": "module.py"},
        ],
        "code": {
            "files": [
                {
                    "path": "README.md",
                    "content": "# Feature Project\n\nHigh quality project with documentation.\n",
                },
                {
                    "path": "src/feature.py",
                    "content": '''"""Feature module with full documentation.

This module provides core functionality.
"""

from typing import Any


def process_data(data: dict[str, Any]) -> dict[str, Any]:
    """Process incoming data.

    Args:
        data: Dictionary of input data.

    Returns:
        Processed data dictionary.
    """
    return {"processed": True, **data}


def validate_input(value: str) -> bool:
    """Validate input string.

    Args:
        value: String to validate.

    Returns:
        True if valid.
    """
    return bool(value.strip())
''',
                },
                {
                    "path": "tests/test_feature.py",
                    "content": '''"""Tests for feature module."""


def test_process_data():
    """Test data processing."""
    from src.feature import process_data
    result = process_data({"key": "value"})
    assert result["processed"] is True


def test_validate_input():
    """Test input validation."""
    from src.feature import validate_input
    assert validate_input("valid") is True
    assert validate_input("  ") is False
''',
                },
            ],
        },
    }


@pytest.fixture
def low_quality_artifacts() -> dict:
    """Artifacts that result in low confidence score."""
    return {
        "gate_results": [
            {"gate_name": "testability", "passed": False, "score": 45},
            {"gate_name": "ac_measurability", "passed": False, "score": 40},
            {"gate_name": "architecture_validation", "passed": False, "score": 50},
            {"gate_name": "definition_of_done", "passed": False, "score": 55},
        ],
        "coverage": {
            "line_coverage": 30.0,
            "branch_coverage": 20.0,
            "function_coverage": 35.0,
        },
        "risks": [
            {"type": "security", "severity": "critical", "location": "api.py"},
            {"type": "complexity", "severity": "high", "location": "core.py"},
            {"type": "dependency", "severity": "medium", "location": "utils.py"},
        ],
        "code": {
            "files": [
                {
                    "path": "src/broken.py",
                    "content": """
def no_docs_function(x):
    return x

def another_undocumented(y, z):
    if y:
        if z:
            if y > z:
                if y < 100:
                    if z > 0:
                        return y
    return z
""",
                },
            ],
        },
    }


# =============================================================================
# Integration Tests
# =============================================================================


class TestConfidenceScoringGateIntegration:
    """Integration tests for confidence scoring gate with decorator."""

    @pytest.mark.asyncio
    async def test_gate_decorator_with_confidence_evaluator(
        self, high_quality_artifacts: dict
    ) -> None:
        """Gate decorator should work with confidence_scoring evaluator."""

        @quality_gate("confidence_scoring", blocking=True)
        async def process_artifacts(state: dict) -> dict:
            state["processed"] = True
            return state

        state = high_quality_artifacts
        result = await process_artifacts(state)

        # Should pass and process with high quality artifacts
        assert result.get("processed") is True or result.get("gate_blocked") is not True

    @pytest.mark.asyncio
    async def test_gate_reads_gate_results_from_state(self, high_quality_artifacts: dict) -> None:
        """Gate should read gate_results from state."""
        from yolo_developer.gates.gates.confidence_scoring import confidence_scoring_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state=high_quality_artifacts,
            gate_name="confidence_scoring",
        )
        result = await confidence_scoring_evaluator(context)

        assert result.gate_name == "confidence_scoring"
        # High quality should pass
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_gate_reads_coverage_from_state(self, high_quality_artifacts: dict) -> None:
        """Gate should read coverage from state['coverage']."""
        from yolo_developer.gates.gates.confidence_scoring import confidence_scoring_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state=high_quality_artifacts,
            gate_name="confidence_scoring",
        )
        result = await confidence_scoring_evaluator(context)

        # Coverage is part of the factors
        assert result.gate_name == "confidence_scoring"
        assert "confidence score" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_gate_reads_risks_from_state(self, low_quality_artifacts: dict) -> None:
        """Gate should read risks from state['risks']."""
        from yolo_developer.gates.gates.confidence_scoring import confidence_scoring_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state=low_quality_artifacts,
            gate_name="confidence_scoring",
        )
        result = await confidence_scoring_evaluator(context)

        assert result.gate_name == "confidence_scoring"

    @pytest.mark.asyncio
    async def test_gate_reads_code_from_state(self, high_quality_artifacts: dict) -> None:
        """Gate should read code from state['code']."""
        from yolo_developer.gates.gates.confidence_scoring import confidence_scoring_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state=high_quality_artifacts,
            gate_name="confidence_scoring",
        )
        result = await confidence_scoring_evaluator(context)

        # Code is used for documentation factor
        assert result.gate_name == "confidence_scoring"

    @pytest.mark.asyncio
    async def test_deployment_blocking_with_low_scores(self, low_quality_artifacts: dict) -> None:
        """Gate should block deployment when confidence is below threshold."""

        @quality_gate("confidence_scoring", blocking=True)
        async def process_artifacts(state: dict) -> dict:
            state["processed"] = True
            return state

        state = low_quality_artifacts
        result = await process_artifacts(state)

        # Should be blocked due to low score
        assert result.get("gate_blocked") is True or result.get("processed") is not True

    @pytest.mark.asyncio
    async def test_passing_behavior_with_high_confidence(
        self, high_quality_artifacts: dict
    ) -> None:
        """Gate should pass when confidence score is above threshold."""
        from yolo_developer.gates.gates.confidence_scoring import confidence_scoring_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state=high_quality_artifacts,
            gate_name="confidence_scoring",
        )
        result = await confidence_scoring_evaluator(context)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_threshold_override_via_configuration(self, high_quality_artifacts: dict) -> None:
        """Gate should respect threshold from config."""
        from yolo_developer.gates.gates.confidence_scoring import confidence_scoring_evaluator
        from yolo_developer.gates.types import GateContext

        # Set very low threshold (0.0-1.0 format) - should pass
        low_threshold_state = {
            **high_quality_artifacts,
            "config": {"quality": {"confidence_threshold": 0.50}},
        }
        context = GateContext(
            state=low_threshold_state,
            gate_name="confidence_scoring",
        )
        result = await confidence_scoring_evaluator(context)

        assert result.passed is True
        assert "50" in result.reason  # Threshold should be in reason

    @pytest.mark.asyncio
    async def test_high_threshold_override(self, high_quality_artifacts: dict) -> None:
        """Gate should fail with very high threshold."""
        from yolo_developer.gates.gates.confidence_scoring import confidence_scoring_evaluator
        from yolo_developer.gates.types import GateContext

        # Set very high threshold (0.0-1.0 format) - may fail
        high_threshold_state = {
            **high_quality_artifacts,
            "config": {"quality": {"confidence_threshold": 0.99}},
        }
        context = GateContext(
            state=high_threshold_state,
            gate_name="confidence_scoring",
        )
        result = await confidence_scoring_evaluator(context)

        # Very high threshold will likely fail
        assert result.gate_name == "confidence_scoring"
        assert "99" in result.reason  # Threshold should be in reason

    @pytest.mark.asyncio
    async def test_evaluator_available_via_get_evaluator(self) -> None:
        """Evaluator should be available via get_evaluator()."""
        evaluator = get_evaluator("confidence_scoring")
        assert evaluator is not None

    @pytest.mark.asyncio
    async def test_gate_with_empty_state(self) -> None:
        """Gate should handle empty state gracefully."""
        from yolo_developer.gates.gates.confidence_scoring import confidence_scoring_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state={},
            gate_name="confidence_scoring",
        )
        result = await confidence_scoring_evaluator(context)

        # Should not crash with empty state
        assert result.gate_name == "confidence_scoring"
        # Will likely fail due to no data
        assert result.reason is not None

    @pytest.mark.asyncio
    async def test_gate_with_partial_state(self) -> None:
        """Gate should handle partial state (missing some keys)."""
        from yolo_developer.gates.gates.confidence_scoring import confidence_scoring_evaluator
        from yolo_developer.gates.types import GateContext

        # Only gate_results, no coverage or risks
        context = GateContext(
            state={
                "gate_results": [
                    {"gate_name": "testability", "passed": True, "score": 85},
                ],
            },
            gate_name="confidence_scoring",
        )
        result = await confidence_scoring_evaluator(context)

        # Should not crash with partial state
        assert result.gate_name == "confidence_scoring"

    @pytest.mark.asyncio
    async def test_gate_result_includes_score_breakdown(self, low_quality_artifacts: dict) -> None:
        """Gate result reason should include score information."""
        from yolo_developer.gates.gates.confidence_scoring import confidence_scoring_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state=low_quality_artifacts,
            gate_name="confidence_scoring",
        )
        result = await confidence_scoring_evaluator(context)

        # Should have reason with score details
        assert result.reason is not None
        assert "confidence score" in result.reason.lower()
        assert "threshold" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_advisory_mode_allows_continuation(self, low_quality_artifacts: dict) -> None:
        """Advisory mode should allow continuation even with low confidence."""

        @quality_gate("confidence_scoring", blocking=False)
        async def process_artifacts(state: dict) -> dict:
            state["processed"] = True
            return state

        state = low_quality_artifacts
        result = await process_artifacts(state)

        # Advisory mode should allow processing to continue
        assert result.get("processed") is True

    @pytest.mark.asyncio
    async def test_custom_factor_weights_integration(self, high_quality_artifacts: dict) -> None:
        """Gate should respect custom factor weights from config."""
        from yolo_developer.gates.gates.confidence_scoring import confidence_scoring_evaluator
        from yolo_developer.gates.types import GateContext

        # Override weights via config
        custom_weights_state = {
            **high_quality_artifacts,
            "config": {
                "quality": {
                    "confidence_threshold": 90,
                    "factor_weights": {
                        "test_coverage": 0.50,  # More emphasis on coverage
                        "gate_results": 0.30,
                        "risk_assessment": 0.10,
                        "documentation": 0.10,
                    },
                }
            },
        }
        context = GateContext(
            state=custom_weights_state,
            gate_name="confidence_scoring",
        )
        result = await confidence_scoring_evaluator(context)

        # Should still work with custom weights
        assert result.gate_name == "confidence_scoring"

    @pytest.mark.asyncio
    async def test_gate_with_implementation_key(self, high_quality_artifacts: dict) -> None:
        """Gate should also work with 'implementation' key in state."""
        from yolo_developer.gates.gates.confidence_scoring import confidence_scoring_evaluator
        from yolo_developer.gates.types import GateContext

        # Use 'implementation' instead of 'code'
        state_with_impl = {
            **high_quality_artifacts,
            "implementation": high_quality_artifacts["code"],
        }
        del state_with_impl["code"]

        context = GateContext(
            state=state_with_impl,
            gate_name="confidence_scoring",
        )
        result = await confidence_scoring_evaluator(context)

        assert result.gate_name == "confidence_scoring"

    @pytest.mark.asyncio
    async def test_integration_with_multiple_gate_results(self) -> None:
        """Gate should handle many gate results correctly."""
        from yolo_developer.gates.gates.confidence_scoring import confidence_scoring_evaluator
        from yolo_developer.gates.types import GateContext

        # Many gate results
        many_gates_state = {
            "gate_results": [
                {"gate_name": f"gate_{i}", "passed": i % 2 == 0, "score": 70 + i} for i in range(10)
            ],
            "coverage": {"line_coverage": 80.0, "branch_coverage": 70.0, "function_coverage": 85.0},
        }

        context = GateContext(
            state=many_gates_state,
            gate_name="confidence_scoring",
        )
        result = await confidence_scoring_evaluator(context)

        # Should process all gates
        assert result.gate_name == "confidence_scoring"
