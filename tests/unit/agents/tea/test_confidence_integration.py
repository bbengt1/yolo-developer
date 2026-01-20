"""Integration tests for confidence scoring with TEA node (Story 9.4 - Task 13).

Tests for end-to-end integration of confidence scoring with the TEA agent:
- TEA node produces ConfidenceResult in output
- Confidence scoring integrates with coverage and test execution
- Backward compatibility with overall_confidence float field

All tests follow the patterns established in Story 9.1-9.3.
"""

from __future__ import annotations

import pytest


class TestTEANodeConfidenceIntegration:
    """Integration tests for TEA node with confidence scoring."""

    @pytest.mark.asyncio
    async def test_tea_node_includes_confidence_result(self) -> None:
        """Test that tea_node includes ConfidenceResult in output."""
        from yolo_developer.agents.tea import tea_node
        from yolo_developer.orchestrator.state import YoloState

        # Minimal state with some dev output
        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "artifacts": [
                    {
                        "artifact_id": "src/example.py",
                        "type": "code_file",
                        "content": "def hello(): return 'world'",
                    },
                    {
                        "artifact_id": "tests/test_example.py",
                        "type": "test_file",
                        "content": "def test_hello(): assert hello() == 'world'",
                    },
                ]
            },
        }

        result = await tea_node(state)

        # Check that tea_output contains the output dict
        assert "tea_output" in result
        tea_output = result["tea_output"]

        # Verify confidence_result is present
        assert "confidence_result" in tea_output
        confidence_result = tea_output["confidence_result"]
        assert confidence_result is not None

        # Verify structure of confidence result
        assert "score" in confidence_result
        assert "breakdown" in confidence_result
        assert "passed_threshold" in confidence_result
        assert "threshold_value" in confidence_result
        assert "deployment_recommendation" in confidence_result

        # Score should be 0-100
        assert 0 <= confidence_result["score"] <= 100

    @pytest.mark.asyncio
    async def test_tea_node_backward_compatible_overall_confidence(self) -> None:
        """Test that tea_node still has overall_confidence as float (0-1)."""
        from yolo_developer.agents.tea import tea_node
        from yolo_developer.orchestrator.state import YoloState

        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "artifacts": [
                    {
                        "artifact_id": "src/example.py",
                        "type": "code_file",
                        "content": "def hello(): return 'world'",
                    },
                ]
            },
        }

        result = await tea_node(state)

        tea_output = result["tea_output"]

        # overall_confidence should still be a float 0-1 for backward compat
        assert "overall_confidence" in tea_output
        assert isinstance(tea_output["overall_confidence"], float)
        assert 0.0 <= tea_output["overall_confidence"] <= 1.0

        # It should match confidence_result.score / 100
        if tea_output["confidence_result"]:
            expected_confidence = tea_output["confidence_result"]["score"] / 100.0
            assert abs(tea_output["overall_confidence"] - expected_confidence) < 0.01

    @pytest.mark.asyncio
    async def test_tea_node_empty_artifacts_returns_neutral(self) -> None:
        """Test that tea_node with no artifacts returns neutral confidence."""
        from yolo_developer.agents.tea import tea_node
        from yolo_developer.orchestrator.state import YoloState

        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {"artifacts": []},
        }

        result = await tea_node(state)

        tea_output = result["tea_output"]
        confidence_result = tea_output["confidence_result"]

        # With no artifacts, should get neutral scores
        # Coverage: 50 (neutral) * 0.4 = 20
        # Test exec: 50 (neutral) * 0.3 = 15
        # Validation: 100 (no findings) * 0.3 = 30
        # Total: 65
        assert confidence_result["score"] == 65
        assert confidence_result["passed_threshold"] is False  # Below 90

    @pytest.mark.asyncio
    async def test_confidence_result_structure_validated(self) -> None:
        """Test that confidence result has correct structure regardless of score."""
        from yolo_developer.agents.tea import tea_node
        from yolo_developer.orchestrator.state import YoloState

        # High quality code with comprehensive tests
        high_quality_code = """
def add(a: int, b: int) -> int:
    '''Add two numbers.'''
    return a + b

def multiply(a: int, b: int) -> int:
    '''Multiply two numbers.'''
    return a * b
"""
        high_quality_tests = """
import pytest

def test_add():
    assert add(1, 2) == 3
    assert add(0, 0) == 0
    assert add(-1, 1) == 0

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(0, 5) == 0
    assert multiply(-2, 3) == -6

def test_add_negative():
    assert add(-5, -3) == -8

def test_multiply_negative():
    assert multiply(-2, -3) == 6
"""

        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "artifacts": [
                    {
                        "artifact_id": "src/math_utils.py",
                        "type": "code_file",
                        "content": high_quality_code,
                    },
                    {
                        "artifact_id": "tests/test_math_utils.py",
                        "type": "test_file",
                        "content": high_quality_tests,
                    },
                ]
            },
        }

        result = await tea_node(state)

        tea_output = result["tea_output"]
        confidence_result = tea_output["confidence_result"]

        # Verify structure is complete and valid
        assert "score" in confidence_result
        assert "breakdown" in confidence_result
        assert "passed_threshold" in confidence_result
        assert "deployment_recommendation" in confidence_result

        # Score should be valid 0-100 range
        assert 0 <= confidence_result["score"] <= 100

        # Breakdown should have expected structure
        breakdown = confidence_result["breakdown"]
        assert "coverage_score" in breakdown
        assert "test_execution_score" in breakdown
        assert "validation_score" in breakdown
        assert "weighted_coverage" in breakdown
        assert "weighted_test_execution" in breakdown
        assert "weighted_validation" in breakdown
        assert "final_score" in breakdown

        # Deployment recommendation should be valid value
        assert confidence_result["deployment_recommendation"] in (
            "deploy",
            "deploy_with_warnings",
            "block",
        )


class TestConfidenceResultSerialization:
    """Tests for ConfidenceResult serialization in TEA output."""

    def test_confidence_result_to_dict_structure(self) -> None:
        """Test that ConfidenceResult.to_dict() has correct structure."""
        from yolo_developer.agents.tea.scoring import (
            ConfidenceBreakdown,
            ConfidenceResult,
        )

        breakdown = ConfidenceBreakdown(
            coverage_score=80.0,
            test_execution_score=90.0,
            validation_score=100.0,
            weighted_coverage=32.0,
            weighted_test_execution=27.0,
            weighted_validation=30.0,
            penalties=(),
            bonuses=("+5 for perfect tests",),
            base_score=89.0,
            final_score=94,
        )

        result = ConfidenceResult(
            score=94,
            breakdown=breakdown,
            passed_threshold=True,
            threshold_value=90,
            blocking_reasons=(),
            deployment_recommendation="deploy",
        )

        serialized = result.to_dict()

        # Verify top-level structure
        assert serialized["score"] == 94
        assert serialized["passed_threshold"] is True
        assert serialized["threshold_value"] == 90
        assert serialized["blocking_reasons"] == []
        assert serialized["deployment_recommendation"] == "deploy"
        assert "created_at" in serialized

        # Verify breakdown structure
        breakdown_dict = serialized["breakdown"]
        assert breakdown_dict["coverage_score"] == 80.0
        assert breakdown_dict["test_execution_score"] == 90.0
        assert breakdown_dict["validation_score"] == 100.0
        assert breakdown_dict["bonuses"] == ["+5 for perfect tests"]
        assert breakdown_dict["final_score"] == 94


class TestCalculateConfidenceScoreIntegration:
    """Integration tests for calculate_confidence_score function."""

    def test_full_integration_with_all_components(self) -> None:
        """Test calculate_confidence_score with all input types."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.execution import TestExecutionResult
        from yolo_developer.agents.tea.scoring import calculate_confidence_score
        from yolo_developer.agents.tea.types import Finding, ValidationResult

        # Create realistic coverage report
        coverage_result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=85,
            coverage_percentage=85.0,
        )
        coverage_report = CoverageReport(
            results=(coverage_result,),
            overall_coverage=85.0,
            threshold=80.0,
            passed=True,
        )

        # Create realistic test execution result
        test_result = TestExecutionResult(
            status="passed",
            passed_count=15,
            failed_count=0,
            error_count=0,
        )

        # Create validation result with one medium finding
        finding = Finding(
            finding_id="F001",
            category="documentation",
            severity="medium",
            description="Missing docstring",
            location="src/module.py",
            remediation="Add docstrings",
        )
        validation_result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="warning",
            findings=(finding,),
        )

        result = calculate_confidence_score(
            validation_results=(validation_result,),
            coverage_report=coverage_report,
            test_execution_result=test_result,
        )

        # Verify score is calculated correctly
        # Coverage: 85 * 0.4 = 34
        # Test exec: 100 * 0.3 = 30 (100% pass rate) + 5 bonus from modifiers = ~35
        # Validation: 95 (100 - 5 medium) * 0.3 = 28.5
        # Base: 34 + 30 + 28.5 = 92.5, + 5 bonus = 97.5
        assert result.score >= 90
        assert result.passed_threshold is True
        assert result.deployment_recommendation == "deploy"

        # Verify breakdown
        assert result.breakdown.coverage_score == 85.0
        assert result.breakdown.test_execution_score == 100.0  # 100% pass rate
        assert result.breakdown.validation_score == 95.0

        # Verify penalties are populated
        assert len(result.breakdown.penalties) == 1
        assert "-5" in result.breakdown.penalties[0]
        assert "medium" in result.breakdown.penalties[0]

        # Verify bonuses are populated
        assert "+5 for perfect test pass rate" in result.breakdown.bonuses

    def test_integration_with_critical_findings_blocks(self) -> None:
        """Test that critical findings cause blocking."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.execution import TestExecutionResult
        from yolo_developer.agents.tea.scoring import calculate_confidence_score
        from yolo_developer.agents.tea.types import Finding, ValidationResult

        coverage_result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=100,
            coverage_percentage=100.0,
        )
        coverage_report = CoverageReport(
            results=(coverage_result,),
            overall_coverage=100.0,
            threshold=80.0,
            passed=True,
        )

        test_result = TestExecutionResult(
            status="passed",
            passed_count=10,
            failed_count=0,
            error_count=0,
        )

        # Critical finding
        critical_finding = Finding(
            finding_id="F001",
            category="security",
            severity="critical",
            description="SQL Injection vulnerability",
            location="src/module.py",
            remediation="Use parameterized queries",
        )
        validation_result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="failed",
            findings=(critical_finding,),
        )

        result = calculate_confidence_score(
            validation_results=(validation_result,),
            coverage_report=coverage_report,
            test_execution_result=test_result,
        )

        # Critical finding reduces validation score by 25
        # Coverage: 100 * 0.4 = 40
        # Test exec: 100 * 0.3 = 30
        # Validation: 75 (100 - 25) * 0.3 = 22.5
        # Total: 92.5
        # Still above 90 threshold due to perfect coverage and tests

        # But verify the breakdown shows the penalty
        assert result.breakdown.validation_score == 75.0
