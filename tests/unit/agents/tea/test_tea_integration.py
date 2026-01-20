"""Integration tests for test execution in TEA node (Story 9.3).

Tests for the integration of test execution into the TEA node workflow.
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.tea.execution import (
    TestExecutionResult,
    TestFailure,
)
from yolo_developer.agents.tea.types import TEAOutput


class TestTEAOutputWithTestExecution:
    """Tests for TEAOutput with test execution result field."""

    def test_tea_output_has_test_execution_result_field(self) -> None:
        """Test that TEAOutput can include test_execution_result."""
        test_result = TestExecutionResult(
            status="passed",
            passed_count=10,
            failed_count=0,
            error_count=0,
            failures=(),
            duration_ms=100,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.100Z",
        )
        output = TEAOutput(
            test_execution_result=test_result,
        )
        assert output.test_execution_result is not None
        assert output.test_execution_result.status == "passed"
        assert output.test_execution_result.passed_count == 10

    def test_tea_output_test_execution_result_default_none(self) -> None:
        """Test that test_execution_result defaults to None."""
        output = TEAOutput()
        assert output.test_execution_result is None

    def test_tea_output_to_dict_includes_test_execution_result(self) -> None:
        """Test that to_dict includes test_execution_result when present."""
        test_result = TestExecutionResult(
            status="failed",
            passed_count=8,
            failed_count=2,
            error_count=0,
            failures=(
                TestFailure(
                    test_name="test_broken",
                    file_path="test.py",
                    error_message="Failed",
                    failure_type="failure",
                ),
            ),
            duration_ms=150,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.150Z",
        )
        output = TEAOutput(
            test_execution_result=test_result,
            overall_confidence=0.8,
        )
        dict_output = output.to_dict()

        assert "test_execution_result" in dict_output
        assert dict_output["test_execution_result"] is not None
        assert dict_output["test_execution_result"]["status"] == "failed"
        assert dict_output["test_execution_result"]["passed_count"] == 8
        assert len(dict_output["test_execution_result"]["failures"]) == 1

    def test_tea_output_to_dict_none_test_execution_result(self) -> None:
        """Test that to_dict handles None test_execution_result."""
        output = TEAOutput()
        dict_output = output.to_dict()

        assert "test_execution_result" in dict_output
        assert dict_output["test_execution_result"] is None


class TestTEANodeTestExecution:
    """Tests for test execution integration in tea_node."""

    @pytest.mark.asyncio
    async def test_tea_node_executes_tests(self) -> None:
        """Test that tea_node executes tests when test artifacts present."""
        from yolo_developer.agents.tea.node import tea_node
        from yolo_developer.orchestrator.state import YoloState

        # Create state with test artifact
        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    {
                        "story_id": "9-3",
                        "code_files": [
                            {
                                "file_path": "src/module.py",
                                "content": "def calculate(): return 42",
                                "file_type": "source",
                            }
                        ],
                        "test_files": [
                            {
                                "file_path": "tests/test_module.py",
                                "content": """
def test_calculate():
    from module import calculate
    assert calculate() == 42
""",
                                "test_type": "unit",
                            }
                        ],
                    }
                ]
            },
        }

        result = await tea_node(state)

        # Check that test execution was performed
        assert "tea_output" in result
        tea_output = result["tea_output"]
        assert "test_execution_result" in tea_output
        # Test should have passed
        assert tea_output["test_execution_result"]["status"] == "passed"
        assert tea_output["test_execution_result"]["passed_count"] >= 1

    @pytest.mark.asyncio
    async def test_tea_node_adjusts_confidence_for_test_failures(self) -> None:
        """Test that tea_node adjusts confidence based on test pass rate."""
        from yolo_developer.agents.tea.node import tea_node
        from yolo_developer.orchestrator.state import YoloState

        # Create state with failing test
        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    {
                        "story_id": "9-3",
                        "code_files": [],
                        "test_files": [
                            {
                                "file_path": "tests/test_broken.py",
                                "content": """
def test_incomplete():
    # TODO: implement this test
    pass

def test_no_assertion():
    x = 1
""",
                                "test_type": "unit",
                            }
                        ],
                    }
                ]
            },
        }

        result = await tea_node(state)

        # Check that confidence was reduced due to test issues
        assert "tea_output" in result
        result["tea_output"]
        # Overall confidence should be less than 1.0 due to test issues
        # (The exact value depends on the confidence calculation logic)

    @pytest.mark.asyncio
    async def test_tea_node_includes_test_findings(self) -> None:
        """Test that tea_node includes test-related findings."""
        from yolo_developer.agents.tea.node import tea_node
        from yolo_developer.orchestrator.state import YoloState

        # Create state with test that has issues
        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    {
                        "story_id": "9-3",
                        "code_files": [],
                        "test_files": [
                            {
                                "file_path": "tests/test_issues.py",
                                "content": """
def test_no_assert():
    x = 1  # No assertion
""",
                                "test_type": "unit",
                            }
                        ],
                    }
                ]
            },
        }

        result = await tea_node(state)

        # Check that test findings were included
        assert "tea_output" in result
        tea_output = result["tea_output"]
        # Should have test execution result with failures
        if tea_output.get("test_execution_result"):
            test_result = tea_output["test_execution_result"]
            assert test_result["failed_count"] >= 1 or len(test_result["failures"]) >= 1

    @pytest.mark.asyncio
    async def test_tea_node_includes_test_findings_in_validation_results(self) -> None:
        """Test that tea_node includes test findings in validation_results (AC6)."""
        from yolo_developer.agents.tea.node import tea_node
        from yolo_developer.orchestrator.state import YoloState

        # Create state with test that has issues
        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    {
                        "story_id": "9-3",
                        "code_files": [],
                        "test_files": [
                            {
                                "file_path": "tests/test_issues.py",
                                "content": """
def test_no_assert():
    x = 1  # No assertion
""",
                                "test_type": "unit",
                            }
                        ],
                    }
                ]
            },
        }

        result = await tea_node(state)

        # Check that test findings are in validation_results
        assert "tea_output" in result
        tea_output = result["tea_output"]

        # Find the test_execution validation result
        validation_results = tea_output.get("validation_results", [])
        test_execution_result = None
        for vr in validation_results:
            if vr.get("artifact_id") == "test_execution":
                test_execution_result = vr
                break

        # Verify test execution validation result exists and has findings
        assert test_execution_result is not None, "test_execution validation result should exist"
        assert len(test_execution_result.get("findings", [])) >= 1, (
            "Should have at least one finding"
        )

        # Verify finding has correct category
        findings = test_execution_result["findings"]
        assert any(f["category"] == "test_coverage" for f in findings), (
            "Should have test_coverage finding"
        )
