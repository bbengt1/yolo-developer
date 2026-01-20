"""Integration tests for testability audit (Story 9.6).

Tests for:
- Full testability audit flow
- TEA node integration with testability report
- TestabilityReport included in TEAOutput
- Testability findings in validation results
"""

from __future__ import annotations

import pytest


class TestTestabilityAuditIntegration:
    """Integration tests for the full testability audit flow."""

    def test_full_audit_flow(self) -> None:
        """Test complete audit flow from code files to report."""
        from yolo_developer.agents.tea.testability import audit_testability

        # Code with multiple testability issues
        code_files = [
            {
                "artifact_id": "src/module.py",
                "content": """
cache = []  # Global state

class Service:
    def __init__(self):
        self.db = Database()  # Tight coupling

    def process(self, data):
        import json  # Hidden dependency
        return json.dumps(data)
""",
            }
        ]

        report = audit_testability(code_files)

        # Verify report structure
        assert report.score.base_score == 100
        assert report.metrics.files_analyzed == 1
        assert report.metrics.total_issues >= 3  # At least 3 issues
        assert len(report.recommendations) > 0
        assert report.created_at is not None

        # Verify issues are found
        patterns_found = {issue.pattern_type for issue in report.issues}
        assert "global_state" in patterns_found
        assert "tight_coupling" in patterns_found
        assert "hidden_dependency" in patterns_found

    def test_audit_with_clean_code(self) -> None:
        """Test audit with clean code produces high score."""
        from yolo_developer.agents.tea.testability import audit_testability

        # Clean code without testability issues
        code_files = [
            {
                "artifact_id": "src/clean.py",
                "content": """
from __future__ import annotations

class Service:
    def __init__(self, db) -> None:
        self.db = db  # Dependency injection

    def get_data(self) -> dict:
        return self.db.query()
""",
            }
        ]

        report = audit_testability(code_files)

        assert report.score.score == 100
        assert report.metrics.total_issues == 0
        assert report.metrics.files_with_issues == 0

    def test_finding_conversion_integration(self) -> None:
        """Test that testability issues convert to findings correctly."""
        from yolo_developer.agents.tea.testability import (
            audit_testability,
            convert_testability_issues_to_findings,
        )

        code_files = [
            {
                "artifact_id": "src/module.py",
                "content": "cache = []",  # Global state
            }
        ]

        report = audit_testability(code_files)
        findings = convert_testability_issues_to_findings(report.issues)

        assert len(findings) == 1
        finding = findings[0]

        # Verify finding structure
        assert finding.finding_id.startswith("F-")
        assert finding.category == "code_quality"
        assert finding.severity == "critical"  # Global state is critical
        assert "Testability:" in finding.description
        assert "src/module.py" in finding.location

    def test_metrics_aggregation(self) -> None:
        """Test metrics are correctly aggregated across files."""
        from yolo_developer.agents.tea.testability import audit_testability

        code_files = [
            {"artifact_id": "src/a.py", "content": "data = {}"},  # Global state
            {"artifact_id": "src/b.py", "content": "items = []"},  # Global state
            {"artifact_id": "src/c.py", "content": "def foo(): pass"},  # Clean
        ]

        report = audit_testability(code_files)

        assert report.metrics.files_analyzed == 3
        assert report.metrics.files_with_issues == 2
        assert report.metrics.total_issues == 2
        assert report.metrics.issues_by_severity["critical"] == 2

    def test_score_with_mixed_severities(self) -> None:
        """Test score calculation with mixed severity issues."""
        from yolo_developer.agents.tea.testability import audit_testability

        # Create a file that generates issues of different severities
        lines = ["cache = []"]  # Critical - global state
        lines.append(
            """
class Service:
    def __init__(self):
        self.db = Database()  # High - tight coupling
"""
        )
        # Long method (medium)
        lines.append("def long_func():")
        lines.extend(["    pass"] * 54)  # > 50 lines

        code_files = [{"artifact_id": "src/mixed.py", "content": "\n".join(lines)}]

        report = audit_testability(code_files)

        # Score should be reduced by penalties
        assert report.score.score < 100
        assert "critical_penalty" in report.score.breakdown
        assert report.score.breakdown["critical_penalty"] < 0


class TestTEANodeTestabilityIntegration:
    """Tests for TEA node integration with testability audit."""

    @pytest.mark.asyncio
    async def test_tea_node_includes_testability_report(self) -> None:
        """Test that tea_node output includes testability report."""
        from yolo_developer.agents.tea import tea_node
        from yolo_developer.orchestrator.state import YoloState

        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    {
                        "story_id": "test-story",
                        "code_files": [
                            {
                                "file_path": "src/module.py",
                                "content": "cache = []",  # Global state
                                "file_type": "source",
                            }
                        ],
                        "test_files": [],
                    }
                ]
            },
        }

        result = await tea_node(state)

        # Verify testability report is in output
        assert "tea_output" in result
        tea_output = result["tea_output"]

        assert "testability_report" in tea_output
        assert tea_output["testability_report"] is not None
        assert tea_output["testability_report"]["score"]["score"] < 100

    @pytest.mark.asyncio
    async def test_tea_node_testability_findings_in_validation(self) -> None:
        """Test that testability issues appear as validation findings."""
        from yolo_developer.agents.tea import tea_node
        from yolo_developer.orchestrator.state import YoloState

        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    {
                        "story_id": "test-story",
                        "code_files": [
                            {
                                "file_path": "src/module.py",
                                "content": "data = {}",  # Global state
                                "file_type": "source",
                            }
                        ],
                        "test_files": [],
                    }
                ]
            },
        }

        result = await tea_node(state)

        # Verify testability audit appears in validation results
        tea_output = result["tea_output"]
        validation_results = tea_output["validation_results"]

        # Find testability audit result
        testability_result = next(
            (r for r in validation_results if r["artifact_id"] == "testability_audit"),
            None,
        )

        assert testability_result is not None
        assert len(testability_result["findings"]) > 0
        assert any("Testability:" in f["description"] for f in testability_result["findings"])

    @pytest.mark.asyncio
    async def test_tea_node_processing_notes_include_testability(self) -> None:
        """Test that processing notes include testability summary."""
        from yolo_developer.agents.tea import tea_node
        from yolo_developer.orchestrator.state import YoloState

        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    {
                        "story_id": "test-story",
                        "code_files": [
                            {
                                "file_path": "src/module.py",
                                "content": "cache = []",
                                "file_type": "source",
                            }
                        ],
                        "test_files": [],
                    }
                ]
            },
        }

        result = await tea_node(state)

        processing_notes = result["tea_output"]["processing_notes"]
        assert "Testability:" in processing_notes
        assert "score=" in processing_notes
        assert "issues=" in processing_notes

    @pytest.mark.asyncio
    async def test_tea_node_with_no_code_files(self) -> None:
        """Test tea_node handles no code files gracefully."""
        from yolo_developer.agents.tea import tea_node
        from yolo_developer.orchestrator.state import YoloState

        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    {
                        "story_id": "test-story",
                        "code_files": [],
                        "test_files": [],
                    }
                ]
            },
        }

        result = await tea_node(state)

        # Testability report should be None or empty
        tea_output = result["tea_output"]
        # With no code files, testability_report will be None
        assert tea_output["testability_report"] is None


class TestTEAOutputTestabilityReport:
    """Tests for TestabilityReport in TEAOutput."""

    def test_tea_output_with_testability_report(self) -> None:
        """Test TEAOutput includes testability_report field."""
        from yolo_developer.agents.tea.testability import (
            TestabilityMetrics,
            TestabilityReport,
            TestabilityScore,
        )
        from yolo_developer.agents.tea.types import TEAOutput

        score = TestabilityScore(score=80, base_score=100, breakdown={"critical_penalty": -20})
        metrics = TestabilityMetrics(
            total_issues=1,
            issues_by_severity={"critical": 1, "high": 0, "medium": 0, "low": 0},
            issues_by_pattern={"global_state": 1},
            files_analyzed=5,
            files_with_issues=1,
        )
        testability_report = TestabilityReport(
            issues=(),
            score=score,
            metrics=metrics,
            recommendations=("Fix global state",),
        )

        output = TEAOutput(
            validation_results=(),
            processing_notes="Test",
            testability_report=testability_report,
        )

        assert output.testability_report is not None
        assert output.testability_report.score.score == 80

    def test_tea_output_to_dict_includes_testability(self) -> None:
        """Test TEAOutput.to_dict() serializes testability_report."""
        from yolo_developer.agents.tea.testability import (
            TestabilityMetrics,
            TestabilityReport,
            TestabilityScore,
        )
        from yolo_developer.agents.tea.types import TEAOutput

        score = TestabilityScore(score=90, base_score=100, breakdown={})
        metrics = TestabilityMetrics(
            total_issues=0,
            issues_by_severity={},
            issues_by_pattern={},
            files_analyzed=3,
            files_with_issues=0,
        )
        testability_report = TestabilityReport(
            issues=(),
            score=score,
            metrics=metrics,
            recommendations=(),
        )

        output = TEAOutput(
            validation_results=(),
            processing_notes="Test",
            testability_report=testability_report,
        )

        result = output.to_dict()

        assert "testability_report" in result
        assert result["testability_report"] is not None
        assert result["testability_report"]["score"]["score"] == 90

    def test_tea_output_to_dict_without_testability(self) -> None:
        """Test TEAOutput.to_dict() handles None testability_report."""
        from yolo_developer.agents.tea.types import TEAOutput

        output = TEAOutput(
            validation_results=(),
            processing_notes="Test",
            testability_report=None,
        )

        result = output.to_dict()

        assert "testability_report" in result
        assert result["testability_report"] is None
