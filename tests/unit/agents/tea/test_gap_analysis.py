"""Unit tests for gap analysis types and functions (Story 9.8).

Tests cover:
- TestGap: Type creation and serialization
- GapPriority: Type creation and serialization
- TestSuggestion: Type creation and serialization
- GapAnalysisSummary: Type creation and serialization
- GapAnalysisReport: Type creation and serialization
- Immutability (frozen dataclass)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


class TestGapType:
    """Tests for GapType Literal type (AC: 1)."""

    def test_gap_type_values(self) -> None:
        """GapType accepts valid values."""
        from yolo_developer.agents.tea.gap_analysis import GapType

        # These should all be valid GapType values
        valid_types: list[GapType] = ["no_tests", "partial_coverage", "untested_branch"]
        assert len(valid_types) == 3


class TestGapSeverity:
    """Tests for GapSeverity Literal type (AC: 2)."""

    def test_gap_severity_values(self) -> None:
        """GapSeverity accepts valid values."""
        from yolo_developer.agents.tea.gap_analysis import GapSeverity

        valid_severities: list[GapSeverity] = ["critical", "high", "medium", "low"]
        assert len(valid_severities) == 4


class TestSuggestionTestType:
    """Tests for SuggestionTestType Literal type (AC: 3)."""

    def test_suggestion_test_type_values(self) -> None:
        """SuggestionTestType accepts valid values."""
        from yolo_developer.agents.tea.gap_analysis import SuggestionTestType

        valid_types: list[SuggestionTestType] = ["unit", "integration", "e2e"]
        assert len(valid_types) == 3


class TestTestGapType:
    """Tests for TestGap dataclass (AC: 1)."""

    def test_test_gap_creation(self) -> None:
        """TestGap can be created with required fields."""
        from yolo_developer.agents.tea.gap_analysis import TestGap

        gap = TestGap(
            gap_id="GAP-a1b2c3-001",
            file_path="src/module.py",
            function_names=("calculate_total", "process_data"),
            uncovered_lines=((10, 20), (30, 40)),
            gap_type="no_tests",
            description="Module has no tests",
        )

        assert gap.gap_id == "GAP-a1b2c3-001"
        assert gap.file_path == "src/module.py"
        assert gap.function_names == ("calculate_total", "process_data")
        assert gap.uncovered_lines == ((10, 20), (30, 40))
        assert gap.gap_type == "no_tests"
        assert gap.description == "Module has no tests"

    def test_test_gap_to_dict(self) -> None:
        """TestGap.to_dict() returns dictionary representation."""
        from yolo_developer.agents.tea.gap_analysis import TestGap

        gap = TestGap(
            gap_id="GAP-a1b2c3-001",
            file_path="src/module.py",
            function_names=("func1",),
            uncovered_lines=((10, 15),),
            gap_type="partial_coverage",
            description="Partial coverage",
        )

        result = gap.to_dict()

        assert result["gap_id"] == "GAP-a1b2c3-001"
        assert result["file_path"] == "src/module.py"
        assert result["function_names"] == ["func1"]
        assert result["uncovered_lines"] == [[10, 15]]
        assert result["gap_type"] == "partial_coverage"
        assert result["description"] == "Partial coverage"

    def test_test_gap_immutability(self) -> None:
        """TestGap is immutable (frozen dataclass)."""
        from yolo_developer.agents.tea.gap_analysis import TestGap

        gap = TestGap(
            gap_id="GAP-001",
            file_path="src/test.py",
            function_names=(),
            uncovered_lines=(),
            gap_type="no_tests",
            description="No tests",
        )

        with pytest.raises(FrozenInstanceError):
            gap.gap_id = "GAP-002"  # type: ignore[misc]

    def test_test_gap_default_values(self) -> None:
        """TestGap has correct default values."""
        from yolo_developer.agents.tea.gap_analysis import TestGap

        gap = TestGap(
            gap_id="GAP-001",
            file_path="src/test.py",
            function_names=(),
            uncovered_lines=(),
            gap_type="no_tests",
            description="",
        )

        assert gap.function_names == ()
        assert gap.uncovered_lines == ()

    def test_test_gap_whitespace_description(self) -> None:
        """TestGap handles whitespace-only description."""
        from yolo_developer.agents.tea.gap_analysis import TestGap

        gap = TestGap(
            gap_id="GAP-001",
            file_path="src/test.py",
            function_names=(),
            uncovered_lines=(),
            gap_type="no_tests",
            description="   \t\n  ",
        )

        # Whitespace-only descriptions should be accepted (no validation at dataclass level)
        assert gap.description == "   \t\n  "
        result = gap.to_dict()
        assert result["description"] == "   \t\n  "


class TestGapPriorityType:
    """Tests for GapPriority dataclass (AC: 2)."""

    def test_gap_priority_creation(self) -> None:
        """GapPriority can be created with required fields."""
        from yolo_developer.agents.tea.gap_analysis import GapPriority

        priority = GapPriority(
            gap_id="GAP-a1b2c3-001",
            severity="critical",
            risk_score=95,
            priority_rank=1,
        )

        assert priority.gap_id == "GAP-a1b2c3-001"
        assert priority.severity == "critical"
        assert priority.risk_score == 95
        assert priority.priority_rank == 1

    def test_gap_priority_to_dict(self) -> None:
        """GapPriority.to_dict() returns dictionary representation."""
        from yolo_developer.agents.tea.gap_analysis import GapPriority

        priority = GapPriority(
            gap_id="GAP-001",
            severity="high",
            risk_score=75,
            priority_rank=2,
        )

        result = priority.to_dict()

        assert result["gap_id"] == "GAP-001"
        assert result["severity"] == "high"
        assert result["risk_score"] == 75
        assert result["priority_rank"] == 2

    def test_gap_priority_immutability(self) -> None:
        """GapPriority is immutable (frozen dataclass)."""
        from yolo_developer.agents.tea.gap_analysis import GapPriority

        priority = GapPriority(
            gap_id="GAP-001",
            severity="low",
            risk_score=25,
            priority_rank=5,
        )

        with pytest.raises(FrozenInstanceError):
            priority.risk_score = 50  # type: ignore[misc]


class TestTestSuggestionType:
    """Tests for TestSuggestion dataclass (AC: 3)."""

    def test_test_suggestion_creation(self) -> None:
        """TestSuggestion can be created with required fields."""
        from yolo_developer.agents.tea.gap_analysis import TestSuggestion

        suggestion = TestSuggestion(
            suggestion_id="SUG-001",
            target_gap_id="GAP-a1b2c3-001",
            test_type="unit",
            description="Add unit test for calculate_total function",
            estimated_impact=5.0,
            example_signature="test_calculate_total_returns_correct_sum",
        )

        assert suggestion.suggestion_id == "SUG-001"
        assert suggestion.target_gap_id == "GAP-a1b2c3-001"
        assert suggestion.test_type == "unit"
        assert suggestion.description == "Add unit test for calculate_total function"
        assert suggestion.estimated_impact == 5.0
        assert suggestion.example_signature == "test_calculate_total_returns_correct_sum"

    def test_test_suggestion_to_dict(self) -> None:
        """TestSuggestion.to_dict() returns dictionary representation."""
        from yolo_developer.agents.tea.gap_analysis import TestSuggestion

        suggestion = TestSuggestion(
            suggestion_id="SUG-002",
            target_gap_id="GAP-002",
            test_type="integration",
            description="Add integration test",
            estimated_impact=10.0,
            example_signature="test_api_integration",
        )

        result = suggestion.to_dict()

        assert result["suggestion_id"] == "SUG-002"
        assert result["target_gap_id"] == "GAP-002"
        assert result["test_type"] == "integration"
        assert result["description"] == "Add integration test"
        assert result["estimated_impact"] == 10.0
        assert result["example_signature"] == "test_api_integration"

    def test_test_suggestion_immutability(self) -> None:
        """TestSuggestion is immutable (frozen dataclass)."""
        from yolo_developer.agents.tea.gap_analysis import TestSuggestion

        suggestion = TestSuggestion(
            suggestion_id="SUG-001",
            target_gap_id="GAP-001",
            test_type="unit",
            description="Test",
            estimated_impact=5.0,
            example_signature="test_something",
        )

        with pytest.raises(FrozenInstanceError):
            suggestion.estimated_impact = 10.0  # type: ignore[misc]


class TestGapAnalysisSummaryType:
    """Tests for GapAnalysisSummary dataclass (AC: 5)."""

    def test_gap_analysis_summary_creation(self) -> None:
        """GapAnalysisSummary can be created with required fields."""
        from yolo_developer.agents.tea.gap_analysis import GapAnalysisSummary

        summary = GapAnalysisSummary(
            total_gaps=10,
            critical_gaps=2,
            high_gaps=3,
            medium_gaps=3,
            low_gaps=2,
            total_suggestions=15,
            estimated_effort="2-4 hours",
        )

        assert summary.total_gaps == 10
        assert summary.critical_gaps == 2
        assert summary.high_gaps == 3
        assert summary.medium_gaps == 3
        assert summary.low_gaps == 2
        assert summary.total_suggestions == 15
        assert summary.estimated_effort == "2-4 hours"

    def test_gap_analysis_summary_to_dict(self) -> None:
        """GapAnalysisSummary.to_dict() returns dictionary representation."""
        from yolo_developer.agents.tea.gap_analysis import GapAnalysisSummary

        summary = GapAnalysisSummary(
            total_gaps=5,
            critical_gaps=1,
            high_gaps=2,
            medium_gaps=1,
            low_gaps=1,
            total_suggestions=8,
            estimated_effort="1-2 hours",
        )

        result = summary.to_dict()

        assert result["total_gaps"] == 5
        assert result["critical_gaps"] == 1
        assert result["high_gaps"] == 2
        assert result["medium_gaps"] == 1
        assert result["low_gaps"] == 1
        assert result["total_suggestions"] == 8
        assert result["estimated_effort"] == "1-2 hours"

    def test_gap_analysis_summary_immutability(self) -> None:
        """GapAnalysisSummary is immutable (frozen dataclass)."""
        from yolo_developer.agents.tea.gap_analysis import GapAnalysisSummary

        summary = GapAnalysisSummary(
            total_gaps=5,
            critical_gaps=1,
            high_gaps=2,
            medium_gaps=1,
            low_gaps=1,
            total_suggestions=8,
            estimated_effort="1 hour",
        )

        with pytest.raises(FrozenInstanceError):
            summary.total_gaps = 10  # type: ignore[misc]


class TestGapAnalysisReportType:
    """Tests for GapAnalysisReport dataclass (AC: 4)."""

    def test_gap_analysis_report_creation(self) -> None:
        """GapAnalysisReport can be created with required fields."""
        from yolo_developer.agents.tea.gap_analysis import (
            GapAnalysisReport,
            GapAnalysisSummary,
            GapPriority,
            TestGap,
            TestSuggestion,
        )

        gap = TestGap(
            gap_id="GAP-001",
            file_path="src/test.py",
            function_names=("func1",),
            uncovered_lines=((1, 10),),
            gap_type="no_tests",
            description="No tests",
        )
        priority = GapPriority(
            gap_id="GAP-001",
            severity="high",
            risk_score=80,
            priority_rank=1,
        )
        suggestion = TestSuggestion(
            suggestion_id="SUG-001",
            target_gap_id="GAP-001",
            test_type="unit",
            description="Add test",
            estimated_impact=10.0,
            example_signature="test_func1",
        )
        summary = GapAnalysisSummary(
            total_gaps=1,
            critical_gaps=0,
            high_gaps=1,
            medium_gaps=0,
            low_gaps=0,
            total_suggestions=1,
            estimated_effort="30 minutes",
        )

        report = GapAnalysisReport(
            gaps=(gap,),
            priorities=(priority,),
            suggestions=(suggestion,),
            summary=summary,
            coverage_baseline=75.0,
            projected_coverage=85.0,
        )

        assert len(report.gaps) == 1
        assert len(report.priorities) == 1
        assert len(report.suggestions) == 1
        assert report.summary.total_gaps == 1
        assert report.coverage_baseline == 75.0
        assert report.projected_coverage == 85.0
        assert report.created_at is not None

    def test_gap_analysis_report_to_dict(self) -> None:
        """GapAnalysisReport.to_dict() returns dictionary representation."""
        from yolo_developer.agents.tea.gap_analysis import (
            GapAnalysisReport,
            GapAnalysisSummary,
        )

        summary = GapAnalysisSummary(
            total_gaps=0,
            critical_gaps=0,
            high_gaps=0,
            medium_gaps=0,
            low_gaps=0,
            total_suggestions=0,
            estimated_effort="None",
        )

        report = GapAnalysisReport(
            gaps=(),
            priorities=(),
            suggestions=(),
            summary=summary,
            coverage_baseline=90.0,
            projected_coverage=90.0,
        )

        result = report.to_dict()

        assert result["gaps"] == []
        assert result["priorities"] == []
        assert result["suggestions"] == []
        assert result["summary"]["total_gaps"] == 0
        assert result["coverage_baseline"] == 90.0
        assert result["projected_coverage"] == 90.0
        assert "created_at" in result

    def test_gap_analysis_report_immutability(self) -> None:
        """GapAnalysisReport is immutable (frozen dataclass)."""
        from yolo_developer.agents.tea.gap_analysis import (
            GapAnalysisReport,
            GapAnalysisSummary,
        )

        summary = GapAnalysisSummary(
            total_gaps=0,
            critical_gaps=0,
            high_gaps=0,
            medium_gaps=0,
            low_gaps=0,
            total_suggestions=0,
            estimated_effort="None",
        )

        report = GapAnalysisReport(
            gaps=(),
            priorities=(),
            suggestions=(),
            summary=summary,
            coverage_baseline=80.0,
            projected_coverage=80.0,
        )

        with pytest.raises(FrozenInstanceError):
            report.coverage_baseline = 90.0  # type: ignore[misc]

    def test_gap_analysis_report_default_values(self) -> None:
        """GapAnalysisReport has correct default values."""
        from yolo_developer.agents.tea.gap_analysis import (
            GapAnalysisReport,
            GapAnalysisSummary,
        )

        summary = GapAnalysisSummary(
            total_gaps=0,
            critical_gaps=0,
            high_gaps=0,
            medium_gaps=0,
            low_gaps=0,
            total_suggestions=0,
            estimated_effort="None",
        )

        report = GapAnalysisReport(
            gaps=(),
            priorities=(),
            suggestions=(),
            summary=summary,
            coverage_baseline=100.0,
            projected_coverage=100.0,
        )

        assert report.gaps == ()
        assert report.priorities == ()
        assert report.suggestions == ()


# =============================================================================
# Task 2: Gap Identification Tests (AC: 1)
# =============================================================================


class TestGapIdGeneration:
    """Tests for gap ID generation (AC: 1)."""

    def test_generate_gap_id_format(self) -> None:
        """_generate_gap_id produces correct format."""
        from yolo_developer.agents.tea.gap_analysis import _generate_gap_id

        gap_id = _generate_gap_id("src/module.py", 1)

        # Format: GAP-{hash[:6]}-{seq:03d}
        assert gap_id.startswith("GAP-")
        parts = gap_id.split("-")
        assert len(parts) == 3
        assert len(parts[1]) == 6  # 6-char hash
        assert parts[2] == "001"

    def test_generate_gap_id_sequence(self) -> None:
        """_generate_gap_id increments sequence correctly."""
        from yolo_developer.agents.tea.gap_analysis import _generate_gap_id

        gap_id1 = _generate_gap_id("src/module.py", 1)
        gap_id2 = _generate_gap_id("src/module.py", 2)
        gap_id10 = _generate_gap_id("src/module.py", 10)

        assert gap_id1.endswith("-001")
        assert gap_id2.endswith("-002")
        assert gap_id10.endswith("-010")

    def test_generate_gap_id_different_files(self) -> None:
        """_generate_gap_id produces different hashes for different files."""
        from yolo_developer.agents.tea.gap_analysis import _generate_gap_id

        gap_id1 = _generate_gap_id("src/module_a.py", 1)
        gap_id2 = _generate_gap_id("src/module_b.py", 1)

        # Same sequence but different hash
        assert gap_id1 != gap_id2
        assert gap_id1.split("-")[1] != gap_id2.split("-")[1]


class TestIdentifyUntestedFunctions:
    """Tests for identify_untested_functions (AC: 1)."""

    def test_identify_untested_functions_zero_coverage(self) -> None:
        """identify_untested_functions finds files with 0% coverage."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.gap_analysis import identify_untested_functions

        report = CoverageReport(
            results=(
                CoverageResult(
                    file_path="src/untested.py",
                    lines_total=50,
                    lines_covered=0,
                    coverage_percentage=0.0,
                    uncovered_lines=((1, 50),),
                ),
            ),
            overall_coverage=0.0,
            threshold=80.0,
            passed=False,
        )

        gaps = identify_untested_functions(report)

        assert len(gaps) == 1
        assert gaps[0].file_path == "src/untested.py"
        assert gaps[0].gap_type == "no_tests"

    def test_identify_untested_functions_empty_report(self) -> None:
        """identify_untested_functions returns empty for empty report."""
        from yolo_developer.agents.tea.coverage import CoverageReport
        from yolo_developer.agents.tea.gap_analysis import identify_untested_functions

        report = CoverageReport(results=(), overall_coverage=100.0, threshold=80.0, passed=True)

        gaps = identify_untested_functions(report)

        assert len(gaps) == 0


class TestIdentifyPartialCoverageGaps:
    """Tests for identify_partial_coverage_gaps (AC: 1)."""

    def test_identify_partial_coverage_gaps_below_threshold(self) -> None:
        """identify_partial_coverage_gaps finds files below threshold."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.gap_analysis import identify_partial_coverage_gaps

        report = CoverageReport(
            results=(
                CoverageResult(
                    file_path="src/partial.py",
                    lines_total=100,
                    lines_covered=60,
                    coverage_percentage=60.0,
                    uncovered_lines=((30, 50), (70, 90)),
                ),
            ),
            overall_coverage=60.0,
            threshold=80.0,
            passed=False,
        )

        gaps = identify_partial_coverage_gaps(report, threshold=80.0)

        assert len(gaps) == 1
        assert gaps[0].file_path == "src/partial.py"
        assert gaps[0].gap_type == "partial_coverage"

    def test_identify_partial_coverage_gaps_above_threshold(self) -> None:
        """identify_partial_coverage_gaps skips files above threshold."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.gap_analysis import identify_partial_coverage_gaps

        report = CoverageReport(
            results=(
                CoverageResult(
                    file_path="src/covered.py",
                    lines_total=100,
                    lines_covered=90,
                    coverage_percentage=90.0,
                    uncovered_lines=((90, 100),),
                ),
            ),
            overall_coverage=90.0,
            threshold=80.0,
            passed=True,
        )

        gaps = identify_partial_coverage_gaps(report, threshold=80.0)

        assert len(gaps) == 0


class TestIdentifyUntestedBranches:
    """Tests for identify_untested_branches (AC: 1)."""

    def test_identify_untested_branches_with_uncovered_ranges(self) -> None:
        """identify_untested_branches finds files with uncovered line ranges."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.gap_analysis import identify_untested_branches

        report = CoverageReport(
            results=(
                CoverageResult(
                    file_path="src/branches.py",
                    lines_total=100,
                    lines_covered=85,
                    coverage_percentage=85.0,
                    uncovered_lines=((20, 25), (40, 45)),  # Branch gaps
                ),
            ),
            overall_coverage=85.0,
            threshold=80.0,
            passed=True,
        )

        gaps = identify_untested_branches(report)

        assert len(gaps) == 1
        assert gaps[0].file_path == "src/branches.py"
        assert gaps[0].gap_type == "untested_branch"
        assert len(gaps[0].uncovered_lines) == 2


class TestIdentifyGaps:
    """Tests for combined identify_gaps function (AC: 1)."""

    def test_identify_gaps_combines_all_types(self) -> None:
        """identify_gaps returns all gap types combined."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.gap_analysis import identify_gaps

        report = CoverageReport(
            results=(
                CoverageResult(
                    file_path="src/untested.py",
                    lines_total=50,
                    lines_covered=0,
                    coverage_percentage=0.0,
                    uncovered_lines=((1, 50),),
                ),
                CoverageResult(
                    file_path="src/partial.py",
                    lines_total=100,
                    lines_covered=60,
                    coverage_percentage=60.0,
                    uncovered_lines=((30, 50),),
                ),
            ),
            overall_coverage=30.0,
            threshold=80.0,
            passed=False,
        )

        gaps = identify_gaps(report)

        assert len(gaps) >= 2
        assert isinstance(gaps, tuple)

    def test_identify_gaps_empty_report(self) -> None:
        """identify_gaps returns empty tuple for empty report."""
        from yolo_developer.agents.tea.coverage import CoverageReport
        from yolo_developer.agents.tea.gap_analysis import identify_gaps

        report = CoverageReport(results=(), overall_coverage=100.0, threshold=80.0, passed=True)

        gaps = identify_gaps(report)

        assert gaps == ()

    def test_identify_gaps_returns_tuple(self) -> None:
        """identify_gaps returns tuple (immutable)."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.gap_analysis import identify_gaps

        report = CoverageReport(
            results=(
                CoverageResult(
                    file_path="src/test.py",
                    lines_total=10,
                    lines_covered=0,
                    coverage_percentage=0.0,
                    uncovered_lines=((1, 10),),
                ),
            ),
            overall_coverage=0.0,
            threshold=80.0,
            passed=False,
        )

        gaps = identify_gaps(report)

        assert isinstance(gaps, tuple)


# =============================================================================
# Task 3: Gap Prioritization Tests (AC: 2)
# =============================================================================


class TestCalculateRiskScore:
    """Tests for _calculate_risk_score (AC: 2)."""

    def test_calculate_risk_score_critical_path(self) -> None:
        """_calculate_risk_score returns high score for critical paths."""
        from yolo_developer.agents.tea.gap_analysis import TestGap, _calculate_risk_score

        gap = TestGap(
            gap_id="GAP-001",
            file_path="src/orchestrator/workflow.py",
            function_names=(),
            uncovered_lines=((1, 10),),
            gap_type="no_tests",
            description="No tests",
        )

        score = _calculate_risk_score(gap, critical_paths=("orchestrator/",))

        assert score >= 80  # Critical paths should get high scores

    def test_calculate_risk_score_non_critical_path(self) -> None:
        """_calculate_risk_score returns lower score for non-critical paths."""
        from yolo_developer.agents.tea.gap_analysis import TestGap, _calculate_risk_score

        gap = TestGap(
            gap_id="GAP-001",
            file_path="src/utils/helpers.py",
            function_names=(),
            uncovered_lines=((1, 5),),
            gap_type="partial_coverage",
            description="Partial coverage",
        )

        score = _calculate_risk_score(gap, critical_paths=("orchestrator/", "agents/"))

        assert score < 80


class TestDetermineSeverity:
    """Tests for _determine_severity (AC: 2)."""

    def test_determine_severity_critical_path(self) -> None:
        """_determine_severity returns critical for critical paths."""
        from yolo_developer.agents.tea.gap_analysis import TestGap, _determine_severity

        gap = TestGap(
            gap_id="GAP-001",
            file_path="src/agents/analyst.py",
            function_names=(),
            uncovered_lines=(),
            gap_type="no_tests",
            description="",
        )

        severity = _determine_severity(gap, risk_score=90, critical_paths=("agents/",))

        assert severity == "critical"

    def test_determine_severity_high_risk_score(self) -> None:
        """_determine_severity returns high for high risk scores."""
        from yolo_developer.agents.tea.gap_analysis import TestGap, _determine_severity

        gap = TestGap(
            gap_id="GAP-001",
            file_path="src/utils/helpers.py",
            function_names=(),
            uncovered_lines=(),
            gap_type="no_tests",
            description="",
        )

        severity = _determine_severity(gap, risk_score=75, critical_paths=())

        assert severity == "high"

    def test_determine_severity_low_risk_score(self) -> None:
        """_determine_severity returns low for low risk scores."""
        from yolo_developer.agents.tea.gap_analysis import TestGap, _determine_severity

        gap = TestGap(
            gap_id="GAP-001",
            file_path="src/utils/helpers.py",
            function_names=(),
            uncovered_lines=(),
            gap_type="partial_coverage",
            description="",
        )

        severity = _determine_severity(gap, risk_score=25, critical_paths=())

        assert severity == "low"


class TestPrioritizeGaps:
    """Tests for prioritize_gaps (AC: 2)."""

    def test_prioritize_gaps_assigns_ranks(self) -> None:
        """prioritize_gaps assigns sequential priority ranks."""
        from yolo_developer.agents.tea.gap_analysis import TestGap, prioritize_gaps

        gaps = (
            TestGap(gap_id="GAP-001", file_path="src/a.py", function_names=(), uncovered_lines=(), gap_type="no_tests", description=""),
            TestGap(gap_id="GAP-002", file_path="src/b.py", function_names=(), uncovered_lines=(), gap_type="no_tests", description=""),
        )

        priorities = prioritize_gaps(gaps, critical_paths=())

        assert len(priorities) == 2
        ranks = [p.priority_rank for p in priorities]
        assert 1 in ranks
        assert 2 in ranks

    def test_prioritize_gaps_critical_paths_first(self) -> None:
        """prioritize_gaps puts critical paths at higher priority."""
        from yolo_developer.agents.tea.gap_analysis import TestGap, prioritize_gaps

        gaps = (
            TestGap(gap_id="GAP-001", file_path="src/utils.py", function_names=(), uncovered_lines=(), gap_type="no_tests", description=""),
            TestGap(gap_id="GAP-002", file_path="src/agents/core.py", function_names=(), uncovered_lines=(), gap_type="no_tests", description=""),
        )

        priorities = prioritize_gaps(gaps, critical_paths=("agents/",))

        # Find priority for agents/core.py
        agent_priority = next(p for p in priorities if p.gap_id == "GAP-002")
        assert agent_priority.priority_rank == 1  # Should be first

    def test_prioritize_gaps_returns_tuple(self) -> None:
        """prioritize_gaps returns immutable tuple."""
        from yolo_developer.agents.tea.gap_analysis import TestGap, prioritize_gaps

        gaps = (
            TestGap(gap_id="GAP-001", file_path="src/a.py", function_names=(), uncovered_lines=(), gap_type="no_tests", description=""),
        )

        priorities = prioritize_gaps(gaps)

        assert isinstance(priorities, tuple)

    def test_prioritize_gaps_empty_tuple(self) -> None:
        """prioritize_gaps handles empty tuple input."""
        from yolo_developer.agents.tea.gap_analysis import prioritize_gaps

        priorities = prioritize_gaps(())

        assert priorities == ()
        assert isinstance(priorities, tuple)


# =============================================================================
# Task 4: Test Suggestion Generation Tests (AC: 3)
# =============================================================================


class TestGenerateSuggestionId:
    """Tests for _generate_suggestion_id (AC: 3)."""

    def test_generate_suggestion_id_format(self) -> None:
        """_generate_suggestion_id produces correct format."""
        from yolo_developer.agents.tea.gap_analysis import _generate_suggestion_id

        suggestion_id = _generate_suggestion_id(1)

        assert suggestion_id == "SUG-001"

    def test_generate_suggestion_id_sequence(self) -> None:
        """_generate_suggestion_id increments correctly."""
        from yolo_developer.agents.tea.gap_analysis import _generate_suggestion_id

        assert _generate_suggestion_id(1) == "SUG-001"
        assert _generate_suggestion_id(10) == "SUG-010"
        assert _generate_suggestion_id(100) == "SUG-100"


class TestDetermineTestType:
    """Tests for _determine_test_type (AC: 3)."""

    def test_determine_test_type_unit_default(self) -> None:
        """_determine_test_type returns unit for most gaps."""
        from yolo_developer.agents.tea.gap_analysis import TestGap, _determine_test_type

        gap = TestGap(
            gap_id="GAP-001",
            file_path="src/utils/helpers.py",
            function_names=("helper_func",),
            uncovered_lines=(),
            gap_type="no_tests",
            description="",
        )

        test_type = _determine_test_type(gap)

        assert test_type == "unit"


class TestGenerateTestSignature:
    """Tests for _generate_test_signature (AC: 3)."""

    def test_generate_test_signature_from_file(self) -> None:
        """_generate_test_signature generates from file path."""
        from yolo_developer.agents.tea.gap_analysis import TestGap, _generate_test_signature

        gap = TestGap(
            gap_id="GAP-001",
            file_path="src/utils/helpers.py",
            function_names=(),
            uncovered_lines=(),
            gap_type="no_tests",
            description="",
        )

        signature = _generate_test_signature(gap)

        assert signature.startswith("test_")
        assert "helpers" in signature.lower()


class TestEstimateImpact:
    """Tests for _estimate_impact (AC: 3)."""

    def test_estimate_impact_returns_float(self) -> None:
        """_estimate_impact returns coverage improvement estimate."""
        from yolo_developer.agents.tea.gap_analysis import TestGap, _estimate_impact

        gap = TestGap(
            gap_id="GAP-001",
            file_path="src/module.py",
            function_names=(),
            uncovered_lines=((1, 10),),
            gap_type="no_tests",
            description="",
        )

        impact = _estimate_impact(gap, coverage_baseline=70.0)

        assert isinstance(impact, float)
        assert impact > 0


class TestGenerateTestSuggestions:
    """Tests for generate_test_suggestions (AC: 3)."""

    def test_generate_test_suggestions_creates_suggestions(self) -> None:
        """generate_test_suggestions creates suggestions for gaps."""
        from yolo_developer.agents.tea.gap_analysis import (
            GapPriority,
            TestGap,
            generate_test_suggestions,
        )

        gaps = (
            TestGap(gap_id="GAP-001", file_path="src/a.py", function_names=(), uncovered_lines=((1, 10),), gap_type="no_tests", description=""),
        )
        priorities = (
            GapPriority(gap_id="GAP-001", severity="high", risk_score=80, priority_rank=1),
        )

        suggestions = generate_test_suggestions(gaps, priorities, coverage_baseline=70.0)

        assert len(suggestions) >= 1
        assert suggestions[0].target_gap_id == "GAP-001"

    def test_generate_test_suggestions_respects_max_limit(self) -> None:
        """generate_test_suggestions respects max_suggestions limit."""
        from yolo_developer.agents.tea.gap_analysis import (
            GapPriority,
            TestGap,
            generate_test_suggestions,
        )

        # Create 5 gaps
        gaps = tuple(
            TestGap(gap_id=f"GAP-{i:03d}", file_path=f"src/{i}.py", function_names=(), uncovered_lines=((1, 10),), gap_type="no_tests", description="")
            for i in range(1, 6)
        )
        priorities = tuple(
            GapPriority(gap_id=f"GAP-{i:03d}", severity="medium", risk_score=50, priority_rank=i)
            for i in range(1, 6)
        )

        suggestions = generate_test_suggestions(gaps, priorities, max_suggestions=3)

        assert len(suggestions) == 3

    def test_generate_test_suggestions_returns_tuple(self) -> None:
        """generate_test_suggestions returns immutable tuple."""
        from yolo_developer.agents.tea.gap_analysis import (
            GapPriority,
            TestGap,
            generate_test_suggestions,
        )

        gaps = (
            TestGap(gap_id="GAP-001", file_path="src/a.py", function_names=(), uncovered_lines=(), gap_type="no_tests", description=""),
        )
        priorities = (
            GapPriority(gap_id="GAP-001", severity="low", risk_score=30, priority_rank=1),
        )

        suggestions = generate_test_suggestions(gaps, priorities)

        assert isinstance(suggestions, tuple)


# =============================================================================
# Task 5: Summary Generation Tests (AC: 5)
# =============================================================================


class TestCountGapsBySeverity:
    """Tests for _count_gaps_by_severity (AC: 5)."""

    def test_count_gaps_by_severity(self) -> None:
        """_count_gaps_by_severity returns correct counts."""
        from yolo_developer.agents.tea.gap_analysis import GapPriority, _count_gaps_by_severity

        priorities = (
            GapPriority(gap_id="GAP-001", severity="critical", risk_score=95, priority_rank=1),
            GapPriority(gap_id="GAP-002", severity="critical", risk_score=90, priority_rank=2),
            GapPriority(gap_id="GAP-003", severity="high", risk_score=75, priority_rank=3),
            GapPriority(gap_id="GAP-004", severity="medium", risk_score=50, priority_rank=4),
            GapPriority(gap_id="GAP-005", severity="low", risk_score=20, priority_rank=5),
        )

        counts = _count_gaps_by_severity(priorities)

        assert counts["critical"] == 2
        assert counts["high"] == 1
        assert counts["medium"] == 1
        assert counts["low"] == 1


class TestEstimateTotalEffort:
    """Tests for _estimate_total_effort (AC: 5)."""

    def test_estimate_total_effort_small(self) -> None:
        """_estimate_total_effort returns small estimate for few suggestions."""
        from yolo_developer.agents.tea.gap_analysis import (
            TestSuggestion,
            _estimate_total_effort,
        )

        suggestions = (
            TestSuggestion(suggestion_id="SUG-001", target_gap_id="GAP-001", test_type="unit", description="", estimated_impact=5.0, example_signature="test_a"),
        )

        effort = _estimate_total_effort(suggestions)

        assert "hour" in effort.lower() or "minute" in effort.lower()

    def test_estimate_total_effort_returns_string(self) -> None:
        """_estimate_total_effort returns human-readable string."""
        from yolo_developer.agents.tea.gap_analysis import _estimate_total_effort

        effort = _estimate_total_effort(())

        assert isinstance(effort, str)


class TestGenerateSummary:
    """Tests for generate_summary (AC: 5)."""

    def test_generate_summary_creates_summary(self) -> None:
        """generate_summary creates complete summary."""
        from yolo_developer.agents.tea.gap_analysis import (
            GapPriority,
            TestGap,
            TestSuggestion,
            generate_summary,
        )

        gaps = (
            TestGap(gap_id="GAP-001", file_path="src/a.py", function_names=(), uncovered_lines=(), gap_type="no_tests", description=""),
            TestGap(gap_id="GAP-002", file_path="src/b.py", function_names=(), uncovered_lines=(), gap_type="no_tests", description=""),
        )
        priorities = (
            GapPriority(gap_id="GAP-001", severity="high", risk_score=80, priority_rank=1),
            GapPriority(gap_id="GAP-002", severity="low", risk_score=30, priority_rank=2),
        )
        suggestions = (
            TestSuggestion(suggestion_id="SUG-001", target_gap_id="GAP-001", test_type="unit", description="", estimated_impact=5.0, example_signature="test_a"),
        )

        summary = generate_summary(gaps, priorities, suggestions)

        assert summary.total_gaps == 2
        assert summary.high_gaps == 1
        assert summary.low_gaps == 1
        assert summary.total_suggestions == 1


# =============================================================================
# Task 6: Report Generation Tests (AC: 4)
# =============================================================================


class TestGenerateGapAnalysisReport:
    """Tests for generate_gap_analysis_report (AC: 4)."""

    def test_generate_gap_analysis_report_full_pipeline(self) -> None:
        """generate_gap_analysis_report runs full analysis pipeline."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.gap_analysis import generate_gap_analysis_report

        coverage_report = CoverageReport(
            results=(
                CoverageResult(
                    file_path="src/untested.py",
                    lines_total=50,
                    lines_covered=0,
                    coverage_percentage=0.0,
                    uncovered_lines=((1, 50),),
                ),
                CoverageResult(
                    file_path="src/partial.py",
                    lines_total=100,
                    lines_covered=60,
                    coverage_percentage=60.0,
                    uncovered_lines=((30, 50),),
                ),
            ),
            overall_coverage=40.0,
            threshold=80.0,
            passed=False,
        )

        report = generate_gap_analysis_report(coverage_report)

        assert len(report.gaps) >= 2
        assert len(report.priorities) == len(report.gaps)
        assert report.coverage_baseline == 40.0
        assert report.summary.total_gaps >= 2

    def test_generate_gap_analysis_report_empty_coverage(self) -> None:
        """generate_gap_analysis_report handles empty coverage."""
        from yolo_developer.agents.tea.coverage import CoverageReport
        from yolo_developer.agents.tea.gap_analysis import generate_gap_analysis_report

        coverage_report = CoverageReport(
            results=(),
            overall_coverage=100.0,
            threshold=80.0,
            passed=True,
        )

        report = generate_gap_analysis_report(coverage_report)

        assert len(report.gaps) == 0
        assert report.summary.total_gaps == 0

    def test_generate_gap_analysis_report_with_critical_paths(self) -> None:
        """generate_gap_analysis_report respects critical paths."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.gap_analysis import generate_gap_analysis_report

        coverage_report = CoverageReport(
            results=(
                CoverageResult(
                    file_path="src/agents/tea.py",
                    lines_total=100,
                    lines_covered=0,
                    coverage_percentage=0.0,
                    uncovered_lines=((1, 100),),
                ),
            ),
            overall_coverage=0.0,
            threshold=80.0,
            passed=False,
        )

        report = generate_gap_analysis_report(
            coverage_report,
            critical_paths=("agents/",),
        )

        assert len(report.priorities) == 1
        assert report.priorities[0].severity == "critical"

    def test_generate_gap_analysis_report_calculates_projected_coverage(self) -> None:
        """generate_gap_analysis_report calculates projected coverage."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.gap_analysis import generate_gap_analysis_report

        coverage_report = CoverageReport(
            results=(
                CoverageResult(
                    file_path="src/test.py",
                    lines_total=100,
                    lines_covered=50,
                    coverage_percentage=50.0,
                    uncovered_lines=((51, 100),),
                ),
            ),
            overall_coverage=50.0,
            threshold=80.0,
            passed=False,
        )

        report = generate_gap_analysis_report(coverage_report)

        assert report.coverage_baseline == 50.0
        assert report.projected_coverage >= report.coverage_baseline


# =============================================================================
# Task 7: Export Functions Tests (AC: 6)
# =============================================================================


class TestExportToJson:
    """Tests for export_to_json (AC: 6)."""

    def test_export_to_json_returns_valid_json(self) -> None:
        """export_to_json returns valid JSON string."""
        import json

        from yolo_developer.agents.tea.gap_analysis import (
            GapAnalysisReport,
            GapAnalysisSummary,
            export_to_json,
        )

        summary = GapAnalysisSummary(
            total_gaps=0, critical_gaps=0, high_gaps=0, medium_gaps=0, low_gaps=0, total_suggestions=0, estimated_effort="None"
        )
        report = GapAnalysisReport(gaps=(), priorities=(), suggestions=(), summary=summary, coverage_baseline=100.0, projected_coverage=100.0)

        json_str = export_to_json(report)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "gaps" in parsed
        assert "summary" in parsed

    def test_export_to_json_includes_all_data(self) -> None:
        """export_to_json includes all report data."""
        import json

        from yolo_developer.agents.tea.gap_analysis import (
            GapAnalysisReport,
            GapAnalysisSummary,
            GapPriority,
            TestGap,
            TestSuggestion,
            export_to_json,
        )

        gap = TestGap(gap_id="GAP-001", file_path="src/a.py", function_names=(), uncovered_lines=(), gap_type="no_tests", description="")
        priority = GapPriority(gap_id="GAP-001", severity="high", risk_score=80, priority_rank=1)
        suggestion = TestSuggestion(suggestion_id="SUG-001", target_gap_id="GAP-001", test_type="unit", description="Test", estimated_impact=5.0, example_signature="test_a")
        summary = GapAnalysisSummary(
            total_gaps=1, critical_gaps=0, high_gaps=1, medium_gaps=0, low_gaps=0, total_suggestions=1, estimated_effort="1 hour"
        )
        report = GapAnalysisReport(gaps=(gap,), priorities=(priority,), suggestions=(suggestion,), summary=summary, coverage_baseline=70.0, projected_coverage=75.0)

        json_str = export_to_json(report)
        parsed = json.loads(json_str)

        assert len(parsed["gaps"]) == 1
        assert len(parsed["priorities"]) == 1
        assert len(parsed["suggestions"]) == 1
        assert parsed["coverage_baseline"] == 70.0


class TestExportToMarkdown:
    """Tests for export_to_markdown (AC: 6)."""

    def test_export_to_markdown_returns_string(self) -> None:
        """export_to_markdown returns markdown string."""
        from yolo_developer.agents.tea.gap_analysis import (
            GapAnalysisReport,
            GapAnalysisSummary,
            export_to_markdown,
        )

        summary = GapAnalysisSummary(
            total_gaps=0, critical_gaps=0, high_gaps=0, medium_gaps=0, low_gaps=0, total_suggestions=0, estimated_effort="None"
        )
        report = GapAnalysisReport(gaps=(), priorities=(), suggestions=(), summary=summary, coverage_baseline=100.0, projected_coverage=100.0)

        markdown = export_to_markdown(report)

        assert isinstance(markdown, str)
        assert "# Gap Analysis Report" in markdown

    def test_export_to_markdown_includes_sections(self) -> None:
        """export_to_markdown includes all sections."""
        from yolo_developer.agents.tea.gap_analysis import (
            GapAnalysisReport,
            GapAnalysisSummary,
            GapPriority,
            TestGap,
            TestSuggestion,
            export_to_markdown,
        )

        gap = TestGap(gap_id="GAP-001", file_path="src/a.py", function_names=(), uncovered_lines=(), gap_type="no_tests", description="No tests")
        priority = GapPriority(gap_id="GAP-001", severity="high", risk_score=80, priority_rank=1)
        suggestion = TestSuggestion(suggestion_id="SUG-001", target_gap_id="GAP-001", test_type="unit", description="Add test", estimated_impact=5.0, example_signature="test_a")
        summary = GapAnalysisSummary(
            total_gaps=1, critical_gaps=0, high_gaps=1, medium_gaps=0, low_gaps=0, total_suggestions=1, estimated_effort="1 hour"
        )
        report = GapAnalysisReport(gaps=(gap,), priorities=(priority,), suggestions=(suggestion,), summary=summary, coverage_baseline=70.0, projected_coverage=75.0)

        markdown = export_to_markdown(report)

        assert "## Summary" in markdown
        assert "## Gaps" in markdown
        assert "## Suggestions" in markdown


class TestExportToCsv:
    """Tests for export_to_csv (AC: 6)."""

    def test_export_to_csv_returns_string(self) -> None:
        """export_to_csv returns CSV string."""
        from yolo_developer.agents.tea.gap_analysis import (
            GapAnalysisReport,
            GapAnalysisSummary,
            export_to_csv,
        )

        summary = GapAnalysisSummary(
            total_gaps=0, critical_gaps=0, high_gaps=0, medium_gaps=0, low_gaps=0, total_suggestions=0, estimated_effort="None"
        )
        report = GapAnalysisReport(gaps=(), priorities=(), suggestions=(), summary=summary, coverage_baseline=100.0, projected_coverage=100.0)

        csv_str = export_to_csv(report)

        assert isinstance(csv_str, str)
        # Should have header row
        assert "gap_id" in csv_str

    def test_export_to_csv_includes_gaps_and_priorities(self) -> None:
        """export_to_csv includes gap and priority data."""
        from yolo_developer.agents.tea.gap_analysis import (
            GapAnalysisReport,
            GapAnalysisSummary,
            GapPriority,
            TestGap,
            export_to_csv,
        )

        gap = TestGap(gap_id="GAP-001", file_path="src/a.py", function_names=(), uncovered_lines=(), gap_type="no_tests", description="No tests")
        priority = GapPriority(gap_id="GAP-001", severity="high", risk_score=80, priority_rank=1)
        summary = GapAnalysisSummary(
            total_gaps=1, critical_gaps=0, high_gaps=1, medium_gaps=0, low_gaps=0, total_suggestions=0, estimated_effort="1 hour"
        )
        report = GapAnalysisReport(gaps=(gap,), priorities=(priority,), suggestions=(), summary=summary, coverage_baseline=70.0, projected_coverage=75.0)

        csv_str = export_to_csv(report)

        assert "GAP-001" in csv_str
        assert "src/a.py" in csv_str
        assert "high" in csv_str


# =============================================================================
# Task 8: TEA Integration Tests (AC: 7)
# =============================================================================


class TestTEAOutputIntegration:
    """Tests for TEAOutput gap_analysis_report field (AC: 7)."""

    def test_tea_output_has_gap_analysis_report_field(self) -> None:
        """TEAOutput has gap_analysis_report field."""
        from yolo_developer.agents.tea.types import TEAOutput

        output = TEAOutput()

        assert hasattr(output, "gap_analysis_report")
        assert output.gap_analysis_report is None

    def test_tea_output_to_dict_includes_gap_analysis(self) -> None:
        """TEAOutput.to_dict() includes gap_analysis_report."""
        from yolo_developer.agents.tea.gap_analysis import (
            GapAnalysisReport,
            GapAnalysisSummary,
        )
        from yolo_developer.agents.tea.types import TEAOutput

        summary = GapAnalysisSummary(
            total_gaps=1, critical_gaps=0, high_gaps=1, medium_gaps=0, low_gaps=0, total_suggestions=1, estimated_effort="1 hour"
        )
        gap_report = GapAnalysisReport(gaps=(), priorities=(), suggestions=(), summary=summary, coverage_baseline=80.0, projected_coverage=85.0)

        output = TEAOutput(gap_analysis_report=gap_report)
        result = output.to_dict()

        assert "gap_analysis_report" in result
        assert result["gap_analysis_report"]["coverage_baseline"] == 80.0


class TestGapAnalysisExports:
    """Tests for gap analysis exports from __init__.py."""

    def test_gap_analysis_types_exported(self) -> None:
        """Gap analysis types are exported from tea module."""
        from yolo_developer.agents.tea import (
            GapAnalysisReport,
            GapAnalysisSummary,
            GapPriority,
            GapSeverity,
            GapType,
            SuggestionTestType,
            TestGap,
            TestSuggestion,
        )

        # Just verify imports work
        assert GapType is not None
        assert GapSeverity is not None
        assert SuggestionTestType is not None
        assert TestGap is not None
        assert GapPriority is not None
        assert TestSuggestion is not None
        assert GapAnalysisSummary is not None
        assert GapAnalysisReport is not None

    def test_gap_analysis_functions_exported(self) -> None:
        """Gap analysis functions are exported from tea module."""
        from yolo_developer.agents.tea import (
            export_to_csv,
            export_to_json,
            export_to_markdown,
            generate_gap_analysis_report,
            generate_summary,
            generate_test_suggestions,
            identify_gaps,
            prioritize_gaps,
        )

        # Just verify imports work
        assert identify_gaps is not None
        assert prioritize_gaps is not None
        assert generate_test_suggestions is not None
        assert generate_summary is not None
        assert generate_gap_analysis_report is not None
        assert export_to_json is not None
        assert export_to_markdown is not None
        assert export_to_csv is not None
