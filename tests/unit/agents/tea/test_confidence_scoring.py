"""Unit tests for confidence scoring calculations (Story 9.4 - Tasks 3-8, 11, 12).

Tests for the confidence scoring calculation functions:
- _calculate_coverage_score: Coverage contribution calculation
- _calculate_test_execution_score: Test execution contribution calculation
- _calculate_validation_score: Validation findings contribution calculation
- _apply_score_modifiers: Score bonus/penalty application
- calculate_confidence_score: Main confidence calculation
- check_deployment_threshold: Threshold validation

All tests follow the patterns established in Story 9.1-9.3.
"""

from __future__ import annotations


class TestCalculateCoverageScore:
    """Tests for _calculate_coverage_score function (Task 3)."""

    def test_none_coverage_returns_neutral_score(self) -> None:
        """Test that None coverage report returns neutral score (50)."""
        from yolo_developer.agents.tea.scoring import _calculate_coverage_score

        score = _calculate_coverage_score(None)

        assert score == 50.0

    def test_full_coverage_returns_100(self) -> None:
        """Test that 100% coverage returns 100 score."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.scoring import _calculate_coverage_score

        result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=100,
            coverage_percentage=100.0,
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=100.0,
            threshold=80.0,
            passed=True,
        )

        score = _calculate_coverage_score(report)

        assert score == 100.0

    def test_zero_coverage_returns_0(self) -> None:
        """Test that 0% coverage returns 0 score."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.scoring import _calculate_coverage_score

        result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=0,
            coverage_percentage=0.0,
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=0.0,
            threshold=80.0,
            passed=False,
        )

        score = _calculate_coverage_score(report)

        assert score == 0.0

    def test_partial_coverage_returns_percentage(self) -> None:
        """Test that partial coverage returns corresponding percentage."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.scoring import _calculate_coverage_score

        result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=75,
            coverage_percentage=75.0,
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=75.0,
            threshold=80.0,
            passed=False,
        )

        score = _calculate_coverage_score(report)

        assert score == 75.0

    def test_critical_path_coverage_boost(self) -> None:
        """Test that all critical paths covered adds +10 bonus."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.scoring import _calculate_coverage_score

        result = CoverageResult(
            file_path="src/agents/tea/node.py",
            lines_total=100,
            lines_covered=90,
            coverage_percentage=90.0,
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=90.0,
            threshold=80.0,
            passed=True,
            critical_files_coverage={"src/agents/tea/node.py": 100.0},
        )

        score = _calculate_coverage_score(report)

        # Should be 90 + 10 bonus for 100% critical path coverage = 100 (capped)
        assert score == 100.0


class TestCalculateTestExecutionScore:
    """Tests for _calculate_test_execution_score function (Task 4)."""

    def test_none_result_returns_neutral_score(self) -> None:
        """Test that None test result returns neutral score (50)."""
        from yolo_developer.agents.tea.scoring import _calculate_test_execution_score

        score = _calculate_test_execution_score(None)

        assert score == 50.0

    def test_all_tests_pass_returns_100(self) -> None:
        """Test that 100% pass rate returns 100 (bonus applied separately in modifiers)."""
        from yolo_developer.agents.tea.execution import TestExecutionResult
        from yolo_developer.agents.tea.scoring import _calculate_test_execution_score

        result = TestExecutionResult(
            status="passed",
            passed_count=10,
            failed_count=0,
            error_count=0,
        )

        score = _calculate_test_execution_score(result)

        # 100% pass rate = 100 (bonus is applied in _apply_score_modifiers)
        assert score == 100.0

    def test_all_tests_fail_returns_0(self) -> None:
        """Test that 0% pass rate returns 0."""
        from yolo_developer.agents.tea.execution import TestExecutionResult, TestFailure
        from yolo_developer.agents.tea.scoring import _calculate_test_execution_score

        failure = TestFailure(
            test_name="test_example",
            file_path="test.py",
            error_message="Assertion failed",
            failure_type="failure",
        )
        result = TestExecutionResult(
            status="failed",
            passed_count=0,
            failed_count=5,
            error_count=0,
            failures=(failure,) * 5,
        )

        score = _calculate_test_execution_score(result)

        assert score == 0.0

    def test_mixed_pass_fail_returns_proportional_score(self) -> None:
        """Test that mixed pass/fail returns proportional score."""
        from yolo_developer.agents.tea.execution import TestExecutionResult
        from yolo_developer.agents.tea.scoring import _calculate_test_execution_score

        result = TestExecutionResult(
            status="failed",
            passed_count=8,
            failed_count=2,
            error_count=0,
        )

        score = _calculate_test_execution_score(result)

        # 8/10 = 80%
        assert score == 80.0

    def test_errors_apply_penalty(self) -> None:
        """Test that errors apply -15 penalty per error."""
        from yolo_developer.agents.tea.execution import TestExecutionResult
        from yolo_developer.agents.tea.scoring import _calculate_test_execution_score

        result = TestExecutionResult(
            status="error",
            passed_count=10,
            failed_count=0,
            error_count=2,
        )

        score = _calculate_test_execution_score(result)

        # 10/12 = 83.33% - 30 (2 * 15) = 53.33
        expected = (10 / 12) * 100.0 - (2 * 15)
        assert abs(score - expected) < 0.1

    def test_error_penalty_capped_at_45(self) -> None:
        """Test that error penalty is capped at -45."""
        from yolo_developer.agents.tea.execution import TestExecutionResult
        from yolo_developer.agents.tea.scoring import _calculate_test_execution_score

        result = TestExecutionResult(
            status="error",
            passed_count=10,
            failed_count=0,
            error_count=10,  # 10 * 15 = 150, but capped at 45
        )

        score = _calculate_test_execution_score(result)

        # Base score - 45 (capped penalty)
        # 10/20 = 50% - 45 = 5
        expected = (10 / 20) * 100.0 - 45
        assert abs(score - expected) < 0.1

    def test_no_tests_returns_neutral(self) -> None:
        """Test that empty test result returns neutral score."""
        from yolo_developer.agents.tea.execution import TestExecutionResult
        from yolo_developer.agents.tea.scoring import _calculate_test_execution_score

        result = TestExecutionResult(
            status="passed",
            passed_count=0,
            failed_count=0,
            error_count=0,
        )

        score = _calculate_test_execution_score(result)

        # No tests = neutral score
        assert score == 50.0


class TestCalculateValidationScore:
    """Tests for _calculate_validation_score function (Task 5)."""

    def test_no_findings_returns_100(self) -> None:
        """Test that no findings returns 100 score."""
        from yolo_developer.agents.tea.scoring import _calculate_validation_score
        from yolo_developer.agents.tea.types import ValidationResult

        result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="passed",
            findings=(),
        )

        score, penalties = _calculate_validation_score((result,))

        assert score == 100.0
        assert penalties == []

    def test_critical_finding_penalty(self) -> None:
        """Test that critical findings reduce score by 25 each."""
        from yolo_developer.agents.tea.scoring import _calculate_validation_score
        from yolo_developer.agents.tea.types import Finding, ValidationResult

        finding = Finding(
            finding_id="F001",
            category="test_coverage",
            severity="critical",
            description="Critical issue",
            location="src/module.py",
            remediation="Fix it",
        )
        result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="failed",
            findings=(finding,),
        )

        score, penalties = _calculate_validation_score((result,))

        assert score == 75.0  # 100 - 25
        assert len(penalties) == 1
        assert "-25" in penalties[0]
        assert "critical" in penalties[0]

    def test_critical_findings_capped_at_75(self) -> None:
        """Test that critical findings penalty is capped at -75."""
        from yolo_developer.agents.tea.scoring import _calculate_validation_score
        from yolo_developer.agents.tea.types import Finding, ValidationResult

        findings = tuple(
            Finding(
                finding_id=f"F{i:03d}",
                category="test_coverage",
                severity="critical",
                description=f"Critical issue {i}",
                location="src/module.py",
                remediation="Fix it",
            )
            for i in range(5)  # 5 * 25 = 125, but capped at 75
        )
        result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="failed",
            findings=findings,
        )

        score, penalties = _calculate_validation_score((result,))

        assert score == 25.0  # 100 - 75 (capped)
        assert any("capped" in p for p in penalties)

    def test_high_finding_penalty(self) -> None:
        """Test that high findings reduce score by 10 each."""
        from yolo_developer.agents.tea.scoring import _calculate_validation_score
        from yolo_developer.agents.tea.types import Finding, ValidationResult

        finding = Finding(
            finding_id="F001",
            category="code_quality",
            severity="high",
            description="High issue",
            location="src/module.py",
            remediation="Fix it",
        )
        result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="warning",
            findings=(finding,),
        )

        score, penalties = _calculate_validation_score((result,))

        assert score == 90.0  # 100 - 10
        assert len(penalties) == 1
        assert "-10" in penalties[0]
        assert "high" in penalties[0]

    def test_high_findings_capped_at_40(self) -> None:
        """Test that high findings penalty is capped at -40."""
        from yolo_developer.agents.tea.scoring import _calculate_validation_score
        from yolo_developer.agents.tea.types import Finding, ValidationResult

        findings = tuple(
            Finding(
                finding_id=f"F{i:03d}",
                category="code_quality",
                severity="high",
                description=f"High issue {i}",
                location="src/module.py",
                remediation="Fix it",
            )
            for i in range(10)  # 10 * 10 = 100, but capped at 40
        )
        result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="failed",
            findings=findings,
        )

        score, penalties = _calculate_validation_score((result,))

        assert score == 60.0  # 100 - 40 (capped)
        assert any("capped" in p for p in penalties)

    def test_medium_finding_penalty(self) -> None:
        """Test that medium findings reduce score by 5 each."""
        from yolo_developer.agents.tea.scoring import _calculate_validation_score
        from yolo_developer.agents.tea.types import Finding, ValidationResult

        finding = Finding(
            finding_id="F001",
            category="documentation",
            severity="medium",
            description="Medium issue",
            location="src/module.py",
            remediation="Fix it",
        )
        result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="warning",
            findings=(finding,),
        )

        score, penalties = _calculate_validation_score((result,))

        assert score == 95.0  # 100 - 5
        assert len(penalties) == 1
        assert "-5" in penalties[0]
        assert "medium" in penalties[0]

    def test_low_finding_penalty(self) -> None:
        """Test that low findings reduce score by 2 each."""
        from yolo_developer.agents.tea.scoring import _calculate_validation_score
        from yolo_developer.agents.tea.types import Finding, ValidationResult

        finding = Finding(
            finding_id="F001",
            category="documentation",
            severity="low",
            description="Low issue",
            location="src/module.py",
            remediation="Fix it",
        )
        result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="warning",
            findings=(finding,),
        )

        score, penalties = _calculate_validation_score((result,))

        assert score == 98.0  # 100 - 2
        assert len(penalties) == 1
        assert "-2" in penalties[0]

    def test_info_finding_penalty(self) -> None:
        """Test that info findings reduce score by 1 each."""
        from yolo_developer.agents.tea.scoring import _calculate_validation_score
        from yolo_developer.agents.tea.types import Finding, ValidationResult

        finding = Finding(
            finding_id="F001",
            category="documentation",
            severity="info",
            description="Info item",
            location="src/module.py",
            remediation="Consider it",
        )
        result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="passed",
            findings=(finding,),
        )

        score, penalties = _calculate_validation_score((result,))

        assert score == 99.0  # 100 - 1
        assert len(penalties) == 1
        assert "-1" in penalties[0]

    def test_empty_results_returns_100(self) -> None:
        """Test that empty validation results returns 100."""
        from yolo_developer.agents.tea.scoring import _calculate_validation_score

        score, penalties = _calculate_validation_score(())

        assert score == 100.0
        assert penalties == []

    def test_mixed_severities(self) -> None:
        """Test mixed severity findings apply correct cumulative penalties."""
        from yolo_developer.agents.tea.scoring import _calculate_validation_score
        from yolo_developer.agents.tea.types import Finding, ValidationResult

        findings = (
            Finding(
                finding_id="F001",
                category="test_coverage",
                severity="critical",
                description="Critical",
                location="src/module.py",
                remediation="Fix",
            ),
            Finding(
                finding_id="F002",
                category="code_quality",
                severity="high",
                description="High",
                location="src/module.py",
                remediation="Fix",
            ),
            Finding(
                finding_id="F003",
                category="documentation",
                severity="medium",
                description="Medium",
                location="src/module.py",
                remediation="Fix",
            ),
        )
        result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="failed",
            findings=findings,
        )

        score, penalties = _calculate_validation_score((result,))

        # 100 - 25 (critical) - 10 (high) - 5 (medium) = 60
        assert score == 60.0
        assert len(penalties) == 3  # One for each severity


class TestApplyScoreModifiers:
    """Tests for _apply_score_modifiers function (Task 6)."""

    def test_perfect_test_pass_adds_bonus(self) -> None:
        """Test that perfect test pass rate adds +5 bonus."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown, _apply_score_modifiers

        breakdown = ConfidenceBreakdown(
            coverage_score=100.0,
            test_execution_score=100.0,  # All tests passed - triggers bonus
            validation_score=100.0,
            weighted_coverage=40.0,
            weighted_test_execution=30.0,
            weighted_validation=30.0,
            base_score=100.0,
            final_score=100,
        )

        final_score, reasons = _apply_score_modifiers(100.0, breakdown, perfect_tests=True)

        assert final_score == 100.0  # Capped at 100
        assert any("+5" in reason for reason in reasons)

    def test_score_capped_at_100(self) -> None:
        """Test that score is capped at 100 after bonuses."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown, _apply_score_modifiers

        breakdown = ConfidenceBreakdown(
            coverage_score=100.0,
            test_execution_score=100.0,
            validation_score=100.0,
            weighted_coverage=40.0,
            weighted_test_execution=30.0,
            weighted_validation=30.0,
            base_score=98.0,
            final_score=98,
        )

        final_score, _ = _apply_score_modifiers(98.0, breakdown, perfect_tests=True)

        assert final_score == 100.0  # Capped at 100

    def test_score_capped_at_0(self) -> None:
        """Test that score is capped at 0 after penalties."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown, _apply_score_modifiers

        breakdown = ConfidenceBreakdown(
            coverage_score=0.0,
            test_execution_score=0.0,
            validation_score=0.0,
            weighted_coverage=0.0,
            weighted_test_execution=0.0,
            weighted_validation=0.0,
            base_score=0.0,
            final_score=0,
        )

        final_score, _ = _apply_score_modifiers(-50.0, breakdown, perfect_tests=False)

        assert final_score == 0.0  # Capped at 0

    def test_no_modifiers_returns_base_score(self) -> None:
        """Test that no modifiers returns base score unchanged."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown, _apply_score_modifiers

        breakdown = ConfidenceBreakdown(
            coverage_score=80.0,
            test_execution_score=80.0,
            validation_score=80.0,
            weighted_coverage=32.0,
            weighted_test_execution=24.0,
            weighted_validation=24.0,
            base_score=80.0,
            final_score=80,
        )

        final_score, reasons = _apply_score_modifiers(80.0, breakdown, perfect_tests=False)

        assert final_score == 80.0
        assert len(reasons) == 0


class TestCheckDeploymentThreshold:
    """Tests for check_deployment_threshold function (Task 8, 12)."""

    def test_score_above_threshold_passes(self) -> None:
        """Test that score >= threshold passes."""
        from yolo_developer.agents.tea.scoring import check_deployment_threshold

        passed, recommendation, reasons, finding = check_deployment_threshold(95, threshold=90)

        assert passed is True
        assert recommendation == "deploy"
        assert len(reasons) == 0
        assert finding is None

    def test_score_equals_threshold_passes(self) -> None:
        """Test that score == threshold passes."""
        from yolo_developer.agents.tea.scoring import check_deployment_threshold

        passed, recommendation, reasons, finding = check_deployment_threshold(90, threshold=90)

        assert passed is True
        assert recommendation == "deploy"
        assert len(reasons) == 0
        assert finding is None

    def test_score_below_threshold_blocks(self) -> None:
        """Test that score far below threshold blocks."""
        from yolo_developer.agents.tea.scoring import check_deployment_threshold

        # Score 79 is more than 10 points below threshold 90, so it blocks
        passed, recommendation, reasons, _finding = check_deployment_threshold(79, threshold=90)

        assert passed is False
        assert recommendation == "block"
        assert len(reasons) > 0
        assert "79" in reasons[0]
        assert "90" in reasons[0]

    def test_score_close_to_threshold_warns(self) -> None:
        """Test that score within 10 points of threshold warns."""
        from yolo_developer.agents.tea.scoring import check_deployment_threshold

        passed, recommendation, reasons, finding = check_deployment_threshold(82, threshold=90)

        assert passed is False
        assert recommendation == "deploy_with_warnings"
        assert len(reasons) > 0
        assert finding is None  # No critical finding for warnings

    def test_score_far_below_threshold_blocks(self) -> None:
        """Test that score far below threshold (>10 points) blocks."""
        from yolo_developer.agents.tea.scoring import check_deployment_threshold

        passed, recommendation, reasons, _finding = check_deployment_threshold(70, threshold=90)

        assert passed is False
        assert recommendation == "block"
        assert len(reasons) > 0

    def test_custom_threshold(self) -> None:
        """Test that custom threshold is respected."""
        from yolo_developer.agents.tea.scoring import check_deployment_threshold

        # Would fail with default threshold (90) but passes with custom (70)
        passed, recommendation, _, finding = check_deployment_threshold(75, threshold=70)

        assert passed is True
        assert recommendation == "deploy"
        assert finding is None

    def test_threshold_zero(self) -> None:
        """Test that threshold of 0 works correctly."""
        from yolo_developer.agents.tea.scoring import check_deployment_threshold

        passed, recommendation, _, finding = check_deployment_threshold(0, threshold=0)

        assert passed is True
        assert recommendation == "deploy"
        assert finding is None

    def test_blocking_generates_critical_finding_ac4(self) -> None:
        """Test AC4: blocking generates a Finding with severity='critical'."""
        from yolo_developer.agents.tea.scoring import check_deployment_threshold

        passed, recommendation, _reasons, finding = check_deployment_threshold(70, threshold=90)

        assert passed is False
        assert recommendation == "block"
        assert finding is not None
        assert finding.severity == "critical"
        assert finding.category == "test_coverage"
        assert "70" in finding.description
        assert "90" in finding.description
        assert finding.remediation is not None
        assert len(finding.remediation) > 0

    def test_blocking_finding_has_unique_id(self) -> None:
        """Test that blocking finding has a unique finding_id."""
        from yolo_developer.agents.tea.scoring import check_deployment_threshold

        _, _, _, finding1 = check_deployment_threshold(70, threshold=90)
        _, _, _, finding2 = check_deployment_threshold(65, threshold=90)

        assert finding1 is not None
        assert finding2 is not None
        assert finding1.finding_id != finding2.finding_id
        assert "70" in finding1.finding_id
        assert "65" in finding2.finding_id


class TestCalculateConfidenceScore:
    """Tests for calculate_confidence_score function (Task 7)."""

    def test_full_confidence_score(self) -> None:
        """Test calculating confidence with all components perfect."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.execution import TestExecutionResult
        from yolo_developer.agents.tea.scoring import calculate_confidence_score
        from yolo_developer.agents.tea.types import ValidationResult

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
        validation_result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="passed",
            findings=(),
        )

        result = calculate_confidence_score(
            validation_results=(validation_result,),
            coverage_report=coverage_report,
            test_execution_result=test_result,
        )

        assert result.score == 100
        assert result.passed_threshold is True
        assert result.deployment_recommendation == "deploy"

    def test_none_inputs_returns_neutral(self) -> None:
        """Test that None inputs return neutral scores."""
        from yolo_developer.agents.tea.scoring import calculate_confidence_score

        result = calculate_confidence_score(
            validation_results=(),
            coverage_report=None,
            test_execution_result=None,
        )

        # Coverage: 50 (neutral) * 0.4 = 20
        # Test exec: 50 (neutral) * 0.3 = 15
        # Validation: 100 (no findings) * 0.3 = 30
        # Total: 20 + 15 + 30 = 65
        assert result.score == 65
        assert result.passed_threshold is False

    def test_score_breakdown_correct(self) -> None:
        """Test that score breakdown is correctly populated."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.execution import TestExecutionResult
        from yolo_developer.agents.tea.scoring import calculate_confidence_score
        from yolo_developer.agents.tea.types import ValidationResult

        coverage_result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=80,
            coverage_percentage=80.0,
        )
        coverage_report = CoverageReport(
            results=(coverage_result,),
            overall_coverage=80.0,
            threshold=80.0,
            passed=True,
        )
        test_result = TestExecutionResult(
            status="passed",
            passed_count=10,
            failed_count=0,
            error_count=0,
        )
        validation_result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="passed",
            findings=(),
        )

        result = calculate_confidence_score(
            validation_results=(validation_result,),
            coverage_report=coverage_report,
            test_execution_result=test_result,
        )

        # Check breakdown has expected structure
        breakdown = result.breakdown
        assert breakdown.coverage_score == 80.0
        assert breakdown.test_execution_score == 100.0  # 100% pass rate
        assert breakdown.validation_score == 100.0
        assert breakdown.weighted_coverage == 32.0  # 80 * 0.4
        assert breakdown.weighted_test_execution == 30.0  # 100 * 0.3
        assert breakdown.weighted_validation == 30.0  # 100 * 0.3
        # Perfect test bonus should be in bonuses (applied via _apply_score_modifiers)
        assert "+5 for perfect test pass rate" in breakdown.bonuses

    def test_custom_weights(self) -> None:
        """Test that custom weights are applied."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.execution import TestExecutionResult
        from yolo_developer.agents.tea.scoring import (
            ConfidenceWeight,
            calculate_confidence_score,
        )
        from yolo_developer.agents.tea.types import ValidationResult

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
        validation_result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="passed",
            findings=(),
        )

        custom_weights = ConfidenceWeight(
            coverage_weight=0.5,
            test_execution_weight=0.25,
            validation_weight=0.25,
        )

        result = calculate_confidence_score(
            validation_results=(validation_result,),
            coverage_report=coverage_report,
            test_execution_result=test_result,
            weights=custom_weights,
        )

        # With all 100% scores, still should be 100
        assert result.score == 100
        # But weighted contributions should differ
        assert result.breakdown.weighted_coverage == 50.0  # 100 * 0.5

    def test_blocking_threshold_check(self) -> None:
        """Test that low score triggers blocking."""
        from yolo_developer.agents.tea.coverage import CoverageReport, CoverageResult
        from yolo_developer.agents.tea.execution import TestExecutionResult
        from yolo_developer.agents.tea.scoring import calculate_confidence_score
        from yolo_developer.agents.tea.types import Finding, ValidationResult

        coverage_result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=50,
            coverage_percentage=50.0,
        )
        coverage_report = CoverageReport(
            results=(coverage_result,),
            overall_coverage=50.0,
            threshold=80.0,
            passed=False,
        )
        test_result = TestExecutionResult(
            status="failed",
            passed_count=5,
            failed_count=5,
            error_count=0,
        )
        finding = Finding(
            finding_id="F001",
            category="test_coverage",
            severity="critical",
            description="Critical issue",
            location="src/module.py",
            remediation="Fix",
        )
        validation_result = ValidationResult(
            artifact_id="src/module.py",
            validation_status="failed",
            findings=(finding,),
        )

        result = calculate_confidence_score(
            validation_results=(validation_result,),
            coverage_report=coverage_report,
            test_execution_result=test_result,
        )

        # Coverage: 50 * 0.4 = 20
        # Test exec: 50 * 0.3 = 15
        # Validation: 75 (100-25) * 0.3 = 22.5
        # Total: ~57.5
        assert result.score < 90
        assert result.passed_threshold is False
        assert result.deployment_recommendation in ("block", "deploy_with_warnings")
