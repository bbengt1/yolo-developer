"""Unit tests for testability score calculation and recommendations (Story 9.6).

Tests for:
- calculate_testability_score
- collect_testability_metrics
- generate_testability_recommendations
"""

from __future__ import annotations


class TestCalculateTestabilityScore:
    """Tests for calculate_testability_score function."""

    def test_perfect_score_no_issues(self) -> None:
        """Test perfect score with no issues."""
        from yolo_developer.agents.tea.testability import calculate_testability_score

        score = calculate_testability_score(())
        assert score.score == 100
        assert score.base_score == 100

    def test_critical_penalty(self) -> None:
        """Test critical severity penalty (-20 per occurrence)."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            calculate_testability_score,
        )

        issue = TestabilityIssue(
            issue_id="T-test-001",
            pattern_type="global_state",
            severity="critical",
            location="test.py",
            line_start=1,
            line_end=1,
            description="test",
            impact="test",
            remediation="test",
        )

        score = calculate_testability_score((issue,))
        assert score.score == 80  # 100 - 20
        assert score.breakdown["critical_penalty"] == -20

    def test_high_penalty(self) -> None:
        """Test high severity penalty (-10 per occurrence)."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            calculate_testability_score,
        )

        issue = TestabilityIssue(
            issue_id="T-test-001",
            pattern_type="tight_coupling",
            severity="high",
            location="test.py",
            line_start=1,
            line_end=1,
            description="test",
            impact="test",
            remediation="test",
        )

        score = calculate_testability_score((issue,))
        assert score.score == 90  # 100 - 10
        assert score.breakdown["high_penalty"] == -10

    def test_medium_penalty(self) -> None:
        """Test medium severity penalty (-5 per occurrence)."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            calculate_testability_score,
        )

        issue = TestabilityIssue(
            issue_id="T-test-001",
            pattern_type="long_method",
            severity="medium",
            location="test.py",
            line_start=1,
            line_end=50,
            description="test",
            impact="test",
            remediation="test",
        )

        score = calculate_testability_score((issue,))
        assert score.score == 95  # 100 - 5
        assert score.breakdown["medium_penalty"] == -5

    def test_low_penalty(self) -> None:
        """Test low severity penalty (-2 per occurrence)."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            calculate_testability_score,
        )

        issue = TestabilityIssue(
            issue_id="T-test-001",
            pattern_type="deep_nesting",
            severity="low",
            location="test.py",
            line_start=1,
            line_end=20,
            description="test",
            impact="test",
            remediation="test",
        )

        score = calculate_testability_score((issue,))
        assert score.score == 98  # 100 - 2
        assert score.breakdown["low_penalty"] == -2

    def test_critical_penalty_cap(self) -> None:
        """Test that critical penalty is capped at -60."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            calculate_testability_score,
        )

        # Create 4 critical issues (4 * 20 = 80, should be capped at 60)
        issues = tuple(
            TestabilityIssue(
                issue_id=f"T-test-{i:03d}",
                pattern_type="global_state",
                severity="critical",
                location="test.py",
                line_start=i,
                line_end=i,
                description="test",
                impact="test",
                remediation="test",
            )
            for i in range(4)
        )

        score = calculate_testability_score(issues)
        assert score.score == 40  # 100 - 60 (capped)
        assert score.breakdown["critical_penalty"] == -60

    def test_high_penalty_cap(self) -> None:
        """Test that high penalty is capped at -40."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            calculate_testability_score,
        )

        # Create 5 high issues (5 * 10 = 50, should be capped at 40)
        issues = tuple(
            TestabilityIssue(
                issue_id=f"T-test-{i:03d}",
                pattern_type="tight_coupling",
                severity="high",
                location="test.py",
                line_start=i,
                line_end=i,
                description="test",
                impact="test",
                remediation="test",
            )
            for i in range(5)
        )

        score = calculate_testability_score(issues)
        assert score.score == 60  # 100 - 40 (capped)
        assert score.breakdown["high_penalty"] == -40

    def test_medium_penalty_cap(self) -> None:
        """Test that medium penalty is capped at -20."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            calculate_testability_score,
        )

        # Create 5 medium issues (5 * 5 = 25, should be capped at 20)
        issues = tuple(
            TestabilityIssue(
                issue_id=f"T-test-{i:03d}",
                pattern_type="long_method",
                severity="medium",
                location="test.py",
                line_start=i,
                line_end=i + 50,
                description="test",
                impact="test",
                remediation="test",
            )
            for i in range(5)
        )

        score = calculate_testability_score(issues)
        assert score.score == 80  # 100 - 20 (capped)
        assert score.breakdown["medium_penalty"] == -20

    def test_low_penalty_cap(self) -> None:
        """Test that low penalty is capped at -10."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            calculate_testability_score,
        )

        # Create 6 low issues (6 * 2 = 12, should be capped at 10)
        issues = tuple(
            TestabilityIssue(
                issue_id=f"T-test-{i:03d}",
                pattern_type="deep_nesting",
                severity="low",
                location="test.py",
                line_start=i,
                line_end=i + 10,
                description="test",
                impact="test",
                remediation="test",
            )
            for i in range(6)
        )

        score = calculate_testability_score(issues)
        assert score.score == 90  # 100 - 10 (capped)
        assert score.breakdown["low_penalty"] == -10

    def test_mixed_severities(self) -> None:
        """Test score calculation with mixed severities."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            calculate_testability_score,
        )

        issues = (
            TestabilityIssue(
                issue_id="T-test-001",
                pattern_type="global_state",
                severity="critical",
                location="test.py",
                line_start=1,
                line_end=1,
                description="test",
                impact="test",
                remediation="test",
            ),
            TestabilityIssue(
                issue_id="T-test-002",
                pattern_type="tight_coupling",
                severity="high",
                location="test.py",
                line_start=10,
                line_end=10,
                description="test",
                impact="test",
                remediation="test",
            ),
            TestabilityIssue(
                issue_id="T-test-003",
                pattern_type="deep_nesting",
                severity="low",
                location="test.py",
                line_start=20,
                line_end=30,
                description="test",
                impact="test",
                remediation="test",
            ),
        )

        score = calculate_testability_score(issues)
        # 100 - 20 (critical) - 10 (high) - 2 (low) = 68
        assert score.score == 68
        assert score.breakdown["critical_penalty"] == -20
        assert score.breakdown["high_penalty"] == -10
        assert score.breakdown["low_penalty"] == -2

    def test_score_clamped_to_zero(self) -> None:
        """Test that score cannot go below 0."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            calculate_testability_score,
        )

        # Create many critical issues to exceed 100 penalty
        issues = tuple(
            TestabilityIssue(
                issue_id=f"T-test-{i:03d}",
                pattern_type="global_state",
                severity="critical",
                location=f"test{i}.py",
                line_start=1,
                line_end=1,
                description="test",
                impact="test",
                remediation="test",
            )
            for i in range(10)
        ) + tuple(
            TestabilityIssue(
                issue_id=f"T-test-{i + 10:03d}",
                pattern_type="tight_coupling",
                severity="high",
                location=f"test{i}.py",
                line_start=1,
                line_end=1,
                description="test",
                impact="test",
                remediation="test",
            )
            for i in range(10)
        )

        score = calculate_testability_score(issues)
        # 100 - 60 (critical cap) - 40 (high cap) = 0
        assert score.score == 0


class TestCollectTestabilityMetrics:
    """Tests for collect_testability_metrics function."""

    def test_empty_issues(self) -> None:
        """Test metrics with no issues."""
        from yolo_developer.agents.tea.testability import collect_testability_metrics

        metrics = collect_testability_metrics((), files_analyzed=5)
        assert metrics.total_issues == 0
        assert metrics.files_analyzed == 5
        assert metrics.files_with_issues == 0

    def test_counts_by_severity(self) -> None:
        """Test that metrics correctly count issues by severity."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            collect_testability_metrics,
        )

        issues = (
            TestabilityIssue(
                issue_id="T-test-001",
                pattern_type="global_state",
                severity="critical",
                location="test1.py",
                line_start=1,
                line_end=1,
                description="test",
                impact="test",
                remediation="test",
            ),
            TestabilityIssue(
                issue_id="T-test-002",
                pattern_type="tight_coupling",
                severity="high",
                location="test2.py",
                line_start=1,
                line_end=1,
                description="test",
                impact="test",
                remediation="test",
            ),
            TestabilityIssue(
                issue_id="T-test-003",
                pattern_type="hidden_dependency",
                severity="high",
                location="test3.py",
                line_start=1,
                line_end=1,
                description="test",
                impact="test",
                remediation="test",
            ),
        )

        metrics = collect_testability_metrics(issues, files_analyzed=5)
        assert metrics.total_issues == 3
        assert metrics.issues_by_severity["critical"] == 1
        assert metrics.issues_by_severity["high"] == 2
        assert metrics.issues_by_severity["medium"] == 0
        assert metrics.issues_by_severity["low"] == 0

    def test_counts_by_pattern(self) -> None:
        """Test that metrics correctly count issues by pattern."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            collect_testability_metrics,
        )

        issues = (
            TestabilityIssue(
                issue_id="T-test-001",
                pattern_type="global_state",
                severity="critical",
                location="test1.py",
                line_start=1,
                line_end=1,
                description="test",
                impact="test",
                remediation="test",
            ),
            TestabilityIssue(
                issue_id="T-test-002",
                pattern_type="global_state",
                severity="critical",
                location="test2.py",
                line_start=1,
                line_end=1,
                description="test",
                impact="test",
                remediation="test",
            ),
            TestabilityIssue(
                issue_id="T-test-003",
                pattern_type="long_method",
                severity="medium",
                location="test3.py",
                line_start=1,
                line_end=60,
                description="test",
                impact="test",
                remediation="test",
            ),
        )

        metrics = collect_testability_metrics(issues, files_analyzed=5)
        assert metrics.issues_by_pattern["global_state"] == 2
        assert metrics.issues_by_pattern["long_method"] == 1

    def test_counts_unique_files_with_issues(self) -> None:
        """Test that files_with_issues counts unique files."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            collect_testability_metrics,
        )

        # Two issues in same file, one in different file
        issues = (
            TestabilityIssue(
                issue_id="T-test-001",
                pattern_type="global_state",
                severity="critical",
                location="test1.py",
                line_start=1,
                line_end=1,
                description="test",
                impact="test",
                remediation="test",
            ),
            TestabilityIssue(
                issue_id="T-test-002",
                pattern_type="long_method",
                severity="medium",
                location="test1.py",  # Same file
                line_start=10,
                line_end=70,
                description="test",
                impact="test",
                remediation="test",
            ),
            TestabilityIssue(
                issue_id="T-test-003",
                pattern_type="tight_coupling",
                severity="high",
                location="test2.py",  # Different file
                line_start=1,
                line_end=1,
                description="test",
                impact="test",
                remediation="test",
            ),
        )

        metrics = collect_testability_metrics(issues, files_analyzed=5)
        assert metrics.files_with_issues == 2  # Only 2 unique files


class TestGenerateTestabilityRecommendations:
    """Tests for generate_testability_recommendations function."""

    def test_empty_issues(self) -> None:
        """Test recommendations with no issues."""
        from yolo_developer.agents.tea.testability import generate_testability_recommendations

        recommendations = generate_testability_recommendations(())
        assert len(recommendations) == 0

    def test_prioritizes_critical_first(self) -> None:
        """Test that critical issues are listed first."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            generate_testability_recommendations,
        )

        issues = (
            TestabilityIssue(
                issue_id="T-test-001",
                pattern_type="deep_nesting",
                severity="low",
                location="test1.py",
                line_start=1,
                line_end=20,
                description="Low issue",
                impact="test",
                remediation="Fix low",
            ),
            TestabilityIssue(
                issue_id="T-test-002",
                pattern_type="global_state",
                severity="critical",
                location="test2.py",
                line_start=1,
                line_end=1,
                description="Critical issue",
                impact="test",
                remediation="Fix critical",
            ),
        )

        recommendations = generate_testability_recommendations(issues)
        assert len(recommendations) == 2
        assert recommendations[0].startswith("CRITICAL:")
        assert recommendations[1].startswith("LOW:")

    def test_deduplicates_same_pattern_same_file(self) -> None:
        """Test that duplicate pattern+file combinations are deduplicated."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            generate_testability_recommendations,
        )

        # Two issues of same pattern in same file
        issues = (
            TestabilityIssue(
                issue_id="T-test-001",
                pattern_type="global_state",
                severity="critical",
                location="test.py",
                line_start=1,
                line_end=1,
                description="First global state",
                impact="test",
                remediation="Fix it",
            ),
            TestabilityIssue(
                issue_id="T-test-002",
                pattern_type="global_state",
                severity="critical",
                location="test.py",  # Same file
                line_start=10,
                line_end=10,
                description="Second global state",
                impact="test",
                remediation="Fix it",
            ),
        )

        recommendations = generate_testability_recommendations(issues)
        # Should have only 1 recommendation (deduplicated)
        assert len(recommendations) == 1

    def test_includes_description_and_remediation(self) -> None:
        """Test that recommendations include description and remediation."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            generate_testability_recommendations,
        )

        issue = TestabilityIssue(
            issue_id="T-test-001",
            pattern_type="global_state",
            severity="critical",
            location="test.py",
            line_start=1,
            line_end=1,
            description="Module-level mutable variable",
            impact="test",
            remediation="Use dependency injection",
        )

        recommendations = generate_testability_recommendations((issue,))
        assert "Module-level mutable variable" in recommendations[0]
        assert "Use dependency injection" in recommendations[0]


class TestConvertTestabilityIssuesToFindings:
    """Tests for convert_testability_issues_to_findings function."""

    def test_empty_issues(self) -> None:
        """Test conversion with no issues."""
        from yolo_developer.agents.tea.testability import convert_testability_issues_to_findings

        findings = convert_testability_issues_to_findings(())
        assert len(findings) == 0

    def test_converts_issue_to_finding(self) -> None:
        """Test conversion of testability issue to finding."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            convert_testability_issues_to_findings,
        )

        issue = TestabilityIssue(
            issue_id="T-test-001",
            pattern_type="global_state",
            severity="critical",
            location="test.py",
            line_start=10,
            line_end=15,
            description="Module-level mutable variable",
            impact="Cannot isolate tests",
            remediation="Use dependency injection",
        )

        findings = convert_testability_issues_to_findings((issue,))
        assert len(findings) == 1

        finding = findings[0]
        assert finding.finding_id == "F-test-001"  # T- replaced with F-
        assert finding.category == "code_quality"
        assert finding.severity == "critical"
        assert "Testability:" in finding.description
        assert finding.location == "test.py:10-15"
        assert finding.remediation == "Use dependency injection"

    def test_maps_all_severities(self) -> None:
        """Test that all severity levels are mapped correctly."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            convert_testability_issues_to_findings,
        )

        issues = tuple(
            TestabilityIssue(
                issue_id=f"T-test-{i:03d}",
                pattern_type="global_state",
                severity=severity,
                location="test.py",
                line_start=i,
                line_end=i,
                description="test",
                impact="test",
                remediation="test",
            )
            for i, severity in enumerate(["critical", "high", "medium", "low"])
        )

        findings = convert_testability_issues_to_findings(issues)
        assert len(findings) == 4

        severities = [f.severity for f in findings]
        assert "critical" in severities
        assert "high" in severities
        assert "medium" in severities
        assert "low" in severities
