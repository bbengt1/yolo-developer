"""Unit tests for gate metrics calculator (Story 3.9 - Tasks 5 & 6).

These tests verify the pass/fail rate and summary calculations:
- Pass rate calculations by gate
- Gate summary calculations
- Time range filtering
- Trend analysis
- Edge cases (empty data, single record)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from yolo_developer.gates.metrics_types import GateMetricRecord


class TestCalculatePassRates:
    """Tests for calculate_pass_rates function."""

    def test_function_is_importable(self) -> None:
        """calculate_pass_rates should be importable."""
        from yolo_developer.gates.metrics_calculator import calculate_pass_rates

        assert calculate_pass_rates is not None
        assert callable(calculate_pass_rates)

    def test_empty_records_returns_empty_dict(self) -> None:
        """Empty records list should return empty dict."""
        from yolo_developer.gates.metrics_calculator import calculate_pass_rates

        result = calculate_pass_rates([])

        assert result == {}

    def test_single_gate_all_pass(self) -> None:
        """Single gate with all passes should return 100.0."""
        from yolo_developer.gates.metrics_calculator import calculate_pass_rates

        timestamp = datetime.now(timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
            )
            for _ in range(5)
        ]

        result = calculate_pass_rates(records)

        assert "testability" in result
        assert result["testability"] == 100.0

    def test_single_gate_all_fail(self) -> None:
        """Single gate with all failures should return 0.0."""
        from yolo_developer.gates.metrics_calculator import calculate_pass_rates

        timestamp = datetime.now(timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=False,
                score=0.50,
                threshold=0.80,
                timestamp=timestamp,
            )
            for _ in range(5)
        ]

        result = calculate_pass_rates(records)

        assert result["testability"] == 0.0

    def test_single_gate_mixed_results(self) -> None:
        """Single gate with mixed results should calculate correct rate."""
        from yolo_developer.gates.metrics_calculator import calculate_pass_rates

        timestamp = datetime.now(timezone.utc)
        # 7 pass, 3 fail = 70%
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=i < 7,
                score=0.85 if i < 7 else 0.50,
                threshold=0.80,
                timestamp=timestamp,
            )
            for i in range(10)
        ]

        result = calculate_pass_rates(records)

        assert result["testability"] == 70.0

    def test_multiple_gates(self) -> None:
        """Multiple gates should have separate pass rates."""
        from yolo_developer.gates.metrics_calculator import calculate_pass_rates

        timestamp = datetime.now(timezone.utc)
        records = [
            # testability: 2 pass, 1 fail = 66.67%
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=False,
                score=0.50,
                threshold=0.80,
                timestamp=timestamp,
            ),
            # architecture: 1 pass, 1 fail = 50%
            GateMetricRecord(
                gate_name="architecture",
                passed=True,
                score=0.90,
                threshold=0.80,
                timestamp=timestamp,
            ),
            GateMetricRecord(
                gate_name="architecture",
                passed=False,
                score=0.60,
                threshold=0.80,
                timestamp=timestamp,
            ),
        ]

        result = calculate_pass_rates(records)

        assert len(result) == 2
        assert "testability" in result
        assert "architecture" in result
        assert abs(result["testability"] - 66.67) < 0.01
        assert result["architecture"] == 50.0


class TestCalculateGateSummary:
    """Tests for calculate_gate_summary function."""

    def test_function_is_importable(self) -> None:
        """calculate_gate_summary should be importable."""
        from yolo_developer.gates.metrics_calculator import calculate_gate_summary

        assert calculate_gate_summary is not None
        assert callable(calculate_gate_summary)

    def test_empty_records_returns_zero_summary(self) -> None:
        """Empty records should return zero-filled summary."""
        from yolo_developer.gates.metrics_calculator import calculate_gate_summary

        result = calculate_gate_summary("testability", [])

        assert result.gate_name == "testability"
        assert result.total_evaluations == 0
        assert result.pass_count == 0
        assert result.fail_count == 0
        assert result.pass_rate == 0.0
        assert result.avg_score == 0.0

    def test_summary_includes_all_fields(self) -> None:
        """Summary should include all required fields."""
        from yolo_developer.gates.metrics_calculator import calculate_gate_summary

        timestamp = datetime.now(timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
            )
        ]

        result = calculate_gate_summary("testability", records)

        assert result.gate_name == "testability"
        assert result.total_evaluations == 1
        assert result.pass_count == 1
        assert result.fail_count == 0
        assert result.pass_rate == 100.0
        assert result.avg_score == 0.85

    def test_summary_calculates_correct_counts(self) -> None:
        """Summary should calculate correct pass/fail counts."""
        from yolo_developer.gates.metrics_calculator import calculate_gate_summary

        timestamp = datetime.now(timezone.utc)
        # 7 pass, 3 fail
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=i < 7,
                score=0.85 if i < 7 else 0.50,
                threshold=0.80,
                timestamp=timestamp,
            )
            for i in range(10)
        ]

        result = calculate_gate_summary("testability", records)

        assert result.total_evaluations == 10
        assert result.pass_count == 7
        assert result.fail_count == 3
        assert result.pass_rate == 70.0

    def test_summary_calculates_average_score(self) -> None:
        """Summary should calculate correct average score."""
        from yolo_developer.gates.metrics_calculator import calculate_gate_summary

        timestamp = datetime.now(timezone.utc)
        # Scores: 0.80, 0.85, 0.90 -> avg = 0.85
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=score,
                threshold=0.80,
                timestamp=timestamp,
            )
            for score in [0.80, 0.85, 0.90]
        ]

        result = calculate_gate_summary("testability", records)

        assert result.avg_score == 0.85

    def test_summary_includes_time_range(self) -> None:
        """Summary should include correct period start and end."""
        from yolo_developer.gates.metrics_calculator import calculate_gate_summary

        base_time = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(hours=i),
            )
            for i in range(5)
        ]

        result = calculate_gate_summary("testability", records)

        assert result.period_start == base_time
        assert result.period_end == base_time + timedelta(hours=4)

    def test_summary_filters_by_gate_name(self) -> None:
        """Summary should only include records for specified gate."""
        from yolo_developer.gates.metrics_calculator import calculate_gate_summary

        timestamp = datetime.now(timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=False,
                score=0.50,
                threshold=0.80,
                timestamp=timestamp,
            ),
            GateMetricRecord(
                gate_name="architecture",
                passed=True,
                score=0.90,
                threshold=0.80,
                timestamp=timestamp,
            ),
        ]

        result = calculate_gate_summary("testability", records)

        assert result.total_evaluations == 2
        assert result.pass_count == 1
        assert result.fail_count == 1


class TestFilterRecordsByTimeRange:
    """Tests for filter_records_by_time_range function."""

    def test_function_is_importable(self) -> None:
        """filter_records_by_time_range should be importable."""
        from yolo_developer.gates.metrics_calculator import filter_records_by_time_range

        assert filter_records_by_time_range is not None
        assert callable(filter_records_by_time_range)

    def test_no_filters_returns_all_records(self) -> None:
        """No time filters should return all records."""
        from yolo_developer.gates.metrics_calculator import filter_records_by_time_range

        timestamp = datetime.now(timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
            )
            for _ in range(5)
        ]

        result = filter_records_by_time_range(records)

        assert len(result) == 5

    def test_filter_by_start_time(self) -> None:
        """Should filter records before start_time."""
        from yolo_developer.gates.metrics_calculator import filter_records_by_time_range

        base_time = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(hours=i),
            )
            for i in range(5)
        ]

        # Start from hour 2 (should exclude hours 0 and 1)
        start = base_time + timedelta(hours=2)
        result = filter_records_by_time_range(records, start_time=start)

        assert len(result) == 3

    def test_filter_by_end_time(self) -> None:
        """Should filter records after end_time."""
        from yolo_developer.gates.metrics_calculator import filter_records_by_time_range

        base_time = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(hours=i),
            )
            for i in range(5)
        ]

        # End at hour 2 (should exclude hours 3 and 4)
        end = base_time + timedelta(hours=2)
        result = filter_records_by_time_range(records, end_time=end)

        assert len(result) == 3

    def test_filter_by_time_range(self) -> None:
        """Should filter records outside time range."""
        from yolo_developer.gates.metrics_calculator import filter_records_by_time_range

        base_time = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(hours=i),
            )
            for i in range(10)
        ]

        # Filter to hours 3-6 (inclusive)
        start = base_time + timedelta(hours=3)
        end = base_time + timedelta(hours=6)
        result = filter_records_by_time_range(records, start_time=start, end_time=end)

        assert len(result) == 4


class TestCalculateTrends:
    """Tests for calculate_trends function (Task 6)."""

    def test_function_is_importable(self) -> None:
        """calculate_trends should be importable."""
        from yolo_developer.gates.metrics_calculator import calculate_trends

        assert calculate_trends is not None
        assert callable(calculate_trends)

    def test_empty_records_returns_empty_list(self) -> None:
        """Empty records should return empty list."""
        from yolo_developer.gates.metrics_calculator import calculate_trends

        result = calculate_trends([], "daily")

        assert result == []

    def test_daily_period_supported(self) -> None:
        """Should support daily period aggregation."""
        from yolo_developer.gates.metrics_calculator import calculate_trends

        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(days=i),
            )
            for i in range(5)
        ]

        result = calculate_trends(records, "daily")

        assert len(result) == 5
        assert all(t.period == "daily" for t in result)

    def test_weekly_period_supported(self) -> None:
        """Should support weekly period aggregation."""
        from yolo_developer.gates.metrics_calculator import calculate_trends

        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Create records across 3 weeks
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(weeks=i),
            )
            for i in range(3)
        ]

        result = calculate_trends(records, "weekly")

        assert len(result) == 3
        assert all(t.period == "weekly" for t in result)

    def test_sprint_period_supported(self) -> None:
        """Should support sprint period aggregation."""
        from yolo_developer.gates.metrics_calculator import calculate_trends

        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Create records with sprint IDs
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time,
                sprint_id="sprint-1",
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=False,
                score=0.50,
                threshold=0.80,
                timestamp=base_time + timedelta(days=7),
                sprint_id="sprint-2",
            ),
        ]

        result = calculate_trends(records, "sprint")

        assert len(result) == 2
        assert all(t.period == "sprint" for t in result)

    def test_trend_includes_pass_rate(self) -> None:
        """Trends should include pass rate calculations."""
        from yolo_developer.gates.metrics_calculator import calculate_trends

        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Day 1: 2 pass, Day 2: 1 pass 1 fail
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time,
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(hours=1),
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(days=1),
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=False,
                score=0.50,
                threshold=0.80,
                timestamp=base_time + timedelta(days=1, hours=1),
            ),
        ]

        result = calculate_trends(records, "daily")

        assert len(result) == 2
        # First day: 100% pass rate
        assert result[0].pass_rate == 100.0
        # Second day: 50% pass rate
        assert result[1].pass_rate == 50.0

    def test_trend_includes_evaluation_count(self) -> None:
        """Trends should include evaluation count per period."""
        from yolo_developer.gates.metrics_calculator import calculate_trends

        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Day 1: 3 evals, Day 2: 2 evals
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(hours=i),
            )
            for i in range(3)
        ] + [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(days=1, hours=i),
            )
            for i in range(2)
        ]

        result = calculate_trends(records, "daily")

        assert result[0].evaluation_count == 3
        assert result[1].evaluation_count == 2

    def test_trend_direction_improving(self) -> None:
        """Should detect improving trend direction."""
        from yolo_developer.gates.metrics_calculator import calculate_trends
        from yolo_developer.gates.metrics_types import TrendDirection

        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Day 1: 50% pass rate, Day 2: 100% pass rate (improving)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time,
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=False,
                score=0.50,
                threshold=0.80,
                timestamp=base_time + timedelta(hours=1),
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(days=1),
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(days=1, hours=1),
            ),
        ]

        result = calculate_trends(records, "daily")

        assert result[1].direction == TrendDirection.IMPROVING

    def test_trend_direction_declining(self) -> None:
        """Should detect declining trend direction."""
        from yolo_developer.gates.metrics_calculator import calculate_trends
        from yolo_developer.gates.metrics_types import TrendDirection

        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Day 1: 100% pass rate, Day 2: 50% pass rate (declining)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time,
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(hours=1),
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(days=1),
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=False,
                score=0.50,
                threshold=0.80,
                timestamp=base_time + timedelta(days=1, hours=1),
            ),
        ]

        result = calculate_trends(records, "daily")

        assert result[1].direction == TrendDirection.DECLINING

    def test_trend_direction_stable(self) -> None:
        """Should detect stable trend direction."""
        from yolo_developer.gates.metrics_calculator import calculate_trends
        from yolo_developer.gates.metrics_types import TrendDirection

        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Day 1 and Day 2: same pass rate (stable)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time,
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time + timedelta(days=1),
            ),
        ]

        result = calculate_trends(records, "daily")

        assert result[1].direction == TrendDirection.STABLE

    def test_first_period_is_stable(self) -> None:
        """First period should always be STABLE (no previous to compare)."""
        from yolo_developer.gates.metrics_calculator import calculate_trends
        from yolo_developer.gates.metrics_types import TrendDirection

        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time,
            ),
        ]

        result = calculate_trends(records, "daily")

        assert result[0].direction == TrendDirection.STABLE

    def test_trend_period_label_daily(self) -> None:
        """Daily trends should have date labels."""
        from yolo_developer.gates.metrics_calculator import calculate_trends

        base_time = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time,
            ),
        ]

        result = calculate_trends(records, "daily")

        assert result[0].period_label == "2026-01-05"

    def test_trend_filters_by_gate(self) -> None:
        """Trends should only include specified gate."""
        from yolo_developer.gates.metrics_calculator import calculate_trends

        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=base_time,
            ),
            GateMetricRecord(
                gate_name="architecture",
                passed=False,
                score=0.50,
                threshold=0.80,
                timestamp=base_time,
            ),
        ]

        result = calculate_trends(records, "daily", gate_name="testability")

        assert len(result) == 1
        assert result[0].gate_name == "testability"
        assert result[0].pass_rate == 100.0


class TestGetAgentBreakdown:
    """Tests for get_agent_breakdown function (Task 7)."""

    def test_function_is_importable(self) -> None:
        """get_agent_breakdown should be importable."""
        from yolo_developer.gates.metrics_calculator import get_agent_breakdown

        assert get_agent_breakdown is not None
        assert callable(get_agent_breakdown)

    def test_empty_records_returns_empty_dict(self) -> None:
        """Empty records should return empty dict."""
        from yolo_developer.gates.metrics_calculator import get_agent_breakdown

        result = get_agent_breakdown([])

        assert result == {}

    def test_groups_by_agent_name(self) -> None:
        """Should group records by agent_name."""
        from yolo_developer.gates.metrics_calculator import get_agent_breakdown

        timestamp = datetime.now(timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="analyst",
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="analyst",
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=False,
                score=0.50,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="dev",
            ),
        ]

        result = get_agent_breakdown(records)

        assert "analyst" in result
        assert "dev" in result
        assert result["analyst"].total_evaluations == 2
        assert result["dev"].total_evaluations == 1

    def test_calculates_per_agent_pass_rates(self) -> None:
        """Should calculate pass rates per agent."""
        from yolo_developer.gates.metrics_calculator import get_agent_breakdown

        timestamp = datetime.now(timezone.utc)
        records = [
            # analyst: 2 pass = 100%
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="analyst",
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="analyst",
            ),
            # dev: 1 pass, 1 fail = 50%
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="dev",
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=False,
                score=0.50,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="dev",
            ),
        ]

        result = get_agent_breakdown(records)

        assert result["analyst"].pass_rate == 100.0
        assert result["dev"].pass_rate == 50.0

    def test_excludes_records_without_agent(self) -> None:
        """Should exclude records with None agent_name."""
        from yolo_developer.gates.metrics_calculator import get_agent_breakdown

        timestamp = datetime.now(timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="analyst",
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
                agent_name=None,  # No agent
            ),
        ]

        result = get_agent_breakdown(records)

        assert len(result) == 1
        assert "analyst" in result

    def test_identifies_highest_failure_rates(self) -> None:
        """Should be sortable by failure rate."""
        from yolo_developer.gates.metrics_calculator import get_agent_breakdown

        timestamp = datetime.now(timezone.utc)
        records = [
            # analyst: 0% failures
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="analyst",
            ),
            # dev: 50% failures (1 fail out of 2)
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.85,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="dev",
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=False,
                score=0.50,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="dev",
            ),
            # tester: 100% failures (2 fails out of 2)
            GateMetricRecord(
                gate_name="testability",
                passed=False,
                score=0.50,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="tester",
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=False,
                score=0.40,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="tester",
            ),
        ]

        result = get_agent_breakdown(records)

        # Sort by failure count to identify worst performers
        sorted_by_failures = sorted(result.items(), key=lambda x: x[1].fail_count, reverse=True)

        assert sorted_by_failures[0][0] == "tester"  # Most failures (2)
        assert sorted_by_failures[1][0] == "dev"  # 1 failure
        assert sorted_by_failures[2][0] == "analyst"  # 0 failures

    def test_calculates_average_score_per_agent(self) -> None:
        """Should calculate average score per agent."""
        from yolo_developer.gates.metrics_calculator import get_agent_breakdown

        timestamp = datetime.now(timezone.utc)
        records = [
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.80,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="analyst",
            ),
            GateMetricRecord(
                gate_name="testability",
                passed=True,
                score=0.90,
                threshold=0.80,
                timestamp=timestamp,
                agent_name="analyst",
            ),
        ]

        result = get_agent_breakdown(records)

        # Use approximate comparison for floating-point
        assert abs(result["analyst"].avg_score - 0.85) < 0.001
