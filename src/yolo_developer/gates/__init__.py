"""Quality Gate Framework for YOLO Developer.

This module provides the quality gate infrastructure for validating
agent artifacts at handoff boundaries. Gates can be blocking (prevent
execution on failure) or advisory (log warning and continue).

Exports:
    quality_gate: Decorator for wrapping agent nodes with gates.
    GateResult: Result of a gate evaluation.
    GateMode: Enum for blocking/advisory mode.
    GateContext: Context passed to gate evaluators.
    GateEvaluator: Protocol for gate evaluator functions.
    register_evaluator: Register a gate evaluator by name.
    get_evaluator: Get a registered evaluator by name.
    list_evaluators: List all registered gate names.
    clear_evaluators: Clear all registered evaluators.
    resolve_threshold: Resolve threshold from config with priority order.
    GateIssue: Individual issue found during gate evaluation.
    GateFailureReport: Complete failure report for a gate.
    Severity: Issue severity level (BLOCKING/WARNING).
    generate_failure_report: Generate a failure report from issues.
    format_report_text: Format a report as human-readable text.
    get_remediation_suggestion: Get remediation guidance for an issue.

    Metrics (Story 3.9):
    GateMetricRecord: Single gate evaluation metric record.
    GateMetricsSummary: Aggregated metrics summary.
    GateTrend: Trend analysis for a gate over time.
    TrendDirection: Direction of quality trend (IMPROVING/STABLE/DECLINING).
    GateMetricsStore: Protocol for gate metrics storage backends.
    JsonGateMetricsStore: JSON file-based metrics storage implementation.
    set_metrics_store: Set the metrics store for recording gate evaluations.
    get_metrics_store: Get the currently configured metrics store.
    calculate_pass_rates: Calculate pass rates by gate type.
    calculate_gate_summary: Calculate summary statistics for a gate.
    calculate_trends: Calculate trend analysis over time periods.
    get_agent_breakdown: Get metrics breakdown by agent.
    filter_records_by_time_range: Filter records by time range.

Example - Basic Gate Usage:
    >>> from yolo_developer.gates import quality_gate, register_evaluator
    >>> from yolo_developer.gates import GateContext, GateResult
    >>>
    >>> # Define a gate evaluator
    >>> async def testability_gate(ctx: GateContext) -> GateResult:
    ...     has_tests = "test_coverage" in ctx.state
    ...     return GateResult(
    ...         passed=has_tests,
    ...         gate_name=ctx.gate_name,
    ...         reason=None if has_tests else "Missing test coverage",
    ...     )
    >>>
    >>> # Register the evaluator
    >>> register_evaluator("testability", testability_gate)
    >>>
    >>> # Use the decorator on an agent node
    >>> @quality_gate("testability", blocking=True)
    ... async def analyst_node(state: dict) -> dict:
    ...     # Agent logic here
    ...     return state

Example - Configurable Thresholds:
    Gates support configurable thresholds via the state config. Use
    resolve_threshold() to get the threshold with priority order:
    gate-specific > global > default.

    >>> from yolo_developer.gates import resolve_threshold
    >>>
    >>> # In a gate evaluator:
    >>> async def my_gate(ctx: GateContext) -> GateResult:
    ...     threshold = resolve_threshold(
    ...         gate_name="my_gate",
    ...         state=ctx.state,
    ...         default=0.80,  # Default threshold (0.0-1.0)
    ...     )
    ...     # Use threshold in evaluation logic
    ...     passed = score >= threshold
    ...     return GateResult(passed=passed, gate_name=ctx.gate_name, reason=None)

    Configuration example (yolo.yaml):

    .. code-block:: yaml

        quality:
          test_coverage_threshold: 0.85  # Global
          confidence_threshold: 0.90     # Global

          # Per-gate overrides
          gate_thresholds:
            testability:
              min_score: 0.80
              blocking: true
            architecture_validation:
              min_score: 0.70
              blocking: false  # Advisory mode

Example - Gate Failure Reports:
    Gates use a unified report system for consistent failure messaging.

    >>> from yolo_developer.gates import (
    ...     GateIssue,
    ...     Severity,
    ...     generate_failure_report,
    ...     format_report_text,
    ... )
    >>>
    >>> # Create issues found during evaluation
    >>> issues = [
    ...     GateIssue(
    ...         location="requirement-1",
    ...         issue_type="vague_term",
    ...         description="Term 'fast' is unmeasurable",
    ...         severity=Severity.BLOCKING,
    ...         context={"term": "fast"},
    ...     ),
    ... ]
    >>>
    >>> # Generate a failure report
    >>> report = generate_failure_report(
    ...     gate_name="testability",
    ...     issues=issues,
    ...     score=0.70,
    ...     threshold=0.80,
    ... )
    >>>
    >>> # Format for display
    >>> print(format_report_text(report))
"""

from __future__ import annotations

from yolo_developer.gates.decorator import get_metrics_store, quality_gate, set_metrics_store
from yolo_developer.gates.evaluators import (
    GateEvaluator,
    clear_evaluators,
    get_evaluator,
    list_evaluators,
    register_evaluator,
)
from yolo_developer.gates.metrics_api import (
    get_agent_summary,
    get_gate_metrics,
    get_pass_rates,
    get_trends,
)
from yolo_developer.gates.metrics_calculator import (
    calculate_gate_summary,
    calculate_pass_rates,
    calculate_trends,
    filter_records_by_time_range,
    get_agent_breakdown,
)
from yolo_developer.gates.metrics_store import GateMetricsStore, JsonGateMetricsStore
from yolo_developer.gates.metrics_types import (
    GateMetricRecord,
    GateMetricsSummary,
    GateTrend,
    TrendDirection,
)
from yolo_developer.gates.remediation import get_remediation_suggestion
from yolo_developer.gates.report_generator import format_report_text, generate_failure_report
from yolo_developer.gates.report_types import GateFailureReport, GateIssue, Severity
from yolo_developer.gates.threshold_resolver import resolve_threshold
from yolo_developer.gates.types import GateContext, GateMode, GateResult

__all__ = [
    "GateContext",
    "GateEvaluator",
    "GateFailureReport",
    "GateIssue",
    "GateMetricRecord",
    "GateMetricsStore",
    "GateMetricsSummary",
    "GateMode",
    "GateResult",
    "GateTrend",
    "JsonGateMetricsStore",
    "Severity",
    "TrendDirection",
    "calculate_gate_summary",
    "calculate_pass_rates",
    "calculate_trends",
    "clear_evaluators",
    "filter_records_by_time_range",
    "format_report_text",
    "generate_failure_report",
    "get_agent_breakdown",
    "get_agent_summary",
    "get_evaluator",
    "get_gate_metrics",
    "get_metrics_store",
    "get_pass_rates",
    "get_remediation_suggestion",
    "get_trends",
    "list_evaluators",
    "quality_gate",
    "register_evaluator",
    "resolve_threshold",
    "set_metrics_store",
]
