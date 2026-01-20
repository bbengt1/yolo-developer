"""Semantic validation report types and generation (Story 4.6).

This module provides data models and functions for generating comprehensive
validation reports from seed parsing results:

- ReportFormat: Enum for output formats (JSON, MARKDOWN, RICH)
- QualityMetrics: Dataclass with ambiguity, SOP, and extraction scores
- ValidationReport: Complete validation report with all data
- calculate_quality_score: Calculate quality metrics from parse result
- generate_validation_report: Generate complete report from parse result
- format_report_json: Format report as JSON string
- format_report_markdown: Format report as Markdown string
- format_report_rich: Display report using Rich console

Example:
    >>> from yolo_developer.seed.report import (
    ...     generate_validation_report,
    ...     format_report_json,
    ...     format_report_markdown,
    ...     ReportFormat,
    ... )
    >>> from yolo_developer.seed import parse_seed
    >>>
    >>> result = await parse_seed("Build an e-commerce platform")
    >>> report = generate_validation_report(result)
    >>> print(format_report_json(report))
    >>> print(format_report_markdown(report))

Security Note:
    Reports may contain sensitive information from seed documents.
    Review output before exposing to external parties.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from yolo_developer.seed.types import SeedParseResult

logger = structlog.get_logger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ReportFormat(str, Enum):
    """Output format for validation reports.

    Values:
        JSON: Machine-readable JSON format
        MARKDOWN: Human-readable Markdown format
        RICH: Rich console format for CLI display
    """

    JSON = "json"
    MARKDOWN = "markdown"
    RICH = "rich"


# =============================================================================
# Quality Score Constants
# =============================================================================


# Weights for overall score calculation
AMBIGUITY_WEIGHT = 0.3
SOP_WEIGHT = 0.3
EXTRACTION_WEIGHT = 0.4

# Deduction per ambiguity by severity (for ambiguity_score)
# Note: We use ambiguity_confidence from parse result directly
AMBIGUITY_SEVERITY_DEDUCTIONS = {
    "high": 0.15,
    "medium": 0.08,
    "low": 0.03,
}

# Deduction per SOP conflict by severity (for sop_score)
SOP_SEVERITY_DEDUCTIONS = {
    "hard": 0.20,
    "soft": 0.08,
}


# =============================================================================
# Data Models
# =============================================================================


@dataclass(frozen=True)
class QualityMetrics:
    """Quality metrics calculated from validation results.

    Immutable dataclass representing score breakdown for a parsed seed.
    All scores are normalized to 0.0-1.0 range.

    Attributes:
        ambiguity_score: Score based on ambiguity count/severity (1.0 = no ambiguities)
        sop_score: Score based on SOP conflict count/severity (1.0 = no conflicts)
        extraction_score: Score based on extraction confidence (1.0 = high confidence)
        overall_score: Weighted combination of all scores

    Example:
        >>> metrics = QualityMetrics(
        ...     ambiguity_score=0.85,
        ...     sop_score=0.90,
        ...     extraction_score=0.95,
        ...     overall_score=0.90,
        ... )
    """

    ambiguity_score: float
    sop_score: float
    extraction_score: float
    overall_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all score fields.
        """
        return {
            "ambiguity_score": self.ambiguity_score,
            "sop_score": self.sop_score,
            "extraction_score": self.extraction_score,
            "overall_score": self.overall_score,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Complete validation report for a parsed seed document.

    Immutable dataclass containing the parse result, quality metrics,
    and report metadata.

    Attributes:
        parse_result: The SeedParseResult being reported on
        quality_metrics: Calculated quality scores
        report_id: Unique identifier for this report
        generated_at: ISO timestamp when report was generated
        source_file: Optional source file path

    Example:
        >>> report = ValidationReport(
        ...     parse_result=result,
        ...     quality_metrics=metrics,
        ...     report_id="rpt-abc123",
        ... )
    """

    parse_result: SeedParseResult
    quality_metrics: QualityMetrics
    report_id: str
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Includes summary section with counts and recommendations.

        Returns:
            Dictionary with all report fields and computed sections.
        """
        # Build summary section
        summary = {
            "goals_count": self.parse_result.goal_count,
            "features_count": self.parse_result.feature_count,
            "constraints_count": self.parse_result.constraint_count,
            "ambiguity_count": self.parse_result.ambiguity_count,
            "sop_conflict_count": (
                len(self.parse_result.sop_validation.conflicts)
                if self.parse_result.sop_validation
                else 0
            ),
            "quality_score": self.quality_metrics.overall_score,
        }

        # Build ambiguities section (prioritized)
        ambiguities_list = []
        if self.parse_result.has_ambiguities:
            # Import here to avoid circular import
            from yolo_developer.seed.ambiguity import calculate_question_priority

            for amb in self.parse_result.ambiguities:
                priority = calculate_question_priority(amb)
                ambiguities_list.append(
                    {
                        "priority": priority,
                        **amb.to_dict(),
                    }
                )
            # Sort by priority (highest first)
            ambiguities_list.sort(key=lambda x: -x["priority"])

        # Build SOP conflicts section (severity-sorted)
        sop_conflicts_list = []
        if self.parse_result.sop_validation and self.parse_result.has_sop_conflicts:
            for conflict in self.parse_result.sop_validation.conflicts:
                sop_conflicts_list.append(conflict.to_dict())
            # Sort by severity (HARD first)
            sop_conflicts_list.sort(key=lambda x: 0 if x["severity"] == "hard" else 1)

        # Build recommendations
        recommendations = _generate_recommendations(self)

        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "source_file": self.source_file,
            "summary": summary,
            "quality_metrics": self.quality_metrics.to_dict(),
            "ambiguities": ambiguities_list,
            "sop_conflicts": sop_conflicts_list,
            "parse_result": self.parse_result.to_dict(),
            "recommendations": recommendations,
        }


# =============================================================================
# Quality Score Calculation
# =============================================================================


def calculate_quality_score(result: SeedParseResult) -> QualityMetrics:
    """Calculate quality metrics from a parse result.

    Computes individual scores for ambiguities, SOP conflicts, and
    extraction confidence, then combines them into an overall score.

    Score calculation:
    - ambiguity_score: Uses parse result's ambiguity_confidence
    - sop_score: 1.0 - (HARD*0.20 + SOFT*0.08)
    - extraction_score: 1.0 (placeholder - could be enhanced with confidence data)
    - overall_score: weighted average (0.3*amb + 0.3*sop + 0.4*ext)

    Args:
        result: The SeedParseResult to analyze.

    Returns:
        QualityMetrics with all calculated scores.

    Example:
        >>> result = SeedParseResult(...)
        >>> metrics = calculate_quality_score(result)
        >>> print(f"Overall: {metrics.overall_score:.2f}")
    """
    # Ambiguity score - use the pre-calculated ambiguity_confidence
    ambiguity_score = result.ambiguity_confidence

    # SOP score - calculate from conflicts
    sop_score = _calculate_sop_score(result)

    # Extraction score - based on component confidence
    # For now, use 1.0 as baseline (could be enhanced with actual confidence data)
    extraction_score = _calculate_extraction_score(result)

    # Overall weighted score
    overall_score = (
        ambiguity_score * AMBIGUITY_WEIGHT
        + sop_score * SOP_WEIGHT
        + extraction_score * EXTRACTION_WEIGHT
    )

    # Clamp to [0.0, 1.0]
    overall_score = max(0.0, min(1.0, overall_score))

    logger.debug(
        "quality_score_calculated",
        ambiguity_score=round(ambiguity_score, 3),
        sop_score=round(sop_score, 3),
        extraction_score=round(extraction_score, 3),
        overall_score=round(overall_score, 3),
    )

    return QualityMetrics(
        ambiguity_score=ambiguity_score,
        sop_score=sop_score,
        extraction_score=extraction_score,
        overall_score=overall_score,
    )


def _calculate_sop_score(result: SeedParseResult) -> float:
    """Calculate SOP score based on conflicts.

    Args:
        result: Parse result with optional SOP validation.

    Returns:
        Score from 0.0 to 1.0 (1.0 = no conflicts).
    """
    if result.sop_validation is None:
        return 1.0

    if not result.sop_validation.has_conflicts:
        return 1.0

    # Calculate deductions
    total_deduction = 0.0
    for conflict in result.sop_validation.conflicts:
        severity = conflict.severity.value.lower()
        deduction = SOP_SEVERITY_DEDUCTIONS.get(severity, 0.0)
        total_deduction += deduction

    score = 1.0 - total_deduction
    return max(0.0, min(1.0, score))


def _calculate_extraction_score(result: SeedParseResult) -> float:
    """Calculate extraction score based on parsed components.

    Args:
        result: Parse result with goals, features, constraints.

    Returns:
        Score from 0.0 to 1.0 (1.0 = high confidence extraction).
    """
    # For now, return 1.0 as baseline
    # This could be enhanced to use actual extraction confidence from LLM
    # or based on the completeness of extracted data

    # Simple heuristic: having at least one goal and one feature is good
    has_goals = result.goal_count > 0
    has_features = result.feature_count > 0

    if has_goals and has_features:
        return 1.0
    elif has_goals or has_features:
        return 0.85
    else:
        return 0.7


# =============================================================================
# Report Generation
# =============================================================================


def generate_validation_report(
    result: SeedParseResult,
    source_file: str | None = None,
) -> ValidationReport:
    """Generate a comprehensive validation report from a parse result.

    Args:
        result: The SeedParseResult to report on.
        source_file: Optional source file path.

    Returns:
        ValidationReport with all data and metrics.

    Example:
        >>> report = generate_validation_report(result)
        >>> print(report.quality_metrics.overall_score)
    """
    report_id = f"rpt-{uuid.uuid4().hex[:8]}"
    quality_metrics = calculate_quality_score(result)

    logger.info(
        "validation_report_generated",
        report_id=report_id,
        overall_score=round(quality_metrics.overall_score, 3),
        ambiguity_count=result.ambiguity_count,
        sop_conflict_count=(len(result.sop_validation.conflicts) if result.sop_validation else 0),
    )

    return ValidationReport(
        parse_result=result,
        quality_metrics=quality_metrics,
        report_id=report_id,
        source_file=source_file,
    )


def _generate_recommendations(report: ValidationReport) -> list[str]:
    """Generate actionable recommendations based on report findings.

    Args:
        report: The validation report to analyze.

    Returns:
        List of recommendation strings.
    """
    recommendations: list[str] = []
    result = report.parse_result
    metrics = report.quality_metrics

    # Ambiguity recommendations
    if result.has_ambiguities:
        high_count = sum(1 for amb in result.ambiguities if amb.severity.value == "high")
        if high_count > 0:
            recommendations.append(
                f"Resolve {high_count} high-severity ambiguities before proceeding"
            )
        elif result.ambiguity_count > 3:
            recommendations.append(
                f"Consider clarifying {result.ambiguity_count} ambiguities to improve quality"
            )

    # SOP recommendations
    if result.has_sop_conflicts and result.sop_validation:
        hard_count = result.sop_validation.hard_conflict_count
        soft_count = result.sop_validation.soft_conflict_count
        if hard_count > 0:
            recommendations.append(
                f"BLOCKING: Resolve {hard_count} hard SOP conflicts before implementation"
            )
        if soft_count > 0:
            recommendations.append(f"Consider addressing {soft_count} soft SOP conflicts")

    # Quality score recommendations
    if metrics.overall_score < 0.7:
        recommendations.append("Quality score below threshold - significant improvements needed")
    elif metrics.overall_score < 0.85:
        recommendations.append("Quality score moderate - some improvements recommended")

    # Content recommendations
    if result.goal_count == 0:
        recommendations.append("No goals extracted - consider adding clear project objectives")
    if result.feature_count == 0:
        recommendations.append(
            "No features extracted - consider adding specific feature descriptions"
        )

    return recommendations


# =============================================================================
# Report Formatters
# =============================================================================


def format_report_json(report: ValidationReport, indent: int = 2) -> str:
    """Format validation report as JSON string.

    Args:
        report: The validation report to format.
        indent: JSON indentation level (default: 2).

    Returns:
        Pretty-printed JSON string.

    Example:
        >>> json_str = format_report_json(report)
        >>> print(json_str)
    """
    return json.dumps(report.to_dict(), indent=indent)


def format_report_markdown(report: ValidationReport) -> str:
    """Format validation report as Markdown string.

    Args:
        report: The validation report to format.

    Returns:
        Markdown-formatted string.

    Example:
        >>> md_str = format_report_markdown(report)
        >>> print(md_str)
    """
    lines: list[str] = []
    data = report.to_dict()

    # Header
    lines.append("# Validation Report")
    lines.append("")
    lines.append(f"**Report ID:** {data['report_id']}")
    lines.append(f"**Generated:** {data['generated_at']}")
    if data["source_file"]:
        lines.append(f"**Source:** {data['source_file']}")
    lines.append("")

    # Summary section
    lines.append("## Summary")
    lines.append("")
    summary = data["summary"]
    lines.append(f"- **Goals:** {summary['goals_count']}")
    lines.append(f"- **Features:** {summary['features_count']}")
    lines.append(f"- **Constraints:** {summary['constraints_count']}")
    lines.append(f"- **Ambiguities:** {summary['ambiguity_count']}")
    lines.append(f"- **SOP Conflicts:** {summary['sop_conflict_count']}")
    lines.append("")

    # Quality Metrics section
    lines.append("## Quality Metrics")
    lines.append("")
    metrics = data["quality_metrics"]
    overall_pct = int(metrics["overall_score"] * 100)
    lines.append(f"**Overall Score: {overall_pct}%**")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    lines.append(f"| Ambiguity Score | {int(metrics['ambiguity_score'] * 100)}% |")
    lines.append(f"| SOP Score | {int(metrics['sop_score'] * 100)}% |")
    lines.append(f"| Extraction Score | {int(metrics['extraction_score'] * 100)}% |")
    lines.append("")

    # Ambiguities section
    if data["ambiguities"]:
        lines.append("## Ambiguities")
        lines.append("")
        for i, amb in enumerate(data["ambiguities"], 1):
            severity = amb["severity"].upper()
            lines.append(f"### {i}. [{severity}] {amb['source_text']}")
            lines.append("")
            lines.append(f"- **Type:** {amb['ambiguity_type']}")
            lines.append(f"- **Location:** {amb['location']}")
            lines.append(f"- **Description:** {amb['description']}")
            lines.append(f"- **Priority Score:** {amb['priority']}")
            lines.append("")

    # SOP Conflicts section
    if data["sop_conflicts"]:
        lines.append("## SOP Conflicts")
        lines.append("")
        for i, conflict in enumerate(data["sop_conflicts"], 1):
            severity = conflict["severity"].upper()
            lines.append(f"### {i}. [{severity}] Conflict")
            lines.append("")
            lines.append(f"- **Seed Text:** {conflict['seed_text']}")
            lines.append(f"- **Rule:** {conflict['constraint']['rule_text']}")
            lines.append(f"- **Description:** {conflict['description']}")
            if conflict["resolution_options"]:
                lines.append("- **Resolution Options:**")
                for opt in conflict["resolution_options"]:
                    lines.append(f"  - {opt}")
            lines.append("")

    # Recommendations section
    if data["recommendations"]:
        lines.append("## Recommendations")
        lines.append("")
        for rec in data["recommendations"]:
            lines.append(f"- {rec}")
        lines.append("")

    return "\n".join(lines)


def format_report_rich(
    report: ValidationReport,
    console: Console | None = None,
) -> None:
    """Display validation report using Rich console formatting.

    Args:
        report: The validation report to display.
        console: Rich Console instance (uses default if None).

    Example:
        >>> from rich.console import Console
        >>> console = Console()
        >>> format_report_rich(report, console)
    """
    if console is None:
        console = Console()

    data = report.to_dict()
    metrics = data["quality_metrics"]
    summary = data["summary"]

    # Overall quality score with color coding
    overall_score = metrics["overall_score"]
    if overall_score >= 0.85:
        score_style = "bold green"
        status = "GOOD"
    elif overall_score >= 0.7:
        score_style = "bold yellow"
        status = "MODERATE"
    else:
        score_style = "bold red"
        status = "NEEDS IMPROVEMENT"

    # Summary Panel
    summary_text = (
        f"[bold]Quality Score:[/bold] [{score_style}]{int(overall_score * 100)}% - {status}[/{score_style}]\n\n"
        f"[bold green]Goals:[/bold green] {summary['goals_count']}\n"
        f"[bold blue]Features:[/bold blue] {summary['features_count']}\n"
        f"[bold yellow]Constraints:[/bold yellow] {summary['constraints_count']}\n"
        f"[bold red]Ambiguities:[/bold red] {summary['ambiguity_count']}\n"
        f"[bold red]SOP Conflicts:[/bold red] {summary['sop_conflict_count']}"
    )
    console.print(Panel(summary_text, title="Validation Report Summary", border_style="cyan"))

    # Quality Metrics Table
    metrics_table = Table(title="Quality Metrics", show_header=True, header_style="bold")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Score", justify="center")
    metrics_table.add_column("Weight", justify="center", style="dim")

    # Add rows with color coding
    def score_color(score: float) -> str:
        if score >= 0.85:
            return "green"
        elif score >= 0.7:
            return "yellow"
        else:
            return "red"

    amb_score = metrics["ambiguity_score"]
    sop_score = metrics["sop_score"]
    ext_score = metrics["extraction_score"]

    metrics_table.add_row(
        "Ambiguity Score",
        f"[{score_color(amb_score)}]{int(amb_score * 100)}%[/{score_color(amb_score)}]",
        "30%",
    )
    metrics_table.add_row(
        "SOP Score",
        f"[{score_color(sop_score)}]{int(sop_score * 100)}%[/{score_color(sop_score)}]",
        "30%",
    )
    metrics_table.add_row(
        "Extraction Score",
        f"[{score_color(ext_score)}]{int(ext_score * 100)}%[/{score_color(ext_score)}]",
        "40%",
    )
    metrics_table.add_row(
        "[bold]Overall[/bold]",
        f"[bold {score_color(overall_score)}]{int(overall_score * 100)}%[/bold {score_color(overall_score)}]",
        "100%",
    )

    console.print(metrics_table)
    console.print()

    # Ambiguities Table (if any)
    if data["ambiguities"]:
        amb_table = Table(
            title="Ambiguities Detected",
            show_header=True,
            header_style="bold red",
        )
        amb_table.add_column("#", justify="right", style="dim")
        amb_table.add_column("Priority", justify="center")
        amb_table.add_column("Type", style="cyan")
        amb_table.add_column("Severity", justify="center")
        amb_table.add_column("Description", width=40)

        for i, amb in enumerate(data["ambiguities"], 1):
            severity = amb["severity"]
            severity_style = {"low": "green", "medium": "yellow", "high": "red"}.get(
                severity, "white"
            )
            priority = amb["priority"]
            if priority >= 50:
                priority_label = "[red bold]HIGH[/red bold]"
            elif priority >= 30:
                priority_label = "[yellow]MED[/yellow]"
            else:
                priority_label = "[green]LOW[/green]"

            amb_table.add_row(
                str(i),
                priority_label,
                amb["ambiguity_type"],
                f"[{severity_style}]{severity.upper()}[/{severity_style}]",
                amb["description"][:40] + "..."
                if len(amb["description"]) > 40
                else amb["description"],
            )

        console.print(amb_table)
        console.print()

    # SOP Conflicts Table (if any)
    if data["sop_conflicts"]:
        sop_table = Table(
            title="SOP Conflicts Detected",
            show_header=True,
            header_style="bold red",
        )
        sop_table.add_column("#", justify="right", style="dim")
        sop_table.add_column("Severity", justify="center")
        sop_table.add_column("Category", style="cyan")
        sop_table.add_column("Rule", width=30)
        sop_table.add_column("Conflict", width=30)

        for i, conflict in enumerate(data["sop_conflicts"], 1):
            severity = conflict["severity"]
            if severity == "hard":
                severity_label = "[red bold]HARD[/red bold]"
            else:
                severity_label = "[yellow]SOFT[/yellow]"

            sop_table.add_row(
                str(i),
                severity_label,
                conflict["constraint"]["category"],
                conflict["constraint"]["rule_text"][:30] + "..."
                if len(conflict["constraint"]["rule_text"]) > 30
                else conflict["constraint"]["rule_text"],
                conflict["seed_text"][:30] + "..."
                if len(conflict["seed_text"]) > 30
                else conflict["seed_text"],
            )

        console.print(sop_table)
        console.print()

    # Recommendations Panel (if any)
    if data["recommendations"]:
        rec_text = "\n".join(f"- {rec}" for rec in data["recommendations"])
        console.print(Panel(rec_text, title="Recommendations", border_style="yellow"))
