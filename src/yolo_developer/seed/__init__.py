"""Seed parsing module for yolo-developer.

This module provides functionality for parsing natural language seed documents
and extracting structured components (goals, features, constraints).

Example:
    >>> from yolo_developer.seed import parse_seed, SeedSource
    >>>
    >>> # Parse a simple text seed
    >>> result = await parse_seed("Build an e-commerce platform with auth")
    >>> print(f"Found {result.goal_count} goals, {result.feature_count} features")
    >>>
    >>> # Parse from a file
    >>> with open("requirements.md") as f:
    ...     content = f.read()
    >>> result = await parse_seed(content, filename="requirements.md")
    >>>
    >>> # Parse with ambiguity detection (Story 4.3)
    >>> result = await parse_seed(content, detect_ambiguities=True)
    >>> if result.has_ambiguities:
    ...     for amb in result.ambiguities:
    ...         print(f"- {amb.description}")
    >>>
    >>> # Use question prioritization (Story 4.4)
    >>> from yolo_developer.seed import prioritize_questions, calculate_question_priority
    >>> sorted_ambs = prioritize_questions(list(result.ambiguities))
    >>> for amb in sorted_ambs:
    ...     score = calculate_question_priority(amb)
    ...     print(f"Priority {score}: {amb.description}")
    >>>
    >>> # Validate question quality (Story 4.4)
    >>> from yolo_developer.seed import validate_question_quality
    >>> is_valid, suggestions = validate_question_quality("What response time is needed?")
    >>>
    >>> # SOP constraint validation (Story 4.5)
    >>> from yolo_developer.seed import (
    ...     InMemorySOPStore,
    ...     SOPConstraint,
    ...     SOPCategory,
    ...     ConflictSeverity,
    ...     validate_against_sop,
    ... )
    >>> store = InMemorySOPStore()
    >>> await store.add_constraint(SOPConstraint(
    ...     id="arch-001",
    ...     rule_text="All APIs must use REST conventions",
    ...     category=SOPCategory.ARCHITECTURE,
    ...     source="architecture.md",
    ...     severity=ConflictSeverity.HARD,
    ... ))
    >>> result = await parse_seed(content, validate_sop=True, sop_store=store)
    >>> if not result.sop_passed:
    ...     print(f"Found {len(result.sop_validation.conflicts)} SOP conflicts")
    >>>
    >>> # Generate validation reports (Story 4.6)
    >>> from yolo_developer.seed import (
    ...     generate_validation_report,
    ...     format_report_json,
    ...     format_report_markdown,
    ...     format_report_rich,
    ...     calculate_quality_score,
    ...     ValidationReport,
    ...     QualityMetrics,
    ...     ReportFormat,
    ... )
    >>> report = generate_validation_report(result, source_file="requirements.md")
    >>> print(f"Quality Score: {report.quality_metrics.overall_score:.0%}")
    >>> json_output = format_report_json(report)  # JSON format
    >>> md_output = format_report_markdown(report)  # Markdown format
    >>> format_report_rich(report)  # Rich console output
"""

from __future__ import annotations

from yolo_developer.seed.ambiguity import (
    Ambiguity,
    AmbiguityResult,
    AmbiguitySeverity,
    AmbiguityType,
    AnswerFormat,
    Resolution,
    ResolutionPrompt,
    SeedContext,
    calculate_ambiguity_confidence,
    calculate_question_priority,
    detect_ambiguities,
    prioritize_questions,
    validate_question_quality,
)
from yolo_developer.seed.api import parse_seed
from yolo_developer.seed.parser import (
    LLMSeedParser,
    SeedParser,
    detect_source_format,
    normalize_content,
)
from yolo_developer.seed.report import (
    QualityMetrics,
    ReportFormat,
    ValidationReport,
    calculate_quality_score,
    format_report_json,
    format_report_markdown,
    format_report_rich,
    generate_validation_report,
)
from yolo_developer.seed.sop import (
    ConflictSeverity,
    InMemorySOPStore,
    SOPCategory,
    SOPConflict,
    SOPConstraint,
    SOPStore,
    SOPValidationResult,
    generate_constraint_id,
    validate_against_sop,
)
from yolo_developer.seed.types import (
    ComponentType,
    ConstraintCategory,
    SeedComponent,
    SeedConstraint,
    SeedFeature,
    SeedGoal,
    SeedParseResult,
    SeedSource,
)

__all__ = [
    # Types
    "Ambiguity",
    "AmbiguityResult",
    "AmbiguitySeverity",
    "AmbiguityType",
    "AnswerFormat",  # Story 4.4
    "ComponentType",
    "ConflictSeverity",  # Story 4.5
    "ConstraintCategory",
    "InMemorySOPStore",  # Story 4.5
    "LLMSeedParser",
    "QualityMetrics",  # Story 4.6
    "ReportFormat",  # Story 4.6
    "Resolution",
    "ResolutionPrompt",
    "SOPCategory",  # Story 4.5
    "SOPConflict",  # Story 4.5
    "SOPConstraint",  # Story 4.5
    "SOPStore",  # Story 4.5
    "SOPValidationResult",  # Story 4.5
    "SeedComponent",
    "SeedConstraint",
    "SeedContext",
    "SeedFeature",
    "SeedGoal",
    "SeedParseResult",
    "SeedParser",
    "SeedSource",
    "ValidationReport",  # Story 4.6
    # Functions
    "calculate_ambiguity_confidence",
    "calculate_quality_score",  # Story 4.6
    "calculate_question_priority",  # Story 4.4
    "detect_ambiguities",
    "detect_source_format",
    "format_report_json",  # Story 4.6
    "format_report_markdown",  # Story 4.6
    "format_report_rich",  # Story 4.6
    "generate_constraint_id",  # Story 4.5
    "generate_validation_report",  # Story 4.6
    "normalize_content",
    "parse_seed",
    "prioritize_questions",  # Story 4.4
    "validate_against_sop",  # Story 4.5
    "validate_question_quality",  # Story 4.4
]
