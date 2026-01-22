"""Pattern matching module for codebase pattern consistency (Story 7.8).

This module provides functionality to match design decisions against
learned codebase patterns, ensuring consistency with existing code style,
naming conventions, and architectural patterns.

Key Components:
- Pattern retrieval from ChromaPatternStore (Task 2)
- Naming convention checking (Task 3)
- Architectural style checking (Task 4)
- Deviation detection and justification (Task 5)
- Pass/fail decision logic (Task 6)
- LLM-powered pattern analysis (Task 7)
- Main orchestration function (Task 8)

Example:
    >>> from yolo_developer.agents.architect.pattern_matcher import run_pattern_matching
    >>> from yolo_developer.agents.architect.types import DesignDecision
    >>>
    >>> result = await run_pattern_matching(decisions)
    >>> result.overall_pass
    True

Architecture:
    - Uses ChromaPatternStore from memory layer (Epic 2) for pattern retrieval
    - Follows ADR-001: Frozen dataclasses for immutable results
    - Follows ADR-003: Uses litellm for LLM calls
    - Uses tenacity for retry logic per ADR-007
    - Integrates with architect_node after ATAM review (Story 7.7)

References:
    - FR56: Architect Agent can ensure design patterns match existing codebase
    - Story 7.8: Pattern Matching to Codebase
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import structlog
from litellm import acompletion
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from yolo_developer.config.schema import LLM_CHEAP_MODEL_DEFAULT
from yolo_developer.agents.architect.types import (
    DesignDecision,
    PatternCheckSeverity,
    PatternDeviation,
    PatternMatchingResult,
    PatternViolation,
)
from yolo_developer.memory.pattern_store import ChromaPatternStore
from yolo_developer.memory.patterns import CodePattern, PatternType

logger = structlog.get_logger(__name__)

# All pattern types to check
ALL_PATTERN_TYPES: tuple[PatternType, ...] = (
    PatternType.NAMING_FUNCTION,
    PatternType.NAMING_CLASS,
    PatternType.NAMING_VARIABLE,
    PatternType.NAMING_MODULE,
    PatternType.STRUCTURE_DIRECTORY,
    PatternType.STRUCTURE_FILE,
    PatternType.IMPORT_STYLE,
    PatternType.DESIGN_PATTERN,
)


async def _get_learned_patterns(
    pattern_store: ChromaPatternStore | None,
) -> list[CodePattern]:
    """Retrieve learned patterns from the memory layer.

    Queries the pattern store for all pattern types and aggregates
    the results. Handles the case where the store is not available.

    Args:
        pattern_store: ChromaPatternStore instance or None if unavailable.

    Returns:
        List of CodePattern instances. Empty list if store is None or on error.

    Example:
        >>> patterns = await _get_learned_patterns(store)
        >>> len(patterns)
        15
    """
    if pattern_store is None:
        logger.debug("pattern_store_not_available", action="returning_empty_list")
        return []

    all_patterns: list[CodePattern] = []

    for pattern_type in ALL_PATTERN_TYPES:
        try:
            patterns = await pattern_store.get_patterns_by_type(pattern_type)
            all_patterns.extend(patterns)
            logger.debug(
                "patterns_retrieved",
                pattern_type=pattern_type.value,
                count=len(patterns),
            )
        except Exception as e:
            logger.warning(
                "pattern_retrieval_failed",
                pattern_type=pattern_type.value,
                error=str(e),
                exc_info=True,
            )
            # Continue with other pattern types, don't fail completely
            continue

    logger.info(
        "pattern_retrieval_complete",
        total_patterns=len(all_patterns),
        pattern_types_queried=len(ALL_PATTERN_TYPES),
    )

    return all_patterns


# =============================================================================
# Task 3: Naming Convention Checking
# =============================================================================

# Naming pattern detectors
NAMING_PATTERNS = {
    "snake_case": re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$"),
    "camelCase": re.compile(r"^[a-z][a-zA-Z0-9]*$"),
    "PascalCase": re.compile(r"^[A-Z][a-zA-Z0-9]*$"),
    "SCREAMING_SNAKE_CASE": re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$"),
    "kebab-case": re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$"),
}


def _detect_naming_style(name: str) -> str | None:
    """Detect the naming style of a given identifier.

    Args:
        name: The identifier to analyze.

    Returns:
        The detected naming style or None if no match.
    """
    for style, pattern in NAMING_PATTERNS.items():
        if pattern.match(name):
            return style
    return None


def _extract_identifiers_from_decision(decision: DesignDecision) -> list[tuple[str, str]]:
    """Extract identifiers and their context from a design decision.

    Args:
        decision: The design decision to analyze.

    Returns:
        List of (identifier, context) tuples.
    """
    identifiers: list[tuple[str, str]] = []
    text = f"{decision.description} {decision.rationale}"

    # Look for common code patterns in the text
    # Function/method names: function_name(), methodName()
    func_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
    for match in func_pattern.finditer(text):
        name = match.group(1)
        if len(name) > 2 and name not in {"if", "for", "while", "return", "async", "await"}:
            identifiers.append((name, "function"))

    # Class names: ClassName, class ClassName
    class_pattern = re.compile(r"\b(?:class\s+)?([A-Z][a-zA-Z0-9]*)\b")
    for match in class_pattern.finditer(text):
        name = match.group(1)
        if len(name) > 2:
            identifiers.append((name, "class"))

    return identifiers


def _check_naming_conventions(
    decisions: list[DesignDecision],
    patterns: list[CodePattern],
) -> tuple[list[PatternViolation], float]:
    """Check naming conventions in design decisions against learned patterns.

    Args:
        decisions: List of design decisions to analyze.
        patterns: List of learned codebase patterns.

    Returns:
        Tuple of (list of violations, confidence score).
    """
    violations: list[PatternViolation] = []

    # Extract naming patterns from learned patterns
    naming_patterns: dict[str, CodePattern] = {}
    for pattern in patterns:
        if pattern.pattern_type.value.startswith("naming_"):
            naming_patterns[pattern.pattern_type.value] = pattern

    if not naming_patterns:
        logger.debug("no_naming_patterns_learned", action="skipping_naming_check")
        return [], 1.0  # No patterns to check against, assume conformance

    total_checks = 0
    conforming = 0

    for decision in decisions:
        identifiers = _extract_identifiers_from_decision(decision)

        for name, context in identifiers:
            detected_style = _detect_naming_style(name)
            total_checks += 1

            # Determine expected pattern based on context
            if context == "function":
                expected_pattern = naming_patterns.get("naming_function")
            elif context == "class":
                expected_pattern = naming_patterns.get("naming_class")
            else:
                expected_pattern = naming_patterns.get("naming_variable")

            if expected_pattern and detected_style:
                expected_style = expected_pattern.value
                if detected_style == expected_style:
                    conforming += 1
                else:
                    severity: PatternCheckSeverity = "high" if context == "function" else "medium"
                    violations.append(
                        PatternViolation(
                            pattern_type=f"naming_{context}",
                            expected=expected_style,
                            actual=detected_style,
                            file_context=decision.description[:50],
                            severity=severity,
                            justification=None,
                        )
                    )
            else:
                # Can't determine, count as conforming
                conforming += 1

    confidence = conforming / total_checks if total_checks > 0 else 1.0

    logger.debug(
        "naming_check_complete",
        violations_count=len(violations),
        total_checks=total_checks,
        confidence=confidence,
    )

    return violations, confidence


# =============================================================================
# Task 4: Architectural Style Checking
# =============================================================================


def _check_architectural_style(
    decisions: list[DesignDecision],
    patterns: list[CodePattern],
) -> list[PatternViolation]:
    """Check architectural style consistency in design decisions.

    Args:
        decisions: List of design decisions to analyze.
        patterns: List of learned codebase patterns.

    Returns:
        List of architectural style violations.
    """
    violations: list[PatternViolation] = []

    # Extract structure and design patterns from learned patterns
    structure_patterns: dict[str, CodePattern] = {}
    for pattern in patterns:
        if pattern.pattern_type in (
            PatternType.STRUCTURE_DIRECTORY,
            PatternType.STRUCTURE_FILE,
            PatternType.IMPORT_STYLE,
            PatternType.DESIGN_PATTERN,
        ):
            structure_patterns[pattern.pattern_type.value] = pattern

    if not structure_patterns:
        logger.debug("no_structure_patterns_learned", action="skipping_style_check")
        return []

    for decision in decisions:
        text = f"{decision.description} {decision.rationale}".lower()

        # Check for import style inconsistencies
        import_pattern = structure_patterns.get("import_style")
        if import_pattern:
            if "relative import" in text and import_pattern.value == "absolute":
                violations.append(
                    PatternViolation(
                        pattern_type="import_style",
                        expected="absolute imports",
                        actual="relative imports",
                        file_context=decision.description[:50],
                        severity="low",
                        justification=None,
                    )
                )
            elif "absolute import" in text and import_pattern.value == "relative":
                violations.append(
                    PatternViolation(
                        pattern_type="import_style",
                        expected="relative imports",
                        actual="absolute imports",
                        file_context=decision.description[:50],
                        severity="low",
                        justification=None,
                    )
                )

        # Check for design pattern inconsistencies
        design_pattern = structure_patterns.get("design_pattern")
        if design_pattern:
            known_patterns = ["factory", "builder", "singleton", "repository", "adapter"]
            for pattern_name in known_patterns:
                if pattern_name in text and pattern_name not in design_pattern.value.lower():
                    # A different pattern is being used
                    violations.append(
                        PatternViolation(
                            pattern_type="design_pattern",
                            expected=design_pattern.value,
                            actual=pattern_name.capitalize(),
                            file_context=decision.description[:50],
                            severity="medium",
                            justification=None,
                        )
                    )
                    break

    logger.debug(
        "architectural_style_check_complete",
        violations_count=len(violations),
    )

    return violations


# =============================================================================
# Task 5: Deviation Detection and Justification
# =============================================================================


def _detect_pattern_deviations(
    violations: list[PatternViolation],
    decisions: list[DesignDecision],
) -> list[PatternDeviation]:
    """Detect pattern deviations and check for justifications.

    For each violation, checks if any design decision rationale
    justifies the deviation.

    Args:
        violations: List of pattern violations to analyze.
        decisions: List of design decisions with rationales.

    Returns:
        List of pattern deviations with justification status.
    """
    deviations: list[PatternDeviation] = []

    # Collect all rationales
    rationales = " ".join(d.rationale.lower() for d in decisions)

    for violation in violations:
        # Check if rationale justifies this deviation
        justification_keywords = [
            "intentional",
            "deliberately",
            "by design",
            "required for",
            "necessary for",
            "chosen because",
            "prefer",
            "better for",
            "consistency with",
            "compatibility",
        ]

        found_justification = ""
        is_justified = False

        for keyword in justification_keywords:
            if keyword in rationales and violation.actual.lower() in rationales:
                is_justified = True
                # Extract surrounding context as justification
                idx = rationales.find(keyword)
                found_justification = rationales[max(0, idx - 20) : idx + 80].strip()
                break

        deviation = PatternDeviation(
            pattern_type=violation.pattern_type,
            standard_pattern=violation.expected,
            proposed_pattern=violation.actual,
            justification=found_justification
            if found_justification
            else violation.justification or "",
            is_justified=is_justified,
            severity=violation.severity,
        )
        deviations.append(deviation)

    logger.debug(
        "deviation_detection_complete",
        total_deviations=len(deviations),
        justified_count=sum(1 for d in deviations if d.is_justified),
    )

    return deviations


# =============================================================================
# Task 6: Pass/Fail Decision Logic
# =============================================================================

# Default thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_MAX_HIGH_SEVERITY_VIOLATIONS = 3


def _make_pattern_decision(
    violations: list[PatternViolation],
    deviations: list[PatternDeviation],
    confidence: float,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_high_severity: int = DEFAULT_MAX_HIGH_SEVERITY_VIOLATIONS,
) -> tuple[bool, list[str]]:
    """Make pass/fail decision based on pattern analysis.

    Args:
        violations: List of pattern violations.
        deviations: List of pattern deviations.
        confidence: Calculated confidence score.
        confidence_threshold: Minimum confidence to pass.
        max_high_severity: Maximum allowed high-severity violations.

    Returns:
        Tuple of (pass decision, list of failure reasons).
    """
    failure_reasons: list[str] = []

    # Check for critical unjustified deviations
    critical_unjustified = [
        d for d in deviations if d.severity == "critical" and not d.is_justified
    ]
    if critical_unjustified:
        failure_reasons.append(f"Found {len(critical_unjustified)} critical unjustified deviations")

    # Check confidence threshold
    if confidence < confidence_threshold:
        failure_reasons.append(
            f"Confidence score {confidence:.2f} below threshold {confidence_threshold:.2f}"
        )

    # Check high-severity violation count
    high_severity_violations = [v for v in violations if v.severity in ("critical", "high")]
    if len(high_severity_violations) > max_high_severity:
        failure_reasons.append(
            f"Found {len(high_severity_violations)} high-severity violations "
            f"(max allowed: {max_high_severity})"
        )

    overall_pass = len(failure_reasons) == 0

    logger.debug(
        "pattern_decision_made",
        overall_pass=overall_pass,
        failure_reasons=failure_reasons,
        confidence=confidence,
    )

    return overall_pass, failure_reasons


def _generate_recommendations(
    violations: list[PatternViolation],
    deviations: list[PatternDeviation],
) -> list[str]:
    """Generate recommendations for improving pattern conformance.

    Args:
        violations: List of pattern violations.
        deviations: List of pattern deviations.

    Returns:
        List of recommendation strings.
    """
    recommendations: list[str] = []

    # Recommend fixes for naming violations
    naming_violations = [v for v in violations if v.pattern_type.startswith("naming_")]
    if naming_violations:
        examples = [f"'{v.actual}' → '{v.expected}'" for v in naming_violations[:3]]
        recommendations.append(
            f"Consider renaming identifiers to match codebase conventions: {', '.join(examples)}"
        )

    # Recommend justifications for unjustified deviations
    unjustified = [d for d in deviations if not d.is_justified]
    if unjustified:
        for dev in unjustified[:2]:
            recommendations.append(
                f"Add justification for using {dev.proposed_pattern} instead of {dev.standard_pattern}"
            )

    return recommendations


# =============================================================================
# Task 7: LLM-Powered Pattern Analysis
# =============================================================================

PATTERN_MATCHING_PROMPT = """Analyze the following design decisions against existing codebase patterns.

Design Decisions:
{design_decisions}

Learned Codebase Patterns:
{patterns}

For each design decision:
1. Check naming conventions against learned patterns
2. Check architectural style consistency
3. Identify any pattern deviations
4. For deviations, check if the decision rationale justifies them

Respond in JSON format only (no markdown, no code blocks):
{{
  "overall_pass": true,
  "confidence": 0.85,
  "patterns_checked": ["naming_function", "naming_class", "structure_directory"],
  "violations": [
    {{
      "pattern_type": "naming_function",
      "expected": "snake_case",
      "actual": "camelCase",
      "file_context": "src/myModule.py",
      "severity": "high",
      "justification": null
    }}
  ],
  "deviations": [
    {{
      "pattern_type": "design_pattern",
      "standard_pattern": "Factory pattern",
      "proposed_pattern": "Builder pattern",
      "justification": "Builder provides better configuration flexibility",
      "is_justified": true,
      "severity": "medium"
    }}
  ],
  "recommendations": [
    "Consider renaming getUserData to get_user_data for consistency"
  ],
  "summary": "Design mostly conforms with 2 minor deviations justified"
}}
"""


def _format_decisions_for_llm(decisions: list[DesignDecision]) -> str:
    """Format design decisions for LLM prompt."""
    formatted = []
    for d in decisions:
        formatted.append(
            f"- ID: {d.id}\n"
            f"  Type: {d.decision_type}\n"
            f"  Description: {d.description}\n"
            f"  Rationale: {d.rationale}"
        )
    return "\n".join(formatted)


def _format_patterns_for_llm(patterns: list[CodePattern]) -> str:
    """Format patterns for LLM prompt."""
    formatted = []
    for p in patterns:
        examples = ", ".join(p.examples[:3]) if p.examples else "none"
        formatted.append(
            f"- {p.pattern_type.value}: {p.name} = {p.value} "
            f"(confidence: {p.confidence:.2f}, examples: {examples})"
        )
    return "\n".join(formatted) if formatted else "No patterns learned yet"


def _parse_llm_response(response_text: str) -> dict[str, Any] | None:
    """Parse LLM JSON response safely."""
    try:
        # Try to extract JSON from the response
        # Sometimes LLM wraps JSON in code blocks
        text = response_text.strip()
        if text.startswith("```"):
            # Extract from code block
            lines = text.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        parsed: dict[str, Any] = json.loads(text)
        return parsed
    except json.JSONDecodeError:
        logger.warning("llm_response_json_parse_failed", response_preview=response_text[:100])
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _analyze_patterns_with_llm(
    decisions: list[DesignDecision],
    patterns: list[CodePattern],
) -> PatternMatchingResult | None:
    """LLM-powered pattern analysis with retry.

    Args:
        decisions: List of design decisions to analyze.
        patterns: List of learned codebase patterns.

    Returns:
        PatternMatchingResult if successful, None to trigger fallback.
    """
    model = os.environ.get("YOLO_LLM__ROUTINE_MODEL", LLM_CHEAP_MODEL_DEFAULT)

    prompt = PATTERN_MATCHING_PROMPT.format(
        design_decisions=_format_decisions_for_llm(decisions),
        patterns=_format_patterns_for_llm(patterns),
    )

    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        content = response.choices[0].message.content
        if not content:
            logger.warning("llm_returned_empty_content")
            return None

        parsed = _parse_llm_response(content)
        if not parsed:
            return None

        # Convert parsed response to typed objects
        violations = tuple(
            PatternViolation(
                pattern_type=v.get("pattern_type", "unknown"),
                expected=v.get("expected", ""),
                actual=v.get("actual", ""),
                file_context=v.get("file_context", ""),
                severity=v.get("severity", "medium"),
                justification=v.get("justification"),
            )
            for v in parsed.get("violations", [])
        )

        deviations = tuple(
            PatternDeviation(
                pattern_type=d.get("pattern_type", "unknown"),
                standard_pattern=d.get("standard_pattern", ""),
                proposed_pattern=d.get("proposed_pattern", ""),
                justification=d.get("justification", ""),
                is_justified=d.get("is_justified", False),
                severity=d.get("severity", "medium"),
            )
            for d in parsed.get("deviations", [])
        )

        return PatternMatchingResult(
            overall_pass=parsed.get("overall_pass", True),
            confidence=parsed.get("confidence", 0.8),
            patterns_checked=tuple(parsed.get("patterns_checked", [])),
            violations=violations,
            deviations=deviations,
            recommendations=tuple(parsed.get("recommendations", [])),
            summary=parsed.get("summary", "LLM analysis complete"),
        )

    except Exception as e:
        logger.warning("llm_pattern_analysis_failed", error=str(e), exc_info=True)
        return None


# =============================================================================
# Task 8: Main Pattern Matching Function
# =============================================================================


async def run_pattern_matching(
    decisions: list[DesignDecision],
    pattern_store: ChromaPatternStore | None = None,
) -> PatternMatchingResult:
    """Main entry point for pattern matching analysis.

    Orchestrates pattern retrieval, rule-based checking, and optional
    LLM analysis with fallback.

    Args:
        decisions: List of design decisions to analyze.
        pattern_store: Optional ChromaPatternStore for learned patterns.

    Returns:
        PatternMatchingResult with analysis outcome.

    Example:
        >>> result = await run_pattern_matching(decisions, pattern_store)
        >>> result.overall_pass
        True
    """
    logger.info(
        "pattern_matching_start",
        decision_count=len(decisions),
        has_pattern_store=pattern_store is not None,
    )

    # Handle empty decisions
    if not decisions:
        return PatternMatchingResult(
            overall_pass=True,
            confidence=1.0,
            patterns_checked=(),
            violations=(),
            deviations=(),
            recommendations=(),
            summary="No design decisions to analyze",
        )

    # Retrieve learned patterns
    patterns = await _get_learned_patterns(pattern_store)

    # Try LLM analysis first (if patterns exist)
    if patterns:
        try:
            llm_result = await _analyze_patterns_with_llm(decisions, patterns)
            if llm_result is not None:
                logger.info(
                    "pattern_matching_complete",
                    method="llm",
                    overall_pass=llm_result.overall_pass,
                    confidence=llm_result.confidence,
                )
                return llm_result
        except Exception as e:
            logger.warning("llm_analysis_failed_using_rule_based", error=str(e))

    # Fall back to rule-based analysis
    return await _rule_based_pattern_matching(decisions, patterns)


async def _rule_based_pattern_matching(
    decisions: list[DesignDecision],
    patterns: list[CodePattern],
) -> PatternMatchingResult:
    """Rule-based pattern matching fallback.

    Args:
        decisions: List of design decisions to analyze.
        patterns: List of learned codebase patterns.

    Returns:
        PatternMatchingResult with analysis outcome.
    """
    # Check naming conventions
    naming_violations, naming_confidence = _check_naming_conventions(decisions, patterns)

    # Check architectural style
    style_violations = _check_architectural_style(decisions, patterns)

    # Combine violations
    all_violations = naming_violations + style_violations

    # Detect deviations
    deviations = _detect_pattern_deviations(all_violations, decisions)

    # Calculate overall confidence
    if patterns:
        # Weight naming more heavily
        confidence = naming_confidence * 0.7 + (1.0 - len(style_violations) * 0.1) * 0.3
        confidence = max(0.0, min(1.0, confidence))
    else:
        # No patterns to check against
        confidence = 1.0

    # Make pass/fail decision
    overall_pass, failure_reasons = _make_pattern_decision(all_violations, deviations, confidence)

    # Generate recommendations
    recommendations = _generate_recommendations(all_violations, deviations)

    # Determine which patterns were checked
    patterns_checked = list({p.pattern_type.value for p in patterns})

    # Generate summary
    if overall_pass:
        if not patterns:
            summary = "No learned patterns to check against. Passing by default."
        elif not all_violations:
            summary = "Design conforms to all established codebase patterns."
        else:
            summary = (
                f"Design passes with {len(all_violations)} minor violations. "
                f"All deviations are justified."
            )
    else:
        summary = f"Design fails pattern matching: {'; '.join(failure_reasons)}"

    result = PatternMatchingResult(
        overall_pass=overall_pass,
        confidence=confidence,
        patterns_checked=tuple(patterns_checked),
        violations=tuple(all_violations),
        deviations=tuple(deviations),
        recommendations=tuple(recommendations),
        summary=summary,
    )

    logger.info(
        "pattern_matching_complete",
        method="rule_based",
        overall_pass=result.overall_pass,
        confidence=result.confidence,
        violation_count=len(all_violations),
        deviation_count=len(deviations),
    )

    return result
