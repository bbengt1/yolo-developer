"""Gate implementations package.

This package contains specific gate evaluator implementations.
Each gate module registers its evaluator automatically on import.

Available gates:
    - testability: Validates requirements for testability
    - ac_measurability: Validates acceptance criteria for measurability

Example:
    >>> # Import to register the testability gate
    >>> from yolo_developer.gates.gates import testability
    >>>
    >>> # Or import specific components
    >>> from yolo_developer.gates.gates.testability import (
    ...     TestabilityIssue,
    ...     VAGUE_TERMS,
    ...     detect_vague_terms,
    ...     has_success_criteria,
    ...     testability_evaluator,
    ...     generate_testability_report,
    ... )
    >>>
    >>> # Import AC measurability gate
    >>> from yolo_developer.gates.gates.ac_measurability import (
    ...     ACMeasurabilityIssue,
    ...     SUBJECTIVE_TERMS,
    ...     GWT_PATTERNS,
    ...     detect_subjective_terms,
    ...     has_gwt_structure,
    ...     has_concrete_condition,
    ...     ac_measurability_evaluator,
    ...     generate_ac_measurability_report,
    ... )
"""

from __future__ import annotations

from yolo_developer.gates.gates.ac_measurability import (
    CONCRETE_CONDITION_PATTERNS,
    GWT_PATTERNS,
    SUBJECTIVE_TERMS,
    ACMeasurabilityIssue,
    ac_measurability_evaluator,
    detect_subjective_terms,
    generate_ac_measurability_report,
    generate_improvement_suggestions,
    has_concrete_condition,
    has_gwt_structure,
)
from yolo_developer.gates.gates.testability import (
    VAGUE_TERMS,
    TestabilityIssue,
    detect_vague_terms,
    generate_testability_report,
    has_success_criteria,
    testability_evaluator,
)

__all__ = [
    "CONCRETE_CONDITION_PATTERNS",
    "GWT_PATTERNS",
    "SUBJECTIVE_TERMS",
    "VAGUE_TERMS",
    "ACMeasurabilityIssue",
    "TestabilityIssue",
    "ac_measurability_evaluator",
    "detect_subjective_terms",
    "detect_vague_terms",
    "generate_ac_measurability_report",
    "generate_improvement_suggestions",
    "generate_testability_report",
    "has_concrete_condition",
    "has_gwt_structure",
    "has_success_criteria",
    "testability_evaluator",
]
