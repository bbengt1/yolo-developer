"""Gate implementations package.

This package contains specific gate evaluator implementations.
Each gate module registers its evaluator automatically on import.

Available gates:
    - testability: Validates requirements for testability

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
"""

from __future__ import annotations

from yolo_developer.gates.gates.testability import (
    VAGUE_TERMS,
    TestabilityIssue,
    detect_vague_terms,
    generate_testability_report,
    has_success_criteria,
    testability_evaluator,
)

__all__ = [
    "VAGUE_TERMS",
    "TestabilityIssue",
    "detect_vague_terms",
    "generate_testability_report",
    "has_success_criteria",
    "testability_evaluator",
]
