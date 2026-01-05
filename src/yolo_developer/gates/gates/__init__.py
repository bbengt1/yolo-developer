"""Gate implementations package.

This package contains specific gate evaluator implementations.
Each gate module registers its evaluator automatically on import.

Available gates:
    - testability: Validates requirements for testability
    - ac_measurability: Validates acceptance criteria for measurability
    - architecture_validation: Validates architectural decisions against principles
    - definition_of_done: Validates code against DoD checklist

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
    >>>
    >>> # Import architecture validation gate
    >>> from yolo_developer.gates.gates.architecture_validation import (
    ...     ArchitectureIssue,
    ...     TWELVE_FACTOR_PRINCIPLES,
    ...     SECURITY_ANTI_PATTERNS,
    ...     check_twelve_factor_compliance,
    ...     validate_tech_stack,
    ...     detect_security_anti_patterns,
    ...     calculate_compliance_score,
    ...     architecture_validation_evaluator,
    ...     generate_architecture_report,
    ... )
    >>>
    >>> # Import definition of done gate
    >>> from yolo_developer.gates.gates.definition_of_done import (
    ...     DoDIssue,
    ...     DoDCategory,
    ...     DOD_CHECKLIST_ITEMS,
    ...     SEVERITY_WEIGHTS,
    ...     check_test_presence,
    ...     check_documentation,
    ...     check_code_style,
    ...     check_ac_coverage,
    ...     generate_dod_checklist,
    ...     definition_of_done_evaluator,
    ...     generate_dod_report,
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
from yolo_developer.gates.gates.architecture_validation import (
    SECURITY_ANTI_PATTERNS,
    TWELVE_FACTOR_PRINCIPLES,
    ArchitectureIssue,
    architecture_validation_evaluator,
    calculate_compliance_score,
    check_twelve_factor_compliance,
    detect_security_anti_patterns,
    evaluate_adrs,
    generate_architecture_report,
    validate_tech_stack,
)
from yolo_developer.gates.gates.definition_of_done import (
    DEFAULT_DOD_THRESHOLD,
    DOD_CHECKLIST_ITEMS,
    SEVERITY_WEIGHTS,
    DoDCategory,
    DoDIssue,
    check_ac_coverage,
    check_code_style,
    check_documentation,
    check_test_presence,
    definition_of_done_evaluator,
    generate_dod_checklist,
    generate_dod_report,
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
    "DEFAULT_DOD_THRESHOLD",
    "DOD_CHECKLIST_ITEMS",
    "GWT_PATTERNS",
    "SECURITY_ANTI_PATTERNS",
    "SEVERITY_WEIGHTS",
    "SUBJECTIVE_TERMS",
    "TWELVE_FACTOR_PRINCIPLES",
    "VAGUE_TERMS",
    "ACMeasurabilityIssue",
    "ArchitectureIssue",
    "DoDCategory",
    "DoDIssue",
    "TestabilityIssue",
    "ac_measurability_evaluator",
    "architecture_validation_evaluator",
    "calculate_compliance_score",
    "check_ac_coverage",
    "check_code_style",
    "check_documentation",
    "check_test_presence",
    "check_twelve_factor_compliance",
    "definition_of_done_evaluator",
    "detect_security_anti_patterns",
    "detect_subjective_terms",
    "detect_vague_terms",
    "evaluate_adrs",
    "generate_ac_measurability_report",
    "generate_architecture_report",
    "generate_dod_checklist",
    "generate_dod_report",
    "generate_improvement_suggestions",
    "generate_testability_report",
    "has_concrete_condition",
    "has_gwt_structure",
    "has_success_criteria",
    "testability_evaluator",
    "validate_tech_stack",
]
