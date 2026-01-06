"""Gate implementations package.

This package contains specific gate evaluator implementations.
Each gate module registers its evaluator automatically on import.

Available gates:
    - testability: Validates requirements for testability
    - ac_measurability: Validates acceptance criteria for measurability
    - architecture_validation: Validates architectural decisions against principles
    - definition_of_done: Validates code against DoD checklist
    - confidence_scoring: Calculates confidence score for deployable artifacts

Example:
    >>> # Import to register the testability gate
    >>> from yolo_developer.gates.gates import testability
    >>>
    >>> # Or import specific components
    >>> from yolo_developer.gates.gates.testability import (
    ...     VAGUE_TERMS,
    ...     detect_vague_terms,
    ...     has_success_criteria,
    ...     testability_evaluator,
    ... )
    >>>
    >>> # Import AC measurability gate
    >>> from yolo_developer.gates.gates.ac_measurability import (
    ...     SUBJECTIVE_TERMS,
    ...     GWT_PATTERNS,
    ...     detect_subjective_terms,
    ...     has_gwt_structure,
    ...     has_concrete_condition,
    ...     ac_measurability_evaluator,
    ...     generate_improvement_suggestions,
    ... )
    >>>
    >>> # Import architecture validation gate
    >>> from yolo_developer.gates.gates.architecture_validation import (
    ...     TWELVE_FACTOR_PRINCIPLES,
    ...     SECURITY_ANTI_PATTERNS,
    ...     check_twelve_factor_compliance,
    ...     validate_tech_stack,
    ...     detect_security_anti_patterns,
    ...     calculate_compliance_score,
    ...     architecture_validation_evaluator,
    ... )
    >>>
    >>> # Import definition of done gate
    >>> from yolo_developer.gates.gates.definition_of_done import (
    ...     DoDCategory,
    ...     DOD_CHECKLIST_ITEMS,
    ...     SEVERITY_WEIGHTS,
    ...     check_test_presence,
    ...     check_documentation,
    ...     check_code_style,
    ...     check_ac_coverage,
    ...     generate_dod_checklist,
    ...     definition_of_done_evaluator,
    ... )
    >>>
    >>> # Import confidence scoring gate
    >>> from yolo_developer.gates.gates.confidence_scoring import (
    ...     ConfidenceFactor,
    ...     ConfidenceBreakdown,
    ...     DEFAULT_FACTOR_WEIGHTS,
    ...     DEFAULT_CONFIDENCE_THRESHOLD,
    ...     RISK_SEVERITY_IMPACT,
    ...     calculate_coverage_factor,
    ...     calculate_gate_factor,
    ...     calculate_risk_factor,
    ...     calculate_documentation_factor,
    ...     calculate_confidence_score,
    ...     confidence_scoring_evaluator,
    ...     generate_confidence_report,
    ... )
"""

from __future__ import annotations

from yolo_developer.gates.gates.ac_measurability import (
    CONCRETE_CONDITION_PATTERNS,
    GWT_PATTERNS,
    SUBJECTIVE_TERMS,
    ac_measurability_evaluator,
    detect_subjective_terms,
    generate_improvement_suggestions,
    has_concrete_condition,
    has_gwt_structure,
)
from yolo_developer.gates.gates.architecture_validation import (
    SECURITY_ANTI_PATTERNS,
    TWELVE_FACTOR_PRINCIPLES,
    architecture_validation_evaluator,
    calculate_compliance_score,
    check_twelve_factor_compliance,
    detect_security_anti_patterns,
    evaluate_adrs,
    validate_tech_stack,
)
from yolo_developer.gates.gates.confidence_scoring import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FACTOR_WEIGHTS,
    RISK_SEVERITY_IMPACT,
    WEIGHT_SUM_TOLERANCE,
    ConfidenceBreakdown,
    ConfidenceFactor,
    calculate_confidence_score,
    calculate_coverage_factor,
    calculate_documentation_factor,
    calculate_gate_factor,
    calculate_risk_factor,
    confidence_scoring_evaluator,
    generate_confidence_report,
)
from yolo_developer.gates.gates.definition_of_done import (
    DEFAULT_DOD_THRESHOLD,
    DOD_CHECKLIST_ITEMS,
    SEVERITY_WEIGHTS,
    DoDCategory,
    check_ac_coverage,
    check_code_style,
    check_documentation,
    check_test_presence,
    definition_of_done_evaluator,
    generate_dod_checklist,
)
from yolo_developer.gates.gates.testability import (
    VAGUE_TERMS,
    detect_vague_terms,
    has_success_criteria,
    testability_evaluator,
)

__all__ = [
    "CONCRETE_CONDITION_PATTERNS",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_DOD_THRESHOLD",
    "DEFAULT_FACTOR_WEIGHTS",
    "DOD_CHECKLIST_ITEMS",
    "GWT_PATTERNS",
    "RISK_SEVERITY_IMPACT",
    "SECURITY_ANTI_PATTERNS",
    "SEVERITY_WEIGHTS",
    "SUBJECTIVE_TERMS",
    "TWELVE_FACTOR_PRINCIPLES",
    "VAGUE_TERMS",
    "WEIGHT_SUM_TOLERANCE",
    "ConfidenceBreakdown",
    "ConfidenceFactor",
    "DoDCategory",
    "ac_measurability_evaluator",
    "architecture_validation_evaluator",
    "calculate_compliance_score",
    "calculate_confidence_score",
    "calculate_coverage_factor",
    "calculate_documentation_factor",
    "calculate_gate_factor",
    "calculate_risk_factor",
    "check_ac_coverage",
    "check_code_style",
    "check_documentation",
    "check_test_presence",
    "check_twelve_factor_compliance",
    "confidence_scoring_evaluator",
    "definition_of_done_evaluator",
    "detect_security_anti_patterns",
    "detect_subjective_terms",
    "detect_vague_terms",
    "evaluate_adrs",
    "generate_confidence_report",
    "generate_dod_checklist",
    "generate_improvement_suggestions",
    "has_concrete_condition",
    "has_gwt_structure",
    "has_success_criteria",
    "testability_evaluator",
    "validate_tech_stack",
]
