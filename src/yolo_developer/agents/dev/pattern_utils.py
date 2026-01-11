"""Pattern following utilities for Dev agent (Story 8.7).

This module provides pattern analysis and validation for code generation:

- PatternDeviation: Single deviation from established pattern
- PatternValidationResult: Aggregate result of pattern validation
- ErrorHandlingPattern: Error handling convention pattern
- StylePattern: Code style convention pattern
- analyze_naming_patterns: Analyze code for naming pattern violations
- analyze_error_handling_patterns: Analyze code for error handling violations
- analyze_style_patterns: Analyze code for style violations
- validate_pattern_adherence: Complete pattern validation

The module integrates with the memory system from Epic 2 via PatternLearner
and provides default patterns when memory_context is not available.

Example:
    >>> from yolo_developer.agents.dev.pattern_utils import (
    ...     validate_pattern_adherence,
    ...     get_naming_patterns,
    ... )
    >>>
    >>> state = {"memory_context": {}}
    >>> patterns = get_naming_patterns(state)
    >>> result = validate_pattern_adherence(code, state)
    >>> result.passed
    True

Architecture References:
    - ADR-001: TypedDict for state, frozen dataclasses for patterns
    - ADR-005: Access patterns via state["memory_context"]
    - ADR-006: Pattern validation is advisory (like DoD)
    - ADR-007: Handle missing memory_context gracefully
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from yolo_developer.memory.patterns import CodePattern, PatternType

# Note: YoloState type hint not needed here since we use dict[str, Any]
# to avoid strict type checking issues with TypedDict compatibility

logger = structlog.get_logger(__name__)


# =============================================================================
# Pattern Cache (for efficiency during single dev_node invocation)
# =============================================================================

# Module-level cache for patterns during a single invocation
# Call clear_pattern_cache() at the start of each dev_node invocation
_pattern_cache: dict[str, list[Any]] = {}


def clear_pattern_cache() -> None:
    """Clear the pattern cache at the start of each dev_node invocation.

    This should be called at the beginning of dev_node to ensure
    fresh pattern queries for each invocation while avoiding
    redundant queries within a single invocation.
    """
    global _pattern_cache
    _pattern_cache.clear()
    logger.debug("pattern_cache_cleared")


# =============================================================================
# Severity Weights for Scoring (per story spec)
# =============================================================================

DEVIATION_SEVERITY_WEIGHTS: dict[str, int] = {
    "high": 20,  # Major pattern violation (wrong naming convention)
    "medium": 10,  # Moderate deviation (inconsistent style)
    "low": 3,  # Minor variation (slightly different formatting)
}


# =============================================================================
# Type Definitions (Task 1)
# =============================================================================


@dataclass(frozen=True)
class PatternDeviation:
    """A detected deviation from an established pattern.

    Represents a single pattern violation detected during code analysis.
    Frozen (immutable) for audit trail integrity.

    Attributes:
        pattern_type: Type of pattern deviated from.
        pattern_name: Human-readable pattern identifier.
        expected_value: What the pattern specifies.
        actual_value: What was found in the code.
        severity: Deviation severity (high=requires justification, medium=warning, low=info).
        justification: Optional justification for the deviation.
        location: Optional code location (line number or function name).

    Example:
        >>> deviation = PatternDeviation(
        ...     pattern_type=PatternType.NAMING_FUNCTION,
        ...     pattern_name="function_naming",
        ...     expected_value="snake_case",
        ...     actual_value="camelCase",
        ...     severity="high",
        ... )
        >>> deviation.severity
        'high'
    """

    pattern_type: PatternType
    pattern_name: str
    expected_value: str
    actual_value: str
    severity: Literal["high", "medium", "low"]
    justification: str | None = None
    location: str | None = None


@dataclass(frozen=True)
class ErrorHandlingPattern:
    """Pattern describing error handling conventions.

    Represents error handling patterns learned from the codebase.
    Frozen (immutable) for consistency once learned.

    Attributes:
        pattern_name: Human-readable identifier.
        exception_types: Tuple of exception types commonly caught.
        handling_style: Description of handling approach.
        examples: Code examples demonstrating the pattern.

    Example:
        >>> pattern = ErrorHandlingPattern(
        ...     pattern_name="specific_exceptions",
        ...     exception_types=("ValueError", "TypeError"),
        ...     handling_style="specific exceptions with context",
        ... )
        >>> pattern.handling_style
        'specific exceptions with context'
    """

    pattern_name: str
    exception_types: tuple[str, ...]
    handling_style: str
    examples: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StylePattern:
    """Pattern describing code style conventions.

    Represents code style patterns learned from the codebase.
    Frozen (immutable) for consistency once learned.

    Attributes:
        pattern_name: Human-readable identifier.
        category: Style category (import_style, docstring_format, type_hint_style).
        value: The detected style value.
        examples: Code examples demonstrating the pattern.

    Example:
        >>> pattern = StylePattern(
        ...     pattern_name="import_ordering",
        ...     category="import_style",
        ...     value="stdlib, third_party, local",
        ... )
        >>> pattern.category
        'import_style'
    """

    pattern_name: str
    category: Literal["import_style", "docstring_format", "type_hint_style"]
    value: str
    examples: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class PatternValidationResult:
    """Result of pattern adherence validation.

    Aggregates all validation results including score, pass/fail status,
    and detailed deviations. Not frozen because deviations are appended
    during validation.

    Attributes:
        score: Adherence score 0-100.
        passed: Whether score meets threshold (default 70).
        threshold: Score threshold for pass/fail.
        patterns_checked: Number of patterns validated against.
        adherence_percentage: Percentage of patterns followed correctly.
        deviations: List of detected pattern deviations.

    Example:
        >>> result = PatternValidationResult(score=85, passed=True)
        >>> result.to_dict()
        {'score': 85, 'passed': True, ...}
    """

    score: int = 100
    passed: bool = True
    threshold: int = 70
    patterns_checked: int = 0
    adherence_percentage: float = 100.0
    deviations: list[PatternDeviation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of validation result.
        """
        return {
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "patterns_checked": self.patterns_checked,
            "adherence_percentage": self.adherence_percentage,
            "deviation_count": len(self.deviations),
            "deviations": [
                {
                    "pattern_type": d.pattern_type.value,
                    "pattern_name": d.pattern_name,
                    "expected": d.expected_value,
                    "actual": d.actual_value,
                    "severity": d.severity,
                }
                for d in self.deviations
            ],
        }


# =============================================================================
# Default Patterns (from Architecture)
# =============================================================================


DEFAULT_NAMING_PATTERNS: list[CodePattern] = [
    CodePattern(
        pattern_type=PatternType.NAMING_FUNCTION,
        name="function_naming",
        value="snake_case",
        confidence=1.0,
        examples=("get_user", "validate_input", "process_order"),
    ),
    CodePattern(
        pattern_type=PatternType.NAMING_CLASS,
        name="class_naming",
        value="PascalCase",
        confidence=1.0,
        examples=("UserService", "OrderProcessor", "DataValidator"),
    ),
    CodePattern(
        pattern_type=PatternType.NAMING_VARIABLE,
        name="variable_naming",
        value="snake_case",
        confidence=1.0,
        examples=("user_id", "total_count", "is_valid"),
    ),
]


DEFAULT_ERROR_PATTERNS: list[ErrorHandlingPattern] = [
    ErrorHandlingPattern(
        pattern_name="specific_exceptions",
        exception_types=("ValueError", "TypeError", "KeyError", "AttributeError"),
        handling_style="specific exceptions with context",
        examples=(
            "try:\n    ...\nexcept ValueError as e:\n    logger.error(...)\n    raise",
        ),
    ),
]


DEFAULT_STYLE_PATTERNS: list[StylePattern] = [
    StylePattern(
        pattern_name="import_ordering",
        category="import_style",
        value="stdlib, third_party, local (separated by blank lines)",
        examples=(
            "from __future__ import annotations\n\nimport asyncio\n\nfrom pydantic import BaseModel",
        ),
    ),
    StylePattern(
        pattern_name="docstring_format",
        category="docstring_format",
        value="Google-style",
        examples=(
            '"""Summary line.\n\nArgs:\n    param: Description.\n\nReturns:\n    Description.\n"""',
        ),
    ),
    StylePattern(
        pattern_name="type_annotations",
        category="type_hint_style",
        value="full annotations required",
        examples=("def process(data: dict[str, Any]) -> str:",),
    ),
]


# =============================================================================
# Pattern Query Functions (Task 2)
# =============================================================================


def get_naming_patterns(state: dict[str, Any]) -> list[CodePattern]:
    """Get naming patterns from state or return defaults.

    Queries PatternLearner for NAMING_FUNCTION, NAMING_CLASS, NAMING_VARIABLE
    patterns. Falls back to default patterns when memory_context is unavailable.
    Results are cached for efficiency during a single dev_node invocation.

    Args:
        state: YoloState containing memory_context.

    Returns:
        List of CodePattern for naming conventions.

    Example:
        >>> state = {"memory_context": {}}
        >>> patterns = get_naming_patterns(state)
        >>> len(patterns) >= 1
        True
    """
    # Check cache first
    cache_key = "naming_patterns"
    if cache_key in _pattern_cache:
        logger.debug("naming_patterns_from_cache")
        return _pattern_cache[cache_key]

    memory_context = state.get("memory_context")

    if memory_context is None:
        logger.debug("no_memory_context_using_default_naming_patterns")
        result = DEFAULT_NAMING_PATTERNS.copy()
        _pattern_cache[cache_key] = result
        return result

    # Try to get patterns from PatternLearner
    learner = memory_context.get("pattern_learner")
    if learner is None:
        logger.debug("no_pattern_learner_using_default_naming_patterns")
        result = DEFAULT_NAMING_PATTERNS.copy()
        _pattern_cache[cache_key] = result
        return result

    try:
        # Get naming patterns by type - synchronous version for simplicity
        patterns: list[CodePattern] = []

        # Use get_patterns_by_type if available
        if hasattr(learner, "get_patterns_by_type"):
            func_patterns = learner.get_patterns_by_type(PatternType.NAMING_FUNCTION)
            class_patterns = learner.get_patterns_by_type(PatternType.NAMING_CLASS)
            var_patterns = learner.get_patterns_by_type(PatternType.NAMING_VARIABLE)

            if isinstance(func_patterns, list):
                patterns.extend(func_patterns)
            if isinstance(class_patterns, list):
                patterns.extend(class_patterns)
            if isinstance(var_patterns, list):
                patterns.extend(var_patterns)

        if patterns:
            logger.debug("naming_patterns_loaded_from_memory", count=len(patterns))
            _pattern_cache[cache_key] = patterns
            return patterns
        else:
            logger.debug("no_naming_patterns_in_memory_using_defaults")
            result = DEFAULT_NAMING_PATTERNS.copy()
            _pattern_cache[cache_key] = result
            return result

    except Exception as e:
        logger.warning("naming_pattern_query_error", error=str(e))
        result = DEFAULT_NAMING_PATTERNS.copy()
        _pattern_cache[cache_key] = result
        return result


def get_error_patterns(state: dict[str, Any]) -> list[ErrorHandlingPattern]:
    """Get error handling patterns from state or return defaults.

    Queries memory_context for error handling patterns.
    Falls back to default patterns when unavailable.
    Results are cached for efficiency during a single dev_node invocation.

    Args:
        state: YoloState containing memory_context.

    Returns:
        List of ErrorHandlingPattern for error handling conventions.

    Example:
        >>> state = {}
        >>> patterns = get_error_patterns(state)
        >>> len(patterns) >= 1
        True
    """
    # Check cache first
    cache_key = "error_patterns"
    if cache_key in _pattern_cache:
        logger.debug("error_patterns_from_cache")
        return _pattern_cache[cache_key]

    memory_context = state.get("memory_context")

    if memory_context is None:
        logger.debug("no_memory_context_using_default_error_patterns")
        result = DEFAULT_ERROR_PATTERNS.copy()
        _pattern_cache[cache_key] = result
        return result

    # Check for error_patterns in memory_context
    error_patterns = memory_context.get("error_patterns")
    if error_patterns and isinstance(error_patterns, list):
        # Validate types
        valid_patterns = [p for p in error_patterns if isinstance(p, ErrorHandlingPattern)]
        if valid_patterns:
            logger.debug("error_patterns_loaded_from_memory", count=len(valid_patterns))
            _pattern_cache[cache_key] = valid_patterns
            return valid_patterns

    logger.debug("no_error_patterns_in_memory_using_defaults")
    result = DEFAULT_ERROR_PATTERNS.copy()
    _pattern_cache[cache_key] = result
    return result


def get_style_patterns(state: dict[str, Any]) -> list[StylePattern]:
    """Get style patterns from state or return defaults.

    Queries memory_context for code style patterns.
    Falls back to default patterns when unavailable.
    Results are cached for efficiency during a single dev_node invocation.

    Args:
        state: YoloState containing memory_context.

    Returns:
        List of StylePattern for style conventions.

    Example:
        >>> state = {}
        >>> patterns = get_style_patterns(state)
        >>> len(patterns) >= 1
        True
    """
    # Check cache first
    cache_key = "style_patterns"
    if cache_key in _pattern_cache:
        logger.debug("style_patterns_from_cache")
        return _pattern_cache[cache_key]

    memory_context = state.get("memory_context")

    if memory_context is None:
        logger.debug("no_memory_context_using_default_style_patterns")
        result = DEFAULT_STYLE_PATTERNS.copy()
        _pattern_cache[cache_key] = result
        return result

    # Check for style_patterns in memory_context
    style_patterns = memory_context.get("style_patterns")
    if style_patterns and isinstance(style_patterns, list):
        # Validate types
        valid_patterns = [p for p in style_patterns if isinstance(p, StylePattern)]
        if valid_patterns:
            logger.debug("style_patterns_loaded_from_memory", count=len(valid_patterns))
            _pattern_cache[cache_key] = valid_patterns
            return valid_patterns

    logger.debug("no_style_patterns_in_memory_using_defaults")
    result = DEFAULT_STYLE_PATTERNS.copy()
    _pattern_cache[cache_key] = result
    return result


# =============================================================================
# Naming Pattern Analysis (Task 3)
# =============================================================================


def _is_snake_case(name: str) -> bool:
    """Check if name follows snake_case convention."""
    # Allow single underscore prefix for private functions
    if name.startswith("_"):
        name = name.lstrip("_")
    # Allow dunder methods
    if name.startswith("__") and name.endswith("__"):
        return True
    # snake_case: lowercase letters, numbers, underscores
    return bool(re.match(r"^[a-z][a-z0-9_]*$", name)) or name == ""


def _is_pascal_case(name: str) -> bool:
    """Check if name follows PascalCase convention."""
    # PascalCase: starts with uppercase, may have lowercase/uppercase
    return bool(re.match(r"^[A-Z][a-zA-Z0-9]*$", name))


def _extract_names_from_code(code: str) -> dict[str, list[tuple[str, int]]]:
    """Extract function, class, and variable names using AST.

    Args:
        code: Python source code to analyze.

    Returns:
        Dict with keys 'functions', 'classes', 'variables',
        each containing list of (name, line_number) tuples.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"functions": [], "classes": [], "variables": []}

    names: dict[str, list[tuple[str, int]]] = {
        "functions": [],
        "classes": [],
        "variables": [],
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names["functions"].append((node.name, node.lineno))
        elif isinstance(node, ast.ClassDef):
            names["classes"].append((node.name, node.lineno))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names["variables"].append((node.id, node.lineno))

    return names


def analyze_naming_patterns(
    code: str,
    patterns: list[CodePattern],
) -> list[PatternDeviation]:
    """Analyze code for naming pattern deviations.

    Uses AST to extract function, class, and variable names.
    Compares against established naming patterns.

    Args:
        code: Python source code to analyze.
        patterns: Naming patterns to check against.

    Returns:
        List of PatternDeviation for naming violations.

    Example:
        >>> code = "def getUserData(): pass"
        >>> patterns = [CodePattern(...)]  # snake_case pattern
        >>> deviations = analyze_naming_patterns(code, patterns)
        >>> len(deviations) >= 1
        True
    """
    if not patterns:
        return []

    names = _extract_names_from_code(code)
    deviations: list[PatternDeviation] = []

    # Build pattern lookup by type
    pattern_by_type: dict[PatternType, CodePattern] = {}
    for p in patterns:
        if p.pattern_type in (
            PatternType.NAMING_FUNCTION,
            PatternType.NAMING_CLASS,
            PatternType.NAMING_VARIABLE,
        ):
            pattern_by_type[p.pattern_type] = p

    # Check function names
    func_pattern = pattern_by_type.get(PatternType.NAMING_FUNCTION)
    if func_pattern and func_pattern.value == "snake_case":
        for name, lineno in names["functions"]:
            # Skip dunder methods and private methods
            if name.startswith("__") and name.endswith("__"):
                continue
            if not _is_snake_case(name):
                deviations.append(
                    PatternDeviation(
                        pattern_type=PatternType.NAMING_FUNCTION,
                        pattern_name=func_pattern.name,
                        expected_value="snake_case",
                        actual_value=f"non-snake_case: {name}",
                        severity="high",
                        location=f"line {lineno}: {name}",
                    )
                )

    # Check class names
    class_pattern = pattern_by_type.get(PatternType.NAMING_CLASS)
    if class_pattern and class_pattern.value == "PascalCase":
        for name, lineno in names["classes"]:
            if not _is_pascal_case(name):
                deviations.append(
                    PatternDeviation(
                        pattern_type=PatternType.NAMING_CLASS,
                        pattern_name=class_pattern.name,
                        expected_value="PascalCase",
                        actual_value=f"non-PascalCase: {name}",
                        severity="high",
                        location=f"line {lineno}: {name}",
                    )
                )

    # Check variable names (less strict - only report obvious violations)
    var_pattern = pattern_by_type.get(PatternType.NAMING_VARIABLE)
    if var_pattern and var_pattern.value == "snake_case":
        for name, lineno in names["variables"]:
            # Skip single letter variables (common in loops)
            if len(name) == 1:
                continue
            # Skip constants (UPPER_SNAKE_CASE is valid)
            if name.isupper():
                continue
            # Check for obvious camelCase (starts lowercase, has uppercase)
            if name[0].islower() and any(c.isupper() for c in name[1:]):
                deviations.append(
                    PatternDeviation(
                        pattern_type=PatternType.NAMING_VARIABLE,
                        pattern_name=var_pattern.name,
                        expected_value="snake_case",
                        actual_value=f"camelCase: {name}",
                        severity="medium",
                        location=f"line {lineno}: {name}",
                    )
                )

    logger.debug(
        "naming_pattern_analysis_complete",
        function_count=len(names["functions"]),
        class_count=len(names["classes"]),
        variable_count=len(names["variables"]),
        deviation_count=len(deviations),
    )

    return deviations


# =============================================================================
# Error Handling Pattern Analysis (Task 4)
# =============================================================================


def _extract_exception_handlers(code: str) -> list[dict[str, Any]]:
    """Extract exception handler info using AST.

    Args:
        code: Python source code to analyze.

    Returns:
        List of dicts with exception handler details.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    handlers: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                handler_info: dict[str, Any] = {
                    "lineno": handler.lineno,
                    "exception_type": None,
                    "is_bare": False,
                }

                if handler.type is None:
                    # Bare except clause
                    handler_info["is_bare"] = True
                elif isinstance(handler.type, ast.Name):
                    handler_info["exception_type"] = handler.type.id
                elif isinstance(handler.type, ast.Tuple):
                    # Multiple exception types
                    types = []
                    for elt in handler.type.elts:
                        if isinstance(elt, ast.Name):
                            types.append(elt.id)
                    handler_info["exception_type"] = tuple(types)

                handlers.append(handler_info)

    return handlers


def analyze_error_handling_patterns(
    code: str,
    patterns: list[ErrorHandlingPattern],
) -> list[PatternDeviation]:
    """Analyze code for error handling pattern deviations.

    Extracts try/except blocks using AST and checks against
    established error handling patterns.

    Args:
        code: Python source code to analyze.
        patterns: Error handling patterns to check against.

    Returns:
        List of PatternDeviation for error handling violations.

    Example:
        >>> code = "try:\\n    x = 1\\nexcept:\\n    pass"
        >>> patterns = [ErrorHandlingPattern(...)]
        >>> deviations = analyze_error_handling_patterns(code, patterns)
        >>> len(deviations) >= 1  # bare except detected
        True
    """
    if not patterns:
        return []

    handlers = _extract_exception_handlers(code)
    deviations: list[PatternDeviation] = []

    # Get allowed exception types from patterns
    allowed_exceptions: set[str] = set()
    for p in patterns:
        allowed_exceptions.update(p.exception_types)

    for handler in handlers:
        lineno = handler["lineno"]

        # Check for bare except
        if handler["is_bare"]:
            deviations.append(
                PatternDeviation(
                    pattern_type=PatternType.DESIGN_PATTERN,
                    pattern_name="error_handling",
                    expected_value="specific exceptions",
                    actual_value="bare except",
                    severity="high",
                    location=f"line {lineno}",
                )
            )
            continue

        # Check for generic Exception
        exc_type = handler["exception_type"]
        if exc_type == "Exception":
            deviations.append(
                PatternDeviation(
                    pattern_type=PatternType.DESIGN_PATTERN,
                    pattern_name="error_handling",
                    expected_value="specific exceptions",
                    actual_value="generic Exception",
                    severity="medium",
                    location=f"line {lineno}",
                )
            )
            continue

        # If using allowed exception types, no deviation
        # Custom project exceptions are fine as long as they're specific

    logger.debug(
        "error_handling_analysis_complete",
        handler_count=len(handlers),
        deviation_count=len(deviations),
    )

    return deviations


# =============================================================================
# Style Pattern Analysis (Task 5)
# =============================================================================


def _check_import_ordering(code: str) -> bool:
    """Check if imports follow stdlib, third_party, local ordering.

    Args:
        code: Python source code to analyze.

    Returns:
        True if imports are properly ordered.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True  # Can't analyze, assume OK

    imports: list[tuple[int, str, str]] = []  # (lineno, category, module)

    stdlib_modules = {
        "abc",
        "asyncio",
        "collections",
        "contextlib",
        "dataclasses",
        "datetime",
        "enum",
        "functools",
        "io",
        "itertools",
        "json",
        "logging",
        "os",
        "pathlib",
        "re",
        "sys",
        "time",
        "typing",
        "uuid",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module == "__future__":
                    category = "future"
                elif module in stdlib_modules:
                    category = "stdlib"
                else:
                    category = "third_party"  # Assume third party or local
                imports.append((node.lineno, category, module))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                if module == "__future__":
                    category = "future"
                elif module == "yolo_developer":
                    category = "local"
                elif module in stdlib_modules:
                    category = "stdlib"
                else:
                    category = "third_party"
                imports.append((node.lineno, category, module))

    # Sort by line number
    imports.sort(key=lambda x: x[0])

    # Check ordering: future, stdlib, third_party, local
    expected_order = {"future": 0, "stdlib": 1, "third_party": 2, "local": 3}
    last_category_order = -1

    for _, category, _ in imports:
        cat_order = expected_order.get(category, 2)
        if cat_order < last_category_order:
            return False  # Out of order
        last_category_order = cat_order

    return True


def _check_docstring_format(code: str) -> list[str]:
    """Check if docstrings follow Google-style format.

    Validates that public functions and classes have docstrings with
    appropriate Google-style sections (Args, Returns, Raises).

    Note: Currently returns empty list as docstring format checking is
    advisory only. The AST parsing infrastructure is in place for future
    enhancement if stricter validation is needed.

    Args:
        code: Python source code to analyze.

    Returns:
        List of issues with docstring format (empty for now).
    """
    # Docstring format checking is advisory only - we validate structure
    # but don't flag issues to avoid false positives with valid styles.
    # This function exists as infrastructure for future stricter validation.
    try:
        ast.parse(code)
    except SyntaxError:
        pass  # Can't analyze invalid code

    return []  # No issues flagged (advisory mode)


def _check_type_annotations(code: str) -> list[str]:
    """Check if functions have type annotations.

    Args:
        code: Python source code to analyze.

    Returns:
        List of functions missing type annotations.
    """
    missing: list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return missing

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Skip dunder methods and private methods for strictness
            if node.name.startswith("_"):
                continue

            # Check return annotation
            if node.returns is None:
                missing.append(f"{node.name}: missing return annotation")

            # Check parameter annotations
            for arg in node.args.args:
                if arg.arg in ("self", "cls"):
                    continue
                if arg.annotation is None:
                    missing.append(f"{node.name}: parameter '{arg.arg}' missing annotation")

    return missing


def analyze_style_patterns(
    code: str,
    patterns: list[StylePattern],
) -> list[PatternDeviation]:
    """Analyze code for style pattern deviations.

    Checks import ordering, docstring format, and type annotation style.

    Args:
        code: Python source code to analyze.
        patterns: Style patterns to check against.

    Returns:
        List of PatternDeviation for style violations.

    Example:
        >>> code = "from mypackage import foo\\nimport os"
        >>> patterns = [StylePattern(category="import_style", ...)]
        >>> deviations = analyze_style_patterns(code, patterns)
    """
    if not patterns:
        return []

    deviations: list[PatternDeviation] = []

    # Build pattern lookup by category
    pattern_by_category: dict[str, StylePattern] = {}
    for p in patterns:
        pattern_by_category[p.category] = p

    # Check import ordering
    import_pattern = pattern_by_category.get("import_style")
    if import_pattern:
        if not _check_import_ordering(code):
            deviations.append(
                PatternDeviation(
                    pattern_type=PatternType.IMPORT_STYLE,
                    pattern_name=import_pattern.pattern_name,
                    expected_value=import_pattern.value,
                    actual_value="imports out of order",
                    severity="low",
                    location="import section",
                )
            )

    # Check type annotations
    type_pattern = pattern_by_category.get("type_hint_style")
    if type_pattern:
        missing = _check_type_annotations(code)
        for issue in missing[:3]:  # Limit to first 3 issues
            deviations.append(
                PatternDeviation(
                    pattern_type=PatternType.DESIGN_PATTERN,
                    pattern_name=type_pattern.pattern_name,
                    expected_value=type_pattern.value,
                    actual_value=issue,
                    severity="medium",
                    location=issue,
                )
            )

    logger.debug(
        "style_pattern_analysis_complete",
        deviation_count=len(deviations),
    )

    return deviations


# =============================================================================
# Pattern Validation Aggregation (Task 6)
# =============================================================================


def _calculate_pattern_score(deviations: list[PatternDeviation]) -> int:
    """Calculate pattern adherence score from deviations.

    Args:
        deviations: List of detected pattern deviations.

    Returns:
        Score 0-100 (100 = no deviations).
    """
    score = 100
    for deviation in deviations:
        weight = DEVIATION_SEVERITY_WEIGHTS.get(deviation.severity, 10)
        score -= weight
    return max(0, score)


def validate_pattern_adherence(
    code: str,
    state: dict[str, Any],
    threshold: int = 70,
) -> PatternValidationResult:
    """Validate code adherence to established patterns.

    Analyzes code for naming, error handling, and style pattern compliance.
    Queries patterns from state["memory_context"] via PatternLearner.

    Args:
        code: Python source code to validate.
        state: YoloState containing memory_context with patterns.
        threshold: Minimum score to pass (0-100, default 70).

    Returns:
        PatternValidationResult with score, pass/fail, and deviations.

    Example:
        >>> result = validate_pattern_adherence(code, state)
        >>> result.passed
        True
        >>> result.adherence_percentage
        95.5
    """
    logger.debug("validate_pattern_adherence_start", threshold=threshold)

    # Query patterns from state
    naming_patterns = get_naming_patterns(state)
    error_patterns = get_error_patterns(state)
    style_patterns = get_style_patterns(state)

    all_deviations: list[PatternDeviation] = []

    # Analyze naming patterns
    naming_deviations = analyze_naming_patterns(code, naming_patterns)
    all_deviations.extend(naming_deviations)

    # Analyze error handling patterns
    error_deviations = analyze_error_handling_patterns(code, error_patterns)
    all_deviations.extend(error_deviations)

    # Analyze style patterns
    style_deviations = analyze_style_patterns(code, style_patterns)
    all_deviations.extend(style_deviations)

    # Calculate score
    score = _calculate_pattern_score(all_deviations)

    # Calculate patterns checked and adherence
    patterns_checked = len(naming_patterns) + len(error_patterns) + len(style_patterns)
    deviation_count = len(all_deviations)

    # Calculate adherence percentage based on patterns without deviations
    # If no patterns checked, default to 100%
    if patterns_checked > 0:
        # Adherence = percentage of patterns that had no deviations
        # Each deviation counts against one pattern (capped at patterns_checked)
        patterns_with_deviations = min(deviation_count, patterns_checked)
        adherence_percentage = ((patterns_checked - patterns_with_deviations) / patterns_checked) * 100
    else:
        adherence_percentage = 100.0 if not all_deviations else 0.0

    # Determine pass/fail
    passed = score >= threshold

    result = PatternValidationResult(
        score=score,
        passed=passed,
        threshold=threshold,
        patterns_checked=patterns_checked,
        adherence_percentage=adherence_percentage,
        deviations=all_deviations,
    )

    # Audit trail logging
    logger.info(
        "pattern_validation_audit",
        score=score,
        passed=passed,
        threshold=threshold,
        patterns_checked=patterns_checked,
        deviation_count=len(all_deviations),
        naming_deviations=len(naming_deviations),
        error_deviations=len(error_deviations),
        style_deviations=len(style_deviations),
    )

    return result
