"""Code utilities for Dev agent (Story 8.2).

This module provides utility functions for code validation, extraction,
and maintainability checking.

Key Functions:
    - validate_python_syntax: Validate Python code syntax using ast.parse
    - extract_code_from_response: Extract code from LLM response
    - check_maintainability: Check code for maintainability guidelines

Example:
    >>> from yolo_developer.agents.dev.code_utils import (
    ...     validate_python_syntax,
    ...     extract_code_from_response,
    ... )
    >>>
    >>> is_valid, error = validate_python_syntax("def hello(): pass")
    >>> is_valid
    True
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Syntax Validation (Task 3)
# =============================================================================


def validate_python_syntax(code: str) -> tuple[bool, str | None]:
    """Validate Python code syntax using ast.parse (AC5).

    Args:
        code: Python code string to validate.

    Returns:
        Tuple of (is_valid, error_message).
        error_message is None if code is valid.

    Example:
        >>> is_valid, error = validate_python_syntax("def hello(): pass")
        >>> is_valid
        True
        >>> error is None
        True
        >>> is_valid, error = validate_python_syntax("def broken(")
        >>> is_valid
        False
        >>> "SyntaxError" in str(error) or "Line" in str(error)
        True
    """
    if not code or not code.strip():
        return False, "Empty code provided"

    try:
        ast.parse(code)
        logger.debug("syntax_validation_passed", code_length=len(code))
        return True, None
    except SyntaxError as e:
        error_msg = f"Line {e.lineno}: {e.msg}"
        logger.debug(
            "syntax_validation_failed",
            error=error_msg,
            line=e.lineno,
        )
        return False, error_msg
    except Exception as e:
        # Handle other parsing errors (e.g., encoding issues)
        error_msg = f"Parse error: {e}"
        logger.debug("syntax_validation_error", error=error_msg)
        return False, error_msg


# =============================================================================
# Code Extraction
# =============================================================================


def extract_code_from_response(response: str) -> str:
    """Extract Python code from LLM response.

    Handles responses with:
    - Code in ```python ... ``` blocks
    - Code in ``` ... ``` blocks
    - Raw code without blocks

    Args:
        response: LLM response potentially containing code.

    Returns:
        Extracted code string, or original response if no blocks found.

    Example:
        >>> response = '''```python
        ... def hello():
        ...     return "world"
        ... ```'''
        >>> extract_code_from_response(response)
        'def hello():\\n    return "world"'
    """
    if not response:
        return ""

    # Try to find ```python ... ``` block
    python_pattern = r"```python\n(.*?)```"
    python_matches: list[str] = re.findall(python_pattern, response, re.DOTALL)
    if python_matches:
        # Return the largest match (in case of multiple blocks)
        code: str = max(python_matches, key=len).strip()
        logger.debug("code_extracted_from_python_block", code_length=len(code))
        return code

    # Try to find generic ``` ... ``` block
    generic_pattern = r"```\n?(.*?)```"
    generic_matches: list[str] = re.findall(generic_pattern, response, re.DOTALL)
    if generic_matches:
        code = max(generic_matches, key=len).strip()
        logger.debug("code_extracted_from_generic_block", code_length=len(code))
        return code

    # Return raw response as-is (might be raw code)
    logger.debug("code_no_blocks_found", response_length=len(response))
    return response.strip()


# =============================================================================
# Maintainability Checking (Task 5)
# =============================================================================


@dataclass(frozen=True)
class MaintainabilityWarning:
    """A maintainability warning for generated code.

    Attributes:
        category: Warning category (function_length, nesting_depth, naming).
        message: Human-readable warning message.
        line: Line number where issue was found (if applicable).
        severity: Warning severity (info, warning, error).
    """

    category: str
    message: str
    line: int | None = None
    severity: str = "warning"


@dataclass
class MaintainabilityReport:
    """Report of maintainability analysis for generated code.

    Attributes:
        warnings: List of maintainability warnings.
        function_count: Number of functions found.
        max_function_length: Longest function length in lines.
        max_nesting_depth: Maximum nesting depth found.
        max_cyclomatic_complexity: Highest cyclomatic complexity found.
    """

    warnings: list[MaintainabilityWarning] = field(default_factory=list)
    function_count: int = 0
    max_function_length: int = 0
    max_nesting_depth: int = 0
    max_cyclomatic_complexity: int = 0

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def get_warnings_by_category(self, category: str) -> list[MaintainabilityWarning]:
        """Get warnings filtered by category."""
        return [w for w in self.warnings if w.category == category]


def check_maintainability(code: str) -> MaintainabilityReport:
    """Check code for maintainability guidelines (AC1, AC2, AC3, AC4).

    Analyzes code and returns warnings for:
    - Functions longer than 50 lines
    - Nesting depth greater than 3 levels
    - Cyclomatic complexity greater than 10
    - Non-descriptive variable names

    Note: This is advisory only - warnings don't block code generation.

    Args:
        code: Python code string to analyze.

    Returns:
        MaintainabilityReport with warnings and metrics.

    Example:
        >>> code = '''
        ... def very_long_function():
        ...     x = 1
        ...     y = 2
        ...     # ... many lines ...
        ...     return x + y
        ... '''
        >>> report = check_maintainability(code)
        >>> report.function_count >= 0
        True
    """
    report = MaintainabilityReport()

    if not code or not code.strip():
        return report

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Can't analyze invalid code
        logger.debug("maintainability_check_skipped_invalid_syntax")
        return report

    # Analyze functions
    _analyze_functions(tree, report, code)

    # Analyze nesting depth
    _analyze_nesting(tree, report)

    # Analyze cyclomatic complexity (AC1)
    _analyze_cyclomatic_complexity(tree, report)

    # Analyze naming
    _analyze_naming(tree, report)

    logger.debug(
        "maintainability_check_complete",
        warning_count=len(report.warnings),
        function_count=report.function_count,
        max_function_length=report.max_function_length,
        max_nesting_depth=report.max_nesting_depth,
        max_cyclomatic_complexity=report.max_cyclomatic_complexity,
    )

    return report


def _analyze_functions(tree: ast.AST, report: MaintainabilityReport, code: str) -> None:
    """Analyze functions for length and complexity."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            report.function_count += 1

            # Calculate function length
            if hasattr(node, "end_lineno") and node.end_lineno and node.lineno:
                func_length = node.end_lineno - node.lineno + 1
                report.max_function_length = max(report.max_function_length, func_length)

                if func_length > 50:
                    report.warnings.append(
                        MaintainabilityWarning(
                            category="function_length",
                            message=f"Function '{node.name}' is {func_length} lines "
                            f"(recommended: < 50)",
                            line=node.lineno,
                            severity="warning",
                        )
                    )


def _analyze_nesting(tree: ast.AST, report: MaintainabilityReport) -> None:
    """Analyze nesting depth of control structures."""

    def get_nesting_depth(node: ast.AST, depth: int = 0) -> int:
        """Recursively calculate maximum nesting depth."""
        max_depth = depth

        for child in ast.iter_child_nodes(node):
            child_depth = depth
            # Increment depth for control flow statements
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.Match)):
                child_depth = depth + 1

            max_depth = max(max_depth, get_nesting_depth(child, child_depth))

        return max_depth

    # Check nesting in all function bodies
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            depth = get_nesting_depth(node)
            report.max_nesting_depth = max(report.max_nesting_depth, depth)

            if depth > 3:
                report.warnings.append(
                    MaintainabilityWarning(
                        category="nesting_depth",
                        message=f"Function '{node.name}' has nesting depth {depth} "
                        f"(recommended: <= 3)",
                        line=node.lineno,
                        severity="warning",
                    )
                )


def _analyze_cyclomatic_complexity(tree: ast.AST, report: MaintainabilityReport) -> None:
    """Analyze cyclomatic complexity of functions (AC1).

    Cyclomatic complexity = 1 + number of decision points.
    Decision points include: if, elif, for, while, and, or, except, assert,
    ternary expressions, and comprehension conditions.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = _calculate_cyclomatic_complexity(node)
            report.max_cyclomatic_complexity = max(report.max_cyclomatic_complexity, complexity)

            if complexity > 10:
                report.warnings.append(
                    MaintainabilityWarning(
                        category="cyclomatic_complexity",
                        message=f"Function '{node.name}' has cyclomatic complexity {complexity} "
                        f"(recommended: < 10)",
                        line=node.lineno,
                        severity="warning",
                    )
                )


def _calculate_cyclomatic_complexity(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Calculate cyclomatic complexity for a function.

    Args:
        func_node: AST node for function definition.

    Returns:
        Cyclomatic complexity score (1 = simplest).
    """
    complexity = 1  # Base complexity

    for node in ast.walk(func_node):
        # Decision points that add to complexity
        if isinstance(node, (ast.If, ast.IfExp)):  # if statements and ternary
            complexity += 1
        elif isinstance(node, (ast.For, ast.While)):  # loops
            complexity += 1
        elif isinstance(node, ast.ExceptHandler):  # except clauses
            complexity += 1
        elif isinstance(node, ast.Assert):  # assert statements
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            # and/or operators add (num_operands - 1) decision points
            complexity += len(node.values) - 1
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # Comprehension conditions
            for generator in node.generators:
                complexity += len(generator.ifs)

    return complexity


def _analyze_naming(tree: ast.AST, report: MaintainabilityReport) -> None:
    """Analyze variable and function naming conventions."""
    # Single-letter variables to flag (except i, j, k, n, x, y, z in specific contexts)
    allowed_single_letters = {"i", "j", "k", "n", "x", "y", "z", "_", "e", "f"}

    for node in ast.walk(tree):
        # Check variable assignments
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            name = node.id
            if len(name) == 1 and name not in allowed_single_letters:
                report.warnings.append(
                    MaintainabilityWarning(
                        category="naming",
                        message=f"Single-letter variable '{name}' - use descriptive name",
                        line=getattr(node, "lineno", None),
                        severity="info",
                    )
                )

        # Check for non-snake_case function names
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if not name.startswith("_") and not _is_snake_case(name):
                if not name.startswith("test"):  # Allow test methods
                    report.warnings.append(
                        MaintainabilityWarning(
                            category="naming",
                            message=f"Function '{name}' should use snake_case",
                            line=node.lineno,
                            severity="info",
                        )
                    )


def _is_snake_case(name: str) -> bool:
    """Check if name follows snake_case convention."""
    # Allow dunder methods
    if name.startswith("__") and name.endswith("__"):
        return True
    # Check for snake_case pattern
    return bool(re.match(r"^[a-z][a-z0-9_]*$", name))


# =============================================================================
# Code Generation Helper
# =============================================================================


async def generate_code_with_validation(
    router: Any,
    prompt: str,
    tier: str = "complex",
    max_retries: int = 2,
) -> tuple[str, bool]:
    """Generate code with syntax validation and retry (AC5).

    Calls LLM to generate code, validates syntax, and retries with
    error context if validation fails.

    Args:
        router: LLMRouter instance for making calls.
        prompt: Code generation prompt.
        tier: Model tier to use (default "complex" per ADR-003).
        max_retries: Maximum validation retries (default 2).

    Returns:
        Tuple of (code, is_valid). Code is the best attempt,
        is_valid indicates if syntax is valid.

    Example:
        >>> code, is_valid = await generate_code_with_validation(
        ...     router=router,
        ...     prompt="Generate a hello function",
        ... )
    """
    from yolo_developer.agents.dev.prompts.code_generation import build_retry_prompt

    _ = tier  # Retained for backwards compatibility with older call sites.

    messages = [{"role": "user", "content": prompt}]
    response = await router.call_task(messages=messages, task_type="code_generation")

    code = extract_code_from_response(response)
    is_valid, error = validate_python_syntax(code)

    if is_valid:
        return code, True

    # Retry with error context
    for attempt in range(max_retries):
        logger.info(
            "code_generation_retry",
            attempt=attempt + 1,
            max_retries=max_retries,
            error=error,
        )

        retry_prompt = build_retry_prompt(prompt, error or "Unknown error", code)
        messages = [{"role": "user", "content": retry_prompt}]
        response = await router.call_task(messages=messages, task_type="code_generation")

        code = extract_code_from_response(response)
        is_valid, error = validate_python_syntax(code)

        if is_valid:
            return code, True

    # Return best attempt even if invalid
    logger.warning(
        "code_generation_failed_validation",
        final_error=error,
    )
    return code, False
