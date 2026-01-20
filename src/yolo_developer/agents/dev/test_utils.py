"""Test analysis utilities for Dev agent (Story 8.3).

This module provides utility functions for analyzing code to generate tests:
- Function extraction: Parse code to find public functions to test
- Edge case identification: Suggest edge cases based on signatures and types
- LLM test generation: Generate tests using LLM with validation

Key Functions:
    - extract_public_functions: Extract public function info from code using AST
    - identify_edge_cases: Identify potential edge cases for a function
    - generate_unit_tests_with_llm: Generate unit tests using LLM

Example:
    >>> from yolo_developer.agents.dev.test_utils import (
    ...     extract_public_functions,
    ...     identify_edge_cases,
    ... )
    >>>
    >>> code = "def add(a: int, b: int) -> int: return a + b"
    >>> functions = extract_public_functions(code)
    >>> len(functions)
    1
    >>> functions[0].name
    'add'
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from yolo_developer.llm.router import LLMRouter

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class QualityReport:
    """Report of test quality analysis (AC3).

    Note: This class is intentionally NOT frozen because it's built
    incrementally during validation. The validation functions modify
    the report's fields as they analyze the test code. Freezing would
    require building a complete dict first and creating the instance
    once, which is less readable for the validation logic.

    Attributes:
        warnings: List of quality warnings.
        has_assertions: Whether tests have proper assertions.
        is_deterministic: Whether tests appear deterministic.
        uses_fixtures: Whether tests use fixtures appropriately.

    Example:
        >>> report = QualityReport(has_assertions=True)
        >>> report.is_acceptable()
        True
    """

    warnings: list[str] = field(default_factory=list)
    has_assertions: bool = True
    is_deterministic: bool = True
    uses_fixtures: bool = False

    def is_acceptable(self) -> bool:
        """Check if test quality is acceptable.

        Tests are acceptable if they have assertions and are deterministic.

        Returns:
            True if test quality meets minimum standards.
        """
        return self.has_assertions and self.is_deterministic


@dataclass(frozen=True)
class FunctionInfo:
    """Information about a public function to test.

    Attributes:
        name: Function name.
        signature: Full function signature string.
        docstring: Function docstring (if present).
        parameters: Tuple of parameter names.
        return_type: Return type annotation as string (if present).

    Example:
        >>> info = FunctionInfo(
        ...     name="add",
        ...     signature="def add(a: int, b: int) -> int",
        ...     docstring="Add two numbers.",
        ...     parameters=("a", "b"),
        ...     return_type="int",
        ... )
        >>> info.name
        'add'
    """

    name: str
    signature: str
    docstring: str | None
    parameters: tuple[str, ...]
    return_type: str | None


# =============================================================================
# Function Extraction (AC1)
# =============================================================================


def extract_public_functions(code: str) -> list[FunctionInfo]:
    """Extract public function information from code using AST (AC1).

    Parses Python code and extracts information about all public functions
    (those not starting with underscore) including their name, signature,
    docstring, parameters, and return type.

    Args:
        code: Python code string to analyze.

    Returns:
        List of FunctionInfo for each public function found.
        Returns empty list if code is empty, invalid, or has no functions.

    Example:
        >>> code = '''
        ... def greet(name: str) -> str:
        ...     \"\"\"Greet someone.\"\"\"
        ...     return f"Hello, {name}!"
        ... '''
        >>> funcs = extract_public_functions(code)
        >>> len(funcs)
        1
        >>> funcs[0].name
        'greet'
    """
    if not code or not code.strip():
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.debug("function_extraction_syntax_error")
        return []

    functions: list[FunctionInfo] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private functions (starting with _)
            if node.name.startswith("_"):
                continue

            # Extract function info
            func_info = _extract_function_info(node, code)
            functions.append(func_info)

    logger.debug(
        "functions_extracted",
        function_count=len(functions),
        function_names=[f.name for f in functions],
    )

    return functions


def _extract_function_info(node: ast.FunctionDef | ast.AsyncFunctionDef, code: str) -> FunctionInfo:
    """Extract FunctionInfo from an AST function node.

    Args:
        node: AST node for function definition.
        code: Original source code for signature reconstruction.

    Returns:
        FunctionInfo with extracted details.
    """
    name = node.name

    # Extract parameters
    params: list[str] = []
    for arg in node.args.args:
        params.append(arg.arg)
    for arg in node.args.kwonlyargs:
        params.append(arg.arg)

    # Extract return type
    return_type: str | None = None
    if node.returns:
        return_type = _annotation_to_string(node.returns)

    # Extract docstring
    docstring: str | None = None
    if node.body and isinstance(node.body[0], ast.Expr):
        if isinstance(node.body[0].value, ast.Constant):
            value = node.body[0].value.value
            if isinstance(value, str):
                docstring = value

    # Build signature
    signature = _build_signature(node)

    return FunctionInfo(
        name=name,
        signature=signature,
        docstring=docstring,
        parameters=tuple(params),
        return_type=return_type,
    )


def _annotation_to_string(annotation: ast.expr) -> str:
    """Convert an AST annotation to a string representation.

    Args:
        annotation: AST expression for type annotation.

    Returns:
        String representation of the type annotation.
    """
    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Constant):
        return str(annotation.value)
    elif isinstance(annotation, ast.Subscript):
        # Handle generic types like list[int], dict[str, Any]
        value = _annotation_to_string(annotation.value)
        slice_val = _annotation_to_string(annotation.slice)
        return f"{value}[{slice_val}]"
    elif isinstance(annotation, ast.Tuple):
        # Handle tuple of types
        elts = ", ".join(_annotation_to_string(e) for e in annotation.elts)
        return elts
    elif isinstance(annotation, ast.BinOp):
        # Handle union types (X | Y)
        left = _annotation_to_string(annotation.left)
        right = _annotation_to_string(annotation.right)
        return f"{left} | {right}"
    elif isinstance(annotation, ast.Attribute):
        # Handle module.Type
        return f"{_annotation_to_string(annotation.value)}.{annotation.attr}"
    else:
        return ast.unparse(annotation)


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build function signature string from AST node.

    Args:
        node: AST node for function definition.

    Returns:
        String representation of function signature.
    """
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"

    # Build argument list
    args_parts: list[str] = []

    # Regular arguments
    defaults_offset = len(node.args.args) - len(node.args.defaults)
    for i, arg in enumerate(node.args.args):
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {_annotation_to_string(arg.annotation)}"
        # Add default value if present
        default_idx = i - defaults_offset
        if default_idx >= 0:
            default_val = node.args.defaults[default_idx]
            arg_str += f" = {ast.unparse(default_val)}"
        args_parts.append(arg_str)

    # Keyword-only arguments
    for i, arg in enumerate(node.args.kwonlyargs):
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {_annotation_to_string(arg.annotation)}"
        # Add default value if present (kw_defaults can have None entries)
        if i < len(node.args.kw_defaults):
            kw_default = node.args.kw_defaults[i]
            if kw_default is not None:
                arg_str += f" = {ast.unparse(kw_default)}"
        args_parts.append(arg_str)

    args_str = ", ".join(args_parts)

    # Build return type
    return_str = ""
    if node.returns:
        return_str = f" -> {_annotation_to_string(node.returns)}"

    return f"{prefix} {node.name}({args_str}){return_str}"


# =============================================================================
# Edge Case Identification (AC2)
# =============================================================================


def identify_edge_cases(func_info: FunctionInfo) -> list[str]:
    """Identify potential edge cases for a function based on its signature (AC2).

    Analyzes the function signature and docstring to suggest edge cases
    that should be tested, such as:
    - Empty collections (lists, strings, dicts)
    - None values for optional parameters
    - Zero/negative values for numeric parameters
    - Boundary conditions mentioned in docstrings

    Note: This is a heuristic-based approach using string matching on
    the signature. It may not handle complex nested types perfectly
    (e.g., dict[str, list[int]]). The LLM prompt for test generation
    provides more detailed edge case guidance that complements this.

    Args:
        func_info: FunctionInfo for the function to analyze.

    Returns:
        List of edge case descriptions to test.

    Example:
        >>> info = FunctionInfo(
        ...     name="sum_items",
        ...     signature="def sum_items(items: list[int]) -> int",
        ...     docstring=None,
        ...     parameters=("items",),
        ...     return_type="int",
        ... )
        >>> edge_cases = identify_edge_cases(info)
        >>> any("empty" in ec.lower() for ec in edge_cases)
        True
    """
    edge_cases: list[str] = []
    signature_lower = func_info.signature.lower()

    # Check for string parameters
    if "str" in signature_lower:
        edge_cases.append("Test with empty string ''")
        edge_cases.append("Test with whitespace-only string")

    # Check for list/collection parameters
    if any(t in signature_lower for t in ["list", "tuple", "set", "sequence"]):
        edge_cases.append("Test with empty collection []")
        edge_cases.append("Test with single element")

    # Check for dict parameters
    if "dict" in signature_lower:
        edge_cases.append("Test with empty dict {}")
        edge_cases.append("Test with single key-value pair")

    # Check for Optional/None types
    if "| none" in signature_lower or "none" in signature_lower:
        edge_cases.append("Test with None value")

    # Check for numeric parameters
    if any(t in signature_lower for t in ["int", "float", "number"]):
        edge_cases.append("Test with zero value (0)")
        edge_cases.append("Test with negative value")
        edge_cases.append("Test with large value")

    # Check for bool parameters
    if "bool" in signature_lower:
        edge_cases.append("Test with True")
        edge_cases.append("Test with False")

    # Check docstring for error/exception mentions
    if func_info.docstring:
        docstring_lower = func_info.docstring.lower()
        if any(word in docstring_lower for word in ["raise", "error", "exception", "invalid"]):
            edge_cases.append("Test with invalid input that should raise exception")
            edge_cases.append("Test error message is meaningful")

    # Always suggest basic happy path and failure tests
    if not edge_cases:
        edge_cases.append("Test basic happy path with valid inputs")
        edge_cases.append("Test expected output format")

    logger.debug(
        "edge_cases_identified",
        function_name=func_info.name,
        edge_case_count=len(edge_cases),
    )

    return edge_cases


# =============================================================================
# LLM Test Generation (AC5, AC6)
# =============================================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _call_llm_with_retry(
    router: LLMRouter,
    prompt: str,
) -> str:
    """Call LLM with tenacity retry and exponential backoff (AC5, ADR-007).

    Args:
        router: LLMRouter instance for making LLM calls.
        prompt: The prompt to send to the LLM.

    Returns:
        LLM response string.

    Raises:
        Exception: If all retries fail.
    """
    return await router.call(
        messages=[{"role": "user", "content": prompt}],
        tier="complex",
    )


async def generate_unit_tests_with_llm(
    implementation_code: str,
    functions: list[FunctionInfo],
    module_name: str,
    router: LLMRouter,
    max_retries: int = 2,
    additional_context: str = "",
) -> tuple[str, bool]:
    """Generate unit tests using LLM with syntax validation (AC5, AC6).

    Calls the LLM to generate pytest unit tests for the given implementation,
    validates the generated test syntax, and retries with error context if needed.

    Uses tenacity with exponential backoff for transient LLM failures (AC5, ADR-007).
    Syntax validation failures trigger retry with modified prompt.

    Args:
        implementation_code: Python code to generate tests for.
        functions: List of FunctionInfo for public functions to test.
        module_name: Name of the module being tested.
        router: LLMRouter instance for making LLM calls.
        max_retries: Maximum retries on syntax validation failure. Defaults to 2.
        additional_context: Extra context for test generation. Defaults to empty.

    Returns:
        Tuple of (test_code, is_valid). test_code is the generated test code,
        is_valid indicates if the code passed syntax validation.

    Example:
        >>> code, is_valid = await generate_unit_tests_with_llm(
        ...     implementation_code="def add(a, b): return a + b",
        ...     functions=[...],
        ...     module_name="math_utils",
        ...     router=router,
        ... )
    """
    from yolo_developer.agents.dev.code_utils import (
        extract_code_from_response,
        validate_python_syntax,
    )
    from yolo_developer.agents.dev.prompts.test_generation import (
        build_test_generation_prompt,
        build_test_retry_prompt,
    )

    # Build function list for prompt
    func_names = [f.name for f in functions]

    # Build initial prompt
    prompt = build_test_generation_prompt(
        implementation_code=implementation_code,
        function_list=func_names,
        module_name=module_name,
        additional_context=additional_context,
    )

    try:
        # Call LLM with tenacity retry (AC5, ADR-007)
        response = await _call_llm_with_retry(router, prompt)
    except Exception as e:
        logger.error("llm_test_generation_failed", error=str(e))
        return "", False

    # Extract code from response
    test_code = extract_code_from_response(response)
    is_valid, error = validate_python_syntax(test_code)

    if is_valid:
        logger.info(
            "test_generation_success",
            module_name=module_name,
            function_count=len(functions),
        )
        return test_code, True

    # Retry with error context for syntax validation failures
    for attempt in range(max_retries):
        logger.info(
            "test_generation_syntax_retry",
            attempt=attempt + 1,
            max_retries=max_retries,
            error=error,
        )

        retry_prompt = build_test_retry_prompt(
            original_prompt=prompt,
            error_message=error or "Unknown syntax error",
            previous_tests=test_code,
        )

        try:
            # LLM call with tenacity retry
            response = await _call_llm_with_retry(router, retry_prompt)
        except Exception as e:
            logger.error("llm_test_generation_retry_failed", error=str(e))
            continue

        test_code = extract_code_from_response(response)
        is_valid, error = validate_python_syntax(test_code)

        if is_valid:
            logger.info(
                "test_generation_success_after_retry",
                module_name=module_name,
                attempt=attempt + 1,
            )
            return test_code, True

    # Return best attempt even if invalid
    logger.warning(
        "test_generation_validation_failed",
        module_name=module_name,
        final_error=error,
    )
    return test_code, False


# =============================================================================
# Coverage Calculation (AC4)
# =============================================================================


def calculate_coverage_estimate(code: str, tests: str) -> float:
    """Calculate estimated test coverage using AST analysis (AC4).

    This is a heuristic-based coverage estimation that analyzes:
    - Number of public functions in code
    - Number of test functions that reference implementation functions
    - Ratio of tested functions to total functions

    Note: This is an estimate, not actual runtime coverage. For precise
    coverage, use pytest-cov during test execution.

    Args:
        code: Implementation code string.
        tests: Test code string.

    Returns:
        Estimated coverage as float between 0.0 and 1.0.
        Returns 1.0 if code is empty (nothing to cover).
        Returns 0.0 if tests are empty or code is unparseable.

    Example:
        >>> code = "def add(a, b): return a + b"
        >>> tests = "def test_add(): assert add(1, 2) == 3"
        >>> coverage = calculate_coverage_estimate(code, tests)
        >>> 0.0 <= coverage <= 1.0
        True
    """
    # Handle empty cases
    if not code or not code.strip():
        return 1.0  # Nothing to cover
    if not tests or not tests.strip():
        return 0.0

    # Parse code to find functions
    try:
        code_tree = ast.parse(code)
    except SyntaxError:
        logger.debug("coverage_estimate_syntax_error_in_code")
        return 0.0

    # Get all public function names from implementation
    impl_functions: set[str] = set()
    for node in ast.walk(code_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                impl_functions.add(node.name)

    if not impl_functions:
        return 1.0  # No functions to cover

    # Parse tests to find what functions are tested
    try:
        test_tree = ast.parse(tests)
    except SyntaxError:
        logger.debug("coverage_estimate_syntax_error_in_tests")
        return 0.0

    # Count test functions and what they reference
    tested_functions: set[str] = set()
    test_count = 0

    for node in ast.walk(test_tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("test_"):
                test_count += 1
                # Check if test name hints at what function is tested
                for impl_func in impl_functions:
                    if impl_func in node.name.lower():
                        tested_functions.add(impl_func)

                # Check for function calls in the test body
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        # Direct function calls: func_name()
                        if isinstance(child.func, ast.Name):
                            if child.func.id in impl_functions:
                                tested_functions.add(child.func.id)
                        # Method calls: obj.method_name() or Class.method_name()
                        elif isinstance(child.func, ast.Attribute):
                            if child.func.attr in impl_functions:
                                tested_functions.add(child.func.attr)

    # Calculate coverage ratio
    func_coverage = len(tested_functions) / len(impl_functions)

    # Bonus for having multiple tests per function
    test_density = min(test_count / len(impl_functions), 2.0) / 2.0

    # Combine metrics (weighted)
    coverage = (func_coverage * 0.7) + (test_density * 0.3)

    # Clamp to valid range
    coverage = max(0.0, min(1.0, coverage))

    logger.debug(
        "coverage_estimate_calculated",
        impl_function_count=len(impl_functions),
        tested_function_count=len(tested_functions),
        test_count=test_count,
        coverage=coverage,
    )

    return coverage


def check_coverage_threshold(
    coverage: float,
    threshold: float,
) -> tuple[bool, str]:
    """Check if coverage meets the configured threshold (AC4).

    Compares the calculated coverage against the threshold and returns
    a pass/fail result with an appropriate warning message.

    Args:
        coverage: Calculated coverage value (0.0 to 1.0).
        threshold: Required coverage threshold (0.0 to 1.0).

    Returns:
        Tuple of (passes, warning_message).
        passes is True if coverage >= threshold.
        warning_message is empty string if passes, else describes the gap.

    Example:
        >>> passes, msg = check_coverage_threshold(0.60, 0.80)
        >>> passes
        False
        >>> "60" in msg or "0.6" in msg
        True
    """
    passes = coverage >= threshold

    if passes:
        logger.debug(
            "coverage_threshold_passed",
            coverage=coverage,
            threshold=threshold,
        )
        return True, ""

    # Generate warning message
    coverage_pct = int(coverage * 100)
    threshold_pct = int(threshold * 100)
    gap_pct = threshold_pct - coverage_pct

    message = (
        f"Test coverage ({coverage_pct}%) is below the required threshold "
        f"({threshold_pct}%). Gap: {gap_pct}%."
    )

    logger.warning(
        "coverage_threshold_failed",
        coverage=coverage,
        threshold=threshold,
        gap=gap_pct,
    )

    return False, message


# =============================================================================
# Test Quality Validation (AC3)
# =============================================================================


def validate_test_quality(test_code: str) -> QualityReport:
    """Validate test code quality for isolation and determinism (AC3).

    Analyzes test code for quality issues:
    - Missing assertions in test functions
    - Use of random without seeding (non-deterministic)
    - Use of time.time() without mocking (non-deterministic)
    - Global state mutations that may affect isolation
    - Proper use of fixtures

    Args:
        test_code: Python test code string to analyze.

    Returns:
        QualityReport with quality assessment and warnings.

    Example:
        >>> tests = "def test_x(): assert True"
        >>> report = validate_test_quality(tests)
        >>> report.is_acceptable()
        True
    """
    report = QualityReport()

    if not test_code or not test_code.strip():
        return report

    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        logger.debug("test_quality_validation_syntax_error")
        return report

    # Analyze test functions
    _check_assertions(tree, report)
    _check_determinism(tree, test_code, report)
    _check_fixtures(tree, test_code, report)
    _check_isolation(tree, test_code, report)

    logger.debug(
        "test_quality_validated",
        has_assertions=report.has_assertions,
        is_deterministic=report.is_deterministic,
        uses_fixtures=report.uses_fixtures,
        warning_count=len(report.warnings),
    )

    return report


def _check_assertions(tree: ast.AST, report: QualityReport) -> None:
    """Check that test functions have assertions."""
    test_functions_found = False
    tests_with_assertions = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            test_functions_found = True
            has_assert = False

            for child in ast.walk(node):
                if isinstance(child, ast.Assert):
                    has_assert = True
                    break
                # Also check for pytest.raises
                if isinstance(child, ast.With):
                    for item in child.items:
                        if isinstance(item.context_expr, ast.Call):
                            call = item.context_expr
                            if isinstance(call.func, ast.Attribute):
                                if call.func.attr == "raises":
                                    has_assert = True
                                    break

            if has_assert:
                tests_with_assertions += 1

    if test_functions_found and tests_with_assertions == 0:
        report.has_assertions = False
        report.warnings.append("Test functions appear to lack assertions")


def _check_determinism(tree: ast.AST, test_code: str, report: QualityReport) -> None:
    """Check for non-deterministic patterns."""
    # Check for random usage without seeding
    if "import random" in test_code or "from random" in test_code:
        if "random.seed" not in test_code:
            report.warnings.append(
                "Uses 'random' module without seeding - may cause non-deterministic tests"
            )
            report.is_deterministic = False

    # Check for time.time() usage
    if "time.time()" in test_code and "mock" not in test_code.lower():
        report.warnings.append(
            "Uses 'time.time()' without mocking - may cause non-deterministic tests"
        )

    # Check for datetime.now() usage
    if "datetime.now()" in test_code and "mock" not in test_code.lower():
        report.warnings.append(
            "Uses 'datetime.now()' without mocking - may cause non-deterministic tests"
        )


def _check_fixtures(tree: ast.AST, test_code: str, report: QualityReport) -> None:
    """Check for fixture usage."""
    # Check for @pytest.fixture decorator
    if "@pytest.fixture" in test_code or "pytest.fixture" in test_code:
        report.uses_fixtures = True

    # Check for fixture parameters in test functions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            # If test function has parameters (other than self), likely uses fixtures
            params = [arg.arg for arg in node.args.args if arg.arg != "self"]
            if params:
                report.uses_fixtures = True
                break


def _check_isolation(tree: ast.AST, test_code: str, report: QualityReport) -> None:
    """Check for potential isolation issues."""
    # Check for module-level mutable state
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            # Check if it's a mutable type assignment (list, dict, set)
            if isinstance(node.value, (ast.List, ast.Dict, ast.Set)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper():
                            report.warnings.append(
                                f"Module-level mutable variable '{target.id}' may cause test isolation issues"
                            )
