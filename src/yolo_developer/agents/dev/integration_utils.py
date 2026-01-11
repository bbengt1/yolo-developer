"""Integration test utilities for Dev agent (Story 8.4).

This module provides utility functions for analyzing code to generate
integration tests:
- Component boundary analysis: Detect interactions between modules
- Data flow analysis: Trace data through function calls
- Error scenario detection: Identify error handling patterns

Key Functions:
    - analyze_component_boundaries: Detect component interaction boundaries
    - analyze_data_flow: Trace data flow through functions
    - detect_error_scenarios: Identify error handling patterns
    - validate_integration_test_quality: Check integration test quality

Example:
    >>> from yolo_developer.agents.dev.integration_utils import (
    ...     analyze_component_boundaries,
    ...     ComponentBoundary,
    ... )
    >>>
    >>> boundaries = analyze_component_boundaries([code_file_a, code_file_b])
    >>> len(boundaries)
    2
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from yolo_developer.agents.dev.types import CodeFile
    from yolo_developer.llm.router import LLMRouter

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Types (AC1, AC2, AC3, AC7)
# =============================================================================


@dataclass(frozen=True)
class ComponentBoundary:
    """Represents an interaction boundary between components (AC1, AC7).

    Attributes:
        source_file: Path to the source component file.
        target_file: Path to the target component file.
        interaction_type: Type of interaction (import, function_call, etc.).
        boundary_point: Specific function/class at the boundary.
        is_async: Whether the interaction is async.

    Example:
        >>> boundary = ComponentBoundary(
        ...     source_file="src/caller.py",
        ...     target_file="src/callee.py",
        ...     interaction_type="function_call",
        ...     boundary_point="process_data",
        ...     is_async=False,
        ... )
        >>> boundary.interaction_type
        'function_call'
    """

    source_file: str
    target_file: str
    interaction_type: Literal["import", "function_call", "state_access", "class_instantiation"]
    boundary_point: str
    is_async: bool


@dataclass(frozen=True)
class DataFlowPath:
    """Represents a data flow path through components (AC2).

    Attributes:
        start_point: Where data enters the flow.
        end_point: Where data exits the flow.
        steps: Sequence of transformation steps.
        data_types: Types observed at each step.

    Example:
        >>> flow = DataFlowPath(
        ...     start_point="user_input",
        ...     end_point="database_write",
        ...     steps=("validate", "transform", "save"),
        ...     data_types=("str", "dict", "Model"),
        ... )
        >>> len(flow.steps)
        3
    """

    start_point: str
    end_point: str
    steps: tuple[str, ...]
    data_types: tuple[str, ...]


@dataclass(frozen=True)
class ErrorScenario:
    """Represents an error scenario to test (AC3).

    Attributes:
        trigger: What triggers the error condition.
        handling: How the error is handled.
        recovery: Recovery mechanism if any.
        exception_type: Type of exception raised/caught.

    Example:
        >>> scenario = ErrorScenario(
        ...     trigger="invalid_input",
        ...     handling="try/except",
        ...     recovery="return default",
        ...     exception_type="ValueError",
        ... )
        >>> scenario.exception_type
        'ValueError'
    """

    trigger: str
    handling: str
    recovery: str | None
    exception_type: str | None


@dataclass
class IntegrationTestQualityReport:
    """Report of integration test quality analysis (AC3, AC4).

    Note: Not frozen because warnings are appended incrementally during analysis.

    Attributes:
        warnings: List of quality warnings.
        uses_fixtures: Whether tests use pytest fixtures.
        uses_mocks: Whether external dependencies are mocked.
        has_cleanup: Whether tests clean up state.
        is_async_compliant: Whether async tests have proper markers.

    Example:
        >>> report = IntegrationTestQualityReport(warnings=[])
        >>> report.uses_fixtures = True
        >>> report.is_acceptable()
        False  # needs mocks and cleanup too
    """

    warnings: list[str] = field(default_factory=list)
    uses_fixtures: bool = False
    uses_mocks: bool = False
    has_cleanup: bool = False
    is_async_compliant: bool = True

    def is_acceptable(self) -> bool:
        """Check if test quality is acceptable.

        Integration tests are acceptable if they use fixtures, mocks,
        have cleanup patterns, and async tests have proper markers.

        Returns:
            True if test quality meets minimum standards.
        """
        return (
            self.uses_fixtures
            and self.uses_mocks
            and self.has_cleanup
            and self.is_async_compliant
        )


# =============================================================================
# Component Boundary Analysis (AC1, AC7)
# =============================================================================


def analyze_component_boundaries(code_files: list[CodeFile]) -> list[ComponentBoundary]:
    """Analyze code files to detect component interaction boundaries (AC1, AC7).

    Uses AST parsing to identify:
    1. Import statements between files
    2. Function calls to imported modules
    3. Class instantiations from other modules
    4. Async calls (await expressions)

    Args:
        code_files: List of CodeFile objects to analyze.

    Returns:
        List of ComponentBoundary objects representing detected boundaries.

    Example:
        >>> from yolo_developer.agents.dev.types import CodeFile
        >>> file_a = CodeFile(
        ...     file_path="src/caller.py",
        ...     content="from module_b import helper",
        ...     file_type="source",
        ... )
        >>> boundaries = analyze_component_boundaries([file_a])
    """
    if not code_files:
        return []

    boundaries: list[ComponentBoundary] = []

    # Build map of file -> module name for internal import detection
    file_modules: dict[str, str] = {}
    for cf in code_files:
        if cf.file_type == "source":
            module_name = _extract_module_name(cf.file_path)
            file_modules[cf.file_path] = module_name

    # Also build reverse map: module name -> file path
    module_to_file: dict[str, str] = {v: k for k, v in file_modules.items()}

    for code_file in code_files:
        if code_file.file_type != "source":
            continue

        try:
            tree = ast.parse(code_file.content)
        except SyntaxError:
            logger.debug(
                "boundary_analysis_syntax_error",
                file_path=code_file.file_path,
            )
            continue

        # Track imported names for function call detection
        imported_names: dict[str, str] = {}  # name -> module

        # Analyze the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Handle: import module
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    # Check if this is an internal import
                    if alias.name in module_to_file:
                        target_file = module_to_file[alias.name]
                        boundaries.append(ComponentBoundary(
                            source_file=code_file.file_path,
                            target_file=target_file,
                            interaction_type="import",
                            boundary_point=alias.name,
                            is_async=False,
                        ))
                        imported_names[name] = alias.name

            elif isinstance(node, ast.ImportFrom):
                # Handle: from module import name
                if node.module:
                    # Check if this is an internal import
                    if node.module in module_to_file:
                        target_file = module_to_file[node.module]
                        for alias in node.names:
                            name = alias.asname if alias.asname else alias.name
                            boundaries.append(ComponentBoundary(
                                source_file=code_file.file_path,
                                target_file=target_file,
                                interaction_type="import",
                                boundary_point=alias.name,
                                is_async=False,
                            ))
                            imported_names[name] = node.module

            elif isinstance(node, ast.Call):
                # Detect function calls to imported names
                func_name = _get_call_name(node)
                if func_name and func_name in imported_names:
                    module_name = imported_names[func_name]
                    if module_name in module_to_file:
                        target_file = module_to_file[module_name]
                        is_async = _is_awaited_call(node, tree)
                        boundaries.append(ComponentBoundary(
                            source_file=code_file.file_path,
                            target_file=target_file,
                            interaction_type="function_call",
                            boundary_point=func_name,
                            is_async=is_async,
                        ))

    logger.debug(
        "component_boundaries_analyzed",
        file_count=len(code_files),
        boundary_count=len(boundaries),
    )

    return boundaries


def _extract_module_name(file_path: str) -> str:
    """Extract module name from file path.

    Args:
        file_path: File path like "src/module_name.py"

    Returns:
        Module name like "module_name"
    """
    # Get the filename without path
    filename = file_path.split("/")[-1]
    # Remove .py extension
    return filename.replace(".py", "")


def _get_call_name(node: ast.Call) -> str | None:
    """Get the function name from a Call node.

    Args:
        node: AST Call node.

    Returns:
        Function name if simple call, None otherwise.
    """
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_awaited_call(call_node: ast.Call, tree: ast.AST) -> bool:
    """Check if a call node is awaited.

    Args:
        call_node: The Call node to check.
        tree: Full AST tree to search.

    Returns:
        True if the call is awaited.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Await):
            if node.value is call_node:
                return True
    return False


# =============================================================================
# Data Flow Analysis (AC2)
# =============================================================================


def analyze_data_flow(code_files: list[CodeFile]) -> list[DataFlowPath]:
    """Analyze data flow through code files (AC2).

    Traces data from input to output through function calls,
    identifying transformation points and type changes.

    Args:
        code_files: List of CodeFile objects to analyze.

    Returns:
        List of DataFlowPath objects representing detected flows.

    Example:
        >>> flows = analyze_data_flow([code_file])
        >>> len(flows) >= 1
        True
    """
    if not code_files:
        return []

    flows: list[DataFlowPath] = []

    for code_file in code_files:
        if code_file.file_type != "source":
            continue

        try:
            tree = ast.parse(code_file.content)
        except SyntaxError:
            continue

        # Find functions and their data flow
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                flow = _analyze_function_flow(node)
                if flow:
                    flows.append(flow)

    logger.debug(
        "data_flow_analyzed",
        file_count=len(code_files),
        flow_count=len(flows),
    )

    return flows


def _analyze_function_flow(node: ast.FunctionDef | ast.AsyncFunctionDef) -> DataFlowPath | None:
    """Analyze data flow within a single function.

    Args:
        node: Function definition AST node.

    Returns:
        DataFlowPath if flow detected, None otherwise.
    """
    # Extract parameter types as start point
    param_types: list[str] = []
    for arg in node.args.args:
        if arg.annotation:
            param_types.append(_annotation_to_str(arg.annotation))
        else:
            param_types.append("Any")

    # Extract return type as end point
    return_type = "None"
    if node.returns:
        return_type = _annotation_to_str(node.returns)

    # Find function calls as steps
    steps: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            call_name = _get_call_name(child)
            if call_name and call_name != node.name:  # Avoid recursion
                steps.append(call_name)

    if not steps and not param_types:
        return None

    # Build start point from parameters
    start_point = f"{node.name}({', '.join(param_types)})"
    end_point = f"{node.name} -> {return_type}"

    return DataFlowPath(
        start_point=start_point,
        end_point=end_point,
        steps=tuple(steps) if steps else ("direct",),
        data_types=(*param_types, return_type) if param_types else (return_type,),
    )


def _annotation_to_str(annotation: ast.expr) -> str:
    """Convert AST annotation to string.

    Args:
        annotation: AST expression for type annotation.

    Returns:
        String representation.
    """
    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Constant):
        return str(annotation.value)
    elif isinstance(annotation, ast.Subscript):
        value = _annotation_to_str(annotation.value)
        slice_val = _annotation_to_str(annotation.slice)
        return f"{value}[{slice_val}]"
    else:
        return ast.unparse(annotation)


# =============================================================================
# Error Scenario Detection (AC3)
# =============================================================================


def detect_error_scenarios(code_files: list[CodeFile]) -> list[ErrorScenario]:
    """Detect error handling scenarios in code files (AC3).

    Identifies:
    1. try/except blocks and exception types
    2. raise statements
    3. Fallback patterns (default values on error)

    Args:
        code_files: List of CodeFile objects to analyze.

    Returns:
        List of ErrorScenario objects representing detected scenarios.

    Example:
        >>> scenarios = detect_error_scenarios([code_file])
        >>> any(s.exception_type == "ValueError" for s in scenarios)
        True
    """
    if not code_files:
        return []

    scenarios: list[ErrorScenario] = []

    for code_file in code_files:
        if code_file.file_type != "source":
            continue

        try:
            tree = ast.parse(code_file.content)
        except SyntaxError:
            continue

        # Find try/except blocks
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    scenario = _analyze_exception_handler(handler, node)
                    if scenario:
                        scenarios.append(scenario)

            elif isinstance(node, ast.Raise):
                scenario = _analyze_raise_statement(node)
                if scenario:
                    scenarios.append(scenario)

    logger.debug(
        "error_scenarios_detected",
        file_count=len(code_files),
        scenario_count=len(scenarios),
    )

    return scenarios


def _analyze_exception_handler(
    handler: ast.ExceptHandler,
    try_node: ast.Try,
) -> ErrorScenario | None:
    """Analyze an exception handler.

    Args:
        handler: ExceptHandler AST node.
        try_node: Parent Try node.

    Returns:
        ErrorScenario if valid, None otherwise.
    """
    # Get exception type
    exception_type: str | None = None
    if handler.type:
        if isinstance(handler.type, ast.Name):
            exception_type = handler.type.id
        elif isinstance(handler.type, ast.Tuple):
            # Multiple exceptions
            types = [
                e.id if isinstance(e, ast.Name) else str(e)
                for e in handler.type.elts
            ]
            exception_type = " | ".join(types)

    # Determine if there's a recovery (return statement in handler)
    recovery: str | None = None
    for child in ast.walk(handler):
        if isinstance(child, ast.Return):
            if child.value:
                recovery = "return fallback value"
            else:
                recovery = "return None"
            break

    return ErrorScenario(
        trigger="exception in try block",
        handling="try/except",
        recovery=recovery,
        exception_type=exception_type,
    )


def _analyze_raise_statement(node: ast.Raise) -> ErrorScenario | None:
    """Analyze a raise statement.

    Args:
        node: Raise AST node.

    Returns:
        ErrorScenario if valid, None otherwise.
    """
    exception_type: str | None = None

    if node.exc:
        if isinstance(node.exc, ast.Call):
            # raise ValueError("msg")
            if isinstance(node.exc.func, ast.Name):
                exception_type = node.exc.func.id
        elif isinstance(node.exc, ast.Name):
            # raise exc
            exception_type = node.exc.id

    if not exception_type:
        return None

    return ErrorScenario(
        trigger="validation failure or error condition",
        handling="raise",
        recovery=None,
        exception_type=exception_type,
    )


# =============================================================================
# Integration Test Quality Validation (AC3, AC4)
# =============================================================================


def validate_integration_test_quality(test_code: str) -> IntegrationTestQualityReport:
    """Validate integration test code quality (AC3, AC4).

    Analyzes test code for:
    - Fixture usage (pytest fixtures for shared setup)
    - Mock patterns (external dependency mocking)
    - Cleanup patterns (state restoration)
    - Async test markers (@pytest.mark.asyncio)

    Args:
        test_code: Python test code string to analyze.

    Returns:
        IntegrationTestQualityReport with quality assessment.

    Example:
        >>> report = validate_integration_test_quality(test_code)
        >>> report.uses_fixtures
        True
    """
    report = IntegrationTestQualityReport()

    if not test_code or not test_code.strip():
        return report

    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        report.warnings.append("Test code has syntax errors")
        return report

    # Check for fixture usage
    _check_fixture_usage(tree, test_code, report)

    # Check for mock usage
    _check_mock_usage(test_code, report)

    # Check for cleanup patterns
    _check_cleanup_patterns(tree, test_code, report)

    # Check async compliance
    _check_async_compliance(tree, test_code, report)

    logger.debug(
        "integration_test_quality_validated",
        uses_fixtures=report.uses_fixtures,
        uses_mocks=report.uses_mocks,
        has_cleanup=report.has_cleanup,
        is_async_compliant=report.is_async_compliant,
        warning_count=len(report.warnings),
    )

    return report


def _check_fixture_usage(tree: ast.AST, test_code: str, report: IntegrationTestQualityReport) -> None:
    """Check for pytest fixture usage."""
    # Check for @pytest.fixture decorator
    if "@pytest.fixture" in test_code:
        report.uses_fixtures = True
        return

    # Check for fixture parameters in test functions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            # If test function has parameters (other than self), likely uses fixtures
            params = [arg.arg for arg in node.args.args if arg.arg != "self"]
            if params:
                report.uses_fixtures = True
                return


def _check_mock_usage(test_code: str, report: IntegrationTestQualityReport) -> None:
    """Check for mock/patch usage."""
    mock_patterns = [
        "MagicMock",
        "AsyncMock",
        "patch",
        "mock.Mock",
        "unittest.mock",
        "@patch",
    ]

    for pattern in mock_patterns:
        if pattern in test_code:
            report.uses_mocks = True
            return


def _check_cleanup_patterns(tree: ast.AST, test_code: str, report: IntegrationTestQualityReport) -> None:
    """Check for cleanup/teardown patterns."""
    cleanup_indicators = [
        "autouse=True",  # Fixtures with auto cleanup
        "yield",  # Yield fixtures with cleanup
        "teardown",
        "cleanup",
        "finally:",
    ]

    for indicator in cleanup_indicators:
        if indicator in test_code:
            report.has_cleanup = True
            return

    # Check for fixture with yield (generator fixture)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for child in ast.walk(node):
                if isinstance(child, ast.Yield):
                    report.has_cleanup = True
                    return


def _check_async_compliance(tree: ast.AST, test_code: str, report: IntegrationTestQualityReport) -> None:
    """Check async test compliance with pytest-asyncio."""
    # Find all async test functions
    async_tests: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name.startswith("test_"):
            async_tests.append(node.name)

    if not async_tests:
        # No async tests, so compliant by default
        return

    # Check if pytest.mark.asyncio is present
    has_asyncio_marker = "@pytest.mark.asyncio" in test_code

    if not has_asyncio_marker:
        report.is_async_compliant = False
        report.warnings.append(
            f"Async test functions {async_tests} may need @pytest.mark.asyncio marker"
        )


# =============================================================================
# LLM Integration Test Generation (AC6)
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
    """Call LLM with tenacity retry and exponential backoff (AC6, ADR-007).

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


async def generate_integration_tests_with_llm(
    code_files: list[CodeFile],
    boundaries: list[ComponentBoundary],
    flows: list[DataFlowPath],
    error_scenarios: list[ErrorScenario],
    router: LLMRouter,
    additional_context: str = "",
    max_retries: int = 2,
) -> tuple[str, bool]:
    """Generate integration tests using LLM with syntax validation (AC6).

    Calls the LLM to generate pytest integration tests for the given code files,
    using the detected boundaries, flows, and error scenarios to guide generation.
    Validates the generated test syntax and retries with error context if needed.

    Uses tenacity with exponential backoff for transient LLM failures (ADR-007).
    Syntax validation failures trigger retry with modified prompt.

    Args:
        code_files: List of CodeFile objects to generate tests for.
        boundaries: Detected component boundaries.
        flows: Detected data flow paths.
        error_scenarios: Detected error scenarios.
        router: LLMRouter instance for making LLM calls.
        additional_context: Extra context for test generation. Defaults to empty.
        max_retries: Maximum retries on syntax validation failure. Defaults to 2.

    Returns:
        Tuple of (test_code, is_valid). test_code is the generated test code,
        is_valid indicates if the code passed syntax validation.

    Example:
        >>> code, is_valid = await generate_integration_tests_with_llm(
        ...     code_files=[...],
        ...     boundaries=[...],
        ...     flows=[...],
        ...     error_scenarios=[...],
        ...     router=router,
        ... )
    """
    from yolo_developer.agents.dev.code_utils import (
        extract_code_from_response,
        validate_python_syntax,
    )
    from yolo_developer.agents.dev.prompts.integration_test_generation import (
        build_integration_test_prompt,
        build_integration_test_retry_prompt,
    )

    # Build code files content
    code_content_parts = []
    for cf in code_files:
        if cf.file_type == "source":
            code_content_parts.append(f"# File: {cf.file_path}\n{cf.content}")
    code_files_content = "\n\n".join(code_content_parts)

    # Format boundaries for prompt
    boundaries_text = "\n".join(
        f"- {b.source_file} -> {b.target_file}: {b.interaction_type} at {b.boundary_point}"
        + (" (async)" if b.is_async else "")
        for b in boundaries
    ) if boundaries else "(No boundaries detected)"

    # Format flows for prompt
    flows_text = "\n".join(
        f"- {f.start_point} -> {f.end_point}: {' -> '.join(f.steps)}"
        for f in flows
    ) if flows else "(No flows detected)"

    # Format error scenarios for prompt
    error_text = "\n".join(
        f"- {s.exception_type or 'Unknown'}: {s.trigger} ({s.handling})"
        + (f" -> {s.recovery}" if s.recovery else "")
        for s in error_scenarios
    ) if error_scenarios else "(No error scenarios detected)"

    # Build initial prompt
    prompt = build_integration_test_prompt(
        code_files_content=code_files_content,
        boundaries=boundaries_text,
        data_flows=flows_text,
        error_scenarios=error_text,
        additional_context=additional_context,
    )

    logger.info(
        "llm_integration_test_generation_start",
        file_count=len(code_files),
        boundary_count=len(boundaries),
        flow_count=len(flows),
        error_scenario_count=len(error_scenarios),
        prompt_length=len(prompt),
    )

    try:
        # Call LLM with tenacity retry (ADR-007)
        response = await _call_llm_with_retry(router, prompt)
    except Exception as e:
        logger.error("llm_integration_test_generation_failed", error=str(e))
        return "", False

    # Extract code from response
    test_code = extract_code_from_response(response)
    is_valid, error = validate_python_syntax(test_code)

    if is_valid:
        logger.info(
            "integration_test_generation_success",
            test_code_length=len(test_code),
        )
        return test_code, True

    # Retry with error context for syntax validation failures
    for attempt in range(max_retries):
        logger.info(
            "integration_test_generation_syntax_retry",
            attempt=attempt + 1,
            max_retries=max_retries,
            error=error,
        )

        retry_prompt = build_integration_test_retry_prompt(
            original_prompt=prompt,
            error_message=error or "Unknown syntax error",
            previous_tests=test_code,
        )

        try:
            # LLM call with tenacity retry
            response = await _call_llm_with_retry(router, retry_prompt)
        except Exception as e:
            logger.error("llm_integration_test_generation_retry_failed", error=str(e))
            continue

        test_code = extract_code_from_response(response)
        is_valid, error = validate_python_syntax(test_code)

        if is_valid:
            logger.info(
                "integration_test_generation_success_after_retry",
                attempt=attempt + 1,
            )
            return test_code, True

    # Return best attempt even if invalid
    logger.warning(
        "integration_test_generation_validation_failed",
        final_error=error,
    )
    return test_code, False


