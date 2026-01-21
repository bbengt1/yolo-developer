"""Documentation analysis and generation utilities for Dev agent (Story 8.5).

This module provides utilities for analyzing code documentation status,
detecting complex sections needing comments, validating documentation
quality, and generating documentation with LLM assistance.

Key Functions:
    - extract_documentation_info: Analyze code to identify documentation gaps
    - detect_complex_sections: Find code sections needing explanatory comments
    - validate_documentation_quality: Check docstring completeness and quality
    - generate_documentation_with_llm: LLM-powered documentation enhancement

Example:
    >>> from yolo_developer.agents.dev.doc_utils import (
    ...     extract_documentation_info,
    ...     validate_documentation_quality,
    ... )
    >>>
    >>> code = '''
    ... def hello():
    ...     pass
    ... '''
    >>> info = extract_documentation_info(code)
    >>> info.has_module_docstring
    False
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.dev.code_utils import (
    extract_code_from_response,
    validate_python_syntax,
)
from yolo_developer.agents.dev.prompts.documentation_generation import (
    build_documentation_prompt,
    build_documentation_retry_prompt,
)

if TYPE_CHECKING:
    from yolo_developer.llm.router import LLMRouter

logger = structlog.get_logger(__name__)

# =============================================================================
# Type Definitions (AC1, AC2, AC6)
# =============================================================================


@dataclass(frozen=True)
class ComplexSection:
    """Represents a complex code section that may need comments.

    Complex sections include nested loops, long functions, and complex
    conditionals that benefit from explanatory comments.

    Attributes:
        start_line: Line number where complex section starts.
        end_line: Line number where complex section ends.
        complexity_type: Type of complexity detected.
        function_name: Name of containing function, if any.
        description: Brief description of the complexity.

    Example:
        >>> section = ComplexSection(
        ...     start_line=10,
        ...     end_line=25,
        ...     complexity_type="nested_loop",
        ...     function_name="process",
        ...     description="Nested for loops iterating over matrix",
        ... )
        >>> section.complexity_type
        'nested_loop'
    """

    start_line: int
    end_line: int
    complexity_type: Literal["nested_loop", "long_function", "complex_conditional", "deep_nesting"]
    function_name: str | None
    description: str


@dataclass(frozen=True)
class DocumentationInfo:
    """Analysis results for code documentation status.

    Provides information about documentation coverage and gaps in a
    code module, including missing docstrings and complex sections.

    Attributes:
        has_module_docstring: Whether module has a docstring.
        functions_missing_docstrings: Tuple of function names without docstrings.
        functions_with_incomplete_docstrings: Functions missing Args/Returns/Example.
        complex_sections: Tuple of code sections needing explanatory comments.
        total_public_functions: Total count of public functions.
        documented_functions: Count of functions with docstrings.

    Example:
        >>> info = DocumentationInfo(
        ...     has_module_docstring=True,
        ...     functions_missing_docstrings=("func1",),
        ...     functions_with_incomplete_docstrings=(),
        ...     complex_sections=(),
        ...     total_public_functions=2,
        ...     documented_functions=1,
        ... )
        >>> info.documentation_coverage
        50.0
    """

    has_module_docstring: bool
    functions_missing_docstrings: tuple[str, ...]
    functions_with_incomplete_docstrings: tuple[str, ...]
    complex_sections: tuple[ComplexSection, ...]
    total_public_functions: int
    documented_functions: int

    @property
    def documentation_coverage(self) -> float:
        """Calculate documentation coverage percentage.

        Returns:
            Percentage of public functions with docstrings.
            Returns 100.0 if there are no public functions.
        """
        if self.total_public_functions == 0:
            return 100.0
        return (self.documented_functions / self.total_public_functions) * 100


@dataclass
class DocumentationQualityReport:
    """Report of documentation quality analysis.

    Tracks docstring completeness metrics and generates warnings
    for documentation issues. Not frozen because warnings are
    appended incrementally during analysis.

    Attributes:
        warnings: List of quality warnings.
        has_module_docstring: Whether module docstring exists.
        functions_with_args: Count of functions with Args section.
        functions_with_returns: Count of functions with Returns section.
        functions_with_examples: Count of functions with Example section.
        total_functions: Total functions analyzed.
        type_consistency_issues: Functions where docstring types don't match annotations.

    Example:
        >>> report = DocumentationQualityReport(has_module_docstring=True)
        >>> report.is_acceptable()
        True
    """

    warnings: list[str] = field(default_factory=list)
    has_module_docstring: bool = False
    functions_with_args: int = 0
    functions_with_returns: int = 0
    functions_with_examples: int = 0
    total_functions: int = 0
    type_consistency_issues: list[str] = field(default_factory=list)

    def is_acceptable(self) -> bool:
        """Check if documentation quality is acceptable.

        Documentation is acceptable if module has docstring and
        at least 80% of functions have Args and Returns sections.

        Returns:
            True if documentation meets minimum quality standards.
        """
        if not self.has_module_docstring:
            return False
        if self.total_functions == 0:
            return True
        args_coverage = self.functions_with_args / self.total_functions
        returns_coverage = self.functions_with_returns / self.total_functions
        return args_coverage >= 0.8 and returns_coverage >= 0.8


# =============================================================================
# Documentation Analysis (AC1, AC6)
# =============================================================================


def extract_documentation_info(code: str) -> DocumentationInfo:
    """Analyze code to extract documentation status information.

    Uses AST parsing to identify missing docstrings, incomplete
    documentation, and complex sections needing comments.

    Args:
        code: Python source code to analyze.

    Returns:
        DocumentationInfo with analysis results.

    Example:
        >>> code = '''\"\"\"Module doc.\"\"\"
        ... def func():
        ...     pass
        ... '''
        >>> info = extract_documentation_info(code)
        >>> info.has_module_docstring
        True
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.warning("syntax_error_in_documentation_analysis")
        return DocumentationInfo(
            has_module_docstring=False,
            functions_missing_docstrings=(),
            functions_with_incomplete_docstrings=(),
            complex_sections=(),
            total_public_functions=0,
            documented_functions=0,
        )

    # Check module docstring
    has_module_docstring = ast.get_docstring(tree) is not None

    # Analyze functions
    functions_missing: list[str] = []
    functions_incomplete: list[str] = []
    total_public = 0
    documented = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private functions (starting with _)
            if node.name.startswith("_"):
                continue

            total_public += 1
            docstring = ast.get_docstring(node)

            if docstring is None:
                functions_missing.append(node.name)
            else:
                documented += 1
                # Check for incomplete docstrings
                if not _has_complete_docstring(docstring):
                    functions_incomplete.append(node.name)

    # Detect complex sections
    complex_sections = detect_complex_sections(code)

    return DocumentationInfo(
        has_module_docstring=has_module_docstring,
        functions_missing_docstrings=tuple(functions_missing),
        functions_with_incomplete_docstrings=tuple(functions_incomplete),
        complex_sections=tuple(complex_sections),
        total_public_functions=total_public,
        documented_functions=documented,
    )


def _has_complete_docstring(docstring: str) -> bool:
    """Check if docstring has Args and Returns sections.

    Args:
        docstring: The docstring to check.

    Returns:
        True if docstring has both Args and Returns sections.
    """
    docstring_lower = docstring.lower()
    has_args = "args:" in docstring_lower or "arguments:" in docstring_lower
    has_returns = "returns:" in docstring_lower or "return:" in docstring_lower
    return has_args and has_returns


# =============================================================================
# Complex Section Detection (AC2)
# =============================================================================


def detect_complex_sections(code: str) -> list[ComplexSection]:
    """Detect complex code sections that may need explanatory comments.

    Analyzes code for nested loops, long functions, and complex
    conditionals that would benefit from documentation.

    Args:
        code: Python source code to analyze.

    Returns:
        List of ComplexSection objects describing detected complexity.

    Example:
        >>> code = '''
        ... def process():
        ...     for i in range(10):
        ...         for j in range(10):
        ...             print(i, j)
        ... '''
        >>> sections = detect_complex_sections(code)
        >>> len(sections) >= 1
        True
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.warning("syntax_error_in_complexity_analysis")
        return []

    sections: list[ComplexSection] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check for long functions
            if node.end_lineno is not None and node.lineno is not None:
                func_length = node.end_lineno - node.lineno
                if func_length > 20:
                    sections.append(
                        ComplexSection(
                            start_line=node.lineno,
                            end_line=node.end_lineno,
                            complexity_type="long_function",
                            function_name=node.name,
                            description=f"Function with {func_length} lines",
                        )
                    )

            # Check for nested loops and conditionals within function
            _detect_nesting_in_function(node, sections)

    return sections


def _detect_nesting_in_function(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    sections: list[ComplexSection],
) -> None:
    """Detect nested structures within a function.

    Args:
        func_node: The function node to analyze.
        sections: List to append detected sections to.
    """
    for node in ast.walk(func_node):
        # Detect nested loops
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if child is not node and isinstance(child, (ast.For, ast.While)):
                    if node.lineno is not None and node.end_lineno is not None:
                        sections.append(
                            ComplexSection(
                                start_line=node.lineno,
                                end_line=node.end_lineno,
                                complexity_type="nested_loop",
                                function_name=func_node.name,
                                description="Nested loop structure",
                            )
                        )
                    break  # Only count once per outer loop

        # Detect deeply nested conditionals
        if isinstance(node, ast.If):
            depth = _count_if_depth(node)
            if depth >= 3:
                if node.lineno is not None and node.end_lineno is not None:
                    sections.append(
                        ComplexSection(
                            start_line=node.lineno,
                            end_line=node.end_lineno,
                            complexity_type="deep_nesting",
                            function_name=func_node.name,
                            description=f"Conditional nesting depth of {depth}",
                        )
                    )


def _count_if_depth(node: ast.If, current_depth: int = 1) -> int:
    """Count the maximum nesting depth of if statements.

    Args:
        node: The if statement node to analyze.
        current_depth: Current depth level.

    Returns:
        Maximum depth of nested if statements.
    """
    max_depth = current_depth

    for child in node.body:
        if isinstance(child, ast.If):
            child_depth = _count_if_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)

    for child in node.orelse:
        if isinstance(child, ast.If):
            child_depth = _count_if_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)

    return max_depth


# =============================================================================
# Documentation Quality Validation (AC4, AC6)
# =============================================================================


def validate_documentation_quality(code: str) -> DocumentationQualityReport:
    """Validate documentation quality and generate report.

    Checks docstring completeness including Args, Returns, and Example
    sections. Generates warnings for documentation issues.

    Args:
        code: Python source code to validate.

    Returns:
        DocumentationQualityReport with quality metrics and warnings.

    Example:
        >>> code = '''\"\"\"Module doc.\"\"\"
        ... def func():
        ...     \"\"\"Has docstring.\"\"\"
        ...     pass
        ... '''
        >>> report = validate_documentation_quality(code)
        >>> report.has_module_docstring
        True
    """
    report = DocumentationQualityReport()

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        report.warnings.append(f"Syntax error in code: {e}")
        return report

    # Check module docstring
    module_docstring = ast.get_docstring(tree)
    report.has_module_docstring = module_docstring is not None
    if not report.has_module_docstring:
        report.warnings.append("Missing module docstring")

    # Analyze function docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private functions
            if node.name.startswith("_"):
                continue

            report.total_functions += 1
            docstring = ast.get_docstring(node)

            if docstring is None:
                report.warnings.append(f"Function '{node.name}' missing docstring")
                continue

            # Check for sections
            docstring_lower = docstring.lower()

            if "args:" in docstring_lower or "arguments:" in docstring_lower:
                report.functions_with_args += 1
            else:
                # Only warn if function has parameters
                if node.args.args or node.args.kwonlyargs:
                    report.warnings.append(f"Function '{node.name}' missing Args section")

            if "returns:" in docstring_lower or "return:" in docstring_lower:
                report.functions_with_returns += 1
            else:
                # Only warn if function has return annotation
                if node.returns is not None:
                    report.warnings.append(f"Function '{node.name}' missing Returns section")

            if (
                "example:" in docstring_lower
                or "examples:" in docstring_lower
                or ">>>" in docstring
            ):
                report.functions_with_examples += 1

    return report


# =============================================================================
# LLM Documentation Generation (AC5)
# =============================================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _call_llm_for_documentation(
    prompt: str,
    router: LLMRouter,
) -> str:
    """Call LLM for documentation generation with retry.

    Args:
        prompt: The documentation generation prompt.
        router: LLM router instance.

    Returns:
        LLM response string.
    """
    messages = [{"role": "user", "content": prompt}]
    response = await router.call_task(messages=messages, task_type="documentation")
    return response


async def generate_documentation_with_llm(
    code: str,
    context: str,
    router: LLMRouter,
    max_retries: int = 2,
) -> tuple[str, bool]:
    """Generate documentation for code using LLM.

    Uses LLM to enhance code with comprehensive documentation including
    module docstrings, function docstrings, and explanatory comments.

    Args:
        code: Python source code to document.
        context: Context about the code's purpose.
        router: LLM router for making API calls.
        max_retries: Maximum retry attempts for syntax errors. Defaults to 2.

    Returns:
        Tuple of (documented_code, is_valid) where is_valid indicates
        if the generated code passes syntax validation.

    Raises:
        Exception: If LLM call fails after all retry attempts.

    Note:
        Uses tenacity retry with exponential backoff per ADR-007.
        On persistent failure, returns original code with is_valid=False.
    """
    # Analyze current documentation status
    doc_info = extract_documentation_info(code)
    complex_sections = detect_complex_sections(code)

    # Format analysis for prompt
    analysis_text = format_documentation_info_for_prompt(doc_info)
    complex_text = format_complex_sections_for_prompt(complex_sections)

    # Build initial prompt
    prompt = build_documentation_prompt(
        code_content=code,
        documentation_analysis=analysis_text,
        complex_sections=complex_text,
        additional_context=context,
    )

    # Try to generate documentation
    for attempt in range(max_retries + 1):
        try:
            logger.info(
                "generating_documentation",
                attempt=attempt + 1,
                max_retries=max_retries,
            )

            response = await _call_llm_for_documentation(prompt, router)
            documented_code = extract_code_from_response(response)

            # Validate syntax
            is_valid, error = validate_python_syntax(documented_code)

            if is_valid:
                logger.info("documentation_generated_successfully")
                return documented_code, True

            # Syntax error - build retry prompt
            if attempt < max_retries:
                logger.warning(
                    "documentation_syntax_error_retrying",
                    error=error,
                    attempt=attempt + 1,
                )
                prompt = build_documentation_retry_prompt(
                    original_prompt=prompt,
                    error_message=error or "Unknown syntax error",
                    previous_code=documented_code,
                )
            else:
                logger.error(
                    "documentation_generation_failed_syntax",
                    error=error,
                )
                return documented_code, False

        except Exception as e:
            logger.error("documentation_generation_error", error=str(e))
            if attempt == max_retries:
                # Return original code on final failure
                return code, False

    return code, False


# =============================================================================
# Formatting Utilities
# =============================================================================


def format_documentation_info_for_prompt(info: DocumentationInfo) -> str:
    """Format DocumentationInfo for inclusion in LLM prompt.

    Args:
        info: Documentation analysis results.

    Returns:
        Formatted string describing documentation status.

    Example:
        >>> info = DocumentationInfo(
        ...     has_module_docstring=False,
        ...     functions_missing_docstrings=("func1",),
        ...     functions_with_incomplete_docstrings=(),
        ...     complex_sections=(),
        ...     total_public_functions=1,
        ...     documented_functions=0,
        ... )
        >>> formatted = format_documentation_info_for_prompt(info)
        >>> "module docstring" in formatted.lower()
        True
    """
    lines = []

    # Module docstring status
    if info.has_module_docstring:
        lines.append("✓ Module has docstring")
    else:
        lines.append("✗ Missing module docstring")

    # Coverage
    lines.append(f"Documentation coverage: {info.documentation_coverage:.1f}%")
    lines.append(f"Total public functions: {info.total_public_functions}")
    lines.append(f"Documented functions: {info.documented_functions}")

    # Missing docstrings
    if info.functions_missing_docstrings:
        lines.append("\nFunctions missing docstrings:")
        for func in info.functions_missing_docstrings:
            lines.append(f"  - {func}")

    # Incomplete docstrings
    if info.functions_with_incomplete_docstrings:
        lines.append("\nFunctions with incomplete docstrings:")
        for func in info.functions_with_incomplete_docstrings:
            lines.append(f"  - {func}")

    return "\n".join(lines)


def format_complex_sections_for_prompt(sections: list[ComplexSection]) -> str:
    """Format complex sections list for inclusion in LLM prompt.

    Args:
        sections: List of detected complex sections.

    Returns:
        Formatted string describing complex sections.

    Example:
        >>> sections = []
        >>> formatted = format_complex_sections_for_prompt(sections)
        >>> "none" in formatted.lower() or "no complex" in formatted.lower()
        True
    """
    if not sections:
        return "No complex sections detected that require additional comments."

    lines = ["Complex sections that may benefit from explanatory comments:"]

    for section in sections:
        func_info = f" in {section.function_name}" if section.function_name else ""
        lines.append(
            f"- Lines {section.start_line}-{section.end_line}{func_info}: "
            f"{section.complexity_type} - {section.description}"
        )

    return "\n".join(lines)
