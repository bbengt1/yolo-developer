"""Dev agent module for code implementation and testing (Story 8.1, 8.3).

The Dev agent is responsible for:
- Implementing code from stories with designs
- Writing unit tests for implementations
- Writing integration tests for component interactions
- Generating documentation
- Validating code against Definition of Done

Example:
    >>> from yolo_developer.agents.dev import (
    ...     dev_node,
    ...     DevOutput,
    ...     ImplementationArtifact,
    ...     CodeFile,
    ...     TestFile,
    ...     extract_public_functions,
    ...     FunctionInfo,
    ... )
    >>>
    >>> # Create a code file
    >>> code_file = CodeFile(
    ...     file_path="src/module.py",
    ...     content="def hello(): pass",
    ...     file_type="source",
    ... )
    >>>
    >>> # Extract functions for testing (Story 8.3)
    >>> functions = extract_public_functions(code_file.content)
    >>>
    >>> # Run the dev node
    >>> result = await dev_node(state)

Architecture:
    The dev_node function is a LangGraph node that:
    - Receives YoloState TypedDict as input
    - Returns state update dict (not full state)
    - Never mutates input state
    - Uses async/await for all I/O
    - Integrates with definition_of_done gate (Story 8.1)

References:
    - ADR-001: TypedDict for internal state
    - ADR-005: LangGraph node patterns
    - ADR-006: Quality gate patterns
    - FR57-64: Dev Agent capabilities
"""

from __future__ import annotations

from yolo_developer.agents.dev.node import dev_node
from yolo_developer.agents.dev.test_utils import (
    FunctionInfo,
    QualityReport,
    calculate_coverage_estimate,
    check_coverage_threshold,
    extract_public_functions,
    generate_unit_tests_with_llm,
    identify_edge_cases,
    validate_test_quality,
)
from yolo_developer.agents.dev.types import (
    CodeFile,
    CodeFileType,
    DevOutput,
    ImplementationArtifact,
    ImplementationStatus,
    TestFile,
    TestFileType,
)

__all__ = [
    "CodeFile",
    "CodeFileType",
    "DevOutput",
    "FunctionInfo",
    "ImplementationArtifact",
    "ImplementationStatus",
    "QualityReport",
    "TestFile",
    "TestFileType",
    "calculate_coverage_estimate",
    "check_coverage_threshold",
    "dev_node",
    "extract_public_functions",
    "generate_unit_tests_with_llm",
    "identify_edge_cases",
    "validate_test_quality",
]
