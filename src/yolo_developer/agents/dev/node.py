"""Dev agent node for LangGraph orchestration (Story 8.1, 8.2, 8.3).

This module provides the dev_node function that integrates with the
LangGraph orchestration workflow. The Dev agent produces implementation
artifacts including code files and test files for stories with designs.

Story 8.2 adds LLM-powered code generation with maintainability guidelines.
Story 8.3 adds LLM-powered unit test generation with coverage validation.

Key Concepts:
- **YoloState Input**: Receives state as TypedDict, not Pydantic
- **Immutable Updates**: Returns state update dict, never mutates input
- **Async I/O**: All LLM calls use async/await
- **Structured Logging**: Uses structlog for audit trail
- **Maintainability-First**: Generated code prioritizes readability and simplicity

Example:
    >>> from yolo_developer.agents.dev import dev_node
    >>> from yolo_developer.orchestrator.state import YoloState
    >>>
    >>> state: YoloState = {
    ...     "messages": [...],
    ...     "current_agent": "dev",
    ...     "handoff_context": None,
    ...     "decisions": [],
    ... }
    >>> result = await dev_node(state)
    >>> result["messages"]  # New messages to append
    [AIMessage(...)]

Architecture Note:
    Per ADR-005, this node follows the LangGraph pattern of receiving
    full state and returning only the updates to apply.
"""

from __future__ import annotations

from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.dev.code_utils import (
    check_maintainability,
    extract_code_from_response,
    validate_python_syntax,
)
from yolo_developer.agents.dev.prompts import build_code_generation_prompt
from yolo_developer.agents.dev.test_utils import (
    calculate_coverage_estimate,
    check_coverage_threshold,
    extract_public_functions,
    generate_unit_tests_with_llm,
    validate_test_quality,
)
from yolo_developer.agents.dev.types import (
    CodeFile,
    DevOutput,
    ImplementationArtifact,
    TestFile,
)
from yolo_developer.config import load_config
from yolo_developer.config.schema import LLMConfig
from yolo_developer.gates import quality_gate
from yolo_developer.llm.router import LLMProviderError, LLMRouter
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)

# Module-level LLM router (lazy initialized)
_llm_router: LLMRouter | None = None


def _reset_llm_router() -> None:
    """Reset the global LLM router for testing.

    This function clears the cached router instance, allowing
    tests to start with fresh state.
    """
    global _llm_router
    _llm_router = None


def _get_llm_router() -> LLMRouter | None:
    """Get or create the LLM router instance.

    Returns:
        LLMRouter instance or None if configuration is missing.
    """
    global _llm_router
    if _llm_router is None:
        try:
            config = LLMConfig()
            _llm_router = LLMRouter(config)
        except Exception as e:
            logger.warning("llm_router_init_failed", error=str(e))
            return None
    return _llm_router


def _extract_stories_for_implementation(state: YoloState) -> list[dict[str, Any]]:
    """Extract stories from orchestration state for implementation.

    Stories can be present in state in two ways:
    1. Directly in architect_output key (preferred - has design decisions)
    2. In message metadata from Architect agent messages

    Args:
        state: Current orchestration state.

    Returns:
        List of story dictionaries. Empty list if no stories found.

    Example:
        >>> state = {"architect_output": {"design_decisions": [...]}}
        >>> stories = _extract_stories_for_implementation(state)
        >>> len(stories)
        2
    """
    # First, try to extract from architect_output (direct state key)
    architect_output = state.get("architect_output")
    if architect_output and isinstance(architect_output, dict):
        # Extract story IDs from design decisions
        design_decisions = architect_output.get("design_decisions", [])
        if design_decisions:
            # Build story list from design decisions
            story_ids = {d.get("story_id") for d in design_decisions if d.get("story_id")}
            arch_stories: list[dict[str, Any]] = [
                {"id": sid, "title": f"Story {sid}"} for sid in story_ids
            ]
            if arch_stories:
                logger.info(
                    "stories_extracted_from_architect_output",
                    story_count=len(arch_stories),
                    story_ids=[s.get("id") for s in arch_stories],
                )
                return arch_stories

    # Try pm_output as fallback
    pm_output = state.get("pm_output")
    if pm_output and isinstance(pm_output, dict):
        pm_stories: list[dict[str, Any]] = pm_output.get("stories", [])
        if pm_stories:
            logger.info(
                "stories_extracted_from_pm_output",
                story_count=len(pm_stories),
                story_ids=[s.get("id") for s in pm_stories],
            )
            return pm_stories

    # Fallback: Extract from message metadata (find latest Architect message)
    messages = state.get("messages", [])
    architect_messages = []

    for msg in messages:
        # Check if message has additional_kwargs with agent="architect"
        if hasattr(msg, "additional_kwargs"):
            kwargs = msg.additional_kwargs
            if kwargs.get("agent") == "architect":
                architect_messages.append(msg)

    # Get stories from the latest Architect message
    if architect_messages:
        latest_arch_msg = architect_messages[-1]
        output = latest_arch_msg.additional_kwargs.get("output", {})
        design_decisions = output.get("design_decisions", [])
        if design_decisions:
            story_ids = {d.get("story_id") for d in design_decisions if d.get("story_id")}
            msg_stories: list[dict[str, Any]] = [
                {"id": sid, "title": f"Story {sid}"} for sid in story_ids
            ]
            if msg_stories:
                logger.info(
                    "stories_extracted_from_message",
                    story_count=len(msg_stories),
                    story_ids=[s.get("id") for s in msg_stories],
                )
                return msg_stories

    logger.debug("no_stories_found_in_state")
    return []


def _extract_project_context(state: YoloState) -> dict[str, Any]:
    """Extract project context from state for code generation (Task 6).

    Extracts relevant context including:
    - Architecture constraints from architect_output
    - Project conventions from config
    - Patterns from memory (if available)

    Args:
        state: Current orchestration state.

    Returns:
        Dictionary with project context for prompts.
    """
    context: dict[str, Any] = {
        "conventions": {
            "naming": "snake_case for functions and variables, PascalCase for classes",
            "typing": "Full type annotations required",
            "async": "Use async/await for I/O operations",
            "docstrings": "Google-style docstrings required",
        },
        "patterns": [],
        "constraints": [],
    }

    # Extract architecture constraints
    architect_output = state.get("architect_output")
    if architect_output and isinstance(architect_output, dict):
        design_decisions = architect_output.get("design_decisions", [])
        for decision in design_decisions:
            if isinstance(decision, dict):
                pattern = decision.get("pattern")
                if pattern:
                    context["patterns"].append(pattern)
                constraint = decision.get("constraint")
                if constraint:
                    context["constraints"].append(constraint)

    # Extract from memory if available
    memory_context = state.get("memory_context")
    if memory_context and isinstance(memory_context, dict):
        learned_patterns = memory_context.get("patterns", [])
        context["patterns"].extend(learned_patterns)

    logger.debug(
        "project_context_extracted",
        pattern_count=len(context["patterns"]),
        constraint_count=len(context["constraints"]),
    )

    return context


async def _generate_stub_implementation(story: dict[str, Any]) -> ImplementationArtifact:
    """Generate stub implementation artifact for a story (fallback).

    Creates a stub implementation with placeholder code and test files.
    Used as fallback when LLM generation fails.

    Args:
        story: Story dictionary with id, title, and other fields.

    Returns:
        ImplementationArtifact with stub code and test files.
    """
    story_id = story.get("id", "unknown")
    story_title = story.get("title", "Untitled Story")

    # Generate stub code file
    module_name = story_id.replace("-", "_").replace(".", "_")
    code_content = f'''"""Implementation for {story_title} (Story {story_id}).

This module provides the implementation for the story requirements.
Generated by Dev agent (stub fallback).
"""

from __future__ import annotations


def implement_{module_name}() -> dict[str, str]:
    """Main implementation function for {story_title}.

    Returns:
        dict with status and message.
    """
    return {{"status": "implemented", "story_id": "{story_id}"}}
'''

    code_file = CodeFile(
        file_path=f"src/implementations/{module_name}.py",
        content=code_content,
        file_type="source",
    )

    # Generate stub test files (no router = stub fallback)
    test_files = await _generate_tests(story, [code_file])

    artifact = ImplementationArtifact(
        story_id=story_id,
        code_files=(code_file,),
        test_files=tuple(test_files),
        implementation_status="completed",
        notes=f"Stub implementation for {story_title} (LLM fallback).",
    )

    logger.debug(
        "stub_implementation_generated",
        story_id=story_id,
    )

    return artifact


async def _generate_code_with_llm(
    story: dict[str, Any],
    context: dict[str, Any],
    router: LLMRouter,
) -> tuple[str, bool]:
    """Generate code using LLM with maintainability guidelines (AC5, AC6).

    Args:
        story: Story dictionary with id, title, requirements, etc.
        context: Project context from _extract_project_context.
        router: LLMRouter instance for making calls.

    Returns:
        Tuple of (code, is_valid). code is generated code string,
        is_valid indicates if syntax validation passed.
    """
    story_id = story.get("id", "unknown")
    story_title = story.get("title", "Untitled Story")
    requirements = story.get("requirements", story.get("description", ""))
    acceptance_criteria = story.get("acceptance_criteria", [])
    design_decisions = story.get("design_decisions", {})

    # Build prompt with maintainability guidelines
    prompt = build_code_generation_prompt(
        story_title=story_title,
        requirements=requirements or f"Implement functionality for {story_title}",
        acceptance_criteria=acceptance_criteria if acceptance_criteria else None,
        design_decisions=design_decisions if design_decisions else None,
        additional_context=f"Story ID: {story_id}\n"
        f"Patterns to follow: {', '.join(context.get('patterns', []))}\n"
        f"Constraints: {', '.join(context.get('constraints', []))}",
        include_maintainability=True,
        include_conventions=True,
    )

    logger.info(
        "llm_code_generation_start",
        story_id=story_id,
        prompt_length=len(prompt),
    )

    try:
        # Use "complex" tier per ADR-003 for code generation
        response = await router.call(
            messages=[{"role": "user", "content": prompt}],
            tier="complex",
            temperature=0.7,
            max_tokens=4096,
        )

        # Extract code from response
        code = extract_code_from_response(response)

        # Validate syntax
        is_valid, error = validate_python_syntax(code)

        if is_valid:
            # Check maintainability (advisory only)
            report = check_maintainability(code)
            if report.has_warnings():
                logger.info(
                    "maintainability_warnings",
                    story_id=story_id,
                    warning_count=len(report.warnings),
                    max_function_length=report.max_function_length,
                    max_nesting_depth=report.max_nesting_depth,
                )

            logger.info(
                "llm_code_generation_success",
                story_id=story_id,
                code_length=len(code),
            )
            return code, True
        else:
            logger.warning(
                "llm_code_generation_syntax_error",
                story_id=story_id,
                error=error,
            )
            # Retry with error context
            from yolo_developer.agents.dev.prompts.code_generation import (
                build_retry_prompt,
            )

            retry_prompt = build_retry_prompt(prompt, error or "Unknown error", code)
            retry_response = await router.call(
                messages=[{"role": "user", "content": retry_prompt}],
                tier="complex",
                temperature=0.5,  # Lower temp for fixing
            )

            retry_code = extract_code_from_response(retry_response)
            retry_valid, retry_error = validate_python_syntax(retry_code)

            if retry_valid:
                logger.info(
                    "llm_code_generation_retry_success",
                    story_id=story_id,
                )
                return retry_code, True
            else:
                logger.warning(
                    "llm_code_generation_retry_failed",
                    story_id=story_id,
                    error=retry_error,
                )
                return code, False

    except LLMProviderError as e:
        logger.error(
            "llm_code_generation_provider_error",
            story_id=story_id,
            error=str(e),
        )
        return "", False

    except Exception as e:
        logger.error(
            "llm_code_generation_error",
            story_id=story_id,
            error=str(e),
        )
        return "", False


async def _generate_implementation(
    story: dict[str, Any],
    context: dict[str, Any],
    router: LLMRouter | None = None,
) -> ImplementationArtifact:
    """Generate implementation artifact for a story (Task 4).

    Uses LLM-powered code generation when available, with fallback
    to stub implementation on failure.

    Args:
        story: Story dictionary with id, title, and other fields.
        context: Project context for code generation.
        router: Optional LLMRouter. If None, uses stub implementation.

    Returns:
        ImplementationArtifact with code and test files.

    Example:
        >>> story = {"id": "story-001", "title": "User Authentication"}
        >>> artifact = await _generate_implementation(story, {})
        >>> artifact.implementation_status
        'completed'
    """
    story_id = story.get("id", "unknown")
    story_title = story.get("title", "Untitled Story")

    # Try LLM generation if router available
    if router is not None:
        code, is_valid = await _generate_code_with_llm(story, context, router)

        if is_valid and code:
            # LLM generation succeeded
            module_name = story_id.replace("-", "_").replace(".", "_")

            code_file = CodeFile(
                file_path=f"src/implementations/{module_name}.py",
                content=code,
                file_type="source",
            )

            # Generate test files for the code (LLM-powered, Story 8.3)
            test_files = await _generate_tests(story, [code_file], router)

            artifact = ImplementationArtifact(
                story_id=story_id,
                code_files=(code_file,),
                test_files=tuple(test_files),
                implementation_status="completed",
                notes=f"LLM-generated implementation for {story_title}.",
            )

            logger.info(
                "llm_implementation_generated",
                story_id=story_id,
                code_file_count=len(artifact.code_files),
                test_file_count=len(artifact.test_files),
            )

            return artifact

    # Fallback to stub implementation
    logger.info(
        "falling_back_to_stub_implementation",
        story_id=story_id,
        reason="LLM unavailable or generation failed",
    )
    return await _generate_stub_implementation(story)


async def _generate_tests(
    story: dict[str, Any],
    code_files: list[CodeFile],
    router: LLMRouter | None = None,
    coverage_threshold: float | None = None,
) -> list[TestFile]:
    """Generate test files for a story's code files (Story 8.3).

    Uses LLM-powered test generation with coverage validation and
    quality checks. Falls back to stub tests if LLM unavailable.

    Args:
        story: Story dictionary with id, title, and other fields.
        code_files: List of code files to generate tests for.
        router: Optional LLMRouter for LLM-powered generation.
        coverage_threshold: Minimum coverage threshold (0.0-1.0).
            If None, loads from config.quality.test_coverage_threshold (AC4).

    Returns:
        List of TestFile objects. Empty if no source files.

    Example:
        >>> story = {"id": "story-001", "title": "User Auth"}
        >>> code_files = [CodeFile(...)]
        >>> tests = await _generate_tests(story, code_files, router)
        >>> len(tests) >= 1
        True
    """
    story_id = story.get("id", "unknown")
    story_title = story.get("title", "Untitled Story")

    # Load coverage threshold from config if not provided (AC4)
    if coverage_threshold is None:
        try:
            config = load_config()
            coverage_threshold = config.quality.test_coverage_threshold
        except Exception:
            coverage_threshold = 0.8  # Fallback default

    # Check if there are any source files to test
    source_files = [cf for cf in code_files if cf.file_type == "source"]
    if not source_files:
        logger.debug(
            "tests_generated",
            story_id=story_id,
            test_count=0,
            reason="no_source_files",
        )
        return []

    test_files: list[TestFile] = []

    for code_file in source_files:
        # Extract module name from file path
        file_name = code_file.file_path.split("/")[-1].replace(".py", "")
        module_name = file_name.replace("-", "_")

        # Try LLM-powered test generation
        if router is not None:
            try:
                # Extract public functions from code (AC1)
                functions = extract_public_functions(code_file.content)

                if functions:
                    logger.info(
                        "llm_test_generation_start",
                        story_id=story_id,
                        module_name=module_name,
                        function_count=len(functions),
                    )

                    # Generate tests with LLM (AC5, AC6)
                    test_code, is_valid = await generate_unit_tests_with_llm(
                        implementation_code=code_file.content,
                        functions=functions,
                        module_name=module_name,
                        router=router,
                        max_retries=2,
                        additional_context=f"Story: {story_title} (ID: {story_id})",
                    )

                    if is_valid and test_code:
                        # Validate test quality (AC3)
                        quality_report = validate_test_quality(test_code)

                        if not quality_report.is_acceptable():
                            logger.warning(
                                "test_quality_warning",
                                story_id=story_id,
                                module_name=module_name,
                                warnings=quality_report.warnings,
                                has_assertions=quality_report.has_assertions,
                                is_deterministic=quality_report.is_deterministic,
                            )

                        # Check coverage estimate (AC4)
                        coverage = calculate_coverage_estimate(
                            code_file.content, test_code
                        )
                        meets_threshold, coverage_msg = check_coverage_threshold(
                            coverage, coverage_threshold
                        )

                        if not meets_threshold:
                            logger.warning(
                                "test_coverage_below_threshold",
                                story_id=story_id,
                                module_name=module_name,
                                coverage=coverage,
                                threshold=coverage_threshold,
                                message=coverage_msg,
                            )

                        # Create test file
                        test_path = f"tests/unit/implementations/test_{module_name}.py"
                        test_file = TestFile(
                            file_path=test_path,
                            content=test_code,
                            test_type="unit",
                        )
                        test_files.append(test_file)

                        logger.info(
                            "llm_test_generation_success",
                            story_id=story_id,
                            module_name=module_name,
                            coverage=coverage,
                            quality_acceptable=quality_report.is_acceptable(),
                        )
                        continue  # Move to next code file

            except Exception as e:
                logger.warning(
                    "llm_test_generation_error",
                    story_id=story_id,
                    module_name=module_name,
                    error=str(e),
                )
                # Fall through to stub generation

        # Fallback: Generate stub test file
        test_file = _generate_stub_test(story_id, story_title, module_name, code_file)
        test_files.append(test_file)

    logger.debug(
        "tests_generated",
        story_id=story_id,
        test_count=len(test_files),
    )

    return test_files


def _generate_stub_test(
    story_id: str,
    story_title: str,
    module_name: str,
    code_file: CodeFile,
) -> TestFile:
    """Generate a stub test file when LLM is unavailable.

    Args:
        story_id: Story identifier.
        story_title: Story title for documentation.
        module_name: Module name for import and class naming.
        code_file: The code file being tested.

    Returns:
        TestFile with stub test content.
    """
    test_path = f"tests/unit/implementations/test_{module_name}.py"

    # Extract function names for stub tests
    functions = extract_public_functions(code_file.content)
    func_names = [f.name for f in functions] if functions else ["main"]

    # Generate test methods for each function
    test_methods = []
    for func_name in func_names[:5]:  # Limit to 5 functions
        test_methods.append(f'''
    def test_{func_name}_exists(self) -> None:
        """Test that {func_name} function exists and is callable."""
        # Stub test - full LLM generation in Story 8.3+
        from yolo_developer.implementations.{module_name} import {func_name}

        assert callable({func_name})''')

    test_methods_str = "\n".join(test_methods) if test_methods else '''
    def test_implementation_exists(self) -> None:
        """Test that implementation module exists."""
        # Stub test - full LLM generation in Story 8.3+
        pass'''

    class_name = module_name.title().replace("_", "")
    test_content = f'''"""Unit tests for {story_title} (Story {story_id}).

Generated by Dev agent (stub fallback - LLM unavailable).
"""

from __future__ import annotations

import pytest


class Test{class_name}:
    """Test suite for {story_title}."""
{test_methods_str}
'''

    return TestFile(
        file_path=test_path,
        content=test_content,
        test_type="unit",
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
@quality_gate("definition_of_done", blocking=False)
async def dev_node(state: YoloState) -> dict[str, Any]:
    """Dev agent node for code implementation and testing.

    Receives stories with designs from state and produces implementation
    artifacts including code files and test files.

    Story 8.2 adds LLM-powered code generation with maintainability
    guidelines. Falls back to stub generation if LLM unavailable.

    This function follows the LangGraph node pattern:
    - Receives full state as YoloState TypedDict
    - Returns only the state updates (not full state)
    - Never mutates the input state
    - Uses tenacity for retry with exponential backoff (AC4)

    Args:
        state: Current orchestration state with stories from Architect.

    Returns:
        State update dict with:
        - messages: List of new messages to append
        - decisions: List of new decisions to append
        - dev_output: Serialized DevOutput
        Never includes current_agent (handoff manages that).

    Example:
        >>> state: YoloState = {
        ...     "messages": [...],
        ...     "current_agent": "dev",
        ...     "handoff_context": None,
        ...     "decisions": [],
        ... }
        >>> result = await dev_node(state)
        >>> "messages" in result
        True
    """
    logger.info(
        "dev_node_start",
        current_agent=state.get("current_agent"),
        message_count=len(state.get("messages", [])),
    )

    # Get LLM router (may be None if not configured)
    router = _get_llm_router()

    # Extract project context for code generation (Task 6)
    context = _extract_project_context(state)

    # Extract stories from state (AC2)
    stories = _extract_stories_for_implementation(state)

    # Generate implementations for each story (AC3)
    implementations: list[ImplementationArtifact] = []
    for story in stories:
        artifact = await _generate_implementation(story, context, router)
        implementations.append(artifact)

    # Build output with actual results
    total_code_files = sum(len(impl.code_files) for impl in implementations)
    total_test_files = sum(len(impl.test_files) for impl in implementations)

    # Determine generation method
    llm_used = router is not None
    generation_method = "LLM-powered" if llm_used else "stub"

    output = DevOutput(
        implementations=tuple(implementations),
        processing_notes=f"Processed {len(stories)} stories, "
        f"generated {total_code_files} code files, "
        f"{total_test_files} test files. "
        f"Generation method: {generation_method}.",
    )

    # Create decision record with dev attribution (AC6)
    decision = Decision(
        agent="dev",
        summary=f"Generated {total_code_files} code files, {total_test_files} test files "
        f"for {len(stories)} stories ({generation_method})",
        rationale=f"Processed {len(stories)} stories from Architect. "
        f"Used {generation_method} code generation with maintainability guidelines.",
        related_artifacts=tuple(impl.story_id for impl in implementations),
    )

    # Create output message with dev attribution (AC6)
    message = create_agent_message(
        content=f"Dev processing complete: {total_code_files} code files, "
        f"{total_test_files} test files generated for {len(stories)} stories "
        f"({generation_method}).",
        agent="dev",
        metadata={"output": output.to_dict()},
    )

    logger.info(
        "dev_node_complete",
        story_count=len(stories),
        implementation_count=len(implementations),
        code_file_count=total_code_files,
        test_file_count=total_test_files,
        generation_method=generation_method,
    )

    # Return ONLY the updates, not full state (AC6)
    return {
        "messages": [message],
        "decisions": [decision],
        "dev_output": output.to_dict(),
    }
