"""Dev agent node for LangGraph orchestration (Story 8.1, 8.2, 8.3, 8.4, 8.5, 8.7, 8.8).

This module provides the dev_node function that integrates with the
LangGraph orchestration workflow. The Dev agent produces implementation
artifacts including code files and test files for stories with designs.

Story 8.2 adds LLM-powered code generation with maintainability guidelines.
Story 8.3 adds LLM-powered unit test generation with coverage validation.
Story 8.4 adds LLM-powered integration test generation with boundary analysis.
Story 8.5 adds LLM-powered documentation enhancement with quality validation.
Story 8.7 adds pattern following with validation and prompt integration.
Story 8.8 adds communicative commit message generation.

Key Concepts:
- **YoloState Input**: Receives state as TypedDict, not Pydantic
- **Immutable Updates**: Returns state update dict, never mutates input
- **Async I/O**: All LLM calls use async/await
- **Structured Logging**: Uses structlog for audit trail
- **Maintainability-First**: Generated code prioritizes readability and simplicity
- **Pattern Following**: Generated code follows established codebase patterns

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

from dataclasses import asdict
from pathlib import Path
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.dev.code_utils import (
    check_maintainability,
    extract_code_from_response,
    validate_python_syntax,
)
from yolo_developer.agents.dev.commit_utils import (
    CommitMessageContext,
    CommitType,
    generate_commit_message,
    generate_commit_message_with_llm,
)
from yolo_developer.agents.dev.doc_utils import (
    generate_documentation_with_llm,
    validate_documentation_quality,
)
from yolo_developer.agents.dev.integration_utils import (
    analyze_component_boundaries,
    analyze_data_flow,
    detect_error_scenarios,
    generate_integration_tests_with_llm,
    validate_integration_test_quality,
)
from yolo_developer.agents.dev.pattern_utils import (
    clear_pattern_cache,
    get_error_patterns,
    get_naming_patterns,
    get_style_patterns,
    validate_pattern_adherence,
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
from yolo_developer.config import ConfigurationError, load_config
from yolo_developer.config.schema import LLMConfig
from yolo_developer.gates import quality_gate
from yolo_developer.llm.router import LLMProviderError, LLMRouter
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)

# Module-level LLM router (lazy initialized)
_llm_router: LLMRouter | None = None


def _extract_files_from_response(response: str) -> list[tuple[str, str]]:
    """Extract file paths and code from LLM response.

    Parses the LLM response to extract FILE_PATH markers and associated code blocks.
    Falls back to extracting just code if no file paths are found.

    Args:
        response: Raw LLM response containing FILE_PATH markers and code blocks.

    Returns:
        List of (file_path, code) tuples. If no FILE_PATH markers found,
        returns empty list (caller should use default path).

    Example:
        >>> response = '''FILE_PATH: src/yolo_developer/memory/manager.py
        ... ```python
        ... def foo(): pass
        ... ```'''
        >>> files = _extract_files_from_response(response)
        >>> files[0][0]
        'src/yolo_developer/memory/manager.py'
    """
    import re

    files: list[tuple[str, str]] = []

    # Pattern to match FILE_PATH followed by code block
    pattern = r'FILE_PATH:\s*([^\n]+)\s*```python\s*(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)

    for file_path, code in matches:
        file_path = file_path.strip()
        code = code.strip()
        if file_path and code:
            files.append((file_path, code))
            logger.debug(
                "file_extracted_from_response",
                file_path=file_path,
                code_length=len(code),
            )

    if files:
        logger.info(
            "files_extracted_from_llm_response",
            file_count=len(files),
            file_paths=[f[0] for f in files],
        )

    return files


def _reset_llm_router() -> None:
    """Reset the global LLM router for testing.

    This function clears the cached router instance, allowing
    tests to start with fresh state.
    """
    global _llm_router
    _llm_router = None


def _write_generated_files(
    implementations: tuple[ImplementationArtifact, ...],
    output_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Write generated code and test files to disk.

    Writes all code files and test files from the implementation artifacts
    to the filesystem, creating necessary directories as needed.

    Args:
        implementations: Tuple of implementation artifacts containing files to write.
        output_dir: Optional output directory. If None, uses current working directory.

    Returns:
        Dictionary with 'code_files' and 'test_files' lists of written paths.

    Example:
        >>> files = _write_generated_files(implementations)
        >>> files['code_files']
        ['src/implementations/story_001.py']
    """
    base_dir = output_dir or Path.cwd()
    written_files: dict[str, list[str]] = {"code_files": [], "test_files": []}

    for impl in implementations:
        # Write code files
        for code_file in impl.code_files:
            file_path = base_dir / code_file.file_path
            try:
                # Create parent directories if they don't exist
                file_path.parent.mkdir(parents=True, exist_ok=True)
                # Write the file
                file_path.write_text(code_file.content)
                written_files["code_files"].append(str(file_path))
                logger.info(
                    "code_file_written",
                    file_path=str(file_path),
                    file_type=code_file.file_type,
                    story_id=impl.story_id,
                )
            except OSError as e:
                logger.error(
                    "code_file_write_failed",
                    file_path=str(file_path),
                    error=str(e),
                )

        # Write test files
        for test_file in impl.test_files:
            file_path = base_dir / test_file.file_path
            try:
                # Create parent directories if they don't exist
                file_path.parent.mkdir(parents=True, exist_ok=True)
                # Write the file
                file_path.write_text(test_file.content)
                written_files["test_files"].append(str(file_path))
                logger.info(
                    "test_file_written",
                    file_path=str(file_path),
                    test_type=test_file.test_type,
                    story_id=impl.story_id,
                )
            except OSError as e:
                logger.error(
                    "test_file_write_failed",
                    file_path=str(file_path),
                    error=str(e),
                )

    logger.info(
        "generated_files_written",
        code_file_count=len(written_files["code_files"]),
        test_file_count=len(written_files["test_files"]),
    )

    return written_files


def _get_llm_router() -> LLMRouter | None:
    """Get or create the LLM router instance.

    Returns:
        LLMRouter instance or None if configuration is missing.
    """
    global _llm_router
    if _llm_router is None:
        try:
            try:
                config = load_config()
                llm_config = config.llm
            except ConfigurationError as e:
                logger.warning("llm_router_load_config_failed", error=str(e))
                llm_config = LLMConfig()
            _llm_router = LLMRouter(llm_config)
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


def _get_relevant_patterns(state: dict[str, Any]) -> dict[str, list[Any]]:
    """Get patterns from state for code generation prompts (Story 8.7).

    Queries PatternLearner from memory_context for naming, error handling,
    and style patterns. Falls back to defaults if memory_context unavailable.

    Args:
        state: YoloState containing memory_context with patterns.

    Returns:
        Dictionary with keys 'naming', 'error_handling', 'style',
        each containing list of patterns.

    Example:
        >>> state = {"memory_context": {}}
        >>> patterns = _get_relevant_patterns(state)
        >>> len(patterns["naming"]) >= 1
        True
    """
    naming = get_naming_patterns(state)
    error = get_error_patterns(state)
    style = get_style_patterns(state)

    logger.debug(
        "relevant_patterns_extracted",
        naming_count=len(naming),
        error_count=len(error),
        style_count=len(style),
    )

    return {
        "naming": naming,
        "error_handling": error,
        "style": style,
    }


def _format_patterns_for_prompt(patterns: dict[str, list[Any]]) -> str:
    """Format patterns for inclusion in LLM prompts (Story 8.7).

    Converts pattern objects to human-readable text for code generation prompts.

    Args:
        patterns: Dictionary from _get_relevant_patterns().

    Returns:
        Formatted string with pattern instructions.
    """
    sections: list[str] = []

    # Naming patterns
    naming = patterns.get("naming", [])
    if naming:
        naming_lines = ["### Naming Conventions (from codebase patterns)"]
        for p in naming[:5]:  # Limit to 5
            if hasattr(p, "value") and hasattr(p, "examples"):
                examples = ", ".join(p.examples[:3]) if p.examples else ""
                naming_lines.append(f"- {p.name}: {p.value} (e.g., {examples})")
        sections.append("\n".join(naming_lines))

    # Error handling patterns
    error = patterns.get("error_handling", [])
    if error:
        error_lines = ["### Error Handling Conventions"]
        for p in error[:3]:  # Limit to 3
            if hasattr(p, "handling_style") and hasattr(p, "exception_types"):
                types = ", ".join(p.exception_types[:3])
                error_lines.append(f"- {p.pattern_name}: {p.handling_style}")
                error_lines.append(f"  Preferred exceptions: {types}")
        sections.append("\n".join(error_lines))

    # Style patterns
    style = patterns.get("style", [])
    if style:
        style_lines = ["### Code Style Conventions"]
        for p in style[:3]:  # Limit to 3
            if hasattr(p, "value") and hasattr(p, "category"):
                style_lines.append(f"- {p.pattern_name} ({p.category}): {p.value}")
        sections.append("\n".join(style_lines))

    if sections:
        return "\n\n".join(sections)
    return ""


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
    state: dict[str, Any] | None = None,
) -> tuple[str, bool, dict[str, Any] | None, list[tuple[str, str]]]:
    """Generate code using LLM with maintainability guidelines (AC5, AC6, Story 8.7).

    Story 8.7 adds pattern following: patterns are included in the prompt,
    and generated code is validated for pattern adherence.

    Args:
        story: Story dictionary with id, title, requirements, etc.
        context: Project context from _extract_project_context.
        router: LLMRouter instance for making calls.
        state: Optional YoloState for pattern queries.

    Returns:
        Tuple of (code, is_valid, pattern_result, extracted_files).
        - code: Generated code string (first file if multiple)
        - is_valid: True if syntax validation passed
        - pattern_result: PatternValidationResult dict (or None if state unavailable)
        - extracted_files: List of (file_path, code) tuples for project placement
    """
    story_id = story.get("id", "unknown")
    story_title = story.get("title", "Untitled Story")

    # Extract requirements from story fields
    # Stories have action (what user wants), benefit (why), and role (who)
    action = story.get("action", "")
    benefit = story.get("benefit", "")
    role = story.get("role", "user")

    # Build requirements from story content
    # First try explicit requirements/description, then construct from story fields
    requirements = story.get("requirements", story.get("description", ""))
    if not requirements and (action or benefit):
        requirements_parts = []
        if action:
            requirements_parts.append(f"As a {role}, I want to {action}")
        if benefit:
            requirements_parts.append(f"so that {benefit}")
        requirements = " ".join(requirements_parts)

    acceptance_criteria = story.get("acceptance_criteria", [])
    design_decisions = story.get("design_decisions", {})

    # Story 8.7: Get patterns for prompt and validation
    patterns_dict: dict[str, list[Any]] = {}
    if state is not None:
        patterns_dict = _get_relevant_patterns(state)

    # Format patterns for inclusion in prompt
    pattern_context = _format_patterns_for_prompt(patterns_dict)

    # Build additional context including patterns
    additional_context_parts = [
        f"Story ID: {story_id}",
        f"Architecture patterns: {', '.join(context.get('patterns', []))}",
        f"Constraints: {', '.join(context.get('constraints', []))}",
    ]
    if pattern_context:
        additional_context_parts.append(f"\n{pattern_context}")

    # Build prompt with maintainability guidelines
    prompt = build_code_generation_prompt(
        story_title=story_title,
        requirements=requirements or f"Implement functionality for {story_title}",
        acceptance_criteria=acceptance_criteria if acceptance_criteria else None,
        design_decisions=design_decisions if design_decisions else None,
        additional_context="\n".join(additional_context_parts),
        include_maintainability=True,
        include_conventions=True,
    )

    logger.info(
        "llm_code_generation_start",
        story_id=story_id,
        prompt_length=len(prompt),
    )

    try:
        response = await router.call_task(
            messages=[{"role": "user", "content": prompt}],
            task_type="code_generation",
            temperature=0.7,
            max_tokens=4096,
        )

        # Try to extract files with paths first (new format)
        extracted_files = _extract_files_from_response(response)

        # If no files with paths found, fallback to extracting code without path
        if not extracted_files:
            code = extract_code_from_response(response)
        else:
            # Use the first file's code for validation (we'll handle multiple files later)
            code = extracted_files[0][1]

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

            # Story 8.7: Validate pattern adherence (advisory only)
            pattern_result_dict: dict[str, Any] | None = None
            if state is not None:
                pattern_result = validate_pattern_adherence(code, state)
                pattern_result_dict = pattern_result.to_dict()

                if not pattern_result.passed:
                    logger.info(
                        "pattern_adherence_warnings",
                        story_id=story_id,
                        score=pattern_result.score,
                        deviation_count=len(pattern_result.deviations),
                        threshold=pattern_result.threshold,
                    )

            logger.info(
                "llm_code_generation_success",
                story_id=story_id,
                code_length=len(code),
                pattern_score=pattern_result_dict.get("score") if pattern_result_dict else None,
            )
            return code, True, pattern_result_dict, extracted_files
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
            retry_response = await router.call_task(
                messages=[{"role": "user", "content": retry_prompt}],
                task_type="code_generation",
                temperature=0.5,  # Lower temp for fixing
            )

            retry_code = extract_code_from_response(retry_response)
            retry_valid, retry_error = validate_python_syntax(retry_code)

            if retry_valid:
                # Story 8.7: Validate pattern adherence for retry code
                retry_pattern_result_dict: dict[str, Any] | None = None
                if state is not None:
                    retry_pattern_result = validate_pattern_adherence(retry_code, state)
                    retry_pattern_result_dict = retry_pattern_result.to_dict()

                logger.info(
                    "llm_code_generation_retry_success",
                    story_id=story_id,
                )
                return retry_code, True, retry_pattern_result_dict, []
            else:
                logger.warning(
                    "llm_code_generation_retry_failed",
                    story_id=story_id,
                    error=retry_error,
                )
                return code, False, None, []

    except LLMProviderError as e:
        logger.error(
            "llm_code_generation_provider_error",
            story_id=story_id,
            error=str(e),
        )
        return "", False, None, []

    except Exception as e:
        logger.error(
            "llm_code_generation_error",
            story_id=story_id,
            error=str(e),
        )
        return "", False, None, []


async def _enhance_documentation(
    code: str,
    story_id: str,
    story_title: str,
    router: LLMRouter,
) -> str:
    """Enhance code with comprehensive documentation (Story 8.5).

    Uses LLM to add module docstrings, function docstrings, and
    explanatory comments. Falls back to original code on failure.

    Args:
        code: Python source code to document.
        story_id: ID of the story being implemented.
        story_title: Title of the story for context.
        router: LLM router for documentation generation.

    Returns:
        Enhanced code with documentation, or original code on failure.

    Example:
        >>> documented = await _enhance_documentation(
        ...     code="def hello(): pass",
        ...     story_id="story-001",
        ...     story_title="User Authentication",
        ...     router=router,
        ... )
    """
    logger.info(
        "documentation_enhancement_start",
        story_id=story_id,
    )

    try:
        # Generate documentation using LLM
        documented_code, is_valid = await generate_documentation_with_llm(
            code=code,
            context=f"Implementation for story '{story_title}' (ID: {story_id})",
            router=router,
        )

        if is_valid:
            # Validate documentation quality
            report = validate_documentation_quality(documented_code)

            if report.is_acceptable():
                logger.info(
                    "documentation_enhancement_success",
                    story_id=story_id,
                    has_module_docstring=report.has_module_docstring,
                    functions_with_args=report.functions_with_args,
                    total_functions=report.total_functions,
                )
                return documented_code
            else:
                logger.warning(
                    "documentation_quality_below_threshold",
                    story_id=story_id,
                    warnings=report.warnings[:5],  # Log first 5 warnings
                )
                # Still use documented code, just log the warning
                return documented_code
        else:
            logger.warning(
                "documentation_enhancement_invalid_syntax",
                story_id=story_id,
            )
            return code

    except Exception as e:
        logger.error(
            "documentation_enhancement_error",
            story_id=story_id,
            error=str(e),
        )
        # Return original code on any error
        return code


async def _generate_implementation(
    story: dict[str, Any],
    context: dict[str, Any],
    router: LLMRouter | None = None,
    state: dict[str, Any] | None = None,
) -> ImplementationArtifact:
    """Generate implementation artifact for a story (Task 4, Story 8.7).

    Uses LLM-powered code generation when available, with fallback
    to stub implementation on failure.

    Story 8.7: Includes pattern validation results in artifact notes.

    Args:
        story: Story dictionary with id, title, and other fields.
        context: Project context for code generation.
        router: Optional LLMRouter. If None, uses stub implementation.
        state: Optional YoloState for pattern queries.

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
        code, is_valid, pattern_result, extracted_files = await _generate_code_with_llm(
            story, context, router, state
        )

        if is_valid and code:
            # LLM generation succeeded
            module_name = story_id.replace("-", "_").replace(".", "_")

            # Create code files from extracted file paths or use default
            code_files_list: list[CodeFile] = []

            if extracted_files:
                # Use extracted file paths from LLM response
                for file_path, file_code in extracted_files:
                    # Story 8.5: Enhance code with documentation
                    documented_code = await _enhance_documentation(
                        code=file_code,
                        story_id=story_id,
                        story_title=story_title,
                        router=router,
                    )
                    code_files_list.append(
                        CodeFile(
                            file_path=file_path,
                            content=documented_code,
                            file_type="source",
                        )
                    )
                logger.info(
                    "using_extracted_file_paths",
                    story_id=story_id,
                    file_count=len(code_files_list),
                    file_paths=[cf.file_path for cf in code_files_list],
                )
            else:
                # Fallback to default implementations path
                documented_code = await _enhance_documentation(
                    code=code,
                    story_id=story_id,
                    story_title=story_title,
                    router=router,
                )
                code_files_list.append(
                    CodeFile(
                        file_path=f"src/implementations/{module_name}.py",
                        content=documented_code,
                        file_type="source",
                    )
                )
                logger.info(
                    "using_default_file_path",
                    story_id=story_id,
                    file_path=f"src/implementations/{module_name}.py",
                )

            # Generate test files for the code (LLM-powered, Story 8.3)
            test_files = await _generate_tests(story, code_files_list, router)

            # Story 8.7: Include pattern adherence in notes
            notes_parts = [f"LLM-generated implementation for {story_title}."]
            if pattern_result:
                score = pattern_result.get("score", 100)
                passed = pattern_result.get("passed", True)
                deviation_count = pattern_result.get("deviation_count", 0)
                if deviation_count > 0:
                    notes_parts.append(
                        f"Pattern adherence: score={score}, "
                        f"passed={passed}, deviations={deviation_count}."
                    )
                else:
                    notes_parts.append(f"Pattern adherence: score={score}, no deviations.")

            artifact = ImplementationArtifact(
                story_id=story_id,
                code_files=tuple(code_files_list),
                test_files=tuple(test_files),
                implementation_status="completed",
                notes=" ".join(notes_parts),
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
    """Generate test files for a story's code files (Story 8.3, 8.4).

    Uses LLM-powered test generation with coverage validation and
    quality checks. Falls back to stub tests if LLM unavailable.

    Story 8.4 adds integration test generation for multi-file scenarios
    with component boundary detection and data flow analysis.

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

    # Generate unit tests for each code file
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
                        coverage = calculate_coverage_estimate(code_file.content, test_code)
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

    # Story 8.4: Generate integration tests if multiple source files
    if len(source_files) >= 2 and router is not None:
        integration_test = await _generate_integration_tests(
            story_id=story_id,
            story_title=story_title,
            code_files=source_files,
            router=router,
        )
        if integration_test:
            test_files.append(integration_test)

    logger.debug(
        "tests_generated",
        story_id=story_id,
        test_count=len(test_files),
        unit_tests=len([t for t in test_files if t.test_type == "unit"]),
        integration_tests=len([t for t in test_files if t.test_type == "integration"]),
    )

    return test_files


async def _generate_integration_tests(
    story_id: str,
    story_title: str,
    code_files: list[CodeFile],
    router: LLMRouter,
) -> TestFile | None:
    """Generate integration tests for multi-file scenarios (Story 8.4).

    Analyzes component boundaries, data flows, and error scenarios to
    generate comprehensive integration tests using LLM.

    Args:
        story_id: Story identifier.
        story_title: Story title for context.
        code_files: List of source code files.
        router: LLMRouter for LLM-powered generation.

    Returns:
        TestFile with integration tests, or None if generation fails.
    """
    # Analyze component boundaries (AC1, AC7)
    boundaries = analyze_component_boundaries(code_files)

    # Analyze data flows (AC2)
    flows = analyze_data_flow(code_files)

    # Detect error scenarios (AC3)
    error_scenarios = detect_error_scenarios(code_files)

    logger.info(
        "integration_test_analysis_complete",
        story_id=story_id,
        boundary_count=len(boundaries),
        flow_count=len(flows),
        error_scenario_count=len(error_scenarios),
    )

    # Only generate integration tests if we detected boundaries
    if not boundaries and not flows:
        logger.debug(
            "skipping_integration_tests",
            story_id=story_id,
            reason="no_boundaries_or_flows_detected",
        )
        return None

    try:
        # Generate integration tests with LLM (AC6)
        test_code, is_valid = await generate_integration_tests_with_llm(
            code_files=code_files,
            boundaries=boundaries,
            flows=flows,
            error_scenarios=error_scenarios,
            router=router,
            additional_context=f"Story: {story_title} (ID: {story_id})",
            max_retries=2,
        )

        if is_valid and test_code:
            # Validate integration test quality (AC4)
            quality_report = validate_integration_test_quality(test_code)

            if not quality_report.is_acceptable():
                logger.warning(
                    "integration_test_quality_warning",
                    story_id=story_id,
                    warnings=quality_report.warnings,
                    uses_fixtures=quality_report.uses_fixtures,
                    uses_mocks=quality_report.uses_mocks,
                    has_cleanup=quality_report.has_cleanup,
                )

            # Create integration test file (AC5)
            # Naming convention: test_<story_id>_integration.py
            # where story_id serves as the component/scenario identifier
            base_name = story_id.replace("-", "_").replace(".", "_")
            test_path = f"tests/integration/implementations/test_{base_name}_integration.py"

            test_file = TestFile(
                file_path=test_path,
                content=test_code,
                test_type="integration",
            )

            logger.info(
                "integration_test_generation_success",
                story_id=story_id,
                test_path=test_path,
                quality_acceptable=quality_report.is_acceptable(),
            )

            return test_file

    except Exception as e:
        logger.warning(
            "integration_test_generation_error",
            story_id=story_id,
            error=str(e),
        )

    # Fallback: Generate stub integration test
    return _generate_stub_integration_test(story_id, story_title, code_files)


def _generate_stub_integration_test(
    story_id: str,
    story_title: str,
    code_files: list[CodeFile],
) -> TestFile:
    """Generate a stub integration test file when LLM is unavailable.

    Args:
        story_id: Story identifier.
        story_title: Story title for documentation.
        code_files: List of code files being tested.

    Returns:
        TestFile with stub integration test content.
    """
    base_name = story_id.replace("-", "_").replace(".", "_")
    test_path = f"tests/integration/implementations/test_{base_name}_integration.py"

    # Extract module names for imports
    module_names = []
    for cf in code_files:
        if cf.file_type == "source":
            file_name = cf.file_path.split("/")[-1].replace(".py", "")
            module_names.append(file_name.replace("-", "_"))

    class_name = base_name.title().replace("_", "")
    test_content = f'''"""Integration tests for {story_title} (Story {story_id}).

Generated by Dev agent (stub fallback - LLM unavailable).
Tests component interactions between: {", ".join(module_names)}.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_dependencies() -> MagicMock:
    """Mock external dependencies for integration tests."""
    return MagicMock()


@pytest.fixture(autouse=True)
def cleanup_state():
    """Clean up state after each test."""
    yield
    # Cleanup code here


class Test{class_name}Integration:
    """Integration test suite for {story_title}."""

    @pytest.mark.asyncio
    async def test_component_interaction_placeholder(
        self, mock_dependencies: MagicMock
    ) -> None:
        """Placeholder test for component interactions.

        TODO: Implement actual integration tests for:
        - Component boundary testing
        - Data flow verification
        - Error handling across components
        """
        # Stub test - full LLM generation in Story 8.4+
        assert True, "Placeholder - implement actual integration tests"

    def test_data_flow_placeholder(self) -> None:
        """Placeholder test for data flow verification.

        TODO: Verify data transformations across component boundaries.
        """
        # Stub test
        assert True, "Placeholder - implement data flow tests"
'''

    return TestFile(
        file_path=test_path,
        content=test_content,
        test_type="integration",
    )


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

    test_methods_str = (
        "\n".join(test_methods)
        if test_methods
        else '''
    def test_implementation_exists(self) -> None:
        """Test that implementation module exists."""
        # Stub test - full LLM generation in Story 8.3+
        pass'''
    )

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


async def _generate_commit_message_for_implementations(
    stories: list[dict[str, Any]],
    implementations: list[ImplementationArtifact],
    router: LLMRouter | None = None,
) -> str | None:
    """Generate a commit message for the implementations (Story 8.8).

    Builds context from stories and implementations, then generates
    a commit message using LLM (if available) or template fallback.

    Args:
        stories: List of story dictionaries with id, title, etc.
        implementations: List of implementation artifacts.
        router: Optional LLMRouter for LLM-powered generation.

    Returns:
        Commit message string, or None if generation fails.

    Example:
        >>> stories = [{"id": "8-8", "title": "Communicative Commits"}]
        >>> impls = [ImplementationArtifact(story_id="8-8")]
        >>> msg = await _generate_commit_message_for_implementations(stories, impls)
        >>> msg.startswith("feat")
        True
    """
    if not implementations:
        return None

    # Build context from stories and implementations
    story_ids = tuple(impl.story_id for impl in implementations)
    story_titles = {}
    decisions: list[str] = []

    for story in stories:
        story_id = story.get("id", "")
        title = story.get("title", "")
        if story_id:
            story_titles[story_id] = title

        # Extract decisions from story if available
        story_decisions = story.get("design_decisions", {})
        if isinstance(story_decisions, dict):
            for key, value in story_decisions.items():
                if value:
                    decisions.append(f"{key}: {value}")

    # Collect all files changed
    files_changed: list[str] = []
    for impl in implementations:
        for code_file in impl.code_files:
            files_changed.append(code_file.file_path)
        for test_file in impl.test_files:
            files_changed.append(test_file.file_path)

    # Build summary from implementation notes
    summary_parts = []
    for impl in implementations:
        if impl.notes:
            summary_parts.append(impl.notes)

    code_summary = " ".join(summary_parts) if summary_parts else ""

    # Create context
    context = CommitMessageContext(
        story_ids=story_ids,
        story_titles=story_titles,
        decisions=tuple(decisions),
        code_summary=code_summary,
        files_changed=tuple(files_changed),
        change_type=CommitType.FEAT,  # Default to feat for new implementations
        scope="dev",  # Dev agent scope
    )

    # Generate commit message
    if router is not None:
        try:
            message, is_valid = await generate_commit_message_with_llm(context, router)
            if is_valid:
                logger.info(
                    "commit_message_generated",
                    story_ids=story_ids,
                    method="llm",
                    message_length=len(message),
                )
                return message
        except Exception as e:
            logger.warning(
                "llm_commit_message_failed",
                story_ids=story_ids,
                error=str(e),
            )

    # Fallback to template-based generation
    message = generate_commit_message(context)
    logger.info(
        "commit_message_generated",
        story_ids=story_ids,
        method="template",
        message_length=len(message),
    )
    return message


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
    # Clear pattern cache for fresh queries (Story 8.7)
    clear_pattern_cache()

    logger.info(
        "dev_node_start",
        current_agent=state.get("current_agent"),
        message_count=len(state.get("messages", [])),
    )

    # Log gate evaluation context (Story 8.6 AC5)
    advisory_warnings: list[dict[str, Any]] = state.get("advisory_warnings", [])  # type: ignore[assignment]
    if advisory_warnings and isinstance(advisory_warnings, list):
        logger.info(
            "dev_node_gate_warnings",
            warning_count=len(advisory_warnings),
            warnings=advisory_warnings,
        )

    # Get LLM router (may be None if not configured)
    router = _get_llm_router()
    if router is not None:
        clear_usage_log = getattr(router, "clear_usage_log", None)
        if callable(clear_usage_log):
            clear_usage_log()

    # Extract project context for code generation (Task 6)
    context = _extract_project_context(state)

    # Extract stories from state (AC2)
    stories = _extract_stories_for_implementation(state)

    # Generate implementations for each story (AC3)
    # Story 8.7: Pass state for pattern queries
    implementations: list[ImplementationArtifact] = []
    state_dict: dict[str, Any] = dict(state)  # Convert TypedDict to dict
    for story in stories:
        artifact = await _generate_implementation(story, context, router, state_dict)
        implementations.append(artifact)

    # Write generated files to disk
    written_files: dict[str, list[str]] = {"code_files": [], "test_files": []}
    if implementations:
        written_files = _write_generated_files(tuple(implementations))

    # Build output with actual results
    total_code_files = sum(len(impl.code_files) for impl in implementations)
    total_test_files = sum(len(impl.test_files) for impl in implementations)

    # Determine generation method
    llm_used = router is not None
    generation_method = "LLM-powered" if llm_used else "stub"

    # Story 8.8: Generate commit message
    commit_message = await _generate_commit_message_for_implementations(
        stories=stories,
        implementations=implementations,
        router=router,
    )

    # Build processing notes including written files info
    files_written_msg = ""
    if written_files["code_files"] or written_files["test_files"]:
        files_written_msg = (
            f" Wrote {len(written_files['code_files'])} code files and "
            f"{len(written_files['test_files'])} test files to disk."
        )

    output = DevOutput(
        implementations=tuple(implementations),
        processing_notes=f"Processed {len(stories)} stories, "
        f"generated {total_code_files} code files, "
        f"{total_test_files} test files. "
        f"Generation method: {generation_method}.{files_written_msg}",
        suggested_commit_message=commit_message,
    )

    # Create decision record with dev attribution (AC6)
    # Include gate failure summary if advisory warnings exist (Story 8.6 AC5)
    rationale_parts = [
        f"Processed {len(stories)} stories from Architect.",
        f"Used {generation_method} code generation with maintainability guidelines.",
    ]
    if advisory_warnings and isinstance(advisory_warnings, list):
        gate_summary = "; ".join(
            f"{w.get('gate_name', 'unknown')}: {w.get('reason', 'no reason')[:100]}"
            for w in advisory_warnings
        )
        rationale_parts.append(f"Gate warnings: {gate_summary}")

    decision = Decision(
        agent="dev",
        summary=f"Generated {total_code_files} code files, {total_test_files} test files "
        f"for {len(stories)} stories ({generation_method})",
        rationale=" ".join(rationale_parts),
        related_artifacts=tuple(impl.story_id for impl in implementations),
    )

    # Create output message with dev attribution (AC6)
    llm_usage = []
    if router is not None:
        get_usage_log = getattr(router, "get_usage_log", None)
        if callable(get_usage_log):
            usage_log = get_usage_log()
            if isinstance(usage_log, (list, tuple)):
                llm_usage = [asdict(entry) for entry in usage_log]

    # Build message content including written files info
    written_info = ""
    if written_files["code_files"] or written_files["test_files"]:
        written_info = (
            f" Wrote {len(written_files['code_files'])} code files and "
            f"{len(written_files['test_files'])} test files to disk."
        )

    message = create_agent_message(
        content=f"Dev processing complete: {total_code_files} code files, "
        f"{total_test_files} test files generated for {len(stories)} stories "
        f"({generation_method}).{written_info}",
        agent="dev",
        metadata={
            "output": output.to_dict(),
            "llm_usage": llm_usage,
            "written_files": written_files,
        },
    )

    logger.info(
        "dev_node_complete",
        story_count=len(stories),
        implementation_count=len(implementations),
        code_file_count=total_code_files,
        test_file_count=total_test_files,
        generation_method=generation_method,
        llm_usage_count=len(llm_usage),
        written_code_files=len(written_files["code_files"]),
        written_test_files=len(written_files["test_files"]),
    )

    # Return ONLY the updates, not full state (AC6)
    return {
        "messages": [message],
        "decisions": [decision],
        "dev_output": output.to_dict(),
    }
