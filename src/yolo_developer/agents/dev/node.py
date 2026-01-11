"""Dev agent node for LangGraph orchestration (Story 8.1).

This module provides the dev_node function that integrates with the
LangGraph orchestration workflow. The Dev agent produces implementation
artifacts including code files and test files for stories with designs.

Key Concepts:
- **YoloState Input**: Receives state as TypedDict, not Pydantic
- **Immutable Updates**: Returns state update dict, never mutates input
- **Async I/O**: All LLM calls use async/await
- **Structured Logging**: Uses structlog for audit trail

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

from yolo_developer.agents.dev.types import (
    CodeFile,
    DevOutput,
    ImplementationArtifact,
    TestFile,
)
from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)


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


def _generate_implementation(story: dict[str, Any]) -> ImplementationArtifact:
    """Generate implementation artifact for a story.

    Creates a stub implementation with placeholder code and test files.
    Full LLM-powered implementation will be added in Story 8.2.

    Args:
        story: Story dictionary with id, title, and other fields.

    Returns:
        ImplementationArtifact with stub code and test files.

    Example:
        >>> story = {"id": "story-001", "title": "User Authentication"}
        >>> artifact = _generate_implementation(story)
        >>> artifact.implementation_status
        'completed'
    """
    story_id = story.get("id", "unknown")
    story_title = story.get("title", "Untitled Story")

    # Generate stub code file
    module_name = story_id.replace("-", "_").replace(".", "_")
    code_content = f'''"""Implementation for {story_title} (Story {story_id}).

This module provides the implementation for the story requirements.
Generated by Dev agent (stub - full implementation in Story 8.2+).
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

    # Generate stub test files
    test_files = _generate_tests(story, [code_file])

    artifact = ImplementationArtifact(
        story_id=story_id,
        code_files=(code_file,),
        test_files=tuple(test_files),
        implementation_status="completed",
        notes=f"Stub implementation for {story_title}. Full LLM implementation in Story 8.2+.",
    )

    logger.debug(
        "implementation_generated",
        story_id=story_id,
        code_file_count=len(artifact.code_files),
        test_file_count=len(artifact.test_files),
    )

    return artifact


def _generate_tests(
    story: dict[str, Any], code_files: list[CodeFile]
) -> list[TestFile]:
    """Generate test files for a story's code files.

    Creates a single stub test file for the story. Full LLM-powered
    per-file test generation will be added in Stories 8.3-8.4.

    Args:
        story: Story dictionary with id, title, and other fields.
        code_files: List of code files to generate tests for.

    Returns:
        List containing a single TestFile stub, or empty if no source files.

    Example:
        >>> story = {"id": "story-001", "title": "User Auth"}
        >>> code_files = [CodeFile(...)]
        >>> tests = _generate_tests(story, code_files)
        >>> len(tests)
        1
    """
    story_id = story.get("id", "unknown")
    story_title = story.get("title", "Untitled Story")

    # Check if there are any source files to test
    has_source_files = any(cf.file_type == "source" for cf in code_files)
    if not has_source_files:
        logger.debug(
            "tests_generated",
            story_id=story_id,
            test_count=0,
        )
        return []

    # Generate single test file for the story (stub behavior)
    # Full per-file test generation in Story 8.3
    module_name = story_id.replace("-", "_").replace(".", "_")
    test_path = f"tests/unit/implementations/test_{module_name}.py"

    test_content = f'''"""Unit tests for {story_title} (Story {story_id}).

Generated by Dev agent (stub - full implementation in Story 8.3+).
"""

from __future__ import annotations

import pytest


class Test{module_name.title().replace("_", "")}:
    """Test suite for {story_title}."""

    def test_implementation_returns_dict(self) -> None:
        """Test that implementation returns expected dict."""
        # Stub test - full implementation in Story 8.3
        # Note: Import path assumes src/ is in PYTHONPATH or using editable install
        from yolo_developer.implementations.{module_name} import implement_{module_name}

        result = implement_{module_name}()
        assert isinstance(result, dict)
        assert result["status"] == "implemented"
        assert result["story_id"] == "{story_id}"
'''

    test_file = TestFile(
        file_path=test_path,
        content=test_content,
        test_type="unit",
    )

    logger.debug(
        "tests_generated",
        story_id=story_id,
        test_count=1,
    )

    return [test_file]


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

    # Extract stories from state (AC2)
    stories = _extract_stories_for_implementation(state)

    # Generate implementations for each story (AC3)
    implementations: list[ImplementationArtifact] = []
    for story in stories:
        artifact = _generate_implementation(story)
        implementations.append(artifact)

    # Build output with actual results
    total_code_files = sum(len(impl.code_files) for impl in implementations)
    total_test_files = sum(len(impl.test_files) for impl in implementations)

    output = DevOutput(
        implementations=tuple(implementations),
        processing_notes=f"Processed {len(stories)} stories, "
        f"generated {total_code_files} code files, "
        f"{total_test_files} test files. "
        "Stub implementation - full LLM integration in Story 8.2+.",
    )

    # Create decision record with dev attribution (AC6)
    decision = Decision(
        agent="dev",
        summary=f"Generated {total_code_files} code files, {total_test_files} test files "
        f"for {len(stories)} stories",
        rationale=f"Processed {len(stories)} stories from Architect. "
        "Stub implementation - full LLM-powered code generation in Story 8.2+.",
        related_artifacts=tuple(impl.story_id for impl in implementations),
    )

    # Create output message with dev attribution (AC6)
    message = create_agent_message(
        content=f"Dev processing complete: {total_code_files} code files, "
        f"{total_test_files} test files generated for {len(stories)} stories.",
        agent="dev",
        metadata={"output": output.to_dict()},
    )

    logger.info(
        "dev_node_complete",
        story_count=len(stories),
        implementation_count=len(implementations),
        code_file_count=total_code_files,
        test_file_count=total_test_files,
    )

    # Return ONLY the updates, not full state (AC6)
    return {
        "messages": [message],
        "decisions": [decision],
        "dev_output": output.to_dict(),
    }
