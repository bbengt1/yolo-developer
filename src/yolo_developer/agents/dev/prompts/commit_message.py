"""LLM prompt templates for commit message generation (Story 8.8, Task 4).

This module provides prompt templates for generating commit messages using LLM.
Uses the "routine" tier model per ADR-003 for cost-effective generation.

Functions:
    build_commit_message_prompt: Build initial prompt for LLM generation
    build_retry_prompt: Build retry prompt with validation feedback

Example:
    >>> from yolo_developer.agents.dev.prompts.commit_message import (
    ...     build_commit_message_prompt,
    ... )
    >>> from yolo_developer.agents.dev.commit_utils import CommitMessageContext
    >>>
    >>> context = CommitMessageContext(
    ...     story_ids=("8-8",),
    ...     story_titles={"8-8": "Communicative Commits"},
    ... )
    >>> prompt = build_commit_message_prompt(context)
    >>> "conventional commit" in prompt.lower()
    True
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yolo_developer.agents.dev.commit_utils import (
        CommitMessageContext,
        CommitMessageValidationResult,
    )


# =============================================================================
# Prompt Templates
# =============================================================================

COMMIT_MESSAGE_SYSTEM_PROMPT = """You are an expert at writing clear, concise git commit messages.
Your commit messages follow the conventional commit specification exactly.

Key principles:
1. Subject line uses format: type(scope): description
2. Subject is imperative mood, lowercase, max 50 characters
3. Body explains the "why" not the "how"
4. Always reference the story being implemented
5. Include key decision rationale when provided

Respond with ONLY the commit message, no explanation or markdown formatting."""


COMMIT_MESSAGE_PROMPT_TEMPLATE = """Generate a conventional commit message for the following changes:

## Story Information
- Story IDs: {story_ids}
- Story Titles: {story_titles}

## Change Details
- Change Type: {change_type}
- Scope: {scope}
- Files Changed: {files_changed}
- Summary: {code_summary}

## Decisions Made
{decisions}

## Format Requirements
1. Subject line: {change_type}({scope}): <description>
   - Max 50 characters
   - Imperative mood, lowercase
   - No period at end

2. Body (after blank line):
   - Explain WHY this change was made
   - Reference story: "Story: <id>"
   - Include key decisions if provided

## Example Good Commit Messages

Example 1:
```
feat(dev): add commit message generation

Implement communicative commit messages for Dev agent.
Uses LLM-powered generation with template fallback.

Story: 8-8

- Add CommitMessageContext and validation types
- Implement template-based generation
- Integrate with dev_node output
```

Example 2:
```
fix(gates): handle missing state in DoD validation

Prevents KeyError when memory_context is not present.
Uses defensive .get() with default values.

Story: 8-6

Decision: Use defensive programming for optional state keys
```

Generate the commit message now:"""


RETRY_PROMPT_TEMPLATE = """The previous commit message had validation issues:

## Errors
{errors}

## Warnings
{warnings}

Please fix these issues and generate a corrected commit message.
Remember:
- Subject line must start with: feat|fix|refactor|test|docs|chore|style
- Subject line must be under 72 characters (prefer under 50)
- Must have blank line between subject and body

Original context:
{original_prompt}

Generate the corrected commit message:"""


# =============================================================================
# Prompt Building Functions
# =============================================================================


def build_commit_message_prompt(context: CommitMessageContext) -> str:
    """Build the prompt for LLM commit message generation.

    Formats the context into a structured prompt that guides the LLM
    to generate a conventional commit message.

    Args:
        context: CommitMessageContext with story and change information.

    Returns:
        Formatted prompt string for LLM.

    Example:
        >>> context = CommitMessageContext(
        ...     story_ids=("8-8",),
        ...     story_titles={"8-8": "Communicative Commits"},
        ... )
        >>> prompt = build_commit_message_prompt(context)
        >>> "8-8" in prompt
        True
    """
    # Format story information
    story_ids_str = ", ".join(context.story_ids)

    story_titles_lines = []
    for sid in context.story_ids:
        title = context.story_titles.get(sid, "Untitled")
        story_titles_lines.append(f"  - {sid}: {title}")
    story_titles_str = "\n".join(story_titles_lines) if story_titles_lines else "None"

    # Format scope (clarify when omitted)
    scope_str = context.scope if context.scope else "(omit scope)"

    # Format files changed
    if context.files_changed:
        files_str = ", ".join(context.files_changed[:10])  # Limit to 10 files
        if len(context.files_changed) > 10:
            files_str += f" (+{len(context.files_changed) - 10} more)"
    else:
        files_str = "Not specified"

    # Format decisions
    if context.decisions:
        decisions_lines = ["Decisions to include:"]
        for decision in context.decisions:
            decisions_lines.append(f"- {decision}")
        decisions_str = "\n".join(decisions_lines)
    else:
        decisions_str = "No specific decisions to document."

    # Format code summary
    summary_str = context.code_summary if context.code_summary else "Not specified"

    prompt = COMMIT_MESSAGE_PROMPT_TEMPLATE.format(
        story_ids=story_ids_str,
        story_titles=story_titles_str,
        change_type=context.change_type.value,
        scope=scope_str,
        files_changed=files_str,
        code_summary=summary_str,
        decisions=decisions_str,
    )

    return prompt


def build_retry_prompt(
    original_prompt: str,
    validation_result: CommitMessageValidationResult,
) -> str:
    """Build a retry prompt with validation feedback.

    Used when the initial LLM response fails validation. Includes
    the validation errors and warnings to guide correction.

    Args:
        original_prompt: The original prompt that was used.
        validation_result: The validation result with errors and warnings.

    Returns:
        Retry prompt string with feedback.

    Example:
        >>> from yolo_developer.agents.dev.commit_utils import (
        ...     CommitMessageValidationResult,
        ... )
        >>> result = CommitMessageValidationResult(
        ...     passed=False,
        ...     errors=["Invalid format"],
        ... )
        >>> retry = build_retry_prompt("original prompt", result)
        >>> "Invalid format" in retry
        True
    """
    # Format errors
    if validation_result.errors:
        errors_str = "\n".join(f"- {e}" for e in validation_result.errors)
    else:
        errors_str = "None"

    # Format warnings
    if validation_result.warnings:
        warnings_str = "\n".join(f"- {w}" for w in validation_result.warnings)
    else:
        warnings_str = "None"

    prompt = RETRY_PROMPT_TEMPLATE.format(
        errors=errors_str,
        warnings=warnings_str,
        original_prompt=original_prompt,
    )

    return prompt
