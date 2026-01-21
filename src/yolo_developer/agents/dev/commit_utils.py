"""Commit message utilities for Dev agent (Story 8.8).

This module provides commit message generation and validation for the Dev agent:

- CommitType: Enum for conventional commit types
- CommitMessageContext: Context for generating commit messages
- CommitMessageValidationResult: Result of commit message validation
- generate_commit_message: Template-based commit message generation
- generate_commit_message_with_llm: LLM-powered commit message generation
- validate_commit_message: Validate commit message format

All types follow conventional commit specification:
https://www.conventionalcommits.org/

Example:
    >>> from yolo_developer.agents.dev.commit_utils import (
    ...     CommitType,
    ...     CommitMessageContext,
    ...     generate_commit_message,
    ...     validate_commit_message,
    ... )
    >>>
    >>> context = CommitMessageContext(
    ...     story_ids=("8-8",),
    ...     story_titles={"8-8": "Communicative Commits"},
    ...     change_type=CommitType.FEAT,
    ...     scope="dev",
    ... )
    >>> message = generate_commit_message(context)
    >>> result = validate_commit_message(message)
    >>> result.passed
    True

Architecture:
    - ADR-001: Frozen dataclasses for immutable types (CommitMessageContext)
    - ADR-003: Use "routine" tier for LLM generation (cost-effective)
    - ADR-007: Retry with fallback to template-based generation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from yolo_developer.llm.router import LLMRouter

logger = structlog.get_logger(__name__)


# =============================================================================
# Enums
# =============================================================================


class CommitType(str, Enum):
    """Conventional commit types.

    Inherits from str for easy string formatting and comparison.
    Values match conventional commit specification.

    Example:
        >>> CommitType.FEAT.value
        'feat'
        >>> f"{CommitType.FEAT.value}: add feature"
        'feat: add feature'
        >>> CommitType.FEAT == "feat"  # String comparison works
        True
    """

    FEAT = "feat"
    FIX = "fix"
    REFACTOR = "refactor"
    TEST = "test"
    DOCS = "docs"
    CHORE = "chore"
    STYLE = "style"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class CommitMessageContext:
    """Context for generating a commit message.

    Contains all information needed to generate a meaningful commit message
    including story references, decisions, and change details.

    Frozen dataclass per ADR-001 for immutable internal state.

    Attributes:
        story_ids: Tuple of story IDs being implemented.
        story_titles: Mapping of story ID to title for context.
        decisions: Tuple of decision rationales to include in body.
        code_summary: Brief summary of code changes.
        files_changed: Tuple of file paths changed.
        change_type: Primary type of change (feat, fix, etc.).
        scope: Optional scope for the commit (e.g., "dev", "config").

    Example:
        >>> context = CommitMessageContext(
        ...     story_ids=("8-8",),
        ...     story_titles={"8-8": "Communicative Commits"},
        ...     decisions=("Use conventional commits",),
        ...     change_type=CommitType.FEAT,
        ...     scope="dev",
        ... )
        >>> context.to_dict()
        {'story_ids': ['8-8'], ...}
    """

    story_ids: tuple[str, ...]
    story_titles: dict[str, str] = field(default_factory=dict)
    decisions: tuple[str, ...] = field(default_factory=tuple)
    code_summary: str = ""
    files_changed: tuple[str, ...] = field(default_factory=tuple)
    change_type: CommitType = CommitType.FEAT
    scope: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with lists instead of tuples.
        """
        return {
            "story_ids": list(self.story_ids),
            "story_titles": self.story_titles,
            "decisions": list(self.decisions),
            "code_summary": self.code_summary,
            "files_changed": list(self.files_changed),
            "change_type": self.change_type.value,
            "scope": self.scope,
        }


@dataclass
class CommitMessageValidationResult:
    """Result of commit message validation.

    Contains validation status, extracted components, and any warnings/errors.
    Mutable dataclass to allow building up validation results.

    Attributes:
        passed: Whether validation passed (no errors, warnings OK).
        subject_line: Extracted subject line from message.
        body_lines: Extracted body lines from message.
        warnings: List of warning messages (soft limit violations).
        errors: List of error messages (hard limit violations).

    Example:
        >>> result = CommitMessageValidationResult(
        ...     passed=True,
        ...     subject_line="feat: add feature",
        ...     warnings=["Subject exceeds 50 chars"],
        ... )
        >>> result.to_dict()
        {'passed': True, ...}
    """

    passed: bool = True
    subject_line: str = ""
    body_lines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with validation results and counts.
        """
        return {
            "passed": self.passed,
            "subject_line": self.subject_line,
            "body_line_count": len(self.body_lines),
            "warning_count": len(self.warnings),
            "error_count": len(self.errors),
            "warnings": self.warnings,
            "errors": self.errors,
        }


# =============================================================================
# Constants
# =============================================================================

# Conventional commit pattern: type(scope): description
CONVENTIONAL_COMMIT_PATTERN = re.compile(
    r"^(feat|fix|refactor|test|docs|chore|style)"  # type
    r"(?:\(([^)]+)\))?"  # optional scope in parentheses
    r":\s*"  # colon and optional space
    r"(.+)$",  # description
    re.IGNORECASE,
)

# Subject line limits
SUBJECT_SOFT_LIMIT = 50
SUBJECT_HARD_LIMIT = 72
BODY_LINE_LIMIT = 72


# =============================================================================
# Template-Based Generation (AC1, AC2, AC3, AC4)
# =============================================================================


def generate_commit_message(context: CommitMessageContext) -> str:
    """Generate a commit message from context using templates.

    Creates a conventional commit message with:
    - Subject line: type(scope): description
    - Body: Story reference, decisions, and change summary

    Args:
        context: CommitMessageContext with story and change information.

    Returns:
        Formatted commit message string.

    Example:
        >>> context = CommitMessageContext(
        ...     story_ids=("8-8",),
        ...     story_titles={"8-8": "Communicative Commits"},
        ...     change_type=CommitType.FEAT,
        ...     scope="dev",
        ... )
        >>> msg = generate_commit_message(context)
        >>> msg.startswith("feat(dev):")
        True
    """
    # Build subject line
    subject = _build_subject_line(context)

    # Build body
    body_parts: list[str] = []

    # Add code summary if available
    if context.code_summary:
        body_parts.append(context.code_summary)

    # Add story references (AC2)
    if context.story_ids:
        story_refs = _build_story_references(context)
        body_parts.append(story_refs)

    # Add decisions (AC3)
    if context.decisions:
        decision_text = _build_decision_text(context)
        body_parts.append(decision_text)

    # Combine subject and body
    if body_parts:
        body = "\n\n".join(body_parts)
        message = f"{subject}\n\n{body}"
    else:
        message = subject

    logger.debug(
        "commit_message_generated",
        story_ids=context.story_ids,
        change_type=context.change_type.value,
        message_length=len(message),
    )

    return message


def _build_subject_line(context: CommitMessageContext) -> str:
    """Build the subject line for a commit message.

    Format: type(scope): description

    Args:
        context: CommitMessageContext with change information.

    Returns:
        Subject line string, truncated to fit limits.
    """
    # Start with type
    type_str = context.change_type.value

    # Add scope if present
    if context.scope:
        type_scope = f"{type_str}({context.scope})"
    else:
        type_scope = type_str

    # Build description from story titles or summary
    if context.story_titles and context.story_ids:
        # Use first story title
        first_story_id = context.story_ids[0]
        title = context.story_titles.get(first_story_id, "")
        if title:
            description = _format_description(title)
        else:
            description = f"implement story {first_story_id}"
    elif context.code_summary:
        description = _format_description(context.code_summary)
    else:
        description = "implement changes"

    # Combine and enforce length limits
    subject = f"{type_scope}: {description}"

    # Truncate if needed (prefer soft limit, hard limit is max)
    if len(subject) > SUBJECT_SOFT_LIMIT:
        # Try to truncate description to fit soft limit
        available = SUBJECT_SOFT_LIMIT - len(type_scope) - 2  # ": "
        if available > 10:  # Minimum meaningful description
            description = description[: available - 3] + "..."
            subject = f"{type_scope}: {description}"

    return subject


def _format_description(text: str) -> str:
    """Format description for subject line.

    - Lowercase first letter
    - Remove trailing period
    - Imperative mood (assumed from input)

    Args:
        text: Raw description text.

    Returns:
        Formatted description.
    """
    if not text:
        return ""

    # Lowercase first letter
    description = text[0].lower() + text[1:] if len(text) > 1 else text.lower()

    # Remove trailing period
    description = description.rstrip(".")

    return description


def _build_story_references(context: CommitMessageContext) -> str:
    """Build story reference section for commit body.

    Format:
        Story: 8-8
        Story: 8-9

    Or for single story with title:
        Story: 8-8 (Communicative Commits)

    Args:
        context: CommitMessageContext with story information.

    Returns:
        Formatted story reference string.
    """
    lines: list[str] = []

    for story_id in context.story_ids:
        title = context.story_titles.get(story_id, "")
        if title and len(context.story_ids) == 1:
            # Include title for single story
            lines.append(f"Story: {story_id} ({title})")
        else:
            lines.append(f"Story: {story_id}")

    return "\n".join(lines)


def _build_decision_text(context: CommitMessageContext) -> str:
    """Build decision rationale section for commit body.

    Format:
        Decision: Use conventional commits for clarity
        Decision: Follow ADR-003 for LLM tier selection

    Args:
        context: CommitMessageContext with decisions.

    Returns:
        Formatted decision text.
    """
    if not context.decisions:
        return ""

    if len(context.decisions) == 1:
        return f"Decision: {context.decisions[0]}"

    # Multiple decisions as bullet points
    lines = ["Decisions:"]
    for decision in context.decisions:
        # Wrap long decisions
        wrapped = _wrap_text(f"- {decision}", BODY_LINE_LIMIT)
        lines.append(wrapped)

    return "\n".join(lines)


def _wrap_text(text: str, limit: int) -> str:
    """Wrap text to fit within character limit.

    Simple word-wrap implementation for commit message body.

    Args:
        text: Text to wrap.
        limit: Maximum line length.

    Returns:
        Wrapped text with newlines.
    """
    if len(text) <= limit:
        return text

    words = text.split()
    lines: list[str] = []
    current_line: list[str] = []
    current_length = 0

    for word in words:
        word_len = len(word)
        if current_length + word_len + (1 if current_line else 0) <= limit:
            current_line.append(word)
            current_length += word_len + (1 if current_length > 0 else 0)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_len

    if current_line:
        lines.append(" ".join(current_line))

    return "\n  ".join(lines)  # Indent continuation lines


# =============================================================================
# Commit Message Validation (AC6)
# =============================================================================


def validate_commit_message(message: str) -> CommitMessageValidationResult:
    """Validate a commit message for conventional commit compliance.

    Checks:
    - Conventional commit format in subject line
    - Subject line length (50 char warning, 72 char error)
    - Body formatting (blank line after subject, line wrap at 72)

    Args:
        message: Commit message string to validate.

    Returns:
        CommitMessageValidationResult with passed, warnings, errors.

    Example:
        >>> result = validate_commit_message("feat: add login")
        >>> result.passed
        True
        >>> result = validate_commit_message("invalid subject")
        >>> result.passed
        False
    """
    result = CommitMessageValidationResult()

    if not message or not message.strip():
        result.passed = False
        result.errors.append("Commit message is empty")
        return result

    # Split into lines
    lines = message.strip().split("\n")
    result.subject_line = lines[0]

    # Extract body (after blank line)
    body_start = 1
    if len(lines) > 1:
        # Find blank line separator
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "":
                body_start = i + 1
                break
        result.body_lines = [line for line in lines[body_start:] if line.strip()]

    # Validate subject line format
    match = CONVENTIONAL_COMMIT_PATTERN.match(result.subject_line)
    if not match:
        result.passed = False
        result.errors.append(
            "Subject line does not follow conventional commit format: "
            "<type>(<scope>): <description>"
        )

    # Validate subject line length
    subject_length = len(result.subject_line)
    if subject_length > SUBJECT_HARD_LIMIT:
        result.passed = False
        result.errors.append(
            f"Subject line exceeds hard limit of {SUBJECT_HARD_LIMIT} characters "
            f"(current: {subject_length})"
        )
    elif subject_length > SUBJECT_SOFT_LIMIT:
        result.warnings.append(
            f"Subject line exceeds recommended limit of {SUBJECT_SOFT_LIMIT} characters "
            f"(current: {subject_length})"
        )

    # Validate blank line after subject
    if len(lines) > 1 and lines[1].strip() != "":
        result.warnings.append("Missing blank line between subject and body")

    # Validate body line lengths (skip URLs which are often long)
    url_pattern = re.compile(r"https?://\S+")
    for i, line in enumerate(result.body_lines):
        if len(line) > BODY_LINE_LIMIT:
            # Skip warning for lines containing URLs
            if url_pattern.search(line):
                continue
            result.warnings.append(
                f"Body line {i + 1} exceeds {BODY_LINE_LIMIT} characters (current: {len(line)})"
            )

    logger.debug(
        "commit_message_validated",
        passed=result.passed,
        warning_count=len(result.warnings),
        error_count=len(result.errors),
    )

    return result


# =============================================================================
# LLM-Powered Generation (AC5)
# =============================================================================


async def generate_commit_message_with_llm(
    context: CommitMessageContext,
    router: LLMRouter,
    max_retries: int = 2,
) -> tuple[str, bool]:
    """Generate a commit message using LLM.

    Uses the "routine" tier model for cost-effective generation per ADR-003.
    Falls back to template-based generation on failure per ADR-007.

    Args:
        context: CommitMessageContext with story and change information.
        router: LLMRouter for making LLM calls.
        max_retries: Maximum retry attempts on format issues.

    Returns:
        Tuple of (message, is_valid). Message is the commit message string,
        is_valid indicates if LLM generation succeeded.

    Example:
        >>> message, valid = await generate_commit_message_with_llm(context, router)
        >>> valid
        True
    """
    from yolo_developer.agents.dev.prompts.commit_message import (
        build_commit_message_prompt,
        build_retry_prompt,
    )

    logger.info(
        "llm_commit_message_generation_start",
        story_ids=context.story_ids,
        change_type=context.change_type.value,
    )

    prompt = build_commit_message_prompt(context)

    for attempt in range(max_retries + 1):
        try:
            response = await router.call_task(
                messages=[{"role": "user", "content": prompt}],
                task_type="documentation",
                temperature=0.3,  # Lower for consistency
                max_tokens=512,
            )

            # Extract commit message from response
            message = _extract_commit_message(response)

            # Validate the message
            validation = validate_commit_message(message)

            if validation.passed:
                logger.info(
                    "llm_commit_message_generation_success",
                    story_ids=context.story_ids,
                    attempt=attempt + 1,
                    message_length=len(message),
                )
                return message, True

            # Retry with validation feedback
            if attempt < max_retries:
                logger.warning(
                    "llm_commit_message_retry",
                    attempt=attempt + 1,
                    errors=validation.errors,
                    warnings=validation.warnings,
                )
                prompt = build_retry_prompt(prompt, validation)
            else:
                logger.warning(
                    "llm_commit_message_validation_failed",
                    errors=validation.errors,
                    warnings=validation.warnings,
                )
                # Return the message anyway if it has only warnings
                if not validation.errors:
                    return message, True

        except Exception as e:
            logger.error(
                "llm_commit_message_generation_error",
                error=str(e),
                attempt=attempt + 1,
            )
            if attempt >= max_retries:
                break

    # Fallback to template-based generation
    logger.info(
        "llm_commit_message_fallback_to_template",
        story_ids=context.story_ids,
    )
    fallback_message = generate_commit_message(context)
    return fallback_message, False


def _extract_commit_message(response: str) -> str:
    """Extract commit message from LLM response.

    Handles responses that may include markdown code blocks or explanations.

    Args:
        response: Raw LLM response string.

    Returns:
        Extracted commit message.
    """
    # Check for markdown code block (permissive: any language hint or none)
    code_block_pattern = re.compile(r"```(?:\w*)?\s*\n?(.*?)\n?```", re.DOTALL)
    match = code_block_pattern.search(response)
    if match:
        return match.group(1).strip()

    # Check for lines starting with conventional commit type
    lines = response.strip().split("\n")
    for i, line in enumerate(lines):
        if CONVENTIONAL_COMMIT_PATTERN.match(line):
            # Found the commit message start, take until empty line or end
            message_lines = [line]
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "" and j > i + 1:
                    # Check if this is the subject-body separator or end
                    remaining = lines[j + 1 :] if j + 1 < len(lines) else []
                    if remaining and any(line.strip() for line in remaining):
                        # There's more content, include it
                        message_lines.append("")
                        continue
                    break
                message_lines.append(lines[j])
            return "\n".join(message_lines).strip()

    # Fallback: return the whole response stripped
    return response.strip()
