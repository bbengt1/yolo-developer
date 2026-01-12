"""Unit tests for commit message utilities (Story 8.8, Tasks 8, 9, 10, 11).

Tests for:
- CommitType enum values (Task 8)
- CommitMessageContext dataclass construction and serialization (Task 8)
- CommitMessageValidationResult dataclass construction and serialization (Task 8)
- generate_commit_message template-based generation (Task 9)
- validate_commit_message validation (Task 10)
- generate_commit_message_with_llm LLM generation (Task 11)

These tests follow the red-green-refactor cycle, written before implementation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestCommitType:
    """Test suite for CommitType enum."""

    def test_commit_type_feat_value(self) -> None:
        """Test that FEAT has correct value."""
        from yolo_developer.agents.dev.commit_utils import CommitType

        assert CommitType.FEAT.value == "feat"

    def test_commit_type_fix_value(self) -> None:
        """Test that FIX has correct value."""
        from yolo_developer.agents.dev.commit_utils import CommitType

        assert CommitType.FIX.value == "fix"

    def test_commit_type_refactor_value(self) -> None:
        """Test that REFACTOR has correct value."""
        from yolo_developer.agents.dev.commit_utils import CommitType

        assert CommitType.REFACTOR.value == "refactor"

    def test_commit_type_test_value(self) -> None:
        """Test that TEST has correct value."""
        from yolo_developer.agents.dev.commit_utils import CommitType

        assert CommitType.TEST.value == "test"

    def test_commit_type_docs_value(self) -> None:
        """Test that DOCS has correct value."""
        from yolo_developer.agents.dev.commit_utils import CommitType

        assert CommitType.DOCS.value == "docs"

    def test_commit_type_chore_value(self) -> None:
        """Test that CHORE has correct value."""
        from yolo_developer.agents.dev.commit_utils import CommitType

        assert CommitType.CHORE.value == "chore"

    def test_commit_type_style_value(self) -> None:
        """Test that STYLE has correct value."""
        from yolo_developer.agents.dev.commit_utils import CommitType

        assert CommitType.STYLE.value == "style"

    def test_commit_type_is_string_enum(self) -> None:
        """Test that CommitType inherits from str for string operations."""
        from yolo_developer.agents.dev.commit_utils import CommitType

        # Should be usable as a string via .value
        assert f"{CommitType.FEAT.value}: description" == "feat: description"
        # String comparison should work
        assert CommitType.FEAT == "feat"


class TestCommitMessageContext:
    """Test suite for CommitMessageContext dataclass."""

    def test_context_minimal_construction(self) -> None:
        """Test construction with only required fields."""
        from yolo_developer.agents.dev.commit_utils import CommitMessageContext

        context = CommitMessageContext(story_ids=("8-8",))

        assert context.story_ids == ("8-8",)
        assert context.story_titles == {}
        assert context.decisions == ()
        assert context.code_summary == ""
        assert context.files_changed == ()
        assert context.scope is None

    def test_context_full_construction(self) -> None:
        """Test construction with all fields."""
        from yolo_developer.agents.dev.commit_utils import CommitMessageContext, CommitType

        context = CommitMessageContext(
            story_ids=("8-8", "8-9"),
            story_titles={"8-8": "Communicative Commits", "8-9": "Another Story"},
            decisions=("Use conventional commits", "Add story references"),
            code_summary="Add commit message generation",
            files_changed=("commit_utils.py", "node.py"),
            change_type=CommitType.FEAT,
            scope="dev",
        )

        assert context.story_ids == ("8-8", "8-9")
        assert context.story_titles == {"8-8": "Communicative Commits", "8-9": "Another Story"}
        assert context.decisions == ("Use conventional commits", "Add story references")
        assert context.code_summary == "Add commit message generation"
        assert context.files_changed == ("commit_utils.py", "node.py")
        assert context.change_type == CommitType.FEAT
        assert context.scope == "dev"

    def test_context_is_frozen(self) -> None:
        """Test that CommitMessageContext is immutable (frozen)."""
        from yolo_developer.agents.dev.commit_utils import CommitMessageContext

        context = CommitMessageContext(story_ids=("8-8",))

        with pytest.raises(AttributeError):
            context.story_ids = ("different",)  # type: ignore[misc]

    def test_context_to_dict(self) -> None:
        """Test serialization to dictionary."""
        from yolo_developer.agents.dev.commit_utils import CommitMessageContext, CommitType

        context = CommitMessageContext(
            story_ids=("8-8",),
            story_titles={"8-8": "Communicative Commits"},
            decisions=("Decision 1",),
            code_summary="Summary",
            files_changed=("file1.py",),
            change_type=CommitType.FEAT,
            scope="dev",
        )

        result = context.to_dict()

        assert result["story_ids"] == ["8-8"]
        assert result["story_titles"] == {"8-8": "Communicative Commits"}
        assert result["decisions"] == ["Decision 1"]
        assert result["code_summary"] == "Summary"
        assert result["files_changed"] == ["file1.py"]
        assert result["change_type"] == "feat"
        assert result["scope"] == "dev"

    def test_context_default_change_type(self) -> None:
        """Test default change_type is FEAT."""
        from yolo_developer.agents.dev.commit_utils import CommitMessageContext, CommitType

        context = CommitMessageContext(story_ids=("8-8",))

        assert context.change_type == CommitType.FEAT


class TestCommitMessageValidationResult:
    """Test suite for CommitMessageValidationResult dataclass."""

    def test_validation_result_default_values(self) -> None:
        """Test default values for validation result."""
        from yolo_developer.agents.dev.commit_utils import CommitMessageValidationResult

        result = CommitMessageValidationResult()

        assert result.passed is True
        assert result.subject_line == ""
        assert result.body_lines == []
        assert result.warnings == []
        assert result.errors == []

    def test_validation_result_with_warnings(self) -> None:
        """Test validation result with warnings."""
        from yolo_developer.agents.dev.commit_utils import CommitMessageValidationResult

        result = CommitMessageValidationResult(
            passed=True,
            subject_line="feat: add feature",
            body_lines=["Description line"],
            warnings=["Subject line exceeds 50 characters"],
            errors=[],
        )

        assert result.passed is True
        assert result.subject_line == "feat: add feature"
        assert result.body_lines == ["Description line"]
        assert len(result.warnings) == 1
        assert len(result.errors) == 0

    def test_validation_result_with_errors(self) -> None:
        """Test validation result with errors (passed=False)."""
        from yolo_developer.agents.dev.commit_utils import CommitMessageValidationResult

        result = CommitMessageValidationResult(
            passed=False,
            subject_line="invalid subject",
            body_lines=[],
            warnings=[],
            errors=["Subject line does not follow conventional commit format"],
        )

        assert result.passed is False
        assert len(result.errors) == 1

    def test_validation_result_to_dict(self) -> None:
        """Test serialization to dictionary."""
        from yolo_developer.agents.dev.commit_utils import CommitMessageValidationResult

        result = CommitMessageValidationResult(
            passed=True,
            subject_line="feat(dev): add commit utils",
            body_lines=["Line 1", "Line 2"],
            warnings=["Warning 1"],
            errors=[],
        )

        data = result.to_dict()

        assert data["passed"] is True
        assert data["subject_line"] == "feat(dev): add commit utils"
        assert data["body_line_count"] == 2
        assert data["warning_count"] == 1
        assert data["error_count"] == 0
        assert data["warnings"] == ["Warning 1"]
        assert data["errors"] == []

    def test_validation_result_is_mutable(self) -> None:
        """Test that CommitMessageValidationResult is mutable (not frozen)."""
        from yolo_developer.agents.dev.commit_utils import CommitMessageValidationResult

        result = CommitMessageValidationResult()

        # Should be able to modify
        result.passed = False
        result.warnings.append("New warning")

        assert result.passed is False
        assert len(result.warnings) == 1


# =============================================================================
# Task 9: Tests for Template-Based Message Generation (AC: 1, 2, 3, 4)
# =============================================================================


class TestGenerateCommitMessage:
    """Test suite for generate_commit_message function."""

    def test_generate_with_minimal_context(self) -> None:
        """Test generation with only required fields (AC1)."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            generate_commit_message,
        )

        context = CommitMessageContext(story_ids=("8-8",))
        message = generate_commit_message(context)

        # Should have conventional commit format
        assert message.startswith("feat:")
        # Should include story reference
        assert "Story: 8-8" in message

    def test_generate_with_story_title(self) -> None:
        """Test generation includes story title in subject (AC1, AC2)."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            CommitType,
            generate_commit_message,
        )

        context = CommitMessageContext(
            story_ids=("8-8",),
            story_titles={"8-8": "Communicative Commits"},
            change_type=CommitType.FEAT,
        )
        message = generate_commit_message(context)

        # Subject should include story title
        lines = message.split("\n")
        assert "communicative" in lines[0].lower()
        # Story reference in body
        assert "Story: 8-8" in message

    def test_generate_with_scope(self) -> None:
        """Test generation includes scope (AC1)."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            CommitType,
            generate_commit_message,
        )

        context = CommitMessageContext(
            story_ids=("8-8",),
            change_type=CommitType.FEAT,
            scope="dev",
        )
        message = generate_commit_message(context)

        # Should have scope in parentheses
        assert message.startswith("feat(dev):")

    def test_generate_with_decisions(self) -> None:
        """Test generation includes decision rationale (AC3)."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            generate_commit_message,
        )

        context = CommitMessageContext(
            story_ids=("8-8",),
            decisions=("Use conventional commits for clarity",),
        )
        message = generate_commit_message(context)

        # Should include decision in body
        assert "Decision:" in message
        assert "conventional commits" in message

    def test_generate_with_multiple_decisions(self) -> None:
        """Test generation with multiple decisions (AC3)."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            generate_commit_message,
        )

        context = CommitMessageContext(
            story_ids=("8-8",),
            decisions=(
                "Use conventional commits",
                "Follow ADR-003 for LLM tier",
            ),
        )
        message = generate_commit_message(context)

        # Should include both decisions
        assert "Decisions:" in message
        assert "conventional commits" in message
        assert "ADR-003" in message

    def test_generate_with_code_summary(self) -> None:
        """Test generation includes code summary (AC4)."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            generate_commit_message,
        )

        context = CommitMessageContext(
            story_ids=("8-8",),
            code_summary="Add commit message generation utilities",
        )
        message = generate_commit_message(context)

        # Code summary should be in body
        assert "Add commit message generation utilities" in message

    def test_generate_subject_length_enforcement(self) -> None:
        """Test subject line respects length limits (AC4)."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            generate_commit_message,
        )

        context = CommitMessageContext(
            story_ids=("8-8",),
            story_titles={
                "8-8": "A Very Long Story Title That Exceeds The Fifty Character Limit"
            },
        )
        message = generate_commit_message(context)

        # Subject should be truncated
        lines = message.split("\n")
        # Should be reasonably short (soft limit is 50, but truncation adds "...")
        assert len(lines[0]) <= 72  # Hard limit

    def test_generate_with_multiple_stories(self) -> None:
        """Test generation with multiple story references (AC2)."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            generate_commit_message,
        )

        context = CommitMessageContext(
            story_ids=("8-8", "8-9"),
            story_titles={"8-8": "Communicative Commits", "8-9": "Another Story"},
        )
        message = generate_commit_message(context)

        # Should include both story references
        assert "Story: 8-8" in message
        assert "Story: 8-9" in message

    def test_generate_different_change_types(self) -> None:
        """Test generation with different commit types (AC1)."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            CommitType,
            generate_commit_message,
        )

        for commit_type in [CommitType.FIX, CommitType.REFACTOR, CommitType.DOCS]:
            context = CommitMessageContext(
                story_ids=("8-8",),
                change_type=commit_type,
            )
            message = generate_commit_message(context)

            # Should start with the correct type
            assert message.startswith(f"{commit_type.value}:")


# =============================================================================
# Task 10: Tests for Commit Message Validation (AC: 6)
# =============================================================================


class TestValidateCommitMessage:
    """Test suite for validate_commit_message function."""

    def test_validate_valid_message(self) -> None:
        """Test validation passes for valid message."""
        from yolo_developer.agents.dev.commit_utils import validate_commit_message

        result = validate_commit_message("feat: add login feature")

        assert result.passed is True
        assert result.subject_line == "feat: add login feature"
        assert len(result.errors) == 0

    def test_validate_with_scope(self) -> None:
        """Test validation passes for message with scope."""
        from yolo_developer.agents.dev.commit_utils import validate_commit_message

        result = validate_commit_message("feat(auth): add login feature")

        assert result.passed is True
        assert len(result.errors) == 0

    def test_validate_invalid_format(self) -> None:
        """Test validation fails for invalid format."""
        from yolo_developer.agents.dev.commit_utils import validate_commit_message

        result = validate_commit_message("This is not a conventional commit")

        assert result.passed is False
        assert len(result.errors) > 0
        assert "conventional commit format" in result.errors[0].lower()

    def test_validate_empty_message(self) -> None:
        """Test validation fails for empty message."""
        from yolo_developer.agents.dev.commit_utils import validate_commit_message

        result = validate_commit_message("")

        assert result.passed is False
        assert "empty" in result.errors[0].lower()

    def test_validate_long_subject_warning(self) -> None:
        """Test validation warns for subject exceeding 50 chars."""
        from yolo_developer.agents.dev.commit_utils import validate_commit_message

        # 51 character subject
        long_subject = "feat: " + "x" * 45  # "feat: " is 6 chars, total 51
        result = validate_commit_message(long_subject)

        assert result.passed is True  # Warning, not error
        assert len(result.warnings) > 0
        assert "50" in result.warnings[0]

    def test_validate_long_subject_error(self) -> None:
        """Test validation fails for subject exceeding 72 chars."""
        from yolo_developer.agents.dev.commit_utils import validate_commit_message

        # 73 character subject
        very_long_subject = "feat: " + "x" * 67  # "feat: " is 6 chars, total 73
        result = validate_commit_message(very_long_subject)

        assert result.passed is False
        assert len(result.errors) > 0
        assert "72" in result.errors[0]

    def test_validate_missing_blank_line(self) -> None:
        """Test validation warns for missing blank line between subject and body."""
        from yolo_developer.agents.dev.commit_utils import validate_commit_message

        message = "feat: add feature\nThis is the body without blank line"
        result = validate_commit_message(message)

        # Should pass but with warning
        assert result.passed is True
        assert len(result.warnings) > 0
        assert "blank line" in result.warnings[0].lower()

    def test_validate_with_proper_body(self) -> None:
        """Test validation passes for message with proper body formatting."""
        from yolo_developer.agents.dev.commit_utils import validate_commit_message

        message = """feat: add feature

This is the body with proper blank line separation.

Story: 8-8"""
        result = validate_commit_message(message)

        assert result.passed is True
        assert len(result.body_lines) > 0

    def test_validate_extracts_body_lines(self) -> None:
        """Test validation extracts body lines correctly."""
        from yolo_developer.agents.dev.commit_utils import validate_commit_message

        message = """feat: add feature

Line 1 of body
Line 2 of body"""
        result = validate_commit_message(message)

        assert result.subject_line == "feat: add feature"
        assert len(result.body_lines) == 2

    def test_validate_all_commit_types(self) -> None:
        """Test validation accepts all conventional commit types."""
        from yolo_developer.agents.dev.commit_utils import validate_commit_message

        types = ["feat", "fix", "refactor", "test", "docs", "chore", "style"]
        for commit_type in types:
            result = validate_commit_message(f"{commit_type}: description")
            assert result.passed is True, f"Failed for type: {commit_type}"


# =============================================================================
# Task 11: Tests for LLM-Powered Generation (AC: 5)
# =============================================================================


class TestGenerateCommitMessageWithLLM:
    """Test suite for generate_commit_message_with_llm function."""

    @pytest.mark.asyncio
    async def test_llm_generation_success(self) -> None:
        """Test LLM generation succeeds with valid response."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            CommitType,
            generate_commit_message_with_llm,
        )

        # Mock router
        mock_router = MagicMock()
        mock_router.call = AsyncMock(
            return_value="""feat(dev): add commit message utilities

Implement communicative commit messages for Dev agent.

Story: 8-8"""
        )

        context = CommitMessageContext(
            story_ids=("8-8",),
            story_titles={"8-8": "Communicative Commits"},
            change_type=CommitType.FEAT,
            scope="dev",
        )

        message, is_valid = await generate_commit_message_with_llm(context, mock_router)

        assert is_valid is True
        assert message.startswith("feat(dev):")
        mock_router.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_generation_uses_routine_tier(self) -> None:
        """Test LLM generation uses routine tier per ADR-003."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            generate_commit_message_with_llm,
        )

        mock_router = MagicMock()
        mock_router.call = AsyncMock(return_value="feat: add feature")

        context = CommitMessageContext(story_ids=("8-8",))

        await generate_commit_message_with_llm(context, mock_router)

        # Verify "routine" tier was used
        call_kwargs = mock_router.call.call_args[1]
        assert call_kwargs["tier"] == "routine"

    @pytest.mark.asyncio
    async def test_llm_generation_retry_on_invalid_format(self) -> None:
        """Test LLM generation retries on invalid format."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            generate_commit_message_with_llm,
        )

        mock_router = MagicMock()
        # First call returns invalid, second returns valid
        mock_router.call = AsyncMock(
            side_effect=[
                "Invalid commit message without format",
                "feat: valid message after retry",
            ]
        )

        context = CommitMessageContext(story_ids=("8-8",))

        message, is_valid = await generate_commit_message_with_llm(
            context, mock_router, max_retries=1
        )

        assert is_valid is True
        assert message.startswith("feat:")
        assert mock_router.call.call_count == 2

    @pytest.mark.asyncio
    async def test_llm_generation_fallback_to_template(self) -> None:
        """Test LLM generation falls back to template on failure."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            generate_commit_message_with_llm,
        )

        mock_router = MagicMock()
        # All calls fail
        mock_router.call = AsyncMock(side_effect=Exception("LLM error"))

        context = CommitMessageContext(
            story_ids=("8-8",),
            story_titles={"8-8": "Communicative Commits"},
        )

        message, is_valid = await generate_commit_message_with_llm(
            context, mock_router, max_retries=1
        )

        # Should fall back to template (is_valid=False indicates fallback)
        assert is_valid is False
        assert message.startswith("feat:")  # Template format
        assert "Story: 8-8" in message

    @pytest.mark.asyncio
    async def test_llm_generation_extracts_from_code_block(self) -> None:
        """Test LLM generation extracts message from markdown code block."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            generate_commit_message_with_llm,
        )

        mock_router = MagicMock()
        mock_router.call = AsyncMock(
            return_value="""Here's the commit message:

```
feat: add feature

Story: 8-8
```

This follows conventional commit format."""
        )

        context = CommitMessageContext(story_ids=("8-8",))

        message, is_valid = await generate_commit_message_with_llm(context, mock_router)

        assert is_valid is True
        assert message.startswith("feat:")
        # Should not include the explanation text
        assert "Here's the commit message" not in message

    @pytest.mark.asyncio
    async def test_llm_generation_max_retries_exhausted(self) -> None:
        """Test LLM generation handles max retries exhausted."""
        from yolo_developer.agents.dev.commit_utils import (
            CommitMessageContext,
            generate_commit_message_with_llm,
        )

        mock_router = MagicMock()
        # All calls return invalid format
        mock_router.call = AsyncMock(return_value="Invalid message")

        context = CommitMessageContext(story_ids=("8-8",))

        message, is_valid = await generate_commit_message_with_llm(
            context, mock_router, max_retries=2
        )

        # Should fall back to template
        assert is_valid is False
        # 3 calls total: initial + 2 retries
        assert mock_router.call.call_count == 3


# =============================================================================
# Tests for Module Exports (AC: 8)
# =============================================================================


class TestModuleExports:
    """Test that commit utilities are properly exported from dev module."""

    def test_commit_type_exported_from_dev_module(self) -> None:
        """Test that CommitType is exported from yolo_developer.agents.dev."""
        from yolo_developer.agents.dev import CommitType

        assert CommitType.FEAT.value == "feat"

    def test_commit_message_context_exported_from_dev_module(self) -> None:
        """Test that CommitMessageContext is exported from yolo_developer.agents.dev."""
        from yolo_developer.agents.dev import CommitMessageContext

        context = CommitMessageContext(story_ids=("8-8",))
        assert context.story_ids == ("8-8",)

    def test_commit_message_validation_result_exported_from_dev_module(self) -> None:
        """Test that CommitMessageValidationResult is exported."""
        from yolo_developer.agents.dev import CommitMessageValidationResult

        result = CommitMessageValidationResult()
        assert result.passed is True

    def test_generate_commit_message_exported_from_dev_module(self) -> None:
        """Test that generate_commit_message is exported."""
        from yolo_developer.agents.dev import generate_commit_message

        assert callable(generate_commit_message)

    def test_validate_commit_message_exported_from_dev_module(self) -> None:
        """Test that validate_commit_message is exported."""
        from yolo_developer.agents.dev import validate_commit_message

        assert callable(validate_commit_message)

    def test_generate_commit_message_with_llm_exported_from_dev_module(self) -> None:
        """Test that generate_commit_message_with_llm is exported."""
        from yolo_developer.agents.dev import generate_commit_message_with_llm

        assert callable(generate_commit_message_with_llm)
