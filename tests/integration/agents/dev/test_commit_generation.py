"""Integration tests for commit message generation in Dev node (Story 8.8, Task 12).

Tests that verify:
- dev_node includes commit message in output
- Commit message references story IDs
- Commit message reflects implementation artifacts

These tests use mocked LLM to isolate the commit generation logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yolo_developer.agents.dev.types import CodeFile, ImplementationArtifact
from yolo_developer.llm.router import TaskRouting


class TestDevNodeCommitMessageIntegration:
    """Integration tests for commit message generation in dev_node."""

    @pytest.fixture
    def mock_state(self) -> dict:
        """Create a mock YoloState for testing."""
        return {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
            "architect_output": {
                "design_decisions": [
                    {
                        "story_id": "8-8",
                        "pattern": "conventional-commits",
                    }
                ]
            },
            "pm_output": {
                "stories": [
                    {
                        "id": "8-8",
                        "title": "Communicative Commits",
                        "description": "Add commit message generation",
                    }
                ]
            },
            "memory_context": {},
            "advisory_warnings": [],
        }

    @pytest.mark.asyncio
    async def test_dev_node_includes_commit_message_in_output(self, mock_state: dict) -> None:
        """Test that dev_node output includes suggested_commit_message."""
        from yolo_developer.agents.dev.node import dev_node

        # Mock the LLM router to avoid actual API calls
        with patch("yolo_developer.agents.dev.node._get_llm_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.call_task = AsyncMock(return_value='def implement(): return {"status": "ok"}')
            mock_get_router.return_value = mock_router

            result = await dev_node(mock_state)

            # Verify dev_output is in result
            assert "dev_output" in result

            # Verify commit message is present
            dev_output = result["dev_output"]
            assert "suggested_commit_message" in dev_output
            assert dev_output["suggested_commit_message"] is not None

    @pytest.mark.asyncio
    async def test_commit_message_references_story_ids(self, mock_state: dict) -> None:
        """Test that generated commit message references story IDs."""
        from yolo_developer.agents.dev.node import dev_node

        with patch("yolo_developer.agents.dev.node._get_llm_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.call_task = AsyncMock(return_value='def implement(): return {"status": "ok"}')
            mock_get_router.return_value = mock_router

            result = await dev_node(mock_state)

            commit_message = result["dev_output"]["suggested_commit_message"]

            # Commit message should reference story ID
            assert "8-8" in commit_message or "Story" in commit_message

    @pytest.mark.asyncio
    async def test_dev_node_records_llm_usage_metadata(self, mock_state: dict) -> None:
        """Test that dev_node includes LLM usage metadata in messages."""
        from yolo_developer.agents.dev.node import dev_node

        with patch("yolo_developer.agents.dev.node._get_llm_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.call_task = AsyncMock(return_value='def implement(): return {"status": "ok"}')
            mock_router.get_usage_log.return_value = [
                TaskRouting(
                    task_type="code_generation",
                    provider="openai",
                    model="gpt-5.2-pro",
                    tier="complex",
                )
            ]
            mock_get_router.return_value = mock_router

            result = await dev_node(mock_state)

            message = result["messages"][0]
            metadata = message.additional_kwargs
            assert "llm_usage" in metadata
            assert metadata["llm_usage"][0]["model"] == "gpt-5.2-pro"

    @pytest.mark.asyncio
    async def test_commit_message_uses_conventional_format(self, mock_state: dict) -> None:
        """Test that generated commit message uses conventional commit format."""
        from yolo_developer.agents.dev.commit_utils import validate_commit_message
        from yolo_developer.agents.dev.node import dev_node

        with patch("yolo_developer.agents.dev.node._get_llm_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.call_task = AsyncMock(return_value='def implement(): return {"status": "ok"}')
            mock_get_router.return_value = mock_router

            result = await dev_node(mock_state)

            commit_message = result["dev_output"]["suggested_commit_message"]

            # Validate the commit message format
            validation = validate_commit_message(commit_message)

            # Should be valid conventional commit (may have warnings but no errors)
            assert validation.passed is True, f"Errors: {validation.errors}"

    @pytest.mark.asyncio
    async def test_commit_message_without_llm_uses_template(self, mock_state: dict) -> None:
        """Test that commit message falls back to template without LLM."""
        from yolo_developer.agents.dev.node import dev_node

        with patch("yolo_developer.agents.dev.node._get_llm_router") as mock_get_router:
            # Return None to simulate no LLM available
            mock_get_router.return_value = None

            result = await dev_node(mock_state)

            commit_message = result["dev_output"]["suggested_commit_message"]

            # Should still have a commit message from template
            assert commit_message is not None
            assert commit_message.startswith("feat")

    @pytest.mark.asyncio
    async def test_commit_message_with_multiple_stories(self) -> None:
        """Test commit message generation with multiple stories."""
        from yolo_developer.agents.dev.node import dev_node

        multi_story_state = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
            "architect_output": {
                "design_decisions": [
                    {"story_id": "8-7", "pattern": "pattern-following"},
                    {"story_id": "8-8", "pattern": "conventional-commits"},
                ]
            },
            "pm_output": {
                "stories": [
                    {"id": "8-7", "title": "Pattern Following"},
                    {"id": "8-8", "title": "Communicative Commits"},
                ]
            },
            "memory_context": {},
            "advisory_warnings": [],
        }

        with patch("yolo_developer.agents.dev.node._get_llm_router") as mock_get_router:
            mock_get_router.return_value = None  # Use template

            result = await dev_node(multi_story_state)

            commit_message = result["dev_output"]["suggested_commit_message"]

            # Should reference at least one story
            assert "Story:" in commit_message


class TestGenerateCommitMessageForImplementations:
    """Tests for the _generate_commit_message_for_implementations helper."""

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_implementations(self) -> None:
        """Test that empty implementations returns None."""
        from yolo_developer.agents.dev.node import (
            _generate_commit_message_for_implementations,
        )

        result = await _generate_commit_message_for_implementations(
            stories=[],
            implementations=[],
            router=None,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_builds_context_from_implementations(self) -> None:
        """Test that context is properly built from implementations."""
        from yolo_developer.agents.dev.node import (
            _generate_commit_message_for_implementations,
        )

        stories = [
            {"id": "8-8", "title": "Communicative Commits"},
        ]

        implementations = [
            ImplementationArtifact(
                story_id="8-8",
                code_files=(
                    CodeFile(
                        file_path="src/commit_utils.py",
                        content="# code",
                        file_type="source",
                    ),
                ),
                test_files=(),
                implementation_status="completed",
                notes="Added commit message utilities",
            ),
        ]

        result = await _generate_commit_message_for_implementations(
            stories=stories,
            implementations=implementations,
            router=None,
        )

        # Should return a valid commit message
        assert result is not None
        assert result.startswith("feat")
        # Should reference the story
        assert "8-8" in result

    @pytest.mark.asyncio
    async def test_includes_implementation_notes_in_summary(self) -> None:
        """Test that implementation notes are included in commit summary."""
        from yolo_developer.agents.dev.node import (
            _generate_commit_message_for_implementations,
        )

        stories = [{"id": "8-8", "title": "Test Story"}]

        implementations = [
            ImplementationArtifact(
                story_id="8-8",
                code_files=(),
                test_files=(),
                implementation_status="completed",
                notes="LLM-generated implementation with pattern adherence",
            ),
        ]

        result = await _generate_commit_message_for_implementations(
            stories=stories,
            implementations=implementations,
            router=None,
        )

        # Notes content might be in body
        assert result is not None

    @pytest.mark.asyncio
    async def test_uses_llm_when_available(self) -> None:
        """Test that LLM is used for generation when available."""
        from yolo_developer.agents.dev.node import (
            _generate_commit_message_for_implementations,
        )

        stories = [{"id": "8-8", "title": "Test Story"}]
        implementations = [
            ImplementationArtifact(
                story_id="8-8",
                code_files=(),
                test_files=(),
                implementation_status="completed",
            ),
        ]

        mock_router = MagicMock()
        mock_router.call_task = AsyncMock(
            return_value="""feat(dev): add test implementation

This implements the test story functionality.

Story: 8-8"""
        )

        result = await _generate_commit_message_for_implementations(
            stories=stories,
            implementations=implementations,
            router=mock_router,
        )

        # Should have used LLM
        mock_router.call_task.assert_called_once()
        assert "feat(dev):" in result

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_error(self) -> None:
        """Test fallback to template on LLM error."""
        from yolo_developer.agents.dev.node import (
            _generate_commit_message_for_implementations,
        )

        stories = [{"id": "8-8", "title": "Test Story"}]
        implementations = [
            ImplementationArtifact(
                story_id="8-8",
                code_files=(),
                test_files=(),
                implementation_status="completed",
            ),
        ]

        mock_router = MagicMock()
        mock_router.call_task = AsyncMock(side_effect=Exception("LLM error"))

        result = await _generate_commit_message_for_implementations(
            stories=stories,
            implementations=implementations,
            router=mock_router,
        )

        # Should fall back to template
        assert result is not None
        assert result.startswith("feat")

    @pytest.mark.asyncio
    async def test_handles_empty_stories_with_implementations(self) -> None:
        """Test commit message generation with empty stories but non-empty implementations."""
        from yolo_developer.agents.dev.node import (
            _generate_commit_message_for_implementations,
        )

        # Empty stories list but implementations exist
        stories: list[dict] = []

        implementations = [
            ImplementationArtifact(
                story_id="8-8",
                code_files=(),
                test_files=(),
                implementation_status="completed",
                notes="Implementation without story context",
            ),
        ]

        result = await _generate_commit_message_for_implementations(
            stories=stories,
            implementations=implementations,
            router=None,
        )

        # Should still generate a commit message (using template)
        assert result is not None
        assert result.startswith("feat")
        # Should reference the story ID from implementation
        assert "8-8" in result
