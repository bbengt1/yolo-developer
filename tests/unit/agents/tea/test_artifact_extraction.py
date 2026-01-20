"""Unit tests for TEA agent artifact extraction (Story 9.1).

Tests for the _extract_artifacts_for_validation function.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import structlog

from yolo_developer.agents.tea.node import _extract_artifacts_for_validation
from yolo_developer.orchestrator.state import YoloState


@pytest.fixture
def empty_state() -> YoloState:
    """Create an empty state for testing."""
    return {
        "messages": [],
        "current_agent": "tea",
        "handoff_context": None,
        "decisions": [],
    }


@pytest.fixture
def state_with_dev_output() -> YoloState:
    """Create a state with dev_output for testing."""
    return {
        "messages": [],
        "current_agent": "tea",
        "handoff_context": None,
        "decisions": [],
        "dev_output": {
            "implementations": [
                {
                    "story_id": "story-001",
                    "code_files": [
                        {
                            "file_path": "src/auth.py",
                            "content": "def authenticate(): pass",
                            "file_type": "source",
                        },
                        {
                            "file_path": "src/utils.py",
                            "content": "def helper(): pass",
                            "file_type": "source",
                        },
                    ],
                    "test_files": [
                        {
                            "file_path": "tests/test_auth.py",
                            "content": "def test_auth(): assert True",
                            "test_type": "unit",
                        },
                    ],
                },
            ],
        },
    }


@pytest.fixture
def state_with_message_metadata() -> YoloState:
    """Create a state with dev message metadata for testing."""
    message = MagicMock()
    message.additional_kwargs = {
        "agent": "dev",
        "output": {
            "implementations": [
                {
                    "story_id": "story-002",
                    "code_files": [
                        {
                            "file_path": "src/main.py",
                            "content": "def main(): pass",
                            "file_type": "source",
                        },
                    ],
                    "test_files": [
                        {
                            "file_path": "tests/test_main.py",
                            "content": "def test_main(): assert True",
                            "test_type": "unit",
                        },
                    ],
                },
            ],
        },
    }

    return {
        "messages": [message],
        "current_agent": "tea",
        "handoff_context": None,
        "decisions": [],
    }


class TestExtractArtifactsFromDevOutput:
    """Tests for extracting artifacts from dev_output."""

    def test_extracts_code_files_from_dev_output(self, state_with_dev_output: YoloState) -> None:
        """Test that code files are extracted from dev_output."""
        artifacts = _extract_artifacts_for_validation(state_with_dev_output)

        code_files = [a for a in artifacts if a["type"] == "code_file"]
        assert len(code_files) == 2
        assert any(a["artifact_id"] == "src/auth.py" for a in code_files)
        assert any(a["artifact_id"] == "src/utils.py" for a in code_files)

    def test_extracts_test_files_from_dev_output(self, state_with_dev_output: YoloState) -> None:
        """Test that test files are extracted from dev_output."""
        artifacts = _extract_artifacts_for_validation(state_with_dev_output)

        test_files = [a for a in artifacts if a["type"] == "test_file"]
        assert len(test_files) == 1
        assert test_files[0]["artifact_id"] == "tests/test_auth.py"
        assert test_files[0]["test_type"] == "unit"

    def test_includes_story_id_in_artifacts(self, state_with_dev_output: YoloState) -> None:
        """Test that story_id is included in each artifact."""
        artifacts = _extract_artifacts_for_validation(state_with_dev_output)

        for artifact in artifacts:
            assert artifact["story_id"] == "story-001"

    def test_includes_content_in_artifacts(self, state_with_dev_output: YoloState) -> None:
        """Test that content is included in each artifact."""
        artifacts = _extract_artifacts_for_validation(state_with_dev_output)

        for artifact in artifacts:
            assert "content" in artifact
            assert artifact["content"] != ""


class TestExtractArtifactsFromMessageMetadata:
    """Tests for extracting artifacts from message metadata."""

    def test_extracts_from_message_when_no_dev_output(
        self, state_with_message_metadata: YoloState
    ) -> None:
        """Test fallback to message metadata when dev_output is missing."""
        artifacts = _extract_artifacts_for_validation(state_with_message_metadata)

        assert len(artifacts) == 2
        assert any(a["artifact_id"] == "src/main.py" for a in artifacts)
        assert any(a["artifact_id"] == "tests/test_main.py" for a in artifacts)

    def test_message_extraction_includes_story_id(
        self, state_with_message_metadata: YoloState
    ) -> None:
        """Test that story_id is extracted from message metadata."""
        artifacts = _extract_artifacts_for_validation(state_with_message_metadata)

        for artifact in artifacts:
            assert artifact["story_id"] == "story-002"


class TestExtractArtifactsEmptyState:
    """Tests for empty state handling."""

    def test_returns_empty_list_for_empty_state(self, empty_state: YoloState) -> None:
        """Test that empty list is returned for empty state."""
        artifacts = _extract_artifacts_for_validation(empty_state)
        assert artifacts == []

    def test_returns_empty_list_for_empty_dev_output(self) -> None:
        """Test that empty list is returned for empty dev_output."""
        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {},
        }
        artifacts = _extract_artifacts_for_validation(state)
        assert artifacts == []

    def test_returns_empty_list_for_empty_implementations(self) -> None:
        """Test that empty list is returned for empty implementations."""
        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {"implementations": []},
        }
        artifacts = _extract_artifacts_for_validation(state)
        assert artifacts == []


class TestExtractArtifactsLogging:
    """Tests for logging during artifact extraction."""

    def test_logs_artifact_count(
        self, state_with_dev_output: YoloState, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that artifact count is logged."""
        # Configure structlog to use standard logging for capture
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(0),
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )

        with caplog.at_level("INFO"):
            artifacts = _extract_artifacts_for_validation(state_with_dev_output)

        # Verify artifacts were extracted
        assert len(artifacts) == 3

    def test_logs_no_artifacts_when_empty(
        self, empty_state: YoloState, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that debug log is emitted when no artifacts found."""
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(0),
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )

        with caplog.at_level("DEBUG"):
            artifacts = _extract_artifacts_for_validation(empty_state)

        assert artifacts == []


class TestExtractArtifactsRobustness:
    """Tests for robustness of artifact extraction."""

    def test_handles_malformed_code_file(self) -> None:
        """Test handling of malformed code file entries."""
        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    {
                        "story_id": "story-001",
                        "code_files": [
                            "not a dict",  # Malformed
                            {"file_path": "valid.py", "content": "pass", "file_type": "source"},
                        ],
                        "test_files": [],
                    },
                ],
            },
        }
        artifacts = _extract_artifacts_for_validation(state)
        # Should only extract the valid file
        assert len(artifacts) == 1
        assert artifacts[0]["artifact_id"] == "valid.py"

    def test_handles_missing_file_path(self) -> None:
        """Test handling of code file without file_path."""
        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    {
                        "story_id": "story-001",
                        "code_files": [
                            {"content": "pass", "file_type": "source"},  # No file_path
                        ],
                        "test_files": [],
                    },
                ],
            },
        }
        artifacts = _extract_artifacts_for_validation(state)
        # Should still extract with "unknown" as artifact_id
        assert len(artifacts) == 1
        assert artifacts[0]["artifact_id"] == "unknown"

    def test_handles_non_dict_implementation(self) -> None:
        """Test handling of non-dict implementation entries."""
        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    "not a dict",
                    {
                        "story_id": "story-001",
                        "code_files": [
                            {"file_path": "valid.py", "content": "pass", "file_type": "source"},
                        ],
                        "test_files": [],
                    },
                ],
            },
        }
        artifacts = _extract_artifacts_for_validation(state)
        # Should only extract from the valid implementation
        assert len(artifacts) == 1
