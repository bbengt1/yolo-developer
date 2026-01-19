"""Unit tests for SDK exceptions (Story 13.1).

Tests cover:
- SDKError base exception
- Exception attributes and message formatting
- Exception chaining
- All SDK-specific exception types
"""

from __future__ import annotations

from yolo_developer.sdk.exceptions import (
    ClientNotInitializedError,
    ProjectNotFoundError,
    SDKError,
    SeedValidationError,
    WorkflowExecutionError,
)


class TestSDKError:
    """Tests for SDKError base exception."""

    def test_sdk_error_basic_message(self) -> None:
        """Test SDKError with basic message."""
        error = SDKError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.original_error is None
        assert error.details == {}

    def test_sdk_error_with_original_error(self) -> None:
        """Test SDKError wrapping another exception."""
        original = ValueError("Original error")
        error = SDKError("Wrapped error", original_error=original)

        assert "Wrapped error" in str(error)
        assert "Original error" in str(error)
        assert error.original_error is original

    def test_sdk_error_with_details(self) -> None:
        """Test SDKError with additional details."""
        error = SDKError("Error occurred", details={"key": "value", "count": 42})

        assert error.details == {"key": "value", "count": 42}

    def test_sdk_error_inheritance(self) -> None:
        """Test that SDKError inherits from Exception."""
        error = SDKError("Test")
        assert isinstance(error, Exception)


class TestClientNotInitializedError:
    """Tests for ClientNotInitializedError."""

    def test_default_message(self) -> None:
        """Test ClientNotInitializedError default message."""
        error = ClientNotInitializedError()

        assert "not been initialized" in str(error).lower()
        assert "init()" in str(error)

    def test_custom_message(self) -> None:
        """Test ClientNotInitializedError with custom message."""
        error = ClientNotInitializedError("Custom initialization message")

        assert str(error) == "Custom initialization message"

    def test_inherits_from_sdk_error(self) -> None:
        """Test that ClientNotInitializedError inherits from SDKError."""
        error = ClientNotInitializedError()
        assert isinstance(error, SDKError)


class TestWorkflowExecutionError:
    """Tests for WorkflowExecutionError."""

    def test_basic_workflow_error(self) -> None:
        """Test WorkflowExecutionError with basic message."""
        error = WorkflowExecutionError("Workflow failed")

        assert error.message == "Workflow failed"
        assert error.workflow_id is None
        assert error.agent is None

    def test_workflow_error_with_context(self) -> None:
        """Test WorkflowExecutionError with workflow context."""
        error = WorkflowExecutionError(
            "Agent crashed",
            workflow_id="workflow-123",
            agent="analyst",
        )

        assert error.workflow_id == "workflow-123"
        assert error.agent == "analyst"

    def test_workflow_error_with_original(self) -> None:
        """Test WorkflowExecutionError chaining."""
        original = RuntimeError("LLM timeout")
        error = WorkflowExecutionError(
            "Workflow failed due to timeout",
            workflow_id="wf-456",
            original_error=original,
        )

        assert error.original_error is original
        assert "timeout" in str(error).lower()

    def test_inherits_from_sdk_error(self) -> None:
        """Test that WorkflowExecutionError inherits from SDKError."""
        error = WorkflowExecutionError("Test")
        assert isinstance(error, SDKError)


class TestSeedValidationError:
    """Tests for SeedValidationError."""

    def test_basic_seed_error(self) -> None:
        """Test SeedValidationError with basic message."""
        error = SeedValidationError("Seed validation failed")

        assert error.message == "Seed validation failed"
        assert error.seed_id is None
        assert error.validation_errors == []

    def test_seed_error_with_validation_errors(self) -> None:
        """Test SeedValidationError with validation details."""
        error = SeedValidationError(
            "Seed has issues",
            seed_id="seed-123",
            validation_errors=["Missing goals", "Ambiguous requirements"],
        )

        assert error.seed_id == "seed-123"
        assert len(error.validation_errors) == 2
        assert "Missing goals" in error.validation_errors

    def test_inherits_from_sdk_error(self) -> None:
        """Test that SeedValidationError inherits from SDKError."""
        error = SeedValidationError("Test")
        assert isinstance(error, SDKError)


class TestProjectNotFoundError:
    """Tests for ProjectNotFoundError."""

    def test_basic_project_error(self) -> None:
        """Test ProjectNotFoundError with basic message."""
        error = ProjectNotFoundError("Project not found")

        assert error.message == "Project not found"
        assert error.project_path is None

    def test_project_error_with_path(self) -> None:
        """Test ProjectNotFoundError with path details."""
        error = ProjectNotFoundError(
            "No project at /path/to/project",
            project_path="/path/to/project",
        )

        assert error.project_path == "/path/to/project"

    def test_inherits_from_sdk_error(self) -> None:
        """Test that ProjectNotFoundError inherits from SDKError."""
        error = ProjectNotFoundError("Test")
        assert isinstance(error, SDKError)


class TestExceptionChaining:
    """Tests for exception chaining behavior."""

    def test_exception_chain_preserved(self) -> None:
        """Test that exception chain is preserved with raise from."""
        original = ValueError("Root cause")

        try:
            try:
                raise original
            except ValueError as e:
                raise SDKError("Wrapped", original_error=e) from e
        except SDKError as sdk_error:
            assert sdk_error.__cause__ is original
            assert sdk_error.original_error is original

    def test_multiple_exception_levels(self) -> None:
        """Test multi-level exception chaining."""
        root = OSError("Disk error")
        mid = SDKError("Config load failed", original_error=root)
        top = WorkflowExecutionError(
            "Workflow init failed",
            original_error=mid,
        )

        assert top.original_error is mid
        assert mid.original_error is root
