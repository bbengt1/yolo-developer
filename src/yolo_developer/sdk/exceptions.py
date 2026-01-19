"""SDK-specific exception hierarchy (Stories 13.1, 13.5).

This module provides SDK-specific exception types that wrap and enhance
underlying errors with helpful messages while preserving the original
exception chain for debugging.

Example:
    >>> from yolo_developer.sdk.exceptions import SDKError, ClientNotInitializedError
    >>>
    >>> # Raise SDK-specific error
    >>> raise SDKError("Something went wrong")
    >>>
    >>> # Wrap underlying exception
    >>> try:
    ...     config = load_config()
    ... except ConfigurationError as e:
    ...     raise SDKError("Failed to initialize client", original_error=e) from e

References:
    - FR106-FR111: Python SDK requirements
    - AC5: SDK-specific exceptions with helpful error messages
    - Story 13.5: Agent hooks and HookExecutionError
"""

from __future__ import annotations

from typing import Any


class SDKError(Exception):
    """Base exception for all SDK errors.

    All SDK exceptions inherit from this class, making it easy to catch
    any SDK-related error with a single except clause.

    Attributes:
        message: Human-readable error description.
        original_error: The underlying exception that caused this error, if any.
        details: Additional context about the error.

    Example:
        >>> try:
        ...     client.run()
        ... except SDKError as e:
        ...     print(f"SDK Error: {e.message}")
        ...     if e.original_error:
        ...         print(f"Caused by: {e.original_error}")
    """

    def __init__(
        self,
        message: str,
        *,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize SDKError.

        Args:
            message: Human-readable error description.
            original_error: The underlying exception that caused this error.
            details: Additional context about the error.
        """
        super().__init__(message)
        self.message = message
        self.original_error = original_error
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.original_error:
            return f"{self.message} (caused by: {self.original_error})"
        return self.message


class ClientNotInitializedError(SDKError):
    """Raised when operations are attempted on an uninitialized client.

    This error occurs when attempting to use client methods before
    the client has been properly initialized or configured.

    Example:
        >>> client = YoloClient()
        >>> # If project hasn't been initialized
        >>> client.run()  # Raises ClientNotInitializedError
    """

    def __init__(
        self,
        message: str = "Client has not been initialized. Call init() first.",
        *,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ClientNotInitializedError.

        Args:
            message: Human-readable error description.
            original_error: The underlying exception that caused this error.
            details: Additional context about the error.
        """
        super().__init__(message, original_error=original_error, details=details)


class WorkflowExecutionError(SDKError):
    """Raised when workflow execution fails.

    This error occurs when the run() or run_async() methods encounter
    an error during workflow execution.

    Attributes:
        workflow_id: The ID of the workflow that failed, if available.
        agent: The agent that was executing when the error occurred.

    Example:
        >>> try:
        ...     result = await client.run_async(seed_content="Build something")
        ... except WorkflowExecutionError as e:
        ...     print(f"Workflow failed: {e.message}")
        ...     if e.agent:
        ...         print(f"Failed during agent: {e.agent}")
    """

    def __init__(
        self,
        message: str,
        *,
        workflow_id: str | None = None,
        agent: str | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize WorkflowExecutionError.

        Args:
            message: Human-readable error description.
            workflow_id: The ID of the workflow that failed.
            agent: The agent that was executing when the error occurred.
            original_error: The underlying exception that caused this error.
            details: Additional context about the error.
        """
        super().__init__(message, original_error=original_error, details=details)
        self.workflow_id = workflow_id
        self.agent = agent


class SeedValidationError(SDKError):
    """Raised when seed document validation fails.

    This error occurs when the seed() method fails to validate
    the provided seed document.

    Attributes:
        seed_id: The ID of the seed that failed validation, if available.
        validation_errors: List of specific validation failures.

    Example:
        >>> try:
        ...     result = client.seed(content="Incomplete requirements")
        ... except SeedValidationError as e:
        ...     print(f"Seed validation failed: {e.message}")
        ...     for error in e.validation_errors:
        ...         print(f"  - {error}")
    """

    def __init__(
        self,
        message: str,
        *,
        seed_id: str | None = None,
        validation_errors: list[str] | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize SeedValidationError.

        Args:
            message: Human-readable error description.
            seed_id: The ID of the seed that failed validation.
            validation_errors: List of specific validation failures.
            original_error: The underlying exception that caused this error.
            details: Additional context about the error.
        """
        super().__init__(message, original_error=original_error, details=details)
        self.seed_id = seed_id
        self.validation_errors = validation_errors or []


class ConfigurationAPIError(SDKError):
    """Raised when configuration API operations fail.

    This error occurs when update_config(), validate_config(), or save_config()
    methods encounter an error.

    Attributes:
        field: The configuration field that caused the error, if applicable.
        validation_errors: List of validation error messages.

    Example:
        >>> try:
        ...     client.update_config(quality={"test_coverage_threshold": 2.0})
        ... except ConfigurationAPIError as e:
        ...     print(f"Config error: {e.message}")
        ...     for error in e.validation_errors:
        ...         print(f"  - {error}")
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        validation_errors: list[str] | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ConfigurationAPIError.

        Args:
            message: Human-readable error description.
            field: The configuration field that caused the error.
            validation_errors: List of validation error messages.
            original_error: The underlying exception that caused this error.
            details: Additional context about the error.
        """
        super().__init__(message, original_error=original_error, details=details)
        self.field = field
        self.validation_errors = validation_errors or []


class ProjectNotFoundError(SDKError):
    """Raised when a project cannot be found.

    This error occurs when operations require a project directory
    that doesn't exist or doesn't contain a valid YOLO configuration.

    Attributes:
        project_path: The path that was searched for a project.

    Example:
        >>> client = YoloClient(project_path="/nonexistent/path")
        >>> # If the path doesn't exist or isn't a YOLO project
        >>> client.status()  # Raises ProjectNotFoundError
    """

    def __init__(
        self,
        message: str,
        *,
        project_path: str | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ProjectNotFoundError.

        Args:
            message: Human-readable error description.
            project_path: The path that was searched for a project.
            original_error: The underlying exception that caused this error.
            details: Additional context about the error.
        """
        super().__init__(message, original_error=original_error, details=details)
        self.project_path = project_path


class HookExecutionError(SDKError):
    """Raised when a hook execution fails.

    This error is raised when a registered hook (pre or post execution)
    raises an exception during workflow execution. Hook errors are logged
    and recorded in the audit trail but do not block workflow execution.

    Attributes:
        hook_id: The ID of the hook that failed.
        agent: The agent the hook was executing for.
        phase: The execution phase ("pre" or "post").

    Example:
        >>> # Hooks don't block workflow, but you can inspect failures
        >>> try:
        ...     result = await client.run_async(seed_content="Build a REST API")
        ... except WorkflowExecutionError as e:
        ...     # Check if any hook errors were recorded
        ...     for result in hook_results:
        ...         if not result.success:
        ...             print(f"Hook {result.hook_id} failed: {result.error}")
    """

    def __init__(
        self,
        message: str,
        *,
        hook_id: str | None = None,
        agent: str | None = None,
        phase: str | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize HookExecutionError.

        Args:
            message: Human-readable error description.
            hook_id: The ID of the hook that failed.
            agent: The agent the hook was executing for.
            phase: The execution phase ("pre" or "post").
            original_error: The underlying exception that caused this error.
            details: Additional context about the error.
        """
        super().__init__(message, original_error=original_error, details=details)
        self.hook_id = hook_id
        self.agent = agent
        self.phase = phase
