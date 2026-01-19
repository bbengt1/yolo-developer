"""SDK-specific type definitions (Stories 13.1, 13.4, 13.5).

This module provides type definitions for SDK operation results,
enabling full type safety for SDK consumers.

Example:
    >>> from yolo_developer.sdk.types import SeedResult, RunResult, StatusResult
    >>>
    >>> # Type hints for SDK operations
    >>> async def process_seed(content: str) -> SeedResult:
    ...     client = YoloClient()
    ...     return await client.seed_async(content)

References:
    - FR106-FR111: Python SDK requirements
    - AC3: Complete type hints for all SDK operations
    - Story 13.4: Configuration API types
    - Story 13.5: Agent hooks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, runtime_checkable


def _utc_now() -> datetime:
    """Return current UTC datetime for default factory."""
    return datetime.now(timezone.utc)


# =============================================================================
# Hook Protocols (Story 13.5)
# =============================================================================


@runtime_checkable
class PreHook(Protocol):
    """Protocol for pre-execution hooks.

    Pre-hooks fire before an agent begins execution, receiving a read-only
    snapshot of the current workflow state. They can return state modifications
    to inject into the workflow.

    Example:
        >>> def inject_context(agent: str, state: dict) -> dict | None:
        ...     '''Inject custom context before agent runs.'''
        ...     return {"custom_context": "my data"}
        >>>
        >>> # Or return None for no modifications
        >>> def log_state(agent: str, state: dict) -> dict | None:
        ...     '''Log state without modifications.'''
        ...     print(f"Agent {agent} starting with state keys: {state.keys()}")
        ...     return None
    """

    def __call__(self, agent: str, state: dict[str, Any]) -> dict[str, Any] | None:
        """Execute before agent runs.

        Args:
            agent: Name of the agent about to execute (e.g., "analyst", "pm").
            state: Current workflow state (read-only snapshot).

        Returns:
            Dict of state modifications to inject, or None for no changes.
        """
        ...


@runtime_checkable
class PostHook(Protocol):
    """Protocol for post-execution hooks.

    Post-hooks fire after an agent completes execution, receiving both the
    input state and the agent's output. They can modify the output before
    it's applied to the workflow state.

    Example:
        >>> def log_decisions(agent: str, state: dict, output: dict) -> dict | None:
        ...     '''Log agent decisions without modifications.'''
        ...     print(f"Agent {agent} made decisions: {output.get('decisions', [])}")
        ...     return None  # Don't modify output
        >>>
        >>> def filter_output(agent: str, state: dict, output: dict) -> dict | None:
        ...     '''Modify agent output.'''
        ...     if agent == "dev":
        ...         output["additional_checks"] = True
        ...     return output
    """

    def __call__(
        self, agent: str, state: dict[str, Any], output: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Execute after agent completes.

        Args:
            agent: Name of the agent that executed.
            state: Input state the agent received.
            output: Output from the agent.

        Returns:
            Modified output dict, or None to use original output.
        """
        ...


@dataclass(frozen=True)
class HookRegistration:
    """Registration record for an agent hook.

    Attributes:
        hook_id: Unique identifier for this hook registration.
        agent: Target agent name or "*" for all agents.
        phase: Execution phase ("pre" or "post").
        callback: The hook function to execute.
        timestamp: When the hook was registered.

    Example:
        >>> registration = client.list_hooks()[0]
        >>> print(f"Hook {registration.hook_id} targets {registration.agent}")
        >>> print(f"Phase: {registration.phase}")
    """

    hook_id: str
    agent: str
    phase: Literal["pre", "post"]
    callback: PreHook | PostHook
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True)
class HookResult:
    """Result of hook execution for a single hook.

    Attributes:
        hook_id: ID of the hook that executed.
        agent: Agent the hook was executed for.
        phase: Execution phase ("pre" or "post").
        success: Whether the hook executed without error.
        modifications: State/output modifications returned by the hook.
        error: Error message if hook failed.
        timestamp: When the hook executed.

    Example:
        >>> # After workflow run, inspect hook execution results
        >>> for result in hook_results:
        ...     if not result.success:
        ...         print(f"Hook {result.hook_id} failed: {result.error}")
    """

    hook_id: str
    agent: str
    phase: Literal["pre", "post"]
    success: bool
    modifications: dict[str, Any] | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True)
class SeedResult:
    """Result of a seed operation.

    Attributes:
        seed_id: Unique identifier for the processed seed.
        status: Status of the seed operation ('accepted', 'rejected', 'pending').
        goal_count: Number of goals extracted from the seed.
        feature_count: Number of features extracted from the seed.
        constraint_count: Number of constraints extracted from the seed.
        ambiguities: List of detected ambiguities in the seed.
        quality_score: Overall quality score (0.0-1.0).
        warnings: List of warning messages.
        timestamp: When the seed was processed.

    Example:
        >>> result = client.seed("Build an e-commerce platform")
        >>> if result.status == "accepted":
        ...     print(f"Seed {result.seed_id} accepted with score {result.quality_score:.0%}")
    """

    seed_id: str
    status: Literal["accepted", "rejected", "pending"]
    goal_count: int = 0
    feature_count: int = 0
    constraint_count: int = 0
    ambiguities: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    warnings: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True)
class RunResult:
    """Result of a workflow execution.

    Attributes:
        workflow_id: Unique identifier for the workflow execution.
        status: Final status of the workflow ('completed', 'failed', 'cancelled').
        agents_executed: List of agents that executed during the workflow.
        stories_completed: Number of stories completed.
        stories_total: Total number of stories in the sprint.
        duration_seconds: Total execution duration in seconds.
        artifacts: Paths to generated artifacts.
        errors: List of errors that occurred during execution.
        timestamp: When the workflow completed.

    Example:
        >>> result = await client.run_async(seed_content="Build a REST API")
        >>> print(f"Workflow {result.workflow_id}: {result.status}")
        >>> print(f"Completed {result.stories_completed}/{result.stories_total} stories")
    """

    workflow_id: str
    status: Literal["completed", "failed", "cancelled"]
    agents_executed: list[str] = field(default_factory=list)
    stories_completed: int = 0
    stories_total: int = 0
    duration_seconds: float = 0.0
    artifacts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True)
class StatusResult:
    """Result of a status query.

    Attributes:
        project_name: Name of the current project.
        project_path: Path to the project directory.
        is_initialized: Whether the project is initialized.
        current_sprint: Current sprint information, if any.
        active_agent: Currently executing agent, if any.
        workflow_status: Status of any running workflow.
        last_activity: Timestamp of last activity.
        stats: Additional status statistics.

    Example:
        >>> status = client.status()
        >>> if status.is_initialized:
        ...     print(f"Project: {status.project_name}")
        ...     if status.active_agent:
        ...         print(f"Running: {status.active_agent}")
    """

    project_name: str
    project_path: str
    is_initialized: bool
    current_sprint: str | None = None
    active_agent: str | None = None
    workflow_status: Literal["idle", "running", "paused"] = "idle"
    last_activity: datetime | None = None
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditEntry:
    """An entry in the audit trail.

    Attributes:
        entry_id: Unique identifier for the audit entry.
        timestamp: When the entry was created.
        agent: Agent that created the entry.
        decision_type: Type of decision recorded.
        content: Content of the decision or action.
        rationale: Rationale for the decision.
        metadata: Additional metadata.

    Example:
        >>> entries = client.get_audit(agent_filter="analyst")
        >>> for entry in entries:
        ...     print(f"[{entry.timestamp}] {entry.agent}: {entry.content}")
    """

    entry_id: str
    timestamp: datetime
    agent: str
    decision_type: str
    content: str
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitResult:
    """Result of project initialization.

    Attributes:
        project_path: Path to the initialized project.
        project_name: Name of the project.
        config_created: Whether a new config file was created.
        directories_created: List of directories that were created.
        timestamp: When initialization occurred.

    Example:
        >>> result = client.init(project_name="my-project")
        >>> print(f"Project initialized at {result.project_path}")
    """

    project_path: str
    project_name: str
    config_created: bool = False
    directories_created: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True)
class ConfigValidationIssue:
    """A single configuration validation issue.

    Attributes:
        field: The field path that has an issue (e.g., "quality.test_coverage_threshold").
        message: Human-readable description of the issue.
        severity: Whether this is an 'error' (fatal) or 'warning' (non-fatal).

    Example:
        >>> result = client.validate_config()
        >>> for issue in result.issues:
        ...     print(f"[{issue.severity}] {issue.field}: {issue.message}")
    """

    field: str
    message: str
    severity: Literal["error", "warning"] = "warning"


@dataclass(frozen=True)
class ConfigValidationResult:
    """Result of configuration validation.

    Attributes:
        is_valid: True if no fatal errors exist (warnings are OK).
        issues: List of validation issues (both errors and warnings).
        timestamp: When validation occurred.

    Example:
        >>> result = client.validate_config()
        >>> if result.is_valid:
        ...     print("Configuration is valid")
        ... else:
        ...     for issue in result.issues:
        ...         if issue.severity == "error":
        ...             print(f"Error: {issue.message}")
    """

    is_valid: bool
    issues: list[ConfigValidationIssue] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utc_now)

    @property
    def errors(self) -> list[ConfigValidationIssue]:
        """Return only error-level issues."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ConfigValidationIssue]:
        """Return only warning-level issues."""
        return [i for i in self.issues if i.severity == "warning"]


@dataclass(frozen=True)
class ConfigUpdateResult:
    """Result of a configuration update operation.

    Attributes:
        success: Whether the update was successful.
        previous_values: Dictionary of previous values for changed fields.
        new_values: Dictionary of new values for changed fields.
        persisted: Whether changes were persisted to disk.
        validation: Validation result for the new configuration.
        timestamp: When the update occurred.

    Example:
        >>> result = client.update_config(
        ...     quality={"test_coverage_threshold": 0.85},
        ...     persist=True
        ... )
        >>> if result.success:
        ...     print("Configuration updated")
        ...     if result.persisted:
        ...         print("Changes saved to yolo.yaml")
    """

    success: bool
    previous_values: dict[str, Any] = field(default_factory=dict)
    new_values: dict[str, Any] = field(default_factory=dict)
    persisted: bool = False
    validation: ConfigValidationResult | None = None
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True)
class ConfigSaveResult:
    """Result of saving configuration to disk.

    Attributes:
        success: Whether the save was successful.
        config_path: Path where the configuration was saved.
        secrets_excluded: List of secret fields that were excluded from the file.
        timestamp: When the save occurred.

    Example:
        >>> result = client.save_config()
        >>> if result.success:
        ...     print(f"Configuration saved to {result.config_path}")
    """

    success: bool
    config_path: str
    secrets_excluded: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utc_now)
