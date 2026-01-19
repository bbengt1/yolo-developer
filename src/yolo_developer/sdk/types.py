"""SDK-specific type definitions (Story 13.1).

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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


def _utc_now() -> datetime:
    """Return current UTC datetime for default factory."""
    return datetime.now(timezone.utc)


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
