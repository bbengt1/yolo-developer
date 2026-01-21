"""MCP tool implementations for YOLO Developer.

This module provides the MCP tools that expose YOLO Developer functionality
to external clients via the FastMCP server.

Tools:
    - yolo_seed: Provide seed requirements for autonomous development
    - yolo_run: Execute autonomous sprint
    - yolo_status: Query sprint status
    - yolo_audit: Query audit trail entries

Example:
    >>> from yolo_developer.mcp import mcp
    >>> # yolo_seed tool is automatically registered when this module is imported
    >>> # MCP clients can call: await client.call_tool("yolo_seed", {"content": "..."})

References:
    - Story 14.2: yolo_seed MCP Tool
    - ADR-004: FastMCP 2.x for MCP server implementation
    - FR112-FR117: MCP protocol integration requirements
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import structlog
from langchain_core.messages import HumanMessage

from yolo_developer.audit import (
    AuditFilters,
    InMemoryTraceabilityStore,
    JsonDecisionStore,
    get_audit_filter_service,
)
from yolo_developer.audit.traceability_types import VALID_ARTIFACT_TYPES
from yolo_developer.audit.types import VALID_DECISION_TYPES
from yolo_developer.mcp.server import mcp
from yolo_developer.seed import parse_seed
from yolo_developer.seed.rejection import validate_quality_thresholds
from yolo_developer.seed.report import generate_validation_report


@dataclass
class StoredSeed:
    """A stored seed with metadata.

    Attributes:
        seed_id: Unique identifier for the seed (UUID).
        content: The seed requirements content.
        source: Source type - either "text" or "file".
        created_at: Timestamp when the seed was created.
        content_length: Length of the content in characters.
        file_path: Original file path if source is "file".
    """

    seed_id: str
    content: str
    source: Literal["text", "file"]
    created_at: datetime
    content_length: int
    file_path: str | None = None


@dataclass
class StoredSprint:
    """A stored sprint with metadata for MCP status queries.

    Attributes:
        sprint_id: Unique identifier for the sprint.
        seed_id: Seed identifier used to start the sprint.
        status: Current sprint status ("running", "completed", "failed").
        started_at: Timestamp when sprint execution started.
        thread_id: Thread ID used for checkpointing.
        completed_at: Timestamp when sprint completed (if finished).
        error: Error message if sprint failed.
    """

    sprint_id: str
    seed_id: str
    status: Literal["running", "completed", "failed"]
    started_at: datetime
    thread_id: str
    completed_at: datetime | None = None
    error: str | None = None


logger = structlog.get_logger(__name__)

# In-memory storage for seeds
# This is a simple MVP implementation; can be replaced with persistent storage later
_seeds: dict[str, StoredSeed] = {}
_seeds_lock = threading.Lock()  # Thread safety for concurrent MCP requests
_sprints: dict[str, StoredSprint] = {}
_sprints_lock = threading.Lock()
_sprint_tasks: dict[str, asyncio.Task[None]] = {}
_sprint_tasks_lock = threading.Lock()


def _parse_iso_timestamp(
    value: str | None, field_name: str
) -> tuple[str | None, datetime | None, str | None]:
    """Parse an ISO-8601 timestamp string, returning normalized value or error."""
    if value is None:
        return None, None, None
    stripped = value.strip()
    if not stripped:
        return None, None, f"{field_name} cannot be empty"
    try:
        normalized = stripped[:-1] + "+00:00" if stripped.endswith("Z") else stripped
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None, None, f"{field_name} must be an ISO-8601 timestamp"
    return parsed.isoformat(), parsed, None


def store_seed(
    content: str,
    source: Literal["text", "file"],
    file_path: str | None = None,
) -> StoredSeed:
    """Store a seed and return the stored seed object.

    This function is thread-safe for concurrent MCP requests.

    Args:
        content: The seed requirements content.
        source: Source type - "text" for direct content, "file" for file-based.
        file_path: Original file path if source is "file".

    Returns:
        StoredSeed: The stored seed with generated ID and metadata.

    Example:
        >>> seed = store_seed("Build a REST API", source="text")
        >>> print(seed.seed_id)
        '550e8400-e29b-41d4-a716-446655440000'
    """
    seed_id = str(uuid.uuid4())
    seed = StoredSeed(
        seed_id=seed_id,
        content=content,
        source=source,
        created_at=datetime.now(timezone.utc),
        content_length=len(content),
        file_path=file_path,
    )
    with _seeds_lock:
        _seeds[seed_id] = seed
    return seed


def get_seed(seed_id: str) -> StoredSeed | None:
    """Retrieve a seed by its ID.

    This function is thread-safe for concurrent MCP requests.

    Args:
        seed_id: The unique identifier of the seed.

    Returns:
        StoredSeed if found, None otherwise.

    Example:
        >>> seed = get_seed("550e8400-e29b-41d4-a716-446655440000")
        >>> if seed:
        ...     print(seed.content)
    """
    with _seeds_lock:
        return _seeds.get(seed_id)


def clear_seeds() -> None:
    """Clear all stored seeds.

    This function is primarily intended for testing to reset state between tests.
    It is thread-safe for concurrent MCP requests.

    Note:
        This is exposed in the public API for test fixtures. Production code
        should generally not call this function.
    """
    with _seeds_lock:
        _seeds.clear()


def store_sprint(
    seed_id: str,
    thread_id: str,
) -> StoredSprint:
    """Store a sprint and return the stored sprint object."""
    sprint_id = f"sprint-{uuid.uuid4().hex[:8]}"
    sprint = StoredSprint(
        sprint_id=sprint_id,
        seed_id=seed_id,
        status="running",
        started_at=datetime.now(timezone.utc),
        thread_id=thread_id,
    )
    with _sprints_lock:
        _sprints[sprint_id] = sprint
    return sprint


def get_sprint(sprint_id: str) -> StoredSprint | None:
    """Retrieve a sprint by its ID."""
    with _sprints_lock:
        return _sprints.get(sprint_id)


def _get_sprint_snapshot(sprint_id: str) -> StoredSprint | None:
    """Return a snapshot of sprint data to avoid concurrent mutation issues."""
    with _sprints_lock:
        sprint = _sprints.get(sprint_id)
        if sprint is None:
            return None
        return replace(sprint)


def clear_sprints() -> None:
    """Clear all stored sprints.

    Intended for tests to reset state between runs.
    """
    with _sprints_lock:
        _sprints.clear()
    with _sprint_tasks_lock:
        for task in _sprint_tasks.values():
            task.cancel()
        _sprint_tasks.clear()


def _register_sprint_task(sprint_id: str, task: asyncio.Task[None]) -> None:
    """Track a sprint task for cleanup and status monitoring."""
    with _sprint_tasks_lock:
        _sprint_tasks[sprint_id] = task


def _clear_sprint_task(sprint_id: str) -> None:
    """Remove a completed sprint task from the registry."""
    with _sprint_tasks_lock:
        _sprint_tasks.pop(sprint_id, None)


def _update_sprint_status(
    sprint_id: str,
    status: Literal["running", "completed", "failed"],
    *,
    completed_at: datetime | None = None,
    error: str | None = None,
) -> None:
    with _sprints_lock:
        sprint = _sprints.get(sprint_id)
        if sprint is None:
            return
        sprint.status = status
        sprint.completed_at = completed_at
        sprint.error = error


async def _run_sprint(
    sprint_id: str,
    thread_id: str,
    seed_id: str,
    seed_content: str,
) -> None:
    """Execute the sprint workflow and update sprint status."""
    from yolo_developer.orchestrator import WorkflowConfig, create_initial_state, stream_workflow
    from yolo_developer.orchestrator.session import SessionManager
    from yolo_developer.orchestrator.state import YoloState

    session_manager = SessionManager(Path(".yolo/sessions"))
    config = WorkflowConfig(entry_point="analyst", enable_checkpointing=True)
    initial_state = create_initial_state(
        starting_agent="analyst",
        messages=[HumanMessage(content=seed_content)],
    )
    final_state: YoloState | None = None

    try:
        async for event in stream_workflow(
            initial_state,
            config=config,
            thread_id=thread_id,
        ):
            agent_name = next(iter(event.keys())) if event else None
            if agent_name and agent_name in event:
                final_state = cast(YoloState, event[agent_name])
                await session_manager.save_session(final_state, session_id=thread_id)

        if final_state is not None:
            await session_manager.save_session(final_state, session_id=thread_id)

        _update_sprint_status(
            sprint_id,
            "completed",
            completed_at=datetime.now(timezone.utc),
        )
        logger.info(
            "mcp_sprint_completed",
            sprint_id=sprint_id,
            seed_id=seed_id,
        )
    except Exception as exc:
        _update_sprint_status(
            sprint_id,
            "failed",
            completed_at=datetime.now(timezone.utc),
            error=str(exc),
        )
        logger.exception(
            "mcp_sprint_failed",
            sprint_id=sprint_id,
            seed_id=seed_id,
            error=str(exc),
        )
    finally:
        _clear_sprint_task(sprint_id)


@mcp.tool
async def yolo_seed(
    content: str | None = None,
    file_path: str | None = None,
) -> dict[str, Any]:
    """Provide seed requirements for autonomous development.

    Provide EITHER content (as text) OR file_path (to read from).
    If both are provided, content takes precedence.

    Args:
        content: Seed requirements as plain text.
        file_path: Path to a file containing seed requirements.

    Returns:
        dict: Response with status, seed_id (if successful), and metadata.
            Success: {"status": "accepted", "seed_id": "...", "content_length": N, "source": "text"|"file"}
            Error: {"status": "error", "error": "Error message"}

    Example:
        >>> result = await yolo_seed(content="Build a REST API for user management")
        >>> print(result["seed_id"])
    """
    # Validate input: at least one of content or file_path must be provided
    if content is None and file_path is None:
        return {
            "status": "error",
            "error": "Either content or file_path must be provided",
        }

    # If content is provided, use it (takes precedence over file_path)
    if content is not None:
        # Validate content is not empty or whitespace-only
        if not content.strip():
            return {
                "status": "error",
                "error": "Content cannot be empty or whitespace-only",
            }

        seed = store_seed(content=content, source="text")
        return {
            "status": "accepted",
            "seed_id": seed.seed_id,
            "content_length": seed.content_length,
            "source": "text",
        }

    # file_path is provided (and content is None)
    assert file_path is not None  # For type checker

    # Validate file exists
    path = Path(file_path)
    if not path.exists():
        return {
            "status": "error",
            "error": f"File not found: {file_path}",
        }

    if not path.is_file():
        return {
            "status": "error",
            "error": f"Path is not a file: {file_path}",
        }

    # Read file content
    try:
        file_content = path.read_text(encoding="utf-8")
    except OSError as e:
        return {
            "status": "error",
            "error": f"Error reading file: {e}",
        }

    # Validate file content is not empty
    if not file_content.strip():
        return {
            "status": "error",
            "error": "File content is empty or whitespace-only",
        }

    seed = store_seed(content=file_content, source="file", file_path=file_path)
    return {
        "status": "accepted",
        "seed_id": seed.seed_id,
        "content_length": seed.content_length,
        "source": "file",
    }


@mcp.tool
async def yolo_run(
    seed_id: str | None = None,
    seed_content: str | None = None,
) -> dict[str, Any]:
    """Execute a sprint based on a provided seed.

    Provide either seed_id (preferred) or seed_content. If both are provided,
    seed_id takes precedence.
    """
    if seed_id is None and seed_content is None:
        return {
            "status": "error",
            "error": "Either seed_id or seed_content must be provided",
        }

    seed: StoredSeed | None = None
    if seed_id is not None:
        seed = get_seed(seed_id)
        if seed is None:
            return {
                "status": "error",
                "error": f"Seed not found for seed_id: {seed_id}",
            }
    else:
        assert seed_content is not None
        if not seed_content.strip():
            return {
                "status": "error",
                "error": "seed_content cannot be empty or whitespace-only",
            }

        seed_result = await parse_seed(seed_content)
        report = generate_validation_report(seed_result)
        rejection = validate_quality_thresholds(report.quality_metrics)
        if not rejection.passed:
            return {
                "status": "error",
                "error": "Seed failed validation; run yolo_seed for details",
            }

        seed = store_seed(content=seed_content, source="text")
        seed_id = seed.seed_id

    thread_id = f"thread-{uuid.uuid4().hex[:8]}"
    sprint = store_sprint(seed_id=seed_id, thread_id=thread_id)
    task = asyncio.create_task(
        _run_sprint(
            sprint_id=sprint.sprint_id,
            thread_id=thread_id,
            seed_id=seed_id,
            seed_content=seed.content,
        )
    )
    _register_sprint_task(sprint.sprint_id, task)

    return {
        "status": "started",
        "sprint_id": sprint.sprint_id,
        "seed_id": seed_id,
        "thread_id": thread_id,
        "started_at": sprint.started_at.isoformat(),
    }


@mcp.tool
async def yolo_status(sprint_id: str) -> dict[str, Any]:
    """Query the status of a sprint by sprint_id."""
    if not sprint_id or not sprint_id.strip():
        return {
            "status": "error",
            "error": "sprint_id must be provided",
        }

    sprint = _get_sprint_snapshot(sprint_id)
    if sprint is None:
        return {
            "status": "error",
            "error": "Sprint not found",
        }

    return {
        "status": sprint.status,
        "sprint_id": sprint.sprint_id,
        "seed_id": sprint.seed_id,
        "thread_id": sprint.thread_id,
        "started_at": sprint.started_at.isoformat(),
        "completed_at": sprint.completed_at.isoformat() if sprint.completed_at else None,
        "error": sprint.error,
    }


@mcp.tool
async def yolo_audit(
    agent: str | None = None,
    decision_type: str | None = None,
    artifact_type: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Access audit trail entries with optional filters and pagination."""
    if not isinstance(limit, int) or not isinstance(offset, int):
        return {
            "status": "error",
            "error": "limit and offset must be integers",
        }
    if limit < 0 or offset < 0:
        return {
            "status": "error",
            "error": "limit and offset must be >= 0",
        }
    if decision_type is not None and decision_type not in VALID_DECISION_TYPES:
        return {
            "status": "error",
            "error": f"decision_type must be one of: {sorted(VALID_DECISION_TYPES)}",
        }
    if artifact_type is not None and artifact_type not in VALID_ARTIFACT_TYPES:
        return {
            "status": "error",
            "error": f"artifact_type must be one of: {sorted(VALID_ARTIFACT_TYPES)}",
        }

    normalized_start, start_dt, error = _parse_iso_timestamp(start_time, "start_time")
    if error:
        return {
            "status": "error",
            "error": error,
        }
    normalized_end, end_dt, error = _parse_iso_timestamp(end_time, "end_time")
    if error:
        return {
            "status": "error",
            "error": error,
        }
    if start_dt and end_dt and start_dt > end_dt:
        return {
            "status": "error",
            "error": "start_time must be before or equal to end_time",
        }

    audit_path = Path(".yolo/audit/decisions.json")
    resolved_path = audit_path.resolve()
    if not audit_path.exists():
        return {
            "status": "error",
            "error": f"Audit store not found: {resolved_path}",
        }
    if not audit_path.is_file():
        return {
            "status": "error",
            "error": f"Audit store path is not a file: {resolved_path}",
        }

    try:
        decision_store = JsonDecisionStore(audit_path)
        traceability_store = InMemoryTraceabilityStore()
        filter_service = get_audit_filter_service(
            decision_store=decision_store,
            traceability_store=traceability_store,
            cost_store=None,
        )

        filters = AuditFilters(
            agent_name=agent,
            decision_type=decision_type,
            artifact_type=artifact_type,
            start_time=normalized_start,
            end_time=normalized_end,
        )

        results = await filter_service.filter_all(filters)
        decisions = results.get("decisions", [])
        if artifact_type is not None:
            decisions = [
                decision
                for decision in decisions
                if decision.metadata.get("artifact_type") == artifact_type
            ]

        total = len(decisions)
        if limit == 0:
            paginated = []
        else:
            paginated = decisions[offset : offset + limit]

        entries = [
            {
                "entry_id": decision.id,
                "timestamp": decision.timestamp,
                "agent": decision.agent.agent_name,
                "decision_type": decision.decision_type,
                "content": decision.content,
                "rationale": decision.rationale,
                "metadata": decision.metadata,
            }
            for decision in paginated
        ]

        return {
            "status": "ok",
            "entries": entries,
            "limit": limit,
            "offset": offset,
            "total": total,
        }
    except Exception as exc:
        logger.exception("mcp_audit_failed", error=str(exc))
        return {
            "status": "error",
            "error": f"Failed to retrieve audit entries: {exc}",
        }


__all__ = [
    "StoredSeed",
    "StoredSprint",
    "clear_seeds",
    "clear_sprints",
    "get_seed",
    "get_sprint",
    "store_seed",
    "store_sprint",
    "yolo_audit",
    "yolo_run",
    "yolo_seed",
    "yolo_status",
]
