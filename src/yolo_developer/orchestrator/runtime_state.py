"""Runtime state tracking for live dashboard updates.

This module provides real-time state tracking that can be queried by the
web dashboard during workflow execution. It persists state to a JSON file
that gets updated as agents run.

Key Concepts:
    - **RuntimeState**: Current execution state (active agent, gate results, progress)
    - **RuntimeStateManager**: Singleton that updates state during workflow execution
    - **Thread-safe**: Uses atomic file writes for safe concurrent access

Example:
    >>> from yolo_developer.orchestrator.runtime_state import (
    ...     get_runtime_state_manager,
    ...     RuntimeState,
    ... )
    >>>
    >>> # Update state when agent starts
    >>> manager = get_runtime_state_manager()
    >>> manager.agent_started("analyst")
    >>>
    >>> # Read current state
    >>> state = manager.get_state()
    >>> print(f"Active agent: {state.active_agent}")
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with timezone info."""
    return datetime.now(timezone.utc)


@dataclass
class GateResult:
    """Result of a quality gate evaluation.

    Attributes:
        name: Gate name (e.g., "Testability", "DoD").
        score: Gate score (0.0-1.0).
        passed: Whether the gate passed.
        reason: Optional reason/details for the result.
        evaluated_at: When the gate was evaluated.
    """

    name: str
    score: float
    passed: bool
    reason: str = ""
    evaluated_at: str = field(default_factory=lambda: _utcnow().isoformat())


@dataclass
class AgentState:
    """State of an individual agent.

    Attributes:
        name: Agent name (e.g., "analyst", "dev").
        state: Current state ("idle", "active", "completed", "error").
        started_at: When the agent started (if active/completed).
        completed_at: When the agent completed (if completed).
    """

    name: str
    state: str = "idle"
    started_at: str | None = None
    completed_at: str | None = None


@dataclass
class StoryProgress:
    """Progress of story implementation.

    Attributes:
        story_id: Story identifier.
        status: Current status ("pending", "in_progress", "completed", "failed").
        started_at: When implementation started.
        completed_at: When implementation completed.
    """

    story_id: str
    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None


@dataclass
class RuntimeState:
    """Current runtime state of workflow execution.

    Attributes:
        workflow_status: Overall status ("idle", "running", "completed", "error").
        active_agent: Currently executing agent (None if idle).
        thread_id: Current workflow thread ID.
        agents: State of each agent.
        gates: Results of quality gates.
        stories: Progress of stories.
        stories_completed: Count of completed stories.
        stories_total: Total story count.
        last_updated: When state was last updated.
        started_at: When workflow started.
        eta_minutes: Estimated time remaining (approximate).
    """

    workflow_status: str = "idle"
    active_agent: str | None = None
    thread_id: str | None = None
    agents: list[AgentState] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    stories: list[StoryProgress] = field(default_factory=list)
    stories_completed: int = 0
    stories_total: int = 0
    last_updated: str = field(default_factory=lambda: _utcnow().isoformat())
    started_at: str | None = None
    eta_minutes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflow_status": self.workflow_status,
            "active_agent": self.active_agent,
            "thread_id": self.thread_id,
            "agents": [asdict(a) for a in self.agents],
            "gates": [asdict(g) for g in self.gates],
            "stories": [asdict(s) for s in self.stories],
            "stories_completed": self.stories_completed,
            "stories_total": self.stories_total,
            "last_updated": self.last_updated,
            "started_at": self.started_at,
            "eta_minutes": self.eta_minutes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeState:
        """Create RuntimeState from dictionary."""
        return cls(
            workflow_status=data.get("workflow_status", "idle"),
            active_agent=data.get("active_agent"),
            thread_id=data.get("thread_id"),
            agents=[
                AgentState(**a) for a in data.get("agents", [])
            ],
            gates=[
                GateResult(**g) for g in data.get("gates", [])
            ],
            stories=[
                StoryProgress(**s) for s in data.get("stories", [])
            ],
            stories_completed=data.get("stories_completed", 0),
            stories_total=data.get("stories_total", 0),
            last_updated=data.get("last_updated", _utcnow().isoformat()),
            started_at=data.get("started_at"),
            eta_minutes=data.get("eta_minutes", 0),
        )


class RuntimeStateManager:
    """Manages runtime state for the workflow dashboard.

    This class provides thread-safe updates to a JSON state file that
    can be read by the web dashboard for live updates.

    Attributes:
        state_file: Path to the runtime state JSON file.

    Example:
        >>> manager = RuntimeStateManager()
        >>> manager.workflow_started("thread-123")
        >>> manager.agent_started("analyst")
        >>> state = manager.get_state()
    """

    _instance: RuntimeStateManager | None = None
    _lock = threading.Lock()

    def __new__(cls, state_dir: str | Path | None = None) -> RuntimeStateManager:
        """Singleton pattern for global state management."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, state_dir: str | Path | None = None) -> None:
        """Initialize the runtime state manager.

        Args:
            state_dir: Directory for state file. Defaults to .yolo/
        """
        if getattr(self, "_initialized", False):
            return

        if state_dir is None:
            state_dir = Path(".yolo")

        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / "runtime_state.json"
        self._file_lock = threading.Lock()
        self._initialized = True

        # Initialize with default agents
        self._default_agents = ["analyst", "pm", "architect", "dev", "tea", "sm"]

    def _write_state(self, state: RuntimeState) -> None:
        """Write state to file atomically."""
        state.last_updated = _utcnow().isoformat()
        temp_path = self._state_file.with_suffix(".tmp")

        with self._file_lock:
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(state.to_dict(), f, indent=2)
                temp_path.rename(self._state_file)
            except Exception as e:
                logger.warning("runtime_state_write_failed", error=str(e))
                if temp_path.exists():
                    temp_path.unlink()

    def _read_state(self) -> RuntimeState:
        """Read state from file or return default."""
        with self._file_lock:
            if not self._state_file.exists():
                return self._create_default_state()

            try:
                with open(self._state_file, encoding="utf-8") as f:
                    data = json.load(f)
                return RuntimeState.from_dict(data)
            except Exception as e:
                logger.warning("runtime_state_read_failed", error=str(e))
                return self._create_default_state()

    def _create_default_state(self) -> RuntimeState:
        """Create default state with all agents idle."""
        return RuntimeState(
            workflow_status="idle",
            agents=[AgentState(name=a) for a in self._default_agents],
            gates=[],
            stories=[],
        )

    def get_state(self) -> RuntimeState:
        """Get the current runtime state.

        Returns:
            Current RuntimeState.
        """
        return self._read_state()

    def reset(self) -> None:
        """Reset state to default (all idle)."""
        state = self._create_default_state()
        self._write_state(state)
        logger.debug("runtime_state_reset")

    def workflow_started(self, thread_id: str) -> None:
        """Mark workflow as started.

        Args:
            thread_id: The workflow thread ID.
        """
        state = self._read_state()
        state.workflow_status = "running"
        state.thread_id = thread_id
        state.started_at = _utcnow().isoformat()
        state.active_agent = None

        # Reset all agents to idle
        state.agents = [AgentState(name=a) for a in self._default_agents]

        self._write_state(state)
        logger.info("runtime_state_workflow_started", thread_id=thread_id)

    def workflow_completed(self) -> None:
        """Mark workflow as completed."""
        state = self._read_state()
        state.workflow_status = "completed"
        state.active_agent = None
        self._write_state(state)
        logger.info("runtime_state_workflow_completed")

    def workflow_error(self, error: str) -> None:
        """Mark workflow as errored.

        Args:
            error: Error message.
        """
        state = self._read_state()
        state.workflow_status = "error"
        state.active_agent = None
        self._write_state(state)
        logger.error("runtime_state_workflow_error", error=error)

    def agent_started(self, agent_name: str) -> None:
        """Mark an agent as started/active.

        Args:
            agent_name: Name of the agent starting.
        """
        state = self._read_state()
        state.active_agent = agent_name

        # Update agent state
        for agent in state.agents:
            if agent.name == agent_name:
                agent.state = "active"
                agent.started_at = _utcnow().isoformat()
            elif agent.state == "active":
                # Previous active agent is now waiting
                agent.state = "waiting"

        self._write_state(state)
        logger.debug("runtime_state_agent_started", agent=agent_name)

    def agent_completed(self, agent_name: str) -> None:
        """Mark an agent as completed.

        Args:
            agent_name: Name of the agent completing.
        """
        state = self._read_state()

        if state.active_agent == agent_name:
            state.active_agent = None

        for agent in state.agents:
            if agent.name == agent_name:
                agent.state = "completed"
                agent.completed_at = _utcnow().isoformat()

        self._write_state(state)
        logger.debug("runtime_state_agent_completed", agent=agent_name)

    def gate_evaluated(
        self,
        gate_name: str,
        score: float,
        passed: bool,
        reason: str = "",
    ) -> None:
        """Record a gate evaluation result.

        Args:
            gate_name: Name of the gate (e.g., "Testability").
            score: Gate score (0.0-1.0).
            passed: Whether the gate passed.
            reason: Optional reason/details.
        """
        state = self._read_state()

        # Update existing gate or add new
        gate_found = False
        for gate in state.gates:
            if gate.name == gate_name:
                gate.score = score
                gate.passed = passed
                gate.reason = reason
                gate.evaluated_at = _utcnow().isoformat()
                gate_found = True
                break

        if not gate_found:
            state.gates.append(
                GateResult(
                    name=gate_name,
                    score=score,
                    passed=passed,
                    reason=reason,
                )
            )

        self._write_state(state)
        logger.debug(
            "runtime_state_gate_evaluated",
            gate=gate_name,
            score=score,
            passed=passed,
        )

    def story_started(self, story_id: str) -> None:
        """Mark a story as started.

        Args:
            story_id: Story identifier.
        """
        state = self._read_state()

        # Update existing or add new
        story_found = False
        for story in state.stories:
            if story.story_id == story_id:
                story.status = "in_progress"
                story.started_at = _utcnow().isoformat()
                story_found = True
                break

        if not story_found:
            state.stories.append(
                StoryProgress(
                    story_id=story_id,
                    status="in_progress",
                    started_at=_utcnow().isoformat(),
                )
            )
            state.stories_total = len(state.stories)

        self._write_state(state)
        logger.debug("runtime_state_story_started", story_id=story_id)

    def story_completed(self, story_id: str, status: str = "completed") -> None:
        """Mark a story as completed.

        Args:
            story_id: Story identifier.
            status: Final status ("completed" or "failed").
        """
        state = self._read_state()

        for story in state.stories:
            if story.story_id == story_id:
                story.status = status
                story.completed_at = _utcnow().isoformat()
                break

        # Update completed count
        state.stories_completed = sum(
            1 for s in state.stories if s.status == "completed"
        )

        self._write_state(state)
        logger.debug(
            "runtime_state_story_completed",
            story_id=story_id,
            status=status,
        )

    def update_progress(
        self,
        stories_completed: int | None = None,
        stories_total: int | None = None,
        eta_minutes: int | None = None,
    ) -> None:
        """Update progress metrics.

        Args:
            stories_completed: Number of completed stories.
            stories_total: Total number of stories.
            eta_minutes: Estimated time remaining.
        """
        state = self._read_state()

        if stories_completed is not None:
            state.stories_completed = stories_completed
        if stories_total is not None:
            state.stories_total = stories_total
        if eta_minutes is not None:
            state.eta_minutes = eta_minutes

        self._write_state(state)


# Global singleton accessor
_manager: RuntimeStateManager | None = None


def get_runtime_state_manager(state_dir: str | Path | None = None) -> RuntimeStateManager:
    """Get the global runtime state manager instance.

    Args:
        state_dir: Optional directory for state file.

    Returns:
        The singleton RuntimeStateManager instance.
    """
    global _manager
    if _manager is None:
        _manager = RuntimeStateManager(state_dir)
    return _manager


def reset_runtime_state_manager() -> None:
    """Reset the global manager instance (for testing)."""
    global _manager
    _manager = None
    RuntimeStateManager._instance = None
