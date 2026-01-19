"""JSON file-based decision store implementation (Story 13.3).

This module provides a JSON file-based implementation of the DecisionStore protocol.
It persists decisions to a JSON file for cross-session access.

The implementation uses thread-safe file operations with file locking for concurrent access.

Example:
    >>> from yolo_developer.audit.json_decision_store import JsonDecisionStore
    >>> from pathlib import Path
    >>>
    >>> store = JsonDecisionStore(Path(".yolo/audit/decisions.json"))
    >>> decision_id = await store.log_decision(decision)
    >>> # Decision is now persisted to disk

References:
    - FR81: System can log all agent decisions with rationale
    - FR108: Python SDK for programmatic access
    - Story 13.3: Audit Trail Access - Persistent Storage
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from yolo_developer.audit.types import (
    AgentIdentity,
    Decision,
    DecisionContext,
)

if TYPE_CHECKING:
    from yolo_developer.audit.store import DecisionFilters


class JsonDecisionStore:
    """JSON file-based implementation of the DecisionStore protocol.

    Stores decisions in a JSON file for persistence across sessions.
    Thread-safe with file locking for concurrent access.

    This implementation is suitable for:
    - Single-user development workflows
    - Small to medium audit trails
    - Projects without database requirements

    For high-concurrency or large-scale use, consider a database implementation.

    Attributes:
        _file_path: Path to the JSON file for storage.
        _lock: Threading lock for concurrent access safety.

    Example:
        >>> store = JsonDecisionStore(Path(".yolo/audit/decisions.json"))
        >>> await store.log_decision(decision)
        >>> decisions = await store.get_decisions()
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize the JSON decision store.

        Args:
            file_path: Path to the JSON file for storing decisions.
                       Parent directories will be created if they don't exist.
        """
        self._file_path = file_path
        self._lock = threading.Lock()
        # Ensure parent directory exists
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    async def log_decision(self, decision: Decision) -> str:
        """Store a decision and return its ID.

        Thread-safe operation that appends the decision to the JSON file.

        Args:
            decision: The Decision to store.

        Returns:
            The decision ID.
        """
        with self._lock:
            decisions = self._load_decisions()
            decisions[decision.id] = decision.to_dict()
            self._save_decisions(decisions)
        return decision.id

    async def get_decision(self, decision_id: str) -> Decision | None:
        """Retrieve a decision by its ID.

        Thread-safe operation that looks up a decision from the JSON file.

        Args:
            decision_id: The ID of the decision to retrieve.

        Returns:
            The Decision if found, None otherwise.
        """
        with self._lock:
            decisions = self._load_decisions()
            decision_dict = decisions.get(decision_id)
            if decision_dict is None:
                return None
            return self._dict_to_decision(decision_dict)

    async def get_decisions(
        self,
        filters: DecisionFilters | None = None,
    ) -> list[Decision]:
        """Query decisions with optional filters.

        Returns decisions in chronological order (oldest first).
        Applies filters if provided.

        Args:
            filters: Optional filters to apply. None returns all decisions.

        Returns:
            List of matching Decision objects, ordered by timestamp.
        """
        with self._lock:
            decisions_dict = self._load_decisions()
            decisions = [
                self._dict_to_decision(d) for d in decisions_dict.values()
            ]

        # Apply filters if provided
        if filters is not None:
            decisions = self._apply_filters(decisions, filters)

        # Sort by timestamp (chronological order)
        decisions.sort(key=lambda d: d.timestamp)

        return decisions

    async def get_decision_count(self) -> int:
        """Get the total number of stored decisions.

        Returns:
            Count of all stored decisions.
        """
        with self._lock:
            decisions = self._load_decisions()
            return len(decisions)

    def _load_decisions(self) -> dict[str, Any]:
        """Load decisions from the JSON file.

        Returns:
            Dictionary mapping decision IDs to decision dictionaries.
            Returns empty dict if file doesn't exist or is invalid.
        """
        if not self._file_path.exists():
            return {}
        try:
            content = self._file_path.read_text(encoding="utf-8")
            if not content.strip():
                return {}
            data = json.loads(content)
            # Handle both list format (legacy) and dict format
            if isinstance(data, list):
                return {d["id"]: d for d in data}
            if isinstance(data, dict):
                return data
            return {}
        except (json.JSONDecodeError, KeyError):
            return {}

    def _save_decisions(self, decisions: dict[str, Any]) -> None:
        """Save decisions to the JSON file.

        Args:
            decisions: Dictionary mapping decision IDs to decision dictionaries.
        """
        self._file_path.write_text(
            json.dumps(decisions, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _dict_to_decision(self, d: dict[str, Any]) -> Decision:
        """Convert a dictionary to a Decision object.

        Args:
            d: Dictionary representation of a decision.

        Returns:
            Decision object reconstructed from the dictionary.
        """
        agent_dict = d.get("agent", {})
        context_dict = d.get("context", {})

        agent = AgentIdentity(
            agent_name=agent_dict.get("agent_name", ""),
            agent_type=agent_dict.get("agent_type", ""),
            session_id=agent_dict.get("session_id", ""),
        )

        context = DecisionContext(
            sprint_id=context_dict.get("sprint_id"),
            story_id=context_dict.get("story_id"),
            artifact_id=context_dict.get("artifact_id"),
            parent_decision_id=context_dict.get("parent_decision_id"),
            trace_links=tuple(context_dict.get("trace_links", [])),
        )

        return Decision(
            id=d.get("id", ""),
            decision_type=d.get("decision_type", "requirement_analysis"),
            content=d.get("content", ""),
            rationale=d.get("rationale", ""),
            agent=agent,
            context=context,
            timestamp=d.get("timestamp", ""),
            metadata=d.get("metadata", {}),
            severity=d.get("severity", "info"),
        )

    def _apply_filters(
        self,
        decisions: list[Decision],
        filters: DecisionFilters,
    ) -> list[Decision]:
        """Apply filters to a list of decisions.

        Args:
            decisions: List of decisions to filter.
            filters: Filters to apply.

        Returns:
            Filtered list of decisions.
        """
        result = decisions

        # Filter by agent_name
        if filters.agent_name is not None:
            result = [d for d in result if d.agent.agent_name == filters.agent_name]

        # Filter by decision_type
        if filters.decision_type is not None:
            result = [d for d in result if d.decision_type == filters.decision_type]

        # Filter by start_time (inclusive)
        if filters.start_time is not None:
            result = [d for d in result if d.timestamp >= filters.start_time]

        # Filter by end_time (inclusive)
        if filters.end_time is not None:
            result = [d for d in result if d.timestamp <= filters.end_time]

        # Filter by sprint_id
        if filters.sprint_id is not None:
            result = [d for d in result if d.context.sprint_id == filters.sprint_id]

        # Filter by story_id
        if filters.story_id is not None:
            result = [d for d in result if d.context.story_id == filters.story_id]

        return result
