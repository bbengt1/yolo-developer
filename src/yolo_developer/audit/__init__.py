"""Audit trail and decision logging module (Story 11.1).

This module provides functionality for logging and tracking agent decisions
throughout the development workflow. It implements FR81: System can log all
agent decisions with rationale.

Components:
    - Type definitions: Decision, AgentIdentity, DecisionContext, etc.
    - Store protocol: DecisionStore for pluggable storage backends
    - In-memory store: InMemoryDecisionStore for testing and single-session use
    - Logger: DecisionLogger for high-level decision logging with auto-generated IDs

Example:
    >>> from yolo_developer.audit import (
    ...     DecisionLogger,
    ...     DecisionContext,
    ...     InMemoryDecisionStore,
    ...     get_logger,
    ... )
    >>>
    >>> # Create store and logger
    >>> store = InMemoryDecisionStore()
    >>> logger = get_logger(store)
    >>>
    >>> # Log a decision
    >>> decision_id = await logger.log(
    ...     agent_name="analyst",
    ...     agent_type="analyst",
    ...     decision_type="requirement_analysis",
    ...     content="OAuth2 authentication required",
    ...     rationale="Industry standard security",
    ...     context=DecisionContext(sprint_id="sprint-1"),
    ... )
    >>>
    >>> # Retrieve decisions
    >>> decisions = await store.get_decisions()

References:
    - FR81: System can log all agent decisions with rationale
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

# Logger
from yolo_developer.audit.logger import (
    DecisionLogger,
    get_logger,
)

# In-memory implementation
from yolo_developer.audit.memory_store import InMemoryDecisionStore

# Store protocol and filters
from yolo_developer.audit.store import (
    DecisionFilters,
    DecisionStore,
)

# Types
from yolo_developer.audit.types import (
    VALID_DECISION_SEVERITIES,
    VALID_DECISION_TYPES,
    AgentIdentity,
    Decision,
    DecisionContext,
    DecisionSeverity,
    DecisionType,
)

__all__ = [
    "VALID_DECISION_SEVERITIES",
    "VALID_DECISION_TYPES",
    "AgentIdentity",
    "Decision",
    "DecisionContext",
    "DecisionFilters",
    "DecisionLogger",
    "DecisionSeverity",
    "DecisionStore",
    "DecisionType",
    "InMemoryDecisionStore",
    "get_logger",
]
