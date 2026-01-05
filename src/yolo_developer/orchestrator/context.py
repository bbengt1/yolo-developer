"""Context preservation for agent handoffs in YOLO Developer.

This module provides dataclasses and functions for preserving context when
agents hand off work to each other during orchestration. Key concepts:

- **Decision**: Captures a significant decision made by an agent, including
  the agent identifier, summary, rationale, timestamp, and related artifacts.
- **HandoffContext**: Contains all context needed for the receiving agent,
  including decisions made and memory store references.
- **State Integrity**: Functions to compute checksums and validate that
  state is preserved correctly during handoffs.

Example:
    >>> from yolo_developer.orchestrator.context import (
    ...     Decision,
    ...     HandoffContext,
    ...     create_handoff_context,
    ... )
    >>>
    >>> # Create a decision
    >>> decision = Decision(
    ...     agent="analyst",
    ...     summary="Prioritized security over performance",
    ...     rationale="User explicitly requested secure design",
    ... )
    >>>
    >>> # Create handoff context
    >>> state_update = create_handoff_context(
    ...     source_agent="analyst",
    ...     target_agent="pm",
    ...     decisions=[decision],
    ...     memory_refs=["req-001"],
    ... )

Security Note:
    State checksums use SHA-256 for integrity validation. This is sufficient
    for detecting accidental corruption but is not cryptographically secure
    against intentional tampering.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> datetime:
    """Return current UTC datetime with timezone info.

    Using timezone-aware datetime for consistency and to avoid
    deprecation warnings from naive datetime comparisons.
    """
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Decision:
    """A significant decision made by an agent during processing.

    Captures the context of important decisions for audit trail and
    context preservation across agent handoffs. Decisions are immutable
    to ensure integrity throughout the processing pipeline.

    Attributes:
        agent: The agent that made the decision (e.g., "analyst", "pm").
        summary: Brief description of the decision.
        rationale: Why this decision was made.
        timestamp: When the decision was made. Defaults to current UTC time.
        related_artifacts: Keys of related embeddings/relationships in memory.

    Example:
        >>> decision = Decision(
        ...     agent="analyst",
        ...     summary="Prioritized security requirements",
        ...     rationale="User explicitly requested secure design first",
        ...     related_artifacts=("req-001", "req-002"),
        ... )
        >>> decision.agent
        'analyst'
    """

    agent: str
    summary: str
    rationale: str
    timestamp: datetime = field(default_factory=_utcnow)
    related_artifacts: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class HandoffContext:
    """Context passed during agent handoffs.

    Contains decisions made by the source agent and references to relevant
    memory store entries. This enables the target agent to understand prior
    decisions and access relevant context without re-querying everything.

    Attributes:
        source_agent: Agent handing off work (e.g., "analyst").
        target_agent: Agent receiving work (e.g., "pm").
        decisions: Key decisions from source agent processing.
        memory_refs: Keys to relevant embeddings in memory store.
        timestamp: When the handoff occurred. Defaults to current UTC time.

    Example:
        >>> decision = Decision(agent="analyst", summary="test", rationale="test")
        >>> context = HandoffContext(
        ...     source_agent="analyst",
        ...     target_agent="pm",
        ...     decisions=(decision,),
        ...     memory_refs=("req-001", "req-002"),
        ... )
        >>> context.target_agent
        'pm'
    """

    source_agent: str
    target_agent: str
    decisions: tuple[Decision, ...] = field(default_factory=tuple)
    memory_refs: tuple[str, ...] = field(default_factory=tuple)
    timestamp: datetime = field(default_factory=_utcnow)


def create_handoff_context(
    source_agent: str,
    target_agent: str,
    decisions: list[Decision] | None = None,
    memory_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Create a handoff context and return state update dict.

    This function creates a HandoffContext object and returns a dictionary
    suitable for updating LangGraph state. The returned dict includes both
    the handoff_context and updates current_agent to the target.

    Args:
        source_agent: Agent handing off work.
        target_agent: Agent receiving work.
        decisions: Decisions made during processing. Defaults to empty.
        memory_refs: Keys to relevant memory entries. Defaults to empty.

    Returns:
        State update dict with:
        - handoff_context: The created HandoffContext
        - current_agent: Set to target_agent

    Example:
        >>> result = create_handoff_context(
        ...     source_agent="analyst",
        ...     target_agent="pm",
        ...     decisions=[Decision(agent="analyst", summary="s", rationale="r")],
        ...     memory_refs=["req-001"],
        ... )
        >>> result["current_agent"]
        'pm'
        >>> result["handoff_context"].source_agent
        'analyst'
    """
    context = HandoffContext(
        source_agent=source_agent,
        target_agent=target_agent,
        decisions=tuple(decisions) if decisions else (),
        memory_refs=tuple(memory_refs) if memory_refs else (),
    )

    return {
        "handoff_context": context,
        "current_agent": target_agent,
    }


# Default keys to exclude from state integrity validation.
# These are expected to change during handoffs:
# - current_agent: Updated to target agent
# - handoff_context: Created fresh for each handoff
# - messages: Accumulated via reducer
_DEFAULT_EXCLUDE_KEYS = frozenset({"current_agent", "handoff_context", "messages"})

# Minimal exclude keys for compute_state_checksum when used standalone
# Only handoff_context is excluded by default since it always changes
_CHECKSUM_EXCLUDE_KEYS = frozenset({"handoff_context"})


def compute_state_checksum(
    state: dict[str, Any],
    exclude_keys: AbstractSet[str] | None = None,
) -> str:
    """Compute a SHA-256 checksum of state for integrity validation.

    Creates a deterministic hash of the state dictionary, excluding specified
    keys. Useful for verifying that critical state fields are preserved
    during agent handoffs.

    Args:
        state: The state dict to checksum.
        exclude_keys: Keys to exclude from checksum. Defaults to
            {"handoff_context"} (via _CHECKSUM_EXCLUDE_KEYS).
            For handoff validation, use validate_state_integrity() instead,
            which excludes additional transient keys.

    Returns:
        64-character lowercase hex string (SHA-256 hash).

    Example:
        >>> state = {"data": "value", "handoff_context": "ctx"}
        >>> checksum = compute_state_checksum(state)
        >>> len(checksum)
        64

    See Also:
        validate_state_integrity: For handoff-specific validation with
            broader default exclusions.
    """
    # Use minimal default - only handoff_context which always changes
    exclude: AbstractSet[str] = exclude_keys if exclude_keys is not None else _CHECKSUM_EXCLUDE_KEYS

    # Filter and sort for deterministic serialization
    filtered = {k: v for k, v in sorted(state.items()) if k not in exclude}

    # Serialize with custom encoder for non-JSON types (datetime, etc.)
    serialized = json.dumps(filtered, default=str, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def validate_state_integrity(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    exclude_keys: AbstractSet[str] | None = None,
) -> bool:
    """Validate that state integrity was preserved during handoff.

    Compares checksums of before and after states, excluding keys that
    are expected to change during handoffs. This is the recommended function
    for handoff validation as it excludes all transient keys by default.

    Args:
        before_state: State before handoff.
        after_state: State after handoff.
        exclude_keys: Keys to exclude from comparison. Defaults to
            {"current_agent", "handoff_context", "messages"} via
            _DEFAULT_EXCLUDE_KEYS.

    Returns:
        True if integrity preserved (checksums match), False otherwise.

    Example:
        >>> before = {"data": "value", "current_agent": "analyst"}
        >>> after = {"data": "value", "current_agent": "pm"}
        >>> validate_state_integrity(before, after)
        True

    See Also:
        compute_state_checksum: Lower-level function with minimal defaults.
    """
    # Use default exclude keys if not specified
    exclude = exclude_keys if exclude_keys is not None else _DEFAULT_EXCLUDE_KEYS

    before_checksum = compute_state_checksum(before_state, exclude)
    after_checksum = compute_state_checksum(after_state, exclude)

    return before_checksum == after_checksum
