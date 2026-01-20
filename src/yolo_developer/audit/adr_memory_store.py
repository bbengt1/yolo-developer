"""In-memory implementation of ADRStore (Story 11.8).

This module provides an in-memory implementation of the ADRStore
protocol for testing and single-session use.

The implementation follows the same pattern as InMemoryDecisionStore from
Story 11.1, using threading.Lock for thread-safe concurrent access.

Example:
    >>> from yolo_developer.audit.adr_memory_store import InMemoryADRStore
    >>> from yolo_developer.audit.adr_types import AutoADR
    >>>
    >>> store = InMemoryADRStore()
    >>> adr = AutoADR(
    ...     id="ADR-001",
    ...     title="Use PostgreSQL",
    ...     status="proposed",
    ...     context="Database needed.",
    ...     decision="Selected PostgreSQL.",
    ...     consequences="ACID compliance.",
    ...     source_decision_id="dec-123",
    ...     story_ids=("1-2-database",),
    ... )
    >>> await store.store_adr(adr)
    'ADR-001'

References:
    - FR88: System can generate Architecture Decision Records automatically
    - Story 11.1: InMemoryDecisionStore implementation pattern
    - Story 11.6: InMemoryCostStore implementation pattern

"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from yolo_developer.audit.adr_types import AutoADR

_logger = structlog.get_logger(__name__)


class InMemoryADRStore:
    """In-memory implementation of ADRStore protocol.

    Stores ADRs in memory with thread-safe access.
    Maintains indices for fast lookup by story_id.

    Attributes:
        _adrs: Dictionary mapping ADR IDs to ADRs.
        _story_index: Index mapping story_id to list of ADR IDs.
        _adr_counter: Counter for generating sequential ADR numbers.
        _lock: Thread lock for concurrent access safety.

    Example:
        >>> store = InMemoryADRStore()
        >>> await store.store_adr(adr)
        >>> await store.get_adr("ADR-001")

    """

    def __init__(self) -> None:
        """Initialize the in-memory ADR store."""
        self._adrs: dict[str, AutoADR] = {}
        self._story_index: dict[str, list[str]] = defaultdict(list)
        self._adr_counter: int = 0
        self._lock = threading.Lock()
        _logger.debug("in_memory_adr_store_initialized")

    async def store_adr(self, adr: AutoADR) -> str:
        """Store a new ADR.

        Args:
            adr: The ADR to store.

        Returns:
            The ADR ID.

        """
        with self._lock:
            self._adrs[adr.id] = adr
            # Update story index
            for story_id in adr.story_ids:
                self._story_index[story_id].append(adr.id)

        _logger.debug(
            "adr_stored",
            adr_id=adr.id,
            story_ids=adr.story_ids,
        )
        return adr.id

    async def get_adr(self, adr_id: str) -> AutoADR | None:
        """Retrieve an ADR by ID.

        Args:
            adr_id: The ID of the ADR to retrieve.

        Returns:
            The ADR if found, None otherwise.

        """
        with self._lock:
            result = self._adrs.get(adr_id)

        _logger.debug(
            "adr_retrieved",
            adr_id=adr_id,
            found=result is not None,
        )
        return result

    async def get_adrs_by_story(self, story_id: str) -> list[AutoADR]:
        """Get all ADRs linked to a story.

        Args:
            story_id: The story ID to filter by.

        Returns:
            List of ADRs linked to the story.

        """
        with self._lock:
            adr_ids = self._story_index.get(story_id, [])
            result = [self._adrs[adr_id] for adr_id in adr_ids if adr_id in self._adrs]

        _logger.debug(
            "adrs_retrieved_by_story",
            story_id=story_id,
            count=len(result),
        )
        return result

    async def get_all_adrs(self) -> list[AutoADR]:
        """Get all stored ADRs, ordered by created_at descending.

        Returns:
            List of all ADRs, ordered by created_at descending (newest first).

        """
        with self._lock:
            adrs = list(self._adrs.values())
            result = sorted(adrs, key=lambda a: a.created_at, reverse=True)

        _logger.debug("all_adrs_retrieved", count=len(result))
        return result

    async def get_next_adr_number(self) -> int:
        """Get the next available ADR number.

        Returns:
            The next sequential ADR number.

        """
        with self._lock:
            self._adr_counter += 1
            result = self._adr_counter

        _logger.debug("next_adr_number", number=result)
        return result

    async def get_adrs_by_time_range(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[AutoADR]:
        """Get ADRs within a time range.

        Filters ADRs by their created_at timestamp, returning those
        that fall within the specified range (inclusive).

        Args:
            start_time: ISO 8601 timestamp for range start (inclusive).
                If None, no lower bound is applied.
            end_time: ISO 8601 timestamp for range end (inclusive).
                If None, no upper bound is applied.

        Returns:
            List of ADRs within the time range, ordered by created_at
            descending (newest first).

        """
        with self._lock:
            adrs = list(self._adrs.values())

            # Filter by time range
            if start_time is not None:
                adrs = [a for a in adrs if a.created_at >= start_time]
            if end_time is not None:
                adrs = [a for a in adrs if a.created_at <= end_time]

            result = sorted(adrs, key=lambda a: a.created_at, reverse=True)

        _logger.debug(
            "adrs_retrieved_by_time_range",
            start_time=start_time,
            end_time=end_time,
            count=len(result),
        )
        return result
