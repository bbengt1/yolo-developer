"""ADR store protocol for auto-generated ADRs (Story 11.8).

This module defines the ADRStore protocol for storing and retrieving
auto-generated Architecture Decision Records.

The protocol follows the same pattern as DecisionStore (Story 11.1) and
CostStore (Story 11.6), enabling multiple storage implementations.

Example:
    >>> from yolo_developer.audit.adr_store import ADRStore
    >>> from yolo_developer.audit.adr_memory_store import InMemoryADRStore
    >>>
    >>> store: ADRStore = InMemoryADRStore()
    >>> adr_id = await store.store_adr(adr)
    >>> retrieved = await store.get_adr(adr_id)

References:
    - FR88: System can generate Architecture Decision Records automatically
    - Story 11.1: DecisionStore protocol pattern
    - Story 11.6: CostStore protocol pattern
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from yolo_developer.audit.adr_types import AutoADR


class ADRStore(Protocol):
    """Protocol for ADR storage implementations.

    Defines the interface for storing and retrieving auto-generated
    Architecture Decision Records. Implementations must support:
    - Storing new ADRs
    - Retrieving ADRs by ID
    - Querying ADRs by story
    - Listing all ADRs
    - Generating sequential ADR numbers

    Example:
        >>> class MyADRStore:
        ...     async def store_adr(self, adr: AutoADR) -> str: ...
        ...     async def get_adr(self, adr_id: str) -> AutoADR | None: ...
        ...     # ... implement all protocol methods
    """

    async def store_adr(self, adr: AutoADR) -> str:
        """Store a new ADR.

        Args:
            adr: The ADR to store.

        Returns:
            The ADR ID.

        Example:
            >>> adr_id = await store.store_adr(adr)
            >>> adr_id
            'ADR-001'
        """
        ...

    async def get_adr(self, adr_id: str) -> AutoADR | None:
        """Retrieve an ADR by ID.

        Args:
            adr_id: The ID of the ADR to retrieve (e.g., "ADR-001").

        Returns:
            The ADR if found, None otherwise.

        Example:
            >>> adr = await store.get_adr("ADR-001")
            >>> adr.title if adr else "Not found"
            'Use PostgreSQL'
        """
        ...

    async def get_adrs_by_story(self, story_id: str) -> list[AutoADR]:
        """Get all ADRs linked to a story.

        Args:
            story_id: The story ID to filter by.

        Returns:
            List of ADRs linked to the story. Empty list if none found.

        Example:
            >>> adrs = await store.get_adrs_by_story("1-2-database-setup")
            >>> len(adrs)
            2
        """
        ...

    async def get_all_adrs(self) -> list[AutoADR]:
        """Get all stored ADRs.

        Returns:
            List of all ADRs, ordered by created_at descending
            (newest first).

        Example:
            >>> all_adrs = await store.get_all_adrs()
            >>> all_adrs[0].id  # Most recent
            'ADR-005'
        """
        ...

    async def get_next_adr_number(self) -> int:
        """Get the next available ADR number for ID generation.

        Used when creating new ADRs to generate sequential IDs
        in the format ADR-{number:03d}.

        Returns:
            The next sequential ADR number (1, 2, 3, ...).

        Example:
            >>> next_num = await store.get_next_adr_number()
            >>> f"ADR-{next_num:03d}"
            'ADR-001'
        """
        ...

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

        Example:
            >>> adrs = await store.get_adrs_by_time_range(
            ...     start_time="2026-01-01T00:00:00Z",
            ...     end_time="2026-01-31T23:59:59Z",
            ... )
        """
        ...
