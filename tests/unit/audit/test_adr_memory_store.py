"""Tests for InMemoryADRStore (Story 11.8).

Tests for the in-memory ADR store implementation including
storage, retrieval, querying, and thread safety.
"""

from __future__ import annotations

import asyncio

import pytest

from yolo_developer.audit.adr_memory_store import InMemoryADRStore
from yolo_developer.audit.adr_types import AutoADR


def _make_adr(
    adr_id: str = "ADR-001",
    title: str = "Test ADR",
    story_ids: tuple[str, ...] = (),
    created_at: str = "2026-01-15T10:00:00+00:00",
) -> AutoADR:
    """Create a test ADR with default values."""
    return AutoADR(
        id=adr_id,
        title=title,
        status="proposed",
        context="Test context",
        decision="Test decision",
        consequences="Test consequences",
        source_decision_id=f"dec-{adr_id}",
        story_ids=story_ids,
        created_at=created_at,
    )


class TestInMemoryADRStoreBasicOperations:
    """Tests for basic store operations."""

    @pytest.mark.asyncio
    async def test_store_adr_returns_id(self) -> None:
        """Test that store_adr returns the ADR ID."""
        store = InMemoryADRStore()
        adr = _make_adr("ADR-001")

        result = await store.store_adr(adr)

        assert result == "ADR-001"

    @pytest.mark.asyncio
    async def test_get_adr_retrieves_stored_adr(self) -> None:
        """Test that get_adr retrieves a previously stored ADR."""
        store = InMemoryADRStore()
        adr = _make_adr("ADR-001", "My ADR Title")
        await store.store_adr(adr)

        result = await store.get_adr("ADR-001")

        assert result is not None
        assert result.id == "ADR-001"
        assert result.title == "My ADR Title"

    @pytest.mark.asyncio
    async def test_get_adr_returns_none_for_missing(self) -> None:
        """Test that get_adr returns None for non-existent ADR."""
        store = InMemoryADRStore()

        result = await store.get_adr("ADR-999")

        assert result is None

    @pytest.mark.asyncio
    async def test_store_multiple_adrs(self) -> None:
        """Test storing and retrieving multiple ADRs."""
        store = InMemoryADRStore()
        adr1 = _make_adr("ADR-001", "First ADR")
        adr2 = _make_adr("ADR-002", "Second ADR")

        await store.store_adr(adr1)
        await store.store_adr(adr2)

        result1 = await store.get_adr("ADR-001")
        result2 = await store.get_adr("ADR-002")

        assert result1 is not None
        assert result1.title == "First ADR"
        assert result2 is not None
        assert result2.title == "Second ADR"


class TestInMemoryADRStoreStoryIndex:
    """Tests for story-based ADR queries."""

    @pytest.mark.asyncio
    async def test_get_adrs_by_story_single_match(self) -> None:
        """Test getting ADRs linked to a single story."""
        store = InMemoryADRStore()
        adr = _make_adr("ADR-001", story_ids=("1-2-database",))
        await store.store_adr(adr)

        result = await store.get_adrs_by_story("1-2-database")

        assert len(result) == 1
        assert result[0].id == "ADR-001"

    @pytest.mark.asyncio
    async def test_get_adrs_by_story_multiple_matches(self) -> None:
        """Test getting multiple ADRs linked to the same story."""
        store = InMemoryADRStore()
        adr1 = _make_adr("ADR-001", story_ids=("1-2-database",))
        adr2 = _make_adr("ADR-002", story_ids=("1-2-database",))
        await store.store_adr(adr1)
        await store.store_adr(adr2)

        result = await store.get_adrs_by_story("1-2-database")

        assert len(result) == 2
        adr_ids = {adr.id for adr in result}
        assert adr_ids == {"ADR-001", "ADR-002"}

    @pytest.mark.asyncio
    async def test_get_adrs_by_story_no_matches(self) -> None:
        """Test getting ADRs for a story with no linked ADRs."""
        store = InMemoryADRStore()
        adr = _make_adr("ADR-001", story_ids=("1-2-database",))
        await store.store_adr(adr)

        result = await store.get_adrs_by_story("1-3-other")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_adrs_by_story_adr_with_multiple_stories(self) -> None:
        """Test ADR linked to multiple stories appears in queries for each."""
        store = InMemoryADRStore()
        adr = _make_adr("ADR-001", story_ids=("story-a", "story-b"))
        await store.store_adr(adr)

        result_a = await store.get_adrs_by_story("story-a")
        result_b = await store.get_adrs_by_story("story-b")

        assert len(result_a) == 1
        assert result_a[0].id == "ADR-001"
        assert len(result_b) == 1
        assert result_b[0].id == "ADR-001"


class TestInMemoryADRStoreGetAllAdrs:
    """Tests for get_all_adrs method."""

    @pytest.mark.asyncio
    async def test_get_all_adrs_empty_store(self) -> None:
        """Test get_all_adrs on empty store."""
        store = InMemoryADRStore()

        result = await store.get_all_adrs()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_adrs_returns_all(self) -> None:
        """Test get_all_adrs returns all stored ADRs."""
        store = InMemoryADRStore()
        for i in range(5):
            adr = _make_adr(f"ADR-{i:03d}")
            await store.store_adr(adr)

        result = await store.get_all_adrs()

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_get_all_adrs_ordered_by_created_at_descending(self) -> None:
        """Test get_all_adrs returns ADRs ordered by created_at descending."""
        store = InMemoryADRStore()
        # Store in ascending order
        adr1 = _make_adr("ADR-001", created_at="2026-01-01T00:00:00+00:00")
        adr2 = _make_adr("ADR-002", created_at="2026-01-02T00:00:00+00:00")
        adr3 = _make_adr("ADR-003", created_at="2026-01-03T00:00:00+00:00")
        await store.store_adr(adr1)
        await store.store_adr(adr2)
        await store.store_adr(adr3)

        result = await store.get_all_adrs()

        # Should be in descending order (newest first)
        assert result[0].id == "ADR-003"
        assert result[1].id == "ADR-002"
        assert result[2].id == "ADR-001"


class TestInMemoryADRStoreNextNumber:
    """Tests for get_next_adr_number method."""

    @pytest.mark.asyncio
    async def test_get_next_adr_number_starts_at_one(self) -> None:
        """Test that first call returns 1."""
        store = InMemoryADRStore()

        result = await store.get_next_adr_number()

        assert result == 1

    @pytest.mark.asyncio
    async def test_get_next_adr_number_increments(self) -> None:
        """Test that subsequent calls increment the number."""
        store = InMemoryADRStore()

        n1 = await store.get_next_adr_number()
        n2 = await store.get_next_adr_number()
        n3 = await store.get_next_adr_number()

        assert n1 == 1
        assert n2 == 2
        assert n3 == 3

    @pytest.mark.asyncio
    async def test_get_next_adr_number_independent_of_stored_adrs(self) -> None:
        """Test that number increments even without storing ADRs."""
        store = InMemoryADRStore()

        # Get numbers without storing
        await store.get_next_adr_number()
        await store.get_next_adr_number()
        result = await store.get_next_adr_number()

        assert result == 3


class TestInMemoryADRStoreThreadSafety:
    """Tests for thread-safe concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_store_operations(self) -> None:
        """Test concurrent store_adr calls don't corrupt state."""
        store = InMemoryADRStore()

        async def store_adr(i: int) -> str:
            adr = _make_adr(f"ADR-{i:03d}")
            return await store.store_adr(adr)

        # Run 50 concurrent stores
        tasks = [store_adr(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        # All should complete without error
        assert len(results) == 50

        # All should be stored
        all_adrs = await store.get_all_adrs()
        assert len(all_adrs) == 50

    @pytest.mark.asyncio
    async def test_concurrent_get_next_number(self) -> None:
        """Test concurrent get_next_adr_number returns unique numbers."""
        store = InMemoryADRStore()

        async def get_number() -> int:
            return await store.get_next_adr_number()

        # Run 100 concurrent number requests
        tasks = [get_number() for _ in range(100)]
        results = await asyncio.gather(*tasks)

        # All numbers should be unique
        assert len(set(results)) == 100
        # Should be sequential from 1 to 100
        assert set(results) == set(range(1, 101))

    @pytest.mark.asyncio
    async def test_thread_safe_with_concurrent_read_write(self) -> None:
        """Test thread safety with concurrent read and write operations."""
        store = InMemoryADRStore()

        # Store some initial ADRs
        for i in range(10):
            adr = _make_adr(f"ADR-{i:03d}")
            await store.store_adr(adr)

        async def read_operation() -> int:
            all_adrs = await store.get_all_adrs()
            return len(all_adrs)

        async def write_operation(i: int) -> str:
            adr = _make_adr(f"ADR-W{i:03d}")
            return await store.store_adr(adr)

        # Mix reads and writes concurrently
        read_tasks = [read_operation() for _ in range(20)]
        write_tasks = [write_operation(i) for i in range(20)]
        all_tasks = read_tasks + write_tasks

        results = await asyncio.gather(*all_tasks)

        # All should complete without error
        assert len(results) == 40

        # Final count should be 30 (10 initial + 20 new)
        final_count = await store.get_all_adrs()
        assert len(final_count) == 30


class TestInMemoryADRStoreTimeRange:
    """Tests for get_adrs_by_time_range method."""

    @pytest.mark.asyncio
    async def test_get_adrs_by_time_range_empty_store(self) -> None:
        """Test time range query on empty store."""
        store = InMemoryADRStore()

        result = await store.get_adrs_by_time_range(
            start_time="2026-01-01T00:00:00+00:00",
            end_time="2026-01-31T23:59:59+00:00",
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_get_adrs_by_time_range_filters_correctly(self) -> None:
        """Test that time range filtering works correctly."""
        store = InMemoryADRStore()
        # Store ADRs with different timestamps
        adr1 = _make_adr("ADR-001", created_at="2026-01-10T00:00:00+00:00")
        adr2 = _make_adr("ADR-002", created_at="2026-01-15T00:00:00+00:00")
        adr3 = _make_adr("ADR-003", created_at="2026-01-20T00:00:00+00:00")
        adr4 = _make_adr("ADR-004", created_at="2026-01-25T00:00:00+00:00")
        await store.store_adr(adr1)
        await store.store_adr(adr2)
        await store.store_adr(adr3)
        await store.store_adr(adr4)

        # Query middle range
        result = await store.get_adrs_by_time_range(
            start_time="2026-01-12T00:00:00+00:00",
            end_time="2026-01-22T00:00:00+00:00",
        )

        assert len(result) == 2
        adr_ids = {adr.id for adr in result}
        assert adr_ids == {"ADR-002", "ADR-003"}

    @pytest.mark.asyncio
    async def test_get_adrs_by_time_range_start_only(self) -> None:
        """Test time range with only start_time specified."""
        store = InMemoryADRStore()
        adr1 = _make_adr("ADR-001", created_at="2026-01-10T00:00:00+00:00")
        adr2 = _make_adr("ADR-002", created_at="2026-01-20T00:00:00+00:00")
        await store.store_adr(adr1)
        await store.store_adr(adr2)

        result = await store.get_adrs_by_time_range(
            start_time="2026-01-15T00:00:00+00:00",
        )

        assert len(result) == 1
        assert result[0].id == "ADR-002"

    @pytest.mark.asyncio
    async def test_get_adrs_by_time_range_end_only(self) -> None:
        """Test time range with only end_time specified."""
        store = InMemoryADRStore()
        adr1 = _make_adr("ADR-001", created_at="2026-01-10T00:00:00+00:00")
        adr2 = _make_adr("ADR-002", created_at="2026-01-20T00:00:00+00:00")
        await store.store_adr(adr1)
        await store.store_adr(adr2)

        result = await store.get_adrs_by_time_range(
            end_time="2026-01-15T00:00:00+00:00",
        )

        assert len(result) == 1
        assert result[0].id == "ADR-001"

    @pytest.mark.asyncio
    async def test_get_adrs_by_time_range_no_bounds(self) -> None:
        """Test time range with no bounds returns all ADRs."""
        store = InMemoryADRStore()
        adr1 = _make_adr("ADR-001", created_at="2026-01-10T00:00:00+00:00")
        adr2 = _make_adr("ADR-002", created_at="2026-01-20T00:00:00+00:00")
        await store.store_adr(adr1)
        await store.store_adr(adr2)

        result = await store.get_adrs_by_time_range()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_adrs_by_time_range_ordered_descending(self) -> None:
        """Test that results are ordered by created_at descending."""
        store = InMemoryADRStore()
        adr1 = _make_adr("ADR-001", created_at="2026-01-10T00:00:00+00:00")
        adr2 = _make_adr("ADR-002", created_at="2026-01-15T00:00:00+00:00")
        adr3 = _make_adr("ADR-003", created_at="2026-01-20T00:00:00+00:00")
        await store.store_adr(adr1)
        await store.store_adr(adr2)
        await store.store_adr(adr3)

        result = await store.get_adrs_by_time_range()

        # Should be newest first
        assert result[0].id == "ADR-003"
        assert result[1].id == "ADR-002"
        assert result[2].id == "ADR-001"

    @pytest.mark.asyncio
    async def test_get_adrs_by_time_range_inclusive_bounds(self) -> None:
        """Test that time range bounds are inclusive."""
        store = InMemoryADRStore()
        adr = _make_adr("ADR-001", created_at="2026-01-15T00:00:00+00:00")
        await store.store_adr(adr)

        # Query with exact bounds
        result = await store.get_adrs_by_time_range(
            start_time="2026-01-15T00:00:00+00:00",
            end_time="2026-01-15T00:00:00+00:00",
        )

        assert len(result) == 1
        assert result[0].id == "ADR-001"
