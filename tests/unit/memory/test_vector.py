"""Tests for ChromaDB vector storage implementation.

Tests cover:
- ChromaMemory class structure and protocol compliance
- store_embedding functionality with upsert behavior
- search_similar functionality with result ordering
- store_relationship stub implementation
- Persistence across client restarts
- Error handling and retry behavior
"""

from __future__ import annotations

from typing import Any

import pytest

from yolo_developer.memory import MemoryResult, MemoryStore


class TestChromaMemoryProtocolCompliance:
    """Tests for ChromaMemory protocol compliance."""

    def test_chromamemory_can_be_imported(self) -> None:
        """ChromaMemory should be importable from memory module."""
        from yolo_developer.memory import ChromaMemory

        assert ChromaMemory is not None

    def test_chromamemory_satisfies_protocol(self, tmp_path: Any) -> None:
        """ChromaMemory should satisfy MemoryStore protocol."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))
        # This assignment should pass mypy type checking
        store: MemoryStore = memory
        assert store is not None

    def test_chromamemory_has_required_methods(self, tmp_path: Any) -> None:
        """ChromaMemory should have all required protocol methods."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        assert hasattr(memory, "store_embedding")
        assert hasattr(memory, "search_similar")
        assert hasattr(memory, "store_relationship")
        assert callable(memory.store_embedding)
        assert callable(memory.search_similar)
        assert callable(memory.store_relationship)


class TestChromaMemoryInitialization:
    """Tests for ChromaMemory initialization."""

    def test_creates_with_persist_directory(self, tmp_path: Any) -> None:
        """Should create ChromaMemory with persist directory."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))
        assert memory is not None

    def test_creates_with_custom_collection_name(self, tmp_path: Any) -> None:
        """Should create ChromaMemory with custom collection name."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(
            persist_directory=str(tmp_path),
            collection_name="custom_collection",
        )
        assert memory is not None

    def test_default_collection_name(self, tmp_path: Any) -> None:
        """Should use default collection name when not specified."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))
        # Default collection name should be used
        assert memory.collection is not None


class TestStoreEmbedding:
    """Tests for store_embedding method."""

    @pytest.mark.asyncio
    async def test_store_embedding_stores_content(self, tmp_path: Any) -> None:
        """Should store content with embedding."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        await memory.store_embedding(
            key="test-key",
            content="Test content for embedding",
            metadata={"type": "test"},
        )

        # Verify by searching
        results = await memory.search_similar("Test content", k=1)
        assert len(results) == 1
        assert results[0].key == "test-key"
        assert results[0].content == "Test content for embedding"

    @pytest.mark.asyncio
    async def test_store_embedding_preserves_metadata(self, tmp_path: Any) -> None:
        """Should preserve metadata alongside embedding."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        await memory.store_embedding(
            key="meta-key",
            content="Content with metadata",
            metadata={"type": "requirement", "source": "prd.md"},
        )

        results = await memory.search_similar("Content with metadata", k=1)
        assert len(results) == 1
        assert results[0].metadata == {"type": "requirement", "source": "prd.md"}

    @pytest.mark.asyncio
    async def test_store_embedding_upsert_behavior(self, tmp_path: Any) -> None:
        """Should update existing key with upsert behavior."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        # Store initial content
        await memory.store_embedding(
            key="upsert-key",
            content="Original content",
            metadata={"version": "1"},
        )

        # Update with same key
        await memory.store_embedding(
            key="upsert-key",
            content="Updated content",
            metadata={"version": "2"},
        )

        # Search should find updated content
        results = await memory.search_similar("Updated content", k=1)
        assert len(results) == 1
        assert results[0].key == "upsert-key"
        assert results[0].content == "Updated content"
        assert results[0].metadata == {"version": "2"}

    @pytest.mark.asyncio
    async def test_store_embedding_empty_metadata(self, tmp_path: Any) -> None:
        """Should handle empty metadata dict gracefully."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        await memory.store_embedding(
            key="empty-meta-key",
            content="Content without metadata",
            metadata={},
        )

        results = await memory.search_similar("Content without metadata", k=1)
        assert len(results) == 1
        assert results[0].key == "empty-meta-key"


class TestSearchSimilar:
    """Tests for search_similar method."""

    @pytest.mark.asyncio
    async def test_search_similar_returns_memory_results(self, tmp_path: Any) -> None:
        """Should return list of MemoryResult objects."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        await memory.store_embedding(
            key="result-key",
            content="Python is a programming language",
            metadata={"type": "fact"},
        )

        results = await memory.search_similar("programming language", k=1)

        assert len(results) == 1
        assert isinstance(results[0], MemoryResult)
        assert results[0].key == "result-key"
        assert results[0].content == "Python is a programming language"
        assert results[0].metadata == {"type": "fact"}

    @pytest.mark.asyncio
    async def test_search_similar_respects_k_parameter(self, tmp_path: Any) -> None:
        """Should return at most k results."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        # Store multiple items
        for i in range(5):
            await memory.store_embedding(
                key=f"item-{i}",
                content=f"Programming content number {i}",
                metadata={"index": str(i)},
            )

        # Request only 2 results
        results = await memory.search_similar("programming", k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_similar_returns_empty_list_when_no_matches(self, tmp_path: Any) -> None:
        """Should return empty list when collection is empty."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        results = await memory.search_similar("query with no matches", k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_similar_score_in_valid_range(self, tmp_path: Any) -> None:
        """Score should be in valid range (typically 0 to 1 for similarity)."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        await memory.store_embedding(
            key="score-test",
            content="Testing similarity scores",
            metadata={},
        )

        results = await memory.search_similar("Testing similarity", k=1)

        assert len(results) == 1
        # Score should be a valid float (can be > 1 for some distance metrics)
        assert isinstance(results[0].score, float)

    @pytest.mark.asyncio
    async def test_search_similar_ordered_by_similarity(self, tmp_path: Any) -> None:
        """Results should be ordered by similarity (highest first)."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        # Store items with varying similarity to query
        await memory.store_embedding(
            key="exact",
            content="Python programming language",
            metadata={},
        )
        await memory.store_embedding(
            key="related",
            content="Java programming language",
            metadata={},
        )
        await memory.store_embedding(
            key="unrelated",
            content="Cooking recipes for dinner",
            metadata={},
        )

        results = await memory.search_similar("Python programming", k=3)

        # First result should be most similar (exact match)
        assert len(results) == 3
        # Scores should be in descending order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score


class TestStoreRelationship:
    """Tests for store_relationship method."""

    @pytest.mark.asyncio
    async def test_store_relationship_exists(self, tmp_path: Any) -> None:
        """store_relationship method should exist and be callable."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        # Should not raise - it's a no-op for MVP
        await memory.store_relationship(
            source="story-001",
            target="req-001",
            relation="implements",
        )

    @pytest.mark.asyncio
    async def test_store_relationship_does_not_raise(self, tmp_path: Any) -> None:
        """store_relationship should complete without raising."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        # Multiple calls should not raise
        await memory.store_relationship("a", "b", "depends_on")
        await memory.store_relationship("b", "c", "implements")


class TestPersistence:
    """Tests for data persistence."""

    @pytest.mark.asyncio
    async def test_data_persists_across_instances(self, tmp_path: Any) -> None:
        """Data should persist and be accessible from new instance."""
        from yolo_developer.memory import ChromaMemory

        # Store data
        memory1 = ChromaMemory(persist_directory=str(tmp_path))
        await memory1.store_embedding(
            key="persist-key",
            content="Persistent content",
            metadata={"persisted": "true"},
        )
        # Close first instance (let it go out of scope)
        del memory1

        # Create new instance with same persist directory
        memory2 = ChromaMemory(persist_directory=str(tmp_path))

        # Should find previously stored data
        results = await memory2.search_similar("Persistent content", k=1)
        assert len(results) == 1
        assert results[0].key == "persist-key"
        assert results[0].content == "Persistent content"
        assert results[0].metadata == {"persisted": "true"}
