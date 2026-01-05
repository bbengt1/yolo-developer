"""Integration tests for memory persistence.

Tests cover:
- Full lifecycle with real ChromaDB persistence
- Data persistence across client restarts
- Error recovery with retry behavior
- JSONGraphStore persistence across restarts
- ChromaMemory and JSONGraphStore integration
"""

from __future__ import annotations

from typing import Any

import pytest


class TestChromaDBPersistence:
    """Integration tests for ChromaDB persistence behavior."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_store_search_persist(self, tmp_path: Any) -> None:
        """Full lifecycle: store, search, persist, reload, search again."""
        from yolo_developer.memory import ChromaMemory

        persist_dir = str(tmp_path / "chroma_test")

        # Create first instance and store data
        memory1 = ChromaMemory(persist_directory=persist_dir)

        # Store multiple embeddings
        await memory1.store_embedding(
            key="doc-1",
            content="Python is a high-level programming language",
            metadata={"type": "language", "category": "programming"},
        )
        await memory1.store_embedding(
            key="doc-2",
            content="JavaScript runs in web browsers",
            metadata={"type": "language", "category": "web"},
        )
        await memory1.store_embedding(
            key="doc-3",
            content="SQL is used for database queries",
            metadata={"type": "language", "category": "database"},
        )

        # Verify search works
        results = await memory1.search_similar("programming languages", k=3)
        assert len(results) == 3
        assert any(r.key == "doc-1" for r in results)

        # Close first instance
        del memory1

        # Create second instance with same persist directory
        memory2 = ChromaMemory(persist_directory=persist_dir)

        # Verify data persisted and is searchable
        results = await memory2.search_similar("programming languages", k=3)
        assert len(results) == 3

        # Verify specific document can be found
        python_results = await memory2.search_similar("Python high-level", k=1)
        assert len(python_results) == 1
        assert python_results[0].key == "doc-1"
        assert python_results[0].metadata["type"] == "language"

    @pytest.mark.asyncio
    async def test_upsert_behavior_across_instances(self, tmp_path: Any) -> None:
        """Upsert behavior: update should work across client restarts."""
        from yolo_developer.memory import ChromaMemory

        persist_dir = str(tmp_path / "chroma_upsert")

        # Create first instance and store data
        memory1 = ChromaMemory(persist_directory=persist_dir)
        await memory1.store_embedding(
            key="update-test",
            content="Original content version 1",
            metadata={"version": "1"},
        )
        del memory1

        # Create second instance and update
        memory2 = ChromaMemory(persist_directory=persist_dir)
        await memory2.store_embedding(
            key="update-test",
            content="Updated content version 2",
            metadata={"version": "2"},
        )
        del memory2

        # Create third instance and verify update persisted
        memory3 = ChromaMemory(persist_directory=persist_dir)
        results = await memory3.search_similar("Updated content", k=1)
        assert len(results) == 1
        assert results[0].key == "update-test"
        assert results[0].content == "Updated content version 2"
        assert results[0].metadata["version"] == "2"

    @pytest.mark.asyncio
    async def test_empty_collection_persistence(self, tmp_path: Any) -> None:
        """Empty collection should persist correctly."""
        from yolo_developer.memory import ChromaMemory

        persist_dir = str(tmp_path / "chroma_empty")

        # Create and close without storing anything
        memory1 = ChromaMemory(persist_directory=persist_dir)
        results1 = await memory1.search_similar("anything", k=5)
        assert results1 == []
        del memory1

        # Verify empty collection works after restart
        memory2 = ChromaMemory(persist_directory=persist_dir)
        results2 = await memory2.search_similar("anything", k=5)
        assert results2 == []

    @pytest.mark.asyncio
    async def test_large_content_persistence(self, tmp_path: Any) -> None:
        """Large content should persist correctly."""
        from yolo_developer.memory import ChromaMemory

        persist_dir = str(tmp_path / "chroma_large")

        # Create large content
        large_content = "This is a test document. " * 100  # ~2500 characters

        memory1 = ChromaMemory(persist_directory=persist_dir)
        await memory1.store_embedding(
            key="large-doc",
            content=large_content,
            metadata={"size": "large"},
        )
        del memory1

        # Verify large content persisted
        memory2 = ChromaMemory(persist_directory=persist_dir)
        results = await memory2.search_similar("test document", k=1)
        assert len(results) == 1
        assert results[0].key == "large-doc"
        assert results[0].content == large_content

    @pytest.mark.asyncio
    async def test_multiple_collections_isolation(self, tmp_path: Any) -> None:
        """Different collections in same directory should be isolated."""
        from yolo_developer.memory import ChromaMemory

        persist_dir = str(tmp_path / "chroma_multi")

        # Create two collections
        memory_a = ChromaMemory(persist_directory=persist_dir, collection_name="collection_a")
        memory_b = ChromaMemory(persist_directory=persist_dir, collection_name="collection_b")

        # Store in collection A
        await memory_a.store_embedding(
            key="a-doc",
            content="Content for collection A",
            metadata={"collection": "a"},
        )

        # Store in collection B
        await memory_b.store_embedding(
            key="b-doc",
            content="Content for collection B",
            metadata={"collection": "b"},
        )

        # Verify isolation - A should not see B's data
        results_a = await memory_a.search_similar("Content", k=5)
        assert len(results_a) == 1
        assert results_a[0].key == "a-doc"

        results_b = await memory_b.search_similar("Content", k=5)
        assert len(results_b) == 1
        assert results_b[0].key == "b-doc"


class TestRetryConfiguration:
    """Integration tests for retry configuration."""

    @pytest.mark.asyncio
    async def test_operations_complete_successfully(self, tmp_path: Any) -> None:
        """Verify ChromaDB operations complete successfully under normal conditions."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        # Store operation should complete without error
        await memory.store_embedding(
            key="test-1",
            content="Test content for retry verification",
            metadata={"test": "true"},
        )

        # Search operation should complete without error
        results = await memory.search_similar("Test content", k=1)
        assert len(results) == 1
        assert results[0].key == "test-1"

    @pytest.mark.asyncio
    async def test_chromadberror_is_importable(self) -> None:
        """Verify ChromaDBError exception class is exported."""
        from yolo_developer.memory import ChromaDBError

        # ChromaDBError should be importable and usable
        assert ChromaDBError is not None
        error = ChromaDBError(
            message="Test error",
            operation="test_op",
            original_error=RuntimeError("original"),
        )
        assert error.operation == "test_op"
        assert isinstance(error.original_error, RuntimeError)

    @pytest.mark.asyncio
    async def test_collection_count_grows_with_embeddings(self, tmp_path: Any) -> None:
        """Verify collection grows when embeddings are stored."""
        from yolo_developer.memory import ChromaMemory

        memory = ChromaMemory(persist_directory=str(tmp_path))

        # Initially empty - search returns nothing
        initial_results = await memory.search_similar("anything", k=10)
        assert len(initial_results) == 0

        # Store some data
        await memory.store_embedding(
            key="test-1",
            content="Test content 1",
            metadata={},
        )
        await memory.store_embedding(
            key="test-2",
            content="Test content 2",
            metadata={},
        )

        # Now search should find both
        results = await memory.search_similar("Test content", k=10)
        assert len(results) == 2


class TestJSONGraphStorePersistence:
    """Integration tests for JSONGraphStore persistence behavior."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_store_query_persist(self, tmp_path: Any) -> None:
        """Full lifecycle: store, query, persist, reload, query again."""
        from yolo_developer.memory import JSONGraphStore

        persist_path = str(tmp_path / "graph_test.json")

        # Create first instance and store data
        store1 = JSONGraphStore(persist_path=persist_path)

        # Store multiple relationships
        await store1.store_relationship("story-001", "req-001", "implements")
        await store1.store_relationship("story-001", "req-002", "implements")
        await store1.store_relationship("story-002", "req-003", "implements")
        await store1.store_relationship("story-002", "story-001", "depends_on")

        # Verify query works
        results = await store1.get_relationships(source="story-001")
        assert len(results) == 2

        # Close first instance
        del store1

        # Create second instance with same persist path
        store2 = JSONGraphStore(persist_path=persist_path)

        # Verify data persisted and is queryable
        all_results = await store2.get_relationships()
        assert len(all_results) == 4

        # Verify specific queries work
        story1_results = await store2.get_relationships(source="story-001")
        assert len(story1_results) == 2
        assert all(r.source == "story-001" for r in story1_results)

        depends_results = await store2.get_relationships(relation="depends_on")
        assert len(depends_results) == 1
        assert depends_results[0].source == "story-002"
        assert depends_results[0].target == "story-001"

    @pytest.mark.asyncio
    async def test_transitive_queries_across_restarts(self, tmp_path: Any) -> None:
        """Transitive queries should work after persistence reload."""
        from yolo_developer.memory import JSONGraphStore

        persist_path = str(tmp_path / "graph_transitive.json")

        # Create chain: A -> B -> C -> D
        store1 = JSONGraphStore(persist_path=persist_path)
        await store1.store_relationship("A", "B", "links")
        await store1.store_relationship("B", "C", "links")
        await store1.store_relationship("C", "D", "links")
        del store1

        # Reload and verify transitive queries work
        store2 = JSONGraphStore(persist_path=persist_path)
        related_depth1 = await store2.get_related("A", depth=1)
        assert set(related_depth1) == {"B"}

        related_depth2 = await store2.get_related("A", depth=2)
        assert set(related_depth2) == {"B", "C"}

        related_depth3 = await store2.get_related("A", depth=3)
        assert set(related_depth3) == {"B", "C", "D"}

    @pytest.mark.asyncio
    async def test_duplicate_prevention_across_restarts(self, tmp_path: Any) -> None:
        """Duplicate relationships should be prevented across restarts."""
        from yolo_developer.memory import JSONGraphStore

        persist_path = str(tmp_path / "graph_duplicates.json")

        # Store relationship in first instance
        store1 = JSONGraphStore(persist_path=persist_path)
        await store1.store_relationship("A", "B", "links")
        del store1

        # Try to store same relationship in second instance
        store2 = JSONGraphStore(persist_path=persist_path)
        await store2.store_relationship("A", "B", "links")  # Duplicate
        await store2.store_relationship("A", "C", "links")  # New

        results = await store2.get_relationships()
        assert len(results) == 2  # Only 2, not 3

    @pytest.mark.asyncio
    async def test_graph_error_is_importable(self) -> None:
        """Verify JSONGraphError exception class is exported."""
        from yolo_developer.memory import JSONGraphError

        # JSONGraphError should be importable and usable
        assert JSONGraphError is not None
        error = JSONGraphError(
            message="Test error",
            operation="save",
            original_error=OSError("disk full"),
        )
        assert error.operation == "save"
        assert isinstance(error.original_error, OSError)


class TestChromaMemoryGraphIntegration:
    """Integration tests for ChromaMemory with JSONGraphStore."""

    @pytest.mark.asyncio
    async def test_chromamemory_delegates_to_graphstore(self, tmp_path: Any) -> None:
        """ChromaMemory should delegate store_relationship to graph store."""
        from yolo_developer.memory import ChromaMemory, JSONGraphStore

        chroma_dir = str(tmp_path / "chroma")
        graph_path = str(tmp_path / "graph.json")

        # Create integrated memory
        graph_store = JSONGraphStore(persist_path=graph_path)
        memory = ChromaMemory(persist_directory=chroma_dir, graph_store=graph_store)

        # Store relationship via ChromaMemory
        await memory.store_relationship("story-001", "req-001", "implements")
        await memory.store_relationship("story-002", "req-002", "depends_on")

        # Verify it's stored in graph store
        results = await graph_store.get_relationships()
        assert len(results) == 2

        # Verify filters work
        implements = await graph_store.get_relationships(relation="implements")
        assert len(implements) == 1
        assert implements[0].source == "story-001"

    @pytest.mark.asyncio
    async def test_full_memory_lifecycle_with_graph(self, tmp_path: Any) -> None:
        """Full lifecycle with both vector and graph storage."""
        from yolo_developer.memory import ChromaMemory, JSONGraphStore

        chroma_dir = str(tmp_path / "chroma")
        graph_path = str(tmp_path / "graph.json")

        # Create first instance with integrated stores
        graph1 = JSONGraphStore(persist_path=graph_path)
        memory1 = ChromaMemory(persist_directory=chroma_dir, graph_store=graph1)

        # Store embedding
        await memory1.store_embedding(
            key="req-001",
            content="User authentication via OAuth2 protocol",
            metadata={"type": "requirement"},
        )

        # Store relationship
        await memory1.store_relationship("story-001", "req-001", "implements")

        del memory1
        del graph1

        # Create second instance
        graph2 = JSONGraphStore(persist_path=graph_path)
        memory2 = ChromaMemory(persist_directory=chroma_dir, graph_store=graph2)

        # Verify embedding persisted
        results = await memory2.search_similar("OAuth authentication", k=1)
        assert len(results) == 1
        assert results[0].key == "req-001"

        # Verify relationship persisted
        rel_results = await graph2.get_relationships(source="story-001")
        assert len(rel_results) == 1
        assert rel_results[0].target == "req-001"

    @pytest.mark.asyncio
    async def test_memory_without_graph_store_logs_warning(
        self, tmp_path: Any, caplog: Any
    ) -> None:
        """ChromaMemory without graph_store should log warning on store_relationship."""
        import logging

        from yolo_developer.memory import ChromaMemory

        chroma_dir = str(tmp_path / "chroma")
        memory = ChromaMemory(persist_directory=chroma_dir)  # No graph_store

        with caplog.at_level(logging.WARNING):
            await memory.store_relationship("a", "b", "links")

        # Verify warning was logged
        assert "no graph_store configured" in caplog.text
