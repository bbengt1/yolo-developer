"""Integration tests for project isolation.

Tests that data stored in one project is completely isolated from another project.
Verifies all memory stores respect project boundaries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_developer.memory import (
    ChromaDecisionStore,
    ChromaMemory,
    ChromaPatternStore,
    MemoryFactory,
)
from yolo_developer.memory.decisions import Decision, DecisionType
from yolo_developer.memory.patterns import CodePattern, PatternType


class TestChromaMemoryIsolation:
    """Integration tests for ChromaMemory project isolation."""

    @pytest.mark.asyncio
    async def test_chromamemory_collections_isolated(self, tmp_path: Path) -> None:
        """Data stored in Project A is not visible in Project B."""
        # Create two instances with different project IDs
        memory_a = ChromaMemory(
            persist_directory=str(tmp_path),
            project_id="project-a",
        )
        memory_b = ChromaMemory(
            persist_directory=str(tmp_path),
            project_id="project-b",
        )

        # Store content in Project A
        await memory_a.store_embedding(
            key="doc-001",
            content="Secret document from Project A",
            metadata={"type": "secret"},
        )

        # Query from Project B - should NOT find Project A's content
        results_b = await memory_b.search_similar("Secret document", k=10)
        assert len(results_b) == 0, "Project B should not see Project A's data"

        # Query from Project A - should find the content
        results_a = await memory_a.search_similar("Secret document", k=10)
        assert len(results_a) == 1
        assert results_a[0].key == "doc-001"

    @pytest.mark.asyncio
    async def test_chromamemory_collections_have_different_names(self, tmp_path: Path) -> None:
        """Different projects use different collection names."""
        memory_a = ChromaMemory(
            persist_directory=str(tmp_path),
            project_id="project-alpha",
        )
        memory_b = ChromaMemory(
            persist_directory=str(tmp_path),
            project_id="project-beta",
        )

        # Verify collection names are different
        assert memory_a.collection.name != memory_b.collection.name
        assert "project-alpha" in memory_a.collection.name
        assert "project-beta" in memory_b.collection.name


class TestJSONGraphStoreIsolation:
    """Integration tests for JSONGraphStore project isolation."""

    @pytest.mark.asyncio
    async def test_graph_stores_use_separate_files(self, tmp_path: Path) -> None:
        """Different projects use separate JSON files."""
        factory_a = MemoryFactory(
            project_id="project-a",
            base_directory=str(tmp_path),
        )
        factory_b = MemoryFactory(
            project_id="project-b",
            base_directory=str(tmp_path),
        )

        graph_a = factory_a.create_graph_store()
        graph_b = factory_b.create_graph_store()

        # Verify different file paths
        assert graph_a.persist_path != graph_b.persist_path
        assert "project-a" in str(graph_a.persist_path)
        assert "project-b" in str(graph_b.persist_path)

    @pytest.mark.asyncio
    async def test_graph_relationships_isolated(self, tmp_path: Path) -> None:
        """Relationships in Project A are invisible to Project B."""
        factory_a = MemoryFactory(
            project_id="project-a",
            base_directory=str(tmp_path),
        )
        factory_b = MemoryFactory(
            project_id="project-b",
            base_directory=str(tmp_path),
        )

        graph_a = factory_a.create_graph_store()
        graph_b = factory_b.create_graph_store()

        # Store relationship in Project A
        await graph_a.store_relationship("story-001", "req-001", "implements")

        # Query from Project B - should be empty
        results_b = await graph_b.get_relationships(source="story-001")
        assert len(results_b) == 0, "Project B should not see Project A's relationships"

        # Query from Project A - should find the relationship
        results_a = await graph_a.get_relationships(source="story-001")
        assert len(results_a) == 1
        assert results_a[0].target == "req-001"


class TestPatternStoreIsolation:
    """Integration tests for ChromaPatternStore project isolation."""

    @pytest.mark.asyncio
    async def test_pattern_stores_isolated(self, tmp_path: Path) -> None:
        """Patterns in Project A are not found in Project B."""
        store_a = ChromaPatternStore(
            persist_directory=str(tmp_path),
            project_id="project-a",
        )
        store_b = ChromaPatternStore(
            persist_directory=str(tmp_path),
            project_id="project-b",
        )

        # Store pattern in Project A
        pattern = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
        )
        await store_a.store_pattern(pattern)

        # Search from Project B - should not find Project A's pattern
        results_b = await store_b.search_patterns(query="function naming", k=10)
        assert len(results_b) == 0, "Project B should not see Project A's patterns"

        # Search from Project A - should find the pattern
        results_a = await store_a.search_patterns(query="function naming", k=10)
        assert len(results_a) == 1

    @pytest.mark.asyncio
    async def test_pattern_stores_have_different_collections(self, tmp_path: Path) -> None:
        """Different projects use different collection names."""
        store_a = ChromaPatternStore(
            persist_directory=str(tmp_path),
            project_id="project-alpha",
        )
        store_b = ChromaPatternStore(
            persist_directory=str(tmp_path),
            project_id="project-beta",
        )

        # Verify collection names include project IDs
        assert store_a.project_id == "project-alpha"
        assert store_b.project_id == "project-beta"


class TestDecisionStoreIsolation:
    """Integration tests for ChromaDecisionStore project isolation."""

    @pytest.mark.asyncio
    async def test_decision_stores_isolated(self, tmp_path: Path) -> None:
        """Decisions in Project A are not found in Project B."""
        store_a = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="project-a",
        )
        store_b = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="project-b",
        )

        # Store decision in Project A
        decision = Decision(
            id="dec-001",
            agent_type="Architect",
            context="Choosing database",
            rationale="PostgreSQL for reliability",
            decision_type=DecisionType.ARCHITECTURE_CHOICE,
        )
        await store_a.store_decision(decision)

        # Search from Project B - should not find Project A's decision
        results_b = await store_b.search_decisions(query="database choice", k=10)
        assert len(results_b) == 0, "Project B should not see Project A's decisions"

        # Search from Project A - should find the decision
        results_a = await store_a.search_decisions(query="database choice", k=10)
        assert len(results_a) == 1

    @pytest.mark.asyncio
    async def test_decision_stores_have_different_collections(self, tmp_path: Path) -> None:
        """Different projects use different collection names."""
        store_a = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="project-alpha",
        )
        store_b = ChromaDecisionStore(
            persist_directory=str(tmp_path),
            project_id="project-beta",
        )

        # Verify project IDs are different
        assert store_a.project_id == "project-alpha"
        assert store_b.project_id == "project-beta"


class TestMemoryFactoryIsolation:
    """Integration tests for MemoryFactory project isolation."""

    @pytest.mark.asyncio
    async def test_factory_creates_isolated_stores(self, tmp_path: Path) -> None:
        """Factory creates all stores with consistent project isolation."""
        factory_a = MemoryFactory(
            project_id="factory-project-a",
            base_directory=str(tmp_path),
        )
        factory_b = MemoryFactory(
            project_id="factory-project-b",
            base_directory=str(tmp_path),
        )

        stores_a = factory_a.get_all_stores()
        stores_b = factory_b.get_all_stores()

        # Verify vector stores are isolated
        await stores_a["vector"].store_embedding(
            key="test-doc",
            content="Test document for Project A",
            metadata={},
        )
        results_b = await stores_b["vector"].search_similar("Test document", k=10)
        assert len(results_b) == 0, "Vector store should be isolated"

        # Verify graph stores are isolated
        await stores_a["graph"].store_relationship("a", "b", "test")
        results_b_graph = await stores_b["graph"].get_relationships(source="a")
        assert len(results_b_graph) == 0, "Graph store should be isolated"

    @pytest.mark.asyncio
    async def test_context_switching_between_projects(self, tmp_path: Path) -> None:
        """Switching from Project A to Project B isolates data correctly."""
        # Work on Project A
        factory_a = MemoryFactory(
            project_id="context-a",
            base_directory=str(tmp_path),
        )
        vector_a = factory_a.create_vector_store()
        await vector_a.store_embedding(
            key="secret-a",
            content="Secret information for Project A only",
            metadata={"project": "a"},
        )

        # Switch to Project B - new factory instance
        factory_b = MemoryFactory(
            project_id="context-b",
            base_directory=str(tmp_path),
        )
        vector_b = factory_b.create_vector_store()

        # Verify Project A's data is NOT accessible
        results = await vector_b.search_similar("Secret information", k=10)
        assert len(results) == 0, "After context switch, old project data should not be accessible"

        # Store different data in Project B
        await vector_b.store_embedding(
            key="secret-b",
            content="Different secret for Project B",
            metadata={"project": "b"},
        )

        # Verify Project B sees its own data
        results_b = await vector_b.search_similar("secret", k=10)
        assert len(results_b) == 1
        assert results_b[0].metadata["project"] == "b"

        # Verify Project A still sees its own data (not affected by B)
        results_a = await vector_a.search_similar("secret", k=10)
        assert len(results_a) == 1
        assert results_a[0].metadata["project"] == "a"


class TestDefaultProjectIdBackwardCompatibility:
    """Tests that default project ID maintains backward compatibility."""

    @pytest.mark.asyncio
    async def test_chromamemory_default_project_id(self, tmp_path: Path) -> None:
        """ChromaMemory without project_id uses default for backward compatibility."""
        # Create without explicit project_id
        memory = ChromaMemory(persist_directory=str(tmp_path))

        # Store and retrieve
        await memory.store_embedding(
            key="test",
            content="Test content",
            metadata={},
        )
        results = await memory.search_similar("Test", k=5)
        assert len(results) == 1

        # Collection name should include "default"
        assert "default" in memory.collection.name

    @pytest.mark.asyncio
    async def test_factory_default_project_id(self, tmp_path: Path) -> None:
        """MemoryFactory without project_id uses default for backward compatibility."""
        factory = MemoryFactory(base_directory=str(tmp_path))
        assert factory.project_id == "default"

        # All stores should work with default project_id
        stores = factory.get_all_stores()
        assert stores["pattern"].project_id == "default"
        assert stores["decision"].project_id == "default"
