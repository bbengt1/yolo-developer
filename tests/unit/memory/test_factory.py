"""Unit tests for MemoryFactory.

Tests for the factory pattern that creates project-isolated memory stores.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_developer.memory.factory import MemoryFactory
from yolo_developer.memory.isolation import InvalidProjectIdError


class TestMemoryFactoryInit:
    """Tests for MemoryFactory initialization."""

    def test_factory_init_with_valid_project_id(self, tmp_path: Path) -> None:
        """Factory initializes with valid project ID."""
        factory = MemoryFactory(
            project_id="my-project",
            base_directory=str(tmp_path),
        )
        assert factory.project_id == "my-project"

    def test_factory_init_with_default_project_id(self, tmp_path: Path) -> None:
        """Factory uses default project ID when not specified."""
        factory = MemoryFactory(base_directory=str(tmp_path))
        assert factory.project_id == "default"

    def test_factory_init_validates_project_id(self, tmp_path: Path) -> None:
        """Factory raises InvalidProjectIdError for invalid project ID."""
        with pytest.raises(InvalidProjectIdError):
            MemoryFactory(
                project_id="invalid@project!",
                base_directory=str(tmp_path),
            )

    def test_factory_init_empty_project_id_raises_error(self, tmp_path: Path) -> None:
        """Factory raises InvalidProjectIdError for empty project ID."""
        with pytest.raises(InvalidProjectIdError):
            MemoryFactory(
                project_id="",
                base_directory=str(tmp_path),
            )

    def test_factory_stores_base_directory(self, tmp_path: Path) -> None:
        """Factory stores the base directory path."""
        factory = MemoryFactory(
            project_id="test",
            base_directory=str(tmp_path),
        )
        assert factory.base_directory == str(tmp_path)


class TestMemoryFactoryCreateVectorStore:
    """Tests for create_vector_store method."""

    def test_create_vector_store_returns_chroma_memory(self, tmp_path: Path) -> None:
        """create_vector_store returns a ChromaMemory instance."""
        from yolo_developer.memory.vector import ChromaMemory

        factory = MemoryFactory(
            project_id="test-project",
            base_directory=str(tmp_path),
        )
        store = factory.create_vector_store()
        assert isinstance(store, ChromaMemory)

    def test_create_vector_store_uses_project_id_in_collection(self, tmp_path: Path) -> None:
        """Vector store uses project ID in collection name."""
        factory = MemoryFactory(
            project_id="my-project",
            base_directory=str(tmp_path),
        )
        store = factory.create_vector_store()
        # Collection name should include project_id
        assert "my-project" in store.collection.name


class TestMemoryFactoryCreateGraphStore:
    """Tests for create_graph_store method."""

    def test_create_graph_store_returns_json_graph_store(self, tmp_path: Path) -> None:
        """create_graph_store returns a JSONGraphStore instance."""
        from yolo_developer.memory.graph import JSONGraphStore

        factory = MemoryFactory(
            project_id="test-project",
            base_directory=str(tmp_path),
        )
        store = factory.create_graph_store()
        assert isinstance(store, JSONGraphStore)

    def test_create_graph_store_uses_project_id_in_path(self, tmp_path: Path) -> None:
        """Graph store uses project ID in file path."""
        factory = MemoryFactory(
            project_id="my-project",
            base_directory=str(tmp_path),
        )
        store = factory.create_graph_store()
        # Persist path should include project_id
        assert "my-project" in str(store.persist_path)


class TestMemoryFactoryCreatePatternStore:
    """Tests for create_pattern_store method."""

    def test_create_pattern_store_returns_chroma_pattern_store(self, tmp_path: Path) -> None:
        """create_pattern_store returns a ChromaPatternStore instance."""
        from yolo_developer.memory.pattern_store import ChromaPatternStore

        factory = MemoryFactory(
            project_id="test-project",
            base_directory=str(tmp_path),
        )
        store = factory.create_pattern_store()
        assert isinstance(store, ChromaPatternStore)

    def test_create_pattern_store_uses_project_id(self, tmp_path: Path) -> None:
        """Pattern store uses factory's project ID."""
        factory = MemoryFactory(
            project_id="my-project",
            base_directory=str(tmp_path),
        )
        store = factory.create_pattern_store()
        assert store.project_id == "my-project"


class TestMemoryFactoryCreateDecisionStore:
    """Tests for create_decision_store method."""

    def test_create_decision_store_returns_chroma_decision_store(self, tmp_path: Path) -> None:
        """create_decision_store returns a ChromaDecisionStore instance."""
        from yolo_developer.memory.decision_store import ChromaDecisionStore

        factory = MemoryFactory(
            project_id="test-project",
            base_directory=str(tmp_path),
        )
        store = factory.create_decision_store()
        assert isinstance(store, ChromaDecisionStore)

    def test_create_decision_store_uses_project_id(self, tmp_path: Path) -> None:
        """Decision store uses factory's project ID."""
        factory = MemoryFactory(
            project_id="my-project",
            base_directory=str(tmp_path),
        )
        store = factory.create_decision_store()
        assert store.project_id == "my-project"


class TestMemoryFactoryGetAllStores:
    """Tests for get_all_stores method."""

    def test_get_all_stores_returns_dict(self, tmp_path: Path) -> None:
        """get_all_stores returns a dictionary of stores."""
        factory = MemoryFactory(
            project_id="test-project",
            base_directory=str(tmp_path),
        )
        stores = factory.get_all_stores()
        assert isinstance(stores, dict)

    def test_get_all_stores_contains_all_store_types(self, tmp_path: Path) -> None:
        """get_all_stores returns all four store types."""
        factory = MemoryFactory(
            project_id="test-project",
            base_directory=str(tmp_path),
        )
        stores = factory.get_all_stores()
        assert "vector" in stores
        assert "graph" in stores
        assert "pattern" in stores
        assert "decision" in stores

    def test_get_all_stores_all_use_same_project_id(self, tmp_path: Path) -> None:
        """All stores from get_all_stores use the same project ID."""
        factory = MemoryFactory(
            project_id="unified-project",
            base_directory=str(tmp_path),
        )
        stores = factory.get_all_stores()

        # Pattern and decision stores have project_id attribute
        assert stores["pattern"].project_id == "unified-project"
        assert stores["decision"].project_id == "unified-project"


class TestMemoryFactoryProjectIsolation:
    """Tests for project isolation between factory instances."""

    def test_different_projects_get_different_stores(self, tmp_path: Path) -> None:
        """Different project IDs result in different store instances."""
        factory_a = MemoryFactory(
            project_id="project-a",
            base_directory=str(tmp_path),
        )
        factory_b = MemoryFactory(
            project_id="project-b",
            base_directory=str(tmp_path),
        )

        store_a = factory_a.create_vector_store()
        store_b = factory_b.create_vector_store()

        # Collection names should be different
        assert store_a.collection.name != store_b.collection.name

    def test_same_project_id_creates_consistent_stores(self, tmp_path: Path) -> None:
        """Same project ID creates stores with same project ID configuration."""
        factory = MemoryFactory(
            project_id="same-project",
            base_directory=str(tmp_path),
        )

        pattern_store = factory.create_pattern_store()
        decision_store = factory.create_decision_store()

        assert pattern_store.project_id == decision_store.project_id
