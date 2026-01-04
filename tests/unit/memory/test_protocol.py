"""Tests for memory store protocol and types.

Tests cover:
- MemoryResult dataclass instantiation and field access
- MemoryStore protocol definition and structural typing
- Import paths work correctly from package root
"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Protocol


class TestMemoryResult:
    """Tests for MemoryResult dataclass."""

    def test_memory_result_instantiation(self) -> None:
        """MemoryResult should be instantiable with required fields."""
        from yolo_developer.memory import MemoryResult

        result = MemoryResult(
            key="test-key",
            content="test content",
            score=0.95,
            metadata={"source": "test"},
        )

        assert result.key == "test-key"
        assert result.content == "test content"
        assert result.score == 0.95
        assert result.metadata == {"source": "test"}

    def test_memory_result_is_dataclass(self) -> None:
        """MemoryResult should be a proper dataclass."""
        from yolo_developer.memory import MemoryResult

        assert is_dataclass(MemoryResult)

    def test_memory_result_fields_have_correct_types(self) -> None:
        """MemoryResult fields should have correct type annotations."""
        from yolo_developer.memory import MemoryResult

        # Check field annotations exist
        annotations = MemoryResult.__annotations__
        assert "key" in annotations
        assert "content" in annotations
        assert "score" in annotations
        assert "metadata" in annotations


class TestMemoryStoreProtocol:
    """Tests for MemoryStore protocol."""

    def test_memory_store_is_protocol(self) -> None:
        """MemoryStore should be a Protocol class."""
        from yolo_developer.memory import MemoryStore

        # MemoryStore should be a Protocol
        assert issubclass(type(MemoryStore), type(Protocol))

    def test_mock_implementation_satisfies_protocol(self) -> None:
        """A class implementing required methods should satisfy the protocol."""
        from yolo_developer.memory import MemoryResult, MemoryStore

        class MockMemory:
            async def store_embedding(
                self, key: str, content: str, metadata: dict[str, Any]
            ) -> None:
                pass

            async def search_similar(self, query: str, k: int = 5) -> list[MemoryResult]:
                return []

            async def store_relationship(self, source: str, target: str, relation: str) -> None:
                pass

        # This should work without errors (duck typing)
        memory: MemoryStore = MockMemory()
        assert memory is not None

    def test_protocol_has_required_methods(self) -> None:
        """Protocol should define store_embedding, search_similar, store_relationship."""
        from yolo_developer.memory import MemoryStore

        # Check that the protocol has the expected method signatures
        assert hasattr(MemoryStore, "store_embedding")
        assert hasattr(MemoryStore, "search_similar")
        assert hasattr(MemoryStore, "store_relationship")


class TestMemoryResultImmutability:
    """Tests for MemoryResult immutability (frozen dataclass)."""

    def test_memory_result_is_immutable(self) -> None:
        """MemoryResult should be immutable (frozen dataclass)."""
        import pytest

        from yolo_developer.memory import MemoryResult

        result = MemoryResult(
            key="test-key",
            content="test content",
            score=0.95,
            metadata={"source": "test"},
        )

        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(AttributeError):
            result.key = "new-key"  # type: ignore[misc]


class TestMemoryModuleImports:
    """Tests for memory module import paths."""

    def test_imports_from_package_root(self) -> None:
        """MemoryStore and MemoryResult should be importable from memory package."""
        from yolo_developer.memory import MemoryResult, MemoryStore

        assert MemoryStore is not None
        assert MemoryResult is not None

    def test_imports_from_protocol_module(self) -> None:
        """Types should also be importable from protocol submodule."""
        from yolo_developer.memory.protocol import MemoryResult, MemoryStore

        assert MemoryStore is not None
        assert MemoryResult is not None
