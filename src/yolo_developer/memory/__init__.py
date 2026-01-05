"""Memory module for YOLO Developer.

This module provides the memory abstraction layer for storing and retrieving
project artifacts, embeddings, and relationships.

Exports:
    MemoryStore: Protocol defining the memory store interface.
    MemoryResult: Dataclass for similarity search results.
    ChromaMemory: ChromaDB implementation of MemoryStore.
    JSONGraphStore: JSON-based graph storage for relationships.
    Relationship: Dataclass representing a graph edge.
    RelationshipResult: Dataclass for relationship query results.

Example:
    >>> from yolo_developer.memory import MemoryStore, MemoryResult, ChromaMemory
    >>>
    >>> # MemoryStore can be used as a type annotation
    >>> def process_context(memory: MemoryStore) -> None:
    ...     pass
    >>>
    >>> # ChromaMemory provides a concrete implementation
    >>> memory = ChromaMemory(persist_directory=".yolo/memory")
    >>>
    >>> # JSONGraphStore for relationship storage
    >>> from yolo_developer.memory import JSONGraphStore
    >>> graph = JSONGraphStore(persist_path=".yolo/memory/graph.json")
    >>> await graph.store_relationship("story-001", "req-001", "implements")
"""

from __future__ import annotations

from yolo_developer.memory.graph import (
    JSONGraphError,
    JSONGraphStore,
    Relationship,
    RelationshipResult,
)
from yolo_developer.memory.protocol import MemoryResult, MemoryStore
from yolo_developer.memory.vector import ChromaDBError, ChromaMemory

__all__ = [
    "ChromaDBError",
    "ChromaMemory",
    "JSONGraphError",
    "JSONGraphStore",
    "MemoryResult",
    "MemoryStore",
    "Relationship",
    "RelationshipResult",
]
