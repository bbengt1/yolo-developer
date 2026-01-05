"""Memory module for YOLO Developer.

This module provides the memory abstraction layer for storing and retrieving
project artifacts, embeddings, and relationships.

Exports:
    MemoryStore: Protocol defining the memory store interface.
    MemoryResult: Dataclass for similarity search results.
    ChromaMemory: ChromaDB implementation of MemoryStore.

Example:
    >>> from yolo_developer.memory import MemoryStore, MemoryResult, ChromaMemory
    >>>
    >>> # MemoryStore can be used as a type annotation
    >>> def process_context(memory: MemoryStore) -> None:
    ...     pass
    >>>
    >>> # ChromaMemory provides a concrete implementation
    >>> memory = ChromaMemory(persist_directory=".yolo/memory")
"""

from __future__ import annotations

from yolo_developer.memory.protocol import MemoryResult, MemoryStore
from yolo_developer.memory.vector import ChromaDBError, ChromaMemory

__all__ = [
    "ChromaDBError",
    "ChromaMemory",
    "MemoryResult",
    "MemoryStore",
]
