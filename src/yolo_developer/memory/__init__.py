"""Memory module for YOLO Developer.

This module provides the memory abstraction layer for storing and retrieving
project artifacts, embeddings, and relationships.

Exports:
    MemoryStore: Protocol defining the memory store interface.
    MemoryResult: Dataclass for similarity search results.

Example:
    >>> from yolo_developer.memory import MemoryStore, MemoryResult
    >>>
    >>> # MemoryStore can be used as a type annotation
    >>> def process_context(memory: MemoryStore) -> None:
    ...     pass
"""

from __future__ import annotations

from yolo_developer.memory.protocol import MemoryResult, MemoryStore

__all__ = [
    "MemoryResult",
    "MemoryStore",
]
