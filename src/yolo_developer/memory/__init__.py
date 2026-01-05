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
    CodePattern: Dataclass for learned code patterns.
    PatternType: Enum of pattern categories.
    PatternResult: Dataclass for pattern search results.
    ChromaPatternStore: ChromaDB-backed pattern storage.
    PatternLearner: Orchestrates pattern learning from codebases.
    PatternLearningResult: Result of pattern learning process.
    CodebaseScanner: Scans codebases for source files.
    ScanResult: Result of scanning a codebase.

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
    >>>
    >>> # Pattern learning
    >>> from yolo_developer.memory import PatternLearner, CodePattern, PatternType
    >>> learner = PatternLearner(
    ...     persist_directory=".yolo/patterns",
    ...     project_id="my-project",
    ... )
    >>> result = await learner.learn_from_codebase(Path("/my/project"))
    >>> print(f"Found {result.patterns_found} patterns")
"""

from __future__ import annotations

from yolo_developer.memory.graph import (
    JSONGraphError,
    JSONGraphStore,
    Relationship,
    RelationshipResult,
)
from yolo_developer.memory.learning import PatternLearner, PatternLearningResult
from yolo_developer.memory.pattern_store import ChromaPatternStore
from yolo_developer.memory.patterns import CodePattern, PatternResult, PatternType
from yolo_developer.memory.protocol import MemoryResult, MemoryStore
from yolo_developer.memory.scanner import CodebaseScanner, ScanResult
from yolo_developer.memory.vector import ChromaDBError, ChromaMemory

__all__ = [
    "ChromaDBError",
    "ChromaMemory",
    "ChromaPatternStore",
    "CodePattern",
    "CodebaseScanner",
    "JSONGraphError",
    "JSONGraphStore",
    "MemoryResult",
    "MemoryStore",
    "PatternLearner",
    "PatternLearningResult",
    "PatternResult",
    "PatternType",
    "Relationship",
    "RelationshipResult",
    "ScanResult",
]
