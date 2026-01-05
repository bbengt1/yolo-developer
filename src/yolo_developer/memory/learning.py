"""Pattern learning orchestrator for code pattern learning.

This module provides the PatternLearner class that coordinates the codebase
scanner, naming analyzer, and structure analyzer to learn patterns from
existing codebases and store them for later retrieval.

Example:
    >>> from yolo_developer.memory.learning import PatternLearner
    >>> from pathlib import Path
    >>>
    >>> learner = PatternLearner(
    ...     persist_directory=".yolo/patterns",
    ...     project_id="my-project",
    ... )
    >>> result = await learner.learn_from_codebase(Path("/my/project"))
    >>> print(f"Found {result.patterns_found} patterns in {result.files_scanned} files")
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yolo_developer.memory.analyzers.naming import NamingAnalyzer
from yolo_developer.memory.analyzers.structure import StructureAnalyzer
from yolo_developer.memory.pattern_store import ChromaPatternStore
from yolo_developer.memory.patterns import CodePattern, PatternResult, PatternType
from yolo_developer.memory.scanner import CodebaseScanner

logger = logging.getLogger(__name__)

# Cache TTL in seconds (5 minutes default)
DEFAULT_CACHE_TTL = 300


@dataclass
class CacheEntry:
    """Entry in the pattern cache."""

    data: Any
    timestamp: float
    ttl: float = DEFAULT_CACHE_TTL

    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        return (time.time() - self.timestamp) < self.ttl


@dataclass
class PatternLearningResult:
    """Result of learning patterns from a codebase.

    Contains statistics about the learning process including
    the number of files scanned, lines analyzed, and patterns found.

    Attributes:
        files_scanned: Number of source files analyzed.
        total_lines: Total lines of code across all files.
        patterns_found: Number of patterns detected.
        pattern_types_found: List of pattern type values found.
    """

    files_scanned: int
    total_lines: int
    patterns_found: int
    pattern_types_found: list[str]


class PatternLearner:
    """Orchestrates pattern learning from codebases.

    Coordinates the codebase scanner with naming and structure analyzers
    to detect code patterns, then stores them for later retrieval.

    Attributes:
        persist_directory: Directory for pattern storage.
        project_id: Identifier for project isolation.

    Example:
        >>> learner = PatternLearner(
        ...     persist_directory=".yolo/patterns",
        ...     project_id="my-project",
        ... )
        >>> result = await learner.learn_from_codebase(Path("/my/project"))
        >>> patterns = await learner.get_patterns_by_type(PatternType.NAMING_FUNCTION)
    """

    def __init__(
        self,
        persist_directory: str,
        project_id: str | None = None,
        cache_ttl: float = DEFAULT_CACHE_TTL,
    ) -> None:
        """Initialize the pattern learner.

        Args:
            persist_directory: Directory for pattern storage persistence.
            project_id: Identifier for project isolation. Defaults to "default".
            cache_ttl: Cache time-to-live in seconds. Defaults to 300 (5 minutes).
        """
        self.persist_directory = persist_directory
        self.project_id = project_id or "default"
        self._cache_ttl = cache_ttl

        # Initialize components
        self._scanner = CodebaseScanner()
        self._naming_analyzer = NamingAnalyzer()
        self._structure_analyzer = StructureAnalyzer()
        self._pattern_store = ChromaPatternStore(
            persist_directory=persist_directory,
            project_id=project_id,
        )

        # Pattern cache for frequently-queried patterns
        self._cache: dict[str, CacheEntry] = {}

        logger.debug(
            "Initialized pattern learner",
            extra={
                "project_id": self.project_id,
                "persist_directory": persist_directory,
                "cache_ttl": cache_ttl,
            },
        )

    async def learn_from_codebase(
        self,
        root_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> PatternLearningResult:
        """Learn patterns from a codebase.

        Scans the codebase, runs analyzers, and stores detected patterns.

        Args:
            root_path: Root directory of the codebase to analyze.
            progress_callback: Optional callback for progress reporting.
                Called with (current_file, total_files).

        Returns:
            PatternLearningResult with statistics about the learning process.
        """
        logger.info(
            "Starting pattern learning",
            extra={"root_path": str(root_path), "project_id": self.project_id},
        )

        # Step 1: Scan the codebase
        scan_result = await self._scanner.scan(
            root_path, progress_callback=progress_callback
        )

        if not scan_result.files:
            logger.info(
                "No files found to analyze",
                extra={"root_path": str(root_path)},
            )
            return PatternLearningResult(
                files_scanned=0,
                total_lines=0,
                patterns_found=0,
                pattern_types_found=[],
            )

        # Step 2: Run naming analyzer
        naming_patterns = await self._naming_analyzer.analyze(scan_result.files)

        # Step 3: Run structure analyzer
        structure_patterns = await self._structure_analyzer.analyze(root_path)

        # Combine all patterns
        all_patterns = naming_patterns + structure_patterns

        # Step 4: Store patterns
        pattern_types_found: list[str] = []
        for pattern in all_patterns:
            await self._pattern_store.store_pattern(pattern)
            if pattern.pattern_type.value not in pattern_types_found:
                pattern_types_found.append(pattern.pattern_type.value)

        # Invalidate cache after learning new patterns
        self.invalidate_cache()

        result = PatternLearningResult(
            files_scanned=len(scan_result.files),
            total_lines=scan_result.total_lines,
            patterns_found=len(all_patterns),
            pattern_types_found=pattern_types_found,
        )

        logger.info(
            "Pattern learning complete",
            extra={
                "files_scanned": result.files_scanned,
                "total_lines": result.total_lines,
                "patterns_found": result.patterns_found,
                "pattern_types": result.pattern_types_found,
            },
        )

        return result

    async def search_patterns(
        self,
        query: str = "",
        pattern_type: PatternType | None = None,
        k: int = 5,
    ) -> list[PatternResult]:
        """Search for patterns by semantic similarity.

        Args:
            query: Search query for semantic matching.
            pattern_type: Optional filter by pattern type.
            k: Maximum number of results to return.

        Returns:
            List of PatternResult instances ordered by similarity.
        """
        return await self._pattern_store.search_patterns(
            query=query,
            pattern_type=pattern_type,
            k=k,
        )

    async def get_patterns_by_type(
        self,
        pattern_type: PatternType,
    ) -> list[CodePattern]:
        """Get all patterns of a specific type.

        Uses cache for frequently-queried patterns to improve performance.

        Args:
            pattern_type: The type of patterns to retrieve.

        Returns:
            List of CodePattern instances of the specified type.
        """
        cache_key = f"by_type:{pattern_type.value}"

        # Check cache first
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if entry.is_valid():
                logger.debug(
                    "Cache hit for patterns by type",
                    extra={"pattern_type": pattern_type.value},
                )
                return entry.data

        # Cache miss - fetch from store
        patterns = await self._pattern_store.get_patterns_by_type(pattern_type)

        # Store in cache
        self._cache[cache_key] = CacheEntry(
            data=patterns,
            timestamp=time.time(),
            ttl=self._cache_ttl,
        )

        return patterns

    def invalidate_cache(self) -> None:
        """Invalidate all cached patterns.

        Call this after learning new patterns to ensure fresh data.
        """
        self._cache.clear()
        logger.debug("Pattern cache invalidated")

    async def get_relevant_patterns(
        self,
        context: str,
        pattern_types: list[PatternType],
        k: int = 5,
    ) -> list[CodePattern]:
        """Get patterns relevant to a given context.

        Searches for patterns matching the context string across the specified
        pattern types, and returns results ranked by confidence.

        This method is optimized for agent workflow use, combining semantic
        search with confidence-based ranking to find the most useful patterns.

        Args:
            context: Context string describing the code being generated.
            pattern_types: List of pattern types to search across.
                If empty, searches all pattern types.
            k: Maximum number of patterns to return.

        Returns:
            List of CodePattern instances sorted by confidence (descending).
        """
        all_patterns: list[CodePattern] = []

        if not pattern_types:
            # Search across all patterns
            results = await self._pattern_store.search_patterns(query=context, k=k * 2)
            all_patterns = [r.pattern for r in results]
        else:
            # Search each pattern type and combine results
            for pattern_type in pattern_types:
                type_patterns = await self._pattern_store.get_patterns_by_type(
                    pattern_type
                )
                all_patterns.extend(type_patterns)

        # Sort by confidence (descending)
        all_patterns.sort(key=lambda p: p.confidence, reverse=True)

        # Limit to k results
        return all_patterns[:k]
