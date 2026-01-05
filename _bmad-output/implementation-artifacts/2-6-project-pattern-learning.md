# Story 2.6: Project Pattern Learning

Status: done

## Story

As a developer,
I want the system to learn patterns from my existing codebase,
So that generated code matches my project's conventions.

## Acceptance Criteria

1. **AC1: Codebase Analysis Initialization**
   - **Given** I have an existing codebase with Python files
   - **When** I initialize YOLO Developer with `--existing` flag (or equivalent SDK call)
   - **Then** the system scans the codebase and identifies source files
   - **And** the analysis does not modify any existing files
   - **And** a progress indicator shows files being analyzed

2. **AC2: Naming Convention Capture**
   - **Given** the codebase contains functions, classes, and variables
   - **When** pattern analysis runs
   - **Then** function naming style is detected (snake_case, camelCase, etc.)
   - **And** class naming style is detected (PascalCase, etc.)
   - **And** variable naming style is detected
   - **And** module naming patterns are captured
   - **And** these patterns are stored for later retrieval

3. **AC3: Architectural Pattern Identification**
   - **Given** the codebase has a specific structure
   - **When** pattern analysis runs
   - **Then** directory organization patterns are identified
   - **And** common file patterns are detected (e.g., `*_test.py`, `test_*.py`)
   - **And** import patterns are captured (absolute vs relative)
   - **And** common design patterns are detected (e.g., factory, singleton usage)

4. **AC4: Pattern Storage in Memory Layer**
   - **Given** patterns have been analyzed
   - **When** pattern storage completes
   - **Then** patterns are stored in ChromaDB with appropriate embeddings
   - **And** patterns can be queried by type (naming, structure, etc.)
   - **And** patterns persist across sessions
   - **And** patterns are isolated per project

5. **AC5: Pattern Retrieval for Agent Decisions**
   - **Given** patterns have been stored for a project
   - **When** an agent needs to generate code
   - **Then** relevant patterns can be retrieved via memory store
   - **And** patterns are returned with confidence scores
   - **And** multiple pattern types can be combined in queries
   - **And** retrieval is fast enough for agent workflows (<500ms)

## Tasks / Subtasks

- [x] Task 1: Define Pattern Data Structures (AC: 2, 3, 4)
  - [x] Create `src/yolo_developer/memory/patterns.py` module
  - [x] Define `CodePattern` dataclass with pattern_type, name, value, confidence, examples
  - [x] Define `NamingPattern` for captured naming conventions
  - [x] Define `StructurePattern` for directory/file organization
  - [x] Define `PatternType` enum (NAMING_FUNCTION, NAMING_CLASS, NAMING_VARIABLE, NAMING_MODULE, STRUCTURE_DIRECTORY, STRUCTURE_FILE, IMPORT_STYLE, DESIGN_PATTERN)
  - [x] Export from `memory/__init__.py`

- [x] Task 2: Implement Codebase Scanner (AC: 1)
  - [x] Create `src/yolo_developer/memory/scanner.py` module
  - [x] Implement `CodebaseScanner` class with `scan(root_path: Path) -> list[Path]`
  - [x] Support configurable file extensions (default: `.py`)
  - [x] Support configurable ignore patterns (default: `__pycache__`, `.git`, `node_modules`, `.venv`)
  - [x] Implement async file reading with tenacity retry
  - [x] Add progress callback for UI feedback
  - [x] Return list of analyzed file paths with metadata

- [x] Task 3: Implement Naming Convention Analyzer (AC: 2)
  - [x] Create `src/yolo_developer/memory/analyzers/naming.py`
  - [x] Implement `NamingAnalyzer` class using AST parsing
  - [x] Detect function naming style via regex patterns on function names
  - [x] Detect class naming style via regex patterns on class names
  - [x] Detect variable naming style via AST variable assignments
  - [x] Calculate confidence based on consistency ratio
  - [x] Return list of `NamingPattern` objects

- [x] Task 4: Implement Structure Analyzer (AC: 3)
  - [x] Create `src/yolo_developer/memory/analyzers/structure.py`
  - [x] Implement `StructureAnalyzer` class
  - [x] Detect directory organization (flat vs nested, src layout, tests location)
  - [x] Detect test file patterns (`test_*.py` vs `*_test.py`)
  - [x] Detect import style (absolute vs relative)
  - [x] Identify common module organization patterns
  - [x] Return list of `StructurePattern` objects

- [x] Task 5: Implement Pattern Storage (AC: 4)
  - [x] Add `store_pattern(pattern: CodePattern) -> str` to MemoryStore protocol
  - [x] Add `search_patterns(pattern_type: PatternType, query: str, k: int = 5) -> list[PatternResult]`
  - [x] Implement pattern embedding generation (pattern description + examples)
  - [x] Implement ChromaDB collection for patterns (separate from artifact embeddings)
  - [x] Add project isolation via collection naming
  - [x] Ensure persistence via ChromaDB persist_directory

- [x] Task 6: Implement Pattern Learning Orchestrator (AC: 1, 2, 3, 4)
  - [x] Create `src/yolo_developer/memory/learning.py` module
  - [x] Implement `PatternLearner` class coordinating scanner and analyzers
  - [x] Add `learn_from_codebase(root_path: Path) -> PatternLearningResult`
  - [x] Coordinate scanning, analysis, and storage
  - [x] Aggregate results with statistics (files scanned, patterns found)
  - [x] Support incremental learning (skip already-analyzed files)
  - [x] Log progress with structlog

- [x] Task 7: Implement Pattern Query API (AC: 5)
  - [x] Add `get_relevant_patterns(context: str, pattern_types: list[PatternType]) -> list[CodePattern]`
  - [x] Implement pattern ranking by relevance and confidence
  - [x] Add caching for frequently-queried patterns
  - [x] Ensure query response time <500ms
  - [x] Return patterns with usage examples

- [x] Task 8: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/memory/test_patterns.py`
  - [x] Test: CodePattern creation and serialization
  - [x] Test: NamingPattern detection for snake_case functions
  - [x] Test: NamingPattern detection for PascalCase classes
  - [x] Test: StructurePattern detection for test file patterns
  - [x] Test: Pattern storage in ChromaDB
  - [x] Test: Pattern retrieval by type
  - [x] Test: Pattern confidence calculation
  - [x] Create `tests/unit/memory/test_scanner.py`
  - [x] Test: File scanning with ignore patterns
  - [x] Test: Progress callback invocation
  - [x] Create `tests/unit/memory/test_analyzers.py`
  - [x] Test: Naming analyzer with mixed conventions
  - [x] Test: Structure analyzer with different layouts

- [x] Task 9: Write Integration Tests (AC: all)
  - [x] Create `tests/integration/test_pattern_learning.py`
  - [x] Test: Full pattern learning on sample codebase
  - [x] Test: Pattern persistence across sessions
  - [x] Test: Pattern retrieval after learning
  - [x] Test: Project isolation (patterns don't leak between projects)
  - [x] Test: Incremental learning updates existing patterns

## Dev Notes

### Critical Architecture Requirements

**From ADR-002 (Memory Persistence):**
- ChromaDB for vector embeddings of patterns
- Patterns should use a dedicated collection (e.g., `patterns` vs `artifacts`)
- Persistence via persist_directory already configured

**From Architecture Patterns:**
- Async-first design for all I/O operations
- Full type annotations on all functions
- Structured logging with logging/structlog module
- snake_case for all dictionary keys
- Protocol-based abstraction for pattern storage

**From Memory Store Protocol (Story 2.1):**
- Extend existing protocol with pattern-specific methods
- Use consistent async interface
- Maintain project isolation via collection naming

### Implementation Approach

**Pattern Data Structures:**
```python
# src/yolo_developer/memory/patterns.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class PatternType(Enum):
    """Types of code patterns that can be learned."""
    NAMING_FUNCTION = "naming_function"
    NAMING_CLASS = "naming_class"
    NAMING_VARIABLE = "naming_variable"
    NAMING_MODULE = "naming_module"
    STRUCTURE_DIRECTORY = "structure_directory"
    STRUCTURE_FILE = "structure_file"
    IMPORT_STYLE = "import_style"
    DESIGN_PATTERN = "design_pattern"


@dataclass(frozen=True)
class CodePattern:
    """A learned code pattern from the codebase.

    Attributes:
        pattern_type: Category of the pattern.
        name: Human-readable pattern name.
        value: The detected pattern value (e.g., "snake_case").
        confidence: Confidence score 0.0-1.0 based on consistency.
        examples: Sample instances from the codebase.
        source_files: Files where pattern was detected.
        created_at: When pattern was first learned.
    """
    pattern_type: PatternType
    name: str
    value: str
    confidence: float
    examples: tuple[str, ...] = field(default_factory=tuple)
    source_files: tuple[str, ...] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_embedding_text(self) -> str:
        """Generate text for embedding this pattern."""
        examples_str = ", ".join(self.examples[:5])
        return f"{self.pattern_type.value}: {self.name} = {self.value}. Examples: {examples_str}"


@dataclass(frozen=True)
class PatternResult:
    """Result from pattern search."""
    pattern: CodePattern
    similarity: float
```

**Codebase Scanner:**
```python
# src/yolo_developer/memory/scanner.py
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = {".py"}
DEFAULT_IGNORE = {"__pycache__", ".git", "node_modules", ".venv", ".tox", "build", "dist", ".eggs"}


@dataclass
class ScanResult:
    """Result of scanning a codebase."""
    files: list[Path]
    total_lines: int
    skipped: list[Path]


class CodebaseScanner:
    """Scans a codebase for source files."""

    def __init__(
        self,
        extensions: set[str] | None = None,
        ignore_patterns: set[str] | None = None,
    ):
        self.extensions = extensions or DEFAULT_EXTENSIONS
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE

    async def scan(
        self,
        root_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ScanResult:
        """Scan directory for source files.

        Args:
            root_path: Root directory to scan.
            progress_callback: Optional callback(files_scanned, total_files).

        Returns:
            ScanResult with found files and statistics.
        """
        files: list[Path] = []
        skipped: list[Path] = []
        total_lines = 0

        all_files = list(root_path.rglob("*"))
        total = len(all_files)

        for i, path in enumerate(all_files):
            if progress_callback:
                progress_callback(i, total)

            if not path.is_file():
                continue

            # Check ignore patterns
            if any(ignore in path.parts for ignore in self.ignore_patterns):
                skipped.append(path)
                continue

            # Check extension
            if path.suffix not in self.extensions:
                skipped.append(path)
                continue

            files.append(path)

            # Count lines
            try:
                content = await self._read_file(path)
                total_lines += len(content.splitlines())
            except Exception as e:
                logger.warning("Failed to read file", extra={"path": str(path), "error": str(e)})

        logger.info(
            "Codebase scan complete",
            extra={"files": len(files), "lines": total_lines, "skipped": len(skipped)},
        )

        return ScanResult(files=files, total_lines=total_lines, skipped=skipped)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.1, max=1.0))
    async def _read_file(self, path: Path) -> str:
        """Read file content with retry."""
        return await asyncio.to_thread(path.read_text, encoding="utf-8")
```

**Naming Analyzer (AST-based):**
```python
# src/yolo_developer/memory/analyzers/naming.py
from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

from yolo_developer.memory.patterns import CodePattern, PatternType


# Naming style regex patterns
SNAKE_CASE = re.compile(r'^[a-z][a-z0-9]*(_[a-z0-9]+)*$')
PASCAL_CASE = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
CAMEL_CASE = re.compile(r'^[a-z][a-zA-Z0-9]*$')
SCREAMING_SNAKE = re.compile(r'^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$')


def detect_style(name: str) -> str | None:
    """Detect naming style of an identifier."""
    if SNAKE_CASE.match(name):
        return "snake_case"
    if PASCAL_CASE.match(name):
        return "PascalCase"
    if CAMEL_CASE.match(name) and not SNAKE_CASE.match(name):
        return "camelCase"
    if SCREAMING_SNAKE.match(name):
        return "SCREAMING_SNAKE_CASE"
    return None


class NamingAnalyzer:
    """Analyzes naming conventions in Python source code."""

    async def analyze(self, files: list[Path]) -> list[CodePattern]:
        """Analyze naming conventions across files.

        Returns patterns for functions, classes, and variables.
        """
        function_styles: Counter[str] = Counter()
        function_examples: dict[str, list[str]] = {}
        class_styles: Counter[str] = Counter()
        class_examples: dict[str, list[str]] = {}

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        style = detect_style(node.name)
                        if style:
                            function_styles[style] += 1
                            function_examples.setdefault(style, []).append(node.name)

                    elif isinstance(node, ast.ClassDef):
                        style = detect_style(node.name)
                        if style:
                            class_styles[style] += 1
                            class_examples.setdefault(style, []).append(node.name)

            except Exception:
                continue  # Skip files that can't be parsed

        patterns: list[CodePattern] = []

        # Create function naming pattern
        if function_styles:
            dominant_style = function_styles.most_common(1)[0]
            total = sum(function_styles.values())
            confidence = dominant_style[1] / total
            patterns.append(CodePattern(
                pattern_type=PatternType.NAMING_FUNCTION,
                name="function_naming",
                value=dominant_style[0],
                confidence=confidence,
                examples=tuple(function_examples.get(dominant_style[0], [])[:10]),
            ))

        # Create class naming pattern
        if class_styles:
            dominant_style = class_styles.most_common(1)[0]
            total = sum(class_styles.values())
            confidence = dominant_style[1] / total
            patterns.append(CodePattern(
                pattern_type=PatternType.NAMING_CLASS,
                name="class_naming",
                value=dominant_style[0],
                confidence=confidence,
                examples=tuple(class_examples.get(dominant_style[0], [])[:10]),
            ))

        return patterns
```

**Pattern Storage Extension:**
```python
# Add to memory/protocol.py or create memory/pattern_store.py

class PatternStore(Protocol):
    """Protocol for pattern-specific memory operations."""

    async def store_pattern(self, pattern: CodePattern) -> str:
        """Store a code pattern and return its ID."""
        ...

    async def search_patterns(
        self,
        pattern_type: PatternType | None = None,
        query: str = "",
        k: int = 5,
    ) -> list[PatternResult]:
        """Search for patterns by type and/or semantic similarity."""
        ...

    async def get_patterns_by_type(
        self,
        pattern_type: PatternType,
    ) -> list[CodePattern]:
        """Get all patterns of a specific type."""
        ...


# ChromaDB implementation
class ChromaPatternStore:
    """ChromaDB-backed pattern storage."""

    def __init__(self, chroma_client: chromadb.ClientAPI, project_id: str):
        self.collection = chroma_client.get_or_create_collection(
            name=f"patterns_{project_id}",
            metadata={"hnsw:space": "cosine"},
        )

    async def store_pattern(self, pattern: CodePattern) -> str:
        """Store pattern with embedding."""
        pattern_id = f"{pattern.pattern_type.value}_{pattern.name}"

        await asyncio.to_thread(
            self.collection.upsert,
            ids=[pattern_id],
            documents=[pattern.to_embedding_text()],
            metadatas=[{
                "pattern_type": pattern.pattern_type.value,
                "name": pattern.name,
                "value": pattern.value,
                "confidence": pattern.confidence,
                "examples": ",".join(pattern.examples[:5]),
            }],
        )

        return pattern_id
```

### Project Structure Notes

**New/Modified Module Locations:**
```
src/yolo_developer/memory/
├── __init__.py           # Add pattern exports
├── protocol.py           # Existing: Add pattern methods
├── vector.py             # Existing: ChromaDB implementation
├── patterns.py           # NEW: Pattern data structures
├── scanner.py            # NEW: Codebase scanner
├── learning.py           # NEW: Pattern learning orchestrator
└── analyzers/
    ├── __init__.py       # NEW: Analyzer exports
    ├── naming.py         # NEW: Naming convention analyzer
    └── structure.py      # NEW: Structure analyzer
```

**Test Locations:**
```
tests/unit/memory/
├── test_patterns.py      # NEW: Pattern data structure tests
├── test_scanner.py       # NEW: Scanner tests
└── test_analyzers.py     # NEW: Analyzer tests

tests/integration/
└── test_pattern_learning.py  # NEW: End-to-end pattern learning
```

### Previous Story Learnings (from Story 2.5)

1. **LangChain serialization** - Use messages_to_dict/messages_from_dict for message serialization
2. **Frozen dataclasses** - Use frozen=True for immutable pattern data
3. **Tenacity retry** - Apply retry decorator for file I/O operations
4. **Atomic file writes** - Use temp file + rename for safe writes
5. **Project isolation** - Use project-specific collection names in ChromaDB
6. **Type validation** - Validate types with isinstance before processing
7. **Structured logging** - Use extra dict for structured log data
8. **asyncio.to_thread** - Use for blocking I/O in async context

### Dependencies on Previous Stories

- **Story 2.1** (Memory Store Protocol): Pattern storage extends the MemoryStore protocol
- **Story 2.2** (ChromaDB Vector Storage): Pattern embeddings use ChromaDB
- **Story 2.5** (Session Persistence): Patterns persist across sessions via ChromaDB

### Performance Considerations

- **File scanning**: Use async file reading with thread pool
- **AST parsing**: Parse files in parallel using asyncio.gather
- **Pattern queries**: Cache frequently-accessed patterns
- **Embedding generation**: Batch pattern embeddings if many patterns

### Testing Approach

**Unit Tests (isolated components):**
- Test CodePattern creation and to_embedding_text()
- Test NamingAnalyzer with sample AST
- Test StructureAnalyzer with mock directory structure
- Test CodebaseScanner file filtering
- Test ChromaPatternStore CRUD operations

**Integration Tests (component interactions):**
- Test full pattern learning pipeline on test codebase
- Test pattern persistence after session restart
- Test pattern retrieval returns relevant results
- Test project isolation (two projects, separate patterns)

### References

- [Source: architecture.md#ADR-002] - ChromaDB for vector storage
- [Source: epics.md#Story-2.6] - Project Pattern Learning requirements
- [Source: prd.md#FR32] - System can learn project-specific patterns
- [Python AST Documentation](https://docs.python.org/3/library/ast.html)
- [ChromaDB Collections](https://docs.trychroma.com/guides)

---

## Dev Agent Record

### File List

**New Files Created:**
- `src/yolo_developer/memory/patterns.py` - Pattern data structures (CodePattern, PatternResult, PatternType)
- `src/yolo_developer/memory/scanner.py` - Codebase scanner with async file reading
- `src/yolo_developer/memory/pattern_store.py` - ChromaDB-backed pattern storage
- `src/yolo_developer/memory/learning.py` - Pattern learning orchestrator with caching
- `src/yolo_developer/memory/analyzers/__init__.py` - Analyzer package exports
- `src/yolo_developer/memory/analyzers/naming.py` - Naming convention analyzer (AST-based)
- `src/yolo_developer/memory/analyzers/structure.py` - Structure analyzer with design pattern detection
- `tests/unit/memory/test_patterns.py` - Unit tests for pattern data structures
- `tests/unit/memory/test_scanner.py` - Unit tests for codebase scanner
- `tests/unit/memory/test_learning.py` - Unit tests for pattern learner
- `tests/integration/test_pattern_learning.py` - Integration tests for full pipeline

**Modified Files:**
- `src/yolo_developer/memory/__init__.py` - Added pattern-related exports
- `src/yolo_developer/memory/protocol.py` - Added pattern methods to MemoryStore protocol

### Implementation Notes

- Used AST parsing for naming convention detection (functions, classes, variables, modules)
- Added design pattern detection via class name suffix matching (Service, Repository, Factory, Handler, etc.)
- Implemented caching with configurable TTL for frequently-queried patterns
- Project isolation via project-specific ChromaDB collection names
