"""Pattern data structures for project pattern learning.

This module defines dataclasses and enums for representing code patterns
learned from existing codebases. Patterns are used to inform agent decisions
about naming conventions, directory structure, and coding style.

Example:
    >>> from yolo_developer.memory.patterns import CodePattern, PatternType
    >>>
    >>> pattern = CodePattern(
    ...     pattern_type=PatternType.NAMING_FUNCTION,
    ...     name="function_naming",
    ...     value="snake_case",
    ...     confidence=0.95,
    ...     examples=("get_user", "process_order"),
    ... )
    >>> pattern.to_embedding_text()
    'naming_function: function_naming = snake_case. Examples: get_user, process_order'

Security Note:
    Patterns are stored per-project to ensure isolation. Pattern data
    should not contain sensitive information beyond code structure metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class PatternType(Enum):
    """Types of code patterns that can be learned from a codebase.

    Pattern types are categorized into naming conventions, structural
    patterns, import styles, and design patterns. Each type maps to
    a specific aspect of code organization that agents can learn and apply.

    Attributes:
        NAMING_FUNCTION: Function naming convention (e.g., snake_case).
        NAMING_CLASS: Class naming convention (e.g., PascalCase).
        NAMING_VARIABLE: Variable naming convention.
        NAMING_MODULE: Module/file naming convention.
        STRUCTURE_DIRECTORY: Directory organization pattern.
        STRUCTURE_FILE: File organization pattern (e.g., test file naming).
        IMPORT_STYLE: Import statement style (absolute vs relative).
        DESIGN_PATTERN: Common design patterns in use (factory, singleton, etc.).
    """

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
    """A learned code pattern from a codebase.

    Represents a single pattern detected during codebase analysis. Patterns
    are immutable (frozen) to ensure consistency once learned. Each pattern
    has a type, detected value, confidence score, and optional examples.

    Attributes:
        pattern_type: Category of the pattern (naming, structure, etc.).
        name: Human-readable pattern identifier.
        value: The detected pattern value (e.g., "snake_case", "src layout").
        confidence: Confidence score 0.0-1.0 based on consistency in codebase.
        examples: Sample instances from the codebase demonstrating the pattern.
        source_files: Paths to files where this pattern was detected.
        created_at: When the pattern was first learned (defaults to now).

    Example:
        >>> pattern = CodePattern(
        ...     pattern_type=PatternType.NAMING_FUNCTION,
        ...     name="function_naming",
        ...     value="snake_case",
        ...     confidence=0.95,
        ...     examples=("get_user", "validate_input"),
        ... )
        >>> pattern.confidence
        0.95
    """

    pattern_type: PatternType
    name: str
    value: str
    confidence: float
    examples: tuple[str, ...] = field(default_factory=tuple)
    source_files: tuple[str, ...] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding storage.

        Creates a text string suitable for vector embedding that captures
        the pattern's type, name, value, and up to 5 examples. This text
        is used for semantic similarity search in ChromaDB.

        Returns:
            Text representation of the pattern for embedding.

        Example:
            >>> pattern = CodePattern(
            ...     pattern_type=PatternType.NAMING_FUNCTION,
            ...     name="function_naming",
            ...     value="snake_case",
            ...     confidence=0.95,
            ...     examples=("get_user", "process_order"),
            ... )
            >>> pattern.to_embedding_text()
            'naming_function: function_naming = snake_case. Examples: get_user, process_order'
        """
        # Limit to 5 examples to keep embedding text concise
        examples_str = ", ".join(self.examples[:5])
        return f"{self.pattern_type.value}: {self.name} = {self.value}. Examples: {examples_str}"


@dataclass(frozen=True)
class PatternResult:
    """Result from a pattern similarity search.

    Wraps a CodePattern with its similarity score from a vector search.
    Used when querying patterns by semantic similarity.

    Attributes:
        pattern: The matched CodePattern.
        similarity: Similarity score 0.0-1.0 (higher is more similar).

    Example:
        >>> result = PatternResult(pattern=pattern, similarity=0.87)
        >>> result.similarity
        0.87
    """

    pattern: CodePattern
    similarity: float
