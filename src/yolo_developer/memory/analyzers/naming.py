"""Naming convention analyzer for code pattern learning.

This module provides the NamingAnalyzer class for detecting naming conventions
used in Python codebases. It uses AST parsing to extract function, class, and
variable names, then analyzes their naming style patterns.

Example:
    >>> from yolo_developer.memory.analyzers.naming import NamingAnalyzer
    >>> from pathlib import Path
    >>>
    >>> analyzer = NamingAnalyzer()
    >>> patterns = await analyzer.analyze([Path("src/main.py")])
    >>> for pattern in patterns:
    ...     print(f"{pattern.pattern_type}: {pattern.value}")
"""

from __future__ import annotations

import ast
import logging
import re
from collections import Counter
from pathlib import Path

from yolo_developer.memory.patterns import CodePattern, PatternType

logger = logging.getLogger(__name__)

# Regex patterns for naming style detection
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
PASCAL_CASE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
CAMEL_CASE = re.compile(r"^[a-z][a-zA-Z0-9]*$")
SCREAMING_SNAKE = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$")

# Maximum number of examples to include in a pattern
MAX_EXAMPLES = 10


def detect_style(name: str) -> str | None:
    """Detect the naming style of a given identifier.

    Args:
        name: The identifier name to analyze.

    Returns:
        The naming style as a string, or None if unknown.
        Possible values: "snake_case", "PascalCase", "camelCase", "SCREAMING_SNAKE_CASE"
    """
    if not name:
        return None

    # Skip private/dunder names
    if name.startswith("_"):
        return None

    # Check for SCREAMING_SNAKE_CASE first (all caps with underscores)
    if SCREAMING_SNAKE.match(name):
        return "SCREAMING_SNAKE_CASE"

    # Check for PascalCase (starts with uppercase)
    if PASCAL_CASE.match(name) and name[0].isupper():
        # Ensure it has at least one lowercase or is a single char
        if len(name) == 1 or any(c.islower() for c in name):
            return "PascalCase"
        # All caps without underscore could be SCREAMING_SNAKE
        return "SCREAMING_SNAKE_CASE"

    # Check for snake_case
    if SNAKE_CASE.match(name):
        return "snake_case"

    # Check for camelCase (starts with lowercase, has uppercase)
    if CAMEL_CASE.match(name) and name[0].islower():
        if any(c.isupper() for c in name):
            return "camelCase"
        # All lowercase without underscore is still snake_case
        return "snake_case"

    return None


class NamingAnalyzer:
    """Analyzes naming conventions in Python source files.

    Uses AST parsing to extract function, class, and variable names,
    then determines the dominant naming patterns used in the codebase.

    Example:
        >>> analyzer = NamingAnalyzer()
        >>> patterns = await analyzer.analyze([Path("src/module.py")])
        >>> function_patterns = [p for p in patterns if p.pattern_type == PatternType.NAMING_FUNCTION]
    """

    async def analyze(self, files: list[Path]) -> list[CodePattern]:
        """Analyze naming conventions in the given files.

        Args:
            files: List of Python file paths to analyze.

        Returns:
            List of CodePattern instances describing detected naming conventions.
        """
        if not files:
            return []

        # Collect names by category
        function_names: list[str] = []
        class_names: list[str] = []
        variable_names: list[str] = []
        module_names: list[str] = []
        source_files: list[str] = []

        for file_path in files:
            # Collect module/file names (excluding __init__ and test_ prefixed)
            stem = file_path.stem
            if stem != "__init__" and not stem.startswith("test_"):
                module_names.append(stem)
            try:
                content = file_path.read_text(encoding="utf-8")
                tree = ast.parse(content, filename=str(file_path))
                source_files.append(str(file_path))

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                        # Skip private and dunder methods
                        if not node.name.startswith("_"):
                            function_names.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        class_names.append(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                if not target.id.startswith("_"):
                                    variable_names.append(target.id)

            except SyntaxError as e:
                logger.warning(
                    "Failed to parse file",
                    extra={"path": str(file_path), "error": str(e)},
                )
            except Exception as e:
                logger.warning(
                    "Failed to read file",
                    extra={"path": str(file_path), "error": str(e)},
                )

        patterns: list[CodePattern] = []

        # Analyze function naming
        if function_names:
            pattern = self._create_pattern(
                names=function_names,
                pattern_type=PatternType.NAMING_FUNCTION,
                pattern_name="function_naming",
                source_files=source_files,
            )
            if pattern:
                patterns.append(pattern)

        # Analyze class naming
        if class_names:
            pattern = self._create_pattern(
                names=class_names,
                pattern_type=PatternType.NAMING_CLASS,
                pattern_name="class_naming",
                source_files=source_files,
            )
            if pattern:
                patterns.append(pattern)

        # Analyze variable naming
        if variable_names:
            pattern = self._create_pattern(
                names=variable_names,
                pattern_type=PatternType.NAMING_VARIABLE,
                pattern_name="variable_naming",
                source_files=source_files,
            )
            if pattern:
                patterns.append(pattern)

        # Analyze module/file naming
        if module_names:
            pattern = self._create_pattern(
                names=module_names,
                pattern_type=PatternType.NAMING_MODULE,
                pattern_name="module_naming",
                source_files=source_files,
            )
            if pattern:
                patterns.append(pattern)

        return patterns

    def _create_pattern(
        self,
        names: list[str],
        pattern_type: PatternType,
        pattern_name: str,
        source_files: list[str],
    ) -> CodePattern | None:
        """Create a CodePattern from a list of names.

        Determines the dominant naming style and calculates confidence
        based on consistency of the naming convention.

        Args:
            names: List of identifier names to analyze.
            pattern_type: The type of pattern (function, class, variable).
            pattern_name: Human-readable name for the pattern.
            source_files: List of source file paths.

        Returns:
            CodePattern if a dominant style is detected, None otherwise.
        """
        if not names:
            return None

        # Detect style for each name
        styles: list[str] = []
        name_by_style: dict[str, list[str]] = {}

        for name in names:
            style = detect_style(name)
            if style:
                styles.append(style)
                if style not in name_by_style:
                    name_by_style[style] = []
                name_by_style[style].append(name)

        if not styles:
            return None

        # Find dominant style
        style_counts = Counter(styles)
        dominant_style, count = style_counts.most_common(1)[0]
        confidence = count / len(styles)

        # Get examples for the dominant style
        examples = name_by_style.get(dominant_style, [])[:MAX_EXAMPLES]

        return CodePattern(
            pattern_type=pattern_type,
            name=pattern_name,
            value=dominant_style,
            confidence=confidence,
            examples=tuple(examples),
            source_files=tuple(source_files[:MAX_EXAMPLES]),
        )
