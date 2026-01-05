"""Structure analyzer for code pattern learning.

This module provides the StructureAnalyzer class for detecting structural
patterns in Python codebases including directory organization, test file
patterns, and import styles.

Example:
    >>> from yolo_developer.memory.analyzers.structure import StructureAnalyzer
    >>> from pathlib import Path
    >>>
    >>> analyzer = StructureAnalyzer()
    >>> patterns = await analyzer.analyze(Path("/path/to/project"))
    >>> for pattern in patterns:
    ...     print(f"{pattern.pattern_type}: {pattern.value}")
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from yolo_developer.memory.patterns import CodePattern, PatternType

logger = logging.getLogger(__name__)

# Maximum number of examples to include in a pattern
MAX_EXAMPLES = 10

# Design pattern indicators (suffix -> pattern name)
DESIGN_PATTERN_SUFFIXES = {
    "Service": "service_pattern",
    "Repository": "repository_pattern",
    "Store": "repository_pattern",
    "Factory": "factory_pattern",
    "Handler": "handler_pattern",
    "Controller": "controller_pattern",
    "Manager": "manager_pattern",
    "Provider": "provider_pattern",
    "Adapter": "adapter_pattern",
    "Strategy": "strategy_pattern",
}


class StructureAnalyzer:
    """Analyzes structural patterns in Python codebases.

    Detects directory organization, test file naming patterns,
    and import styles used in the codebase.

    Example:
        >>> analyzer = StructureAnalyzer()
        >>> patterns = await analyzer.analyze(Path("/my/project"))
        >>> dir_patterns = [p for p in patterns if p.pattern_type == PatternType.STRUCTURE_DIRECTORY]
    """

    async def analyze(self, root_path: Path) -> list[CodePattern]:
        """Analyze structural patterns in the given directory.

        Args:
            root_path: Root directory of the codebase to analyze.

        Returns:
            List of CodePattern instances describing detected structural patterns.
        """
        if not root_path.exists():
            return []

        patterns: list[CodePattern] = []

        # Analyze directory layout
        layout_pattern = self._analyze_directory_layout(root_path)
        if layout_pattern:
            patterns.append(layout_pattern)

        # Analyze test directory location
        test_dir_pattern = self._analyze_test_directory(root_path)
        if test_dir_pattern:
            patterns.append(test_dir_pattern)

        # Analyze test file patterns
        test_file_pattern = self._analyze_test_file_pattern(root_path)
        if test_file_pattern:
            patterns.append(test_file_pattern)

        # Analyze import style
        import_pattern = await self._analyze_import_style(root_path)
        if import_pattern:
            patterns.append(import_pattern)

        # Analyze design patterns in use
        design_patterns = await self._analyze_design_patterns(root_path)
        patterns.extend(design_patterns)

        return patterns

    def _analyze_directory_layout(self, root_path: Path) -> CodePattern | None:
        """Detect if the project uses src/ layout or flat layout.

        Args:
            root_path: Root directory to analyze.

        Returns:
            CodePattern describing the directory layout, or None if no pattern detected.
        """
        # Check for src/ layout
        src_dir = root_path / "src"
        has_src = src_dir.is_dir()

        # Check for package directories (contain __init__.py)
        packages = []
        for item in root_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                init_file = item / "__init__.py"
                if init_file.exists():
                    packages.append(item.name)

        if not packages and not has_src:
            return None

        if has_src:
            # Check if src contains packages
            src_packages = []
            for item in src_dir.iterdir():
                if item.is_dir():
                    init_file = item / "__init__.py"
                    if init_file.exists():
                        src_packages.append(item.name)

            if src_packages:
                return CodePattern(
                    pattern_type=PatternType.STRUCTURE_DIRECTORY,
                    name="directory_layout",
                    value="src_layout",
                    confidence=1.0,
                    examples=tuple(src_packages[:MAX_EXAMPLES]),
                    source_files=("src/",),
                )

        # Flat layout - packages directly in root
        if packages:
            return CodePattern(
                pattern_type=PatternType.STRUCTURE_DIRECTORY,
                name="directory_layout",
                value="flat_layout",
                confidence=1.0,
                examples=tuple(packages[:MAX_EXAMPLES]),
                source_files=(str(root_path),),
            )

        return None

    def _analyze_test_directory(self, root_path: Path) -> CodePattern | None:
        """Detect the test directory location.

        Args:
            root_path: Root directory to analyze.

        Returns:
            CodePattern describing the test directory, or None if not found.
        """
        # Common test directory names
        test_dirs = ["tests", "test"]

        for dir_name in test_dirs:
            test_dir = root_path / dir_name
            if test_dir.is_dir():
                # Count test files in the directory
                test_files = list(test_dir.rglob("*.py"))
                test_file_names = [f.name for f in test_files if "test" in f.name.lower()]

                if test_files:
                    return CodePattern(
                        pattern_type=PatternType.STRUCTURE_DIRECTORY,
                        name="test_directory",
                        value=dir_name,
                        confidence=1.0,
                        examples=tuple(test_file_names[:MAX_EXAMPLES]),
                        source_files=(str(test_dir),),
                    )

        return None

    def _analyze_test_file_pattern(self, root_path: Path) -> CodePattern | None:
        """Detect test file naming pattern (test_*.py vs *_test.py).

        Args:
            root_path: Root directory to analyze.

        Returns:
            CodePattern describing the test file pattern, or None if not found.
        """
        # Find all test files
        test_prefix_files: list[str] = []
        test_suffix_files: list[str] = []

        for py_file in root_path.rglob("*.py"):
            name = py_file.stem  # filename without extension
            if name.startswith("test_"):
                test_prefix_files.append(py_file.name)
            elif name.endswith("_test"):
                test_suffix_files.append(py_file.name)

        total = len(test_prefix_files) + len(test_suffix_files)
        if total == 0:
            return None

        # Determine dominant pattern
        if len(test_prefix_files) >= len(test_suffix_files):
            dominant = "test_prefix"
            count = len(test_prefix_files)
            examples = test_prefix_files
        else:
            dominant = "test_suffix"
            count = len(test_suffix_files)
            examples = test_suffix_files

        confidence = count / total if total > 0 else 0.0

        return CodePattern(
            pattern_type=PatternType.STRUCTURE_FILE,
            name="test_file_pattern",
            value=dominant,
            confidence=confidence,
            examples=tuple(examples[:MAX_EXAMPLES]),
            source_files=(str(root_path),),
        )

    async def _analyze_import_style(self, root_path: Path) -> CodePattern | None:
        """Detect import style (absolute vs relative imports).

        Args:
            root_path: Root directory to analyze.

        Returns:
            CodePattern describing the import style, or None if not detected.
        """
        absolute_imports: list[str] = []
        relative_imports: list[str] = []

        for py_file in root_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content, filename=str(py_file))

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.level > 0:
                            # Relative import (from . import X, from ..x import Y)
                            module = node.module or ""
                            relative_imports.append(f"{'.' * node.level}{module}")
                        elif node.module:
                            # Absolute import from a module (from x import y)
                            absolute_imports.append(node.module)
                    elif isinstance(node, ast.Import):
                        # import x, import x.y - always absolute
                        for alias in node.names:
                            absolute_imports.append(alias.name)

            except SyntaxError:
                logger.warning(
                    "Failed to parse file for import analysis",
                    extra={"path": str(py_file)},
                )
            except Exception as e:
                logger.warning(
                    "Failed to read file for import analysis",
                    extra={"path": str(py_file), "error": str(e)},
                )

        # Filter out standard library and third-party imports
        # Focus on relative vs absolute from the project
        total = len(absolute_imports) + len(relative_imports)
        if total == 0:
            return None

        # Determine dominant style
        if len(relative_imports) > len(absolute_imports):
            dominant = "relative"
            count = len(relative_imports)
            examples = relative_imports
        else:
            dominant = "absolute"
            count = len(absolute_imports)
            examples = absolute_imports

        confidence = count / total if total > 0 else 0.0

        return CodePattern(
            pattern_type=PatternType.IMPORT_STYLE,
            name="import_style",
            value=dominant,
            confidence=confidence,
            examples=tuple(examples[:MAX_EXAMPLES]),
            source_files=(str(root_path),),
        )

    async def _analyze_design_patterns(self, root_path: Path) -> list[CodePattern]:
        """Detect common design patterns based on class naming conventions.

        Identifies patterns like Service, Repository, Factory, Handler, etc.
        based on class name suffixes commonly used in Python codebases.

        Args:
            root_path: Root directory to analyze.

        Returns:
            List of CodePattern instances for detected design patterns.
        """
        # Collect classes by design pattern type
        pattern_classes: dict[str, list[str]] = {}
        source_files: list[str] = []

        for py_file in root_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content, filename=str(py_file))

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name
                        # Check if class name ends with a known pattern suffix
                        for suffix, pattern_name in DESIGN_PATTERN_SUFFIXES.items():
                            if class_name.endswith(suffix) and class_name != suffix:
                                if pattern_name not in pattern_classes:
                                    pattern_classes[pattern_name] = []
                                pattern_classes[pattern_name].append(class_name)
                                if str(py_file) not in source_files:
                                    source_files.append(str(py_file))
                                break  # Only match first pattern

            except SyntaxError:
                logger.warning(
                    "Failed to parse file for design pattern analysis",
                    extra={"path": str(py_file)},
                )
            except Exception as e:
                logger.warning(
                    "Failed to read file for design pattern analysis",
                    extra={"path": str(py_file), "error": str(e)},
                )

        # Create CodePattern for each detected design pattern
        patterns: list[CodePattern] = []
        for pattern_name, classes in pattern_classes.items():
            if classes:
                # Higher confidence if more classes follow the pattern
                confidence = min(1.0, len(classes) * 0.2 + 0.4)
                patterns.append(
                    CodePattern(
                        pattern_type=PatternType.DESIGN_PATTERN,
                        name=pattern_name,
                        value=pattern_name.replace("_pattern", ""),
                        confidence=confidence,
                        examples=tuple(classes[:MAX_EXAMPLES]),
                        source_files=tuple(source_files[:MAX_EXAMPLES]),
                    )
                )

        return patterns
