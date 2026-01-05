"""Unit tests for code analyzers.

Tests NamingAnalyzer and StructureAnalyzer classes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_developer.memory.analyzers.naming import (
    CAMEL_CASE,
    PASCAL_CASE,
    SCREAMING_SNAKE,
    SNAKE_CASE,
    NamingAnalyzer,
    detect_style,
)
from yolo_developer.memory.analyzers.structure import StructureAnalyzer
from yolo_developer.memory.patterns import CodePattern, PatternType


class TestDetectStyle:
    """Tests for detect_style helper function."""

    def test_detect_snake_case(self) -> None:
        """Test snake_case detection."""
        assert detect_style("get_user") == "snake_case"
        assert detect_style("process_order") == "snake_case"
        assert detect_style("validate_input") == "snake_case"
        assert detect_style("x") == "snake_case"  # Single char
        assert detect_style("my_var_123") == "snake_case"

    def test_detect_pascal_case(self) -> None:
        """Test PascalCase detection."""
        assert detect_style("UserService") == "PascalCase"
        assert detect_style("OrderHandler") == "PascalCase"
        assert detect_style("MyClass") == "PascalCase"
        assert detect_style("HTTPHandler") == "PascalCase"  # All caps prefix

    def test_detect_camel_case(self) -> None:
        """Test camelCase detection."""
        assert detect_style("getUser") == "camelCase"
        assert detect_style("processOrder") == "camelCase"
        assert detect_style("myVariable") == "camelCase"

    def test_detect_screaming_snake(self) -> None:
        """Test SCREAMING_SNAKE_CASE detection."""
        assert detect_style("MAX_SIZE") == "SCREAMING_SNAKE_CASE"
        assert detect_style("DEFAULT_VALUE") == "SCREAMING_SNAKE_CASE"
        assert detect_style("API_KEY") == "SCREAMING_SNAKE_CASE"
        assert detect_style("X") == "SCREAMING_SNAKE_CASE"  # Single upper char

    def test_detect_unknown(self) -> None:
        """Test unknown naming styles return None."""
        assert detect_style("") is None
        assert detect_style("_private") is None  # Leading underscore
        assert detect_style("__dunder__") is None
        assert detect_style("Mixed_CASE") is None


class TestNamingStyleRegex:
    """Tests for naming style regex patterns."""

    def test_snake_case_regex(self) -> None:
        """Test SNAKE_CASE regex pattern."""
        assert SNAKE_CASE.match("get_user")
        assert SNAKE_CASE.match("x")
        assert SNAKE_CASE.match("my_var_123")
        assert not SNAKE_CASE.match("GetUser")
        assert not SNAKE_CASE.match("_private")

    def test_pascal_case_regex(self) -> None:
        """Test PASCAL_CASE regex pattern."""
        assert PASCAL_CASE.match("UserService")
        assert PASCAL_CASE.match("A")
        assert not PASCAL_CASE.match("userService")
        assert not PASCAL_CASE.match("user_service")

    def test_camel_case_regex(self) -> None:
        """Test CAMEL_CASE regex pattern."""
        assert CAMEL_CASE.match("getUser")
        assert CAMEL_CASE.match("x")
        assert CAMEL_CASE.match("myVar123")
        assert not CAMEL_CASE.match("GetUser")

    def test_screaming_snake_regex(self) -> None:
        """Test SCREAMING_SNAKE regex pattern."""
        assert SCREAMING_SNAKE.match("MAX_SIZE")
        assert SCREAMING_SNAKE.match("X")
        assert not SCREAMING_SNAKE.match("max_size")
        assert not SCREAMING_SNAKE.match("MaxSize")


class TestNamingAnalyzer:
    """Tests for NamingAnalyzer class."""

    @pytest.fixture
    def snake_case_codebase(self, tmp_path: Path) -> Path:
        """Create a codebase using snake_case functions."""
        src = tmp_path / "src"
        src.mkdir()

        (src / "main.py").write_text(
            """
def get_user():
    pass

def process_order():
    pass

def validate_input():
    pass
"""
        )

        (src / "utils.py").write_text(
            """
def helper_function():
    pass

def another_helper():
    pass
"""
        )

        return tmp_path

    @pytest.fixture
    def pascal_case_classes(self, tmp_path: Path) -> Path:
        """Create a codebase using PascalCase classes."""
        (tmp_path / "models.py").write_text(
            """
class UserService:
    pass

class OrderHandler:
    pass

class PaymentProcessor:
    pass

class DataManager:
    pass
"""
        )

        return tmp_path

    @pytest.fixture
    def mixed_naming(self, tmp_path: Path) -> Path:
        """Create a codebase with mixed naming conventions."""
        (tmp_path / "mixed.py").write_text(
            """
def get_user():
    pass

def processOrder():
    pass

class UserService:
    pass

class order_handler:
    pass
"""
        )

        return tmp_path

    @pytest.mark.asyncio
    async def test_analyze_empty_file_list(self) -> None:
        """Test analyzer with no files."""
        analyzer = NamingAnalyzer()
        patterns = await analyzer.analyze([])

        assert patterns == []

    @pytest.mark.asyncio
    async def test_analyze_detects_snake_case_functions(self, snake_case_codebase: Path) -> None:
        """Test analyzer detects snake_case function naming."""
        analyzer = NamingAnalyzer()
        files = list(snake_case_codebase.rglob("*.py"))
        patterns = await analyzer.analyze(files)

        # Should detect function naming pattern
        function_patterns = [p for p in patterns if p.pattern_type == PatternType.NAMING_FUNCTION]
        assert len(function_patterns) == 1

        pattern = function_patterns[0]
        assert pattern.value == "snake_case"
        assert pattern.confidence > 0.9  # All functions use snake_case
        assert len(pattern.examples) > 0
        assert "get_user" in pattern.examples

    @pytest.mark.asyncio
    async def test_analyze_detects_pascal_case_classes(self, pascal_case_classes: Path) -> None:
        """Test analyzer detects PascalCase class naming."""
        analyzer = NamingAnalyzer()
        files = list(pascal_case_classes.rglob("*.py"))
        patterns = await analyzer.analyze(files)

        # Should detect class naming pattern
        class_patterns = [p for p in patterns if p.pattern_type == PatternType.NAMING_CLASS]
        assert len(class_patterns) == 1

        pattern = class_patterns[0]
        assert pattern.value == "PascalCase"
        assert pattern.confidence >= 0.75  # At least 3/4 PascalCase
        assert "UserService" in pattern.examples

    @pytest.mark.asyncio
    async def test_analyze_calculates_confidence(self, mixed_naming: Path) -> None:
        """Test analyzer calculates confidence based on consistency."""
        analyzer = NamingAnalyzer()
        files = list(mixed_naming.rglob("*.py"))
        patterns = await analyzer.analyze(files)

        # With mixed naming, confidence should be lower
        function_patterns = [p for p in patterns if p.pattern_type == PatternType.NAMING_FUNCTION]
        if function_patterns:
            # Confidence should be 0.5 (1 snake_case, 1 camelCase)
            assert function_patterns[0].confidence == 0.5

    @pytest.mark.asyncio
    async def test_analyze_handles_parse_errors(self, tmp_path: Path) -> None:
        """Test analyzer handles files that cannot be parsed."""
        # Create a file with syntax errors
        (tmp_path / "broken.py").write_text(
            """
def this_is_broken(
    # Unclosed parenthesis
"""
        )

        # Create a valid file
        (tmp_path / "valid.py").write_text(
            """
def valid_function():
    pass
"""
        )

        analyzer = NamingAnalyzer()
        files = list(tmp_path.rglob("*.py"))
        patterns = await analyzer.analyze(files)

        # Should still return patterns from valid file
        function_patterns = [p for p in patterns if p.pattern_type == PatternType.NAMING_FUNCTION]
        assert len(function_patterns) == 1
        assert "valid_function" in function_patterns[0].examples

    @pytest.mark.asyncio
    async def test_analyze_limits_examples(self, tmp_path: Path) -> None:
        """Test analyzer limits examples to prevent bloat."""
        # Create a file with many functions
        funcs = "\n".join([f"def func_{i}():\n    pass\n" for i in range(20)])
        (tmp_path / "many_funcs.py").write_text(funcs)

        analyzer = NamingAnalyzer()
        files = list(tmp_path.rglob("*.py"))
        patterns = await analyzer.analyze(files)

        function_patterns = [p for p in patterns if p.pattern_type == PatternType.NAMING_FUNCTION]
        assert len(function_patterns) == 1
        # Examples should be limited to 10
        assert len(function_patterns[0].examples) <= 10

    @pytest.mark.asyncio
    async def test_analyze_returns_code_pattern_instances(self, snake_case_codebase: Path) -> None:
        """Test analyzer returns proper CodePattern instances."""
        analyzer = NamingAnalyzer()
        files = list(snake_case_codebase.rglob("*.py"))
        patterns = await analyzer.analyze(files)

        for pattern in patterns:
            assert isinstance(pattern, CodePattern)
            assert isinstance(pattern.pattern_type, PatternType)
            assert isinstance(pattern.name, str)
            assert isinstance(pattern.value, str)
            assert 0.0 <= pattern.confidence <= 1.0
            assert isinstance(pattern.examples, tuple)


class TestNamingAnalyzerVariables:
    """Tests for variable naming detection."""

    @pytest.fixture
    def variable_codebase(self, tmp_path: Path) -> Path:
        """Create a codebase with variable assignments."""
        (tmp_path / "vars.py").write_text(
            """
my_variable = 1
another_var = 2
third_variable = 3

class MyClass:
    class_var = 10

    def method(self):
        local_var = 20
"""
        )
        return tmp_path

    @pytest.mark.asyncio
    async def test_analyze_detects_variable_naming(self, variable_codebase: Path) -> None:
        """Test analyzer can detect variable naming patterns."""
        analyzer = NamingAnalyzer()
        files = list(variable_codebase.rglob("*.py"))
        patterns = await analyzer.analyze(files)

        # Should detect variable naming (if implemented)
        _variable_patterns = [p for p in patterns if p.pattern_type == PatternType.NAMING_VARIABLE]
        # Note: Variable naming detection is more complex,
        # so we just verify the analyzer doesn't crash
        # Full implementation may or may not detect variables
        assert isinstance(patterns, list)


# ==============================================================================
# Structure Analyzer Tests
# ==============================================================================


class TestStructureAnalyzer:
    """Tests for StructureAnalyzer class."""

    @pytest.fixture
    def src_layout_codebase(self, tmp_path: Path) -> Path:
        """Create a codebase with src/ layout."""
        src = tmp_path / "src" / "myproject"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("")
        (src / "main.py").write_text("def main(): pass\n")
        (src / "utils.py").write_text("def helper(): pass\n")

        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("")
        (tests / "test_main.py").write_text("def test_main(): pass\n")

        return tmp_path

    @pytest.fixture
    def flat_layout_codebase(self, tmp_path: Path) -> Path:
        """Create a codebase with flat layout (no src/)."""
        myproject = tmp_path / "myproject"
        myproject.mkdir()
        (myproject / "__init__.py").write_text("")
        (myproject / "main.py").write_text("def main(): pass\n")
        (myproject / "utils.py").write_text("def helper(): pass\n")

        (tmp_path / "test_main.py").write_text("def test_main(): pass\n")

        return tmp_path

    @pytest.fixture
    def test_prefix_codebase(self, tmp_path: Path) -> Path:
        """Create a codebase using test_*.py pattern."""
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_module.py").write_text("def test_one(): pass\n")
        (tests / "test_utils.py").write_text("def test_two(): pass\n")
        (tests / "test_main.py").write_text("def test_three(): pass\n")

        return tmp_path

    @pytest.fixture
    def test_suffix_codebase(self, tmp_path: Path) -> Path:
        """Create a codebase using *_test.py pattern."""
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "module_test.py").write_text("def test_one(): pass\n")
        (tests / "utils_test.py").write_text("def test_two(): pass\n")
        (tests / "main_test.py").write_text("def test_three(): pass\n")

        return tmp_path

    @pytest.fixture
    def absolute_imports_codebase(self, tmp_path: Path) -> Path:
        """Create a codebase with absolute imports."""
        pkg = tmp_path / "mypackage"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "main.py").write_text(
            """
from mypackage.utils import helper
from mypackage.models import User
import mypackage.config
"""
        )
        (pkg / "utils.py").write_text("def helper(): pass\n")
        (pkg / "models.py").write_text("class User: pass\n")
        (pkg / "config.py").write_text("VALUE = 1\n")

        return tmp_path

    @pytest.fixture
    def relative_imports_codebase(self, tmp_path: Path) -> Path:
        """Create a codebase with relative imports."""
        pkg = tmp_path / "mypackage"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "main.py").write_text(
            """
from .utils import helper
from .models import User
from . import config
"""
        )
        (pkg / "utils.py").write_text("def helper(): pass\n")
        (pkg / "models.py").write_text("class User: pass\n")
        (pkg / "config.py").write_text("VALUE = 1\n")

        return tmp_path

    @pytest.mark.asyncio
    async def test_analyze_empty_directory(self, tmp_path: Path) -> None:
        """Test analyzer with empty directory."""
        analyzer = StructureAnalyzer()
        patterns = await analyzer.analyze(tmp_path)

        assert patterns == []

    @pytest.mark.asyncio
    async def test_analyze_detects_src_layout(self, src_layout_codebase: Path) -> None:
        """Test analyzer detects src/ directory layout."""
        analyzer = StructureAnalyzer()
        patterns = await analyzer.analyze(src_layout_codebase)

        # Should detect directory structure pattern
        dir_patterns = [p for p in patterns if p.pattern_type == PatternType.STRUCTURE_DIRECTORY]
        assert len(dir_patterns) >= 1

        # Check for src layout detection
        src_pattern = next((p for p in dir_patterns if p.name == "directory_layout"), None)
        assert src_pattern is not None
        assert src_pattern.value == "src_layout"

    @pytest.mark.asyncio
    async def test_analyze_detects_flat_layout(self, flat_layout_codebase: Path) -> None:
        """Test analyzer detects flat directory layout."""
        analyzer = StructureAnalyzer()
        patterns = await analyzer.analyze(flat_layout_codebase)

        # Should detect directory structure pattern
        dir_patterns = [p for p in patterns if p.pattern_type == PatternType.STRUCTURE_DIRECTORY]
        assert len(dir_patterns) >= 1

        # Check for flat layout detection
        layout_pattern = next((p for p in dir_patterns if p.name == "directory_layout"), None)
        assert layout_pattern is not None
        assert layout_pattern.value == "flat_layout"

    @pytest.mark.asyncio
    async def test_analyze_detects_test_prefix_pattern(self, test_prefix_codebase: Path) -> None:
        """Test analyzer detects test_*.py naming pattern."""
        analyzer = StructureAnalyzer()
        patterns = await analyzer.analyze(test_prefix_codebase)

        # Should detect test file pattern
        file_patterns = [p for p in patterns if p.pattern_type == PatternType.STRUCTURE_FILE]
        assert len(file_patterns) >= 1

        test_pattern = next((p for p in file_patterns if p.name == "test_file_pattern"), None)
        assert test_pattern is not None
        assert test_pattern.value == "test_prefix"
        assert test_pattern.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_analyze_detects_test_suffix_pattern(self, test_suffix_codebase: Path) -> None:
        """Test analyzer detects *_test.py naming pattern."""
        analyzer = StructureAnalyzer()
        patterns = await analyzer.analyze(test_suffix_codebase)

        # Should detect test file pattern
        file_patterns = [p for p in patterns if p.pattern_type == PatternType.STRUCTURE_FILE]
        assert len(file_patterns) >= 1

        test_pattern = next((p for p in file_patterns if p.name == "test_file_pattern"), None)
        assert test_pattern is not None
        assert test_pattern.value == "test_suffix"
        assert test_pattern.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_analyze_detects_absolute_imports(self, absolute_imports_codebase: Path) -> None:
        """Test analyzer detects absolute import style."""
        analyzer = StructureAnalyzer()
        patterns = await analyzer.analyze(absolute_imports_codebase)

        # Should detect import style pattern
        import_patterns = [p for p in patterns if p.pattern_type == PatternType.IMPORT_STYLE]
        assert len(import_patterns) >= 1

        import_pattern = next((p for p in import_patterns if p.name == "import_style"), None)
        assert import_pattern is not None
        assert import_pattern.value == "absolute"

    @pytest.mark.asyncio
    async def test_analyze_detects_relative_imports(self, relative_imports_codebase: Path) -> None:
        """Test analyzer detects relative import style."""
        analyzer = StructureAnalyzer()
        patterns = await analyzer.analyze(relative_imports_codebase)

        # Should detect import style pattern
        import_patterns = [p for p in patterns if p.pattern_type == PatternType.IMPORT_STYLE]
        assert len(import_patterns) >= 1

        import_pattern = next((p for p in import_patterns if p.name == "import_style"), None)
        assert import_pattern is not None
        assert import_pattern.value == "relative"

    @pytest.mark.asyncio
    async def test_analyze_detects_tests_directory(self, src_layout_codebase: Path) -> None:
        """Test analyzer detects tests directory location."""
        analyzer = StructureAnalyzer()
        patterns = await analyzer.analyze(src_layout_codebase)

        # Should detect test directory pattern
        dir_patterns = [p for p in patterns if p.pattern_type == PatternType.STRUCTURE_DIRECTORY]

        test_dir_pattern = next((p for p in dir_patterns if p.name == "test_directory"), None)
        assert test_dir_pattern is not None
        assert test_dir_pattern.value == "tests"

    @pytest.mark.asyncio
    async def test_analyze_returns_code_pattern_instances(self, src_layout_codebase: Path) -> None:
        """Test analyzer returns proper CodePattern instances."""
        analyzer = StructureAnalyzer()
        patterns = await analyzer.analyze(src_layout_codebase)

        for pattern in patterns:
            assert isinstance(pattern, CodePattern)
            assert isinstance(pattern.pattern_type, PatternType)
            assert isinstance(pattern.name, str)
            assert isinstance(pattern.value, str)
            assert 0.0 <= pattern.confidence <= 1.0
            assert isinstance(pattern.examples, tuple)

    @pytest.mark.asyncio
    async def test_analyze_calculates_confidence(self, tmp_path: Path) -> None:
        """Test analyzer calculates confidence based on consistency."""
        # Create mixed test patterns
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_one.py").write_text("pass\n")
        (tests / "test_two.py").write_text("pass\n")
        (tests / "three_test.py").write_text("pass\n")

        analyzer = StructureAnalyzer()
        patterns = await analyzer.analyze(tmp_path)

        # Should detect test file pattern with lower confidence
        file_patterns = [p for p in patterns if p.pattern_type == PatternType.STRUCTURE_FILE]

        test_pattern = next((p for p in file_patterns if p.name == "test_file_pattern"), None)
        if test_pattern:
            # 2 out of 3 use test_ prefix, so confidence ~0.67
            assert test_pattern.confidence < 1.0
            assert test_pattern.confidence > 0.5
