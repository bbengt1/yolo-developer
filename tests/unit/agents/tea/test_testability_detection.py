"""Unit tests for testability pattern detection (Story 9.6).

Tests for all pattern detection functions:
- _detect_global_state
- _detect_tight_coupling
- _detect_hidden_dependencies
- _detect_complex_conditionals
- _detect_long_methods
- _detect_deep_nesting

Note on Testing Strategy:
    These tests intentionally import and test private functions (prefixed with _)
    as white-box unit tests. This allows fine-grained validation of each pattern
    detector's behavior in isolation. Integration tests in test_testability_integration.py
    validate the public API (audit_testability) for black-box testing.

    This dual approach ensures:
    1. Individual detectors work correctly (white-box, this file)
    2. The public API produces correct aggregate results (black-box, integration tests)
"""

from __future__ import annotations


class TestDetectGlobalState:
    """Tests for _detect_global_state function."""

    def test_detects_module_level_mutable_list(self) -> None:
        """Test detection of module-level mutable list."""
        from yolo_developer.agents.tea.testability import _detect_global_state

        content = """
cache = []
def foo():
    cache.append(1)
"""
        issues = _detect_global_state(content, "test.py")
        assert len(issues) == 1
        assert issues[0].pattern_type == "global_state"
        assert issues[0].severity == "critical"
        assert "cache" in issues[0].description

    def test_detects_module_level_mutable_dict(self) -> None:
        """Test detection of module-level mutable dict."""
        from yolo_developer.agents.tea.testability import _detect_global_state

        content = """
config = {}
def foo():
    config["key"] = "value"
"""
        issues = _detect_global_state(content, "test.py")
        assert len(issues) == 1
        assert issues[0].pattern_type == "global_state"
        assert "config" in issues[0].description

    def test_detects_dict_call(self) -> None:
        """Test detection of dict() call at module level."""
        from yolo_developer.agents.tea.testability import _detect_global_state

        content = """
registry = dict()
"""
        issues = _detect_global_state(content, "test.py")
        assert len(issues) == 1
        assert "registry" in issues[0].description

    def test_ignores_private_variables(self) -> None:
        """Test that private variables (_var) are ignored."""
        from yolo_developer.agents.tea.testability import _detect_global_state

        content = """
_cache = []
"""
        issues = _detect_global_state(content, "test.py")
        assert len(issues) == 0

    def test_ignores_constants(self) -> None:
        """Test that UPPER_CASE constants are ignored."""
        from yolo_developer.agents.tea.testability import _detect_global_state

        content = """
DEFAULT_CONFIG = {}
"""
        issues = _detect_global_state(content, "test.py")
        assert len(issues) == 0

    def test_handles_syntax_error(self) -> None:
        """Test graceful handling of syntax errors."""
        from yolo_developer.agents.tea.testability import _detect_global_state

        content = "def broken("  # Invalid syntax
        issues = _detect_global_state(content, "test.py")
        assert len(issues) == 0

    def test_empty_content(self) -> None:
        """Test handling of empty content."""
        from yolo_developer.agents.tea.testability import _detect_global_state

        issues = _detect_global_state("", "test.py")
        assert len(issues) == 0


class TestDetectTightCoupling:
    """Tests for _detect_tight_coupling function."""

    def test_detects_instantiation_in_init(self) -> None:
        """Test detection of class instantiation in __init__."""
        from yolo_developer.agents.tea.testability import _detect_tight_coupling

        content = """
class Service:
    def __init__(self):
        self.db = Database()
"""
        issues = _detect_tight_coupling(content, "test.py")
        assert len(issues) == 1
        assert issues[0].pattern_type == "tight_coupling"
        assert issues[0].severity == "high"
        assert "Database" in issues[0].description

    def test_ignores_builtin_types(self) -> None:
        """Test that builtin types like Dict, List are ignored."""
        from yolo_developer.agents.tea.testability import _detect_tight_coupling

        content = """
class Service:
    def __init__(self):
        self.data = Dict()
        self.items = List()
"""
        issues = _detect_tight_coupling(content, "test.py")
        assert len(issues) == 0

    def test_ignores_parameter_assignment(self) -> None:
        """Test that dependency injection pattern is not flagged."""
        from yolo_developer.agents.tea.testability import _detect_tight_coupling

        content = """
class Service:
    def __init__(self, db):
        self.db = db
"""
        issues = _detect_tight_coupling(content, "test.py")
        assert len(issues) == 0

    def test_handles_syntax_error(self) -> None:
        """Test graceful handling of syntax errors."""
        from yolo_developer.agents.tea.testability import _detect_tight_coupling

        content = "class broken"  # Invalid syntax
        issues = _detect_tight_coupling(content, "test.py")
        assert len(issues) == 0


class TestDetectHiddenDependencies:
    """Tests for _detect_hidden_dependencies function."""

    def test_detects_import_in_function(self) -> None:
        """Test detection of import inside function."""
        from yolo_developer.agents.tea.testability import _detect_hidden_dependencies

        content = """
def foo():
    import json
    return json.dumps({})
"""
        issues = _detect_hidden_dependencies(content, "test.py")
        assert len(issues) == 1
        assert issues[0].pattern_type == "hidden_dependency"
        assert issues[0].severity == "high"
        assert "json" in issues[0].description

    def test_detects_from_import_in_function(self) -> None:
        """Test detection of from-import inside function."""
        from yolo_developer.agents.tea.testability import _detect_hidden_dependencies

        content = """
def foo():
    from os import path
    return path.exists("file.txt")
"""
        issues = _detect_hidden_dependencies(content, "test.py")
        assert len(issues) == 1
        assert "path" in issues[0].description

    def test_ignores_module_level_imports(self) -> None:
        """Test that module-level imports are not flagged."""
        from yolo_developer.agents.tea.testability import _detect_hidden_dependencies

        content = """
import json

def foo():
    return json.dumps({})
"""
        issues = _detect_hidden_dependencies(content, "test.py")
        assert len(issues) == 0

    def test_handles_syntax_error(self) -> None:
        """Test graceful handling of syntax errors."""
        from yolo_developer.agents.tea.testability import _detect_hidden_dependencies

        content = "def broken("  # Invalid syntax
        issues = _detect_hidden_dependencies(content, "test.py")
        assert len(issues) == 0

    def test_detects_import_in_async_function(self) -> None:
        """Test detection of import inside async function."""
        from yolo_developer.agents.tea.testability import _detect_hidden_dependencies

        content = """
async def async_foo():
    import aiohttp
    return await aiohttp.get("url")
"""
        issues = _detect_hidden_dependencies(content, "test.py")
        assert len(issues) == 1
        assert issues[0].pattern_type == "hidden_dependency"
        assert "aiohttp" in issues[0].description
        assert "async_foo" in issues[0].description


class TestDetectComplexConditionals:
    """Tests for _detect_complex_conditionals function."""

    def test_detects_high_complexity(self) -> None:
        """Test detection of function with cyclomatic complexity > 10."""
        from yolo_developer.agents.tea.testability import _detect_complex_conditionals

        # Create a function with high complexity
        content = """
def complex_func(a, b, c, d, e, f, g, h, i, j, k, l):
    if a:
        if b:
            pass
        elif c:
            pass
        elif d:
            pass
    if e:
        if f:
            pass
    if g and h:
        pass
    if i or j:
        pass
    for x in range(k):
        if l:
            pass
    while True:
        break
"""
        issues = _detect_complex_conditionals(content, "test.py")
        assert len(issues) == 1
        assert issues[0].pattern_type == "complex_conditional"
        assert issues[0].severity == "medium"
        assert "cyclomatic complexity" in issues[0].description.lower()

    def test_ignores_low_complexity(self) -> None:
        """Test that simple functions are not flagged."""
        from yolo_developer.agents.tea.testability import _detect_complex_conditionals

        content = """
def simple_func(a):
    if a:
        return True
    return False
"""
        issues = _detect_complex_conditionals(content, "test.py")
        assert len(issues) == 0

    def test_handles_syntax_error(self) -> None:
        """Test graceful handling of syntax errors."""
        from yolo_developer.agents.tea.testability import _detect_complex_conditionals

        content = "def broken("  # Invalid syntax
        issues = _detect_complex_conditionals(content, "test.py")
        assert len(issues) == 0


class TestDetectLongMethods:
    """Tests for _detect_long_methods function."""

    def test_detects_long_function(self) -> None:
        """Test detection of function > 50 lines."""
        from yolo_developer.agents.tea.testability import _detect_long_methods

        # Create a 55-line function
        lines = ["def long_func():"]
        lines.extend(["    pass"] * 54)  # 54 lines of pass statements
        content = "\n".join(lines)

        issues = _detect_long_methods(content, "test.py")
        assert len(issues) == 1
        assert issues[0].pattern_type == "long_method"
        assert issues[0].severity == "medium"
        assert "long_func" in issues[0].description
        assert "55" in issues[0].description or "lines" in issues[0].description.lower()

    def test_ignores_short_function(self) -> None:
        """Test that short functions are not flagged."""
        from yolo_developer.agents.tea.testability import _detect_long_methods

        content = """
def short_func():
    return True
"""
        issues = _detect_long_methods(content, "test.py")
        assert len(issues) == 0

    def test_detects_long_async_function(self) -> None:
        """Test detection of async function > 50 lines."""
        from yolo_developer.agents.tea.testability import _detect_long_methods

        lines = ["async def long_async_func():"]
        lines.extend(["    await something()"] * 54)
        content = "\n".join(lines)

        issues = _detect_long_methods(content, "test.py")
        assert len(issues) == 1
        assert "long_async_func" in issues[0].description

    def test_handles_syntax_error(self) -> None:
        """Test graceful handling of syntax errors."""
        from yolo_developer.agents.tea.testability import _detect_long_methods

        content = "def broken("  # Invalid syntax
        issues = _detect_long_methods(content, "test.py")
        assert len(issues) == 0


class TestDetectDeepNesting:
    """Tests for _detect_deep_nesting function."""

    def test_detects_deep_nesting(self) -> None:
        """Test detection of nesting > 4 levels."""
        from yolo_developer.agents.tea.testability import _detect_deep_nesting

        content = """
def deeply_nested():
    if True:
        if True:
            if True:
                if True:
                    if True:
                        pass
"""
        issues = _detect_deep_nesting(content, "test.py")
        assert len(issues) == 1
        assert issues[0].pattern_type == "deep_nesting"
        assert issues[0].severity == "low"
        assert "nesting" in issues[0].description.lower()

    def test_ignores_shallow_nesting(self) -> None:
        """Test that shallow nesting is not flagged."""
        from yolo_developer.agents.tea.testability import _detect_deep_nesting

        content = """
def shallow_func():
    if True:
        if True:
            pass
"""
        issues = _detect_deep_nesting(content, "test.py")
        assert len(issues) == 0

    def test_detects_mixed_nesting(self) -> None:
        """Test detection of mixed nesting types (if/for/while/with)."""
        from yolo_developer.agents.tea.testability import _detect_deep_nesting

        content = """
def mixed_nesting():
    if True:
        for x in range(10):
            while True:
                with open("f") as f:
                    if True:
                        pass
"""
        issues = _detect_deep_nesting(content, "test.py")
        assert len(issues) == 1

    def test_handles_syntax_error(self) -> None:
        """Test graceful handling of syntax errors."""
        from yolo_developer.agents.tea.testability import _detect_deep_nesting

        content = "def broken("  # Invalid syntax
        issues = _detect_deep_nesting(content, "test.py")
        assert len(issues) == 0


class TestEdgeCases:
    """Tests for edge cases in pattern detection."""

    def test_empty_file(self) -> None:
        """Test handling of empty file."""
        from yolo_developer.agents.tea.testability import audit_testability

        report = audit_testability([{"artifact_id": "empty.py", "content": ""}])
        assert report.score.score == 100
        assert report.metrics.total_issues == 0
        assert report.metrics.files_analyzed == 0

    def test_file_with_only_comments(self) -> None:
        """Test handling of file with only comments."""
        from yolo_developer.agents.tea.testability import audit_testability

        content = '''
# This is a comment
# Another comment
"""
Docstring
"""
'''
        report = audit_testability([{"artifact_id": "comments.py", "content": content}])
        assert report.metrics.total_issues == 0

    def test_malformed_code(self) -> None:
        """Test handling of malformed Python code."""
        from yolo_developer.agents.tea.testability import audit_testability

        content = "def broken( { invalid }"
        report = audit_testability([{"artifact_id": "malformed.py", "content": content}])
        # Should not crash, return empty results
        assert report.score.score == 100

    def test_multiple_files(self) -> None:
        """Test auditing multiple files."""
        from yolo_developer.agents.tea.testability import audit_testability

        files = [
            {"artifact_id": "file1.py", "content": "cache = []"},
            {"artifact_id": "file2.py", "content": "data = {}"},
        ]
        report = audit_testability(files)
        assert report.metrics.files_analyzed == 2
        assert report.metrics.total_issues == 2
        assert report.metrics.files_with_issues == 2
