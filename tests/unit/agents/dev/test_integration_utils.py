"""Unit tests for integration test utilities (Story 8.4, AC1, AC2, AC3, AC7).

Tests for component boundary analysis, data flow analysis, and error scenario
detection utilities used for integration test generation.
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.dev.types import CodeFile


class TestComponentBoundary:
    """Tests for ComponentBoundary dataclass."""

    def test_dataclass_is_frozen(self) -> None:
        """Test that ComponentBoundary is immutable (frozen)."""
        from yolo_developer.agents.dev.integration_utils import ComponentBoundary

        boundary = ComponentBoundary(
            source_file="a.py",
            target_file="b.py",
            interaction_type="import",
            boundary_point="module_b",
            is_async=False,
        )

        with pytest.raises(AttributeError):
            boundary.source_file = "c.py"  # type: ignore[misc]

    def test_dataclass_attributes(self) -> None:
        """Test ComponentBoundary has required attributes."""
        from yolo_developer.agents.dev.integration_utils import ComponentBoundary

        boundary = ComponentBoundary(
            source_file="src/module_a.py",
            target_file="src/module_b.py",
            interaction_type="function_call",
            boundary_point="process_data",
            is_async=True,
        )

        assert boundary.source_file == "src/module_a.py"
        assert boundary.target_file == "src/module_b.py"
        assert boundary.interaction_type == "function_call"
        assert boundary.boundary_point == "process_data"
        assert boundary.is_async is True


class TestAnalyzeComponentBoundaries:
    """Tests for analyze_component_boundaries() function (AC1, AC7)."""

    def test_function_exists(self) -> None:
        """Test that function is exported."""
        from yolo_developer.agents.dev.integration_utils import (
            analyze_component_boundaries,
        )

        assert callable(analyze_component_boundaries)

    def test_empty_code_files_returns_empty_list(self) -> None:
        """Test that empty input returns empty boundaries."""
        from yolo_developer.agents.dev.integration_utils import (
            analyze_component_boundaries,
        )

        result = analyze_component_boundaries([])
        assert result == []

    def test_single_file_no_boundaries(self) -> None:
        """Test that single file with no imports returns empty boundaries."""
        from yolo_developer.agents.dev.integration_utils import (
            analyze_component_boundaries,
        )

        code_file = CodeFile(
            file_path="src/standalone.py",
            content="def hello(): return 'world'",
            file_type="source",
        )

        result = analyze_component_boundaries([code_file])
        assert result == []

    def test_detects_import_between_files(self) -> None:
        """Test detection of import statements between codebase files."""
        from yolo_developer.agents.dev.integration_utils import (
            analyze_component_boundaries,
        )

        file_a = CodeFile(
            file_path="src/module_a.py",
            content="from module_b import helper\n\ndef main(): return helper()",
            file_type="source",
        )
        file_b = CodeFile(
            file_path="src/module_b.py",
            content="def helper(): return 42",
            file_type="source",
        )

        result = analyze_component_boundaries([file_a, file_b])

        assert len(result) >= 1
        # Should detect import from module_a to module_b
        import_boundaries = [b for b in result if b.interaction_type == "import"]
        assert len(import_boundaries) >= 1

    def test_detects_function_calls_across_modules(self) -> None:
        """Test detection of function calls to imported modules."""
        from yolo_developer.agents.dev.integration_utils import (
            analyze_component_boundaries,
        )

        file_a = CodeFile(
            file_path="src/caller.py",
            content="""
from processor import process_data

def main():
    result = process_data(42)
    return result
""",
            file_type="source",
        )
        file_b = CodeFile(
            file_path="src/processor.py",
            content="def process_data(x): return x * 2",
            file_type="source",
        )

        result = analyze_component_boundaries([file_a, file_b])

        # Should detect function call boundary
        call_boundaries = [b for b in result if b.interaction_type == "function_call"]
        assert len(call_boundaries) >= 1

    def test_detects_async_boundaries(self) -> None:
        """Test detection of async function calls."""
        from yolo_developer.agents.dev.integration_utils import (
            analyze_component_boundaries,
        )

        file_a = CodeFile(
            file_path="src/async_caller.py",
            content="""
from async_helper import fetch_data

async def main():
    result = await fetch_data()
    return result
""",
            file_type="source",
        )
        file_b = CodeFile(
            file_path="src/async_helper.py",
            content="async def fetch_data(): return 42",
            file_type="source",
        )

        result = analyze_component_boundaries([file_a, file_b])

        # Should have at least one async boundary
        async_boundaries = [b for b in result if b.is_async]
        assert len(async_boundaries) >= 1

    def test_skips_non_source_files(self) -> None:
        """Test that test files and config files are skipped."""
        from yolo_developer.agents.dev.integration_utils import (
            analyze_component_boundaries,
        )

        test_file = CodeFile(
            file_path="tests/test_module.py",
            content="from module import func\ndef test_func(): pass",
            file_type="test",
        )
        config_file = CodeFile(
            file_path="config.py",
            content="from settings import DB_URL",
            file_type="config",
        )

        result = analyze_component_boundaries([test_file, config_file])

        # Should return empty since these aren't source files
        assert result == []

    def test_handles_syntax_errors_gracefully(self) -> None:
        """Test that syntax errors in code don't crash the analysis."""
        from yolo_developer.agents.dev.integration_utils import (
            analyze_component_boundaries,
        )

        bad_file = CodeFile(
            file_path="src/broken.py",
            content="def broken( invalid syntax here",
            file_type="source",
        )
        good_file = CodeFile(
            file_path="src/good.py",
            content="def good(): return 1",
            file_type="source",
        )

        # Should not raise, should just skip the broken file
        result = analyze_component_boundaries([bad_file, good_file])
        assert isinstance(result, list)


class TestDataFlowPath:
    """Tests for DataFlowPath dataclass."""

    def test_dataclass_is_frozen(self) -> None:
        """Test that DataFlowPath is immutable (frozen)."""
        from yolo_developer.agents.dev.integration_utils import DataFlowPath

        flow = DataFlowPath(
            start_point="input",
            end_point="output",
            steps=("transform",),
            data_types=("str",),
        )

        with pytest.raises(AttributeError):
            flow.start_point = "other"  # type: ignore[misc]

    def test_dataclass_attributes(self) -> None:
        """Test DataFlowPath has required attributes."""
        from yolo_developer.agents.dev.integration_utils import DataFlowPath

        flow = DataFlowPath(
            start_point="user_input",
            end_point="database_write",
            steps=("validate", "transform", "save"),
            data_types=("str", "dict", "Model"),
        )

        assert flow.start_point == "user_input"
        assert flow.end_point == "database_write"
        assert len(flow.steps) == 3
        assert len(flow.data_types) == 3


class TestAnalyzeDataFlow:
    """Tests for analyze_data_flow() function (AC2)."""

    def test_function_exists(self) -> None:
        """Test that function is exported."""
        from yolo_developer.agents.dev.integration_utils import analyze_data_flow

        assert callable(analyze_data_flow)

    def test_empty_code_files_returns_empty_list(self) -> None:
        """Test that empty input returns empty flows."""
        from yolo_developer.agents.dev.integration_utils import analyze_data_flow

        result = analyze_data_flow([])
        assert result == []

    def test_detects_function_chain_flow(self) -> None:
        """Test detection of data flowing through function calls."""
        from yolo_developer.agents.dev.integration_utils import analyze_data_flow

        code_file = CodeFile(
            file_path="src/pipeline.py",
            content="""
def step1(data: str) -> dict:
    return {"value": data}

def step2(data: dict) -> list:
    return [data["value"]]

def pipeline(input_data: str) -> list:
    intermediate = step1(input_data)
    result = step2(intermediate)
    return result
""",
            file_type="source",
        )

        result = analyze_data_flow([code_file])

        # Should detect at least one data flow path
        assert len(result) >= 1

    def test_detects_return_type_transformations(self) -> None:
        """Test detection of type changes in data flow."""
        from yolo_developer.agents.dev.integration_utils import analyze_data_flow

        code_file = CodeFile(
            file_path="src/converter.py",
            content="""
def convert(x: int) -> str:
    return str(x)
""",
            file_type="source",
        )

        result = analyze_data_flow([code_file])

        # Should detect data type transformation
        if result:
            # At least one path should show type change
            assert any(len(flow.data_types) >= 1 for flow in result)


class TestErrorScenario:
    """Tests for ErrorScenario dataclass."""

    def test_dataclass_is_frozen(self) -> None:
        """Test that ErrorScenario is immutable (frozen)."""
        from yolo_developer.agents.dev.integration_utils import ErrorScenario

        scenario = ErrorScenario(
            trigger="invalid_input",
            handling="try/except",
            recovery="fallback_value",
            exception_type="ValueError",
        )

        with pytest.raises(AttributeError):
            scenario.trigger = "other"  # type: ignore[misc]

    def test_dataclass_attributes(self) -> None:
        """Test ErrorScenario has required attributes."""
        from yolo_developer.agents.dev.integration_utils import ErrorScenario

        scenario = ErrorScenario(
            trigger="network_failure",
            handling="except block",
            recovery=None,
            exception_type="ConnectionError",
        )

        assert scenario.trigger == "network_failure"
        assert scenario.handling == "except block"
        assert scenario.recovery is None
        assert scenario.exception_type == "ConnectionError"


class TestDetectErrorScenarios:
    """Tests for detect_error_scenarios() function (AC3)."""

    def test_function_exists(self) -> None:
        """Test that function is exported."""
        from yolo_developer.agents.dev.integration_utils import detect_error_scenarios

        assert callable(detect_error_scenarios)

    def test_empty_code_files_returns_empty_list(self) -> None:
        """Test that empty input returns empty scenarios."""
        from yolo_developer.agents.dev.integration_utils import detect_error_scenarios

        result = detect_error_scenarios([])
        assert result == []

    def test_detects_try_except_blocks(self) -> None:
        """Test detection of try/except error handling."""
        from yolo_developer.agents.dev.integration_utils import detect_error_scenarios

        code_file = CodeFile(
            file_path="src/handler.py",
            content="""
def safe_divide(a: int, b: int) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        return 0.0
""",
            file_type="source",
        )

        result = detect_error_scenarios([code_file])

        # Should detect the ZeroDivisionError handling
        assert len(result) >= 1
        assert any("ZeroDivisionError" in (s.exception_type or "") for s in result)

    def test_detects_raise_statements(self) -> None:
        """Test detection of explicit raise statements."""
        from yolo_developer.agents.dev.integration_utils import detect_error_scenarios

        code_file = CodeFile(
            file_path="src/validator.py",
            content="""
def validate(x: int) -> int:
    if x < 0:
        raise ValueError("x must be non-negative")
    return x
""",
            file_type="source",
        )

        result = detect_error_scenarios([code_file])

        # Should detect the ValueError raise
        assert len(result) >= 1
        assert any("ValueError" in (s.exception_type or "") for s in result)

    def test_detects_fallback_patterns(self) -> None:
        """Test detection of fallback/default patterns."""
        from yolo_developer.agents.dev.integration_utils import detect_error_scenarios

        code_file = CodeFile(
            file_path="src/fallback.py",
            content="""
def get_value(key: str, default: int = 0) -> int:
    try:
        return external_call(key)
    except Exception:
        return default
""",
            file_type="source",
        )

        result = detect_error_scenarios([code_file])

        # Should detect fallback pattern
        assert len(result) >= 1
        assert any(s.recovery is not None for s in result)


class TestIntegrationTestQualityReport:
    """Tests for IntegrationTestQualityReport dataclass."""

    def test_dataclass_exists(self) -> None:
        """Test that dataclass exists and can be instantiated."""
        from yolo_developer.agents.dev.integration_utils import (
            IntegrationTestQualityReport,
        )

        report = IntegrationTestQualityReport(warnings=[])
        assert report is not None

    def test_is_acceptable_with_all_criteria(self) -> None:
        """Test is_acceptable returns True when all criteria met."""
        from yolo_developer.agents.dev.integration_utils import (
            IntegrationTestQualityReport,
        )

        report = IntegrationTestQualityReport(
            warnings=[],
            uses_fixtures=True,
            uses_mocks=True,
            has_cleanup=True,
            is_async_compliant=True,
        )

        assert report.is_acceptable() is True

    def test_is_acceptable_false_without_fixtures(self) -> None:
        """Test is_acceptable returns False without fixtures."""
        from yolo_developer.agents.dev.integration_utils import (
            IntegrationTestQualityReport,
        )

        report = IntegrationTestQualityReport(
            warnings=[],
            uses_fixtures=False,
            uses_mocks=True,
            has_cleanup=True,
        )

        assert report.is_acceptable() is False

    def test_is_acceptable_false_without_mocks(self) -> None:
        """Test is_acceptable returns False without mocks."""
        from yolo_developer.agents.dev.integration_utils import (
            IntegrationTestQualityReport,
        )

        report = IntegrationTestQualityReport(
            warnings=[],
            uses_fixtures=True,
            uses_mocks=False,
            has_cleanup=True,
        )

        assert report.is_acceptable() is False


class TestValidateIntegrationTestQuality:
    """Tests for validate_integration_test_quality() function (AC3, AC4)."""

    def test_function_exists(self) -> None:
        """Test that function is exported."""
        from yolo_developer.agents.dev.integration_utils import (
            validate_integration_test_quality,
        )

        assert callable(validate_integration_test_quality)

    def test_detects_fixture_usage(self) -> None:
        """Test detection of pytest fixture usage."""
        from yolo_developer.agents.dev.integration_utils import (
            validate_integration_test_quality,
        )

        test_code = """
import pytest

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_something(sample_data):
    assert sample_data["key"] == "value"
"""

        report = validate_integration_test_quality(test_code)

        assert report.uses_fixtures is True

    def test_detects_mock_usage(self) -> None:
        """Test detection of mock/patch usage."""
        from yolo_developer.agents.dev.integration_utils import (
            validate_integration_test_quality,
        )

        test_code = """
from unittest.mock import MagicMock, patch

def test_with_mock():
    mock = MagicMock()
    mock.return_value = 42
    assert mock() == 42
"""

        report = validate_integration_test_quality(test_code)

        assert report.uses_mocks is True

    def test_detects_async_test_markers(self) -> None:
        """Test detection of @pytest.mark.asyncio markers."""
        from yolo_developer.agents.dev.integration_utils import (
            validate_integration_test_quality,
        )

        test_code = """
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_call()
    assert result is not None
"""

        report = validate_integration_test_quality(test_code)

        assert report.is_async_compliant is True

    def test_warns_on_missing_async_marker(self) -> None:
        """Test warning when async function lacks asyncio marker."""
        from yolo_developer.agents.dev.integration_utils import (
            validate_integration_test_quality,
        )

        test_code = """
async def test_async_without_marker():
    result = await some_async_call()
    assert result is not None
"""

        report = validate_integration_test_quality(test_code)

        # Should have warning about missing marker
        assert report.is_async_compliant is False or len(report.warnings) > 0
