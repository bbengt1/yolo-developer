"""Integration tests for integration test generation (Story 8.4).

Tests the full flow from code files through component analysis to
integration test generation, validating the end-to-end pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from yolo_developer.agents.dev.integration_utils import (
    analyze_component_boundaries,
    analyze_data_flow,
    detect_error_scenarios,
    generate_integration_tests_with_llm,
    validate_integration_test_quality,
)
from yolo_developer.agents.dev.types import CodeFile

if TYPE_CHECKING:
    pass


class TestIntegrationTestGenerationPipeline:
    """Integration tests for the full integration test generation pipeline."""

    @pytest.fixture
    def multi_file_codebase(self) -> list[CodeFile]:
        """Sample codebase with multiple interacting files."""
        file_a = CodeFile(
            file_path="src/service.py",
            content="""
from repository import save_data, load_data

async def process_item(item: dict) -> dict:
    \"\"\"Process an item and save to repository.\"\"\"
    try:
        result = await save_data(item)
        return {"status": "success", "id": result}
    except ValueError as e:
        return {"status": "error", "message": str(e)}
""",
            file_type="source",
        )

        file_b = CodeFile(
            file_path="src/repository.py",
            content="""
async def save_data(data: dict) -> str:
    \"\"\"Save data to storage.\"\"\"
    if not data:
        raise ValueError("Data cannot be empty")
    return "saved-123"

async def load_data(item_id: str) -> dict:
    \"\"\"Load data from storage.\"\"\"
    return {"id": item_id, "value": "test"}
""",
            file_type="source",
        )

        return [file_a, file_b]

    @pytest.fixture
    def mock_router(self) -> MagicMock:
        """Mock LLM router for testing."""
        router = MagicMock()
        router.call = AsyncMock(
            return_value='''```python
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_repository():
    """Mock repository for integration tests."""
    repo = MagicMock()
    repo.save_data = AsyncMock(return_value="test-id")
    return repo

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after each test."""
    yield
    # Cleanup code

@pytest.mark.asyncio
async def test_service_repository_integration(mock_repository):
    """Test service calls repository correctly."""
    assert True
```'''
        )
        return router

    def test_boundary_analysis_detects_imports(self, multi_file_codebase: list[CodeFile]) -> None:
        """Test that boundary analysis detects imports between modules."""
        boundaries = analyze_component_boundaries(multi_file_codebase)

        # Should detect imports from service.py to repository.py
        import_boundaries = [b for b in boundaries if b.interaction_type == "import"]
        assert len(import_boundaries) >= 1

        # Should detect function call boundaries
        call_boundaries = [b for b in boundaries if b.interaction_type == "function_call"]
        # service.py calls save_data from repository
        assert len(call_boundaries) >= 1

    def test_data_flow_analysis_traces_paths(self, multi_file_codebase: list[CodeFile]) -> None:
        """Test that data flow analysis traces transformation paths."""
        flows = analyze_data_flow(multi_file_codebase)

        # Should detect data flows in functions
        assert len(flows) >= 1

        # Check flow through process_item
        process_flows = [f for f in flows if "process_item" in f.start_point]
        assert len(process_flows) >= 1

    def test_error_scenario_detection(self, multi_file_codebase: list[CodeFile]) -> None:
        """Test that error scenarios are detected."""
        scenarios = detect_error_scenarios(multi_file_codebase)

        # Should detect the ValueError in repository.py
        value_errors = [s for s in scenarios if s.exception_type == "ValueError"]
        assert len(value_errors) >= 1

        # Should detect the try/except in service.py
        try_excepts = [s for s in scenarios if s.handling == "try/except"]
        assert len(try_excepts) >= 1

    @pytest.mark.asyncio
    async def test_full_generation_pipeline(
        self,
        multi_file_codebase: list[CodeFile],
        mock_router: MagicMock,
    ) -> None:
        """Test full pipeline from analysis to test generation."""
        # Step 1: Analyze boundaries
        boundaries = analyze_component_boundaries(multi_file_codebase)
        assert len(boundaries) >= 1

        # Step 2: Analyze data flow
        flows = analyze_data_flow(multi_file_codebase)
        assert len(flows) >= 1

        # Step 3: Detect error scenarios
        errors = detect_error_scenarios(multi_file_codebase)
        assert len(errors) >= 1

        # Step 4: Generate integration tests with LLM
        test_code, is_valid = await generate_integration_tests_with_llm(
            code_files=multi_file_codebase,
            boundaries=boundaries,
            flows=flows,
            error_scenarios=errors,
            router=mock_router,
        )

        # Should generate valid test code
        assert is_valid
        assert len(test_code) > 0
        assert "pytest" in test_code

        # Step 5: Validate generated test quality
        report = validate_integration_test_quality(test_code)

        # Should detect good test practices
        assert report.uses_fixtures
        assert report.uses_mocks
        assert report.has_cleanup
        assert report.is_async_compliant

    def test_quality_report_integration(self) -> None:
        """Test quality report with realistic test code."""
        test_code = '''
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.fixture
def mock_service():
    """Mock service for integration tests."""
    return MagicMock()

@pytest.fixture(autouse=True)
def cleanup_state():
    """Clean up state after each test."""
    yield
    # Cleanup happens here

@pytest.mark.asyncio
async def test_integration_flow(mock_service):
    """Test integration between components."""
    mock_service.process = AsyncMock(return_value={"status": "ok"})
    result = await mock_service.process({"data": "test"})
    assert result["status"] == "ok"
'''

        report = validate_integration_test_quality(test_code)

        # Should pass all quality checks
        assert report.uses_fixtures is True
        assert report.uses_mocks is True
        assert report.has_cleanup is True
        assert report.is_async_compliant is True
        assert report.is_acceptable() is True
        assert len(report.warnings) == 0


class TestBoundaryDetectionEdgeCases:
    """Test edge cases in component boundary detection."""

    def test_handles_empty_files(self) -> None:
        """Test graceful handling of empty file list."""
        boundaries = analyze_component_boundaries([])
        assert boundaries == []

    def test_handles_syntax_errors(self) -> None:
        """Test graceful handling of files with syntax errors."""
        bad_file = CodeFile(
            file_path="src/broken.py",
            content="def broken( invalid syntax",
            file_type="source",
        )
        good_file = CodeFile(
            file_path="src/good.py",
            content="def good(): return 1",
            file_type="source",
        )

        # Should not raise, should skip broken file
        boundaries = analyze_component_boundaries([bad_file, good_file])
        assert isinstance(boundaries, list)

    def test_skips_non_source_files(self) -> None:
        """Test that test and config files are skipped."""
        test_file = CodeFile(
            file_path="tests/test_module.py",
            content="from module import func",
            file_type="test",
        )

        boundaries = analyze_component_boundaries([test_file])
        assert boundaries == []


class TestDataFlowEdgeCases:
    """Test edge cases in data flow analysis."""

    def test_handles_recursive_functions(self) -> None:
        """Test handling of recursive function calls."""
        recursive_file = CodeFile(
            file_path="src/recursive.py",
            content="""
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
            file_type="source",
        )

        flows = analyze_data_flow([recursive_file])
        # Should handle without infinite loop
        assert isinstance(flows, list)


class TestErrorScenarioEdgeCases:
    """Test edge cases in error scenario detection."""

    def test_detects_multiple_exception_types(self) -> None:
        """Test detection of multi-exception handlers."""
        multi_except_file = CodeFile(
            file_path="src/handler.py",
            content="""
def handle():
    try:
        risky_operation()
    except (ValueError, TypeError) as e:
        return str(e)
""",
            file_type="source",
        )

        scenarios = detect_error_scenarios([multi_except_file])
        assert len(scenarios) >= 1

        # Should capture both exception types
        multi_type = [s for s in scenarios if s.exception_type and "|" in s.exception_type]
        assert len(multi_type) >= 1
