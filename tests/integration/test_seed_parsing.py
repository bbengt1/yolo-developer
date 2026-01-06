"""Integration tests for seed parsing (Story 4.1 - Task 11).

Tests end-to-end seed parsing with mocked LLM responses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from yolo_developer.seed.api import parse_seed
from yolo_developer.seed.parser import LLMSeedParser
from yolo_developer.seed.types import SeedSource

# Path to fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "seeds"


# =============================================================================
# Mock LLM Responses
# =============================================================================


def create_mock_llm_response(
    goals: int = 1, features: int = 2, constraints: int = 1
) -> dict[str, Any]:
    """Create a mock LLM response with specified counts."""
    return {
        "goals": [
            {
                "title": f"Goal {i}",
                "description": f"Description for goal {i}",
                "priority": i,
                "rationale": f"Rationale for goal {i}",
            }
            for i in range(1, goals + 1)
        ],
        "features": [
            {
                "name": f"Feature {i}",
                "description": f"Description for feature {i}",
                "user_value": f"User value for feature {i}",
                "related_goals": ["Goal 1"] if goals > 0 else [],
            }
            for i in range(1, features + 1)
        ],
        "constraints": [
            {
                "category": "technical",
                "description": f"Constraint {i}",
                "impact": f"Impact of constraint {i}",
                "related_items": [],
            }
            for i in range(1, constraints + 1)
        ],
    }


# =============================================================================
# Integration Tests
# =============================================================================


class TestParseSeedIntegration:
    """Integration tests for parse_seed API."""

    @pytest.mark.asyncio
    async def test_parse_simple_text_seed(self) -> None:
        """Test parsing a simple text seed."""
        content = "Build a blog application with user comments"
        mock_response = create_mock_llm_response(goals=1, features=2, constraints=0)

        with patch.object(LLMSeedParser, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await parse_seed(content)

            assert result.source == SeedSource.TEXT
            assert result.goal_count == 1
            assert result.feature_count == 2
            assert result.constraint_count == 0
            assert "blog application" in result.raw_content

    @pytest.mark.asyncio
    async def test_parse_simple_seed_file(self) -> None:
        """Test parsing the simple seed fixture file."""
        content = (FIXTURES_DIR / "simple_seed.txt").read_text()
        mock_response = create_mock_llm_response(goals=1, features=3, constraints=1)

        with patch.object(LLMSeedParser, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await parse_seed(content, filename="simple_seed.txt")

            assert result.source == SeedSource.FILE
            assert result.goal_count == 1
            assert result.feature_count == 3
            assert result.constraint_count == 1

    @pytest.mark.asyncio
    async def test_parse_complex_markdown_seed(self) -> None:
        """Test parsing the complex markdown seed fixture."""
        content = (FIXTURES_DIR / "complex_seed.md").read_text()
        mock_response = create_mock_llm_response(goals=3, features=10, constraints=3)

        with patch.object(LLMSeedParser, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await parse_seed(content, filename="complex_seed.md")

            assert result.source == SeedSource.FILE
            assert result.goal_count == 3
            assert result.feature_count == 10
            assert result.constraint_count == 3
            # Verify original content is preserved
            assert "E-Commerce Platform" in result.raw_content

    @pytest.mark.asyncio
    async def test_parse_edge_case_seed(self) -> None:
        """Test parsing seed with unicode and special characters."""
        content = (FIXTURES_DIR / "edge_case_seed.txt").read_text()
        mock_response = create_mock_llm_response(goals=1, features=3, constraints=2)

        with patch.object(LLMSeedParser, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await parse_seed(content, filename="edge_case_seed.txt")

            assert result.source == SeedSource.FILE
            assert result.goal_count == 1  # Now we verify the actual count
            # Verify unicode is preserved in raw_content
            assert "🚀" in result.raw_content  # Emoji preserved
            assert "日本語" in result.raw_content  # Japanese preserved
            assert "Ελληνικά" in result.raw_content  # Greek preserved

    @pytest.mark.asyncio
    async def test_parse_with_explicit_source(self) -> None:
        """Test parsing with explicitly specified source."""
        content = "Some content"
        mock_response = create_mock_llm_response(goals=1, features=1, constraints=0)

        with patch.object(LLMSeedParser, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await parse_seed(content, source=SeedSource.URL)

            assert result.source == SeedSource.URL
            assert result.goal_count == 1  # Verify mock was called

    @pytest.mark.asyncio
    async def test_parse_with_preprocessing_disabled(self) -> None:
        """Test parsing with preprocessing disabled."""
        content = "# Heading\n- Item"
        mock_response = create_mock_llm_response(goals=1, features=1, constraints=0)

        with patch.object(LLMSeedParser, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await parse_seed(content, preprocess=False)

            # Should still parse successfully
            assert result.goal_count == 1

    @pytest.mark.asyncio
    async def test_parse_returns_error_metadata_on_llm_failure(self) -> None:
        """Test that LLM failures are handled gracefully."""
        content = "Some content"

        with patch.object(LLMSeedParser, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("API Error")

            result = await parse_seed(content)

            # Should return empty result with error metadata
            assert result.goal_count == 0
            assert result.feature_count == 0
            assert result.constraint_count == 0
            # Error should be in metadata
            metadata_dict = dict(result.metadata)
            assert "error" in metadata_dict

    @pytest.mark.asyncio
    async def test_parse_result_serialization(self) -> None:
        """Test that parse results can be serialized to dict."""
        content = "Build an app"
        mock_response = create_mock_llm_response(goals=2, features=3, constraints=1)

        with patch.object(LLMSeedParser, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await parse_seed(content)
            serialized = result.to_dict()

            # Verify serialization
            assert isinstance(serialized, dict)
            assert "goals" in serialized
            assert "features" in serialized
            assert "constraints" in serialized
            assert len(serialized["goals"]) == 2
            assert len(serialized["features"]) == 3
            assert len(serialized["constraints"]) == 1


class TestLLMParserIntegration:
    """Integration tests for LLMSeedParser directly."""

    @pytest.mark.asyncio
    async def test_llm_parser_direct_invocation(self) -> None:
        """Test LLMSeedParser.parse with mocked _call_llm."""
        from yolo_developer.seed.types import SeedSource

        llm_parser = LLMSeedParser()
        content = "Build a web app"

        mock_response = create_mock_llm_response(goals=1, features=2, constraints=1)

        with patch.object(LLMSeedParser, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await llm_parser.parse(content, SeedSource.TEXT)

            assert result.goal_count == 1
            assert result.feature_count == 2
            assert result.constraint_count == 1
            # Verify the parser was initialized with default settings
            assert llm_parser.model == "gpt-4o-mini"
            assert llm_parser.temperature == 0.1

    @pytest.mark.asyncio
    async def test_llm_parser_with_custom_model(self) -> None:
        """Test LLMSeedParser with custom model configuration."""
        from yolo_developer.seed.types import SeedSource

        llm_parser = LLMSeedParser(model="gpt-4", temperature=0.2)
        content = "Build a mobile app"

        mock_response = create_mock_llm_response(goals=2, features=4, constraints=1)

        with patch.object(LLMSeedParser, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            result = await llm_parser.parse(content, SeedSource.TEXT)

            assert result.goal_count == 2
            assert result.feature_count == 4
            assert llm_parser.model == "gpt-4"
            assert llm_parser.temperature == 0.2
