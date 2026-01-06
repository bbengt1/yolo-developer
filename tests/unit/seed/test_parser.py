"""Unit tests for seed parser infrastructure (Story 4.1 - Task 2).

Tests the parser protocol and utilities:
- SeedParser protocol
- detect_source_format utility
- normalize_content utility
- LLMSeedParser retry logic
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from yolo_developer.seed.parser import (
    LLMSeedParser,
    _parse_markdown,
    _parse_plain_text,
    detect_source_format,
    normalize_content,
)
from yolo_developer.seed.types import SeedSource

# =============================================================================
# detect_source_format Tests
# =============================================================================


class TestDetectSourceFormat:
    """Tests for detect_source_format utility."""

    def test_detect_txt_file(self) -> None:
        """Test detection of .txt file extension."""
        result = detect_source_format("Some content", filename="requirements.txt")
        assert result == SeedSource.FILE

    def test_detect_md_file(self) -> None:
        """Test detection of .md file extension."""
        result = detect_source_format("# Heading\nContent", filename="seed.md")
        assert result == SeedSource.FILE

    def test_detect_markdown_file(self) -> None:
        """Test detection of .markdown file extension."""
        result = detect_source_format("Content", filename="doc.markdown")
        assert result == SeedSource.FILE

    def test_detect_text_source_no_filename(self) -> None:
        """Test detection of text source when no filename provided."""
        result = detect_source_format("Just some plain text content")
        assert result == SeedSource.TEXT

    def test_detect_url_http(self) -> None:
        """Test detection of http URL."""
        result = detect_source_format("http://example.com/seed.txt")
        assert result == SeedSource.URL

    def test_detect_url_https(self) -> None:
        """Test detection of https URL."""
        result = detect_source_format("https://example.com/requirements.md")
        assert result == SeedSource.URL

    def test_detect_text_with_url_in_content(self) -> None:
        """Test that URLs in content don't trigger URL detection."""
        content = "Build a web scraper that fetches https://example.com/api"
        result = detect_source_format(content)
        assert result == SeedSource.TEXT

    def test_detect_preserves_explicit_source(self) -> None:
        """Test that explicit source overrides detection."""
        # This tests the function behavior with filename present
        result = detect_source_format("content", filename="data.json")
        assert result == SeedSource.FILE


# =============================================================================
# normalize_content Tests
# =============================================================================


class TestNormalizeContent:
    """Tests for normalize_content utility."""

    def test_normalize_strips_leading_trailing_whitespace(self) -> None:
        """Test that leading/trailing whitespace is removed."""
        result = normalize_content("   Hello World   ")
        assert result == "Hello World"

    def test_normalize_preserves_internal_newlines(self) -> None:
        """Test that internal newlines are preserved."""
        content = "Line 1\nLine 2\nLine 3"
        result = normalize_content(content)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_normalize_converts_crlf_to_lf(self) -> None:
        """Test that Windows line endings are converted to Unix."""
        content = "Line 1\r\nLine 2\r\nLine 3"
        result = normalize_content(content)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_normalize_converts_cr_to_lf(self) -> None:
        """Test that old Mac line endings are converted to Unix."""
        content = "Line 1\rLine 2\rLine 3"
        result = normalize_content(content)
        assert result == "Line 1\nLine 2\nLine 3"

    def test_normalize_collapses_multiple_blank_lines(self) -> None:
        """Test that multiple blank lines are collapsed to one."""
        content = "Para 1\n\n\n\nPara 2"
        result = normalize_content(content)
        assert result == "Para 1\n\nPara 2"

    def test_normalize_handles_empty_string(self) -> None:
        """Test normalization of empty string."""
        result = normalize_content("")
        assert result == ""

    def test_normalize_handles_whitespace_only(self) -> None:
        """Test normalization of whitespace-only string."""
        result = normalize_content("   \n\n   \t   ")
        assert result == ""

    def test_normalize_handles_unicode(self) -> None:
        """Test normalization preserves unicode characters."""
        content = "Build app with emoji support"
        result = normalize_content(content)
        assert "" in result

    def test_normalize_removes_null_bytes(self) -> None:
        """Test that null bytes are removed."""
        content = "Hello\x00World"
        result = normalize_content(content)
        assert "\x00" not in result
        assert "Hello" in result
        assert "World" in result

    def test_normalize_preserves_code_blocks(self) -> None:
        """Test that code block formatting is preserved."""
        content = """```python
def hello():
    pass
```"""
        result = normalize_content(content)
        assert "```python" in result
        assert "def hello():" in result


# =============================================================================
# SeedParser Protocol Tests
# =============================================================================


class TestSeedParserProtocol:
    """Tests for SeedParser protocol compliance."""

    def test_parser_protocol_has_parse_method(self) -> None:
        """Test that SeedParser protocol defines parse method."""
        from yolo_developer.seed.parser import SeedParser

        # Check protocol has the parse method
        assert hasattr(SeedParser, "parse")

    def test_parser_protocol_is_runtime_checkable(self) -> None:
        """Test that SeedParser can be used for isinstance checks."""

        from yolo_developer.seed.parser import SeedParser

        # SeedParser should be a Protocol
        assert hasattr(SeedParser, "__protocol_attrs__") or hasattr(SeedParser, "_is_protocol")


# =============================================================================
# _parse_plain_text Tests (Task 7)
# =============================================================================


class TestParsePlainText:
    """Tests for _parse_plain_text preprocessor."""

    def test_parse_plain_text_adds_line_numbers(self) -> None:
        """Test that line numbers are added to non-empty lines."""
        content = "Line one\nLine two\nLine three"
        result = _parse_plain_text(content)
        assert "[L1]" in result
        assert "[L2]" in result
        assert "[L3]" in result

    def test_parse_plain_text_preserves_empty_lines(self) -> None:
        """Test that empty lines are preserved without annotation."""
        content = "Line one\n\nLine three"
        result = _parse_plain_text(content)
        lines = result.split("\n")
        assert lines[1] == ""  # Empty line preserved

    def test_parse_plain_text_handles_list_items(self) -> None:
        """Test that list items are recognized."""
        content = "Intro\n- Item 1\n- Item 2\n* Item 3"
        result = _parse_plain_text(content)
        assert "[L2]" in result
        assert "Item 1" in result

    def test_parse_plain_text_handles_numbered_lists(self) -> None:
        """Test that numbered lists are recognized."""
        content = "Steps:\n1. First\n2. Second\n3) Third"
        result = _parse_plain_text(content)
        assert "First" in result
        assert "Second" in result
        assert "Third" in result

    def test_parse_plain_text_empty_content(self) -> None:
        """Test parsing empty content."""
        result = _parse_plain_text("")
        assert result == ""

    def test_parse_plain_text_preserves_content(self) -> None:
        """Test that original content is preserved."""
        content = "Build an e-commerce platform"
        result = _parse_plain_text(content)
        assert "Build an e-commerce platform" in result


# =============================================================================
# _parse_markdown Tests (Task 8)
# =============================================================================


class TestParseMarkdown:
    """Tests for _parse_markdown preprocessor."""

    def test_parse_markdown_detects_headings(self) -> None:
        """Test that headings are annotated with level."""
        content = "# H1\n## H2\n### H3"
        result = _parse_markdown(content)
        assert "[H1]" in result
        assert "[H2]" in result
        assert "[H3]" in result

    def test_parse_markdown_detects_lists(self) -> None:
        """Test that list items are annotated."""
        content = "Features:\n- Feature 1\n- Feature 2"
        result = _parse_markdown(content)
        assert "[LIST]" in result

    def test_parse_markdown_detects_code_blocks(self) -> None:
        """Test that code blocks are annotated."""
        content = "```python\ncode here\n```"
        result = _parse_markdown(content)
        assert "[CODE_START]" in result
        assert "[CODE_END]" in result

    def test_parse_markdown_detects_emphasis(self) -> None:
        """Test that bold text is annotated."""
        content = "This is **important** text"
        result = _parse_markdown(content)
        assert "[EMPHASIS]" in result

    def test_parse_markdown_adds_line_numbers(self) -> None:
        """Test that line numbers are added."""
        content = "# Project\nDescription here"
        result = _parse_markdown(content)
        assert "[L1]" in result
        assert "[L2]" in result

    def test_parse_markdown_preserves_code_content(self) -> None:
        """Test that code block content is preserved."""
        content = "```python\ndef hello():\n    pass\n```"
        result = _parse_markdown(content)
        assert "def hello():" in result
        assert "pass" in result

    def test_parse_markdown_empty_content(self) -> None:
        """Test parsing empty content."""
        result = _parse_markdown("")
        assert result == ""

    def test_parse_markdown_preserves_empty_lines(self) -> None:
        """Test that empty lines are preserved."""
        content = "# Heading\n\nParagraph"
        result = _parse_markdown(content)
        lines = result.split("\n")
        assert lines[1] == ""  # Empty line preserved

    def test_parse_markdown_numbered_lists(self) -> None:
        """Test that numbered lists are detected."""
        content = "Steps:\n1. First\n2. Second"
        result = _parse_markdown(content)
        assert "[LIST]" in result


# =============================================================================
# LLMSeedParser Retry Logic Tests (Fix from Code Review)
# =============================================================================


def _create_mock_llm_response(content: str) -> MagicMock:
    """Create a mock LLM response with given content."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    return mock


class TestLLMSeedParserRetryLogic:
    """Tests for LLMSeedParser._call_llm retry behavior."""

    @pytest.mark.asyncio
    async def test_retry_on_json_decode_error_then_success(self) -> None:
        """Test that _call_llm retries on JSONDecodeError and succeeds."""
        from yolo_developer.seed import parser

        llm_parser = LLMSeedParser()
        valid_response = {"goals": [], "features": [], "constraints": []}

        call_count = 0

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _create_mock_llm_response("not json")
            return _create_mock_llm_response(json.dumps(valid_response))

        with patch.object(parser.litellm, "acompletion", mock_acompletion):
            result = await llm_parser._call_llm("test content")

            assert result == valid_response
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_json_decode_error(self) -> None:
        """Test that _call_llm raises after 3 failed attempts."""
        from yolo_developer.seed import parser

        llm_parser = LLMSeedParser()
        call_count = 0

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _create_mock_llm_response("not valid json")

        with patch.object(parser.litellm, "acompletion", mock_acompletion):
            with pytest.raises(json.JSONDecodeError):
                await llm_parser._call_llm("test content")

            # Should have tried 3 times (initial + 2 retries)
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self) -> None:
        """Test that successful call doesn't trigger retry."""
        from yolo_developer.seed import parser

        llm_parser = LLMSeedParser()
        valid_response = {"goals": [], "features": [], "constraints": []}
        call_count = 0

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _create_mock_llm_response(json.dumps(valid_response))

        with patch.object(parser.litellm, "acompletion", mock_acompletion):
            result = await llm_parser._call_llm("test content")

            assert result == valid_response
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_handles_json_in_code_block(self) -> None:
        """Test that JSON wrapped in markdown code block is parsed."""
        from yolo_developer.seed import parser

        llm_parser = LLMSeedParser()
        valid_response = {"goals": [], "features": [], "constraints": []}

        # Response wrapped in markdown code block
        wrapped = f"```json\n{json.dumps(valid_response)}\n```"

        async def mock_acompletion(*args, **kwargs):
            return _create_mock_llm_response(wrapped)

        with patch.object(parser.litellm, "acompletion", mock_acompletion):
            result = await llm_parser._call_llm("test content")

            assert result == valid_response
