"""Unit tests for seed API module (Story 4.1 - Code Review Fix 6).

Tests the _looks_like_markdown helper function.
"""

from yolo_developer.seed.api import _looks_like_markdown

# =============================================================================
# _looks_like_markdown Tests
# =============================================================================


class TestLooksLikeMarkdown:
    """Tests for _looks_like_markdown detection function."""

    def test_detects_headings(self) -> None:
        """Test that multiple headings trigger markdown detection."""
        content = "# Heading 1\n## Heading 2\nSome text"
        assert _looks_like_markdown(content) is True

    def test_detects_code_blocks(self) -> None:
        """Test that code blocks trigger markdown detection."""
        content = "Some text\n```python\ncode here\n```\nMore text\n```\nmore code\n```"
        assert _looks_like_markdown(content) is True

    def test_detects_bold_text(self) -> None:
        """Test that bold markers trigger markdown detection."""
        content = "This is **bold** text\nAnd this is **also bold**\nMore text"
        assert _looks_like_markdown(content) is True

    def test_detects_underscore_bold(self) -> None:
        """Test that underscore bold markers trigger markdown detection."""
        content = "This is __bold__ text\nAnd this is __also bold__"
        assert _looks_like_markdown(content) is True

    def test_detects_blockquotes(self) -> None:
        """Test that blockquotes trigger markdown detection."""
        content = "> Quote 1\n> Quote 2\n> Quote 3"
        assert _looks_like_markdown(content) is True

    def test_mixed_markdown_indicators(self) -> None:
        """Test that mixed indicators trigger detection."""
        content = "# Heading\n**Bold text**\nSome plain text"
        assert _looks_like_markdown(content) is True

    def test_plain_text_not_detected(self) -> None:
        """Test that plain text is not detected as markdown."""
        content = "This is just plain text\nWith multiple lines\nBut no markdown"
        assert _looks_like_markdown(content) is False

    def test_single_heading_not_detected(self) -> None:
        """Test that a single markdown indicator is not enough."""
        content = "# Just one heading\nBut no other markdown"
        assert _looks_like_markdown(content) is False

    def test_empty_content(self) -> None:
        """Test that empty content is not detected as markdown."""
        assert _looks_like_markdown("") is False

    def test_whitespace_only(self) -> None:
        """Test that whitespace-only content is not detected."""
        assert _looks_like_markdown("   \n\n   ") is False

    def test_list_items_not_counted(self) -> None:
        """Test that list items alone don't trigger detection."""
        content = "- Item 1\n- Item 2\n- Item 3"
        # Lists are not counted as markdown indicators in the current implementation
        assert _looks_like_markdown(content) is False

    def test_code_block_with_heading(self) -> None:
        """Test combination of code block and heading."""
        content = "# API Reference\n\n```python\ndef hello():\n    pass\n```"
        assert _looks_like_markdown(content) is True

    def test_threshold_exactly_two(self) -> None:
        """Test that exactly 2 indicators triggers detection."""
        content = "# Heading\n**Bold**"
        assert _looks_like_markdown(content) is True

    def test_threshold_less_than_two(self) -> None:
        """Test that fewer than 2 indicators doesn't trigger."""
        content = "# Just one heading"
        assert _looks_like_markdown(content) is False
