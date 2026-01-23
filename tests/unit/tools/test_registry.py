"""Unit tests for tool registry (Issue #17)."""

from __future__ import annotations

import pytest

from yolo_developer.config import CLIToolConfig, ToolsConfig
from yolo_developer.tools.claude_code import ClaudeCodeClient
from yolo_developer.tools.registry import ToolRegistry


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_init_with_defaults(self) -> None:
        """Verify registry initialization with default config."""
        config = ToolsConfig()
        registry = ToolRegistry(config)
        assert registry.config == config
        assert registry.cwd is None

    def test_init_with_cwd(self) -> None:
        """Verify registry initialization with custom cwd."""
        config = ToolsConfig()
        registry = ToolRegistry(config, cwd="/tmp/project")
        assert registry.cwd == "/tmp/project"

    def test_available_none_enabled(self) -> None:
        """Verify available returns empty list when no tools enabled."""
        config = ToolsConfig()
        registry = ToolRegistry(config)
        assert registry.available() == []

    def test_available_claude_code_enabled(self) -> None:
        """Verify available includes claude_code when enabled."""
        config = ToolsConfig(
            claude_code=CLIToolConfig(enabled=True),
        )
        registry = ToolRegistry(config)
        assert "claude_code" in registry.available()

    def test_available_aider_enabled(self) -> None:
        """Verify available includes aider when enabled."""
        config = ToolsConfig(
            aider=CLIToolConfig(enabled=True),
        )
        registry = ToolRegistry(config)
        assert "aider" in registry.available()

    def test_available_multiple_enabled(self) -> None:
        """Verify available includes all enabled tools."""
        config = ToolsConfig(
            claude_code=CLIToolConfig(enabled=True),
            aider=CLIToolConfig(enabled=True),
        )
        registry = ToolRegistry(config)
        available = registry.available()
        assert "claude_code" in available
        assert "aider" in available

    def test_is_available_true(self) -> None:
        """Verify is_available returns True for enabled tool."""
        config = ToolsConfig(
            claude_code=CLIToolConfig(enabled=True),
        )
        registry = ToolRegistry(config)
        assert registry.is_available("claude_code") is True

    def test_is_available_false(self) -> None:
        """Verify is_available returns False for disabled tool."""
        config = ToolsConfig()
        registry = ToolRegistry(config)
        assert registry.is_available("claude_code") is False

    def test_is_available_unknown_tool(self) -> None:
        """Verify is_available returns False for unknown tool."""
        config = ToolsConfig()
        registry = ToolRegistry(config)
        assert registry.is_available("unknown_tool") is False


class TestToolRegistryGet:
    """Tests for ToolRegistry.get() method."""

    def test_get_disabled_tool_returns_none(self) -> None:
        """Verify get returns None for disabled tool."""
        config = ToolsConfig()
        registry = ToolRegistry(config)
        assert registry.get("claude_code") is None

    def test_get_unknown_tool_returns_none(self) -> None:
        """Verify get returns None for unknown tool."""
        config = ToolsConfig()
        registry = ToolRegistry(config)
        assert registry.get("unknown_tool") is None

    def test_get_claude_code_enabled(self) -> None:
        """Verify get returns ClaudeCodeClient when enabled."""
        config = ToolsConfig(
            claude_code=CLIToolConfig(enabled=True),
        )
        registry = ToolRegistry(config)
        tool = registry.get("claude_code")
        assert tool is not None
        assert isinstance(tool, ClaudeCodeClient)

    def test_get_caches_instance(self) -> None:
        """Verify get returns same instance on repeated calls."""
        config = ToolsConfig(
            claude_code=CLIToolConfig(enabled=True),
        )
        registry = ToolRegistry(config)
        tool1 = registry.get("claude_code")
        tool2 = registry.get("claude_code")
        assert tool1 is tool2

    def test_get_passes_cwd(self) -> None:
        """Verify get passes cwd to tool client."""
        config = ToolsConfig(
            claude_code=CLIToolConfig(enabled=True),
        )
        registry = ToolRegistry(config, cwd="/tmp/project")
        tool = registry.get("claude_code")
        assert tool is not None
        assert tool.cwd == "/tmp/project"


class TestToolRegistryGetOrRaise:
    """Tests for ToolRegistry.get_or_raise() method."""

    def test_get_or_raise_disabled_tool(self) -> None:
        """Verify get_or_raise raises for disabled tool."""
        config = ToolsConfig()
        registry = ToolRegistry(config)
        with pytest.raises(ValueError) as exc_info:
            registry.get_or_raise("claude_code")
        assert "claude_code" in str(exc_info.value)
        assert "not available" in str(exc_info.value)

    def test_get_or_raise_unknown_tool(self) -> None:
        """Verify get_or_raise raises for unknown tool."""
        config = ToolsConfig()
        registry = ToolRegistry(config)
        with pytest.raises(ValueError) as exc_info:
            registry.get_or_raise("unknown_tool")
        assert "unknown_tool" in str(exc_info.value)

    def test_get_or_raise_enabled_tool(self) -> None:
        """Verify get_or_raise returns tool when enabled."""
        config = ToolsConfig(
            claude_code=CLIToolConfig(enabled=True),
        )
        registry = ToolRegistry(config)
        tool = registry.get_or_raise("claude_code")
        assert isinstance(tool, ClaudeCodeClient)

    def test_get_or_raise_shows_available_tools(self) -> None:
        """Verify get_or_raise shows available tools in error."""
        config = ToolsConfig(
            aider=CLIToolConfig(enabled=True),
        )
        registry = ToolRegistry(config)
        with pytest.raises(ValueError) as exc_info:
            registry.get_or_raise("claude_code")
        assert "aider" in str(exc_info.value)


class TestToolRegistryContains:
    """Tests for ToolRegistry.__contains__ method."""

    def test_contains_enabled_tool(self) -> None:
        """Verify 'in' operator works for enabled tool."""
        config = ToolsConfig(
            claude_code=CLIToolConfig(enabled=True),
        )
        registry = ToolRegistry(config)
        assert "claude_code" in registry

    def test_contains_disabled_tool(self) -> None:
        """Verify 'in' operator works for disabled tool."""
        config = ToolsConfig()
        registry = ToolRegistry(config)
        assert "claude_code" not in registry


class TestToolRegistryIteration:
    """Tests for ToolRegistry iteration."""

    def test_iter_no_tools(self) -> None:
        """Verify iteration over empty registry."""
        config = ToolsConfig()
        registry = ToolRegistry(config)
        tools = list(registry)
        assert tools == []

    def test_iter_with_tools(self) -> None:
        """Verify iteration over registry with tools."""
        config = ToolsConfig(
            claude_code=CLIToolConfig(enabled=True),
            aider=CLIToolConfig(enabled=True),
        )
        registry = ToolRegistry(config)
        tools = list(registry)
        assert "claude_code" in tools
        assert "aider" in tools

    def test_iter_in_for_loop(self) -> None:
        """Verify registry can be used in for loop."""
        config = ToolsConfig(
            claude_code=CLIToolConfig(enabled=True),
        )
        registry = ToolRegistry(config)
        count = 0
        for tool_name in registry:
            assert tool_name == "claude_code"
            count += 1
        assert count == 1


class TestToolRegistryModuleExports:
    """Tests for tools module exports."""

    def test_tool_registry_importable(self) -> None:
        """Verify ToolRegistry can be imported from tools module."""
        from yolo_developer.tools import ToolRegistry

        assert ToolRegistry is not None

    def test_claude_code_client_importable(self) -> None:
        """Verify ClaudeCodeClient can be imported from tools module."""
        from yolo_developer.tools import ClaudeCodeClient

        assert ClaudeCodeClient is not None

    def test_base_classes_importable(self) -> None:
        """Verify base classes can be imported from tools module."""
        from yolo_developer.tools import (
            BaseCLITool,
            ToolDisabledError,
            ToolError,
            ToolNotFoundError,
            ToolResult,
        )

        assert BaseCLITool is not None
        assert ToolResult is not None
        assert ToolError is not None
        assert ToolNotFoundError is not None
        assert ToolDisabledError is not None
