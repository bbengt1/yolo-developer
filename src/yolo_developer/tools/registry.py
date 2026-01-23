"""Tool registry for managing external CLI tools.

This module provides a registry for lazily initializing and managing
external CLI tool integrations.

Example:
    >>> from yolo_developer.config import load_config
    >>> from yolo_developer.tools import ToolRegistry
    >>> config = load_config()
    >>> registry = ToolRegistry(config.tools)
    >>> claude = registry.get("claude_code")
    >>> if claude:
    ...     result = await claude.run("Analyze this code")
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from yolo_developer.tools.claude_code import ClaudeCodeClient

if TYPE_CHECKING:
    from yolo_developer.config import ToolsConfig
    from yolo_developer.tools.base import BaseCLITool


class ToolRegistry:
    """Registry for external CLI tools.

    Provides lazy initialization and lookup of CLI tool clients.
    Tools are only instantiated when first requested.

    Attributes:
        config: Tools configuration from YoloConfig.
        cwd: Working directory for tool execution.

    Example:
        >>> registry = ToolRegistry(config.tools, cwd="/path/to/project")
        >>> if "claude_code" in registry.available():
        ...     claude = registry.get("claude_code")
        ...     result = await claude.run("What files are in this project?")
    """

    # Registry of known tool names to their client classes
    _tool_classes: dict[str, type[BaseCLITool]] = {
        "claude_code": ClaudeCodeClient,
    }

    def __init__(self, config: ToolsConfig, cwd: str | None = None) -> None:
        """Initialize the tool registry.

        Args:
            config: Tools configuration from YoloConfig.tools.
            cwd: Working directory for tool execution.
        """
        self.config = config
        self.cwd = cwd
        self._tools: dict[str, BaseCLITool] = {}

    def get(self, name: str) -> BaseCLITool | None:
        """Get a tool by name, lazily initializing if needed.

        Only returns the tool if it is enabled in configuration.
        Returns None if the tool is disabled or unknown.

        Args:
            name: Tool name (e.g., "claude_code", "aider").

        Returns:
            Tool client instance if enabled, None otherwise.

        Example:
            >>> claude = registry.get("claude_code")
            >>> if claude:
            ...     result = await claude.run("Analyze this code")
        """
        if name in self._tools:
            return self._tools[name]

        tool = self._create_tool(name)
        if tool is not None:
            self._tools[name] = tool

        return tool

    def _create_tool(self, name: str) -> BaseCLITool | None:
        """Create a tool instance if enabled.

        Args:
            name: Tool name.

        Returns:
            Tool instance if enabled, None otherwise.
        """
        if name == "claude_code":
            if self.config.claude_code.enabled:
                return ClaudeCodeClient(self.config.claude_code, self.cwd)
        elif name == "aider":
            # Aider support can be added here when implemented
            # if self.config.aider.enabled:
            #     return AiderClient(self.config.aider, self.cwd)
            pass

        return None

    def available(self) -> list[str]:
        """List available (enabled) tools.

        Returns:
            List of tool names that are enabled in configuration.

        Example:
            >>> tools = registry.available()
            >>> print(f"Available tools: {tools}")
        """
        tools: list[str] = []

        if self.config.claude_code.enabled:
            tools.append("claude_code")

        if self.config.aider.enabled:
            tools.append("aider")

        return tools

    def is_available(self, name: str) -> bool:
        """Check if a specific tool is available.

        A tool is available if it is enabled in configuration.

        Args:
            name: Tool name to check.

        Returns:
            True if the tool is enabled, False otherwise.

        Example:
            >>> if registry.is_available("claude_code"):
            ...     claude = registry.get("claude_code")
        """
        return name in self.available()

    def get_or_raise(self, name: str) -> BaseCLITool:
        """Get a tool by name, raising if unavailable.

        Args:
            name: Tool name.

        Returns:
            Tool client instance.

        Raises:
            ValueError: If the tool is not available.

        Example:
            >>> try:
            ...     claude = registry.get_or_raise("claude_code")
            ... except ValueError as e:
            ...     print(f"Tool not available: {e}")
        """
        tool = self.get(name)
        if tool is None:
            available = self.available()
            if available:
                msg = f"Tool '{name}' is not available. Available tools: {available}"
            else:
                msg = f"Tool '{name}' is not available. No tools are enabled."
            raise ValueError(msg)
        return tool

    def __contains__(self, name: str) -> bool:
        """Check if a tool is available using 'in' operator.

        Example:
            >>> if "claude_code" in registry:
            ...     claude = registry.get("claude_code")
        """
        return self.is_available(name)

    def __iter__(self) -> Iterator[str]:
        """Iterate over available tool names.

        Example:
            >>> for tool_name in registry:
            ...     tool = registry.get(tool_name)
        """
        return iter(self.available())
