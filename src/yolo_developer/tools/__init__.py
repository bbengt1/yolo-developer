"""External CLI tool integrations for YOLO Developer.

This module provides interfaces for integrating with external CLI-based
AI development tools like Claude Code and Aider.

Example:
    >>> from yolo_developer.tools import ClaudeCodeClient, ToolRegistry
    >>> from yolo_developer.config import load_config
    >>> config = load_config()
    >>> registry = ToolRegistry(config.tools)
    >>> claude = registry.get("claude_code")
    >>> if claude:
    ...     result = await claude.run("Analyze this code")
"""

from __future__ import annotations

from yolo_developer.tools.base import (
    BaseCLITool,
    ToolDisabledError,
    ToolError,
    ToolNotFoundError,
    ToolResult,
)
from yolo_developer.tools.claude_code import ClaudeCodeClient
from yolo_developer.tools.registry import ToolRegistry

__all__ = [
    "BaseCLITool",
    "ClaudeCodeClient",
    "ToolDisabledError",
    "ToolError",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolResult",
]
