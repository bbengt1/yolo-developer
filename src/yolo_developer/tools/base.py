"""Base classes for external CLI tool integration.

This module provides the abstract base class and result types for integrating
with external CLI-based AI development tools like Claude Code and Aider.

Example:
    >>> from yolo_developer.tools.base import BaseCLITool, ToolResult
    >>> class MyTool(BaseCLITool):
    ...     @property
    ...     def binary_name(self) -> str:
    ...         return "mytool"
    ...     def build_args(self, prompt: str, **kwargs) -> list[str]:
    ...         return ["--prompt", prompt]
"""

from __future__ import annotations

import asyncio
import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yolo_developer.config import CLIToolConfig


@dataclass
class ToolResult:
    """Result from CLI tool execution.

    Attributes:
        success: Whether the tool execution succeeded (exit code 0).
        output: Raw stdout output from the tool.
        parsed: Parsed JSON output if output_format is json and parsing succeeded.
        exit_code: Process exit code (0 for success).
        stderr: Raw stderr output from the tool.

    Example:
        >>> result = ToolResult(success=True, output='{"answer": 42}')
        >>> result.success
        True
    """

    success: bool
    output: str
    parsed: dict[str, Any] | None = None
    exit_code: int = 0
    stderr: str = ""


@dataclass
class ToolError:
    """Error details from CLI tool execution.

    Attributes:
        code: Error code or category.
        message: Human-readable error message.
        details: Additional error context.

    Example:
        >>> error = ToolError(code="TIMEOUT", message="Tool timed out after 300s")
    """

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class ToolNotFoundError(FileNotFoundError):
    """Raised when a CLI tool binary is not found in PATH."""

    def __init__(self, binary_name: str, path: str | None = None) -> None:
        self.binary_name = binary_name
        self.path = path
        if path:
            msg = f"Tool binary not found at specified path: {path}"
        else:
            msg = f"Tool binary '{binary_name}' not found in PATH"
        super().__init__(msg)


class ToolDisabledError(RuntimeError):
    """Raised when attempting to use a disabled tool."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' is disabled in configuration")


class BaseCLITool(ABC):
    """Abstract base class for external CLI tool integration.

    Provides a common interface for wrapping CLI-based AI development tools.
    Subclasses must implement `binary_name` and `build_args` methods.

    Attributes:
        config: Tool configuration from YoloConfig.
        cwd: Working directory for tool execution.

    Example:
        >>> class ClaudeCode(BaseCLITool):
        ...     @property
        ...     def binary_name(self) -> str:
        ...         return "claude"
        ...     def build_args(self, prompt: str, **kwargs) -> list[str]:
        ...         return ["--print", "--prompt", prompt]
    """

    def __init__(self, config: CLIToolConfig, cwd: str | None = None) -> None:
        """Initialize the CLI tool wrapper.

        Args:
            config: Tool configuration from YoloConfig.tools.
            cwd: Working directory for tool execution. Defaults to current directory.
        """
        self.config = config
        self.cwd = cwd
        self._binary_path: str | None = None

    @property
    @abstractmethod
    def binary_name(self) -> str:
        """Name of the CLI binary to execute.

        Returns:
            The binary name (e.g., "claude", "aider").
        """
        ...

    @property
    def binary_path(self) -> str:
        """Resolved path to the CLI binary.

        Uses configured path if set, otherwise searches PATH.

        Returns:
            Full path to the binary.

        Raises:
            ToolNotFoundError: If the binary is not found.
        """
        if self._binary_path is None:
            if self.config.path:
                self._binary_path = self.config.path
            else:
                found = shutil.which(self.binary_name)
                if found is None:
                    raise ToolNotFoundError(self.binary_name)
                self._binary_path = found
        return self._binary_path

    @property
    def is_available(self) -> bool:
        """Check if the tool binary is available.

        Returns:
            True if the binary exists and is executable.
        """
        try:
            _ = self.binary_path
            return True
        except ToolNotFoundError:
            return False

    @abstractmethod
    def build_args(self, prompt: str, **kwargs: Any) -> list[str]:
        """Build command-line arguments for the tool.

        Args:
            prompt: The prompt or task to send to the tool.
            **kwargs: Additional options for argument building.

        Returns:
            List of command-line arguments (excluding the binary itself).
        """
        ...

    async def run(self, prompt: str, **kwargs: Any) -> ToolResult:
        """Execute the CLI tool with the given prompt.

        Args:
            prompt: The prompt or task to send to the tool.
            **kwargs: Additional options passed to build_args.

        Returns:
            ToolResult with execution outcome.

        Note:
            Returns a failed ToolResult instead of raising exceptions
            for runtime errors (timeout, execution failure). Only raises
            ToolDisabledError if the tool is disabled in config.
        """
        if not self.config.enabled:
            return ToolResult(
                success=False,
                output="",
                exit_code=-1,
                stderr=f"Tool '{self.binary_name}' is disabled in configuration",
            )

        try:
            binary = self.binary_path
        except ToolNotFoundError as e:
            return ToolResult(
                success=False,
                output="",
                exit_code=-1,
                stderr=str(e),
            )

        args = [binary, *self.build_args(prompt, **kwargs)]

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=self.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.timeout,
            )
        except TimeoutError:
            return ToolResult(
                success=False,
                output="",
                exit_code=-1,
                stderr=f"Tool timed out after {self.config.timeout}s",
            )
        except OSError as e:
            return ToolResult(
                success=False,
                output="",
                exit_code=-1,
                stderr=f"Failed to execute tool: {e}",
            )

        output = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")

        parsed = None
        if self.config.output_format == "json" and proc.returncode == 0:
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                # Output wasn't valid JSON, leave parsed as None
                pass

        return ToolResult(
            success=proc.returncode == 0,
            output=output,
            parsed=parsed,
            exit_code=proc.returncode or 0,
            stderr=stderr_str,
        )

    async def run_or_raise(self, prompt: str, **kwargs: Any) -> ToolResult:
        """Execute the CLI tool, raising on failure.

        Like run(), but raises exceptions for failures instead of
        returning failed ToolResults.

        Args:
            prompt: The prompt or task to send to the tool.
            **kwargs: Additional options passed to build_args.

        Returns:
            ToolResult with successful execution outcome.

        Raises:
            ToolDisabledError: If the tool is disabled.
            ToolNotFoundError: If the binary is not found.
            RuntimeError: If execution fails.
        """
        if not self.config.enabled:
            raise ToolDisabledError(self.binary_name)

        # This will raise ToolNotFoundError if not found
        _ = self.binary_path

        result = await self.run(prompt, **kwargs)

        if not result.success:
            raise RuntimeError(f"Tool execution failed (exit {result.exit_code}): {result.stderr}")

        return result
