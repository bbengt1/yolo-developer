"""Unit tests for tools base module (Issue #17)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from yolo_developer.config import CLIToolConfig
from yolo_developer.tools.base import (
    BaseCLITool,
    ToolDisabledError,
    ToolNotFoundError,
    ToolResult,
)


class DummyTool(BaseCLITool):
    """Dummy tool for testing BaseCLITool."""

    @property
    def binary_name(self) -> str:
        return "dummy"

    def build_args(self, prompt: str, **kwargs: Any) -> list[str]:
        args = ["--prompt", prompt]
        if kwargs.get("verbose"):
            args.append("--verbose")
        return args


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_tool_result_success(self) -> None:
        """Verify ToolResult can represent success."""
        result = ToolResult(success=True, output="Hello, world!")
        assert result.success is True
        assert result.output == "Hello, world!"
        assert result.parsed is None
        assert result.exit_code == 0
        assert result.stderr == ""

    def test_tool_result_failure(self) -> None:
        """Verify ToolResult can represent failure."""
        result = ToolResult(
            success=False,
            output="",
            exit_code=1,
            stderr="Error: command not found",
        )
        assert result.success is False
        assert result.exit_code == 1
        assert result.stderr == "Error: command not found"

    def test_tool_result_with_parsed_json(self) -> None:
        """Verify ToolResult can hold parsed JSON."""
        result = ToolResult(
            success=True,
            output='{"answer": 42}',
            parsed={"answer": 42},
        )
        assert result.parsed == {"answer": 42}


class TestToolNotFoundError:
    """Tests for ToolNotFoundError."""

    def test_error_without_path(self) -> None:
        """Verify error message when binary not in PATH."""
        error = ToolNotFoundError("mytool")
        assert error.binary_name == "mytool"
        assert error.path is None
        assert "mytool" in str(error)
        assert "not found in PATH" in str(error)

    def test_error_with_path(self) -> None:
        """Verify error message when binary not at specified path."""
        error = ToolNotFoundError("mytool", "/usr/bin/mytool")
        assert error.binary_name == "mytool"
        assert error.path == "/usr/bin/mytool"
        assert "/usr/bin/mytool" in str(error)


class TestToolDisabledError:
    """Tests for ToolDisabledError."""

    def test_error_message(self) -> None:
        """Verify error message includes tool name."""
        error = ToolDisabledError("claude")
        assert error.tool_name == "claude"
        assert "claude" in str(error)
        assert "disabled" in str(error)


class TestBaseCLITool:
    """Tests for BaseCLITool abstract class."""

    def test_init_with_defaults(self) -> None:
        """Verify tool initialization with default config."""
        config = CLIToolConfig(enabled=True)
        tool = DummyTool(config)
        assert tool.config == config
        assert tool.cwd is None

    def test_init_with_cwd(self) -> None:
        """Verify tool initialization with custom cwd."""
        config = CLIToolConfig(enabled=True)
        tool = DummyTool(config, cwd="/tmp/project")
        assert tool.cwd == "/tmp/project"

    def test_binary_name_property(self) -> None:
        """Verify binary_name is implemented by subclass."""
        config = CLIToolConfig(enabled=True)
        tool = DummyTool(config)
        assert tool.binary_name == "dummy"

    def test_build_args_basic(self) -> None:
        """Verify build_args creates correct arguments."""
        config = CLIToolConfig(enabled=True)
        tool = DummyTool(config)
        args = tool.build_args("test prompt")
        assert args == ["--prompt", "test prompt"]

    def test_build_args_with_kwargs(self) -> None:
        """Verify build_args handles kwargs."""
        config = CLIToolConfig(enabled=True)
        tool = DummyTool(config)
        args = tool.build_args("test prompt", verbose=True)
        assert "--verbose" in args

    def test_binary_path_with_config_path(self) -> None:
        """Verify binary_path uses config path when set."""
        config = CLIToolConfig(enabled=True, path="/custom/path/dummy")
        tool = DummyTool(config)
        assert tool.binary_path == "/custom/path/dummy"

    def test_binary_path_not_found_raises(self) -> None:
        """Verify binary_path raises when not in PATH."""
        config = CLIToolConfig(enabled=True)
        tool = DummyTool(config)
        with patch("shutil.which", return_value=None):
            with pytest.raises(ToolNotFoundError) as exc_info:
                _ = tool.binary_path
            assert exc_info.value.binary_name == "dummy"

    def test_binary_path_found_in_path(self) -> None:
        """Verify binary_path finds binary in PATH."""
        config = CLIToolConfig(enabled=True)
        tool = DummyTool(config)
        with patch("shutil.which", return_value="/usr/bin/dummy"):
            assert tool.binary_path == "/usr/bin/dummy"

    def test_is_available_true(self) -> None:
        """Verify is_available returns True when binary found."""
        config = CLIToolConfig(enabled=True, path="/usr/bin/dummy")
        tool = DummyTool(config)
        assert tool.is_available is True

    def test_is_available_false(self) -> None:
        """Verify is_available returns False when binary not found."""
        config = CLIToolConfig(enabled=True)
        tool = DummyTool(config)
        with patch("shutil.which", return_value=None):
            assert tool.is_available is False


class TestBaseCLIToolRun:
    """Tests for BaseCLITool.run() method."""

    @pytest.mark.asyncio
    async def test_run_disabled_tool(self) -> None:
        """Verify run returns failure for disabled tool."""
        config = CLIToolConfig(enabled=False)
        tool = DummyTool(config)
        result = await tool.run("test")
        assert result.success is False
        assert "disabled" in result.stderr

    @pytest.mark.asyncio
    async def test_run_binary_not_found(self) -> None:
        """Verify run returns failure when binary not found."""
        config = CLIToolConfig(enabled=True)
        tool = DummyTool(config)
        with patch("shutil.which", return_value=None):
            result = await tool.run("test")
            assert result.success is False
            assert "not found" in result.stderr

    @pytest.mark.asyncio
    async def test_run_success(self) -> None:
        """Verify run returns success on successful execution."""
        config = CLIToolConfig(enabled=True, output_format="text")
        tool = DummyTool(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/dummy"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await tool.run("test prompt")
            assert result.success is True
            assert result.output == "output"
            assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_run_failure(self) -> None:
        """Verify run returns failure on non-zero exit."""
        config = CLIToolConfig(enabled=True, output_format="text")
        tool = DummyTool(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error message"))
        mock_proc.returncode = 1

        with (
            patch("shutil.which", return_value="/usr/bin/dummy"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await tool.run("test prompt")
            assert result.success is False
            assert result.exit_code == 1
            assert result.stderr == "error message"

    @pytest.mark.asyncio
    async def test_run_timeout(self) -> None:
        """Verify run handles timeout."""
        config = CLIToolConfig(enabled=True, timeout=1)
        tool = DummyTool(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError)

        with (
            patch("shutil.which", return_value="/usr/bin/dummy"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await tool.run("test prompt")
            assert result.success is False
            assert "timed out" in result.stderr

    @pytest.mark.asyncio
    async def test_run_json_parsing(self) -> None:
        """Verify run parses JSON output when configured."""
        config = CLIToolConfig(enabled=True, output_format="json")
        tool = DummyTool(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b'{"result": "success"}', b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/dummy"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await tool.run("test prompt")
            assert result.success is True
            assert result.parsed == {"result": "success"}

    @pytest.mark.asyncio
    async def test_run_json_parsing_failure(self) -> None:
        """Verify run handles invalid JSON gracefully."""
        config = CLIToolConfig(enabled=True, output_format="json")
        tool = DummyTool(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"not json", b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/dummy"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await tool.run("test prompt")
            assert result.success is True
            assert result.parsed is None
            assert result.output == "not json"


class TestBaseCLIToolRunOrRaise:
    """Tests for BaseCLITool.run_or_raise() method."""

    @pytest.mark.asyncio
    async def test_run_or_raise_disabled(self) -> None:
        """Verify run_or_raise raises for disabled tool."""
        config = CLIToolConfig(enabled=False)
        tool = DummyTool(config)
        with pytest.raises(ToolDisabledError):
            await tool.run_or_raise("test")

    @pytest.mark.asyncio
    async def test_run_or_raise_not_found(self) -> None:
        """Verify run_or_raise raises for missing binary."""
        config = CLIToolConfig(enabled=True)
        tool = DummyTool(config)
        with patch("shutil.which", return_value=None):
            with pytest.raises(ToolNotFoundError):
                await tool.run_or_raise("test")

    @pytest.mark.asyncio
    async def test_run_or_raise_failure(self) -> None:
        """Verify run_or_raise raises on execution failure."""
        config = CLIToolConfig(enabled=True, output_format="text")
        tool = DummyTool(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))
        mock_proc.returncode = 1

        with (
            patch("shutil.which", return_value="/usr/bin/dummy"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await tool.run_or_raise("test")
            assert "exit 1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_or_raise_success(self) -> None:
        """Verify run_or_raise returns result on success."""
        config = CLIToolConfig(enabled=True, output_format="text")
        tool = DummyTool(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/dummy"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = await tool.run_or_raise("test")
            assert result.success is True
