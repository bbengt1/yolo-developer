"""Unit tests for Claude Code client (Issue #17)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from yolo_developer.config import CLIToolConfig
from yolo_developer.tools.claude_code import ClaudeCodeClient


class TestClaudeCodeClient:
    """Tests for ClaudeCodeClient."""

    def test_binary_name(self) -> None:
        """Verify binary name is claude."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)
        assert client.binary_name == "claude"

    def test_build_args_basic(self) -> None:
        """Verify build_args creates correct basic arguments."""
        config = CLIToolConfig(enabled=True, output_format="json")
        client = ClaudeCodeClient(config)
        args = client.build_args("test prompt")

        assert "--print" in args
        assert "--output-format=json" in args
        assert "--prompt" in args
        assert "test prompt" in args

    def test_build_args_text_format(self) -> None:
        """Verify build_args handles text output format."""
        config = CLIToolConfig(enabled=True, output_format="text")
        client = ClaudeCodeClient(config)
        args = client.build_args("test prompt")

        assert "--print" in args
        assert "--output-format=json" not in args

    def test_build_args_plan_mode(self) -> None:
        """Verify build_args adds plan flag."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)
        args = client.build_args("test prompt", plan_mode=True)

        assert "--plan" in args

    def test_build_args_resume(self) -> None:
        """Verify build_args adds resume flag."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)
        args = client.build_args("test prompt", resume="session-123")

        assert "--resume" in args
        assert "session-123" in args

    def test_build_args_model(self) -> None:
        """Verify build_args adds model flag."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)
        args = client.build_args("test prompt", model="opus")

        assert "--model" in args
        assert "opus" in args

    def test_build_args_allowed_tools(self) -> None:
        """Verify build_args adds allowedTools flag."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)
        args = client.build_args("test prompt", allowedTools=["Read", "Write", "Bash"])

        assert "--allowedTools" in args
        assert "Read,Write,Bash" in args

    def test_build_args_max_turns(self) -> None:
        """Verify build_args adds max-turns flag."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)
        args = client.build_args("test prompt", max_turns=5)

        assert "--max-turns" in args
        assert "5" in args

    def test_build_args_extra_args(self) -> None:
        """Verify build_args includes extra_args from config."""
        config = CLIToolConfig(
            enabled=True,
            extra_args=["--dangerously-skip-permissions"],
        )
        client = ClaudeCodeClient(config)
        args = client.build_args("test prompt")

        assert "--dangerously-skip-permissions" in args

    def test_build_args_prompt_is_last(self) -> None:
        """Verify prompt is at the end of args."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)
        args = client.build_args("my prompt", plan_mode=True, model="sonnet")

        # Find the position of --prompt
        prompt_idx = args.index("--prompt")
        assert prompt_idx == len(args) - 2
        assert args[-1] == "my prompt"


class TestClaudeCodeClientMethods:
    """Tests for ClaudeCodeClient convenience methods."""

    @pytest.mark.asyncio
    async def test_implement_basic(self) -> None:
        """Verify implement method calls run with plan mode."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            result = await client.implement("Add authentication")

            assert result.success is True
            # Verify plan mode is enabled by default
            call_args = mock_exec.call_args[0]
            assert "--plan" in call_args

    @pytest.mark.asyncio
    async def test_implement_with_context(self) -> None:
        """Verify implement method prepends context."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            await client.implement(
                "Add authentication",
                context="This is a Python web app",
            )

            # Verify prompt includes context
            call_args = mock_exec.call_args[0]
            prompt_idx = call_args.index("--prompt")
            prompt = call_args[prompt_idx + 1]
            assert "This is a Python web app" in prompt
            assert "Add authentication" in prompt

    @pytest.mark.asyncio
    async def test_analyze_basic(self) -> None:
        """Verify analyze method creates correct prompt."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            await client.analyze("How is auth implemented?")

            call_args = mock_exec.call_args[0]
            prompt_idx = call_args.index("--prompt")
            prompt = call_args[prompt_idx + 1]
            assert "Analyze:" in prompt

    @pytest.mark.asyncio
    async def test_analyze_with_scope(self) -> None:
        """Verify analyze method includes scope."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            await client.analyze("How is auth implemented?", scope="src/auth/")

            call_args = mock_exec.call_args[0]
            prompt_idx = call_args.index("--prompt")
            prompt = call_args[prompt_idx + 1]
            assert "Scope: src/auth/" in prompt

    @pytest.mark.asyncio
    async def test_test_basic(self) -> None:
        """Verify test method creates correct prompt."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            await client.test("tests/unit/")

            call_args = mock_exec.call_args[0]
            prompt_idx = call_args.index("--prompt")
            prompt = call_args[prompt_idx + 1]
            assert "Run tests" in prompt
            assert "tests/unit/" in prompt

    @pytest.mark.asyncio
    async def test_test_with_fix_failures(self) -> None:
        """Verify test method includes fix instruction."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            await client.test("tests/unit/", fix_failures=True)

            call_args = mock_exec.call_args[0]
            prompt_idx = call_args.index("--prompt")
            prompt = call_args[prompt_idx + 1]
            assert "fix" in prompt.lower()

    @pytest.mark.asyncio
    async def test_review_basic(self) -> None:
        """Verify review method creates correct prompt."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            await client.review("src/tools/")

            call_args = mock_exec.call_args[0]
            prompt_idx = call_args.index("--prompt")
            prompt = call_args[prompt_idx + 1]
            assert "Review:" in prompt

    @pytest.mark.asyncio
    async def test_review_with_focus(self) -> None:
        """Verify review method includes focus areas."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            await client.review("src/tools/", focus="security")

            call_args = mock_exec.call_args[0]
            prompt_idx = call_args.index("--prompt")
            prompt = call_args[prompt_idx + 1]
            assert "Focus on: security" in prompt

    @pytest.mark.asyncio
    async def test_refactor_basic(self) -> None:
        """Verify refactor method creates correct prompt."""
        config = CLIToolConfig(enabled=True)
        client = ClaudeCodeClient(config)

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            await client.refactor("src/config/loader.py", "Extract validation")

            call_args = mock_exec.call_args[0]
            prompt_idx = call_args.index("--prompt")
            prompt = call_args[prompt_idx + 1]
            assert "Refactor" in prompt
            assert "src/config/loader.py" in prompt
            assert "Extract validation" in prompt
            # Plan mode should be enabled by default
            assert "--plan" in call_args
