"""Claude Code CLI integration.

This module provides a client for integrating with Claude Code CLI,
enabling YOLO Developer to delegate complex tasks to Claude Code.

Example:
    >>> from yolo_developer.config import CLIToolConfig
    >>> from yolo_developer.tools.claude_code import ClaudeCodeClient
    >>> config = CLIToolConfig(enabled=True, timeout=600)
    >>> client = ClaudeCodeClient(config)
    >>> result = await client.run("Analyze this codebase")
    >>> if result.success:
    ...     print(result.output)
"""

from __future__ import annotations

from typing import Any

from yolo_developer.tools.base import BaseCLITool, ToolResult


class ClaudeCodeClient(BaseCLITool):
    """Client for Claude Code CLI integration.

    Wraps the Claude Code CLI to enable programmatic access to Claude Code's
    capabilities including code analysis, implementation, and testing.

    Example:
        >>> from yolo_developer.config import CLIToolConfig
        >>> config = CLIToolConfig(enabled=True, timeout=600)
        >>> client = ClaudeCodeClient(config, cwd="/path/to/project")
        >>> result = await client.implement("Add user authentication")
    """

    @property
    def binary_name(self) -> str:
        """Return the Claude Code binary name."""
        return "claude"

    def build_args(self, prompt: str, **kwargs: Any) -> list[str]:
        """Build command-line arguments for Claude Code.

        Args:
            prompt: The prompt to send to Claude Code.
            **kwargs: Additional options:
                - plan_mode (bool): Enter plan mode for structured planning.
                - resume (str): Session ID to resume.
                - model (str): Model to use (e.g., "sonnet", "opus").
                - allowedTools (list[str]): Restrict available tools.
                - max_turns (int): Maximum conversation turns.

        Returns:
            List of command-line arguments.
        """
        args = ["--print"]

        # Output format
        if self.config.output_format == "json":
            args.append("--output-format=json")

        # Plan mode for structured implementation planning
        if kwargs.get("plan_mode"):
            args.append("--plan")

        # Resume existing session
        if kwargs.get("resume"):
            args.extend(["--resume", kwargs["resume"]])

        # Model selection
        if kwargs.get("model"):
            args.extend(["--model", kwargs["model"]])

        # Tool restrictions
        if kwargs.get("allowedTools"):
            tools = kwargs["allowedTools"]
            if isinstance(tools, list):
                args.extend(["--allowedTools", ",".join(tools)])

        # Max turns limit
        if kwargs.get("max_turns"):
            args.extend(["--max-turns", str(kwargs["max_turns"])])

        # Add any extra args from config
        args.extend(self.config.extra_args)

        # Prompt must be last
        args.extend(["--prompt", prompt])

        return args

    async def implement(
        self,
        task: str,
        *,
        plan_mode: bool = True,
        context: str | None = None,
        model: str | None = None,
    ) -> ToolResult:
        """Request implementation of a task.

        Sends an implementation request to Claude Code, optionally using
        plan mode for structured planning before execution.

        Args:
            task: Description of the task to implement.
            plan_mode: Use plan mode for structured planning (default True).
            context: Additional context to prepend to the task.
            model: Model to use for this request.

        Returns:
            ToolResult with the implementation outcome.

        Example:
            >>> result = await client.implement(
            ...     "Add user authentication with JWT tokens",
            ...     plan_mode=True,
            ... )
        """
        prompt = task
        if context:
            prompt = f"{context}\n\n{task}"

        return await self.run(prompt, plan_mode=plan_mode, model=model)

    async def analyze(
        self,
        query: str,
        *,
        scope: str | None = None,
        model: str | None = None,
    ) -> ToolResult:
        """Request code analysis.

        Sends an analysis request to Claude Code for understanding
        code structure, patterns, or behavior.

        Args:
            query: The analysis question or request.
            scope: Optional scope restriction (e.g., file paths, modules).
            model: Model to use for this request.

        Returns:
            ToolResult with the analysis outcome.

        Example:
            >>> result = await client.analyze(
            ...     "How is authentication implemented in this codebase?"
            ... )
        """
        prompt = f"Analyze: {query}"
        if scope:
            prompt = f"{prompt}\n\nScope: {scope}"

        return await self.run(prompt, model=model)

    async def test(
        self,
        scope: str,
        *,
        fix_failures: bool = False,
        model: str | None = None,
    ) -> ToolResult:
        """Run tests and analyze results.

        Requests Claude Code to run tests for the specified scope
        and report results.

        Args:
            scope: Test scope (e.g., file path, test name pattern).
            fix_failures: Whether to attempt fixing failures.
            model: Model to use for this request.

        Returns:
            ToolResult with test execution and analysis.

        Example:
            >>> result = await client.test(
            ...     "tests/unit/config/",
            ...     fix_failures=True,
            ... )
        """
        if fix_failures:
            prompt = f"Run tests for {scope}, analyze any failures, and fix them"
        else:
            prompt = f"Run tests for {scope} and report results"

        return await self.run(prompt, model=model)

    async def review(
        self,
        target: str,
        *,
        focus: str | None = None,
        model: str | None = None,
    ) -> ToolResult:
        """Request code review.

        Sends a code review request to Claude Code for the specified target.

        Args:
            target: What to review (e.g., file path, PR number, diff).
            focus: Optional focus areas (e.g., "security", "performance").
            model: Model to use for this request.

        Returns:
            ToolResult with review feedback.

        Example:
            >>> result = await client.review(
            ...     "src/yolo_developer/tools/",
            ...     focus="security and error handling",
            ... )
        """
        prompt = f"Review: {target}"
        if focus:
            prompt = f"{prompt}\n\nFocus on: {focus}"

        return await self.run(prompt, model=model)

    async def refactor(
        self,
        target: str,
        goal: str,
        *,
        plan_mode: bool = True,
        model: str | None = None,
    ) -> ToolResult:
        """Request code refactoring.

        Sends a refactoring request to Claude Code with a specific goal.

        Args:
            target: What to refactor (e.g., file path, function name).
            goal: The refactoring goal or pattern to apply.
            plan_mode: Use plan mode for structured planning (default True).
            model: Model to use for this request.

        Returns:
            ToolResult with refactoring outcome.

        Example:
            >>> result = await client.refactor(
            ...     "src/yolo_developer/config/loader.py",
            ...     "Extract validation logic into separate functions",
            ... )
        """
        prompt = f"Refactor {target}: {goal}"

        return await self.run(prompt, plan_mode=plan_mode, model=model)
