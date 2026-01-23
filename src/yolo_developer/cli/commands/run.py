"""YOLO run command implementation (Story 12.4, 12.9).

This module provides the yolo run command which executes autonomous sprints
using the multi-agent orchestration system.

The command supports:
- Real-time activity display with Rich Live (Story 12.9)
- Agent transition visualization
- Interrupt handling with graceful shutdown
- Resume from checkpoint
- JSON output for automation

Example:
    yolo run                        # Start a new sprint
    yolo run --dry-run              # Validate without executing
    yolo run --verbose              # Show detailed progress
    yolo run --resume               # Resume from last checkpoint
"""

from __future__ import annotations

import asyncio
import json
import signal
import time
import uuid
from pathlib import Path
from typing import Any

import structlog
from langchain_core.messages import BaseMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from yolo_developer.cli.activity import ActivityDisplay
from yolo_developer.cli.display import error_panel, info_panel, success_panel, warning_panel
from yolo_developer.config import ConfigurationError, load_config

logger = structlog.get_logger(__name__)
console = Console()

# Global flag for interrupt handling
# Note: This is not thread-safe. For CLI usage this is acceptable since
# commands run as single processes. For library usage, consider using
# threading.Event or a context-based approach.
_interrupted = False

# Agent descriptions for activity display (Story 12.9)
AGENT_DESCRIPTIONS: dict[str, str] = {
    "analyst": "Analyzing requirements and extracting insights",
    "pm": "Creating stories and managing backlog",
    "architect": "Designing system architecture",
    "dev": "Implementing code changes",
    "tea": "Validating quality and test coverage",
    "sm": "Orchestrating sprint activities",
    "escalate": "Escalating to human for review",
}


def _get_agent_description(agent_name: str) -> str:
    """Get a human-readable description for an agent.

    Args:
        agent_name: Name of the agent.

    Returns:
        Description of what the agent is doing.
    """
    return AGENT_DESCRIPTIONS.get(agent_name, f"Running {agent_name} agent")


def check_seed_exists() -> bool:
    """Check if a seed has been parsed and is available.

    Returns:
        True if a seed file/state exists, False otherwise.
    """
    # Check for seed state file in .yolo directory
    seed_state_path = Path(".yolo/seed_state.json")
    return seed_state_path.exists()


def get_seed_messages() -> list[BaseMessage]:
    """Load seed requirements as messages for the workflow.

    Returns:
        List of HumanMessage objects containing seed requirements.
    """
    seed_state_path = Path(".yolo/seed_state.json")

    if not seed_state_path.exists():
        return []

    try:
        with open(seed_state_path) as f:
            seed_data = json.load(f)

        # Extract requirements from seed data
        messages: list[BaseMessage] = []

        def extract_description(item: Any) -> str:
            """Extract description from item, handling both dict and string formats."""
            if isinstance(item, dict):
                return str(item.get("description", item.get("name", str(item))))
            return str(item)

        # Add goals as requirements
        goals = seed_data.get("goals", [])
        if goals:
            goals_text = "\n".join(f"- {extract_description(g)}" for g in goals)
            messages.append(HumanMessage(content=f"Project Goals:\n{goals_text}"))

        # Add features as requirements
        features = seed_data.get("features", [])
        if features:
            features_text = "\n".join(f"- {extract_description(f)}" for f in features)
            messages.append(HumanMessage(content=f"Features:\n{features_text}"))

        # Add constraints
        constraints = seed_data.get("constraints", [])
        if constraints:
            constraints_text = "\n".join(f"- {extract_description(c)}" for c in constraints)
            messages.append(HumanMessage(content=f"Constraints:\n{constraints_text}"))

        return messages
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("failed_to_load_seed", error=str(e))
        return []


async def execute_workflow(
    thread_id: str | None = None,
    verbose: bool = False,
    json_output: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute the workflow with real-time progress display.

    Args:
        thread_id: Optional thread ID for checkpointing.
        verbose: Show detailed output.
        json_output: Output JSON instead of formatted display.
        resume: Resume from checkpoint.

    Returns:
        Final workflow state as dictionary.
    """
    from yolo_developer.orchestrator import (
        SessionManager,
        SessionNotFoundError,
        WorkflowConfig,
        create_initial_state,
        stream_workflow,
    )

    global _interrupted

    # Session manager for persistence
    session_dir = Path(".yolo/sessions")
    session_manager = SessionManager(session_dir)

    # Generate thread ID if not provided
    effective_thread_id = thread_id or str(uuid.uuid4())

    # Handle resume: load existing state or create fresh
    if resume and thread_id:
        try:
            state, metadata = await session_manager.load_session(thread_id)
            initial_state = state
            starting_agent = metadata.current_agent or "analyst"
            if not json_output:
                console.print(f"[green]Resuming from checkpoint: {thread_id}[/green]")
                console.print(f"[dim]Last agent: {starting_agent}[/dim]")
            logger.info("session_resumed", thread_id=thread_id, agent=starting_agent)
        except SessionNotFoundError:
            if not json_output:
                console.print(
                    f"[yellow]No checkpoint found for {thread_id}, starting fresh[/yellow]"
                )
            messages = get_seed_messages()
            initial_state = create_initial_state(starting_agent="analyst", messages=messages)
    else:
        # Create initial state from seed
        messages = get_seed_messages()
        initial_state = create_initial_state(starting_agent="analyst", messages=messages)

    # Configure workflow
    config = WorkflowConfig(entry_point="analyst", enable_checkpointing=True)

    # Track execution stats
    start_time = time.time()
    event_count = 0
    agents_executed: list[str] = []
    current_agent = "analyst"
    final_state: dict[str, Any] = {}

    if not json_output:
        console.print(
            Panel(
                f"[bold green]Starting Sprint Execution[/bold green]\n\n"
                f"Thread ID: [cyan]{effective_thread_id}[/cyan]\n"
                f"Resume: [yellow]{resume}[/yellow]",
                title="YOLO Run",
                border_style="green",
            )
        )

    try:
        # Create activity display (Story 12.9)
        if json_output:
            # JSON mode: simple event processing without display
            async for event in stream_workflow(
                initial_state,
                config=config,
                thread_id=effective_thread_id,
            ):
                if _interrupted:
                    logger.info("workflow_interrupted", thread_id=effective_thread_id)
                    break

                event_count += 1

                # Extract agent name from event
                agent_name = next(iter(event.keys())) if event else "unknown"
                if agent_name != current_agent:
                    agents_executed.append(agent_name)
                    current_agent = agent_name

                # Store latest state
                if agent_name in event:
                    final_state = event[agent_name]
        else:
            # Rich activity display (Story 12.9)
            with ActivityDisplay(verbose=verbose) as activity:
                async for event in stream_workflow(
                    initial_state,
                    config=config,
                    thread_id=effective_thread_id,
                ):
                    if _interrupted:
                        logger.info("workflow_interrupted", thread_id=effective_thread_id)
                        break

                    event_count += 1
                    elapsed = time.time() - start_time

                    # Extract agent name from event
                    agent_name = next(iter(event.keys())) if event else "unknown"

                    # Detect agent transition
                    if agent_name != current_agent:
                        if current_agent:  # Not first agent
                            activity.add_transition(
                                from_agent=current_agent,
                                to_agent=agent_name,
                                reason="handoff",
                                elapsed=elapsed,
                            )
                        agents_executed.append(agent_name)
                        current_agent = agent_name

                    # Get description from state
                    description = _get_agent_description(agent_name)

                    # Update activity display
                    activity.update(
                        agent=agent_name,
                        description=description,
                        elapsed=elapsed,
                    )

                    # Store latest state
                    if agent_name in event:
                        final_state = event[agent_name]

    except KeyboardInterrupt:
        _interrupted = True
        logger.info("keyboard_interrupt", thread_id=effective_thread_id)

    elapsed_time = time.time() - start_time

    # Preserve state on interrupt for resumption (AC3)
    if _interrupted and final_state:
        try:
            await session_manager.save_session(
                final_state,  # type: ignore[arg-type]
                session_id=effective_thread_id,
            )
            logger.info("state_preserved_on_interrupt", thread_id=effective_thread_id)
        except Exception as e:
            logger.warning("failed_to_preserve_state", error=str(e))

    # Build result
    result = {
        "thread_id": effective_thread_id,
        "elapsed_time": elapsed_time,
        "event_count": event_count,
        "agents_executed": agents_executed,
        "interrupted": _interrupted,
        "final_state": final_state,
        "decisions": final_state.get("decisions", []) if final_state else [],
    }

    return result


def display_summary(result: dict[str, Any], json_output: bool = False) -> None:
    """Display execution summary.

    Args:
        result: Execution result dictionary.
        json_output: Output as JSON.
    """
    if json_output:
        # Output JSON for automation
        output = {
            "status": "interrupted" if result["interrupted"] else "completed",
            "thread_id": result["thread_id"],
            "elapsed_time_seconds": round(result["elapsed_time"], 2),
            "event_count": result["event_count"],
            "agents_executed": result["agents_executed"],
            "decision_count": len(result.get("decisions", [])),
        }
        console.print(json.dumps(output, indent=2))
        return

    # Create summary table
    table = Table(title="Sprint Execution Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    status = "⚠️ Interrupted" if result["interrupted"] else "✅ Completed"
    table.add_row("Status", status)
    table.add_row("Thread ID", result["thread_id"])
    table.add_row("Elapsed Time", f"{result['elapsed_time']:.2f}s")
    table.add_row("Events Processed", str(result["event_count"]))
    table.add_row("Agents Executed", ", ".join(result["agents_executed"]) or "None")
    table.add_row("Decisions Made", str(len(result.get("decisions", []))))

    console.print(table)

    if result["interrupted"]:
        console.print(
            Panel(
                f"[yellow]Sprint was interrupted.[/yellow]\n\n"
                f"To resume, run:\n"
                f"[cyan]yolo run --resume --thread-id {result['thread_id']}[/cyan]",
                title="Resume Instructions",
                border_style="yellow",
            )
        )


def run_async_workflow(
    thread_id: str | None,
    verbose: bool,
    json_output: bool,
    resume: bool,
) -> dict[str, Any]:
    """Run the async workflow in the event loop.

    Args:
        thread_id: Optional thread ID.
        verbose: Verbose output.
        json_output: JSON output.
        resume: Resume from checkpoint.

    Returns:
        Execution result.
    """
    return asyncio.run(
        execute_workflow(
            thread_id=thread_id,
            verbose=verbose,
            json_output=json_output,
            resume=resume,
        )
    )


def run_command(
    dry_run: bool = False,
    verbose: bool = False,
    json_output: bool = False,
    resume: bool = False,
    thread_id: str | None = None,
    agents: list[str] | None = None,
    max_iterations: int | None = None,
    timeout: int | None = None,
    watch: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Execute the run command.

    This command triggers the multi-agent orchestration to execute
    a sprint based on seed requirements and project configuration.

    Args:
        dry_run: Validate only, don't execute workflow.
        verbose: Show detailed output.
        json_output: Output as JSON.
        resume: Resume from checkpoint.
        thread_id: Specific thread ID for checkpointing.
        agents: Optional list of agents to run (currently informational only).
        max_iterations: Optional maximum iterations override (not yet enforced).
        timeout: Optional timeout override in seconds (not yet enforced).
        watch: Enable watch mode (currently a no-op; live output is default).
        output_dir: Optional output directory (not yet enforced).
    """
    global _interrupted
    _interrupted = False

    logger.debug(
        "run_command_invoked",
        dry_run=dry_run,
        verbose=verbose,
        json_output=json_output,
        resume=resume,
        thread_id=thread_id,
        agents=agents,
        max_iterations=max_iterations,
        timeout=timeout,
        watch=watch,
        output_dir=str(output_dir) if output_dir else None,
    )

    if any([agents, max_iterations, timeout, watch, output_dir]):
        logger.warning(
            "run_options_not_supported",
            agents=agents,
            max_iterations=max_iterations,
            timeout=timeout,
            watch=watch,
            output_dir=str(output_dir) if output_dir else None,
        )
        if not json_output:
            warning_panel(
                "Some options are not yet supported and will be ignored: "
                "--agents, --max-iterations, --timeout, --watch, --output-dir."
            )

    # Load configuration
    try:
        config = load_config()
        if verbose and not json_output:
            info_panel(f"Loaded configuration for project: {config.project_name}")
    except ConfigurationError as e:
        error_panel(f"Configuration error: {e}")
        raise SystemExit(1) from e

    # Check if seed exists (unless resuming)
    if not resume and not check_seed_exists():
        if json_output:
            console.print(json.dumps({"error": "No seed found. Run 'yolo seed' first."}))
        else:
            error_panel(
                "No seed found.\n\n"
                "Please run 'yolo seed <requirements.md>' first to parse your requirements."
            )
        raise SystemExit(1)

    # Dry run mode - validate only
    if dry_run:
        if json_output:
            output = {
                "status": "validated",
                "seed_exists": check_seed_exists(),
                "config_loaded": True,
                "project_name": config.project_name,
                "agents": agents or [],
                "max_iterations": max_iterations,
                "timeout": timeout,
                "watch": watch,
                "output_dir": str(output_dir) if output_dir else None,
            }
            console.print(json.dumps(output, indent=2))
        else:
            success_panel(
                f"Dry run validation successful!\n\n"
                f"Project: {config.project_name}\n"
                f"Seed: {'Found' if check_seed_exists() else 'Not found'}\n\n"
                f"Ready to execute. Remove --dry-run to start the sprint."
            )
        return

    # Register interrupt handler
    def handle_interrupt(signum: int, frame: Any) -> None:
        global _interrupted
        _interrupted = True
        if not json_output:
            console.print("\n[yellow]Interrupt received, finishing current task...[/yellow]")

    signal.signal(signal.SIGINT, handle_interrupt)

    # Execute workflow
    try:
        result = run_async_workflow(
            thread_id=thread_id,
            verbose=verbose,
            json_output=json_output,
            resume=resume,
        )

        # Display summary
        display_summary(result, json_output=json_output)

    except Exception as e:
        logger.exception("workflow_execution_failed", error=str(e))
        if json_output:
            console.print(json.dumps({"error": str(e)}))
        else:
            error_panel(f"Workflow execution failed: {e}")
        raise SystemExit(1) from e
