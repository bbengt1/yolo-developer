"""Main CLI entry point for YOLO Developer."""

from __future__ import annotations

from pathlib import Path
import warnings

import typer
from rich.console import Console

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*There is no current event loop.*",
)

from yolo_developer.cli.commands.init import init_command
from yolo_developer.cli.commands.mcp import app as mcp_app
from yolo_developer.cli.commands.scan import scan_command

app = typer.Typer(
    name="yolo",
    help="YOLO Developer - Autonomous multi-agent AI development system",
    no_args_is_help=True,
)
console = Console()


@app.command("init")
def init(
    path: str | None = typer.Argument(
        None,
        help="Directory to initialize the project in. Defaults to current directory.",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Project name. Defaults to directory name.",
    ),
    author: str | None = typer.Option(
        None,
        "--author",
        "-a",
        help="Author name for pyproject.toml.",
    ),
    email: str | None = typer.Option(
        None,
        "--email",
        "-e",
        help="Author email for pyproject.toml.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Prompt for project details interactively.",
    ),
    no_input: bool = typer.Option(
        False,
        "--no-input",
        help="Use all defaults without prompting.",
    ),
    existing: bool = typer.Option(
        False,
        "--existing",
        "--brownfield",
        help="Add YOLO to an existing project (brownfield mode).",
    ),
    scan_only: bool = typer.Option(
        False,
        "--scan-only",
        help="Scan brownfield project without making changes.",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Skip prompts during brownfield scanning.",
    ),
    hint: str | None = typer.Option(
        None,
        "--hint",
        help="Hint about project type (e.g., 'fastapi api').",
    ),
) -> None:
    """Initialize a new YOLO Developer project.

    Creates a new Python project with all required dependencies for
    autonomous multi-agent development using the BMad Method.

    In interactive mode (-i), prompts for project name, author, and email
    with sensible defaults from git config.

    Use --no-input to skip all prompts and use defaults.

    Use --existing for brownfield projects to add YOLO without
    overwriting existing files.
    """
    init_command(
        path=path,
        name=name,
        author=author,
        email=email,
        interactive=interactive,
        no_input=no_input,
        existing=existing,
        scan_only=scan_only,
        non_interactive=non_interactive,
        hint=hint,
    )


@app.command("scan")
def scan(
    path: str | None = typer.Argument(
        None,
        help="Directory to scan. Defaults to current directory.",
    ),
    scan_depth: int | None = typer.Option(
        None,
        "--scan-depth",
        help="Directory depth to scan.",
    ),
    max_files: int | None = typer.Option(
        None,
        "--max-files",
        help="Maximum files to analyze.",
    ),
    include_git_history: bool | None = typer.Option(
        None,
        "--git-history/--no-git-history",
        help="Include git history analysis.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Prompt for ambiguous findings.",
    ),
    hint: str | None = typer.Option(
        None,
        "--hint",
        help="Hint about project type (e.g., 'fastapi api').",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Overwrite existing project-context.yaml.",
    ),
    write_context: bool = typer.Option(
        True,
        "--write-context/--no-write-context",
        help="Write project-context.yaml to .yolo directory.",
    ),
) -> None:
    """Scan an existing project for brownfield context."""
    scan_command(
        path=path,
        scan_depth=scan_depth,
        max_files=max_files,
        include_git_history=include_git_history,
        interactive=interactive,
        hint=hint,
        refresh=refresh,
        write_context=write_context,
    )


@app.command("version")
def version() -> None:
    """Show YOLO Developer version."""
    from yolo_developer import __version__

    console.print(f"YOLO Developer v{__version__}")


@app.command("seed")
def seed(
    file_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to the seed document file to parse.",
        exists=False,  # We handle existence check in command for better error messages
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output including confidence scores and metadata.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON instead of formatted tables.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Detect ambiguities and prompt for clarification before parsing.",
    ),
    validate_sop: bool = typer.Option(
        False,
        "--validate-sop",
        help="Validate seed against SOP constraints.",
    ),
    sop_store: Path | None = typer.Option(  # noqa: B008
        None,
        "--sop-store",
        help="Path to SOP store JSON file with constraints.",
    ),
    override_soft: bool = typer.Option(
        False,
        "--override-soft",
        help="Auto-override all SOFT SOP conflicts without prompting.",
    ),
    report_format: str | None = typer.Option(
        None,
        "--report-format",
        "-r",
        help="Generate validation report in specified format (json, markdown, rich).",
    ),
    report_output: Path | None = typer.Option(  # noqa: B008
        None,
        "--report-output",
        "-o",
        help="File path to write the validation report to.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force processing even if quality thresholds are not met.",
    ),
) -> None:
    """Parse a seed document into structured components.

    Reads a natural language requirements document and extracts:
    - Goals: High-level project objectives
    - Features: Discrete functional capabilities
    - Constraints: Technical, business, or other limitations

    In interactive mode (--interactive), the command will:
    1. Detect ambiguities in the seed document
    2. Display ambiguities with resolution prompts
    3. Allow you to clarify each ambiguity
    4. Re-parse with your clarifications applied

    With SOP validation (--validate-sop), the command will:
    1. Load constraints from the SOP store
    2. Validate seed against constraints
    3. Report HARD conflicts (block processing)
    4. Prompt for SOFT conflict overrides (or auto-override with --override-soft)

    With validation reports (--report-format), the command will:
    1. Parse the seed document
    2. Calculate quality metrics (ambiguity, SOP, extraction scores)
    3. Generate a comprehensive validation report
    4. Check quality thresholds (reject if below minimum)
    5. Output in the specified format (json, markdown, or rich console)

    Quality thresholds are checked when generating reports. Seeds that fail
    minimum thresholds are rejected with remediation guidance. Use --force
    to proceed despite low quality scores (Story 4.7).

    Examples:
        yolo seed requirements.md
        yolo seed requirements.md --verbose
        yolo seed requirements.md --json
        yolo seed requirements.md --interactive
        yolo seed requirements.md --validate-sop --sop-store constraints.json
        yolo seed requirements.md --validate-sop --override-soft
        yolo seed requirements.md --report-format json
        yolo seed requirements.md --report-format markdown -o report.md
        yolo seed requirements.md --report-format rich
        yolo seed requirements.md --report-format rich --force  # bypass threshold rejection
    """
    from yolo_developer.cli.commands.seed import seed_command

    seed_command(
        file_path=file_path,
        verbose=verbose,
        json_output=json_output,
        interactive=interactive,
        validate_sop=validate_sop,
        sop_store_path=sop_store,
        override_soft=override_soft,
        report_format=report_format,
        report_output=report_output,
        force=force,
    )


@app.command("run")
def run(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Validate configuration and seed without executing the workflow.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output including agent events and decisions.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON instead of formatted display.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume from the last checkpoint instead of starting fresh.",
    ),
    thread_id: str | None = typer.Option(
        None,
        "--thread-id",
        "-t",
        help="Specific thread ID for checkpointing. Auto-generated if not provided.",
    ),
) -> None:
    """Execute an autonomous sprint.

    Starts the multi-agent orchestration system to autonomously
    develop features based on the current sprint backlog.

    The workflow executes agents in sequence: Analyst → PM → Architect → Dev → TEA,
    with the SM Agent coordinating the overall process.

    Examples:
        yolo run                           # Start a new sprint
        yolo run --dry-run                 # Validate without executing
        yolo run --verbose                 # Show detailed progress
        yolo run --resume                  # Resume from last checkpoint
        yolo run --thread-id my-session    # Use specific thread ID
        yolo run --json                    # Output JSON summary
    """
    from yolo_developer.cli.commands.run import run_command

    run_command(
        dry_run=dry_run,
        verbose=verbose,
        json_output=json_output,
        resume=resume,
        thread_id=thread_id,
    )


@app.command("status")
def status(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output including agent health snapshots and metrics.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON instead of formatted display.",
    ),
    health_only: bool = typer.Option(
        False,
        "--health",
        "-H",
        help="Show only health metrics (skip sprint progress and sessions).",
    ),
    sessions_only: bool = typer.Option(
        False,
        "--sessions",
        "-s",
        help="Show only session list (skip sprint progress and health).",
    ),
) -> None:
    """Show sprint progress and status.

    Displays the current sprint status including completed stories,
    in-progress work, system health metrics, and session information.

    The command shows three main sections:
    - Sprint Progress: Stories completed, in-progress, and blocked
    - Health Metrics: Agent idle times, cycle times, and alerts
    - Sessions: Available sessions with resume instructions

    Examples:
        yolo status                    # Show all status information
        yolo status --verbose          # Show detailed health metrics
        yolo status --json             # Output as JSON
        yolo status --health           # Show only health metrics
        yolo status --sessions         # Show only session list
    """
    from yolo_developer.cli.commands.status import status_command

    status_command(
        verbose=verbose,
        json_output=json_output,
        health_only=health_only,
        sessions_only=sessions_only,
    )


@app.command("logs")
def logs(
    agent: str | None = typer.Option(
        None,
        "--agent",
        "-a",
        help="Filter by agent name (case-insensitive). E.g., 'analyst', 'pm', 'dev'.",
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        "-s",
        help="Show decisions after this time. Supports relative (1h, 30m, 7d) or ISO format.",
    ),
    decision_type: str | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by decision type (e.g., 'requirement_analysis', 'architecture_choice').",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of entries to display. Default: 20.",
    ),
    show_all: bool = typer.Option(
        False,
        "--all",
        "-A",
        help="Show all entries without pagination.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output including rationale and context.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON instead of formatted display.",
    ),
) -> None:
    """Browse decision audit trail.

    View the history of agent decisions with filtering and
    search capabilities.

    The command displays recent decisions by default, with options
    to filter by agent, time range, or decision type.

    Examples:
        yolo logs                          # Show recent decisions (default 20)
        yolo logs --agent analyst          # Filter by agent
        yolo logs --since 1h               # Decisions from last hour
        yolo logs --since 2026-01-15       # Decisions since date
        yolo logs --type architecture_choice  # Filter by decision type
        yolo logs --limit 50               # Show 50 entries
        yolo logs --all                    # Show all entries
        yolo logs --verbose                # Show full details
        yolo logs --json                   # Output as JSON
    """
    from yolo_developer.cli.commands.logs import logs_command

    logs_command(
        agent=agent,
        since=since,
        decision_type=decision_type,
        limit=limit,
        show_all=show_all,
        verbose=verbose,
        json_output=json_output,
    )


@app.command("tune")
def tune(
    agent_name: str | None = typer.Argument(
        None,
        help="Agent to view/modify (analyst, pm, architect, dev, sm, tea).",
    ),
    list_agents: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List all configurable agents with their status.",
    ),
    edit: bool = typer.Option(
        False,
        "--edit",
        "-e",
        help="Edit the agent's template using $EDITOR.",
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        "-r",
        help="Reset agent template to default.",
    ),
    export_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--export",
        help="Export agent template to file.",
    ),
    import_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--import",
        help="Import agent template from file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON instead of formatted display.",
    ),
) -> None:
    """Customize agent behavior.

    Modify agent templates and decision-making parameters
    to tune how agents approach development tasks.

    View an agent's current template, modify it with your preferred editor,
    or export/import templates for sharing or backup.

    Examples:
        yolo tune --list                    # List all configurable agents
        yolo tune analyst                   # View analyst agent template
        yolo tune analyst --edit            # Edit analyst template in $EDITOR
        yolo tune analyst --reset           # Reset analyst to default
        yolo tune analyst --export out.yaml # Export template to file
        yolo tune analyst --import my.yaml  # Import template from file
        yolo tune analyst --json            # Output as JSON
    """
    from yolo_developer.cli.commands.tune import tune_command

    tune_command(
        agent_name=agent_name,
        list_agents=list_agents,
        edit=edit,
        reset=reset,
        export_path=export_path,
        import_path=import_path,
        json_output=json_output,
    )


# Config subcommand group
config_app = typer.Typer(
    name="config",
    help="Manage project configuration.",
    no_args_is_help=False,
)
app.add_typer(config_app, name="config")
app.add_typer(mcp_app, name="mcp")


@config_app.callback(invoke_without_command=True)
def config_show(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON instead of formatted display.",
    ),
    no_mask: bool = typer.Option(
        False,
        "--no-mask",
        help="Show API keys unmasked (for debugging).",
    ),
) -> None:
    """Show current project configuration.

    Displays the current configuration from yolo.yaml,
    including LLM settings, quality thresholds, and memory configuration.

    Sensitive values (API keys) are masked by default.

    Examples:
        yolo config                  # Show config with Rich formatting
        yolo config --json           # Show config as JSON
        yolo config --no-mask        # Show API keys unmasked
    """
    if ctx.invoked_subcommand is None:
        from yolo_developer.cli.commands.config import show_config

        show_config(json_output=json_output, no_mask=no_mask)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(
        ...,
        help="Configuration key (supports dotted notation, e.g., 'llm.cheap_model').",
    ),
    value: str = typer.Argument(
        ...,
        help="Value to set.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON instead of formatted display.",
    ),
) -> None:
    """Set a configuration value.

    Updates a specific configuration value in yolo.yaml.
    Supports nested keys using dot notation.

    Examples:
        yolo config set project_name my-project
        yolo config set llm.cheap_model gpt-4o
        yolo config set quality.test_coverage_threshold 0.85
        yolo config set quality.seed_thresholds.overall 0.75
    """
    from yolo_developer.cli.commands.config import set_config_value

    set_config_value(key=key, value=value, json_output=json_output)


@config_app.command("export")
def config_export(
    output: Path = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Output file path. Defaults to 'yolo-config-export.yaml'.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON instead of formatted display.",
    ),
) -> None:
    """Export configuration to a portable YAML file.

    Creates a YAML file containing the current configuration.
    API keys are excluded for security.

    Examples:
        yolo config export                        # Export to yolo-config-export.yaml
        yolo config export -o backup.yaml         # Export to custom path
        yolo config export --json                 # JSON output for scripting
    """
    from yolo_developer.cli.commands.config import export_config_command

    export_config_command(output_path=output, json_output=json_output)


@config_app.command("import")
def config_import(
    file: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to the configuration file to import.",
        exists=False,  # We handle existence check for better error messages
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON instead of formatted display.",
    ),
) -> None:
    """Import configuration from a YAML file.

    Imports configuration from an exported YAML file.
    The configuration is validated before being applied.

    Examples:
        yolo config import backup.yaml            # Import from file
        yolo config import shared-config.yaml     # Import shared config
        yolo config import backup.yaml --json     # JSON output for scripting
    """
    from yolo_developer.cli.commands.config import import_config_command

    import_config_command(source_path=file, json_output=json_output)


if __name__ == "__main__":
    app()
