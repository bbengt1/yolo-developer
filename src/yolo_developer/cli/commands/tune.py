"""YOLO tune command implementation (Story 12.7).

This module provides the yolo tune command which allows users to:
- View agent templates (system prompts, user prompt templates)
- List all configurable agents
- Edit templates using $EDITOR
- Reset templates to defaults
- Export/import templates to/from YAML files

The command supports multiple output modes:
- Default: Rich formatted display with syntax highlighting
- JSON: Machine-readable JSON output

Example:
    >>> from yolo_developer.cli.commands.tune import tune_command
    >>>
    >>> tune_command(list_agents=True)  # List all agents
    >>> tune_command(agent_name="analyst")  # View analyst template
    >>> tune_command(agent_name="analyst", edit=True)  # Edit analyst template

References:
    - FR91: Users can customize agent templates and rules
    - FR103: Users can modify agent templates via yolo tune command
    - Story 12.7: yolo tune command
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
import typer
import yaml
from rich.panel import Panel
from rich.syntax import Syntax

from yolo_developer.cli.display import (
    console,
    create_table,
    info_panel,
    success_panel,
    warning_panel,
)

logger = structlog.get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Agent colors for Rich display (match status/logs patterns)
AGENT_COLORS: dict[str, str] = {
    "analyst": "cyan",
    "pm": "blue",
    "architect": "magenta",
    "dev": "green",
    "tea": "yellow",
    "sm": "white",
}

# Template storage directory
TEMPLATES_DIR = ".yolo/templates"

# =============================================================================
# Agent Template Registry (Task 2)
# =============================================================================

# Maps agent names to their prompt module paths and variable names
CONFIGURABLE_AGENTS: dict[str, dict[str, Any]] = {
    "analyst": {
        "description": "Crystallizes vague requirements into specific, implementable statements",
        "prompt_module": "yolo_developer.agents.prompts.analyst",
        "system_prompt_var": "ANALYST_SYSTEM_PROMPT",
        "user_prompt_var": "ANALYST_USER_PROMPT_TEMPLATE",
    },
    "pm": {
        "description": "Transforms requirements into testable user stories with acceptance criteria",
        "prompt_module": "yolo_developer.agents.pm.llm",
        "system_prompt_var": "PM_SYSTEM_PROMPT",
        "user_prompt_var": None,  # PM uses builder functions
    },
    "architect": {
        "description": "Designs system architecture following 12-Factor principles",
        "prompt_module": "yolo_developer.agents.architect.twelve_factor",
        "system_prompt_var": None,  # Uses TWELVE_FACTOR_PROMPT
        "user_prompt_var": "TWELVE_FACTOR_PROMPT",
    },
    "dev": {
        "description": "Implements maintainable code with tests and documentation",
        "prompt_module": "yolo_developer.agents.dev.prompts.code_generation",
        "system_prompt_var": None,  # Uses guidelines constants
        "user_prompt_var": None,  # Uses builder function
        "guidelines": ["MAINTAINABILITY_GUIDELINES", "PROJECT_CONVENTIONS"],
    },
    "sm": {
        "description": "Orchestrates sprint execution and monitors system health",
        "prompt_module": None,  # SM doesn't have centralized prompts
        "system_prompt_var": None,
        "user_prompt_var": None,
    },
    "tea": {
        "description": "Validates test coverage and calculates deployment confidence",
        "prompt_module": None,  # TEA doesn't have centralized prompts
        "system_prompt_var": None,
        "user_prompt_var": None,
    },
}

# Valid agent names for validation
VALID_AGENTS: frozenset[str] = frozenset(CONFIGURABLE_AGENTS.keys())


# =============================================================================
# Template Storage Functions (Task 4)
# =============================================================================


def _get_templates_dir() -> Path:
    """Get the templates directory path.

    Returns:
        Path to the .yolo/templates directory.
    """
    return Path.cwd() / TEMPLATES_DIR


def _get_template_path(agent_name: str) -> Path:
    """Get the path to an agent's custom template file.

    Args:
        agent_name: Name of the agent.

    Returns:
        Path to the agent's template YAML file.
    """
    return _get_templates_dir() / f"{agent_name}.yaml"


def has_custom_template(agent_name: str) -> bool:
    """Check if an agent has a custom template.

    Args:
        agent_name: Name of the agent.

    Returns:
        True if the agent has a custom template, False otherwise.
    """
    return _get_template_path(agent_name).exists()


def load_custom_template(agent_name: str) -> dict[str, Any] | None:
    """Load a custom template for an agent.

    Args:
        agent_name: Name of the agent.

    Returns:
        Template dictionary or None if no custom template exists.
    """
    template_path = _get_template_path(agent_name)
    if not template_path.exists():
        return None

    try:
        with template_path.open() as f:
            data: dict[str, Any] | None = yaml.safe_load(f)
            return data
    except (yaml.YAMLError, OSError) as e:
        logger.warning("failed_to_load_custom_template", agent=agent_name, error=str(e))
        return None


def save_custom_template(
    agent_name: str,
    system_prompt: str | None,
    user_prompt_template: str | None,
    guidelines: dict[str, str] | None = None,
) -> Path:
    """Save a custom template for an agent.

    Args:
        agent_name: Name of the agent.
        system_prompt: The system prompt content.
        user_prompt_template: The user prompt template content.
        guidelines: Optional dictionary of guideline overrides.

    Returns:
        Path to the saved template file.
    """
    templates_dir = _get_templates_dir()
    templates_dir.mkdir(parents=True, exist_ok=True)

    template_data: dict[str, Any] = {
        "agent": agent_name,
        "version": "1.0",
        "customized_at": datetime.now(timezone.utc).isoformat(),
    }

    if system_prompt is not None:
        template_data["system_prompt"] = system_prompt

    if user_prompt_template is not None:
        template_data["user_prompt_template"] = user_prompt_template

    if guidelines:
        template_data["guidelines"] = guidelines

    template_path = _get_template_path(agent_name)
    with template_path.open("w") as f:
        yaml.dump(template_data, f, default_flow_style=False, allow_unicode=True)

    return template_path


def reset_template(agent_name: str) -> bool:
    """Reset an agent's template to default by removing the custom template.

    Args:
        agent_name: Name of the agent.

    Returns:
        True if a custom template was removed, False if none existed.
    """
    template_path = _get_template_path(agent_name)
    if template_path.exists():
        template_path.unlink()
        return True
    return False


# =============================================================================
# Template Loading Functions (Task 3)
# =============================================================================


def _load_default_template(agent_name: str) -> dict[str, Any]:
    """Load the default template for an agent from its module.

    Args:
        agent_name: Name of the agent.

    Returns:
        Dictionary with template content.
    """
    agent_config = CONFIGURABLE_AGENTS.get(agent_name)
    if not agent_config:
        return {"system_prompt": None, "user_prompt_template": None, "guidelines": {}}

    result: dict[str, Any] = {
        "system_prompt": None,
        "user_prompt_template": None,
        "guidelines": {},
    }

    prompt_module = agent_config.get("prompt_module")
    if not prompt_module:
        return result

    try:
        module = importlib.import_module(prompt_module)

        # Load system prompt
        system_prompt_var = agent_config.get("system_prompt_var")
        if system_prompt_var and hasattr(module, system_prompt_var):
            result["system_prompt"] = getattr(module, system_prompt_var)

        # Load user prompt template
        user_prompt_var = agent_config.get("user_prompt_var")
        if user_prompt_var and hasattr(module, user_prompt_var):
            result["user_prompt_template"] = getattr(module, user_prompt_var)

        # Load guidelines (for dev agent)
        guidelines_vars = agent_config.get("guidelines", [])
        for var_name in guidelines_vars:
            if hasattr(module, var_name):
                result["guidelines"][var_name] = getattr(module, var_name)

    except ImportError as e:
        logger.warning("failed_to_import_prompt_module", module=prompt_module, error=str(e))

    return result


def _get_effective_template(agent_name: str) -> tuple[dict[str, Any], bool]:
    """Get the effective template for an agent (custom or default).

    Args:
        agent_name: Name of the agent.

    Returns:
        Tuple of (template_dict, is_customized).
    """
    custom_template = load_custom_template(agent_name)
    if custom_template:
        return custom_template, True

    default_template = _load_default_template(agent_name)
    return default_template, False


# =============================================================================
# Display Functions (Task 3, Task 7)
# =============================================================================


def _display_agent_template(agent_name: str, json_output: bool = False) -> None:
    """Display an agent's template content.

    Args:
        agent_name: Name of the agent.
        json_output: If True, output as JSON instead of Rich formatting.
    """
    template, is_customized = _get_effective_template(agent_name)
    agent_config = CONFIGURABLE_AGENTS[agent_name]
    agent_color = AGENT_COLORS.get(agent_name, "white")
    status = "customized" if is_customized else "default"

    if json_output:
        output = {
            "agent": agent_name,
            "status": status,
            "customized_at": template.get("customized_at") if is_customized else None,
            "template": {
                "system_prompt": template.get("system_prompt"),
                "user_prompt_template": template.get("user_prompt_template"),
                "guidelines": template.get("guidelines", {}),
            },
        }
        console.print_json(json.dumps(output, default=str))
        return

    # Display header
    console.print()
    status_text = "[yellow](customized)[/yellow]" if is_customized else "[dim](default)[/dim]"
    console.print(
        f"[bold {agent_color}]{agent_name.upper()} Agent Template[/bold {agent_color}] {status_text}"
    )
    console.print(f"[dim]{agent_config['description']}[/dim]")
    console.print()

    # Display system prompt
    system_prompt = template.get("system_prompt")
    if system_prompt:
        syntax = Syntax(system_prompt, "text", theme="monokai", word_wrap=True)
        console.print(
            Panel(syntax, title=f"[{agent_color}]System Prompt[/{agent_color}]", border_style=agent_color)
        )
        console.print()

    # Display user prompt template
    user_prompt = template.get("user_prompt_template")
    if user_prompt:
        syntax = Syntax(user_prompt, "text", theme="monokai", word_wrap=True)
        console.print(
            Panel(
                syntax,
                title=f"[{agent_color}]User Prompt Template[/{agent_color}]",
                border_style=agent_color,
            )
        )
        console.print()

    # Display guidelines (for dev agent)
    guidelines = template.get("guidelines", {})
    if guidelines:
        console.print(f"[bold {agent_color}]Guidelines:[/bold {agent_color}]")
        for name, content in guidelines.items():
            syntax = Syntax(content, "markdown", theme="monokai", word_wrap=True)
            console.print(Panel(syntax, title=f"[dim]{name}[/dim]", border_style="dim"))
            console.print()

    # Show prompt module path
    prompt_module = agent_config.get("prompt_module")
    if prompt_module:
        console.print(f"[dim]Source module: {prompt_module}[/dim]")
    else:
        console.print(f"[dim]Note: {agent_name} agent does not have centralized prompts[/dim]")
    console.print()


def _display_agents_list(json_output: bool = False) -> None:
    """Display list of all configurable agents.

    Args:
        json_output: If True, output as JSON instead of Rich formatting.
    """
    agents_data = []
    for agent_name, config in CONFIGURABLE_AGENTS.items():
        status = "customized" if has_custom_template(agent_name) else "default"
        agents_data.append(
            {
                "name": agent_name,
                "description": config["description"],
                "status": status,
            }
        )

    if json_output:
        console.print_json(json.dumps({"agents": agents_data}))
        return

    # Create Rich table
    columns = [
        ("Agent", "cyan"),
        ("Description", "white"),
        ("Status", "yellow"),
    ]
    table = create_table("Configurable Agents", columns)

    for agent in agents_data:
        agent_color = AGENT_COLORS.get(agent["name"], "white")
        status_style = "yellow" if agent["status"] == "customized" else "dim"
        table.add_row(
            f"[{agent_color}]{agent['name']}[/{agent_color}]",
            agent["description"],
            f"[{status_style}]{agent['status']}[/{status_style}]",
        )

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Use `yolo tune <agent>` to view an agent's template[/dim]")
    console.print("[dim]Use `yolo tune <agent> --edit` to customize an agent[/dim]")
    console.print()


# =============================================================================
# Edit Functions (Task 5)
# =============================================================================


def _edit_template(agent_name: str) -> bool:
    """Edit an agent's template using $EDITOR.

    Args:
        agent_name: Name of the agent.

    Returns:
        True if template was modified and saved, False otherwise.
    """
    editor = os.environ.get("EDITOR", "vi")
    template, _is_customized = _get_effective_template(agent_name)

    # Create YAML content for editing
    edit_content = {
        "agent": agent_name,
        "version": "1.0",
    }
    if template.get("system_prompt"):
        edit_content["system_prompt"] = template["system_prompt"]
    if template.get("user_prompt_template"):
        edit_content["user_prompt_template"] = template["user_prompt_template"]
    if template.get("guidelines"):
        edit_content["guidelines"] = template["guidelines"]

    # Write to temp file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{agent_name}_template.yaml",
        delete=False,
    ) as f:
        yaml.dump(edit_content, f, default_flow_style=False, allow_unicode=True)
        temp_path = f.name

    try:
        # Open in editor
        console.print(f"[dim]Opening {agent_name} template in {editor}...[/dim]")
        result = subprocess.run([editor, temp_path], check=False)

        if result.returncode != 0:
            warning_panel(
                f"Editor exited with code {result.returncode}. Template not saved.",
                title="Edit Cancelled",
            )
            return False

        # Read modified content
        try:
            with open(temp_path) as f:
                modified_content = yaml.safe_load(f)
        except yaml.YAMLError as e:
            warning_panel(
                f"Invalid YAML syntax in edited template:\n{e}",
                title="Edit Cancelled",
            )
            return False

        if not modified_content:
            warning_panel("Empty template. Changes not saved.", title="Edit Cancelled")
            return False

        # Validate structure
        if "agent" not in modified_content or modified_content["agent"] != agent_name:
            warning_panel(
                f"Template must have 'agent: {agent_name}'. Changes not saved.",
                title="Invalid Template",
            )
            return False

        # Save the modified template
        save_custom_template(
            agent_name=agent_name,
            system_prompt=modified_content.get("system_prompt"),
            user_prompt_template=modified_content.get("user_prompt_template"),
            guidelines=modified_content.get("guidelines"),
        )

        success_panel(
            f"Template for {agent_name} agent saved successfully.\n"
            f"[dim]Changes will take effect on next run.[/dim]",
            title="Template Updated",
        )
        return True

    finally:
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)


# =============================================================================
# Export/Import Functions (Task 6)
# =============================================================================


def _export_template(agent_name: str, export_path: Path) -> bool:
    """Export an agent's template to a file.

    Args:
        agent_name: Name of the agent.
        export_path: Path to export the template to.

    Returns:
        True if export succeeded, False otherwise.
    """
    template, is_customized = _get_effective_template(agent_name)

    export_data = {
        "agent": agent_name,
        "version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "status": "customized" if is_customized else "default",
    }

    if template.get("system_prompt"):
        export_data["system_prompt"] = template["system_prompt"]
    if template.get("user_prompt_template"):
        export_data["user_prompt_template"] = template["user_prompt_template"]
    if template.get("guidelines"):
        export_data["guidelines"] = template["guidelines"]

    try:
        with export_path.open("w") as f:
            yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)

        success_panel(
            f"Template for {agent_name} agent exported to:\n[bold]{export_path}[/bold]",
            title="Export Complete",
        )
        return True

    except OSError as e:
        warning_panel(f"Failed to export template: {e}", title="Export Failed")
        return False


def _import_template(agent_name: str, import_path: Path) -> bool:
    """Import an agent's template from a file.

    Args:
        agent_name: Name of the agent.
        import_path: Path to import the template from.

    Returns:
        True if import succeeded, False otherwise.
    """
    if not import_path.exists():
        warning_panel(
            f"Import file not found: {import_path}",
            title="Import Failed",
        )
        return False

    try:
        with import_path.open() as f:
            import_data = yaml.safe_load(f)

        if not import_data:
            warning_panel("Import file is empty.", title="Import Failed")
            return False

        # Validate agent name matches (if specified in file)
        if "agent" in import_data and import_data["agent"] != agent_name:
            warning_panel(
                f"Template is for '{import_data['agent']}' but you specified '{agent_name}'.\n"
                f"[dim]Use --import with the correct agent name.[/dim]",
                title="Agent Mismatch",
            )
            return False

        # Save as custom template
        save_custom_template(
            agent_name=agent_name,
            system_prompt=import_data.get("system_prompt"),
            user_prompt_template=import_data.get("user_prompt_template"),
            guidelines=import_data.get("guidelines"),
        )

        success_panel(
            f"Template for {agent_name} agent imported from:\n[bold]{import_path}[/bold]\n\n"
            f"[dim]Changes will take effect on next run.[/dim]",
            title="Import Complete",
        )
        return True

    except yaml.YAMLError as e:
        warning_panel(f"Invalid YAML in import file: {e}", title="Import Failed")
        return False
    except OSError as e:
        warning_panel(f"Failed to read import file: {e}", title="Import Failed")
        return False


# =============================================================================
# Main Command Function
# =============================================================================


def tune_command(
    agent_name: str | None = None,
    list_agents: bool = False,
    edit: bool = False,
    reset: bool = False,
    export_path: Path | None = None,
    import_path: Path | None = None,
    json_output: bool = False,
) -> None:
    """Execute the tune command.

    Args:
        agent_name: Agent to view/modify.
        list_agents: List all configurable agents.
        edit: Edit the agent's template using $EDITOR.
        reset: Reset agent template to default.
        export_path: Path to export template to.
        import_path: Path to import template from.
        json_output: Output results as JSON.
    """
    logger.debug(
        "tune_command_invoked",
        agent_name=agent_name,
        list_agents=list_agents,
        edit=edit,
        reset=reset,
        export_path=str(export_path) if export_path else None,
        import_path=str(import_path) if import_path else None,
        json_output=json_output,
    )

    # Normalize agent name to lowercase
    normalized_agent = agent_name.lower() if agent_name else None

    # --list flag: show all agents
    if list_agents:
        _display_agents_list(json_output=json_output)
        return

    # No agent and no --list: show help
    if normalized_agent is None:
        if not json_output:
            info_panel(
                "Use [bold]yolo tune --list[/bold] to see all configurable agents.\n"
                "Use [bold]yolo tune <agent>[/bold] to view an agent's template.",
                title="Agent Templates",
            )
        else:
            console.print_json(json.dumps({"error": "No agent specified. Use --list to see available agents."}))
        return

    # Validate agent name
    if normalized_agent not in VALID_AGENTS:
        valid_agents = ", ".join(sorted(VALID_AGENTS))
        warning_panel(
            f"Unknown agent: '{agent_name}'\n\n"
            f"[dim]Valid agents:[/dim]\n  {valid_agents}",
            title="Invalid Agent",
        )
        raise typer.Exit(code=1)

    # --reset flag: reset to default
    if reset:
        if reset_template(normalized_agent):
            success_panel(
                f"Template for {normalized_agent} agent reset to default.\n"
                f"[dim]Changes will take effect on next run.[/dim]",
                title="Template Reset",
            )
        else:
            info_panel(
                f"Agent {normalized_agent} is already using the default template.",
                title="No Change",
            )
        return

    # --export flag: export template
    if export_path:
        _export_template(normalized_agent, export_path)
        return

    # --import flag: import template
    if import_path:
        _import_template(normalized_agent, import_path)
        return

    # --edit flag: edit template
    if edit:
        _edit_template(normalized_agent)
        return

    # Default: display template
    _display_agent_template(normalized_agent, json_output=json_output)
