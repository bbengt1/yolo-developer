"""YOLO config command implementation (Story 12.8).

This module provides the yolo config command which manages project configuration.

Subcommands:
- `yolo config` - Show current configuration
- `yolo config set key value` - Set a configuration value
- `yolo config export` - Export configuration to file
- `yolo config import file` - Import configuration from file

Examples:
    yolo config                              # Show config
    yolo config --json                       # Show config as JSON
    yolo config set llm.cheap_model gpt-4o   # Set nested value
    yolo config export                       # Export to yolo-config-export.yaml
    yolo config export -o custom.yaml        # Export to custom path
    yolo config import backup.yaml           # Import from file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
import typer
import yaml
from rich.tree import Tree

from yolo_developer.cli.display import (
    console,
    error_panel,
    success_panel,
    warning_panel,
)
from yolo_developer.config import (
    ConfigurationError,
    YoloConfig,
    export_config,
    import_config,
    load_config,
)

logger = structlog.get_logger(__name__)

# Configuration keys that cannot be set via CLI (security)
PROTECTED_KEYS = frozenset(
    {
        "llm.openai_api_key",
        "llm.anthropic_api_key",
    }
)

# Valid top-level and nested configuration keys
VALID_CONFIG_KEYS = frozenset(
    {
        "project_name",
        "llm.cheap_model",
        "llm.premium_model",
        "llm.best_model",
        "quality.test_coverage_threshold",
        "quality.confidence_threshold",
        "quality.seed_thresholds.overall",
        "quality.seed_thresholds.ambiguity",
        "quality.seed_thresholds.sop",
        "memory.persist_path",
        "memory.vector_store_type",
        "memory.graph_store_type",
    }
)

# Default export filename
DEFAULT_EXPORT_FILENAME = "yolo-config-export.yaml"

# Masked value placeholder
MASKED_VALUE = "****"


def _load_config_safe() -> YoloConfig | None:
    """Load configuration, returning None if it doesn't exist or fails.

    Returns:
        YoloConfig if successful, None otherwise.
    """
    try:
        return load_config()
    except ConfigurationError:
        return None


def _mask_api_keys(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Mask API keys in configuration dictionary.

    Args:
        config_dict: Configuration dictionary to mask.

    Returns:
        Dictionary with API keys masked.
    """
    result = dict(config_dict)
    if "llm" in result:
        llm = dict(result["llm"])
        if llm.get("openai_api_key"):
            llm["openai_api_key"] = MASKED_VALUE
        if llm.get("anthropic_api_key"):
            llm["anthropic_api_key"] = MASKED_VALUE
        result["llm"] = llm
    return result


def _config_to_dict(config: YoloConfig, mask_secrets: bool = True) -> dict[str, Any]:
    """Convert configuration to dictionary.

    Args:
        config: YoloConfig instance.
        mask_secrets: Whether to mask API keys.

    Returns:
        Dictionary representation of configuration.
    """
    data = config.model_dump(mode="json")
    if mask_secrets:
        data = _mask_api_keys(data)
    return data


def _display_config_tree(config: YoloConfig, mask_secrets: bool = True) -> None:
    """Display configuration as a Rich tree.

    Args:
        config: YoloConfig instance to display.
        mask_secrets: Whether to mask API keys.
    """
    tree = Tree("[bold cyan]yolo.yaml[/bold cyan]")

    # Project name
    tree.add(f"[cyan]project_name:[/cyan] {config.project_name}")

    # LLM section
    llm_branch = tree.add("[cyan]llm:[/cyan]")
    llm_branch.add(f"cheap_model: {config.llm.cheap_model}")
    llm_branch.add(f"premium_model: {config.llm.premium_model}")
    llm_branch.add(f"best_model: {config.llm.best_model}")

    # Mask API keys
    openai_display = (
        MASKED_VALUE if config.llm.openai_api_key and mask_secrets else "[dim]not set[/dim]"
    )
    anthropic_display = (
        MASKED_VALUE if config.llm.anthropic_api_key and mask_secrets else "[dim]not set[/dim]"
    )
    if config.llm.openai_api_key and not mask_secrets:
        openai_display = config.llm.openai_api_key.get_secret_value()
    if config.llm.anthropic_api_key and not mask_secrets:
        anthropic_display = config.llm.anthropic_api_key.get_secret_value()

    llm_branch.add(f"openai_api_key: {openai_display}")
    llm_branch.add(f"anthropic_api_key: {anthropic_display}")

    # Quality section
    quality_branch = tree.add("[cyan]quality:[/cyan]")
    quality_branch.add(f"test_coverage_threshold: {config.quality.test_coverage_threshold}")
    quality_branch.add(f"confidence_threshold: {config.quality.confidence_threshold}")

    # Seed thresholds
    seed_branch = quality_branch.add("[cyan]seed_thresholds:[/cyan]")
    seed_branch.add(f"overall: {config.quality.seed_thresholds.overall}")
    seed_branch.add(f"ambiguity: {config.quality.seed_thresholds.ambiguity}")
    seed_branch.add(f"sop: {config.quality.seed_thresholds.sop}")

    # Gate thresholds (if any)
    if config.quality.gate_thresholds:
        gates_branch = quality_branch.add("[cyan]gate_thresholds:[/cyan]")
        for gate_name, gate_config in config.quality.gate_thresholds.items():
            gates_branch.add(f"{gate_name}: min_score={gate_config.min_score}")

    # Critical paths
    if config.quality.critical_paths:
        paths_str = ", ".join(config.quality.critical_paths)
        quality_branch.add(f"critical_paths: [{paths_str}]")

    # Memory section
    memory_branch = tree.add("[cyan]memory:[/cyan]")
    memory_branch.add(f"persist_path: {config.memory.persist_path}")
    memory_branch.add(f"vector_store_type: {config.memory.vector_store_type}")
    memory_branch.add(f"graph_store_type: {config.memory.graph_store_type}")

    console.print(tree)


def _get_nested_value(data: dict[str, Any], key_path: str) -> Any:
    """Get a value from nested dict using dotted key path.

    Args:
        data: Dictionary to traverse.
        key_path: Dotted key path (e.g., 'llm.cheap_model').

    Returns:
        The value at the key path.

    Raises:
        KeyError: If the key path is invalid.
    """
    keys = key_path.split(".")
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Unknown configuration key: {key_path}")
        current = current[key]
    return current


def _set_nested_value(data: dict[str, Any], key_path: str, value: Any) -> None:
    """Set a value in nested dict using dotted key path.

    Creates intermediate dictionaries if they don't exist. This is safe because
    the key path is validated against VALID_CONFIG_KEYS before this function
    is called.

    Args:
        data: Dictionary to modify.
        key_path: Dotted key path (e.g., 'llm.cheap_model').
        value: Value to set.

    Raises:
        KeyError: If the key path traverses a non-dict value.
    """
    keys = key_path.split(".")
    current = data
    for key in keys[:-1]:
        if not isinstance(current, dict):
            raise KeyError(f"Cannot traverse non-dict at key: {key_path}")
        if key not in current:
            current[key] = {}
        current = current[key]

    final_key = keys[-1]
    if not isinstance(current, dict):
        raise KeyError(f"Cannot set value at non-dict path: {key_path}")
    current[final_key] = value


def _convert_value_type(value: str, key_path: str) -> Any:
    """Convert string value to appropriate type based on key.

    Args:
        value: String value to convert.
        key_path: Key path to determine expected type.

    Returns:
        Converted value.
    """
    # Float fields
    float_keys = {
        "quality.test_coverage_threshold",
        "quality.confidence_threshold",
        "quality.seed_thresholds.overall",
        "quality.seed_thresholds.ambiguity",
        "quality.seed_thresholds.sop",
    }

    if key_path in float_keys:
        try:
            return float(value)
        except ValueError:
            return value

    # Boolean handling
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    return value


def _validate_key(key: str) -> str | None:
    """Validate a configuration key.

    Args:
        key: Key to validate.

    Returns:
        Error message if invalid, None if valid.
    """
    if key in PROTECTED_KEYS:
        return f"Cannot set '{key}' via CLI. Use environment variables instead:\n  YOLO_LLM__OPENAI_API_KEY or YOLO_LLM__ANTHROPIC_API_KEY"

    if key not in VALID_CONFIG_KEYS:
        # Check if it might be a partial match (for better error messages)
        similar = [k for k in VALID_CONFIG_KEYS if key in k or k.startswith(key.split(".")[0])]
        if similar:
            return f"Unknown configuration key: '{key}'\n  Did you mean: {', '.join(sorted(similar)[:3])}?"
        return f"Unknown configuration key: '{key}'"

    return None


def show_config(json_output: bool = False, no_mask: bool = False) -> None:
    """Show current configuration.

    Args:
        json_output: Output as JSON instead of Rich tree.
        no_mask: Show API keys unmasked (for debugging).
    """
    logger.debug("show_config_invoked", json_output=json_output)

    config = _load_config_safe()
    if config is None:
        if json_output:
            print(
                json.dumps(
                    {
                        "error": "No configuration found",
                        "suggestion": "Run 'yolo init' to create a project",
                    }
                )
            )
        else:
            warning_panel(
                "No configuration file found (yolo.yaml).\n\n"
                "Run 'yolo init' to create a new project with configuration.",
                title="Configuration Not Found",
            )
        raise typer.Exit(code=1)

    mask_secrets = not no_mask

    if json_output:
        data = _config_to_dict(config, mask_secrets=mask_secrets)
        print(json.dumps(data, indent=2))
    else:
        _display_config_tree(config, mask_secrets=mask_secrets)


def set_config_value(key: str, value: str, json_output: bool = False) -> None:
    """Set a configuration value.

    Args:
        key: Configuration key (supports dotted notation).
        value: Value to set.
        json_output: Output as JSON instead of Rich display.
    """
    logger.debug("set_config_value_invoked", key=key, value=value)

    # Validate key
    error = _validate_key(key)
    if error:
        if json_output:
            print(json.dumps({"error": error, "status": "failed"}))
        else:
            error_panel(error, title="Invalid Key")
        raise typer.Exit(code=1)

    # Check config file exists
    config_path = Path("yolo.yaml")
    if not config_path.exists():
        if json_output:
            print(
                json.dumps(
                    {"error": "No configuration file found", "suggestion": "Run 'yolo init' first"}
                )
            )
        else:
            warning_panel(
                "No configuration file found (yolo.yaml).\n\n"
                "Run 'yolo init' to create a project first.",
                title="Configuration Not Found",
            )
        raise typer.Exit(code=1)

    # Load existing config
    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        if json_output:
            print(json.dumps({"error": f"Invalid YAML: {e}", "status": "failed"}))
        else:
            error_panel(f"Failed to parse yolo.yaml: {e}", title="YAML Error")
        raise typer.Exit(code=1) from None

    # Get old value (for display)
    try:
        old_value = _get_nested_value(config_data, key)
    except KeyError:
        old_value = None

    # Convert and set new value
    converted_value = _convert_value_type(value, key)
    _set_nested_value(config_data, key, converted_value)

    # Validate the new config by attempting to load it
    try:
        YoloConfig(**config_data)
    except Exception as e:
        if json_output:
            print(json.dumps({"error": f"Invalid configuration: {e}", "status": "failed"}))
        else:
            error_panel(f"Invalid configuration value: {e}", title="Validation Error")
        raise typer.Exit(code=1) from None

    # Write updated config
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)

    if json_output:
        print(
            json.dumps(
                {
                    "key": key,
                    "old_value": old_value,
                    "new_value": converted_value,
                    "status": "success",
                }
            )
        )
    else:
        success_panel(
            f"Configuration updated:\n\n  Key: {key}\n  Old: {old_value}\n  New: {converted_value}",
            title="Configuration Set",
        )


def export_config_command(output_path: Path | None = None, json_output: bool = False) -> None:
    """Export configuration to file.

    Args:
        output_path: Path to export to (default: yolo-config-export.yaml).
        json_output: Output as JSON instead of Rich display.
    """
    logger.debug("export_config_invoked", output_path=output_path)

    config = _load_config_safe()
    if config is None:
        if json_output:
            print(json.dumps({"error": "No configuration found", "status": "failed"}))
        else:
            warning_panel(
                "No configuration file found (yolo.yaml).\n\n"
                "Run 'yolo init' to create a project first.",
                title="Configuration Not Found",
            )
        raise typer.Exit(code=1)

    export_path = output_path or Path(DEFAULT_EXPORT_FILENAME)

    try:
        export_config(config, export_path)
    except Exception as e:
        if json_output:
            print(json.dumps({"error": str(e), "status": "failed"}))
        else:
            error_panel(f"Failed to export configuration: {e}", title="Export Error")
        raise typer.Exit(code=1) from None

    if json_output:
        print(
            json.dumps(
                {
                    "path": str(export_path.absolute()),
                    "status": "success",
                }
            )
        )
    else:
        success_panel(
            f"Configuration exported to: {export_path.absolute()}\n\n"
            f"Note: API keys are excluded for security.\n"
            f"Set them via environment variables when importing.",
            title="Export Complete",
        )


def import_config_command(source_path: Path, json_output: bool = False) -> None:
    """Import configuration from file.

    Args:
        source_path: Path to import from.
        json_output: Output as JSON instead of Rich display.
    """
    logger.debug("import_config_invoked", source_path=source_path)

    if not source_path.exists():
        if json_output:
            print(json.dumps({"error": f"File not found: {source_path}", "status": "failed"}))
        else:
            error_panel(f"File not found: {source_path}", title="Import Error")
        raise typer.Exit(code=1)

    target_path = Path("yolo.yaml")

    try:
        import_config(source_path, target_path)
    except ConfigurationError as e:
        if json_output:
            print(json.dumps({"error": str(e), "status": "failed"}))
        else:
            error_panel(str(e), title="Import Error")
        raise typer.Exit(code=1) from None
    except Exception as e:
        if json_output:
            print(json.dumps({"error": str(e), "status": "failed"}))
        else:
            error_panel(f"Failed to import configuration: {e}", title="Import Error")
        raise typer.Exit(code=1) from None

    if json_output:
        print(
            json.dumps(
                {
                    "source": str(source_path.absolute()),
                    "target": str(target_path.absolute()),
                    "status": "success",
                }
            )
        )
    else:
        success_panel(
            f"Configuration imported from: {source_path}\n"
            f"Written to: {target_path}\n\n"
            f"Remember to set API keys via environment variables.",
            title="Import Complete",
        )


def config_command() -> None:
    """Execute the config command (placeholder - replaced by subcommands)."""
    logger.debug("config_command_invoked")
    # This is called when no subcommand is provided
    # The actual implementation is in show_config
    show_config()
