"""Configuration loading logic for YOLO Developer.

This module provides the load_config function to load configuration from
YAML files with environment variable overrides. The priority order is:
defaults → YAML → environment variables.

Configuration errors are wrapped in ConfigurationError with helpful
messages including line numbers for YAML syntax errors.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from yolo_developer.config.schema import YoloConfig
from yolo_developer.config.validators import validate_config

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Error raised when configuration loading or validation fails.

    This exception provides helpful error messages including:
    - Line and column numbers for YAML syntax errors
    - Field paths for validation errors
    - Actionable guidance for fixing the issue
    """

    pass


def load_config(config_path: Path | None = None) -> YoloConfig:
    """Load configuration from YAML file with environment variable overrides.

    The configuration priority is: defaults → YAML → environment variables.

    Args:
        config_path: Path to the YAML configuration file. If None, defaults to
            'yolo.yaml' in the current directory.

    Returns:
        YoloConfig instance with merged configuration values.

    Raises:
        ConfigurationError: If YAML parsing fails or configuration validation fails.

    Example:
        >>> config = load_config()  # Loads from ./yolo.yaml if it exists
        >>> config = load_config(Path("/path/to/custom.yaml"))
        >>> config.llm.cheap_model
        'gpt-5.2-instant'
    """
    yaml_data: dict[str, Any] = {}

    path = config_path or Path("yolo.yaml")

    if path.exists():
        yaml_data = _load_yaml_file(path)

    # Merge with environment variables (env vars take priority)
    merged_data = _merge_with_env_vars(yaml_data)

    config = _create_config(merged_data)

    # Run comprehensive validation (Story 1.7)
    validation_result = validate_config(config)

    # Raise ConfigurationError if any validation errors exist
    if not validation_result.is_valid:
        error_messages = [f"  {err.field}: {err.message}" for err in validation_result.errors]
        raise ConfigurationError("Configuration validation failed:\n" + "\n".join(error_messages))

    # Log warnings for non-fatal validation issues (e.g., missing API keys)
    for warning in validation_result.warnings:
        logger.warning("%s: %s", warning.field, warning.message)

    return config


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file with error handling.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML data as a dictionary.

    Raises:
        ConfigurationError: If YAML parsing fails.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except yaml.YAMLError as e:
        raise _create_yaml_error(e, path) from e


def _create_yaml_error(error: yaml.YAMLError, path: Path) -> ConfigurationError:
    """Create a helpful ConfigurationError from a YAML parsing error.

    Args:
        error: The original YAML error.
        path: Path to the file that failed to parse.

    Returns:
        ConfigurationError with line number and helpful message.
    """
    if hasattr(error, "problem_mark") and error.problem_mark is not None:
        mark = error.problem_mark
        line = mark.line + 1  # Convert to 1-indexed
        column = mark.column + 1

        problem = getattr(error, "problem", "unknown error")

        msg = (
            f"Invalid YAML at line {line}, column {column} in {path}:\n"
            f"  {problem}\n\n"
            f"  Hint: Check for proper indentation and spacing around line {line}."
        )
        return ConfigurationError(msg)

    # Fallback for errors without position information
    return ConfigurationError(f"Failed to parse YAML file {path}: {error}")


def _merge_with_env_vars(yaml_data: dict[str, Any]) -> dict[str, Any]:
    """Merge YAML data with environment variables.

    Environment variables with YOLO_ prefix override YAML values.
    Uses __ as nested delimiter (e.g., YOLO_LLM__CHEAP_MODEL).

    Args:
        yaml_data: Configuration data from YAML file.

    Returns:
        Merged configuration with env vars taking priority.
    """
    result = dict(yaml_data)
    env_prefix = "YOLO_"
    nested_delimiter = "__"

    for key, value in os.environ.items():
        if not key.startswith(env_prefix):
            continue

        # Remove prefix and split by nested delimiter
        config_key = key[len(env_prefix) :]
        parts = config_key.lower().split(nested_delimiter)

        current: dict[str, Any] = result
        for part in parts[:-1]:
            if part not in current or not isinstance(current.get(part), dict):
                current[part] = {}
            current = current[part]

        current[parts[-1]] = _convert_value(value, tuple(parts))

    return result


def _convert_value(value: str, path: tuple[str, ...]) -> Any:
    """Convert string environment variable to appropriate type.

    Attempts automatic type inference: tries float conversion first
    (which handles integers), then falls back to string. This allows
    new numeric config fields to work without code changes.

    Args:
        value: String value from environment variable.
        path: Full config path parts (e.g., ("quality", "test_coverage_threshold")).

    Returns:
        Converted value (float for numeric values, string otherwise).
    """
    # Known float fields that should always be converted
    float_fields = {
        ("quality", "test_coverage_threshold"),
        ("quality", "confidence_threshold"),
        ("quality", "seed_thresholds", "overall"),
        ("quality", "seed_thresholds", "ambiguity"),
        ("quality", "seed_thresholds", "sop"),
    }

    # Always convert known float fields
    if path in float_fields:
        try:
            return float(value)
        except ValueError:
            return value

    # Try automatic type inference for other fields
    # This allows new numeric fields to work without code changes
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    try:
        # Try float first (also handles integers like "42" -> 42.0)
        float_val = float(value)
        # Return int if it's a whole number
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        return value


def _create_config(merged_data: dict[str, Any]) -> YoloConfig:
    """Create a YoloConfig from merged configuration data.

    Args:
        merged_data: Configuration data merged from YAML and env vars.

    Returns:
        YoloConfig instance.

    Raises:
        ConfigurationError: If configuration validation fails.
    """
    try:
        # Pass merged data directly - env vars already applied
        return YoloConfig(**merged_data)
    except ValidationError as e:
        raise _create_validation_error(e) from e


def _create_validation_error(error: ValidationError) -> ConfigurationError:
    """Create a helpful ConfigurationError from a Pydantic ValidationError.

    Args:
        error: The original Pydantic validation error.

    Returns:
        ConfigurationError with field path and helpful message.
    """
    errors = error.errors()
    if not errors:
        return ConfigurationError(f"Configuration validation failed: {error}")

    messages: list[str] = []
    for err in errors:
        # Build field path (e.g., "quality.test_coverage_threshold")
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "invalid value")
        error_type = err.get("type", "")

        field_msg = f"  {loc}: {msg}"

        # Add hints for common error types
        if error_type == "less_than_equal":
            field_msg += " (value must be <= 1.0)"
        elif error_type == "greater_than_equal":
            field_msg += " (value must be >= 0.0)"

        messages.append(field_msg)

    return ConfigurationError("Invalid configuration value(s):\n" + "\n".join(messages))
