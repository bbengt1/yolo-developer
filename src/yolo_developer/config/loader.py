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

_ALLOW_YAML_SECRETS_ENV = "YOLO_ALLOW_YAML_SECRETS"
_YAML_SECRET_FIELDS = (
    ("llm", "openai_api_key"),
    ("llm", "anthropic_api_key"),
    ("llm", "openai", "api_key"),
)
_YAML_SECRET_ENV_KEYS = {
    "YOLO_LLM__OPENAI__API_KEY": ("openai", "api_key"),
    "YOLO_LLM__OPENAI_API_KEY": ("openai_api_key",),
    "YOLO_LLM__ANTHROPIC_API_KEY": ("anthropic_api_key",),
}
_YAML_ALLOW_KEY = "YOLO_ALLOW_YAML_SECRETS"

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
        yaml_data = _sanitize_yaml_secrets(yaml_data)

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
        if key == _ALLOW_YAML_SECRETS_ENV:
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


def _sanitize_yaml_secrets(yaml_data: dict[str, Any]) -> dict[str, Any]:
    if not yaml_data:
        return yaml_data
    allow_yaml_secrets = _allow_yaml_secrets()
    found_fields = _find_yaml_secret_fields(yaml_data)
    if not found_fields:
        return yaml_data
    if allow_yaml_secrets:
        _normalize_yaml_env_secrets(yaml_data)
        logger.warning(
            "YAML secrets enabled via %s. Do not commit secrets to version control.",
            _ALLOW_YAML_SECRETS_ENV,
        )
        return yaml_data
    _remove_yaml_secret_fields(yaml_data)
    logger.warning(
        "YAML secrets ignored (%s). Set %s=1 to allow for local development only.",
        ", ".join(found_fields),
        _ALLOW_YAML_SECRETS_ENV,
    )
    return yaml_data


def _allow_yaml_secrets() -> bool:
    value = os.environ.get(_ALLOW_YAML_SECRETS_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _find_yaml_secret_fields(yaml_data: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for path in _YAML_SECRET_FIELDS:
        if _get_nested(yaml_data, path) is not None:
            found.append(".".join(path))
    llm_data = yaml_data.get("llm")
    if isinstance(llm_data, dict):
        for key in _YAML_SECRET_ENV_KEYS:
            if key in llm_data:
                found.append(f"llm.{key}")
    return found


def _get_nested(yaml_data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = yaml_data
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _remove_yaml_secret_fields(yaml_data: dict[str, Any]) -> None:
    for path in _YAML_SECRET_FIELDS:
        current: Any = yaml_data
        for part in path[:-1]:
            if not isinstance(current, dict):
                break
            current = current.get(part)
        else:
            if isinstance(current, dict):
                current.pop(path[-1], None)
    llm_data = yaml_data.get("llm")
    if isinstance(llm_data, dict):
        for key in _YAML_SECRET_ENV_KEYS:
            llm_data.pop(key, None)
        llm_data.pop(_YAML_ALLOW_KEY, None)


def _normalize_yaml_env_secrets(yaml_data: dict[str, Any]) -> None:
    llm_data = yaml_data.get("llm")
    if not isinstance(llm_data, dict):
        return
    mapped: list[str] = []
    for env_key, target in _YAML_SECRET_ENV_KEYS.items():
        if env_key in llm_data:
            value = llm_data.pop(env_key)
            _set_nested(llm_data, target, value)
            mapped.append(env_key)
    if _YAML_ALLOW_KEY in llm_data:
        llm_data.pop(_YAML_ALLOW_KEY, None)
        logger.warning(
            "YAML key %s is ignored. Set it as an environment variable instead.",
            _YAML_ALLOW_KEY,
        )
    if mapped:
        logger.warning(
            "Mapped YAML env-style keys (%s) to config fields. Prefer llm.openai_api_key or llm.openai.api_key.",
            ", ".join(mapped),
        )


def _set_nested(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = target
    for part in path[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[path[-1]] = value


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
