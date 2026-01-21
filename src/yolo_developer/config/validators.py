"""Configuration validation for YOLO Developer.

This module provides comprehensive validation beyond Pydantic's built-in
field constraints. Validation happens after config creation to collect
all errors before failing.

Design Decision - Errors vs Warnings:
    Pydantic handles fatal validation errors (required fields, value ranges)
    before this module runs. This module adds additional checks that produce
    WARNINGS (not errors) because:

    1. Path validation: Directories may be created at runtime
    2. API key validation: Keys may be set later or in different environments

    The ValidationResult.errors list exists for future validators that need
    to produce fatal errors. Currently all validators return warnings only.

Validation Categories:
- Path validation: Verify persist_path parent is writable (warning if not)
- API key validation: Check provider-specific requirements based on models (warning if missing)

Example:
    >>> from yolo_developer.config import load_config
    >>> config = load_config()  # Validation runs automatically
    >>> # Or manually:
    >>> from yolo_developer.config.validators import validate_config
    >>> result = validate_config(config)
    >>> if not result.is_valid:
    ...     for error in result.errors:
    ...         print(f"{error.field}: {error.message}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yolo_developer.config.schema import YoloConfig


@dataclass
class ValidationIssue:
    """A single validation error or warning.

    Attributes:
        field: The field path that failed validation (e.g., "memory.persist_path").
        message: Human-readable description of the validation failure.
        value: The actual value that failed validation (optional, may be masked).
        constraint: The expected constraint that was violated (optional).
    """

    field: str
    message: str
    value: str | None = None
    constraint: str | None = None


@dataclass
class ValidationResult:
    """Result of configuration validation.

    Contains separate lists for errors (fatal) and warnings (non-fatal).
    Use is_valid property to check if configuration can proceed.

    Attributes:
        errors: List of fatal validation errors that must be fixed.
        warnings: List of non-fatal warnings (e.g., missing API keys).
    """

    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True if no errors exist (warnings are OK)."""
        return len(self.errors) == 0


def validate_config(config: YoloConfig) -> ValidationResult:
    """Run all validators and collect results.

    This function runs all validation checks and collects errors and warnings
    into a single ValidationResult. Errors are fatal and should prevent
    execution. Warnings are informational and can be logged.

    Args:
        config: The YoloConfig instance to validate.

    Returns:
        ValidationResult containing all errors and warnings.
    """
    result = ValidationResult()

    # Run each validator and collect issues
    # Path validation produces warnings since directories may be created at runtime
    result.warnings.extend(_validate_paths(config))
    result.errors.extend(_validate_provider_api_keys(config))
    result.warnings.extend(_validate_api_keys_for_models(config))

    return result


def _validate_paths(config: YoloConfig) -> list[ValidationIssue]:
    """Validate file paths are accessible.

    Checks that the memory.persist_path parent directory exists and is writable.
    This validation produces warnings (not errors) since directories may be
    created at runtime or the path may be intentionally set for a different
    environment.

    Args:
        config: The YoloConfig instance to validate.

    Returns:
        List of validation warnings for potentially inaccessible paths.
    """
    import os
    from pathlib import Path

    warnings: list[ValidationIssue] = []

    persist_path = Path(config.memory.persist_path)

    # For relative paths, resolve against current directory
    if not persist_path.is_absolute():
        persist_path = Path.cwd() / persist_path

    # Check if the path or its parent exists and is writable
    # Handle PermissionError when traversing restricted directories
    check_path = persist_path
    try:
        while not check_path.exists():
            check_path = check_path.parent
            if check_path == check_path.parent:
                # Reached filesystem root without finding existing directory
                break
    except PermissionError:
        # Can't even check if path exists - definitely not accessible
        warnings.append(
            ValidationIssue(
                field="memory.persist_path",
                message=f"Directory is not accessible: {persist_path.absolute()}",
                value=str(config.memory.persist_path),
                constraint="parent directory should be accessible",
            )
        )
        return warnings

    if check_path.exists() and not os.access(check_path, os.W_OK):
        warnings.append(
            ValidationIssue(
                field="memory.persist_path",
                message=f"Directory may not be writable: {check_path.absolute()}",
                value=str(config.memory.persist_path),
                constraint="parent directory should be writable",
            )
        )

    return warnings


def _validate_api_keys_for_models(config: YoloConfig) -> list[ValidationIssue]:
    """Validate API keys are present for configured model providers.

    Checks if models requiring specific API keys have those keys configured.
    This produces warnings, not errors, since API keys may be set at runtime.

    Model detection patterns:
    - OpenAI: gpt-*, o1-*, o3-*
    - Anthropic: claude-*

    Args:
        config: The YoloConfig instance to validate.

    Returns:
        List of validation warnings for missing API keys.
    """
    warnings: list[ValidationIssue] = []

    if config.llm.provider != "auto" or config.llm.hybrid.enabled:
        return warnings

    models = [
        config.llm.cheap_model,
        config.llm.premium_model,
        config.llm.best_model,
    ]

    # Check for OpenAI models
    openai_prefixes = ("gpt-", "o1-", "o3-")
    openai_models = [m for m in models if m.startswith(openai_prefixes)]
    if openai_models and config.llm.openai_api_key is None:
        warnings.append(
            ValidationIssue(
                field="llm.openai_api_key",
                message=(
                    f"OpenAI models configured ({', '.join(openai_models)}) "
                    "but YOLO_LLM__OPENAI__API_KEY not set"
                ),
            )
        )

    # Check for Anthropic models
    anthropic_models = [m for m in models if m.startswith("claude-")]
    if anthropic_models and config.llm.anthropic_api_key is None:
        warnings.append(
            ValidationIssue(
                field="llm.anthropic_api_key",
                message=(
                    f"Anthropic models configured ({', '.join(anthropic_models)}) "
                    "but YOLO_LLM__ANTHROPIC_API_KEY not set"
                ),
            )
        )

    return warnings


def _validate_provider_api_keys(config: YoloConfig) -> list[ValidationIssue]:
    """Validate API keys based on provider selection.

    When provider or hybrid routing explicitly uses a provider, missing
    API keys become fatal errors.
    """
    errors: list[ValidationIssue] = []

    openai_required = False
    anthropic_required = False

    if config.llm.provider == "openai":
        openai_required = True
    elif config.llm.provider == "anthropic":
        anthropic_required = True
    elif config.llm.provider == "hybrid" or config.llm.hybrid.enabled:
        routing = config.llm.hybrid.routing
        openai_required = "openai" in {
            routing.code_generation,
            routing.code_review,
            routing.documentation,
            routing.testing,
            routing.analysis,
            routing.architecture,
        }
        anthropic_required = "anthropic" in {
            routing.code_generation,
            routing.code_review,
            routing.documentation,
            routing.testing,
            routing.analysis,
            routing.architecture,
        }

    if openai_required and config.llm.openai.api_key is None:
        errors.append(
            ValidationIssue(
                field="llm.openai.api_key",
                message="OpenAI provider selected but no API key configured",
            )
        )

    if anthropic_required and config.llm.anthropic_api_key is None:
        errors.append(
            ValidationIssue(
                field="llm.anthropic_api_key",
                message="Anthropic provider selected but no API key configured",
            )
        )

    return errors
