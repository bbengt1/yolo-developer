"""Configuration schema for YOLO Developer using Pydantic Settings.

This module defines the strongly-typed configuration schema per ADR-008.
Configuration is validated at system boundaries using Pydantic v2.

Environment Variable Pattern
-----------------------------
Environment variables use the ``YOLO_`` prefix with ``__`` as the nested delimiter:

- ``YOLO_PROJECT_NAME``: Project name (required)
- ``YOLO_LLM__PROVIDER``: LLM provider selection (auto/openai/anthropic/hybrid)
- ``YOLO_LLM__CHEAP_MODEL``: LLM model for routine tasks
- ``YOLO_LLM__PREMIUM_MODEL``: LLM model for complex reasoning
- ``YOLO_LLM__BEST_MODEL``: LLM model for critical decisions
- ``YOLO_LLM__OPENAI__API_KEY``: OpenAI API key (nested, preferred)
- ``YOLO_LLM__OPENAI_API_KEY``: OpenAI API key (legacy)
- ``YOLO_LLM__OPENAI__CODE_MODEL``: OpenAI model for code tasks
- ``YOLO_LLM__ANTHROPIC_API_KEY``: Anthropic API key (secrets only via env vars)
- ``YOLO_LLM__HYBRID__ENABLED``: Enable hybrid routing
- ``YOLO_LLM__HYBRID__ROUTING__CODE_GENERATION``: Provider for code generation
- ``YOLO_QUALITY__TEST_COVERAGE_THRESHOLD``: Test coverage threshold (0.0-1.0)
- ``YOLO_QUALITY__CONFIDENCE_THRESHOLD``: Confidence threshold (0.0-1.0)
- ``YOLO_QUALITY__SEED_THRESHOLDS__OVERALL``: Seed overall quality threshold (0.0-1.0, default 0.70)
- ``YOLO_QUALITY__SEED_THRESHOLDS__AMBIGUITY``: Seed ambiguity threshold (0.0-1.0, default 0.60)
- ``YOLO_QUALITY__SEED_THRESHOLDS__SOP``: Seed SOP compliance threshold (0.0-1.0, default 0.80)
- ``YOLO_MEMORY__PERSIST_PATH``: Memory persistence directory
- ``YOLO_MEMORY__VECTOR_STORE_TYPE``: Vector store type (chromadb)
- ``YOLO_MEMORY__GRAPH_STORE_TYPE``: Graph store type (json, neo4j)

Configuration Priority Order
----------------------------
Configuration values are resolved in the following order (later overrides earlier):

1. Defaults (defined in schema)
2. YAML file (yolo.yaml)
3. Environment variables

API Key Security
----------------
API keys (openai_api_key, openai.api_key, anthropic_api_key) are:

- Set via environment variables ONLY (never in YAML files)
- Stored as SecretStr for automatic masking in logs/repr
- Accessible via ``.get_secret_value()`` method when needed
- Never written to config file exports
 - For local development only, set ``YOLO_ALLOW_YAML_SECRETS=1`` to allow YAML secrets

Example Usage
-------------
>>> from yolo_developer.config import load_config
>>> config = load_config()
>>> config.llm.cheap_model
'gpt-5.2-instant'
>>> if config.llm.openai_api_key:
...     api_key = config.llm.openai_api_key.get_secret_value()
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from yolo_developer.github.config import GitHubConfig

OPENAI_CHEAP_MODEL_DEFAULT = "gpt-5.2-instant"
OPENAI_PREMIUM_MODEL_DEFAULT = "gpt-5.2-thinking"
OPENAI_CODE_MODEL_DEFAULT = "gpt-5.2-pro"
OPENAI_REASONING_MODEL_DEFAULT = "gpt-5.2-pro"

LLM_CHEAP_MODEL_DEFAULT = OPENAI_CHEAP_MODEL_DEFAULT
LLM_PREMIUM_MODEL_DEFAULT = "claude-sonnet-4-20250514"
LLM_BEST_MODEL_DEFAULT = "claude-opus-4-5-20251101"


class GateThreshold(BaseModel):
    """Configuration for a single gate's threshold.

    Defines the minimum score and blocking behavior for a quality gate.
    Used in QualityConfig.gate_thresholds for per-gate configuration.

    Attributes:
        min_score: Minimum score (0.0-1.0) for this gate to pass.
        blocking: Whether this gate blocks (True) or is advisory (False).

    Example:
        >>> from yolo_developer.config.schema import GateThreshold
        >>> gate = GateThreshold(min_score=0.85, blocking=True)
        >>> gate.min_score
        0.85
    """

    min_score: float = Field(
        default=0.80,
        description="Minimum score (0.0-1.0) for this gate to pass",
        ge=0.0,
        le=1.0,
    )
    blocking: bool = Field(
        default=True,
        description="Whether this gate blocks or is advisory",
    )


LLMProvider = Literal["auto", "openai", "anthropic", "hybrid"]


class OpenAIConfig(BaseModel):
    """OpenAI/Codex configuration for code-optimized models."""

    api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key (set via YOLO_LLM__OPENAI__API_KEY env var)",
    )
    cheap_model: str = Field(
        default=OPENAI_CHEAP_MODEL_DEFAULT,
        description="OpenAI model for routine tasks",
    )
    premium_model: str = Field(
        default=OPENAI_PREMIUM_MODEL_DEFAULT,
        description="OpenAI model for complex reasoning tasks",
    )
    code_model: str = Field(
        default=OPENAI_CODE_MODEL_DEFAULT,
        description="OpenAI model optimized for code generation and review",
    )
    reasoning_model: str | None = Field(
        default=OPENAI_REASONING_MODEL_DEFAULT,
        description="OpenAI model for deep reasoning tasks (optional)",
    )


class HybridRoutingConfig(BaseModel):
    """Task-based routing configuration for hybrid provider mode."""

    code_generation: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="Provider for code generation tasks",
    )
    code_review: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="Provider for code review tasks",
    )
    architecture: Literal["openai", "anthropic"] = Field(
        default="anthropic",
        description="Provider for architecture tasks",
    )
    analysis: Literal["openai", "anthropic"] = Field(
        default="anthropic",
        description="Provider for analysis tasks",
    )
    documentation: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="Provider for documentation tasks",
    )
    testing: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="Provider for testing tasks",
    )


class HybridConfig(BaseModel):
    """Hybrid routing configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable hybrid routing across providers",
    )
    routing: HybridRoutingConfig = Field(
        default_factory=HybridRoutingConfig,
        description="Task routing configuration for hybrid mode",
    )


class LLMConfig(BaseModel):
    """Configuration for LLM provider settings.

    Defines the model tiers used for different task complexities:
    - cheap_model: For routine, low-complexity tasks
    - premium_model: For complex reasoning tasks
    - best_model: For critical decisions requiring highest quality
    """

    cheap_model: str = Field(
        default=LLM_CHEAP_MODEL_DEFAULT,
        description="LLM model for routine, low-complexity tasks",
    )
    premium_model: str = Field(
        default=LLM_PREMIUM_MODEL_DEFAULT,
        description="LLM model for complex reasoning tasks",
    )
    best_model: str = Field(
        default=LLM_BEST_MODEL_DEFAULT,
        description="LLM model for critical decisions requiring highest quality",
    )

    provider: LLMProvider = Field(
        default="auto",
        description="Primary LLM provider selection (auto/openai/anthropic/hybrid)",
    )
    openai: OpenAIConfig = Field(
        default_factory=OpenAIConfig,
        description="OpenAI/Codex provider configuration",
    )
    hybrid: HybridConfig = Field(
        default_factory=HybridConfig,
        description="Hybrid routing configuration",
    )

    # API keys - read from env only, masked in output (Story 1.6)
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key (set via YOLO_LLM__OPENAI_API_KEY env var, legacy)",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key (set via YOLO_LLM__ANTHROPIC_API_KEY env var)",
    )

    @model_validator(mode="after")
    def _sync_api_keys(self) -> LLMConfig:
        """Keep legacy API key fields in sync with nested OpenAI config."""
        if self.openai.api_key is None and self.openai_api_key is not None:
            self.openai.api_key = self.openai_api_key
        if self.openai_api_key is None and self.openai.api_key is not None:
            self.openai_api_key = self.openai.api_key
        return self


class SeedThresholdConfig(BaseModel):
    """Configuration for seed quality thresholds (Story 4.7).

    Defines the minimum thresholds that must be met for seed validation.
    Seeds with scores below these thresholds are rejected unless --force is used.

    Attributes:
        overall: Minimum overall quality score (0.0-1.0, default 0.70).
        ambiguity: Minimum ambiguity score (0.0-1.0, default 0.60).
        sop: Minimum SOP compliance score (0.0-1.0, default 0.80).

    Example:
        >>> from yolo_developer.config.schema import SeedThresholdConfig
        >>> config = SeedThresholdConfig(overall=0.85, ambiguity=0.75)
        >>> config.overall
        0.85
    """

    overall: float = Field(
        default=0.70,
        description="Minimum overall quality score (0.0-1.0) for seed acceptance",
        ge=0.0,
        le=1.0,
    )
    ambiguity: float = Field(
        default=0.60,
        description="Minimum ambiguity resolution score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    sop: float = Field(
        default=0.80,
        description="Minimum SOP compliance score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )


class QualityConfig(BaseModel):
    """Configuration for quality gate thresholds.

    Defines the minimum thresholds that must be met for quality gates
    to pass. All threshold values must be between 0.0 and 1.0.

    Supports both global thresholds and per-gate configuration via
    the gate_thresholds field.

    Attributes:
        test_coverage_threshold: Global test coverage threshold (0.0-1.0).
        confidence_threshold: Global confidence score threshold (0.0-1.0).
        gate_thresholds: Per-gate threshold configuration overrides.
        seed_thresholds: Seed quality threshold configuration (Story 4.7).
        critical_paths: Path patterns that require 100% test coverage (Story 9.2).

    Example:
        >>> from yolo_developer.config.schema import QualityConfig, GateThreshold
        >>> config = QualityConfig(
        ...     test_coverage_threshold=0.85,
        ...     gate_thresholds={"testability": GateThreshold(min_score=0.90)},
        ...     critical_paths=["orchestrator/", "gates/", "agents/"]
        ... )
    """

    test_coverage_threshold: float = Field(
        default=0.80,
        description="Minimum test coverage ratio (0.0-1.0) required for quality gates",
        ge=0.0,
        le=1.0,
    )
    confidence_threshold: float = Field(
        default=0.90,
        description="Minimum confidence score (0.0-1.0) for deployment approval",
        ge=0.0,
        le=1.0,
    )
    gate_thresholds: dict[str, GateThreshold] = Field(
        default_factory=dict,
        description="Per-gate threshold configuration overrides",
    )
    seed_thresholds: SeedThresholdConfig = Field(
        default_factory=SeedThresholdConfig,
        description="Seed quality threshold configuration (Story 4.7)",
    )
    critical_paths: list[str] = Field(
        default_factory=lambda: ["orchestrator/", "gates/", "agents/"],
        description="Path patterns that require 100% test coverage (Story 9.2)",
    )

    def validate_thresholds(self) -> list[str]:
        """Validate all threshold configurations and return errors.

        Checks that all configured gate thresholds have valid min_score
        values in the range 0.0-1.0.

        Returns:
            List of error messages. Empty list if all thresholds are valid.

        Example:
            >>> config = QualityConfig()
            >>> errors = config.validate_thresholds()
            >>> if errors:
            ...     for error in errors:
            ...         print(f"Error: {error}")
        """
        errors: list[str] = []
        for gate_name, config in self.gate_thresholds.items():
            if not 0.0 <= config.min_score <= 1.0:
                errors.append(
                    f"Gate '{gate_name}' min_score must be 0.0-1.0, got {config.min_score}"
                )
        return errors


class MemoryConfig(BaseModel):
    """Configuration for memory and storage settings.

    Defines how YOLO Developer persists memory, including vector embeddings
    and relationship graphs.
    """

    persist_path: str = Field(
        default=".yolo/memory",
        description="Directory path for persisting memory data",
    )
    vector_store_type: Literal["chromadb"] = Field(
        default="chromadb",
        description="Vector store backend type (chromadb supported)",
    )
    graph_store_type: Literal["json", "neo4j"] = Field(
        default="json",
        description="Graph store backend type (json for MVP, neo4j optional)",
    )


class BrownfieldConfig(BaseModel):
    """Configuration for brownfield project scanning."""

    scan_depth: int = Field(
        default=3,
        description="Directory depth to scan",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "node_modules",
            ".git",
            "__pycache__",
            ".venv",
            "venv",
        ],
        description="Patterns to exclude from scanning",
    )
    include_git_history: bool = Field(
        default=True,
        description="Analyze git history for context",
    )
    max_files_to_analyze: int = Field(
        default=1000,
        description="Maximum files to analyze for patterns",
    )
    interactive: bool = Field(
        default=True,
        description="Prompt user for ambiguous decisions",
    )


class AnalystGatheringConfig(BaseModel):
    """Configuration for interactive requirements gathering."""

    enabled: bool = Field(default=True, description="Enable requirements gathering sessions")
    storage_path: str = Field(
        default=".yolo/sessions", description="Storage path for gathering sessions"
    )
    max_questions_per_phase: int = Field(default=5, description="Maximum questions per phase")


class AnalystConfig(BaseModel):
    """Configuration for analyst features."""

    gathering: AnalystGatheringConfig = Field(default_factory=AnalystGatheringConfig)


class WebUploadConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable uploads")
    max_size_mb: int = Field(default=10, description="Max upload size in MB")
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".md", ".txt", ".pdf", ".docx"],
        description="Allowed upload extensions",
    )
    storage_path: str = Field(default=".yolo/uploads", description="Upload storage path")


CLIToolOutputFormat = Literal["json", "text"]


class CLIToolConfig(BaseModel):
    """Configuration for an external CLI tool.

    Defines settings for integrating with external CLI-based AI development tools
    like Claude Code, Aider, and similar utilities.

    Attributes:
        enabled: Whether this tool is enabled for use.
        path: Custom binary path (defaults to PATH lookup if not set).
        timeout: Maximum execution time in seconds.
        output_format: Expected output format from the tool.
        extra_args: Additional command-line arguments to pass.

    Example:
        >>> from yolo_developer.config.schema import CLIToolConfig
        >>> config = CLIToolConfig(enabled=True, timeout=600)
        >>> config.enabled
        True
    """

    enabled: bool = Field(
        default=False,
        description="Whether this tool is enabled for use",
    )
    path: str | None = Field(
        default=None,
        description="Custom binary path (defaults to PATH lookup if not set)",
    )
    timeout: int = Field(
        default=300,
        description="Maximum execution time in seconds",
        gt=0,
    )
    output_format: CLIToolOutputFormat = Field(
        default="json",
        description="Expected output format from the tool (json or text)",
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Additional command-line arguments to pass to the tool",
    )


class ToolsConfig(BaseModel):
    """Configuration for external CLI tool integrations.

    Defines settings for integrating YOLO Developer with external CLI-based
    AI development tools. Each tool can be individually enabled and configured.

    Attributes:
        claude_code: Configuration for Claude Code CLI integration.
        aider: Configuration for Aider CLI integration.

    Example:
        >>> from yolo_developer.config.schema import ToolsConfig, CLIToolConfig
        >>> config = ToolsConfig(
        ...     claude_code=CLIToolConfig(enabled=True, timeout=600)
        ... )
        >>> config.claude_code.enabled
        True

    Environment Variables:
        - YOLO_TOOLS__CLAUDE_CODE__ENABLED: Enable Claude Code integration
        - YOLO_TOOLS__CLAUDE_CODE__TIMEOUT: Timeout in seconds
        - YOLO_TOOLS__CLAUDE_CODE__PATH: Custom binary path
        - YOLO_TOOLS__AIDER__ENABLED: Enable Aider integration
    """

    claude_code: CLIToolConfig = Field(
        default_factory=CLIToolConfig,
        description="Configuration for Claude Code CLI integration",
    )
    aider: CLIToolConfig = Field(
        default_factory=CLIToolConfig,
        description="Configuration for Aider CLI integration",
    )


class WebConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable web UI")
    host: str = Field(default="127.0.0.1", description="Web host")
    port: int = Field(default=3000, description="Web port")
    api_only: bool = Field(default=False, description="Run API only")
    uploads: WebUploadConfig = Field(default_factory=WebUploadConfig)


class YoloConfig(BaseSettings):
    """Main configuration class for YOLO Developer.

    This is the root configuration that composes all nested configurations.
    Inherits from pydantic_settings.BaseSettings for environment variable support.

    Environment Variable Mapping:
        - YOLO_PROJECT_NAME: Project name (required)
        - YOLO_LLM__CHEAP_MODEL: Override cheap_model
        - YOLO_LLM__PREMIUM_MODEL: Override premium_model
        - YOLO_LLM__BEST_MODEL: Override best_model
        - YOLO_LLM__OPENAI__API_KEY: OpenAI API key (secrets via env only)
        - YOLO_LLM__OPENAI_API_KEY: OpenAI API key (legacy)
        - YOLO_LLM__ANTHROPIC_API_KEY: Anthropic API key (secrets via env only)
        - YOLO_QUALITY__TEST_COVERAGE_THRESHOLD: Override test coverage threshold
        - YOLO_QUALITY__CONFIDENCE_THRESHOLD: Override confidence threshold
        - YOLO_MEMORY__PERSIST_PATH: Override memory persistence path
        - YOLO_MEMORY__VECTOR_STORE_TYPE: Override vector store type
        - YOLO_MEMORY__GRAPH_STORE_TYPE: Override graph store type
        - YOLO_BROWNFIELD__SCAN_DEPTH: Override brownfield scan depth
        - YOLO_BROWNFIELD__MAX_FILES_TO_ANALYZE: Override scan file limit
        - YOLO_BROWNFIELD__INTERACTIVE: Override brownfield interactive mode
        - YOLO_ANALYST__GATHERING__ENABLED: Toggle gathering sessions
        - YOLO_ANALYST__GATHERING__STORAGE_PATH: Session storage path
        - YOLO_WEB__HOST: Web UI host
        - YOLO_WEB__PORT: Web UI port
        - YOLO_GITHUB__TOKEN: GitHub token (env only)
        - YOLO_GITHUB__REPOSITORY: GitHub repo slug (owner/repo)
        - YOLO_TOOLS__CLAUDE_CODE__ENABLED: Enable Claude Code integration
        - YOLO_TOOLS__CLAUDE_CODE__TIMEOUT: Claude Code timeout in seconds
        - YOLO_TOOLS__CLAUDE_CODE__PATH: Custom Claude Code binary path
        - YOLO_TOOLS__AIDER__ENABLED: Enable Aider integration

    Example:
        >>> config = YoloConfig(project_name="my-project")
        >>> config.llm.cheap_model
        'gpt-5.2-instant'
        >>> warnings = config.validate_api_keys()
        >>> if warnings:
        ...     print("Warning: No API keys configured")
    """

    model_config = SettingsConfigDict(
        env_prefix="YOLO_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    # Required fields (no default)
    project_name: str = Field(
        description="Name of the project being developed",
    )

    # Nested configuration models with defaults
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM provider configuration",
    )
    quality: QualityConfig = Field(
        default_factory=QualityConfig,
        description="Quality gate threshold configuration",
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Memory and storage configuration",
    )
    brownfield: BrownfieldConfig = Field(
        default_factory=BrownfieldConfig,
        description="Brownfield scanning configuration",
    )
    analyst: AnalystConfig = Field(
        default_factory=AnalystConfig,
        description="Analyst interactive gathering configuration",
    )
    web: WebConfig = Field(
        default_factory=WebConfig,
        description="Web UI configuration",
    )
    github: GitHubConfig = Field(
        default_factory=GitHubConfig,
        description="GitHub automation configuration",
    )
    tools: ToolsConfig = Field(
        default_factory=ToolsConfig,
        description="External CLI tool integrations configuration",
    )

    def validate_api_keys(self) -> list[str]:
        """Validate API key configuration and return warnings.

        Checks if any API keys are configured. If no API keys are set,
        returns a warning message. This is a warning, not an error,
        because API keys may be set later via SDK or at runtime.

        Returns:
            List of warning messages. Empty list if at least one API key is configured.

        Example:
            >>> config = YoloConfig(project_name="my-project")
            >>> warnings = config.validate_api_keys()
            >>> if warnings:
            ...     for warning in warnings:
            ...         print(f"Warning: {warning}")
        """
        warnings: list[str] = []

        if self.llm.openai.api_key is None and self.llm.anthropic_api_key is None:
            warnings.append(
                "No API keys configured. Set YOLO_LLM__OPENAI__API_KEY or "
                "YOLO_LLM__ANTHROPIC_API_KEY environment variable for LLM operations."
            )

        return warnings
