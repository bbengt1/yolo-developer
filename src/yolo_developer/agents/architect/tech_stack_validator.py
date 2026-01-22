"""Tech stack constraint validation for architect designs (Story 7.6).

This module provides tech stack constraint validation functionality:

- Extract configured tech stack from YoloConfig
- Validate design decisions against configured technologies
- Check version compatibility
- Suggest stack-specific patterns
- LLM-powered analysis with pattern-based fallback

Example:
    >>> from yolo_developer.agents.architect.tech_stack_validator import (
    ...     validate_tech_stack_constraints,
    ... )
    >>>
    >>> validation = await validate_tech_stack_constraints(design_decisions)
    >>> validation.overall_compliance
    True

Architecture:
    - Uses frozen dataclasses per ADR-001
    - Uses litellm for LLM calls per ADR-003
    - Uses tenacity for retry logic per ADR-007
    - All I/O operations are async per ARCH-QUALITY-5
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.architect.types import (
    ConstraintViolation,
    DesignDecision,
    RiskSeverity,
    StackPattern,
    TechStackValidation,
)
from yolo_developer.config import LLM_CHEAP_MODEL_DEFAULT, YoloConfig, load_config

logger = structlog.get_logger(__name__)


# =============================================================================
# Stack-Specific Pattern Templates
# =============================================================================

STACK_PATTERN_TEMPLATES: dict[str, StackPattern] = {
    "pytest": StackPattern(
        pattern_name="pytest-fixtures",
        description="Use pytest fixtures for test setup/teardown",
        rationale="pytest is configured as test framework; fixtures provide clean test isolation",
        applicable_technologies=("pytest", "pytest-asyncio"),
    ),
    "pytest-asyncio": StackPattern(
        pattern_name="async-pytest",
        description="Use pytest-asyncio for async test functions",
        rationale="Project uses async/await patterns throughout; pytest-asyncio provides async test support",
        applicable_technologies=("pytest", "pytest-asyncio"),
    ),
    "uv": StackPattern(
        pattern_name="uv-dependency-management",
        description="Use uv for fast dependency installation and lockfile management",
        rationale="uv is configured as package manager; provides reproducible builds and faster installs",
        applicable_technologies=("uv",),
    ),
    "ruff": StackPattern(
        pattern_name="ruff-linting",
        description="Use ruff for linting and formatting",
        rationale="ruff is configured as linter; provides fast, comprehensive Python code quality checks",
        applicable_technologies=("ruff",),
    ),
    "mypy": StackPattern(
        pattern_name="mypy-type-checking",
        description="Use mypy for static type checking with strict mode",
        rationale="mypy is configured for type checking; ensures type safety across the codebase",
        applicable_technologies=("mypy",),
    ),
    "chromadb": StackPattern(
        pattern_name="chromadb-vector-storage",
        description="Use ChromaDB for vector embeddings and similarity search",
        rationale="ChromaDB is configured as vector store; provides persistent embedding storage",
        applicable_technologies=("chromadb",),
    ),
    "pydantic": StackPattern(
        pattern_name="pydantic-validation",
        description="Use Pydantic models for input validation at system boundaries",
        rationale="Pydantic is used for configuration; provides runtime type validation",
        applicable_technologies=("pydantic", "pydantic-settings"),
    ),
    "structlog": StackPattern(
        pattern_name="structlog-logging",
        description="Use structlog for structured logging with context",
        rationale="structlog is configured for logging; provides structured, contextual log output",
        applicable_technologies=("structlog",),
    ),
    "litellm": StackPattern(
        pattern_name="litellm-abstraction",
        description="Use litellm for LLM provider abstraction",
        rationale="litellm is configured per ADR-003; provides unified interface to multiple LLM providers",
        applicable_technologies=("litellm",),
    ),
    "tenacity": StackPattern(
        pattern_name="tenacity-retry",
        description="Use tenacity for retry logic with exponential backoff",
        rationale="tenacity is configured per ADR-007; provides robust error handling for I/O operations",
        applicable_technologies=("tenacity",),
    ),
    "langgraph": StackPattern(
        pattern_name="langgraph-workflow",
        description="Use LangGraph for agent orchestration workflows",
        rationale="LangGraph is the core orchestration framework; provides stateful agent workflows",
        applicable_technologies=("langgraph",),
    ),
}


# =============================================================================
# Known Technology Categories
# =============================================================================

KNOWN_TECHNOLOGIES: dict[str, str] = {
    # Runtime
    "python": "runtime",
    "python3": "runtime",
    "node": "runtime",
    "nodejs": "runtime",
    # Frameworks
    "langgraph": "framework",
    "fastapi": "framework",
    "django": "framework",
    "flask": "framework",
    "typer": "framework",
    # Databases
    "chromadb": "database",
    "postgresql": "database",
    "postgres": "database",
    "sqlite": "database",
    "neo4j": "database",
    "redis": "database",
    # Testing
    "pytest": "testing",
    "pytest-asyncio": "testing",
    "unittest": "testing",
    # Tooling
    "uv": "tooling",
    "ruff": "tooling",
    "mypy": "tooling",
    "pip": "tooling",
    "poetry": "tooling",
    # LLM
    "litellm": "framework",
    "openai": "framework",
    "anthropic": "framework",
}


# =============================================================================
# Config Tech Stack Extraction (Task 2)
# =============================================================================


def _extract_tech_stack_from_config(config: YoloConfig | None = None) -> dict[str, Any]:
    """Extract tech stack information from YoloConfig.

    Extracts configured technologies from the config, including:
    - LLM models and providers
    - Memory store types
    - Inferred technologies from project structure

    Args:
        config: Optional YoloConfig. If not provided, loads from default location.

    Returns:
        Dictionary with tech stack information.
    """
    if config is None:
        config = load_config()

    logger.debug("extracting_tech_stack", project_name=config.project_name)

    # TODO: MVP implementation uses hard-coded values for most tech stack items.
    # Future enhancement should infer from:
    # 1. pyproject.toml - dependencies and Python version
    # 2. Project structure - detected patterns
    # Currently only LLM config and memory store types come from actual config.
    tech_stack: dict[str, Any] = {
        # Runtime (hard-coded for MVP - should read from pyproject.toml)
        "runtime": "Python 3.10+",
        # LLM Configuration
        "llm_models": [
            config.llm.cheap_model,
            config.llm.premium_model,
            config.llm.best_model,
        ],
        # Memory/Storage
        "vector_store": config.memory.vector_store_type,
        "graph_store": config.memory.graph_store_type,
        # Inferred from project dependencies (known stack)
        "testing": "pytest",
        "tooling": ["uv", "ruff", "mypy"],
        "frameworks": ["langgraph", "pydantic", "typer"],
        "libraries": ["litellm", "structlog", "tenacity"],
    }

    logger.debug(
        "tech_stack_extracted",
        llm_models=tech_stack["llm_models"],
        vector_store=tech_stack["vector_store"],
    )

    return tech_stack


# =============================================================================
# Technology Validation (Task 3)
# =============================================================================


def _extract_technologies_from_decision(decision: DesignDecision) -> list[str]:
    """Extract technology names mentioned in a design decision.

    Uses word boundary matching to avoid false positives (e.g., "node"
    shouldn't match "architect_node").

    Args:
        decision: A design decision to analyze.

    Returns:
        List of technology names found in the decision.
    """
    text = f"{decision.description} {decision.rationale}".lower()
    found: list[str] = []

    for tech in KNOWN_TECHNOLOGIES:
        # Use word boundary matching to avoid false positives
        # e.g., "node" shouldn't match "architect_node" or "node.py"
        if re.search(rf"\b{re.escape(tech)}\b", text):
            found.append(tech)

    return found


def _validate_technology_choices(
    decisions: list[DesignDecision],
    tech_stack: dict[str, Any],
) -> list[ConstraintViolation]:
    """Validate technology choices against configured stack.

    Checks each design decision for technologies that are not in the
    configured tech stack.

    Args:
        decisions: List of design decisions to validate.
        tech_stack: Configured tech stack dictionary.

    Returns:
        List of constraint violations found.
    """
    violations: list[ConstraintViolation] = []

    # Build set of allowed technologies
    allowed: set[str] = set()

    # Add from various config fields
    if "testing" in tech_stack:
        allowed.add(tech_stack["testing"].lower())
    if "database" in tech_stack:
        allowed.add(tech_stack["database"].lower())
    if "vector_store" in tech_stack:
        allowed.add(tech_stack["vector_store"].lower())
    if "graph_store" in tech_stack:
        allowed.add(tech_stack["graph_store"].lower())
    if "tooling" in tech_stack:
        for tool in tech_stack["tooling"]:
            allowed.add(tool.lower())
    if "frameworks" in tech_stack:
        for fw in tech_stack["frameworks"]:
            allowed.add(fw.lower())
    if "libraries" in tech_stack:
        for lib in tech_stack["libraries"]:
            allowed.add(lib.lower())

    # Add common allowed technologies
    allowed.update(["python", "python3", "pytest-asyncio", "pydantic-settings"])

    logger.debug("allowed_technologies", technologies=list(allowed))

    for decision in decisions:
        found_techs = _extract_technologies_from_decision(decision)

        for tech in found_techs:
            if tech not in allowed:
                # Determine severity based on category
                category = KNOWN_TECHNOLOGIES.get(tech, "unknown")
                severity: RiskSeverity = "critical" if category == "database" else "high"

                # Generate alternative suggestion
                alternative = _suggest_alternative(tech, tech_stack)

                violations.append(
                    ConstraintViolation(
                        technology=tech.title(),
                        expected_version=None,
                        actual_version="detected",
                        severity=severity,
                        suggested_alternative=alternative,
                    )
                )

    logger.debug(
        "technology_validation_complete",
        decisions_checked=len(decisions),
        violations_found=len(violations),
    )

    return violations


def _suggest_alternative(technology: str, tech_stack: dict[str, Any]) -> str:
    """Suggest an alternative technology from the configured stack.

    Args:
        technology: The unconfigured technology.
        tech_stack: The configured tech stack.

    Returns:
        Suggestion string for an alternative.
    """
    category = KNOWN_TECHNOLOGIES.get(technology.lower(), "unknown")

    suggestions: dict[str, str] = {
        "database": f"Use {tech_stack.get('vector_store', 'configured database')} as configured",
        "testing": f"Use {tech_stack.get('testing', 'pytest')} as configured",
        "tooling": "Use configured tooling (uv, ruff, mypy)",
        "runtime": "Use Python 3.10+ as configured",
    }

    return suggestions.get(category, f"Verify {technology} is needed or use configured alternative")


# =============================================================================
# Version Compatibility Check (Task 4)
# =============================================================================


def _parse_version(version: str) -> tuple[int, int, int]:
    """Parse a version string into major, minor, patch components.

    Args:
        version: Version string (e.g., "3.10", "3.10.2", "3.10+", "v3.10").

    Returns:
        Tuple of (major, minor, patch) integers.

    Note:
        TODO: MVP implementation handles simple version strings only.
        Future enhancement should support:
        - Range constraints: ">=3.10,<4.0"
        - Tilde constraints: "~3.10"
        - Caret constraints: "^3.10"
        - Wildcards: "3.*"
    """
    # Remove common prefixes (v, V) and trailing suffixes (+)
    version = version.lstrip("vV").rstrip("+").strip()

    parts = version.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0

    return (major, minor, patch)


def _check_version_compatibility(
    technology: str,
    expected_version: str,
    actual_version: str,
) -> ConstraintViolation | None:
    """Check version compatibility between expected and actual versions.

    Args:
        technology: Name of the technology.
        expected_version: Expected/configured version.
        actual_version: Actual version found in design.

    Returns:
        ConstraintViolation if incompatible, None if compatible.
    """
    if expected_version == actual_version:
        return None

    expected = _parse_version(expected_version)
    actual = _parse_version(actual_version)

    # Major version mismatch is critical
    if expected[0] != actual[0]:
        return ConstraintViolation(
            technology=technology,
            expected_version=expected_version,
            actual_version=actual_version,
            severity="critical",
            suggested_alternative=f"Upgrade to {technology} {expected_version}",
        )

    # Minor version mismatch is medium/high
    if expected[1] != actual[1]:
        # If actual is older than expected, it's higher severity
        if actual[1] < expected[1]:
            return ConstraintViolation(
                technology=technology,
                expected_version=expected_version,
                actual_version=actual_version,
                severity="high",
                suggested_alternative=f"Upgrade to {technology} {expected_version}",
            )
        # If actual is newer, it's just a note
        return ConstraintViolation(
            technology=technology,
            expected_version=expected_version,
            actual_version=actual_version,
            severity="medium",
            suggested_alternative=f"Consider standardizing on {technology} {expected_version}",
        )

    # Patch version differences are informational
    return None


# =============================================================================
# Stack-Specific Pattern Suggestion (Task 5)
# =============================================================================


def _suggest_stack_patterns(
    tech_stack: dict[str, Any],
    decisions: list[DesignDecision],
) -> list[StackPattern]:
    """Suggest stack-specific patterns based on configured technologies.

    Args:
        tech_stack: Configured tech stack dictionary.
        decisions: Design decisions for additional context.

    Returns:
        List of applicable stack patterns.
    """
    if not tech_stack:
        return []

    patterns: list[StackPattern] = []
    seen_patterns: set[str] = set()

    # Check testing framework
    testing = tech_stack.get("testing", "").lower()
    if "pytest" in testing and "pytest" not in seen_patterns:
        patterns.append(STACK_PATTERN_TEMPLATES["pytest"])
        patterns.append(STACK_PATTERN_TEMPLATES["pytest-asyncio"])
        seen_patterns.add("pytest")
        seen_patterns.add("pytest-asyncio")

    # Check tooling
    tooling = tech_stack.get("tooling", [])
    for tool in tooling:
        tool_lower = tool.lower()
        if tool_lower in STACK_PATTERN_TEMPLATES and tool_lower not in seen_patterns:
            patterns.append(STACK_PATTERN_TEMPLATES[tool_lower])
            seen_patterns.add(tool_lower)

    # Check vector store
    vector_store = tech_stack.get("vector_store", "").lower()
    if vector_store in STACK_PATTERN_TEMPLATES and vector_store not in seen_patterns:
        patterns.append(STACK_PATTERN_TEMPLATES[vector_store])
        seen_patterns.add(vector_store)

    # Check frameworks
    frameworks = tech_stack.get("frameworks", [])
    for fw in frameworks:
        fw_lower = fw.lower()
        if fw_lower in STACK_PATTERN_TEMPLATES and fw_lower not in seen_patterns:
            patterns.append(STACK_PATTERN_TEMPLATES[fw_lower])
            seen_patterns.add(fw_lower)

    # Check libraries
    libraries = tech_stack.get("libraries", [])
    for lib in libraries:
        lib_lower = lib.lower()
        if lib_lower in STACK_PATTERN_TEMPLATES and lib_lower not in seen_patterns:
            patterns.append(STACK_PATTERN_TEMPLATES[lib_lower])
            seen_patterns.add(lib_lower)

    logger.debug(
        "stack_patterns_suggested",
        pattern_count=len(patterns),
        pattern_names=[p.pattern_name for p in patterns],
    )

    return patterns


# =============================================================================
# LLM-Powered Analysis (Task 6)
# =============================================================================

TECH_STACK_VALIDATION_PROMPT = """Analyze these design decisions against the configured tech stack.

Configured Tech Stack:
{tech_stack}

Design Decisions:
{design_decisions}

Check for:
1. Technologies not in the configured stack
2. Version incompatibilities
3. Stack-specific patterns that should be applied
4. Best practices for this stack combination

For each issue found, provide:
- Technology: What technology is problematic?
- Expected: What was configured?
- Actual: What was used in the design?
- Severity: critical, high, medium, or low
- Suggested Alternative: How to fix?

Also suggest stack-specific patterns with rationale.

Respond in JSON format:
{{
  "overall_compliance": true/false,
  "violations": [
    {{
      "technology": "SQLite",
      "expected_version": null,
      "actual_version": "3.x",
      "severity": "critical",
      "suggested_alternative": "Use ChromaDB as configured for vector storage"
    }}
  ],
  "suggested_patterns": [
    {{
      "pattern_name": "async-pytest",
      "description": "Use pytest-asyncio for async test functions",
      "rationale": "Project uses async/await patterns throughout",
      "applicable_technologies": ["pytest", "pytest-asyncio"]
    }}
  ],
  "summary": "Brief validation summary"
}}
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _call_tech_stack_llm(prompt: str) -> str:
    """Call LLM for tech stack validation with retry logic.

    Args:
        prompt: The prompt to send to the LLM.

    Returns:
        LLM response text.

    Raises:
        Exception: If all retries fail.
    """
    import litellm

    # Note: Uses env var directly for model selection, consistent with other
    # architect modules (risk_identifier.py, adr_generator.py). This allows
    # runtime override without config reload. Falls back to routine-tier default.
    model = os.environ.get("YOLO_LLM__ROUTINE_MODEL", LLM_CHEAP_MODEL_DEFAULT)

    logger.debug("calling_tech_stack_llm", model=model, prompt_length=len(prompt))

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty response")

    return str(content)


async def _analyze_tech_stack_with_llm(
    tech_stack: dict[str, Any],
    decisions: list[DesignDecision],
) -> TechStackValidation | None:
    """Analyze tech stack constraints using LLM.

    Args:
        tech_stack: Configured tech stack dictionary.
        decisions: List of design decisions to validate.

    Returns:
        TechStackValidation if successful, None if LLM fails.
    """
    # Build prompt
    tech_stack_text = json.dumps(tech_stack, indent=2)
    decisions_text = "\n".join(
        f"- {d.decision_type}: {d.description} (Rationale: {d.rationale})" for d in decisions
    )

    prompt = TECH_STACK_VALIDATION_PROMPT.format(
        tech_stack=tech_stack_text,
        design_decisions=decisions_text or "No design decisions provided",
    )

    try:
        response_text = await _call_tech_stack_llm(prompt)

        # Parse JSON response - use regex for robust extraction from code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if json_match:
            response_text = json_match.group(1)

        data = json.loads(response_text.strip())

        # Extract violations
        violations_data = data.get("violations", [])
        violations: list[ConstraintViolation] = []

        for v in violations_data:
            severity = v.get("severity", "medium")
            if severity not in ("critical", "high", "medium", "low"):
                severity = "medium"

            validated_severity: RiskSeverity = severity

            violations.append(
                ConstraintViolation(
                    technology=v.get("technology", "Unknown"),
                    expected_version=v.get("expected_version"),
                    actual_version=v.get("actual_version", "detected"),
                    severity=validated_severity,
                    suggested_alternative=v.get("suggested_alternative", "Review configuration"),
                )
            )

        # Extract patterns
        patterns_data = data.get("suggested_patterns", [])
        patterns: list[StackPattern] = []

        for p in patterns_data:
            applicable = p.get("applicable_technologies", [])
            patterns.append(
                StackPattern(
                    pattern_name=p.get("pattern_name", "unknown"),
                    description=p.get("description", ""),
                    rationale=p.get("rationale", ""),
                    applicable_technologies=tuple(applicable),
                )
            )

        overall_compliance = data.get("overall_compliance", True)
        summary = data.get("summary", "LLM-generated validation")

        return TechStackValidation(
            overall_compliance=overall_compliance,
            violations=tuple(violations),
            suggested_patterns=tuple(patterns),
            summary=summary,
        )

    except json.JSONDecodeError as e:
        logger.warning("llm_response_json_parse_error", error=str(e))
        return None
    except Exception as e:
        logger.warning("llm_tech_stack_analysis_failed", error=str(e))
        return None


# =============================================================================
# Main Validation Function (Task 7)
# =============================================================================


async def validate_tech_stack_constraints(
    decisions: list[DesignDecision],
    config: YoloConfig | None = None,
    use_llm: bool = True,
) -> TechStackValidation:
    """Validate design decisions against configured tech stack.

    Main entry point for tech stack constraint validation. Combines
    rule-based validation with optional LLM analysis.

    Args:
        decisions: List of design decisions to validate.
        config: Optional YoloConfig. If not provided, loads from default location.
        use_llm: Whether to attempt LLM-powered analysis (default True).

    Returns:
        TechStackValidation with compliance status, violations, and patterns.
    """
    logger.info(
        "tech_stack_validation_start",
        decision_count=len(decisions),
        use_llm=use_llm,
    )

    # Extract tech stack from config
    tech_stack = _extract_tech_stack_from_config(config)

    # Try LLM analysis first if enabled
    if use_llm:
        llm_result = await _analyze_tech_stack_with_llm(tech_stack, decisions)
        if llm_result is not None:
            logger.info(
                "tech_stack_validation_complete",
                method="llm",
                overall_compliance=llm_result.overall_compliance,
                violation_count=len(llm_result.violations),
                pattern_count=len(llm_result.suggested_patterns),
            )
            return llm_result

    # Fall back to rule-based validation
    violations = _validate_technology_choices(decisions, tech_stack)
    patterns = _suggest_stack_patterns(tech_stack, decisions)

    # Determine compliance - False if any critical violations
    overall_compliance = not any(v.severity == "critical" for v in violations)

    # Generate summary
    if not violations:
        summary = "All design decisions comply with configured tech stack"
    else:
        critical_count = sum(1 for v in violations if v.severity == "critical")
        high_count = sum(1 for v in violations if v.severity == "high")
        summary_parts = [f"{len(violations)} constraint violations found"]
        if critical_count > 0:
            summary_parts.append(f"{critical_count} critical")
        if high_count > 0:
            summary_parts.append(f"{high_count} high-severity")
        summary = "; ".join(summary_parts)

    result = TechStackValidation(
        overall_compliance=overall_compliance,
        violations=tuple(violations),
        suggested_patterns=tuple(patterns),
        summary=summary,
    )

    logger.info(
        "tech_stack_validation_complete",
        method="rule-based",
        overall_compliance=result.overall_compliance,
        violation_count=len(result.violations),
        pattern_count=len(result.suggested_patterns),
    )

    return result
