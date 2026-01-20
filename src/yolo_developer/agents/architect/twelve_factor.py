"""Twelve-Factor App analysis for Architect agent (Story 7.2).

This module provides 12-Factor App compliance analysis for user stories,
helping identify architectural patterns that follow or violate the
12-Factor methodology.

Key Functions:
    analyze_twelve_factor: Main entry point for 12-Factor analysis
    _analyze_factor: Analyze a single factor for a story

Example:
    >>> from yolo_developer.agents.architect.twelve_factor import analyze_twelve_factor
    >>>
    >>> story = {"title": "Add database", "description": "Store user data"}
    >>> analysis = await analyze_twelve_factor(story)
    >>> analysis.overall_compliance
    0.8

References:
    - https://12factor.net - The Twelve-Factor App methodology
    - FR49: Architect Agent can design system architecture following 12-Factor principles
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.architect.types import (
    TWELVE_FACTORS,
    FactorResult,
    TwelveFactorAnalysis,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# LLM Prompt Template
# =============================================================================

TWELVE_FACTOR_PROMPT = """Analyze the following user story for 12-Factor App compliance.

Story:
{story_content}

For each of the following factors, determine:
1. Does this factor apply to this story? (yes/no)
2. If applicable, is the story compliant? (yes/no/partial)
3. What specific finding led to this assessment?
4. What recommendation would improve compliance?

Factors to analyze: {factors_list}

Respond in JSON format only, with no additional text:
{{
  "factors": [
    {{
      "factor_name": "factor_name_here",
      "applies": true,
      "compliant": true,
      "finding": "description of what was found",
      "recommendation": "improvement suggestion or empty string if compliant"
    }}
  ]
}}
"""

# =============================================================================
# Pattern Definitions for Factor Analysis
# =============================================================================

# Factor III (Config) - Hardcoded configuration patterns
CONFIG_VIOLATION_PATTERNS = [
    # URLs with credentials or specific hosts
    r"(?:postgres(?:ql)?|mysql|redis|mongodb|http|https)://[^\s]+",
    # API keys and secrets
    r"(?:api[_-]?key|secret|password|token)\s*[=:]\s*['\"]?[\w\-]+['\"]?",
    # Hardcoded ports
    r"port\s*[=:]\s*\d{4,5}",
    # Hardcoded IPs
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
]

# Factor III (Config) - Compliant patterns
CONFIG_COMPLIANT_PATTERNS = [
    r"environment\s+variable",
    r"env\s+var",
    r"os\.environ",
    r"os\.getenv",
    r"\$\{?\w+\}?",  # Environment variable references
    r"config(?:uration)?\s+file",
]

# Factor VI (Processes) - Stateful patterns to detect
PROCESS_VIOLATION_PATTERNS = [
    r"session\s+(?:state|storage|memory|local)",
    r"session\s+in\s+(?:local\s+)?memory",
    r"store\s+.*session\s+in\s+(?:local\s+)?memory",
    r"sticky\s+session",
    r"in[_-]?memory\s+(?:cache|storage|state)",
    r"local\s+(?:file|disk|storage|memory)",
    r"/var/(?:uploads|data|storage)",
    r"shared\s+memory",
    r"global\s+(?:state|variable)",
]

# Factor VI (Processes) - Compliant patterns
PROCESS_COMPLIANT_PATTERNS = [
    r"stateless",
    r"(?:external|backing)\s+(?:storage|cache|service)",
    r"(?:redis|memcached)\s+(?:for\s+)?session",
    r"s3|blob\s+storage|object\s+storage",
]

# Factor IV (Backing Services) - Service references
BACKING_SERVICE_PATTERNS = [
    r"(?:postgres|postgresql|mysql|mariadb|mongodb|sqlite|database|db)\b",
    r"(?:redis|memcached|cache)\b",
    r"(?:rabbitmq|kafka|sqs|message\s+queue|amqp)\b",
    r"(?:smtp|email|sendgrid|mailgun)\b",
    r"(?:s3|blob|object\s+storage)\b",
    r"(?:elasticsearch|solr|search)\b",
]

# Connection string patterns (backing services with embedded config - violation)
CONNECTION_STRING_PATTERNS = [
    r"(?:postgres|mysql|redis|mongodb|amqp)://[^\s]+",
    r"(?:host|server)\s*[=:]\s*['\"]?[\w\.\-]+['\"]?",
]


# =============================================================================
# Factor Analyzers
# =============================================================================


def _analyze_codebase(story: dict[str, Any]) -> FactorResult:
    """Analyze Factor I (Codebase) - One codebase tracked in revision control."""
    text = _get_story_text(story).lower()

    # Generally compliant unless multi-repo patterns detected
    multi_repo_patterns = [
        r"multiple\s+repo",
        r"separate\s+codebase",
        r"copy\s+(?:code|source)",
    ]

    for pattern in multi_repo_patterns:
        if re.search(pattern, text):
            return FactorResult(
                factor_name="codebase",
                applies=True,
                compliant=False,
                finding="Multiple codebase or code copying pattern detected",
                recommendation="Maintain a single codebase with multiple deploys",
            )

    # Check for good patterns
    if any(kw in text for kw in ["git", "repository", "version control"]):
        return FactorResult(
            factor_name="codebase",
            applies=True,
            compliant=True,
            finding="Version control patterns mentioned",
            recommendation="",
        )

    return FactorResult(
        factor_name="codebase",
        applies=False,
        compliant=None,
        finding="No codebase-related patterns detected",
        recommendation="",
    )


def _analyze_dependencies(story: dict[str, Any]) -> FactorResult:
    """Analyze Factor II (Dependencies) - Explicitly declare and isolate dependencies."""
    text = _get_story_text(story).lower()

    # Check for dependency declaration patterns
    good_patterns = [
        r"requirements\.txt",
        r"pyproject\.toml",
        r"package\.json",
        r"gemfile",
        r"cargo\.toml",
        r"go\.mod",
        r"pip\s+install",
        r"npm\s+install",
        r"dependency\s+(?:declaration|management)",
    ]

    # Check for bad patterns
    bad_patterns = [
        r"system[_-]?wide\s+(?:install|package)",
        r"global\s+(?:install|package)",
        r"apt[_-]?get",
        r"brew\s+install",
    ]

    has_good = any(re.search(p, text) for p in good_patterns)
    has_bad = any(re.search(p, text) for p in bad_patterns)

    if has_bad:
        return FactorResult(
            factor_name="dependencies",
            applies=True,
            compliant=False,
            finding="System-wide or global dependency installation detected",
            recommendation="Use explicit dependency declaration (requirements.txt, pyproject.toml, etc.)",
        )

    if has_good:
        return FactorResult(
            factor_name="dependencies",
            applies=True,
            compliant=True,
            finding="Explicit dependency declaration pattern detected",
            recommendation="",
        )

    return FactorResult(
        factor_name="dependencies",
        applies=False,
        compliant=None,
        finding="No dependency-related patterns detected",
        recommendation="",
    )


def _analyze_config(story: dict[str, Any]) -> FactorResult:
    """Analyze Factor III (Config) - Store config in the environment."""
    text = _get_story_text(story)

    # Check for compliant patterns first
    for pattern in CONFIG_COMPLIANT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return FactorResult(
                factor_name="config",
                applies=True,
                compliant=True,
                finding="Environment variable or external configuration pattern detected",
                recommendation="",
            )

    # Check for violation patterns
    for pattern in CONFIG_VIOLATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return FactorResult(
                factor_name="config",
                applies=True,
                compliant=False,
                finding="Hardcoded configuration value detected (URL, API key, or credential)",
                recommendation="Use environment variables for all configuration that varies between deploys",
            )

    # Check for generic config keywords
    if any(kw in text.lower() for kw in ["config", "setting", "credential", "secret"]):
        return FactorResult(
            factor_name="config",
            applies=True,
            compliant=True,
            finding="Configuration mentioned without hardcoded values",
            recommendation="",
        )

    return FactorResult(
        factor_name="config",
        applies=False,
        compliant=None,
        finding="No configuration-related patterns detected",
        recommendation="",
    )


def _analyze_backing_services(story: dict[str, Any]) -> FactorResult:
    """Analyze Factor IV (Backing Services) - Treat backing services as attached resources."""
    text = _get_story_text(story)

    # Check if story involves backing services
    has_backing_service = any(re.search(p, text, re.IGNORECASE) for p in BACKING_SERVICE_PATTERNS)

    if not has_backing_service:
        return FactorResult(
            factor_name="backing_services",
            applies=False,
            compliant=None,
            finding="No backing service references detected",
            recommendation="",
        )

    # Check for connection string violations
    for pattern in CONNECTION_STRING_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return FactorResult(
                factor_name="backing_services",
                applies=True,
                compliant=False,
                finding="Hardcoded connection string or host detected",
                recommendation="Externalize connection strings as environment variables or config",
            )

    # Check for compliant patterns
    if any(kw in text.lower() for kw in ["environment variable", "env var", "config"]):
        return FactorResult(
            factor_name="backing_services",
            applies=True,
            compliant=True,
            finding="Backing service with externalized configuration",
            recommendation="",
        )

    return FactorResult(
        factor_name="backing_services",
        applies=True,
        compliant=True,
        finding="Backing service referenced as attached resource",
        recommendation="",
    )


def _analyze_build_release_run(story: dict[str, Any]) -> FactorResult:
    """Analyze Factor V (Build, Release, Run) - Strictly separate build and run stages."""
    text = _get_story_text(story).lower()

    # Check for build/release/run patterns
    build_patterns = [
        r"build\s+(?:stage|step|process)",
        r"(?:ci|cd|pipeline)",
        r"docker\s+build",
        r"compile",
    ]

    # Violation patterns
    bad_patterns = [
        r"modify\s+(?:code|source)\s+(?:in|at)\s+(?:runtime|production)",
        r"hot[_-]?patch",
    ]

    has_bad = any(re.search(p, text) for p in bad_patterns)
    has_good = any(re.search(p, text) for p in build_patterns)

    if has_bad:
        return FactorResult(
            factor_name="build_release_run",
            applies=True,
            compliant=False,
            finding="Runtime code modification pattern detected",
            recommendation="Strictly separate build and run stages; never modify code at runtime",
        )

    if has_good:
        return FactorResult(
            factor_name="build_release_run",
            applies=True,
            compliant=True,
            finding="Build/release/run separation patterns detected",
            recommendation="",
        )

    return FactorResult(
        factor_name="build_release_run",
        applies=False,
        compliant=None,
        finding="No build/release/run patterns detected",
        recommendation="",
    )


def _analyze_processes(story: dict[str, Any]) -> FactorResult:
    """Analyze Factor VI (Processes) - Execute the app as stateless processes."""
    text = _get_story_text(story)

    # Check for compliant patterns first
    for pattern in PROCESS_COMPLIANT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return FactorResult(
                factor_name="processes",
                applies=True,
                compliant=True,
                finding="Stateless process pattern detected",
                recommendation="",
            )

    # Check for violation patterns
    for pattern in PROCESS_VIOLATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return FactorResult(
                factor_name="processes",
                applies=True,
                compliant=False,
                finding="Stateful process pattern detected (session state, local storage, or shared memory)",
                recommendation="Use backing services (Redis, database) for any state that needs to persist",
            )

    # Check for generic state keywords
    if any(kw in text.lower() for kw in ["state", "session", "cache"]):
        return FactorResult(
            factor_name="processes",
            applies=True,
            compliant=True,
            finding="State mentioned without local storage patterns",
            recommendation="",
        )

    return FactorResult(
        factor_name="processes",
        applies=False,
        compliant=None,
        finding="No process state patterns detected",
        recommendation="",
    )


def _analyze_port_binding(story: dict[str, Any]) -> FactorResult:
    """Analyze Factor VII (Port Binding) - Export services via port binding."""
    text = _get_story_text(story).lower()

    # Check for port binding patterns
    patterns = [
        r"port\s+\d+",
        r"bind\s+(?:to\s+)?port",
        r"listen\s+(?:on\s+)?port",
        r"expose\s+(?:port|service)",
        r"http\s+server",
    ]

    has_pattern = any(re.search(p, text) for p in patterns)

    if has_pattern:
        return FactorResult(
            factor_name="port_binding",
            applies=True,
            compliant=True,
            finding="Service exposed via port binding",
            recommendation="",
        )

    return FactorResult(
        factor_name="port_binding",
        applies=False,
        compliant=None,
        finding="No port binding patterns detected",
        recommendation="",
    )


def _analyze_concurrency(story: dict[str, Any]) -> FactorResult:
    """Analyze Factor VIII (Concurrency) - Scale out via the process model."""
    text = _get_story_text(story).lower()

    # Check for scaling patterns
    good_patterns = [
        r"horizontal\s+scal",
        r"scale\s+out",
        r"multiple\s+(?:instance|worker|process)",
        r"load\s+balanc",
    ]

    bad_patterns = [
        r"single\s+(?:instance|server|process)",
        r"vertical\s+scal",
    ]

    has_good = any(re.search(p, text) for p in good_patterns)
    has_bad = any(re.search(p, text) for p in bad_patterns)

    if has_bad and not has_good:
        return FactorResult(
            factor_name="concurrency",
            applies=True,
            compliant=False,
            finding="Single instance or vertical scaling pattern detected",
            recommendation="Design for horizontal scaling via process model",
        )

    if has_good:
        return FactorResult(
            factor_name="concurrency",
            applies=True,
            compliant=True,
            finding="Horizontal scaling pattern detected",
            recommendation="",
        )

    return FactorResult(
        factor_name="concurrency",
        applies=False,
        compliant=None,
        finding="No concurrency patterns detected",
        recommendation="",
    )


def _analyze_disposability(story: dict[str, Any]) -> FactorResult:
    """Analyze Factor IX (Disposability) - Fast startup and graceful shutdown."""
    text = _get_story_text(story).lower()

    # Check for disposability patterns
    good_patterns = [
        r"graceful\s+shutdown",
        r"sigterm",
        r"fast\s+startup",
        r"quick\s+(?:start|boot)",
        r"health\s+check",
    ]

    bad_patterns = [
        r"long\s+(?:startup|initialization)",
        r"warmup\s+(?:time|period)",
    ]

    has_good = any(re.search(p, text) for p in good_patterns)
    has_bad = any(re.search(p, text) for p in bad_patterns)

    if has_bad and not has_good:
        return FactorResult(
            factor_name="disposability",
            applies=True,
            compliant=False,
            finding="Long startup or warmup pattern detected",
            recommendation="Optimize for fast startup and implement graceful shutdown",
        )

    if has_good:
        return FactorResult(
            factor_name="disposability",
            applies=True,
            compliant=True,
            finding="Disposability pattern detected (graceful shutdown/fast startup)",
            recommendation="",
        )

    return FactorResult(
        factor_name="disposability",
        applies=False,
        compliant=None,
        finding="No disposability patterns detected",
        recommendation="",
    )


def _analyze_dev_prod_parity(story: dict[str, Any]) -> FactorResult:
    """Analyze Factor X (Dev/Prod Parity) - Keep environments similar."""
    text = _get_story_text(story).lower()

    # Check for parity patterns
    good_patterns = [
        r"same\s+(?:database|service|environment)",
        r"docker\s+compose",
        r"containeriz",
        r"(?:dev|staging|prod)\s+parity",
    ]

    bad_patterns = [
        r"different\s+(?:database|service)\s+(?:in|for)\s+(?:dev|development)",
        r"sqlite\s+(?:in|for)\s+dev",
        r"mock\s+(?:in|for)\s+dev",
    ]

    has_good = any(re.search(p, text) for p in good_patterns)
    has_bad = any(re.search(p, text) for p in bad_patterns)

    if has_bad and not has_good:
        return FactorResult(
            factor_name="dev_prod_parity",
            applies=True,
            compliant=False,
            finding="Environment divergence pattern detected (different services in dev)",
            recommendation="Use the same backing services in development as in production",
        )

    if has_good:
        return FactorResult(
            factor_name="dev_prod_parity",
            applies=True,
            compliant=True,
            finding="Environment parity pattern detected",
            recommendation="",
        )

    return FactorResult(
        factor_name="dev_prod_parity",
        applies=False,
        compliant=None,
        finding="No environment parity patterns detected",
        recommendation="",
    )


def _analyze_logs(story: dict[str, Any]) -> FactorResult:
    """Analyze Factor XI (Logs) - Treat logs as event streams."""
    text = _get_story_text(story).lower()

    # Check for logging patterns
    good_patterns = [
        r"(?:stdout|stderr)",
        r"log\s+stream",
        r"structured\s+log",
        r"json\s+log",
    ]

    bad_patterns = [
        r"log\s+file",
        r"write\s+(?:to\s+)?log",
        r"/var/log",
        r"rotate\s+log",
    ]

    has_good = any(re.search(p, text) for p in good_patterns)
    has_bad = any(re.search(p, text) for p in bad_patterns)

    if has_bad and not has_good:
        return FactorResult(
            factor_name="logs",
            applies=True,
            compliant=False,
            finding="File-based logging pattern detected",
            recommendation="Write logs to stdout/stderr as event streams",
        )

    if has_good:
        return FactorResult(
            factor_name="logs",
            applies=True,
            compliant=True,
            finding="Stream-based logging pattern detected",
            recommendation="",
        )

    if "log" in text:
        return FactorResult(
            factor_name="logs",
            applies=True,
            compliant=True,
            finding="Logging mentioned without file-based patterns",
            recommendation="",
        )

    return FactorResult(
        factor_name="logs",
        applies=False,
        compliant=None,
        finding="No logging patterns detected",
        recommendation="",
    )


def _analyze_admin_processes(story: dict[str, Any]) -> FactorResult:
    """Analyze Factor XII (Admin Processes) - Run admin tasks as one-off processes."""
    text = _get_story_text(story).lower()

    # Check for admin process patterns
    good_patterns = [
        r"one[_-]?off\s+(?:task|command|process)",
        r"(?:database\s+)?migration",
        r"(?:cli|command[_-]?line)\s+(?:tool|command)",
        r"management\s+command",
    ]

    has_good = any(re.search(p, text) for p in good_patterns)

    if has_good:
        return FactorResult(
            factor_name="admin_processes",
            applies=True,
            compliant=True,
            finding="Admin process pattern detected (one-off task, migration, CLI command)",
            recommendation="",
        )

    return FactorResult(
        factor_name="admin_processes",
        applies=False,
        compliant=None,
        finding="No admin process patterns detected",
        recommendation="",
    )


# =============================================================================
# Factor Analyzer Dispatch
# =============================================================================

FACTOR_ANALYZERS: dict[str, Callable[[dict[str, Any]], FactorResult]] = {
    "codebase": _analyze_codebase,
    "dependencies": _analyze_dependencies,
    "config": _analyze_config,
    "backing_services": _analyze_backing_services,
    "build_release_run": _analyze_build_release_run,
    "processes": _analyze_processes,
    "port_binding": _analyze_port_binding,
    "concurrency": _analyze_concurrency,
    "disposability": _analyze_disposability,
    "dev_prod_parity": _analyze_dev_prod_parity,
    "logs": _analyze_logs,
    "admin_processes": _analyze_admin_processes,
}


def _get_story_text(story: dict[str, Any]) -> str:
    """Extract searchable text from story dict."""
    parts = [
        str(story.get("title", "")),
        str(story.get("description", "")),
        str(story.get("content", "")),
        str(story.get("acceptance_criteria", "")),
    ]
    return " ".join(parts)


def _analyze_factor(story: dict[str, Any], factor_name: str) -> FactorResult:
    """Analyze a single factor for a story.

    Args:
        story: Story dictionary with title, description, etc.
        factor_name: Name of the factor to analyze (from TWELVE_FACTORS).

    Returns:
        FactorResult with analysis outcome.
    """
    analyzer = FACTOR_ANALYZERS.get(factor_name)

    if analyzer is None:
        logger.debug("unknown_factor_requested", factor_name=factor_name)
        return FactorResult(
            factor_name=factor_name,
            applies=False,
            compliant=None,
            finding=f"Unknown factor: {factor_name}",
            recommendation="",
        )

    return analyzer(story)


# =============================================================================
# Main Analysis Function
# =============================================================================


async def analyze_twelve_factor(story: dict[str, Any]) -> TwelveFactorAnalysis:
    """Analyze a story for 12-Factor App compliance.

    Evaluates the story against all 12 factors from the 12-Factor App
    methodology, returning compliance status and recommendations.

    Args:
        story: Story dictionary with keys like 'title', 'description', 'id'.

    Returns:
        TwelveFactorAnalysis with results for all factors.

    Example:
        >>> story = {"title": "Add database", "description": "PostgreSQL setup"}
        >>> analysis = await analyze_twelve_factor(story)
        >>> analysis.overall_compliance
        0.75
    """
    story_id = story.get("id", "unknown")

    logger.info(
        "twelve_factor_analysis_start",
        story_id=story_id,
        story_title=story.get("title", ""),
    )

    # Analyze all 12 factors
    factor_results: dict[str, FactorResult] = {}
    for factor_name in TWELVE_FACTORS:
        result = _analyze_factor(story, factor_name)
        factor_results[factor_name] = result

    # Determine applicable factors
    applicable_factors = tuple(name for name, result in factor_results.items() if result.applies)

    # Calculate overall compliance
    if applicable_factors:
        compliant_count = sum(
            1 for name in applicable_factors if factor_results[name].compliant is True
        )
        overall_compliance = compliant_count / len(applicable_factors)
    else:
        overall_compliance = 1.0  # No applicable factors = fully compliant

    # Aggregate recommendations
    recommendations = tuple(
        result.recommendation
        for result in factor_results.values()
        if result.recommendation and not result.compliant
    )

    analysis = TwelveFactorAnalysis(
        factor_results=factor_results,
        applicable_factors=applicable_factors,
        overall_compliance=overall_compliance,
        recommendations=recommendations,
    )

    logger.info(
        "twelve_factor_analysis_complete",
        story_id=story_id,
        applicable_factors=len(applicable_factors),
        overall_compliance=overall_compliance,
        recommendation_count=len(recommendations),
    )

    return analysis


# =============================================================================
# LLM-Powered Analysis Functions
# =============================================================================


def _parse_llm_response(response: dict[str, Any]) -> dict[str, FactorResult]:
    """Parse LLM response into FactorResult dict.

    Args:
        response: Parsed JSON response from LLM with 'factors' list.

    Returns:
        Dict mapping factor names to FactorResult objects.
    """
    results: dict[str, FactorResult] = {}

    factors = response.get("factors", [])
    for factor in factors:
        factor_name = factor.get("factor_name", "")
        if not factor_name:
            continue

        results[factor_name] = FactorResult(
            factor_name=factor_name,
            applies=factor.get("applies", False),
            compliant=factor.get("compliant"),
            finding=factor.get("finding", ""),
            recommendation=factor.get("recommendation", ""),
        )

    return results


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _call_llm(prompt: str) -> dict[str, Any]:
    """Call LLM with retry logic.

    Uses litellm for provider-agnostic LLM calls with automatic retries
    on transient failures like rate limits.

    Args:
        prompt: The prompt to send to the LLM.

    Returns:
        Parsed JSON response from the LLM.

    Raises:
        Exception: On persistent failures after retries.
    """
    import litellm

    logger.debug("llm_call_start", prompt_length=len(prompt))

    # TODO: Use LLMRouter per ADR-003 when available
    # For now, use configurable model via environment variable
    import os

    model = os.environ.get("YOLO_LLM__ROUTINE_MODEL", "gpt-4o-mini")
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    if content is None:
        logger.warning("llm_returned_empty_content")
        return {"factors": []}

    try:
        result = json.loads(content)
        logger.debug("llm_call_complete", factor_count=len(result.get("factors", [])))
        return result  # type: ignore[no-any-return]
    except json.JSONDecodeError as e:
        logger.error("llm_json_parse_error", error=str(e), content=content[:200])
        return {"factors": []}


async def _analyze_with_llm(
    story: dict[str, Any],
    factors: list[str],
) -> dict[str, FactorResult]:
    """Analyze story using LLM for specified factors.

    Uses LLM to analyze complex stories that require deeper understanding
    than pattern matching can provide.

    Args:
        story: Story dictionary with title, description, etc.
        factors: List of factor names to analyze.

    Returns:
        Dict mapping factor names to FactorResult objects.
    """
    story_content = _get_story_text(story)
    factors_list = ", ".join(factors)

    prompt = TWELVE_FACTOR_PROMPT.format(
        story_content=story_content,
        factors_list=factors_list,
    )

    logger.info(
        "llm_analysis_start",
        story_id=story.get("id", "unknown"),
        factor_count=len(factors),
    )

    response = await _call_llm(prompt)
    results = _parse_llm_response(response)

    logger.info(
        "llm_analysis_complete",
        story_id=story.get("id", "unknown"),
        results_count=len(results),
    )

    return results


async def analyze_twelve_factor_with_llm(
    story: dict[str, Any],
) -> TwelveFactorAnalysis:
    """Analyze story using LLM for all 12 factors.

    This function uses LLM for deeper analysis of complex stories,
    complementing the pattern-based analysis.

    Args:
        story: Story dictionary with title, description, etc.

    Returns:
        TwelveFactorAnalysis with LLM-powered results.
    """
    story_id = story.get("id", "unknown")

    logger.info(
        "twelve_factor_llm_analysis_start",
        story_id=story_id,
        story_title=story.get("title", ""),
    )

    # Use LLM to analyze all factors
    llm_results = await _analyze_with_llm(story, list(TWELVE_FACTORS))

    # Fill in any missing factors with defaults
    factor_results: dict[str, FactorResult] = {}
    for factor_name in TWELVE_FACTORS:
        if factor_name in llm_results:
            factor_results[factor_name] = llm_results[factor_name]
        else:
            factor_results[factor_name] = FactorResult(
                factor_name=factor_name,
                applies=False,
                compliant=None,
                finding="Not analyzed by LLM",
                recommendation="",
            )

    # Determine applicable factors
    applicable_factors = tuple(name for name, result in factor_results.items() if result.applies)

    # Calculate overall compliance
    if applicable_factors:
        compliant_count = sum(
            1 for name in applicable_factors if factor_results[name].compliant is True
        )
        overall_compliance = compliant_count / len(applicable_factors)
    else:
        overall_compliance = 1.0

    # Aggregate recommendations
    recommendations = tuple(
        result.recommendation
        for result in factor_results.values()
        if result.recommendation and not result.compliant
    )

    analysis = TwelveFactorAnalysis(
        factor_results=factor_results,
        applicable_factors=applicable_factors,
        overall_compliance=overall_compliance,
        recommendations=recommendations,
    )

    logger.info(
        "twelve_factor_llm_analysis_complete",
        story_id=story_id,
        applicable_factors=len(applicable_factors),
        overall_compliance=overall_compliance,
    )

    return analysis
