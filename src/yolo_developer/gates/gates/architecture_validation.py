"""Architecture validation gate evaluator.

This module implements the architecture validation quality gate that validates
architectural decisions against:
- 12-Factor principles
- Tech stack constraints
- Security anti-patterns

The gate calculates a compliance score (0-100) and fails if below threshold.

Example:
    >>> from yolo_developer.gates.gates.architecture_validation import architecture_validation_evaluator
    >>> from yolo_developer.gates.types import GateContext
    >>>
    >>> state = {"architecture": {"twelve_factor": {"config": True, ...}}}
    >>> context = GateContext(state=state, gate_name="architecture_validation")
    >>> result = await architecture_validation_evaluator(context)
    >>> result.passed
    True

Security Note:
    This gate performs validation only and does not modify state.
    Architecture content is processed as untrusted input.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from yolo_developer.gates.evaluators import register_evaluator
from yolo_developer.gates.report_generator import format_report_text, generate_failure_report
from yolo_developer.gates.report_types import GateIssue, Severity
from yolo_developer.gates.threshold_resolver import resolve_threshold
from yolo_developer.gates.types import GateContext, GateResult

logger = structlog.get_logger(__name__)

# Default compliance threshold (70% = 0.70 in decimal format)
DEFAULT_COMPLIANCE_THRESHOLD = 0.70

# Severity weights for compliance score calculation
SEVERITY_WEIGHTS: dict[str, int] = {
    "critical": 25,
    "high": 15,
    "medium": 5,
    "low": 1,
}

# 12-Factor principles with descriptions and remediation guidance
TWELVE_FACTOR_PRINCIPLES: dict[str, dict[str, str]] = {
    "codebase": {
        "description": "One codebase tracked in revision control, many deploys",
        "remediation": "Use a single repository for the application; deploy from the same codebase to all environments",
    },
    "dependencies": {
        "description": "Explicitly declare and isolate dependencies",
        "remediation": "Use a dependency manifest (requirements.txt, pyproject.toml) and virtual environments",
    },
    "config": {
        "description": "Store config in the environment",
        "remediation": "Move configuration to environment variables; never hardcode secrets or environment-specific values",
    },
    "backing_services": {
        "description": "Treat backing services as attached resources",
        "remediation": "Access databases, caches, and queues via URLs/credentials from config; make them swappable",
    },
    "build_release_run": {
        "description": "Strictly separate build and run stages",
        "remediation": "Use CI/CD pipelines with distinct build, release, and deployment stages",
    },
    "processes": {
        "description": "Execute the app as one or more stateless processes",
        "remediation": "Store session state in backing services (Redis, database); don't rely on local filesystem",
    },
    "port_binding": {
        "description": "Export services via port binding",
        "remediation": "Make the app self-contained; bind to a port and serve requests directly",
    },
    "concurrency": {
        "description": "Scale out via the process model",
        "remediation": "Design for horizontal scaling; use process managers and container orchestration",
    },
    "disposability": {
        "description": "Maximize robustness with fast startup and graceful shutdown",
        "remediation": "Minimize startup time; handle SIGTERM gracefully; use crash-only design",
    },
    "dev_prod_parity": {
        "description": "Keep development, staging, and production as similar as possible",
        "remediation": "Use same backing services in dev as prod; minimize time between deploys; same team deploys",
    },
    "logs": {
        "description": "Treat logs as event streams",
        "remediation": "Write logs to stdout; use log aggregation services; don't manage log files in app",
    },
    "admin_processes": {
        "description": "Run admin/management tasks as one-off processes",
        "remediation": "Run migrations and scripts as one-off processes in identical environment",
    },
}

# Security anti-patterns with detection patterns, severity, and remediation
SECURITY_ANTI_PATTERNS: dict[str, dict[str, Any]] = {
    "hardcoded_secrets": {
        "patterns": [
            re.compile(r"password\s*=\s*['\"]?[^'\"\s]+", re.IGNORECASE),
            re.compile(r"api_key\s*=\s*['\"]?[^'\"\s]+", re.IGNORECASE),
            re.compile(r"secret\s*=\s*['\"]?[^'\"\s]+", re.IGNORECASE),
            re.compile(r"token\s*=\s*['\"]?[^'\"\s]+", re.IGNORECASE),
        ],
        "severity": "critical",
        "remediation": "Use environment variables or secrets manager",
    },
    "sql_injection": {
        "patterns": [
            re.compile(r"string\s+concatenation", re.IGNORECASE),
            re.compile(r"f-string\s+(?:in\s+)?query", re.IGNORECASE),
            re.compile(r"format\s+string\s+(?:in\s+)?sql", re.IGNORECASE),
        ],
        "severity": "critical",
        "remediation": "Use parameterized queries or ORM",
    },
    "missing_auth": {
        "patterns": [
            re.compile(r"no\s+authentication", re.IGNORECASE),
            re.compile(r"public\s+endpoint", re.IGNORECASE),
            re.compile(r"unprotected\s+route", re.IGNORECASE),
        ],
        "severity": "high",
        "remediation": "Add authentication middleware",
    },
    "insecure_transport": {
        "patterns": [
            re.compile(r"http://(?!localhost|127\.0\.0\.1)", re.IGNORECASE),
            re.compile(r"no\s+TLS", re.IGNORECASE),
            re.compile(r"plain\s+text\s+(?:communication|transport)", re.IGNORECASE),
        ],
        "severity": "high",
        "remediation": "Use HTTPS/TLS for all communications",
    },
    "xss_risk": {
        "patterns": [
            re.compile(r"innerHTML", re.IGNORECASE),
            re.compile(r"dangerouslySetInnerHTML", re.IGNORECASE),
            re.compile(r"unescaped\s+output", re.IGNORECASE),
        ],
        "severity": "high",
        "remediation": "Sanitize and escape user input",
    },
}


def _map_severity(original_severity: str) -> Severity:
    """Map original severity level to Severity enum.

    Args:
        original_severity: Original severity string (critical, high, medium, low).

    Returns:
        Severity.BLOCKING for critical/high, Severity.WARNING for medium/low.
    """
    if original_severity in ("critical", "high"):
        return Severity.BLOCKING
    return Severity.WARNING


def check_twelve_factor_compliance(
    architecture: dict[str, Any],
    decision_id: str = "architecture",
) -> list[GateIssue]:
    """Check architecture against 12-Factor principles.

    Validates that the architecture follows 12-Factor app methodology.

    Args:
        architecture: Architecture dict with optional 'twelve_factor' key.
        decision_id: ID of the decision being checked (for traceability).

    Returns:
        List of GateIssue for any violations found.

    Example:
        >>> arch = {"twelve_factor": {"config": False}}
        >>> issues = check_twelve_factor_compliance(arch)
        >>> len(issues) >= 1
        True
    """
    issues: list[GateIssue] = []
    twelve_factor = architecture.get("twelve_factor", {})

    if not isinstance(twelve_factor, dict):
        return issues

    # Check each principle that is explicitly marked as False
    for principle_name, principle_info in TWELVE_FACTOR_PRINCIPLES.items():
        if principle_name in twelve_factor and twelve_factor[principle_name] is False:
            description = principle_info["description"]
            remediation = principle_info["remediation"]
            original_severity = "high"
            issues.append(
                GateIssue(
                    location=decision_id,
                    issue_type="twelve_factor",
                    description=f"12-Factor violation ({principle_name}): {description}. Remediation: {remediation}",
                    severity=_map_severity(original_severity),
                    context={"original_severity": original_severity, "principle": principle_name},
                )
            )

    return issues


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers.

    Args:
        version_str: Version string like "3.10", "1.0.0", ">=3.9".

    Returns:
        Tuple of version components as integers.
    """
    # Remove comparison operators
    clean = re.sub(r"^[<>=!]+", "", version_str.strip())
    # Split and convert to integers
    try:
        return tuple(int(p) for p in clean.split(".") if p.isdigit())
    except ValueError:
        return ()


def _check_version_constraint(
    used_version: str,
    constraint: str,
) -> bool:
    """Check if a version satisfies a constraint.

    Args:
        used_version: The version being used (e.g., "3.10").
        constraint: The constraint to check (e.g., ">=3.9", "<4.0").

    Returns:
        True if version satisfies constraint, False otherwise.
    """
    if not constraint or not used_version:
        return True

    used = _parse_version(used_version)
    if not used:
        return True  # Can't validate unparseable versions

    # Determine constraint type and required version
    if constraint.startswith(">="):
        required = _parse_version(constraint[2:])
        return used >= required
    elif constraint.startswith("<="):
        required = _parse_version(constraint[2:])
        return used <= required
    elif constraint.startswith(">"):
        required = _parse_version(constraint[1:])
        return used > required
    elif constraint.startswith("<"):
        required = _parse_version(constraint[1:])
        return used < required
    elif constraint.startswith("==") or constraint.startswith("="):
        required = _parse_version(constraint.lstrip("="))
        return used == required
    elif constraint.startswith("!="):
        required = _parse_version(constraint[2:])
        return used != required
    else:
        # Exact match if no operator
        required = _parse_version(constraint)
        return used == required


def validate_tech_stack(
    architecture: dict[str, Any],
    constraints: dict[str, Any],
    decision_id: str = "tech_stack",
) -> list[GateIssue]:
    """Validate tech stack against constraints.

    Checks that the architecture uses only allowed technologies and
    validates version compatibility.

    Args:
        architecture: Architecture dict with optional 'tech_stack' key.
        constraints: Constraints dict with allowed_languages, allowed_frameworks,
            version_constraints, etc.
        decision_id: ID of the decision being checked (for traceability).

    Returns:
        List of GateIssue for any constraint violations.

    Example:
        >>> arch = {"tech_stack": {"languages": ["rust"]}}
        >>> constraints = {"allowed_languages": ["python"]}
        >>> issues = validate_tech_stack(arch, constraints)
        >>> len(issues) >= 1
        True
    """
    issues: list[GateIssue] = []

    if not constraints:
        return issues

    tech_stack = architecture.get("tech_stack", {})
    if not isinstance(tech_stack, dict):
        return issues

    # Get version constraints (e.g., {"python": ">=3.9", "node": ">=18.0"})
    version_constraints = constraints.get("version_constraints", {})

    # Check languages
    allowed_languages = constraints.get("allowed_languages", [])
    if allowed_languages:
        used_languages = tech_stack.get("languages", [])
        language_versions = tech_stack.get("language_versions", {})

        for lang in used_languages:
            lang_lower = lang.lower()
            if lang_lower not in [a.lower() for a in allowed_languages]:
                original_severity = "high"
                issues.append(
                    GateIssue(
                        location=decision_id,
                        issue_type="tech_stack",
                        description=f"Unsupported language: {lang}. Allowed: {', '.join(allowed_languages)}",
                        severity=_map_severity(original_severity),
                        context={"original_severity": original_severity},
                    )
                )
            # Check version constraint
            elif lang_lower in version_constraints or lang in version_constraints:
                constraint = version_constraints.get(lang_lower) or version_constraints.get(lang)
                used_version = language_versions.get(lang) or language_versions.get(lang_lower, "")
                if used_version and not _check_version_constraint(used_version, constraint):
                    original_severity = "high"
                    issues.append(
                        GateIssue(
                            location=decision_id,
                            issue_type="tech_stack",
                            description=f"Version incompatibility: {lang} {used_version} does not satisfy {constraint}",
                            severity=_map_severity(original_severity),
                            context={"original_severity": original_severity},
                        )
                    )

    # Check frameworks
    allowed_frameworks = constraints.get("allowed_frameworks", [])
    if allowed_frameworks:
        used_frameworks = tech_stack.get("frameworks", [])
        framework_versions = tech_stack.get("framework_versions", {})

        for framework in used_frameworks:
            framework_lower = framework.lower()
            if framework_lower not in [a.lower() for a in allowed_frameworks]:
                original_severity = "medium"
                issues.append(
                    GateIssue(
                        location=decision_id,
                        issue_type="tech_stack",
                        description=f"Unsupported framework: {framework}. Allowed: {', '.join(allowed_frameworks)}",
                        severity=_map_severity(original_severity),
                        context={"original_severity": original_severity},
                    )
                )
            # Check version constraint
            elif framework_lower in version_constraints or framework in version_constraints:
                constraint = version_constraints.get(framework_lower) or version_constraints.get(
                    framework
                )
                used_version = framework_versions.get(framework) or framework_versions.get(
                    framework_lower, ""
                )
                if used_version and not _check_version_constraint(used_version, constraint):
                    original_severity = "medium"
                    issues.append(
                        GateIssue(
                            location=decision_id,
                            issue_type="tech_stack",
                            description=f"Version incompatibility: {framework} {used_version} does not satisfy {constraint}",
                            severity=_map_severity(original_severity),
                            context={"original_severity": original_severity},
                        )
                    )

    # Check databases
    allowed_databases = constraints.get("allowed_databases", [])
    if allowed_databases:
        used_databases = tech_stack.get("databases", [])
        for db in used_databases:
            if db.lower() not in [a.lower() for a in allowed_databases]:
                original_severity = "high"
                issues.append(
                    GateIssue(
                        location=decision_id,
                        issue_type="tech_stack",
                        description=f"Unsupported database: {db}. Allowed: {', '.join(allowed_databases)}",
                        severity=_map_severity(original_severity),
                        context={"original_severity": original_severity},
                    )
                )

    return issues


def _extract_text_content(obj: Any, depth: int = 0, max_depth: int = 5) -> str:
    """Recursively extract text content from nested structures.

    Args:
        obj: Object to extract text from.
        depth: Current recursion depth.
        max_depth: Maximum recursion depth.

    Returns:
        Concatenated text content.
    """
    if depth > max_depth:
        return ""

    if isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        parts = []
        for value in obj.values():
            parts.append(_extract_text_content(value, depth + 1, max_depth))
        return " ".join(parts)

    if isinstance(obj, list):
        parts = []
        for item in obj:
            parts.append(_extract_text_content(item, depth + 1, max_depth))
        return " ".join(parts)

    return str(obj) if obj is not None else ""


def detect_security_anti_patterns(
    architecture: dict[str, Any],
    decision_id: str = "security",
) -> list[GateIssue]:
    """Detect security anti-patterns in architecture.

    Scans architecture for known security issues.

    Args:
        architecture: Architecture dict to scan.
        decision_id: ID of the decision being checked (for traceability).

    Returns:
        List of GateIssue for any security issues found.

    Example:
        >>> arch = {"security": {"secrets": "password=secret123"}}
        >>> issues = detect_security_anti_patterns(arch)
        >>> len(issues) >= 1
        True
    """
    issues: list[GateIssue] = []

    # Extract all text content from architecture for scanning
    text_content = _extract_text_content(architecture)

    for pattern_name, pattern_config in SECURITY_ANTI_PATTERNS.items():
        for pattern in pattern_config["patterns"]:
            if pattern.search(text_content):
                original_severity = pattern_config["severity"]
                issues.append(
                    GateIssue(
                        location=decision_id,
                        issue_type="security",
                        description=f"Security anti-pattern detected: {pattern_name}. {pattern_config['remediation']}",
                        severity=_map_severity(original_severity),
                        context={"original_severity": original_severity, "pattern": pattern_name},
                    )
                )
                # Only report each anti-pattern type once
                break

    return issues


def calculate_compliance_score(
    issues: list[GateIssue],
) -> tuple[int, dict[str, int]]:
    """Calculate compliance score from issues.

    Deducts points based on issue severity:
    - critical: 25 points
    - high: 15 points
    - medium: 5 points
    - low: 1 point

    Args:
        issues: List of GateIssue (with original_severity in context).

    Returns:
        Tuple of (score, breakdown) where score is 0-100 and
        breakdown shows counts by severity.

    Example:
        >>> issue = GateIssue(
        ...     location="d-1",
        ...     issue_type="security",
        ...     description="Test",
        ...     severity=Severity.BLOCKING,
        ...     context={"original_severity": "critical"},
        ... )
        >>> score, breakdown = calculate_compliance_score([issue])
        >>> score
        75
    """
    if not issues:
        return 100, {}

    breakdown: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    total_deduction = 0

    for issue in issues:
        # Extract original severity from context, default to high for blocking, low for warning
        severity = issue.context.get("original_severity")
        if severity is None:
            severity = "high" if issue.severity == Severity.BLOCKING else "low"
        if severity in breakdown:
            breakdown[severity] += 1
            total_deduction += SEVERITY_WEIGHTS.get(severity, 0)

    score = max(0, 100 - total_deduction)
    return score, breakdown


def evaluate_adrs(adrs: list[dict[str, Any]]) -> list[GateIssue]:
    """Evaluate Architecture Decision Records for completeness and quality.

    Checks each ADR for required fields and content quality.

    Args:
        adrs: List of ADR dictionaries.

    Returns:
        List of GateIssue for any ADR problems found.

    Example:
        >>> adrs = [{"id": "ADR-001", "title": "Use PostgreSQL"}]
        >>> issues = evaluate_adrs(adrs)
        >>> isinstance(issues, list)
        True
    """
    issues: list[GateIssue] = []

    required_fields = ["id", "title", "status", "context", "decision", "consequences"]

    for adr in adrs:
        if not isinstance(adr, dict):
            continue

        adr_id = adr.get("id", "unknown")

        # Check for required fields
        missing_fields = [f for f in required_fields if f not in adr or not adr[f]]
        if missing_fields:
            original_severity = "medium"
            issues.append(
                GateIssue(
                    location=adr_id,
                    issue_type="adr",
                    description=f"ADR {adr_id} missing required fields: {', '.join(missing_fields)}",
                    severity=_map_severity(original_severity),
                    context={
                        "original_severity": original_severity,
                        "missing_fields": missing_fields,
                    },
                )
            )

        # Check for security anti-patterns in ADR content
        adr_text = _extract_text_content(adr)
        for pattern_name, pattern_config in SECURITY_ANTI_PATTERNS.items():
            for pattern in pattern_config["patterns"]:
                if pattern.search(adr_text):
                    original_severity = pattern_config["severity"]
                    issues.append(
                        GateIssue(
                            location=adr_id,
                            issue_type="security",
                            description=f"ADR {adr_id} contains security anti-pattern: {pattern_name}. {pattern_config['remediation']}",
                            severity=_map_severity(original_severity),
                            context={
                                "original_severity": original_severity,
                                "pattern": pattern_name,
                            },
                        )
                    )
                    break

    return issues


async def architecture_validation_evaluator(context: GateContext) -> GateResult:
    """Evaluate architecture against validation criteria.

    Checks:
    1. 12-Factor principle compliance
    2. Tech stack constraints
    3. Security anti-patterns
    4. ADR completeness and quality
    5. Calculates compliance score

    Args:
        context: Gate context containing state with architecture.

    Returns:
        GateResult indicating pass/fail with detailed reason.

    Example:
        >>> context = GateContext(
        ...     state={"architecture": {"twelve_factor": {"config": True}}},
        ...     gate_name="architecture_validation",
        ... )
        >>> result = await architecture_validation_evaluator(context)
        >>> result.passed
        True
    """
    architecture = context.state.get("architecture")

    # Validate architecture exists
    if architecture is None:
        logger.warning(
            "architecture_validation_gate_invalid_input",
            gate_name=context.gate_name,
            reason="architecture key missing from state",
        )
        return GateResult(
            passed=False,
            gate_name=context.gate_name,
            reason="Invalid input: architecture key missing from state",
        )

    # Validate architecture is a dict
    if not isinstance(architecture, dict):
        logger.warning(
            "architecture_validation_gate_invalid_input",
            gate_name=context.gate_name,
            reason="architecture must be a dict",
            actual_type=type(architecture).__name__,
        )
        return GateResult(
            passed=False,
            gate_name=context.gate_name,
            reason=f"Invalid input: architecture must be a dict, got {type(architecture).__name__}",
        )

    # Get configuration
    config = context.state.get("config", {})
    tech_stack_constraints = config.get("tech_stack", {})

    # Get threshold from config using threshold resolver
    # Note: resolve_threshold returns 0.0-1.0, but this gate uses 0-100 score
    threshold_decimal = resolve_threshold(
        gate_name="architecture_validation",
        state=context.state,
        default=DEFAULT_COMPLIANCE_THRESHOLD,
    )
    threshold = int(threshold_decimal * 100)  # Convert to 0-100 scale

    # Extract primary decision ID if available
    decisions = architecture.get("decisions", [])
    primary_decision_id = "architecture"
    if decisions and isinstance(decisions, list) and len(decisions) > 0:
        first_decision = decisions[0]
        if isinstance(first_decision, dict) and "id" in first_decision:
            primary_decision_id = first_decision["id"]

    # Collect all issues
    all_issues: list[GateIssue] = []

    # Check 12-Factor compliance (use primary decision ID)
    twelve_factor_issues = check_twelve_factor_compliance(architecture, primary_decision_id)
    all_issues.extend(twelve_factor_issues)

    # Validate tech stack
    tech_stack_issues = validate_tech_stack(
        architecture, tech_stack_constraints, primary_decision_id
    )
    all_issues.extend(tech_stack_issues)

    # Detect security anti-patterns
    security_issues = detect_security_anti_patterns(architecture, primary_decision_id)
    all_issues.extend(security_issues)

    # Evaluate ADRs if present (AC6)
    adrs = architecture.get("adrs", [])
    if adrs and isinstance(adrs, list):
        adr_issues = evaluate_adrs(adrs)
        all_issues.extend(adr_issues)

    # Calculate compliance score
    score, breakdown = calculate_compliance_score(all_issues)
    # Convert to 0.0-1.0 scale for GateFailureReport
    score_decimal = score / 100.0
    threshold_decimal_report = threshold / 100.0

    # Log results
    logger.info(
        "architecture_validation_gate_evaluated",
        gate_name=context.gate_name,
        compliance_score=score,
        threshold=threshold,
        issue_count=len(all_issues),
        breakdown=breakdown,
    )

    # Determine if gate passes
    if score < threshold:
        report = generate_failure_report(
            gate_name=context.gate_name,
            issues=all_issues,
            score=score_decimal,
            threshold=threshold_decimal_report,
        )
        report_text = format_report_text(report)

        logger.warning(
            "architecture_validation_gate_failed",
            gate_name=context.gate_name,
            score=score,
            threshold=threshold,
            issue_count=len(all_issues),
        )

        return GateResult(
            passed=False,
            gate_name=context.gate_name,
            reason=report_text,
        )

    # Gate passes
    if all_issues:
        report = generate_failure_report(
            gate_name=context.gate_name,
            issues=all_issues,
            score=score_decimal,
            threshold=threshold_decimal_report,
        )
        report_text = format_report_text(report)
        logger.info(
            "architecture_validation_gate_passed_with_issues",
            gate_name=context.gate_name,
            score=score,
            issue_count=len(all_issues),
        )
        return GateResult(
            passed=True,
            gate_name=context.gate_name,
            reason=report_text,
        )

    logger.info(
        "architecture_validation_gate_passed",
        gate_name=context.gate_name,
        score=score,
    )

    return GateResult(
        passed=True,
        gate_name=context.gate_name,
        reason=None,
    )


# Register the evaluator when module is imported
register_evaluator("architecture_validation", architecture_validation_evaluator)
