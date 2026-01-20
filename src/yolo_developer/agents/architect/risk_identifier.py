"""Technical risk identification for architect designs (Story 7.5).

This module provides technical risk identification functionality:

- Identify technology risks (deprecated, experimental, version conflicts)
- Identify integration risks (API stability, rate limiting, vendor lock-in)
- Identify scalability risks (single points of failure, stateful components)
- Generate design-specific mitigation strategies
- LLM-powered analysis with pattern-based fallback

Example:
    >>> from yolo_developer.agents.architect.risk_identifier import (
    ...     identify_technical_risks,
    ... )
    >>>
    >>> report = await identify_technical_risks(story, design_decisions)
    >>> report.overall_risk_level
    'medium'

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
    DesignDecision,
    MitigationEffort,
    QualityRisk,
    RiskSeverity,
    TechnicalRisk,
    TechnicalRiskCategory,
    TechnicalRiskReport,
    calculate_mitigation_priority,
    calculate_overall_risk_level,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Pattern Definitions for Risk Detection
# =============================================================================

TECHNOLOGY_RISK_PATTERNS: dict[str, tuple[RiskSeverity, str]] = {
    "deprecated": ("high", "Using deprecated API or library"),
    "legacy": ("medium", "Using legacy technology"),
    "end-of-life": ("critical", "Technology reaching end-of-life"),
    "sunset": ("high", "Technology being sunset"),
    "beta": ("medium", "Using beta/preview feature"),
    "alpha": ("high", "Using alpha/unstable feature"),
    "experimental": ("high", "Using experimental feature"),
    "unstable": ("high", "Using unstable API"),
    "breaking change": ("medium", "Potential breaking changes ahead"),
    "migration required": ("medium", "Migration will be required"),
    "incompatible": ("high", "Version incompatibility detected"),
    "version conflict": ("high", "Version conflict detected"),
}

INTEGRATION_RISK_PATTERNS: dict[str, tuple[RiskSeverity, str]] = {
    "external api": ("medium", "Dependency on external API"),
    "third-party": ("medium", "Third-party service dependency"),
    "vendor": ("medium", "Vendor service dependency"),
    "rate limit": ("medium", "Rate limiting concerns"),
    "throttle": ("medium", "Throttling may apply"),
    "quota": ("medium", "Usage quota constraints"),
    "oauth": ("low", "OAuth authentication complexity"),
    "api key": ("low", "API key management required"),
    "credential": ("medium", "Credential management complexity"),
    "proprietary": ("high", "Proprietary technology lock-in"),
    "vendor-specific": ("high", "Vendor-specific implementation"),
    "platform-dependent": ("medium", "Platform-dependent features"),
    "no sla": ("high", "No SLA guarantee from provider"),
    "undocumented": ("high", "Using undocumented API"),
}

SCALABILITY_RISK_PATTERNS: dict[str, tuple[RiskSeverity, str]] = {
    "single instance": ("high", "Single point of failure"),
    "monolithic": ("medium", "Monolithic architecture limits scaling"),
    "centralized": ("medium", "Centralized component may bottleneck"),
    "session state": ("high", "Session state prevents horizontal scaling"),
    "in-memory": ("medium", "In-memory storage limits scaling"),
    "sticky session": ("high", "Sticky sessions prevent load balancing"),
    "single database": ("medium", "Single database may bottleneck"),
    "no replication": ("high", "No replication for data redundancy"),
    "shared database": ("medium", "Shared database may become bottleneck"),
    "synchronous": ("medium", "Synchronous processing limits throughput"),
    "blocking": ("medium", "Blocking operations limit concurrency"),
    "no cache": ("medium", "Lack of caching may impact performance at scale"),
}

COMPATIBILITY_RISK_PATTERNS: dict[str, tuple[RiskSeverity, str]] = {
    "version mismatch": ("high", "Version mismatch between components"),
    "protocol": ("medium", "Protocol compatibility concerns"),
    "backward": ("medium", "Backward compatibility requirements"),
    "forward": ("medium", "Forward compatibility concerns"),
}

OPERATIONAL_RISK_PATTERNS: dict[str, tuple[RiskSeverity, str]] = {
    "no monitoring": ("high", "Lack of monitoring coverage"),
    "no logging": ("medium", "Insufficient logging"),
    "deployment": ("medium", "Deployment complexity"),
    "configuration": ("medium", "Configuration management concerns"),
    "manual": ("medium", "Manual processes may cause errors"),
}


# =============================================================================
# Risk Detection Functions
# =============================================================================


def _get_combined_text(story: dict[str, Any], decisions: list[DesignDecision]) -> str:
    """Extract and combine text from story and decisions for pattern matching."""
    story_parts = [
        str(story.get("title", "")),
        str(story.get("description", "")),
        str(story.get("acceptance_criteria", "")),
    ]
    decision_parts = []
    for d in decisions:
        decision_parts.extend([d.description, d.rationale, d.decision_type])

    return " ".join(story_parts + decision_parts).lower()


def _extract_affected_components(
    story: dict[str, Any], decisions: list[DesignDecision]
) -> tuple[str, ...]:
    """Extract component names from story and decisions."""
    components: list[str] = []

    # Extract from story title
    title = story.get("title", "")
    if title:
        components.append(title)

    # Extract from decision types
    # Note: This heuristic extracts the last word from relevant decision descriptions
    # as a best-effort component name. For more accurate extraction, LLM analysis
    # should be used (enabled by default via use_llm=True in identify_technical_risks).
    for d in decisions:
        if d.decision_type in ("data", "infrastructure", "technology"):
            desc_words = d.description.split()
            if len(desc_words) >= 2:
                components.append(desc_words[-1])

    return tuple(components) if components else ("UnspecifiedComponent",)


def _identify_technology_risks(
    story: dict[str, Any], decisions: list[DesignDecision]
) -> list[TechnicalRisk]:
    """Identify technology-related risks.

    Detects:
    - Deprecated APIs or libraries
    - Experimental/unstable features
    - Version conflicts and compatibility issues

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        List of identified technology risks.
    """
    risks: list[TechnicalRisk] = []
    text = _get_combined_text(story, decisions)
    affected = _extract_affected_components(story, decisions)

    for pattern, (severity, description) in TECHNOLOGY_RISK_PATTERNS.items():
        if pattern in text:
            risks.append(
                TechnicalRisk(
                    category="technology",
                    description=description,
                    severity=severity,
                    affected_components=affected,
                    mitigation="",  # Will be filled by mitigation engine
                    mitigation_effort="medium",
                    mitigation_priority="P2",
                )
            )

    logger.debug(
        "technology_risks_identified",
        risk_count=len(risks),
        patterns_matched=[r.description for r in risks],
    )

    return risks


def _identify_integration_risks(
    story: dict[str, Any], decisions: list[DesignDecision]
) -> list[TechnicalRisk]:
    """Identify integration-related risks.

    Detects:
    - External API dependencies
    - Rate limiting concerns
    - Vendor lock-in
    - Authentication complexity

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        List of identified integration risks.
    """
    risks: list[TechnicalRisk] = []
    text = _get_combined_text(story, decisions)

    # Extract integration-specific affected components
    integration_components: list[str] = []
    for d in decisions:
        if d.decision_type == "integration":
            integration_components.append(d.description.split()[0])

    affected = (
        tuple(integration_components)
        if integration_components
        else _extract_affected_components(story, decisions)
    )

    for pattern, (severity, description) in INTEGRATION_RISK_PATTERNS.items():
        if pattern in text:
            risks.append(
                TechnicalRisk(
                    category="integration",
                    description=description,
                    severity=severity,
                    affected_components=affected,
                    mitigation="",
                    mitigation_effort="medium",
                    mitigation_priority="P2",
                )
            )

    logger.debug(
        "integration_risks_identified",
        risk_count=len(risks),
        patterns_matched=[r.description for r in risks],
    )

    return risks


def _identify_scalability_risks(
    story: dict[str, Any], decisions: list[DesignDecision]
) -> list[TechnicalRisk]:
    """Identify scalability-related risks.

    Detects:
    - Single points of failure
    - Stateful components that prevent horizontal scaling
    - Database bottlenecks

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        List of identified scalability risks.
    """
    risks: list[TechnicalRisk] = []
    text = _get_combined_text(story, decisions)
    affected = _extract_affected_components(story, decisions)

    for pattern, (severity, description) in SCALABILITY_RISK_PATTERNS.items():
        if pattern in text:
            # Link to specific decisions
            linked_decisions = [d.id for d in decisions if pattern in d.description.lower()]
            components = tuple(linked_decisions) if linked_decisions else affected

            risks.append(
                TechnicalRisk(
                    category="scalability",
                    description=description,
                    severity=severity,
                    affected_components=components,
                    mitigation="",
                    mitigation_effort="high",  # Scalability fixes often require more effort
                    mitigation_priority="P2",
                )
            )

    logger.debug(
        "scalability_risks_identified",
        risk_count=len(risks),
        patterns_matched=[r.description for r in risks],
    )

    return risks


def _identify_compatibility_risks(
    story: dict[str, Any], decisions: list[DesignDecision]
) -> list[TechnicalRisk]:
    """Identify compatibility-related risks.

    Detects:
    - Version mismatches between components
    - Protocol compatibility concerns
    - Backward/forward compatibility requirements

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        List of identified compatibility risks.
    """
    risks: list[TechnicalRisk] = []
    text = _get_combined_text(story, decisions)
    affected = _extract_affected_components(story, decisions)

    for pattern, (severity, description) in COMPATIBILITY_RISK_PATTERNS.items():
        if pattern in text:
            risks.append(
                TechnicalRisk(
                    category="compatibility",
                    description=description,
                    severity=severity,
                    affected_components=affected,
                    mitigation="",
                    mitigation_effort="medium",
                    mitigation_priority="P2",
                )
            )

    return risks


def _identify_operational_risks(
    story: dict[str, Any], decisions: list[DesignDecision]
) -> list[TechnicalRisk]:
    """Identify operational-related risks.

    Detects:
    - Missing monitoring or logging coverage
    - Deployment complexity concerns
    - Configuration management issues
    - Manual process dependencies

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        List of identified operational risks.
    """
    risks: list[TechnicalRisk] = []
    text = _get_combined_text(story, decisions)
    affected = _extract_affected_components(story, decisions)

    for pattern, (severity, description) in OPERATIONAL_RISK_PATTERNS.items():
        if pattern in text:
            risks.append(
                TechnicalRisk(
                    category="operational",
                    description=description,
                    severity=severity,
                    affected_components=affected,
                    mitigation="",
                    mitigation_effort="medium",
                    mitigation_priority="P3",
                )
            )

    return risks


# =============================================================================
# Mitigation Suggestion Engine
# =============================================================================

MITIGATION_STRATEGIES: dict[TechnicalRiskCategory, dict[str, str]] = {
    "technology": {
        "deprecated": "Migrate to supported alternative; create migration plan with timeline",
        "legacy": "Evaluate modernization options; document legacy system boundaries",
        "experimental": "Add abstraction layer to isolate experimental features; monitor for stability",
        "version conflict": "Use dependency resolution tools; pin versions explicitly",
        "default": "Document technology risks; plan for upgrade path",
    },
    "integration": {
        "external api": "Implement circuit breaker pattern; add local cache fallback",
        "rate limit": "Implement request throttling; add retry with exponential backoff",
        "vendor": "Create abstraction layer; document vendor dependency",
        "proprietary": "Add adapter pattern for vendor abstraction; evaluate alternatives",
        "credential": "Use secrets management solution; rotate credentials regularly",
        "default": "Document integration points; implement health checks",
    },
    "scalability": {
        "single instance": "Add redundancy; implement failover mechanism",
        "session state": "Externalize session state to distributed cache (Redis)",
        "single database": "Consider read replicas; implement connection pooling",
        "no replication": "Add database replication; implement backup strategy",
        "synchronous": "Convert to async processing where possible",
        "default": "Document scaling constraints; plan horizontal scaling path",
    },
    "compatibility": {
        "version mismatch": "Standardize on compatible versions; add version compatibility tests",
        "protocol": "Document protocol requirements; add protocol version negotiation",
        "default": "Add compatibility tests; document version requirements",
    },
    "operational": {
        "no monitoring": "Add observability (metrics, traces, logs); implement alerting",
        "deployment": "Automate deployment; implement blue-green or canary releases",
        "configuration": "Centralize configuration; use configuration management tool",
        "default": "Document operational procedures; add runbooks",
    },
}


def _generate_design_specific_mitigation(
    risk: TechnicalRisk,
    story: dict[str, Any],
    decisions: list[DesignDecision],
) -> str:
    """Generate a design-specific mitigation strategy.

    Args:
        risk: The technical risk to mitigate.
        story: Story dictionary for context.
        decisions: Design decisions for context.

    Returns:
        Design-specific mitigation string.
    """
    category_strategies = MITIGATION_STRATEGIES.get(risk.category, {})

    # Find matching strategy by keywords in risk description
    description_lower = risk.description.lower()
    mitigation = category_strategies.get("default", "Review and address risk")

    for keyword, strategy in category_strategies.items():
        if keyword != "default" and keyword in description_lower:
            mitigation = strategy
            break

    # Add design-specific context
    story_title = story.get("title", "")
    context_parts: list[str] = []

    if story_title:
        context_parts.append(f"For '{story_title}'")

    # Reference affected components
    if risk.affected_components:
        components_str = ", ".join(risk.affected_components[:3])
        context_parts.append(f"affecting {components_str}")

    # Reference relevant decisions
    relevant_decisions = [d for d in decisions if d.decision_type == risk.category]
    if relevant_decisions:
        context_parts.append(f"review {relevant_decisions[0].decision_type} decision")

    if context_parts:
        specific_context = "; ".join(context_parts)
        return f"{mitigation}. Specific context: {specific_context}."

    return mitigation


def _estimate_mitigation_effort(risk: TechnicalRisk) -> MitigationEffort:
    """Estimate effort required for mitigation based on category and severity."""
    # Scalability fixes typically require more effort
    if risk.category == "scalability":
        if risk.severity in ("critical", "high"):
            return "high"
        return "medium"

    # Technology migrations can be significant
    if risk.category == "technology":
        if "deprecated" in risk.description.lower() or "end-of-life" in risk.description.lower():
            return "high"
        return "medium"

    # Integration risks vary
    if risk.category == "integration":
        if "proprietary" in risk.description.lower() or "vendor" in risk.description.lower():
            return "high"
        return "medium"

    # Default based on severity
    if risk.severity == "critical":
        return "high"
    elif risk.severity == "high":
        return "medium"
    return "low"


def _generate_mitigations(
    risks: list[TechnicalRisk],
    story: dict[str, Any],
    decisions: list[DesignDecision],
) -> list[TechnicalRisk]:
    """Generate mitigations for all risks.

    Creates new TechnicalRisk objects with filled mitigation fields.

    Args:
        risks: List of risks without mitigations.
        story: Story dictionary for context.
        decisions: Design decisions for context.

    Returns:
        List of TechnicalRisk objects with mitigations filled.
    """
    mitigated_risks: list[TechnicalRisk] = []

    for risk in risks:
        mitigation = _generate_design_specific_mitigation(risk, story, decisions)
        effort = _estimate_mitigation_effort(risk)
        priority = calculate_mitigation_priority(risk.severity, effort)

        mitigated_risk = TechnicalRisk(
            category=risk.category,
            description=risk.description,
            severity=risk.severity,
            affected_components=risk.affected_components,
            mitigation=mitigation,
            mitigation_effort=effort,
            mitigation_priority=priority,
        )
        mitigated_risks.append(mitigated_risk)

    return mitigated_risks


# =============================================================================
# LLM-Powered Risk Analysis
# =============================================================================

RISK_IDENTIFICATION_PROMPT = """Identify technical risks in the following design.

Story:
Title: {story_title}
Description: {story_description}

Design Decisions:
{design_decisions}

Identify risks in these categories:
1. Technology: Library maturity, deprecation, version conflicts
2. Integration: External API stability, rate limits, auth complexity, vendor lock-in
3. Scalability: Single points of failure, stateful components, bottlenecks
4. Compatibility: Version mismatches, protocol issues
5. Operational: Monitoring gaps, deployment complexity

For each risk, provide:
- Category: One of the 5 categories above
- Description: What is the risk?
- Severity: critical, high, medium, or low
- Affected components: Which parts of the design are affected?
- Mitigation: How to address this risk (be specific to this design)
- Mitigation effort: high, medium, or low

Respond in JSON format:
{{
  "risks": [
    {{
      "category": "integration",
      "description": "External API has no SLA guarantee",
      "severity": "high",
      "affected_components": ["AuthService", "UserAPI"],
      "mitigation": "Implement circuit breaker and local cache fallback",
      "mitigation_effort": "medium"
    }}
  ],
  "overall_risk_level": "high",
  "summary": "Brief summary of key risks"
}}
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _call_risk_llm(prompt: str) -> str:
    """Call LLM for risk identification with retry logic.

    Args:
        prompt: The prompt to send to the LLM.

    Returns:
        LLM response text.

    Raises:
        Exception: If all retries fail.
    """
    import litellm

    model = os.environ.get("YOLO_LLM__ROUTINE_MODEL", "gpt-4o-mini")

    logger.debug("calling_risk_llm", model=model, prompt_length=len(prompt))

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty response")

    return str(content)


async def _analyze_risks_with_llm(
    story: dict[str, Any],
    decisions: list[DesignDecision],
) -> TechnicalRiskReport | None:
    """Analyze risks using LLM.

    Uses LLM to identify technical risks with design-specific context.

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        TechnicalRiskReport if successful, None if LLM fails.
    """
    # Build prompt
    decisions_text = "\n".join(
        f"- {d.decision_type}: {d.description} (Rationale: {d.rationale})" for d in decisions
    )

    prompt = RISK_IDENTIFICATION_PROMPT.format(
        story_title=story.get("title", "Unknown"),
        story_description=story.get("description", "No description"),
        design_decisions=decisions_text or "No design decisions provided",
    )

    try:
        response_text = await _call_risk_llm(prompt)

        # Parse JSON response - use regex for robust extraction from code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if json_match:
            response_text = json_match.group(1)

        data = json.loads(response_text.strip())

        # Extract risks
        risks_data = data.get("risks", [])
        risks: list[TechnicalRisk] = []

        for r in risks_data:
            category = r.get("category", "technology")
            # Validate category
            if category not in (
                "technology",
                "integration",
                "scalability",
                "compatibility",
                "operational",
            ):
                category = "technology"

            severity = r.get("severity", "medium")
            if severity not in ("critical", "high", "medium", "low"):
                severity = "medium"

            effort_raw = r.get("mitigation_effort", "medium")
            if effort_raw not in ("high", "medium", "low"):
                effort_raw = "medium"

            # Type assertions are safe after validation above
            validated_category: TechnicalRiskCategory = category
            validated_severity: RiskSeverity = severity
            validated_effort: MitigationEffort = effort_raw
            priority = calculate_mitigation_priority(validated_severity, validated_effort)

            risks.append(
                TechnicalRisk(
                    category=validated_category,
                    description=r.get("description", ""),
                    severity=validated_severity,
                    affected_components=tuple(r.get("affected_components", [])),
                    mitigation=r.get("mitigation", ""),
                    mitigation_effort=validated_effort,
                    mitigation_priority=priority,
                )
            )

        overall_risk_level = data.get("overall_risk_level", "low")
        if overall_risk_level not in ("critical", "high", "medium", "low"):
            overall_risk_level = calculate_overall_risk_level(risks)

        summary = data.get("summary", "LLM-generated risk analysis")

        validated_overall: RiskSeverity = overall_risk_level
        return TechnicalRiskReport(
            risks=tuple(risks),
            overall_risk_level=validated_overall,
            summary=summary,
        )

    except json.JSONDecodeError as e:
        logger.warning("llm_response_json_parse_error", error=str(e))
        return None
    except Exception as e:
        logger.warning("llm_risk_analysis_failed", error=str(e))
        return None


# =============================================================================
# Main Risk Identification Function
# =============================================================================


def _convert_quality_risks(
    quality_risks: list[QualityRisk],
) -> list[TechnicalRisk]:
    """Convert quality risks to technical risks for consolidation.

    Args:
        quality_risks: List of QualityRisk from quality evaluation.

    Returns:
        List of TechnicalRisk objects.
    """
    converted: list[TechnicalRisk] = []

    for qr in quality_risks:
        # Map quality attributes to technical categories
        category_map: dict[str, TechnicalRiskCategory] = {
            "performance": "scalability",
            "security": "technology",
            "reliability": "operational",
            "scalability": "scalability",
            "maintainability": "operational",
            "integration": "integration",
            "cost_efficiency": "operational",
        }

        category = category_map.get(qr.attribute, "operational")
        priority = calculate_mitigation_priority(qr.severity, qr.mitigation_effort)

        converted.append(
            TechnicalRisk(
                category=category,
                description=f"[Quality] {qr.description}",
                severity=qr.severity,
                affected_components=(qr.attribute,),
                mitigation=qr.mitigation,
                mitigation_effort=qr.mitigation_effort,
                mitigation_priority=priority,
            )
        )

    return converted


def _generate_summary(risks: list[TechnicalRisk]) -> str:
    """Generate a summary of identified risks.

    Args:
        risks: List of technical risks.

    Returns:
        Summary string describing key risks.
    """
    if not risks:
        return "No technical risks identified"

    # Count by category
    category_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}

    for risk in risks:
        category_counts[risk.category] = category_counts.get(risk.category, 0) + 1
        severity_counts[risk.severity] = severity_counts.get(risk.severity, 0) + 1

    # Build summary
    parts: list[str] = []
    parts.append(f"{len(risks)} technical risks identified")

    # Highlight critical/high
    critical = severity_counts.get("critical", 0)
    high = severity_counts.get("high", 0)
    if critical > 0:
        parts.append(f"{critical} critical")
    if high > 0:
        parts.append(f"{high} high-severity")

    # Top categories
    sorted_categories = sorted(category_counts.items(), key=lambda x: -x[1])
    if sorted_categories:
        top_cat = sorted_categories[0]
        parts.append(f"primarily {top_cat[0]} ({top_cat[1]})")

    return "; ".join(parts)


async def identify_technical_risks(
    story: dict[str, Any],
    decisions: list[DesignDecision],
    quality_risks: list[QualityRisk] | None = None,
    use_llm: bool = True,
) -> TechnicalRiskReport:
    """Identify technical risks in a design.

    Main entry point for technical risk identification. Combines pattern-based
    detection with optional LLM analysis for comprehensive risk coverage.

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.
        quality_risks: Optional quality risks from Story 7.4 to incorporate.
        use_llm: Whether to attempt LLM-powered analysis (default True).

    Returns:
        TechnicalRiskReport with identified risks and mitigations.
    """
    story_id = story.get("id", "unknown")
    logger.info(
        "risk_identification_start",
        story_id=story_id,
        decision_count=len(decisions),
        quality_risk_count=len(quality_risks) if quality_risks else 0,
    )

    # Try LLM analysis first if enabled
    if use_llm:
        llm_result = await _analyze_risks_with_llm(story, decisions)
        if llm_result is not None:
            # Incorporate quality risks if provided
            if quality_risks:
                converted = _convert_quality_risks(quality_risks)
                combined_risks = list(llm_result.risks) + converted
                overall = calculate_overall_risk_level(combined_risks)
                summary = _generate_summary(combined_risks)

                llm_result = TechnicalRiskReport(
                    risks=tuple(combined_risks),
                    overall_risk_level=overall,
                    summary=summary,
                )

            logger.info(
                "risk_identification_complete",
                story_id=story_id,
                method="llm",
                risk_count=len(llm_result.risks),
                overall_risk_level=llm_result.overall_risk_level,
            )
            return llm_result

    # Fall back to pattern-based detection
    all_risks: list[TechnicalRisk] = []

    # Detect risks by category
    all_risks.extend(_identify_technology_risks(story, decisions))
    all_risks.extend(_identify_integration_risks(story, decisions))
    all_risks.extend(_identify_scalability_risks(story, decisions))
    all_risks.extend(_identify_compatibility_risks(story, decisions))
    all_risks.extend(_identify_operational_risks(story, decisions))

    # Incorporate quality risks if provided
    if quality_risks:
        all_risks.extend(_convert_quality_risks(quality_risks))

    # Generate mitigations
    all_risks = _generate_mitigations(all_risks, story, decisions)

    # Calculate overall risk level
    overall_risk_level = calculate_overall_risk_level(all_risks)

    # Generate summary
    summary = _generate_summary(all_risks)

    report = TechnicalRiskReport(
        risks=tuple(all_risks),
        overall_risk_level=overall_risk_level,
        summary=summary,
    )

    logger.info(
        "risk_identification_complete",
        story_id=story_id,
        method="pattern",
        risk_count=len(all_risks),
        overall_risk_level=overall_risk_level,
    )

    return report
