"""Quality attribute evaluation for architect designs (Story 7.4).

This module provides quality attribute evaluation functionality:

- Score individual quality attributes (performance, security, reliability, etc.)
- Detect trade-offs between conflicting attributes
- Identify risks to meeting NFRs
- Suggest mitigations for identified risks
- LLM-powered evaluation with pattern-based fallback

Example:
    >>> from yolo_developer.agents.architect.quality_evaluator import (
    ...     evaluate_quality_attributes,
    ... )
    >>>
    >>> evaluation = await evaluate_quality_attributes(story, design_decisions)
    >>> evaluation.overall_score
    0.75

Architecture:
    - Uses frozen dataclasses per ADR-001
    - Uses litellm for LLM calls per ADR-003
    - Uses tenacity for retry logic per ADR-007
    - All I/O operations are async per ARCH-QUALITY-5
"""

from __future__ import annotations

import json
import os
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.config.schema import LLM_CHEAP_MODEL_DEFAULT
from yolo_developer.agents.architect.types import (
    DesignDecision,
    MitigationEffort,
    QualityAttributeEvaluation,
    QualityRisk,
    QualityTradeOff,
    RiskSeverity,
)

logger = structlog.get_logger(__name__)

# =============================================================================
# Attribute Weights for Overall Score Calculation
# =============================================================================

DEFAULT_BASELINE_SCORE: float = 0.7
"""Default baseline score for quality attribute evaluation.

All scoring functions start with this baseline and adjust based on
positive/negative patterns detected in the story and design decisions.
"""

ATTRIBUTE_WEIGHTS: dict[str, float] = {
    "performance": 0.15,
    "security": 0.20,
    "reliability": 0.20,
    "scalability": 0.15,
    "maintainability": 0.15,
    "integration": 0.10,
    "cost_efficiency": 0.05,
}
"""Weights for calculating overall quality score.

Weights sum to 1.0 and reflect relative importance:
- Security and reliability: highest (0.20 each) - critical for production
- Performance, scalability, maintainability: medium (0.15 each)
- Integration: lower (0.10) - important but not critical
- Cost efficiency: lowest (0.05) - nice to have
"""


# =============================================================================
# Pattern-Based Scoring Functions
# =============================================================================


def _score_performance(story: dict[str, Any], decisions: list[DesignDecision]) -> float:
    """Score a design for performance quality attribute.

    Evaluates story and design decisions for performance-related patterns:
    - Caching strategies
    - Async/await patterns
    - Database optimization
    - Resource efficiency

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        Score from 0.0 to 1.0 indicating performance quality.
    """
    score = DEFAULT_BASELINE_SCORE  # Start with reasonable baseline

    text = _get_story_text(story)
    decision_text = _get_decision_text(decisions)
    combined = f"{text} {decision_text}".lower()

    # Positive patterns (increase score)
    positive_patterns = [
        ("async", 0.05),
        ("await", 0.05),
        ("cache", 0.08),
        ("caching", 0.08),
        ("index", 0.05),
        ("batch", 0.05),
        ("pool", 0.05),
        ("connection pool", 0.08),
        ("lazy load", 0.05),
        ("pagination", 0.05),
    ]

    for pattern, boost in positive_patterns:
        if pattern in combined:
            score += boost

    # Negative patterns (decrease score)
    negative_patterns = [
        ("synchronous", -0.05),
        ("blocking", -0.08),
        ("n+1", -0.10),
        ("no cache", -0.08),
        ("unbounded", -0.05),
    ]

    for pattern, penalty in negative_patterns:
        if pattern in combined:
            score += penalty

    return max(0.0, min(1.0, score))


def _score_security(story: dict[str, Any], decisions: list[DesignDecision]) -> float:
    """Score a design for security quality attribute.

    Evaluates story and design decisions for security-related patterns:
    - Authentication mechanisms
    - Authorization patterns
    - Input validation
    - Data encryption

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        Score from 0.0 to 1.0 indicating security quality.
    """
    score = DEFAULT_BASELINE_SCORE  # Start with reasonable baseline

    text = _get_story_text(story)
    decision_text = _get_decision_text(decisions)
    combined = f"{text} {decision_text}".lower()

    # Check if security is in scope
    security_relevant = any(
        kw in combined
        for kw in ["auth", "user", "login", "password", "token", "api", "credential", "secret"]
    )

    if not security_relevant:
        return 0.85  # High score if security not in scope

    # Positive patterns (increase score)
    positive_patterns = [
        ("authentication", 0.08),
        ("authorization", 0.08),
        ("validation", 0.05),
        ("sanitize", 0.05),
        ("encrypt", 0.08),
        ("hash", 0.05),
        ("jwt", 0.05),
        ("oauth", 0.08),
        ("rbac", 0.05),
        ("secure", 0.03),
        ("tls", 0.05),
        ("https", 0.05),
    ]

    for pattern, boost in positive_patterns:
        if pattern in combined:
            score += boost

    # Negative patterns (decrease score)
    negative_patterns = [
        ("plain text", -0.10),
        ("hardcoded", -0.08),
        ("no validation", -0.10),
        ("sql injection", -0.15),
        ("xss", -0.10),
    ]

    for pattern, penalty in negative_patterns:
        if pattern in combined:
            score += penalty

    return max(0.0, min(1.0, score))


def _score_reliability(story: dict[str, Any], decisions: list[DesignDecision]) -> float:
    """Score a design for reliability quality attribute.

    Evaluates story and design decisions for reliability-related patterns:
    - Error handling
    - Retry mechanisms
    - Fault tolerance
    - Recovery patterns

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        Score from 0.0 to 1.0 indicating reliability quality.
    """
    score = DEFAULT_BASELINE_SCORE  # Start with reasonable baseline

    text = _get_story_text(story)
    decision_text = _get_decision_text(decisions)
    combined = f"{text} {decision_text}".lower()

    # Positive patterns (increase score)
    positive_patterns = [
        ("retry", 0.08),
        ("tenacity", 0.08),
        ("exponential backoff", 0.08),
        ("circuit breaker", 0.10),
        ("fallback", 0.08),
        ("graceful", 0.05),
        ("recovery", 0.05),
        ("checkpoint", 0.08),
        ("idempotent", 0.08),
        ("transaction", 0.05),
        ("rollback", 0.05),
    ]

    for pattern, boost in positive_patterns:
        if pattern in combined:
            score += boost

    # Negative patterns (decrease score)
    negative_patterns = [
        ("no retry", -0.10),
        ("fail fast", -0.05),  # Sometimes intentional
        ("single point", -0.10),
        ("no recovery", -0.10),
    ]

    for pattern, penalty in negative_patterns:
        if pattern in combined:
            score += penalty

    return max(0.0, min(1.0, score))


def _score_scalability(story: dict[str, Any], decisions: list[DesignDecision]) -> float:
    """Score a design for scalability quality attribute.

    Evaluates story and design decisions for scalability-related patterns:
    - Horizontal scaling
    - Stateless design
    - Load distribution
    - Resource management

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        Score from 0.0 to 1.0 indicating scalability quality.
    """
    score = DEFAULT_BASELINE_SCORE  # Start with reasonable baseline

    text = _get_story_text(story)
    decision_text = _get_decision_text(decisions)
    combined = f"{text} {decision_text}".lower()

    # Positive patterns (increase score)
    positive_patterns = [
        ("stateless", 0.10),
        ("horizontal", 0.08),
        ("scale out", 0.08),
        ("distributed", 0.05),
        ("queue", 0.05),
        ("message", 0.05),
        ("partition", 0.05),
        ("sharding", 0.08),
        ("load balanc", 0.08),
        ("elastic", 0.05),
    ]

    for pattern, boost in positive_patterns:
        if pattern in combined:
            score += boost

    # Negative patterns (decrease score)
    negative_patterns = [
        ("stateful", -0.08),
        ("single instance", -0.08),
        ("monolith", -0.05),  # Not always bad
        ("vertical only", -0.05),
        ("sticky session", -0.08),
    ]

    for pattern, penalty in negative_patterns:
        if pattern in combined:
            score += penalty

    return max(0.0, min(1.0, score))


def _score_maintainability(story: dict[str, Any], decisions: list[DesignDecision]) -> float:
    """Score a design for maintainability quality attribute.

    Evaluates story and design decisions for maintainability-related patterns:
    - Code organization
    - Documentation
    - Testing strategy
    - Modularity

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        Score from 0.0 to 1.0 indicating maintainability quality.
    """
    score = DEFAULT_BASELINE_SCORE  # Start with reasonable baseline

    text = _get_story_text(story)
    decision_text = _get_decision_text(decisions)
    combined = f"{text} {decision_text}".lower()

    # Positive patterns (increase score)
    positive_patterns = [
        ("test", 0.05),
        ("unit test", 0.08),
        ("integration test", 0.08),
        ("document", 0.05),
        ("logging", 0.05),
        ("structlog", 0.08),
        ("modular", 0.05),
        ("separation of concerns", 0.08),
        ("clean", 0.03),
        ("type hint", 0.05),
        ("dataclass", 0.05),
        ("pydantic", 0.05),
    ]

    for pattern, boost in positive_patterns:
        if pattern in combined:
            score += boost

    # Negative patterns (decrease score)
    negative_patterns = [
        ("no test", -0.10),
        ("no document", -0.08),
        ("tightly coupled", -0.08),
        ("god class", -0.10),
        ("magic number", -0.05),
    ]

    for pattern, penalty in negative_patterns:
        if pattern in combined:
            score += penalty

    return max(0.0, min(1.0, score))


def _score_integration(story: dict[str, Any], decisions: list[DesignDecision]) -> float:
    """Score a design for integration quality attribute.

    Evaluates story and design decisions for integration-related patterns:
    - API design
    - Protocol compliance
    - External service handling
    - Abstraction layers

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        Score from 0.0 to 1.0 indicating integration quality.
    """
    score = DEFAULT_BASELINE_SCORE  # Start with reasonable baseline

    text = _get_story_text(story)
    decision_text = _get_decision_text(decisions)
    combined = f"{text} {decision_text}".lower()

    # Positive patterns (increase score)
    positive_patterns = [
        ("abstraction", 0.08),
        ("interface", 0.05),
        ("protocol", 0.05),
        ("adapter", 0.08),
        ("litellm", 0.08),
        ("mcp", 0.08),
        ("opentelemetry", 0.05),
        ("api", 0.03),
        ("rest", 0.05),
        ("sdk", 0.05),
    ]

    for pattern, boost in positive_patterns:
        if pattern in combined:
            score += boost

    # Negative patterns (decrease score)
    negative_patterns = [
        ("vendor lock", -0.10),
        ("tight coupling", -0.08),
        ("hard dependency", -0.08),
        ("no abstraction", -0.08),
    ]

    for pattern, penalty in negative_patterns:
        if pattern in combined:
            score += penalty

    return max(0.0, min(1.0, score))


def _score_cost_efficiency(story: dict[str, Any], decisions: list[DesignDecision]) -> float:
    """Score a design for cost efficiency quality attribute.

    Evaluates story and design decisions for cost-related patterns:
    - Resource optimization
    - Caching strategies
    - Model tiering
    - Token efficiency

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        Score from 0.0 to 1.0 indicating cost efficiency quality.
    """
    score = DEFAULT_BASELINE_SCORE  # Start with reasonable baseline

    text = _get_story_text(story)
    decision_text = _get_decision_text(decisions)
    combined = f"{text} {decision_text}".lower()

    # Positive patterns (increase score)
    positive_patterns = [
        ("cache", 0.08),
        ("tiering", 0.08),
        ("cheap model", 0.08),
        ("routine model", 0.05),
        ("optimize", 0.05),
        ("efficient", 0.05),
        ("batch", 0.05),
        ("reuse", 0.05),
    ]

    for pattern, boost in positive_patterns:
        if pattern in combined:
            score += boost

    # Negative patterns (decrease score)
    negative_patterns = [
        ("expensive", -0.05),
        ("always use", -0.05),
        ("no cache", -0.08),
        ("wasteful", -0.10),
    ]

    for pattern, penalty in negative_patterns:
        if pattern in combined:
            score += penalty

    return max(0.0, min(1.0, score))


# =============================================================================
# Helper Functions
# =============================================================================


def _get_story_text(story: dict[str, Any]) -> str:
    """Extract text content from a story dictionary."""
    parts = [
        str(story.get("title", "")),
        str(story.get("description", "")),
        str(story.get("acceptance_criteria", "")),
    ]
    return " ".join(parts)


def _get_decision_text(decisions: list[DesignDecision]) -> str:
    """Extract text content from design decisions."""
    parts = []
    for d in decisions:
        parts.extend([d.description, d.rationale, d.decision_type])
    return " ".join(parts)


def _calculate_overall_score(attribute_scores: dict[str, float]) -> float:
    """Calculate weighted overall score from attribute scores.

    Args:
        attribute_scores: Dict mapping attribute names to scores (0.0-1.0).

    Returns:
        Weighted overall score (0.0-1.0).
    """
    total = 0.0
    weight_sum = 0.0

    for attr, score in attribute_scores.items():
        weight = ATTRIBUTE_WEIGHTS.get(attr, 0.1)
        total += score * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.0

    return total / weight_sum


def _score_to_severity(score: float) -> RiskSeverity:
    """Convert a score to a risk severity level.

    Args:
        score: Score from 0.0 to 1.0.

    Returns:
        Risk severity based on score thresholds.
    """
    if score < 0.3:
        return "critical"
    elif score < 0.5:
        return "high"
    elif score < 0.7:
        return "medium"
    else:
        return "low"


def _effort_for_severity(severity: RiskSeverity) -> MitigationEffort:
    """Estimate mitigation effort based on risk severity.

    Args:
        severity: Risk severity level.

    Returns:
        Estimated mitigation effort.
    """
    if severity == "critical":
        return "high"
    elif severity == "high":
        return "medium"
    else:
        return "low"


# =============================================================================
# Trade-Off Detection
# =============================================================================

# Common trade-off patterns between quality attributes
TRADE_OFF_PATTERNS: list[tuple[str, str, str, str]] = [
    (
        "performance",
        "security",
        "Encryption and authentication add processing overhead",
        "Cache auth tokens, use async encryption, optimize crypto algorithms",
    ),
    (
        "performance",
        "reliability",
        "Caching may serve stale data, sync writes add latency",
        "Configure appropriate TTLs, use async writes with confirmation",
    ),
    (
        "scalability",
        "maintainability",
        "Distributed systems add complexity and debugging difficulty",
        "Start simple, document scaling path, use observability tools",
    ),
    (
        "security",
        "cost_efficiency",
        "Security measures (encryption, auditing) increase resource usage",
        "Optimize security hot paths, use tiered security based on data sensitivity",
    ),
    (
        "reliability",
        "performance",
        "Redundancy and retries add latency and resource consumption",
        "Use circuit breakers, implement smart retry with backoff",
    ),
]


def _detect_trade_offs(
    story: dict[str, Any],
    decisions: list[DesignDecision],
    attribute_scores: dict[str, float],
) -> list[QualityTradeOff]:
    """Detect trade-offs between quality attributes.

    Identifies conflicts where improving one attribute may
    negatively impact another.

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.
        attribute_scores: Scores for each quality attribute.

    Returns:
        List of identified trade-offs.
    """
    trade_offs: list[QualityTradeOff] = []
    text = f"{_get_story_text(story)} {_get_decision_text(decisions)}".lower()

    for attr_a, attr_b, description, resolution in TRADE_OFF_PATTERNS:
        # Check if both attributes are relevant to this story
        score_a = attribute_scores.get(attr_a, 0.7)
        score_b = attribute_scores.get(attr_b, 0.7)

        # Detect trade-off if:
        # 1. Both scores are moderate (not extremely high or low)
        # 2. There's a significant difference between them
        # 3. Story context suggests both attributes are relevant
        if 0.3 < score_a < 0.9 and 0.3 < score_b < 0.9:
            # Check for relevant keywords in story
            relevant_keywords = {
                "performance": ["fast", "latency", "throughput", "speed", "response"],
                "security": ["auth", "secure", "encrypt", "protect", "credential"],
                "reliability": ["retry", "recover", "fault", "failover", "redundant"],
                "scalability": ["scale", "distributed", "load", "concurrent", "growth"],
                "maintainability": ["test", "document", "clean", "readable", "modular"],
                "integration": ["api", "interface", "protocol", "external", "integrate"],
                "cost_efficiency": ["cost", "efficient", "optimize", "cache", "tier"],
            }

            keywords_a = relevant_keywords.get(attr_a, [])
            keywords_b = relevant_keywords.get(attr_b, [])

            has_a = any(kw in text for kw in keywords_a)
            has_b = any(kw in text for kw in keywords_b)

            if has_a and has_b:
                trade_offs.append(
                    QualityTradeOff(
                        attribute_a=attr_a,
                        attribute_b=attr_b,
                        description=description,
                        resolution=resolution,
                    )
                )

    return trade_offs


# =============================================================================
# Risk Identification
# =============================================================================

# Risk descriptions for low-scoring attributes
RISK_DESCRIPTIONS: dict[str, str] = {
    "performance": "Design may not meet response time or throughput requirements",
    "security": "Design may have security vulnerabilities or insufficient protection",
    "reliability": "Design may not handle failures gracefully or recover properly",
    "scalability": "Design may not handle increased load or growth effectively",
    "maintainability": "Design may be difficult to understand, test, or modify",
    "integration": "Design may have tight coupling or integration difficulties",
    "cost_efficiency": "Design may incur higher operational costs than necessary",
}

# Base mitigation strategies for each attribute (used as templates)
BASE_MITIGATION_STRATEGIES: dict[str, str] = {
    "performance": "Add caching, use async operations, optimize database queries, implement connection pooling",
    "security": "Add input validation, implement authentication/authorization, encrypt sensitive data, follow OWASP guidelines",
    "reliability": "Add retry logic with exponential backoff, implement circuit breakers, add health checks, use transactions",
    "scalability": "Design for horizontal scaling, make services stateless, use message queues, implement sharding",
    "maintainability": "Add comprehensive tests, improve documentation, use type hints, follow SOLID principles",
    "integration": "Use abstraction layers, implement adapter pattern, follow protocol standards, avoid vendor lock-in",
    "cost_efficiency": "Implement caching, use model tiering, batch operations, optimize resource usage",
}


def _generate_design_specific_mitigation(
    attribute: str,
    story: dict[str, Any],
    decisions: list[DesignDecision],
) -> str:
    """Generate a design-specific mitigation strategy.

    Combines base mitigation strategy with story and decision context
    to produce actionable, specific recommendations.

    Args:
        attribute: The quality attribute needing mitigation.
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        Design-specific mitigation string.
    """
    base_strategy = BASE_MITIGATION_STRATEGIES.get(
        attribute, f"Review and improve {attribute} aspects of design"
    )

    # Extract context from story and decisions
    story_title = story.get("title", "")
    story_desc = story.get("description", "")
    decision_types = [d.decision_type for d in decisions]

    # Build design-specific context
    context_parts = []

    if story_title:
        context_parts.append(f"For '{story_title}'")

    # Add specific recommendations based on detected patterns
    text = f"{story_title} {story_desc}".lower()

    if attribute == "performance":
        if "database" in text or "data" in decision_types:
            context_parts.append("optimize database queries with indexing")
        if "api" in text or "integration" in decision_types:
            context_parts.append("add response caching for API endpoints")
        if not any(kw in text for kw in ["async", "await"]):
            context_parts.append("consider async operations for I/O")

    elif attribute == "security":
        if "auth" in text or "user" in text:
            context_parts.append("ensure proper authentication flow")
        if "api" in text:
            context_parts.append("add input validation on all endpoints")
        if "security" in decision_types:
            context_parts.append("review security decision implementation")

    elif attribute == "reliability":
        if "external" in text or "api" in text or "integration" in decision_types:
            context_parts.append("add retry logic for external service calls")
        if not any(kw in text for kw in ["retry", "fallback"]):
            context_parts.append("implement fallback mechanisms")

    elif attribute == "scalability":
        if "session" in text or "state" in text:
            context_parts.append("externalize session state for horizontal scaling")
        if "database" in text or "data" in decision_types:
            context_parts.append("consider read replicas or caching layer")

    elif attribute == "maintainability":
        if not any(kw in text for kw in ["test", "document"]):
            context_parts.append("add unit tests and documentation")
        if "pattern" in decision_types:
            context_parts.append("document the pattern usage and rationale")

    elif attribute == "integration":
        if decisions:
            context_parts.append("ensure abstraction layers for external dependencies")
        if "technology" in decision_types:
            context_parts.append("add adapter pattern for technology choices")

    elif attribute == "cost_efficiency":
        if "llm" in text or "model" in text:
            context_parts.append("use model tiering (cheap for routine, smart for complex)")
        context_parts.append("implement caching to reduce redundant operations")

    # Combine base strategy with specific recommendations
    if context_parts:
        specific_recommendations = "; ".join(context_parts)
        return f"{base_strategy}. Specific to this design: {specific_recommendations}."

    return base_strategy


def _identify_risks(
    story: dict[str, Any],
    decisions: list[DesignDecision],
    attribute_scores: dict[str, float],
) -> list[QualityRisk]:
    """Identify risks based on quality attribute scores.

    Creates risks for attributes scoring below acceptable thresholds.
    Generates design-specific mitigation strategies based on story
    and decision context.

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.
        attribute_scores: Scores for each quality attribute.

    Returns:
        List of identified risks with design-specific mitigations.
    """
    risks: list[QualityRisk] = []

    for attr, score in attribute_scores.items():
        # Only create risks for scores below 0.7 threshold
        if score < 0.7:
            severity = _score_to_severity(score)
            effort = _effort_for_severity(severity)

            # Generate design-specific mitigation
            mitigation = _generate_design_specific_mitigation(attr, story, decisions)

            risk = QualityRisk(
                attribute=attr,
                description=RISK_DESCRIPTIONS.get(attr, f"Risk to {attr} quality attribute"),
                severity=severity,
                mitigation=mitigation,
                mitigation_effort=effort,
            )
            risks.append(risk)

    return risks


# =============================================================================
# LLM-Powered Evaluation
# =============================================================================

QUALITY_EVALUATION_PROMPT = """Evaluate the following design for quality attributes.

Story:
Title: {story_title}
Description: {story_description}

Design Decisions:
{decisions_text}

Evaluate against these quality attributes:
1. Performance: Response time, throughput, resource efficiency
2. Security: Authentication, authorization, data protection
3. Reliability: Fault tolerance, recovery, consistency
4. Scalability: Horizontal scaling, load handling
5. Maintainability: Code clarity, testability, documentation
6. Integration: Multi-provider support, protocol compliance, abstraction layers
7. Cost Efficiency: Model tiering, caching, token optimization

For each attribute, provide:
- Score (0.0-1.0): How well does the design address this attribute?
- Any risks or trade-offs

Respond in JSON format:
{{
  "attribute_scores": {{
    "performance": 0.8,
    "security": 0.7,
    "reliability": 0.75,
    "scalability": 0.65,
    "maintainability": 0.8,
    "integration": 0.7,
    "cost_efficiency": 0.75
  }},
  "trade_offs": [
    {{
      "attribute_a": "performance",
      "attribute_b": "security",
      "description": "Brief description of trade-off",
      "resolution": "How to balance both"
    }}
  ],
  "risks": [
    {{
      "attribute": "scalability",
      "description": "Risk description",
      "severity": "medium",
      "mitigation": "Suggested fix",
      "mitigation_effort": "low"
    }}
  ]
}}
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _call_quality_llm(prompt: str) -> str:
    """Call LLM for quality evaluation with retry logic.

    Args:
        prompt: The prompt to send to the LLM.

    Returns:
        LLM response text.

    Raises:
        Exception: If all retries fail.
    """
    import litellm

    model = os.environ.get("YOLO_LLM__ROUTINE_MODEL", LLM_CHEAP_MODEL_DEFAULT)

    logger.debug("calling_quality_llm", model=model, prompt_length=len(prompt))

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty response")

    return str(content)


async def _evaluate_quality_with_llm(
    story: dict[str, Any],
    decisions: list[DesignDecision],
) -> QualityAttributeEvaluation | None:
    """Evaluate quality attributes using LLM.

    Uses LLM to analyze design and provide quality scores,
    trade-offs, and risks.

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.

    Returns:
        QualityAttributeEvaluation if successful, None if LLM fails.
    """
    # Build prompt
    decisions_text = "\n".join(
        f"- {d.decision_type}: {d.description} (Rationale: {d.rationale})" for d in decisions
    )

    prompt = QUALITY_EVALUATION_PROMPT.format(
        story_title=story.get("title", "Unknown"),
        story_description=story.get("description", "No description"),
        decisions_text=decisions_text or "No design decisions provided",
    )

    try:
        response_text = await _call_quality_llm(prompt)

        # Parse JSON response
        # Handle markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        data = json.loads(response_text.strip())

        # Extract attribute scores
        attribute_scores = data.get("attribute_scores", {})
        if not attribute_scores:
            logger.warning("llm_response_missing_scores")
            return None

        # Extract trade-offs
        trade_offs_data = data.get("trade_offs", [])
        trade_offs = tuple(
            QualityTradeOff(
                attribute_a=t.get("attribute_a", "unknown"),
                attribute_b=t.get("attribute_b", "unknown"),
                description=t.get("description", ""),
                resolution=t.get("resolution", ""),
            )
            for t in trade_offs_data
        )

        # Extract risks
        risks_data = data.get("risks", [])
        risks = tuple(
            QualityRisk(
                attribute=r.get("attribute", "unknown"),
                description=r.get("description", ""),
                severity=r.get("severity", "medium"),
                mitigation=r.get("mitigation", ""),
                mitigation_effort=r.get("mitigation_effort", "medium"),
            )
            for r in risks_data
        )

        overall_score = _calculate_overall_score(attribute_scores)

        return QualityAttributeEvaluation(
            attribute_scores=attribute_scores,
            trade_offs=trade_offs,
            risks=risks,
            overall_score=overall_score,
        )

    except json.JSONDecodeError as e:
        logger.warning("llm_response_json_parse_error", error=str(e))
        return None
    except Exception as e:
        logger.warning("llm_evaluation_failed", error=str(e))
        return None


# =============================================================================
# Main Evaluation Function
# =============================================================================


async def evaluate_quality_attributes(
    story: dict[str, Any],
    decisions: list[DesignDecision],
    use_llm: bool = True,
) -> QualityAttributeEvaluation:
    """Evaluate quality attributes for a story design.

    Combines pattern-based scoring with optional LLM analysis to
    produce a comprehensive quality evaluation.

    Args:
        story: Story dictionary with title, description, etc.
        decisions: List of design decisions for this story.
        use_llm: Whether to attempt LLM-powered evaluation (default True).

    Returns:
        QualityAttributeEvaluation with scores, trade-offs, and risks.
    """
    story_id = story.get("id", "unknown")
    logger.info("quality_evaluation_start", story_id=story_id, decision_count=len(decisions))

    # Try LLM evaluation first if enabled
    if use_llm:
        llm_result = await _evaluate_quality_with_llm(story, decisions)
        if llm_result is not None:
            logger.info(
                "quality_evaluation_complete",
                story_id=story_id,
                method="llm",
                overall_score=llm_result.overall_score,
            )
            return llm_result

    # Fall back to pattern-based evaluation
    attribute_scores = {
        "performance": _score_performance(story, decisions),
        "security": _score_security(story, decisions),
        "reliability": _score_reliability(story, decisions),
        "scalability": _score_scalability(story, decisions),
        "maintainability": _score_maintainability(story, decisions),
        "integration": _score_integration(story, decisions),
        "cost_efficiency": _score_cost_efficiency(story, decisions),
    }

    # Detect trade-offs
    trade_offs = _detect_trade_offs(story, decisions, attribute_scores)

    # Identify risks
    risks = _identify_risks(story, decisions, attribute_scores)

    # Calculate overall score
    overall_score = _calculate_overall_score(attribute_scores)

    evaluation = QualityAttributeEvaluation(
        attribute_scores=attribute_scores,
        trade_offs=tuple(trade_offs),
        risks=tuple(risks),
        overall_score=overall_score,
    )

    logger.info(
        "quality_evaluation_complete",
        story_id=story_id,
        method="pattern",
        overall_score=overall_score,
        trade_off_count=len(trade_offs),
        risk_count=len(risks),
    )

    return evaluation
