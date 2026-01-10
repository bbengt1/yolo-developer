"""ADR (Architecture Decision Record) content generation (Story 7.3).

This module provides functions for generating comprehensive ADR content
including context, decision, consequences, and alternatives documentation.

Key Functions:
- _generate_adr_context: Creates context section with 12-Factor analysis
- _generate_adr_decision: Formats the decision and chosen approach
- _generate_adr_consequences: Documents positive/negative effects
- _document_alternatives: Lists alternatives with pros/cons
- _generate_adr_title: Creates descriptive ADR title
- generate_adr: Create a single ADR from a design decision
- generate_adrs: Create multiple ADRs from design decisions (async)
- _generate_adr_with_llm: LLM-powered ADR content generation
- _call_adr_llm: Low-level LLM call with retry

Example:
    >>> from yolo_developer.agents.architect.adr_generator import (
    ...     generate_adr,
    ...     generate_adrs,
    ... )
    >>>
    >>> adr = await generate_adr(decision, analysis)
    >>> adrs = await generate_adrs(decisions, analyses)

Architecture Note:
    Per ADR-001, all functions are designed for immutable data flow.
    Per ADR-003, LLM integration uses litellm with configurable model.
    Per ADR-007, retry with exponential backoff for LLM calls.
"""

from __future__ import annotations

import json
import os
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.architect.types import (
    ADR,
    DesignDecision,
    TwelveFactorAnalysis,
)

logger = structlog.get_logger(__name__)

# =============================================================================
# LLM Prompt Template
# =============================================================================

ADR_GENERATION_PROMPT = """Generate an Architecture Decision Record for the following design decision.

Design Decision:
- Type: {decision_type}
- Description: {description}
- Rationale: {rationale}
- Alternatives Considered: {alternatives}
- Story ID: {story_id}

12-Factor Analysis:
- Compliance: {compliance_percentage}%
- Applicable Factors: {applicable_factors}
- Recommendations: {recommendations}

Generate ADR content in JSON format with these exact fields:
{{
  "title": "Brief, descriptive title (5-10 words)",
  "context": "Why this decision was needed (2-3 sentences)",
  "decision": "What was decided and the chosen approach (2-3 sentences)",
  "consequences": "Positive effects, negative effects, trade-offs (2-4 sentences)"
}}

Respond ONLY with valid JSON, no additional text.
"""


def _generate_adr_context(
    decision: DesignDecision,
    analysis: TwelveFactorAnalysis | None,
) -> str:
    """Generate ADR context section with 12-Factor analysis.

    Creates the context explaining why this decision was needed,
    incorporating 12-Factor compliance information when available.

    Args:
        decision: The design decision to document.
        analysis: Optional 12-Factor analysis for the story.

    Returns:
        Formatted context string for the ADR.

    Example:
        >>> context = _generate_adr_context(decision, analysis)
        >>> "story-001" in context
        True
    """
    parts = []

    # Core context from decision
    parts.append(
        f"This decision was required for story {decision.story_id}: {decision.description}."
    )

    # Decision type context
    type_descriptions = {
        "technology": "A technology choice was needed",
        "pattern": "An architectural pattern selection was needed",
        "integration": "An integration approach was needed",
        "data": "A data architecture decision was needed",
        "security": "A security architecture decision was needed",
        "infrastructure": "An infrastructure decision was needed",
    }
    if decision.decision_type in type_descriptions:
        parts.append(f"{type_descriptions[decision.decision_type]} to proceed with implementation.")

    # 12-Factor compliance context
    if analysis is not None:
        compliance_pct = int(analysis.overall_compliance * 100)
        parts.append(f"12-Factor App compliance for this story: {compliance_pct}%.")

        if analysis.applicable_factors:
            factors_str = ", ".join(analysis.applicable_factors[:3])
            parts.append(f"Relevant factors: {factors_str}.")

    return " ".join(parts)


def _generate_adr_decision(decision: DesignDecision) -> str:
    """Generate ADR decision section.

    Formats the decision clearly stating what was chosen and why.

    Args:
        decision: The design decision to document.

    Returns:
        Formatted decision string for the ADR.

    Example:
        >>> adr_decision = _generate_adr_decision(decision)
        >>> len(adr_decision) > 10
        True
    """
    parts = []

    # What was decided
    parts.append(f"Selected approach: {decision.description}.")

    # Rationale
    parts.append(f"Rationale: {decision.rationale}.")

    # Decision type
    parts.append(f"Decision type: {decision.decision_type}.")

    return " ".join(parts)


def _generate_adr_consequences(
    decision: DesignDecision,
    analysis: TwelveFactorAnalysis | None,
) -> str:
    """Generate ADR consequences section with pros/cons.

    Documents positive and negative effects of the decision,
    including 12-Factor recommendations when compliance is < 100%.

    Args:
        decision: The design decision to document.
        analysis: Optional 12-Factor analysis for context.

    Returns:
        Formatted consequences string for the ADR.

    Example:
        >>> consequences = _generate_adr_consequences(decision, analysis)
        >>> "positive" in consequences.lower() or "pro" in consequences.lower()
        True
    """
    parts = []

    # Positive effects based on decision type
    positive_effects = _get_positive_effects(decision)
    parts.append(f"Positive: {positive_effects}")

    # Negative effects / trade-offs
    negative_effects = _get_negative_effects(decision)
    parts.append(f"Trade-offs: {negative_effects}")

    # 12-Factor recommendations if not fully compliant
    if analysis is not None and analysis.overall_compliance < 1.0:
        if analysis.recommendations:
            recommendations = ", ".join(analysis.recommendations[:2])
            parts.append(f"Recommendations: {recommendations}.")

    return " ".join(parts)


def _get_positive_effects(decision: DesignDecision) -> str:
    """Get positive effects based on decision type."""
    effects_map = {
        "technology": "Enables required functionality with proven technology stack.",
        "pattern": "Provides clear structure and maintainability through established patterns.",
        "integration": "Enables seamless communication between system components.",
        "data": "Ensures data integrity and efficient access patterns.",
        "security": "Strengthens security posture and compliance.",
        "infrastructure": "Improves deployment reliability and scalability.",
    }
    return effects_map.get(decision.decision_type, "Addresses the architectural need.")


def _get_negative_effects(decision: DesignDecision) -> str:
    """Get negative effects / trade-offs based on decision type."""
    effects_map = {
        "technology": "Adds complexity and potential learning curve for team.",
        "pattern": "May introduce initial development overhead for pattern implementation.",
        "integration": "Creates coupling between components that must be managed.",
        "data": "May impact performance for certain access patterns.",
        "security": "May add operational complexity for security management.",
        "infrastructure": "Adds operational complexity and potential cost.",
    }
    return effects_map.get(decision.decision_type, "Requires careful implementation and monitoring.")


def _document_alternatives(decision: DesignDecision) -> str:
    """Document alternatives considered with analysis.

    Lists all alternatives that were considered with brief
    pros/cons for each.

    Args:
        decision: The design decision with alternatives.

    Returns:
        Formatted alternatives documentation.

    Example:
        >>> doc = _document_alternatives(decision)
        >>> "MySQL" in doc if "MySQL" in decision.alternatives_considered else True
        True
    """
    if not decision.alternatives_considered:
        return "No alternatives were formally considered."

    parts = ["Alternatives considered:"]

    for alt in decision.alternatives_considered:
        # Generate brief analysis for each alternative
        analysis = _analyze_alternative(alt)
        parts.append(f"- {alt}: {analysis}")

    return " ".join(parts)


def _analyze_alternative(alternative: str) -> str:
    """Generate brief analysis for an alternative.

    Args:
        alternative: The alternative technology/approach name.

    Returns:
        Brief analysis string explaining why this alternative wasn't selected.
    """
    # Pattern-based analysis - in production, this could be LLM-powered
    alt_lower = alternative.lower()

    # Database alternatives
    if any(db in alt_lower for db in ["mysql", "mariadb"]):
        return "Good compatibility but less feature-rich for complex queries."
    if any(db in alt_lower for db in ["mongodb", "dynamodb", "nosql"]):
        return "Flexible schema but eventual consistency trade-offs."
    if any(db in alt_lower for db in ["sqlite"]):
        return "Simple but limited scalability."
    if any(db in alt_lower for db in ["postgresql", "postgres"]):
        return "Strong ACID compliance but higher operational complexity."

    # Framework alternatives
    if any(fw in alt_lower for fw in ["django", "flask", "fastapi"]):
        return "Viable Python framework with different trade-offs."

    # Cache alternatives
    if any(cache in alt_lower for cache in ["redis", "memcached"]):
        return "Effective caching solution with specific performance characteristics."

    # Message queue alternatives
    if any(mq in alt_lower for mq in ["rabbitmq", "kafka", "sqs"]):
        return "Messaging solution with different delivery guarantees."

    # Generic fallback with more context
    return "Evaluated but not selected; chosen approach better fits project constraints."


def _generate_adr_title(decision: DesignDecision) -> str:
    """Generate descriptive ADR title.

    Creates a clear, descriptive title for the ADR based on
    the decision description and type.

    Args:
        decision: The design decision to title.

    Returns:
        Descriptive title string.

    Example:
        >>> title = _generate_adr_title(decision)
        >>> len(title) > 5
        True
    """
    # Extract key terms from description
    description = decision.description

    # Use decision type as prefix
    type_prefixes = {
        "technology": "Use",
        "pattern": "Apply",
        "integration": "Integrate",
        "data": "Design",
        "security": "Implement",
        "infrastructure": "Deploy",
    }
    prefix = type_prefixes.get(decision.decision_type, "Decide")

    # Clean up description for title
    # Remove common prefixes like "Use", "Implement" if already present
    clean_desc = description
    for p in ["use ", "implement ", "apply ", "design ", "deploy "]:
        if clean_desc.lower().startswith(p):
            clean_desc = clean_desc[len(p):]
            break

    # Capitalize first letter
    if clean_desc:
        clean_desc = clean_desc[0].upper() + clean_desc[1:]

    return f"{prefix} {clean_desc}"


# =============================================================================
# LLM Integration Functions
# =============================================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _call_adr_llm(prompt: str) -> dict[str, Any]:
    """Call LLM for ADR content generation with retry.

    Uses litellm for provider abstraction and tenacity for retry logic.

    Args:
        prompt: The formatted prompt for ADR generation.

    Returns:
        Parsed JSON response from LLM.

    Raises:
        Exception: If LLM call fails after retries.
    """
    import litellm

    logger.debug("adr_llm_call_start", prompt_length=len(prompt))

    # Use configurable model via environment variable
    model = os.environ.get("YOLO_LLM__ROUTINE_MODEL", "gpt-4o-mini")

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    if content is None:
        raise ValueError("LLM returned empty content")

    result: dict[str, Any] = json.loads(content)

    logger.debug("adr_llm_call_complete", response_keys=list(result.keys()))

    return result


async def _generate_adr_with_llm(
    decision: DesignDecision,
    analysis: TwelveFactorAnalysis | None,
) -> dict[str, str]:
    """Generate ADR content using LLM.

    Falls back to pattern-based generation on LLM failure.

    Args:
        decision: The design decision to document.
        analysis: Optional 12-Factor analysis for context.

    Returns:
        Dict with title, context, decision, consequences keys.
    """
    # Build prompt
    alternatives = ", ".join(decision.alternatives_considered) if decision.alternatives_considered else "None"
    compliance_pct = int(analysis.overall_compliance * 100) if analysis else 100
    applicable = ", ".join(analysis.applicable_factors) if analysis and analysis.applicable_factors else "None"
    recommendations = ", ".join(analysis.recommendations) if analysis and analysis.recommendations else "None"

    prompt = ADR_GENERATION_PROMPT.format(
        decision_type=decision.decision_type,
        description=decision.description,
        rationale=decision.rationale,
        alternatives=alternatives,
        story_id=decision.story_id,
        compliance_percentage=compliance_pct,
        applicable_factors=applicable,
        recommendations=recommendations,
    )

    try:
        result = await _call_adr_llm(prompt)
        logger.info("adr_llm_generation_success", decision_id=decision.id)
        return {
            "title": result.get("title", _generate_adr_title(decision)),
            "context": result.get("context", _generate_adr_context(decision, analysis)),
            "decision": result.get("decision", _generate_adr_decision(decision)),
            "consequences": result.get("consequences", _generate_adr_consequences(decision, analysis)),
        }
    except Exception as e:
        logger.warning(
            "adr_llm_generation_fallback",
            decision_id=decision.id,
            error=str(e),
        )
        # Fallback to pattern-based generation
        return {
            "title": _generate_adr_title(decision),
            "context": _generate_adr_context(decision, analysis),
            "decision": _generate_adr_decision(decision),
            "consequences": _generate_adr_consequences(decision, analysis),
        }


# =============================================================================
# Main ADR Generation Functions
# =============================================================================


async def generate_adr(
    decision: DesignDecision,
    analysis: TwelveFactorAnalysis | None,
    adr_number: int = 1,
    additional_story_ids: tuple[str, ...] = (),
    use_llm: bool = False,
) -> ADR:
    """Generate a single ADR from a design decision.

    Creates a complete ADR with all sections populated from the
    design decision and optional 12-Factor analysis.

    Args:
        decision: The design decision to document.
        analysis: Optional 12-Factor analysis for context enrichment.
        adr_number: The sequential ADR number for ID generation.
        additional_story_ids: Additional story IDs to link (beyond decision.story_id).
        use_llm: Whether to use LLM for content generation.

    Returns:
        Frozen ADR dataclass instance.

    Example:
        >>> adr = await generate_adr(decision, analysis, adr_number=1)
        >>> adr.id
        'ADR-001'
    """
    # Generate content (LLM or pattern-based)
    if use_llm:
        content = await _generate_adr_with_llm(decision, analysis)
    else:
        content = {
            "title": _generate_adr_title(decision),
            "context": _generate_adr_context(decision, analysis),
            "decision": _generate_adr_decision(decision),
            "consequences": _generate_adr_consequences(decision, analysis),
        }

    # Add alternatives to consequences
    alternatives_doc = _document_alternatives(decision)
    full_consequences = f"{content['consequences']} {alternatives_doc}"

    # Build story IDs tuple
    story_ids = (decision.story_id, *additional_story_ids)

    # Create immutable ADR
    adr = ADR(
        id=f"ADR-{adr_number:03d}",
        title=content["title"],
        status="proposed",
        context=content["context"],
        decision=content["decision"],
        consequences=full_consequences,
        story_ids=story_ids,
    )

    logger.info(
        "adr_generated",
        adr_id=adr.id,
        story_ids=story_ids,
        decision_type=decision.decision_type,
    )

    return adr


async def generate_adrs(
    decisions: list[DesignDecision],
    twelve_factor_analyses: dict[str, TwelveFactorAnalysis | dict[str, Any]],
    use_llm: bool = False,
) -> list[ADR]:
    """Generate ADRs for a list of design decisions.

    Creates ADRs for significant decisions (technology and pattern types).

    Args:
        decisions: List of design decisions to process.
        twelve_factor_analyses: Dict mapping story IDs to 12-Factor analyses.
        use_llm: Whether to use LLM for content generation.

    Returns:
        List of ADR objects.

    Example:
        >>> adrs = await generate_adrs(decisions, analyses)
        >>> len(adrs)
        2
    """
    if not decisions:
        logger.debug("no_decisions_for_adrs")
        return []

    adrs: list[ADR] = []
    adr_counter = 0

    # Generate ADRs for significant decisions
    significant_types = {"technology", "pattern"}

    for decision in decisions:
        if decision.decision_type in significant_types:
            adr_counter += 1

            # Get analysis for this story
            analysis_data = twelve_factor_analyses.get(decision.story_id)

            # Convert dict to TwelveFactorAnalysis if needed
            if isinstance(analysis_data, dict):
                # It's a serialized dict, create a minimal analysis
                analysis = TwelveFactorAnalysis(
                    factor_results={},
                    applicable_factors=tuple(analysis_data.get("applicable_factors", [])),
                    overall_compliance=analysis_data.get("overall_compliance", 1.0),
                    recommendations=tuple(analysis_data.get("recommendations", [])),
                )
            elif analysis_data is None:
                analysis = None
            else:
                analysis = analysis_data

            adr = await generate_adr(
                decision=decision,
                analysis=analysis,
                adr_number=adr_counter,
                use_llm=use_llm,
            )
            adrs.append(adr)

    logger.info(
        "adrs_generated",
        count=len(adrs),
        adr_ids=[a.id for a in adrs],
    )

    return adrs
