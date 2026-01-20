"""ATAM (Architecture Tradeoff Analysis Method) Review module (Story 7.7).

This module provides ATAM-style architectural review functionality:

- Generate quality attribute scenarios from design decisions
- Detect trade-off conflicts between quality attributes
- Assess risk impact on quality attributes
- Make pass/fail decisions based on configurable thresholds
- LLM-powered analysis with rule-based fallback

Example:
    >>> from yolo_developer.agents.architect.atam_reviewer import (
    ...     run_atam_review,
    ... )
    >>>
    >>> result = await run_atam_review(design_decisions)
    >>> result.overall_pass
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

import litellm
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.architect.types import (
    ATAMReviewResult,
    ATAMRiskAssessment,
    ATAMScenario,
    ATAMTradeOffConflict,
    DesignDecision,
    MitigationFeasibility,
    QualityAttributeEvaluation,
    QualityTradeOff,
    RiskSeverity,
    TechnicalRiskReport,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Quality Attribute Constants
# =============================================================================

QUALITY_ATTRIBUTE_STIMULI: dict[str, tuple[str, str]] = {
    "performance": ("High load of concurrent requests", "Response time meets SLA"),
    "security": ("Unauthorized access attempt", "Access denied with proper audit"),
    "reliability": ("Component failure during operation", "Graceful degradation with recovery"),
    "scalability": ("Traffic spike of 10x normal load", "System scales horizontally"),
    "maintainability": ("New feature request", "Implementation without major refactoring"),
    "integration": ("External service unavailable", "Fallback behavior activates"),
    "cost_efficiency": ("Resource usage during peak", "Costs within budget constraints"),
}


# =============================================================================
# Scenario Generation (Task 2)
# =============================================================================


def _generate_atam_scenarios(
    decisions: list[DesignDecision],
    quality_eval: dict[str, Any] | QualityAttributeEvaluation | None,
) -> list[ATAMScenario]:
    """Generate ATAM quality attribute scenarios from design decisions.

    Creates scenarios that express how the system should respond to stimuli
    for quality attributes affected by design decisions.

    Args:
        decisions: Design decisions to analyze.
        quality_eval: Quality evaluation results (dict or dataclass).

    Returns:
        List of ATAM scenarios for evaluation.
    """
    if not decisions:
        logger.debug("no_decisions_for_scenarios")
        return []

    scenarios: list[ATAMScenario] = []
    scenario_counter = 0

    # Extract attribute scores from quality eval
    attribute_scores: dict[str, float] = {}
    if quality_eval:
        if isinstance(quality_eval, dict):
            attribute_scores = quality_eval.get("attribute_scores", {})
        elif hasattr(quality_eval, "attribute_scores"):
            attribute_scores = dict(quality_eval.attribute_scores)

    # Map decision types to quality attributes
    decision_type_to_attributes: dict[str, list[str]] = {
        "pattern": ["maintainability", "reliability"],
        "technology": ["maintainability", "integration"],
        "security": ["security", "reliability"],
        "data": ["performance", "scalability", "reliability"],
        "infrastructure": ["scalability", "reliability", "cost_efficiency"],
        "integration": ["integration", "reliability"],
    }

    # Track attributes covered
    covered_attributes: set[str] = set()

    for decision in decisions:
        # Get attributes affected by this decision type
        affected_attrs = decision_type_to_attributes.get(
            decision.decision_type, ["maintainability"]
        )

        for attr in affected_attrs:
            if attr in covered_attributes:
                continue  # Avoid duplicate scenarios

            scenario_counter += 1
            covered_attributes.add(attr)

            # Get stimulus/response from templates
            stimulus, response = QUALITY_ATTRIBUTE_STIMULI.get(
                attr, ("Standard operation", "Expected behavior")
            )

            # Generate analysis based on decision
            score = attribute_scores.get(attr, 0.7)
            analysis = _generate_scenario_analysis(decision, attr, score)

            scenarios.append(
                ATAMScenario(
                    scenario_id=f"ATAM-{scenario_counter:03d}",
                    quality_attribute=attr,
                    stimulus=stimulus,
                    response=response,
                    analysis=analysis,
                )
            )

    logger.info(
        "atam_scenarios_generated",
        scenario_count=len(scenarios),
        attributes_covered=list(covered_attributes),
    )

    return scenarios


def _generate_scenario_analysis(decision: DesignDecision, attribute: str, score: float) -> str:
    """Generate analysis text for a scenario based on design decision.

    Args:
        decision: The design decision being analyzed.
        attribute: Quality attribute being evaluated.
        score: Quality score from evaluation (0.0-1.0).

    Returns:
        Analysis text describing how the decision addresses the attribute.
    """
    score_descriptor = (
        "excellent" if score >= 0.8 else "adequate" if score >= 0.6 else "needs improvement"
    )

    return (
        f"Decision '{decision.description}' provides {score_descriptor} support "
        f"for {attribute}. Rationale: {decision.rationale}"
    )


# =============================================================================
# Trade-Off Conflict Detection (Task 3)
# =============================================================================


def _detect_trade_off_conflicts(
    trade_offs: list[QualityTradeOff] | tuple[QualityTradeOff, ...],
    _decisions: list[DesignDecision],
) -> list[ATAMTradeOffConflict]:
    """Detect conflicts between quality attribute trade-offs.

    Identifies situations where trade-off resolutions are in conflict
    (e.g., A improves at expense of B, while B improves at expense of A).

    Args:
        trade_offs: Trade-offs from quality evaluation.
        _decisions: Design decisions for context (reserved for future use).

    Returns:
        List of detected conflicts.
    """
    if len(trade_offs) < 2:
        logger.debug("insufficient_trade_offs_for_conflict", count=len(trade_offs))
        return []

    conflicts: list[ATAMTradeOffConflict] = []

    # Build a map of trade-offs: attr_a -> [(attr_b, resolution)]
    # When attribute_a improves, it impacts attribute_b
    improvement_map: dict[str, list[tuple[str, str]]] = {}  # attr -> [(impacted, resolution)]

    for trade_off in trade_offs:
        attr_a = trade_off.attribute_a
        attr_b = trade_off.attribute_b
        if attr_a not in improvement_map:
            improvement_map[attr_a] = []
        improvement_map[attr_a].append((attr_b, trade_off.resolution))

    # Detect circular conflicts (A improves -> impacts B, B improves -> impacts A)
    for attr_a, impacts_a in improvement_map.items():
        for impacted_by_a, resolution_a in impacts_a:
            # Check if the impacted attribute also improves at expense of attr_a
            if impacted_by_a in improvement_map:
                for impacted_by_b, resolution_b in improvement_map[impacted_by_a]:
                    if impacted_by_b == attr_a:
                        # Found circular conflict - default to medium severity
                        combined_severity: RiskSeverity = "medium"
                        resolution = _generate_resolution_strategy(
                            attr_a, impacted_by_a, resolution_a, resolution_b
                        )

                        conflict = ATAMTradeOffConflict(
                            attribute_a=attr_a,
                            attribute_b=impacted_by_a,
                            description=f"Circular trade-off: {attr_a} and {impacted_by_a} "
                            f"mutually impact each other",
                            severity=combined_severity,
                            resolution_strategy=resolution,
                        )

                        # Avoid duplicate conflicts (A-B same as B-A)
                        existing = {(c.attribute_a, c.attribute_b) for c in conflicts} | {
                            (c.attribute_b, c.attribute_a) for c in conflicts
                        }
                        if (attr_a, impacted_by_a) not in existing:
                            conflicts.append(conflict)

    logger.info("trade_off_conflicts_detected", conflict_count=len(conflicts))

    return conflicts


def _generate_resolution_strategy(
    attr_a: str, attr_b: str, mitigation_a: str, mitigation_b: str
) -> str:
    """Generate a resolution strategy for a trade-off conflict.

    Args:
        attr_a: First attribute in conflict.
        attr_b: Second attribute in conflict.
        mitigation_a: Mitigation for first trade-off.
        mitigation_b: Mitigation for second trade-off.

    Returns:
        Resolution strategy text.
    """
    if mitigation_a and mitigation_b:
        return (
            f"Combine mitigations: '{mitigation_a}' with '{mitigation_b}'. "
            f"Prioritize based on business requirements for {attr_a} vs {attr_b}."
        )
    elif mitigation_a:
        return f"Apply mitigation: {mitigation_a}. Prioritize {attr_a} requirements."
    elif mitigation_b:
        return f"Apply mitigation: {mitigation_b}. Prioritize {attr_b} requirements."
    else:
        return (
            f"No existing mitigations. Consider architectural compromise between "
            f"{attr_a} and {attr_b} based on business priorities."
        )


# =============================================================================
# Risk Impact Assessment (Task 4)
# =============================================================================


# Mapping from risk categories to quality attributes
RISK_CATEGORY_TO_ATTRIBUTES: dict[str, list[str]] = {
    "technology": ["maintainability", "reliability"],
    "integration": ["integration", "reliability"],
    "scalability": ["scalability", "performance"],
    "compatibility": ["maintainability", "integration"],
    "operational": ["reliability", "maintainability"],
}


def _assess_risk_impact(
    risk_report: TechnicalRiskReport | None,
    quality_eval: dict[str, Any] | QualityAttributeEvaluation | None,
) -> list[ATAMRiskAssessment]:
    """Assess the impact of technical risks on quality attributes.

    Maps risks to affected quality attributes and evaluates
    mitigation feasibility.

    Args:
        risk_report: Technical risk report from risk identification.
        quality_eval: Quality evaluation for context.

    Returns:
        List of risk assessments.
    """
    if risk_report is None or not risk_report.risks:
        logger.debug("no_risks_to_assess")
        return []

    assessments: list[ATAMRiskAssessment] = []

    for risk in risk_report.risks:
        # Map risk category to quality attributes
        affected_attrs = RISK_CATEGORY_TO_ATTRIBUTES.get(risk.category, ["reliability"])

        # Evaluate mitigation feasibility based on effort
        feasibility = _evaluate_mitigation_feasibility(
            risk.mitigation, risk.mitigation_effort, risk.severity
        )

        # Determine if risk is unmitigated
        unmitigated = _is_risk_unmitigated(risk.mitigation, risk.severity, feasibility)

        assessment = ATAMRiskAssessment(
            risk_id=f"RISK-{risk.category.upper()[:3]}-{hash(risk.description) % 1000:03d}",
            quality_impact=tuple(affected_attrs),
            mitigation_feasibility=feasibility,
            unmitigated=unmitigated,
        )
        assessments.append(assessment)

    logger.info(
        "risk_impact_assessed",
        assessment_count=len(assessments),
        unmitigated_count=sum(1 for a in assessments if a.unmitigated),
    )

    return assessments


def _evaluate_mitigation_feasibility(
    mitigation: str, effort: str, severity: str
) -> MitigationFeasibility:
    """Evaluate feasibility of implementing a risk mitigation.

    Args:
        mitigation: Mitigation strategy text.
        effort: Mitigation effort level (high/medium/low).
        severity: Risk severity.

    Returns:
        Feasibility level (high/medium/low).
    """
    # No mitigation = low feasibility
    if not mitigation or mitigation.strip() == "":
        return "low"

    # Map effort to feasibility (inverse relationship)
    effort_to_feasibility: dict[str, MitigationFeasibility] = {
        "low": "high",
        "medium": "medium",
        "high": "low",
    }

    return effort_to_feasibility.get(effort, "medium")


def _is_risk_unmitigated(
    mitigation: str, severity: str, feasibility: MitigationFeasibility
) -> bool:
    """Determine if a risk should be considered unmitigated.

    Args:
        mitigation: Mitigation strategy text.
        severity: Risk severity.
        feasibility: Mitigation feasibility.

    Returns:
        True if risk is unmitigated, False otherwise.
    """
    # No mitigation = unmitigated
    if not mitigation or mitigation.strip() == "":
        return True

    # Critical severity with low feasibility = effectively unmitigated
    if severity == "critical" and feasibility == "low":
        return True

    return False


# =============================================================================
# Pass/Fail Decision Logic (Task 5)
# =============================================================================


# Review thresholds (could be made configurable via YoloConfig)
MIN_CONFIDENCE_THRESHOLD = 0.6
MAX_HIGH_CONFLICTS = 2
QUALITY_ATTRIBUTES_COUNT = 7  # Total quality attributes for coverage calculation


def _make_review_decision(
    scenarios: list[ATAMScenario],
    conflicts: list[ATAMTradeOffConflict],
    risk_assessments: list[ATAMRiskAssessment],
) -> tuple[bool, float, list[str]]:
    """Make the ATAM review pass/fail decision.

    Evaluates scenarios, conflicts, and risk assessments against
    configurable thresholds to determine if the design passes.

    Args:
        scenarios: Evaluated ATAM scenarios.
        conflicts: Detected trade-off conflicts.
        risk_assessments: Risk impact assessments.

    Returns:
        Tuple of (pass/fail, confidence score, list of failure reasons).
    """
    failure_reasons: list[str] = []

    # Check for critical unmitigated risks
    unmitigated_critical = [a for a in risk_assessments if a.unmitigated]
    if unmitigated_critical:
        failure_reasons.append(
            f"Critical unmitigated risks: {len(unmitigated_critical)} risks lack feasible mitigations"
        )

    # Check for too many high-severity conflicts
    high_conflicts = [c for c in conflicts if c.severity in ("critical", "high")]
    if len(high_conflicts) > MAX_HIGH_CONFLICTS:
        failure_reasons.append(
            f"Too many high-severity conflicts: {len(high_conflicts)} (max: {MAX_HIGH_CONFLICTS})"
        )

    # Calculate confidence score
    confidence = _calculate_confidence(scenarios, conflicts, risk_assessments)

    # Check confidence threshold
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        failure_reasons.append(
            f"Confidence below threshold: {confidence:.2f} < {MIN_CONFIDENCE_THRESHOLD}"
        )

    # Determine overall pass/fail
    passed = len(failure_reasons) == 0

    logger.info(
        "atam_review_decision",
        passed=passed,
        confidence=confidence,
        failure_count=len(failure_reasons),
    )

    return passed, confidence, failure_reasons


def _calculate_confidence(
    scenarios: list[ATAMScenario],
    conflicts: list[ATAMTradeOffConflict],
    risk_assessments: list[ATAMRiskAssessment],
) -> float:
    """Calculate overall confidence score for the review.

    Confidence is based on:
    - Scenario coverage (how many quality attributes have scenarios)
    - Number of conflicts (fewer = better)
    - Risk mitigation status (more mitigated = better)

    Args:
        scenarios: Evaluated scenarios.
        conflicts: Detected conflicts.
        risk_assessments: Risk assessments.

    Returns:
        Confidence score from 0.0 to 1.0.
    """
    # Scenario coverage component (0.0 - 0.5)
    unique_attributes = len({s.quality_attribute for s in scenarios})
    coverage = min(unique_attributes / QUALITY_ATTRIBUTES_COUNT, 1.0)
    coverage_score = coverage * 0.5

    # Conflict penalty (0.0 - 0.25)
    conflict_penalty = min(len(conflicts) * 0.08, 0.25)
    conflict_score = 0.25 - conflict_penalty

    # Risk mitigation component (0.0 - 0.25)
    if risk_assessments:
        mitigated_ratio = sum(1 for a in risk_assessments if not a.unmitigated) / len(
            risk_assessments
        )
    else:
        mitigated_ratio = 1.0  # No risks = fully mitigated
    risk_score = mitigated_ratio * 0.25

    confidence = coverage_score + conflict_score + risk_score

    # Ensure within bounds
    return max(0.0, min(1.0, confidence))


# =============================================================================
# LLM-Powered Analysis (Task 6)
# =============================================================================


ATAM_REVIEW_PROMPT = """Perform an ATAM-style architectural review of this design.

Design Decisions:
{design_decisions}

Quality Attribute Evaluation:
{quality_eval}

Technical Risk Report:
{risk_report}

Analyze:
1. Create architectural scenarios for quality attributes affected by the design
2. Identify trade-off conflicts where improvements in one attribute harm another
3. Assess risk impact on quality attributes and mitigation feasibility
4. Provide an overall pass/fail recommendation with confidence

Response Requirements:
- overall_pass: true if design is sound, false if critical issues exist
- confidence: 0.0 to 1.0 based on analysis thoroughness
- scenarios_evaluated: quality attribute scenarios with stimulus/response
- trade_off_conflicts: conflicts between attribute optimizations
- risk_assessments: how risks affect quality attributes
- failure_reasons: specific reasons if review fails (empty if passes)
- summary: brief summary of review outcome

Respond in JSON format:
{{
  "overall_pass": true,
  "confidence": 0.85,
  "scenarios_evaluated": [
    {{
      "scenario_id": "ATAM-001",
      "quality_attribute": "performance",
      "stimulus": "100 concurrent API requests",
      "response": "95th percentile < 500ms",
      "analysis": "Design supports async processing"
    }}
  ],
  "trade_off_conflicts": [
    {{
      "attribute_a": "performance",
      "attribute_b": "security",
      "description": "Encryption adds latency",
      "severity": "medium",
      "resolution_strategy": "Use async encryption"
    }}
  ],
  "risk_assessments": [
    {{
      "risk_id": "RISK-001",
      "quality_impact": ["reliability"],
      "mitigation_feasibility": "high",
      "unmitigated": false
    }}
  ],
  "failure_reasons": [],
  "summary": "Design passes ATAM review"
}}
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _call_atam_llm(prompt: str) -> str:
    """Call LLM for ATAM analysis with retry logic.

    Uses YOLO_LLM__ROUTINE_MODEL env var for model selection,
    following the pattern from prior architect stories.

    Args:
        prompt: The ATAM review prompt.

    Returns:
        LLM response text.

    Raises:
        Exception: If all retries fail.
    """
    model = os.environ.get("YOLO_LLM__ROUTINE_MODEL", "gpt-4o-mini")

    logger.debug("calling_atam_llm", model=model, prompt_length=len(prompt))

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty response")

    return str(content)


async def _analyze_atam_with_llm(
    decisions: list[DesignDecision],
    quality_eval: dict[str, Any] | QualityAttributeEvaluation | None,
    risk_report: TechnicalRiskReport | None,
) -> ATAMReviewResult | None:
    """Perform ATAM analysis using LLM.

    Args:
        decisions: Design decisions to review.
        quality_eval: Quality evaluation results.
        risk_report: Technical risk report.

    Returns:
        ATAMReviewResult if successful, None if LLM fails.
    """
    # Format inputs for prompt
    decisions_text = "\n".join(
        f"- {d.decision_type}: {d.description} (Rationale: {d.rationale})" for d in decisions
    )

    quality_text = "Not provided"
    if quality_eval:
        if isinstance(quality_eval, dict):
            quality_text = json.dumps(quality_eval, indent=2)
        elif hasattr(quality_eval, "to_dict"):
            quality_text = json.dumps(quality_eval.to_dict(), indent=2)

    risk_text = "No risks identified"
    if risk_report:
        risk_text = json.dumps(risk_report.to_dict(), indent=2)

    prompt = ATAM_REVIEW_PROMPT.format(
        design_decisions=decisions_text or "No design decisions provided",
        quality_eval=quality_text,
        risk_report=risk_text,
    )

    try:
        response_text = await _call_atam_llm(prompt)

        # Parse JSON response - handle code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if json_match:
            response_text = json_match.group(1)

        data = json.loads(response_text.strip())

        # Parse scenarios
        scenarios_data = data.get("scenarios_evaluated", [])
        scenarios = tuple(
            ATAMScenario(
                scenario_id=s.get("scenario_id", "ATAM-???"),
                quality_attribute=s.get("quality_attribute", "unknown"),
                stimulus=s.get("stimulus", ""),
                response=s.get("response", ""),
                analysis=s.get("analysis", ""),
            )
            for s in scenarios_data
        )

        # Parse conflicts
        conflicts_data = data.get("trade_off_conflicts", [])
        conflicts = tuple(
            ATAMTradeOffConflict(
                attribute_a=c.get("attribute_a", ""),
                attribute_b=c.get("attribute_b", ""),
                description=c.get("description", ""),
                severity=_validate_severity(c.get("severity", "medium")),
                resolution_strategy=c.get("resolution_strategy", ""),
            )
            for c in conflicts_data
        )

        # Parse risk assessments
        assessments_data = data.get("risk_assessments", [])
        assessments = tuple(
            ATAMRiskAssessment(
                risk_id=r.get("risk_id", "RISK-???"),
                quality_impact=tuple(r.get("quality_impact", [])),
                mitigation_feasibility=_validate_feasibility(
                    r.get("mitigation_feasibility", "medium")
                ),
                unmitigated=r.get("unmitigated", False),
            )
            for r in assessments_data
        )

        return ATAMReviewResult(
            overall_pass=data.get("overall_pass", False),
            confidence=float(data.get("confidence", 0.5)),
            scenarios_evaluated=scenarios,
            trade_off_conflicts=conflicts,
            risk_assessments=assessments,
            failure_reasons=tuple(data.get("failure_reasons", [])),
            summary=data.get("summary", "LLM-generated ATAM review"),
        )

    except json.JSONDecodeError as e:
        logger.warning("atam_llm_json_parse_error", error=str(e))
        return None
    except Exception as e:
        logger.warning("atam_llm_analysis_failed", error=str(e))
        return None


def _validate_severity(severity: str) -> RiskSeverity:
    """Validate and return a valid severity value."""
    if severity in ("critical", "high", "medium", "low"):
        return severity  # type: ignore[return-value]
    return "medium"


def _validate_feasibility(feasibility: str) -> MitigationFeasibility:
    """Validate and return a valid feasibility value."""
    if feasibility in ("high", "medium", "low"):
        return feasibility  # type: ignore[return-value]
    return "medium"


# =============================================================================
# Main Review Function (Task 7)
# =============================================================================


async def run_atam_review(
    decisions: list[DesignDecision],
    quality_eval: QualityAttributeEvaluation | None = None,
    risk_report: TechnicalRiskReport | None = None,
    use_llm: bool = True,
) -> ATAMReviewResult:
    """Run ATAM architectural review on design decisions.

    Main entry point for ATAM review. Orchestrates scenario generation,
    conflict detection, risk assessment, and pass/fail decision.

    Args:
        decisions: Design decisions to review.
        quality_eval: Optional quality evaluation from Story 7.4.
        risk_report: Optional risk report from Story 7.5.
        use_llm: Whether to attempt LLM-powered analysis (default True).

    Returns:
        ATAMReviewResult with review outcome.
    """
    logger.info(
        "atam_review_start",
        decision_count=len(decisions),
        has_quality_eval=quality_eval is not None,
        has_risk_report=risk_report is not None,
    )

    # Convert quality_eval to dict if needed for rule-based analysis
    quality_dict: dict[str, Any] | None = None
    if quality_eval:
        if hasattr(quality_eval, "to_dict"):
            quality_dict = quality_eval.to_dict()
        elif isinstance(quality_eval, dict):
            quality_dict = quality_eval

    # Try LLM analysis first
    if use_llm:
        llm_result = await _analyze_atam_with_llm(decisions, quality_dict, risk_report)
        if llm_result is not None:
            logger.info(
                "atam_review_complete",
                method="llm",
                passed=llm_result.overall_pass,
                confidence=llm_result.confidence,
            )
            return llm_result

    # Fall back to rule-based analysis
    logger.debug("atam_fallback_to_rule_based")

    # Generate scenarios
    scenarios = _generate_atam_scenarios(decisions, quality_dict)

    # Detect conflicts
    trade_offs: list[QualityTradeOff] = []
    if quality_eval and hasattr(quality_eval, "trade_offs"):
        trade_offs = list(quality_eval.trade_offs)
    elif quality_dict and "trade_offs" in quality_dict:
        # Convert dict trade-offs back to objects if needed
        pass  # trade_offs stays empty if not proper objects

    conflicts = _detect_trade_off_conflicts(trade_offs, decisions)

    # Assess risk impact
    risk_assessments = _assess_risk_impact(risk_report, quality_dict)

    # Make pass/fail decision
    passed, confidence, failure_reasons = _make_review_decision(
        scenarios, conflicts, risk_assessments
    )

    # Generate summary
    summary = _generate_summary(passed, confidence, scenarios, conflicts, risk_assessments)

    result = ATAMReviewResult(
        overall_pass=passed,
        confidence=confidence,
        scenarios_evaluated=tuple(scenarios),
        trade_off_conflicts=tuple(conflicts),
        risk_assessments=tuple(risk_assessments),
        failure_reasons=tuple(failure_reasons),
        summary=summary,
    )

    logger.info(
        "atam_review_complete",
        method="rule_based",
        passed=passed,
        confidence=confidence,
        scenario_count=len(scenarios),
        conflict_count=len(conflicts),
        risk_count=len(risk_assessments),
    )

    return result


def _generate_summary(
    passed: bool,
    confidence: float,
    scenarios: list[ATAMScenario],
    conflicts: list[ATAMTradeOffConflict],
    risk_assessments: list[ATAMRiskAssessment],
) -> str:
    """Generate a summary of the ATAM review.

    Args:
        passed: Whether review passed.
        confidence: Confidence score.
        scenarios: Evaluated scenarios.
        conflicts: Detected conflicts.
        risk_assessments: Risk assessments.

    Returns:
        Summary text.
    """
    status = "passes" if passed else "fails"
    pct = int(confidence * 100)

    parts = [f"Design {status} ATAM review with {pct}% confidence."]

    if scenarios:
        attrs = ", ".join(sorted({s.quality_attribute for s in scenarios}))
        parts.append(f"Evaluated: {attrs}.")

    if conflicts:
        parts.append(f"{len(conflicts)} trade-off conflict(s) identified.")

    unmitigated = sum(1 for a in risk_assessments if a.unmitigated)
    if unmitigated:
        parts.append(f"{unmitigated} unmitigated risk(s) require attention.")

    return " ".join(parts)
