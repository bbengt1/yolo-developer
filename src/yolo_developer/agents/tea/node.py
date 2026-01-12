"""TEA agent node for LangGraph orchestration (Story 9.1, 9.3).

This module provides the tea_node function that integrates with the
LangGraph orchestration workflow. The TEA (Test Engineering and Assurance)
agent validates implementation artifacts and calculates deployment confidence.

Key Concepts:
- **YoloState Input**: Receives state as TypedDict, not Pydantic
- **Immutable Updates**: Returns state update dict, never mutates input
- **Async I/O**: All LLM calls use async/await
- **Structured Logging**: Uses structlog for audit trail
- **Confidence Scoring**: Calculates deployment readiness scores
- **Test Execution**: Executes tests and adjusts confidence based on results (Story 9.3)

Example:
    >>> from yolo_developer.agents.tea import tea_node
    >>> from yolo_developer.orchestrator.state import YoloState
    >>>
    >>> state: YoloState = {
    ...     "messages": [...],
    ...     "current_agent": "tea",
    ...     "handoff_context": None,
    ...     "decisions": [],
    ... }
    >>> result = await tea_node(state)
    >>> result["messages"]  # New messages to append
    [AIMessage(...)]

Architecture Note:
    Per ADR-005, this node follows the LangGraph pattern of receiving
    full state and returning only the updates to apply.
"""

from __future__ import annotations

from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.tea.types import (
    DeploymentRecommendation,
    Finding,
    TEAOutput,
    ValidationResult,
    ValidationStatus,
)
from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)


def _extract_artifacts_for_validation(state: YoloState) -> list[dict[str, Any]]:
    """Extract artifacts from orchestration state for validation.

    Artifacts can be present in state in two ways:
    1. Directly in dev_output key (preferred - has implementation details)
    2. In message metadata from Dev agent messages

    Args:
        state: Current orchestration state.

    Returns:
        List of artifact dictionaries. Empty list if no artifacts found.

    Example:
        >>> state = {"dev_output": {"implementations": [...]}}
        >>> artifacts = _extract_artifacts_for_validation(state)
        >>> len(artifacts)
        3
    """
    artifacts: list[dict[str, Any]] = []

    # First, try to extract from dev_output (direct state key)
    dev_output = state.get("dev_output")
    if dev_output and isinstance(dev_output, dict):
        implementations = dev_output.get("implementations", [])
        for impl in implementations:
            if not isinstance(impl, dict):
                continue

            # Extract code files
            code_files = impl.get("code_files", [])
            for code_file in code_files:
                if isinstance(code_file, dict):
                    artifacts.append(
                        {
                            "type": "code_file",
                            "artifact_id": code_file.get("file_path", "unknown"),
                            "content": code_file.get("content", ""),
                            "file_type": code_file.get("file_type", "source"),
                            "story_id": impl.get("story_id", "unknown"),
                        }
                    )

            # Extract test files
            test_files = impl.get("test_files", [])
            for test_file in test_files:
                if isinstance(test_file, dict):
                    artifacts.append(
                        {
                            "type": "test_file",
                            "artifact_id": test_file.get("file_path", "unknown"),
                            "content": test_file.get("content", ""),
                            "test_type": test_file.get("test_type", "unit"),
                            "story_id": impl.get("story_id", "unknown"),
                        }
                    )

        if artifacts:
            logger.info(
                "artifacts_extracted_from_dev_output",
                artifact_count=len(artifacts),
                code_files=len([a for a in artifacts if a["type"] == "code_file"]),
                test_files=len([a for a in artifacts if a["type"] == "test_file"]),
            )
            return artifacts

    # Fallback: Extract from message metadata (find latest Dev message)
    messages = state.get("messages", [])
    dev_messages = []

    for msg in messages:
        if hasattr(msg, "additional_kwargs"):
            kwargs = msg.additional_kwargs
            if kwargs.get("agent") == "dev":
                dev_messages.append(msg)

    # Get artifacts from the latest Dev message
    if dev_messages:
        latest_dev_msg = dev_messages[-1]
        output = latest_dev_msg.additional_kwargs.get("output", {})
        implementations = output.get("implementations", [])

        for impl in implementations:
            if not isinstance(impl, dict):
                continue

            # Extract code files from message
            code_files = impl.get("code_files", [])
            for code_file in code_files:
                if isinstance(code_file, dict):
                    artifacts.append(
                        {
                            "type": "code_file",
                            "artifact_id": code_file.get("file_path", "unknown"),
                            "content": code_file.get("content", ""),
                            "file_type": code_file.get("file_type", "source"),
                            "story_id": impl.get("story_id", "unknown"),
                        }
                    )

            # Extract test files from message
            test_files = impl.get("test_files", [])
            for test_file in test_files:
                if isinstance(test_file, dict):
                    artifacts.append(
                        {
                            "type": "test_file",
                            "artifact_id": test_file.get("file_path", "unknown"),
                            "content": test_file.get("content", ""),
                            "test_type": test_file.get("test_type", "unit"),
                            "story_id": impl.get("story_id", "unknown"),
                        }
                    )

        if artifacts:
            logger.info(
                "artifacts_extracted_from_message",
                artifact_count=len(artifacts),
                code_files=len([a for a in artifacts if a["type"] == "code_file"]),
                test_files=len([a for a in artifacts if a["type"] == "test_file"]),
            )
            return artifacts

    logger.debug("no_artifacts_found_in_state")
    return []


def _validate_artifact(
    artifact: dict[str, Any],
    test_content: str = "",
) -> ValidationResult:
    """Validate a single artifact and generate validation result.

    For code files, includes coverage analysis using the provided test content.
    For test files, validates assertion presence and quality.

    Args:
        artifact: Artifact dictionary with type, artifact_id, content, etc.
        test_content: Combined test file content for coverage analysis.

    Returns:
        ValidationResult with findings and scores.

    Example:
        >>> artifact = {"type": "code_file", "artifact_id": "src/main.py", ...}
        >>> result = _validate_artifact(artifact, test_content="def test_main(): assert True")
        >>> result.validation_status
        'passed'
    """
    from yolo_developer.agents.tea.coverage import (
        analyze_coverage,
        check_coverage_threshold,
        validate_critical_paths,
    )

    artifact_id = artifact.get("artifact_id", "unknown")
    artifact_type = artifact.get("type", "unknown")
    content = artifact.get("content", "")

    findings: list[Finding] = []
    recommendations: list[str] = []
    score = 100

    # Validation logic
    if artifact_type == "code_file":
        # Check for docstring presence
        if '"""' not in content and "'''" not in content:
            findings.append(
                Finding(
                    finding_id=f"F-{artifact_id[:8]}-001",
                    category="documentation",
                    severity="medium",
                    description="Module may be missing docstring",
                    location=artifact_id,
                    remediation="Add module-level docstring explaining purpose",
                )
            )
            score -= 10

        # Check for type hints (basic heuristic)
        if "def " in content and "->" not in content:
            findings.append(
                Finding(
                    finding_id=f"F-{artifact_id[:8]}-002",
                    category="code_quality",
                    severity="low",
                    description="Functions may be missing return type hints",
                    location=artifact_id,
                    remediation="Add return type annotations to all functions",
                )
            )
            score -= 5

        # Coverage analysis (Story 9.2)
        code_files = [{"artifact_id": artifact_id, "content": content}]
        test_files = (
            [{"artifact_id": "tests/combined.py", "content": test_content}] if test_content else []
        )

        coverage_report = analyze_coverage(code_files, test_files)

        # Check threshold
        _threshold_passed, threshold_findings = check_coverage_threshold(coverage_report)
        findings.extend(threshold_findings)

        # Validate critical paths
        critical_findings = validate_critical_paths(coverage_report)
        findings.extend(critical_findings)

        # Adjust score based on coverage
        if coverage_report.results:
            coverage_pct = coverage_report.results[0].coverage_percentage
            # Coverage impact: -0.2 points per % below 100
            coverage_penalty = int((100 - coverage_pct) * 0.2)
            score -= coverage_penalty

        recommendations.append("Consider running mypy for comprehensive type checking")

    elif artifact_type == "test_file":
        # Check for test assertions
        if "assert" not in content and "pytest.raises" not in content:
            findings.append(
                Finding(
                    finding_id=f"F-{artifact_id[:8]}-003",
                    category="test_coverage",
                    severity="high",
                    description="Test file may be missing assertions",
                    location=artifact_id,
                    remediation="Ensure tests have meaningful assertions",
                )
            )
            score -= 20

        recommendations.append("Consider adding edge case tests")

    # Determine validation status based on findings
    validation_status: ValidationStatus
    if any(f.severity == "critical" for f in findings):
        validation_status = "failed"
    elif any(f.severity == "high" for f in findings):
        validation_status = "warning"
    elif findings:
        validation_status = "warning"
    else:
        validation_status = "passed"

    logger.debug(
        "artifact_validated",
        artifact_id=artifact_id,
        validation_status=validation_status,
        finding_count=len(findings),
        score=score,
    )

    return ValidationResult(
        artifact_id=artifact_id,
        validation_status=validation_status,
        findings=tuple(findings),
        recommendations=tuple(recommendations),
        score=max(0, score),
    )


def _calculate_overall_confidence(
    results: list[ValidationResult],
) -> tuple[float, DeploymentRecommendation]:
    """Calculate overall confidence score from validation results.

    Uses a weighted approach based on validation status and scores
    to determine deployment readiness.

    Args:
        results: List of ValidationResult objects.

    Returns:
        Tuple of (confidence_score, deployment_recommendation).

    Example:
        >>> results = [ValidationResult(score=90, ...), ValidationResult(score=80, ...)]
        >>> confidence, recommendation = _calculate_overall_confidence(results)
        >>> confidence
        0.85
    """
    if not results:
        return 1.0, "deploy"

    # Calculate weighted average score
    total_score = sum(r.score for r in results)
    avg_score = total_score / len(results)
    confidence = avg_score / 100.0

    # Count issues by severity
    failed_count = sum(1 for r in results if r.validation_status == "failed")
    warning_count = sum(1 for r in results if r.validation_status == "warning")

    # Determine deployment recommendation
    recommendation: DeploymentRecommendation
    if failed_count > 0:
        recommendation = "block"
    elif warning_count > 0:
        recommendation = "deploy_with_warnings"
    else:
        recommendation = "deploy"

    logger.debug(
        "confidence_calculated",
        avg_score=avg_score,
        confidence=confidence,
        failed_count=failed_count,
        warning_count=warning_count,
        recommendation=recommendation,
    )

    return confidence, recommendation


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
@quality_gate("confidence_scoring", blocking=False)
async def tea_node(state: YoloState) -> dict[str, Any]:
    """TEA agent node for validation and quality assurance.

    Receives implementation artifacts from state and validates them
    for quality, test coverage, and deployment readiness.

    This function follows the LangGraph node pattern:
    - Receives full state as YoloState TypedDict
    - Returns only the state updates (not full state)
    - Never mutates the input state
    - Uses tenacity for retry with exponential backoff (AC4)

    Args:
        state: Current orchestration state with dev_output from Dev agent.

    Returns:
        State update dict with:
        - messages: List of new messages to append
        - decisions: List of new decisions to append
        - tea_output: Serialized TEAOutput
        Never includes current_agent (handoff manages that).

    Example:
        >>> state: YoloState = {
        ...     "messages": [...],
        ...     "current_agent": "tea",
        ...     "handoff_context": None,
        ...     "decisions": [],
        ...     "dev_output": {...},
        ... }
        >>> result = await tea_node(state)
        >>> "messages" in result
        True
    """
    logger.info(
        "tea_node_start",
        current_agent=state.get("current_agent"),
        message_count=len(state.get("messages", [])),
    )

    # Extract artifacts for validation (AC2)
    artifacts = _extract_artifacts_for_validation(state)

    # Separate test files for coverage analysis context
    test_artifacts = [a for a in artifacts if a.get("type") == "test_file"]

    # Combine all test content for coverage heuristics
    combined_test_content = "\n".join(a.get("content", "") for a in test_artifacts)

    # Execute tests and get results (Story 9.3)
    from yolo_developer.agents.tea.execution import (
        TestExecutionResult,
        execute_tests,
        generate_test_findings,
    )

    test_execution_result: TestExecutionResult | None = None
    test_findings: list[Finding] = []

    if test_artifacts:
        # Prepare test files for execution
        test_files = [
            {"artifact_id": a.get("artifact_id", "unknown"), "content": a.get("content", "")}
            for a in test_artifacts
        ]
        test_execution_result = execute_tests(test_files)
        test_findings = generate_test_findings(test_execution_result)

        logger.info(
            "test_execution_complete",
            test_count=test_execution_result.passed_count
            + test_execution_result.failed_count
            + test_execution_result.error_count,
            passed=test_execution_result.passed_count,
            failed=test_execution_result.failed_count,
            errors=test_execution_result.error_count,
            status=test_execution_result.status,
        )

    # Validate each artifact (AC3)
    validation_results: list[ValidationResult] = []
    for artifact in artifacts:
        # Pass test content for code file coverage analysis
        if artifact.get("type") == "code_file":
            result = _validate_artifact(artifact, test_content=combined_test_content)
        else:
            result = _validate_artifact(artifact)
        validation_results.append(result)

    # Add test execution findings as a validation result (Story 9.3 AC6)
    if test_findings:
        test_validation_status: ValidationStatus = "passed"
        if any(f.severity == "critical" for f in test_findings):
            test_validation_status = "failed"
        elif any(f.severity in ("high", "medium") for f in test_findings):
            test_validation_status = "warning"

        test_validation_result = ValidationResult(
            artifact_id="test_execution",
            validation_status=test_validation_status,
            findings=tuple(test_findings),
            recommendations=("Review and fix test issues before deployment.",),
            score=max(0, 100 - len(test_findings) * 10),
        )
        validation_results.append(test_validation_result)

        logger.debug(
            "test_findings_added_to_validation",
            finding_count=len(test_findings),
            validation_status=test_validation_status,
        )

    # Calculate coverage report for confidence scoring (Story 9.4)
    from yolo_developer.agents.tea.coverage import analyze_coverage
    from yolo_developer.agents.tea.risk import (
        categorize_risks,
        check_risk_deployment_blocking,
        generate_risk_report,
    )
    from yolo_developer.agents.tea.scoring import calculate_confidence_score

    # Prepare code and test files for coverage analysis
    code_artifacts = [a for a in artifacts if a.get("type") == "code_file"]
    code_files = [
        {"artifact_id": a.get("artifact_id", "unknown"), "content": a.get("content", "")}
        for a in code_artifacts
    ]
    test_files_for_coverage = [
        {"artifact_id": a.get("artifact_id", "unknown"), "content": a.get("content", "")}
        for a in test_artifacts
    ]

    coverage_report = None
    if code_files:
        coverage_report = analyze_coverage(code_files, test_files_for_coverage)

    # Calculate comprehensive confidence score using Story 9.4 system
    confidence_result = calculate_confidence_score(
        validation_results=tuple(validation_results),
        coverage_report=coverage_report,
        test_execution_result=test_execution_result,
    )

    # Use new confidence scoring for overall confidence (backward compatible)
    overall_confidence = confidence_result.score / 100.0  # Convert 0-100 to 0-1
    deployment_recommendation = confidence_result.deployment_recommendation

    logger.debug(
        "confidence_score_calculated",
        score=confidence_result.score,
        passed_threshold=confidence_result.passed_threshold,
        recommendation=deployment_recommendation,
    )

    # Risk categorization (Story 9.5)
    categorized_risks = categorize_risks(tuple(validation_results))
    risk_report = generate_risk_report(categorized_risks)

    # Check if risk should block deployment (in addition to confidence score)
    risk_blocked, risk_blocking_reasons = check_risk_deployment_blocking(risk_report)

    # Update deployment recommendation if risk blocks
    if risk_blocked and deployment_recommendation != "block":
        deployment_recommendation = "block"
        logger.warning(
            "deployment_blocked_by_risk",
            risk_blocking_reasons=risk_blocking_reasons,
        )

    logger.info(
        "risk_categorization_complete",
        critical_count=risk_report.critical_count,
        high_count=risk_report.high_count,
        low_count=risk_report.low_count,
        overall_risk_level=risk_report.overall_risk_level,
        deployment_blocked=risk_report.deployment_blocked,
    )

    # Build processing notes
    total_findings = sum(len(r.findings) for r in validation_results)
    failed_count = sum(1 for r in validation_results if r.validation_status == "failed")
    warning_count = sum(1 for r in validation_results if r.validation_status == "warning")
    passed_count = sum(1 for r in validation_results if r.validation_status == "passed")

    # Include test execution stats in processing notes
    test_stats = ""
    if test_execution_result:
        test_stats = (
            f" Test execution: {test_execution_result.passed_count} passed, "
            f"{test_execution_result.failed_count} failed, "
            f"{test_execution_result.error_count} errors."
        )

    # Include confidence breakdown summary in processing notes
    confidence_summary = (
        f" Confidence breakdown: "
        f"coverage={confidence_result.breakdown.coverage_score:.1f}, "
        f"tests={confidence_result.breakdown.test_execution_score:.1f}, "
        f"validation={confidence_result.breakdown.validation_score:.1f}."
    )

    # Include risk summary in processing notes (Story 9.5)
    risk_summary = (
        f" Risk: {risk_report.critical_count} critical, "
        f"{risk_report.high_count} high, {risk_report.low_count} low."
    )

    processing_notes = (
        f"Validated {len(artifacts)} artifacts. "
        f"Results: {passed_count} passed, {warning_count} warnings, {failed_count} failed. "
        f"Total findings: {total_findings + len(test_findings)}.{test_stats} "
        f"Overall confidence: {overall_confidence:.2%}.{confidence_summary}{risk_summary}"
    )

    # Create TEA output with test execution, confidence, and risk results (Story 9.3, 9.4, 9.5)
    output = TEAOutput(
        validation_results=tuple(validation_results),
        processing_notes=processing_notes,
        overall_confidence=overall_confidence,
        deployment_recommendation=deployment_recommendation,
        test_execution_result=test_execution_result,
        confidence_result=confidence_result,
        risk_report=risk_report,
        overall_risk_level=risk_report.overall_risk_level,
    )

    # Create decision record with TEA attribution (AC6)
    decision = Decision(
        agent="tea",
        summary=f"Validated {len(artifacts)} artifacts. "
        f"Confidence: {overall_confidence:.2%}. "
        f"Recommendation: {deployment_recommendation}.",
        rationale=f"Processed artifacts from dev_output. "
        f"Found {total_findings} findings across {len(validation_results)} validations. "
        f"Deployment recommendation based on validation status distribution.",
        related_artifacts=tuple(a.get("artifact_id", "unknown") for a in artifacts),
    )

    # Create output message with TEA attribution (AC6)
    message = create_agent_message(
        content=f"TEA validation complete: {len(artifacts)} artifacts validated. "
        f"Confidence: {overall_confidence:.2%}. "
        f"Recommendation: {deployment_recommendation}.",
        agent="tea",
        metadata={"output": output.to_dict()},
    )

    logger.info(
        "tea_node_complete",
        artifact_count=len(artifacts),
        validation_count=len(validation_results),
        finding_count=total_findings,
        overall_confidence=overall_confidence,
        deployment_recommendation=deployment_recommendation,
    )

    # Return ONLY the updates, not full state (AC6)
    return {
        "messages": [message],
        "decisions": [decision],
        "tea_output": output.to_dict(),
    }
