"""Analyst agent node for LangGraph orchestration (Story 5.1, 5.2, 5.3).

This module provides the analyst_node function that integrates with the
LangGraph orchestration workflow. The Analyst agent crystallizes requirements
from seed content, identifies gaps, and flags contradictions.

Key Concepts:
- **YoloState Input**: Receives state as TypedDict, not Pydantic
- **Immutable Updates**: Returns state update dict, never mutates input
- **Async I/O**: All LLM calls use async/await
- **Structured Logging**: Uses structlog for audit trail

Example:
    >>> from yolo_developer.agents.analyst import analyst_node
    >>> from yolo_developer.orchestrator.state import YoloState
    >>>
    >>> state: YoloState = {
    ...     "messages": [HumanMessage(content="Build a todo app")],
    ...     "current_agent": "analyst",
    ...     "handoff_context": None,
    ...     "decisions": [],
    ... }
    >>> result = await analyst_node(state)
    >>> result["messages"]  # New messages to append
    [AIMessage(...)]

Architecture Note:
    Per ADR-005, this node follows the LangGraph pattern of receiving
    full state and returning only the updates to apply.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog
from langchain_core.messages import BaseMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.analyst.types import (
    AnalystOutput,
    CrystallizedRequirement,
    GapType,
    IdentifiedGap,
    Severity,
)
from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)

# Flag to enable/disable actual LLM calls (for testing)
_USE_LLM: bool = False

# Vague terms to detect in requirements (Story 5.2)
# These patterns indicate ambiguity that needs crystallization
VAGUE_TERMS: frozenset[str] = frozenset([
    # Quantifier vagueness
    "fast", "quick", "slow", "efficient", "performant",
    "scalable", "responsive", "real-time",
    # Ease vagueness
    "easy", "simple", "straightforward", "intuitive",
    "user-friendly", "seamless",
    # Certainty vagueness
    "should", "might", "could", "may", "possibly",
    "probably", "maybe", "sometimes",
    # Scope vagueness
    "etc", "and so on", "and more", "various", "multiple",
    "several", "many", "few", "some",
    # Quality vagueness
    "good", "better", "best", "nice", "beautiful",
    "clean", "modern", "robust",
])

# Edge case categories to check for each requirement (Story 5.3)
EDGE_CASE_CATEGORIES: dict[str, list[str]] = {
    "input_validation": [
        "Empty/null input handling",
        "Maximum length exceeded",
        "Invalid format handling",
        "Special characters handling",
        "Unicode/encoding handling",
    ],
    "boundary_conditions": [
        "Zero values",
        "Negative values",
        "Maximum integer overflow",
        "Date boundary handling (leap years, etc.)",
        "Time zone handling",
    ],
    "error_conditions": [
        "Network failure handling",
        "Database connection loss",
        "External service timeout",
        "Concurrent modification conflicts",
        "Disk space exhaustion",
    ],
    "state_transitions": [
        "Duplicate submission prevention",
        "Invalid state transition handling",
        "Rollback on partial failure",
        "Recovery from interrupted operations",
    ],
}

# Keywords that map requirement content to relevant edge case categories
EDGE_CASE_KEYWORDS: dict[str, frozenset[str]] = {
    "input_validation": frozenset([
        "input", "form", "field", "submit", "enter", "data", "value",
        "text", "string", "user", "request", "parameter", "argument",
    ]),
    "boundary_conditions": frozenset([
        "number", "count", "limit", "max", "min", "date", "time",
        "range", "quantity", "amount", "size", "length", "age",
    ]),
    "error_conditions": frozenset([
        "api", "call", "fetch", "request", "network", "database", "db",
        "connect", "external", "service", "integrate", "http", "remote",
    ]),
    "state_transitions": frozenset([
        "status", "state", "workflow", "process", "step", "transition",
        "update", "change", "modify", "submit", "save", "transaction",
    ]),
}


def _detect_vague_terms(text: str) -> set[str]:
    """Detect vague terms in requirement text.

    Scans the input text for common vague terms that indicate
    ambiguity needing crystallization. Detection is case-insensitive.

    Args:
        text: The requirement text to analyze.

    Returns:
        Set of detected vague terms (lowercase). Empty set if none found.

    Example:
        >>> _detect_vague_terms("The system should be fast")
        {'should', 'fast'}
        >>> _detect_vague_terms("Response time < 200ms")
        set()
    """
    if not text:
        return set()

    text_lower = text.lower()
    detected: set[str] = set()

    for term in VAGUE_TERMS:
        # Check for the term as a word (not substring of another word)
        # Handle hyphenated terms specially
        if "-" in term:
            if term in text_lower:
                detected.add(term)
        else:
            # Use simple word boundary check
            # Check if term appears surrounded by non-alphanumeric chars
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, text_lower):
                detected.add(term)

    return detected


def _identify_edge_cases(
    requirements: tuple[CrystallizedRequirement, ...],
) -> tuple[IdentifiedGap, ...]:
    """Identify missing edge cases from requirements.

    Analyzes requirements to detect potentially missing edge case handling
    such as error conditions, boundary values, and input validation.

    Args:
        requirements: Tuple of crystallized requirements to analyze.

    Returns:
        Tuple of IdentifiedGap objects for edge cases, sorted by severity.

    Example:
        >>> reqs = (CrystallizedRequirement(
        ...     id="req-001",
        ...     original_text="User can submit form",
        ...     refined_text="User submits form with validation",
        ...     category="functional",
        ...     testable=True,
        ... ),)
        >>> gaps = _identify_edge_cases(reqs)
        >>> any("empty" in g.description.lower() for g in gaps)
        True
    """
    gaps: list[IdentifiedGap] = []
    gap_counter = 1

    for req in requirements:
        # Combine original and refined text for analysis
        text_to_analyze = f"{req.original_text} {req.refined_text}".lower()

        # Check each edge case category
        for category, keywords in EDGE_CASE_KEYWORDS.items():
            # Check if any keyword matches
            matched = any(kw in text_to_analyze for kw in keywords)
            if not matched:
                continue

            # Get edge cases for this category
            edge_cases = EDGE_CASE_CATEGORIES.get(category, [])

            for edge_case in edge_cases:
                # Check if edge case is already mentioned in requirement
                edge_case_lower = edge_case.lower()
                if any(
                    word in text_to_analyze
                    for word in edge_case_lower.split()
                    if len(word) > 4
                ):
                    continue

                # Determine severity based on edge case type
                severity = _assess_edge_case_severity(category, edge_case)

                gap = IdentifiedGap(
                    id=f"gap-{gap_counter:03d}",
                    description=f"Missing edge case: {edge_case}",
                    gap_type=GapType.EDGE_CASE,
                    severity=severity,
                    source_requirements=(req.id,),
                    rationale=(
                        f"Requirement '{req.id}' involves {category.replace('_', ' ')} "
                        f"but does not address {edge_case.lower()}"
                    ),
                )
                gaps.append(gap)
                gap_counter += 1

    # Sort by severity (critical first) and return
    severity_order = {
        Severity.CRITICAL: 0,
        Severity.HIGH: 1,
        Severity.MEDIUM: 2,
        Severity.LOW: 3,
    }
    gaps.sort(key=lambda g: severity_order.get(g.severity, 4))

    return tuple(gaps)


def _assess_edge_case_severity(category: str, edge_case: str) -> Severity:
    """Assess severity of an edge case based on category and type.

    Args:
        category: The edge case category (e.g., "error_conditions").
        edge_case: The specific edge case description.

    Returns:
        Severity level for the edge case.
    """
    edge_case_lower = edge_case.lower()

    # Critical: Security-related or data integrity issues
    if any(
        term in edge_case_lower
        for term in ["overflow", "injection", "unauthorized", "data loss"]
    ):
        return Severity.CRITICAL

    # High: Error conditions that could cause system failures
    if category == "error_conditions":
        return Severity.HIGH

    # High: State transition issues that could corrupt data
    if category == "state_transitions" and any(
        term in edge_case_lower for term in ["rollback", "duplicate", "conflict"]
    ):
        return Severity.HIGH

    # Medium: Input validation and boundary conditions
    if category in ("input_validation", "boundary_conditions"):
        return Severity.MEDIUM

    # Low: Other edge cases
    return Severity.LOW


# Rules for identifying implied requirements (Story 5.3)
# Maps trigger keywords to implied requirements with rationale
IMPLIED_REQUIREMENT_RULES: dict[str, list[tuple[str, str]]] = {
    # Authentication implies these features
    "login": [
        ("Logout functionality", "Login implies users need a way to end their session"),
        ("Password reset/recovery", "Login implies users may forget credentials"),
        ("Session management", "Login implies need to track active sessions"),
    ],
    "authentication": [
        ("Logout functionality", "Authentication implies session termination capability"),
        ("Password policy enforcement", "Authentication implies credential requirements"),
        ("Failed login handling", "Authentication implies need for security controls"),
    ],
    "sign in": [
        ("Sign out functionality", "Sign in implies users need to sign out"),
        ("Password recovery", "Sign in implies credential recovery needs"),
    ],
    # User management implications
    "user registration": [
        ("Email verification", "Registration implies need to verify user identity"),
        ("Account activation flow", "Registration implies activation process"),
        ("Duplicate account prevention", "Registration implies uniqueness constraints"),
    ],
    "create user": [
        ("Delete user capability", "User creation implies user deletion"),
        ("User validation", "User creation implies input validation"),
    ],
    "user account": [
        ("Account deactivation", "Account implies ability to disable"),
        ("Profile management", "Account implies profile editing"),
    ],
    # Data operations implications
    "save": [
        ("Unsaved changes warning", "Save implies user should be warned of data loss"),
        ("Save failure handling", "Save implies recovery from save errors"),
    ],
    "delete": [
        ("Soft delete or undo capability", "Delete implies recovery option"),
        ("Deletion confirmation", "Delete implies preventing accidental deletion"),
        ("Cascade handling", "Delete implies handling related data"),
    ],
    "upload": [
        ("File size limits", "Upload implies size constraints"),
        ("File type validation", "Upload implies format restrictions"),
        ("Upload progress indication", "Upload implies user feedback"),
    ],
    # Communication implications
    "email": [
        ("Email delivery verification", "Email implies delivery confirmation"),
        ("Email bounce handling", "Email implies handling failed delivery"),
    ],
    "notification": [
        ("Notification preferences", "Notifications imply user control over them"),
        ("Notification history", "Notifications imply record keeping"),
    ],
    # Security implications
    "password": [
        ("Password strength requirements", "Password implies security policy"),
        ("Password change capability", "Password implies update functionality"),
        ("Password history tracking", "Password implies preventing reuse"),
    ],
    "permission": [
        ("Permission denied handling", "Permissions imply access denial response"),
        ("Permission auditing", "Permissions imply tracking access"),
    ],
}


def _identify_implied_requirements(
    requirements: tuple[CrystallizedRequirement, ...],
) -> tuple[IdentifiedGap, ...]:
    """Identify implied but unstated requirements.

    Analyzes requirements to find logically implied requirements that
    were not explicitly stated, based on common software patterns.

    Args:
        requirements: Tuple of crystallized requirements to analyze.

    Returns:
        Tuple of IdentifiedGap objects for implied requirements.

    Example:
        >>> reqs = (CrystallizedRequirement(
        ...     id="req-001",
        ...     original_text="User can login",
        ...     refined_text="User authenticates with email and password",
        ...     category="functional",
        ...     testable=True,
        ... ),)
        >>> gaps = _identify_implied_requirements(reqs)
        >>> any("logout" in g.description.lower() for g in gaps)
        True
    """
    gaps: list[IdentifiedGap] = []
    gap_counter = 1
    seen_implications: set[str] = set()

    # Combine all requirement text to check for already-stated requirements
    all_req_text = " ".join(
        f"{r.original_text} {r.refined_text}".lower() for r in requirements
    )

    for req in requirements:
        text_to_analyze = f"{req.original_text} {req.refined_text}".lower()

        # Check each trigger pattern
        for trigger, implications in IMPLIED_REQUIREMENT_RULES.items():
            if trigger not in text_to_analyze:
                continue

            for implied_desc, rationale in implications:
                # Skip if already stated in requirements
                implied_lower = implied_desc.lower()
                key_words = [w for w in implied_lower.split() if len(w) > 4]
                if any(word in all_req_text for word in key_words):
                    continue

                # Skip if we've already suggested this
                if implied_lower in seen_implications:
                    continue
                seen_implications.add(implied_lower)

                # Determine severity based on implied requirement type
                severity = _assess_implied_severity(implied_desc)

                gap = IdentifiedGap(
                    id=f"gap-{gap_counter:03d}",
                    description=f"Implied requirement: {implied_desc}",
                    gap_type=GapType.IMPLIED_REQUIREMENT,
                    severity=severity,
                    source_requirements=(req.id,),
                    rationale=rationale,
                )
                gaps.append(gap)
                gap_counter += 1

    return tuple(gaps)


def _assess_implied_severity(implied_desc: str) -> Severity:
    """Assess severity of an implied requirement.

    Args:
        implied_desc: Description of the implied requirement.

    Returns:
        Severity level for the implied requirement.
    """
    desc_lower = implied_desc.lower()

    # Critical: Security-related implications
    if any(
        term in desc_lower
        for term in ["password", "authentication", "security", "permission"]
    ):
        return Severity.HIGH

    # High: Core functionality implications
    if any(
        term in desc_lower
        for term in ["logout", "session", "verification", "validation"]
    ):
        return Severity.HIGH

    # Medium: User experience implications
    if any(
        term in desc_lower
        for term in ["confirmation", "warning", "preference", "progress"]
    ):
        return Severity.MEDIUM

    # Low: Nice-to-have implications
    return Severity.LOW


# Pattern knowledge base for common software domains (Story 5.3)
DOMAIN_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "authentication": [
        ("User registration", "Standard auth requires user signup capability"),
        ("Password reset/recovery", "Users commonly forget credentials"),
        ("Session management", "Auth requires tracking active sessions"),
        ("Logout functionality", "Sessions must be terminable"),
        ("Multi-factor authentication", "Industry standard for enhanced security"),
        ("Account lockout after failed attempts", "Prevents brute force attacks"),
        ("Password strength validation", "Enforces credential security"),
        ("Remember me functionality", "Common user convenience feature"),
    ],
    "authorization": [
        ("Role-based access control", "Standard pattern for permission management"),
        ("Permission checking", "Core authorization functionality"),
        ("Access denied handling", "Must handle unauthorized access attempts"),
        ("Admin vs user separation", "Common role distinction"),
        ("Permission audit logging", "Compliance and security tracking"),
    ],
    "crud": [
        ("Create operation", "Standard data operation"),
        ("Read/retrieve operation", "Standard data operation"),
        ("Update operation", "Standard data operation"),
        ("Delete operation", "Standard data operation"),
        ("List/pagination", "Standard for multiple records"),
        ("Filtering and search", "Common data discovery feature"),
        ("Soft delete capability", "Data recovery option"),
        ("Bulk operations", "Efficiency for multiple records"),
    ],
    "api": [
        ("Error response handling", "Standard API requirement"),
        ("Input validation", "Security and data integrity"),
        ("Rate limiting", "Prevents abuse and ensures fair usage"),
        ("API versioning", "Backward compatibility strategy"),
        ("Authentication/authorization headers", "API security"),
        ("Request/response logging", "Debugging and auditing"),
        ("CORS configuration", "Cross-origin request handling"),
        ("Health check endpoint", "Monitoring and load balancing"),
    ],
    "data": [
        ("Data validation", "Ensures data integrity"),
        ("Backup and recovery", "Data protection requirement"),
        ("Data migration handling", "Schema evolution support"),
        ("Concurrent access handling", "Multi-user data integrity"),
        ("Transaction management", "Atomic operations"),
        ("Data archival strategy", "Long-term data management"),
    ],
    "user_interface": [
        ("Loading states", "User feedback during operations"),
        ("Error messages", "User communication"),
        ("Form validation", "Input quality assurance"),
        ("Responsive design", "Multi-device support"),
        ("Accessibility features", "Inclusive design"),
        ("Empty states", "Handling no-data scenarios"),
    ],
}

# Keywords to detect domain context from requirements
DOMAIN_KEYWORDS: dict[str, frozenset[str]] = {
    "authentication": frozenset([
        "login", "authenticate", "sign in", "signin", "credential",
        "password", "auth", "sso", "oauth", "identity",
    ]),
    "authorization": frozenset([
        "permission", "role", "access", "authorize", "privilege",
        "admin", "rbac", "acl", "policy", "grant",
    ]),
    "crud": frozenset([
        "create", "read", "update", "delete", "list", "get", "post",
        "put", "patch", "remove", "add", "edit", "modify",
    ]),
    "api": frozenset([
        "api", "endpoint", "rest", "graphql", "request", "response",
        "http", "webhook", "integration", "service",
    ]),
    "data": frozenset([
        "database", "store", "persist", "query", "record", "entity",
        "model", "schema", "migration", "backup", "archive",
    ]),
    "user_interface": frozenset([
        "ui", "interface", "form", "button", "page", "screen",
        "dashboard", "component", "display", "view", "layout",
    ]),
}


def _suggest_from_patterns(
    requirements: tuple[CrystallizedRequirement, ...],
) -> tuple[IdentifiedGap, ...]:
    """Suggest missing features based on domain patterns.

    Analyzes requirements to identify domain context and suggests
    commonly expected features based on industry patterns.

    Args:
        requirements: Tuple of crystallized requirements to analyze.

    Returns:
        Tuple of IdentifiedGap objects for pattern-based suggestions.

    Example:
        >>> reqs = (CrystallizedRequirement(
        ...     id="req-001",
        ...     original_text="Build user login system",
        ...     refined_text="Implement user authentication with email",
        ...     category="functional",
        ...     testable=True,
        ... ),)
        >>> gaps = _suggest_from_patterns(reqs)
        >>> any("registration" in g.description.lower() for g in gaps)
        True
    """
    gaps: list[IdentifiedGap] = []
    gap_counter = 1
    seen_suggestions: set[str] = set()

    # Detect domains mentioned in requirements
    all_req_text = " ".join(
        f"{r.original_text} {r.refined_text}".lower() for r in requirements
    )

    detected_domains: set[str] = set()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in all_req_text for kw in keywords):
            detected_domains.add(domain)

    # For each detected domain, suggest missing patterns
    for domain in detected_domains:
        patterns = DOMAIN_PATTERNS.get(domain, [])

        for pattern_desc, explanation in patterns:
            pattern_lower = pattern_desc.lower()

            # Skip if already covered in requirements
            key_words = [w for w in pattern_lower.split() if len(w) > 4]
            if any(word in all_req_text for word in key_words):
                continue

            # Skip duplicates
            if pattern_lower in seen_suggestions:
                continue
            seen_suggestions.add(pattern_lower)

            # Find source requirement(s) for traceability
            source_reqs: list[str] = []
            for req in requirements:
                req_text = f"{req.original_text} {req.refined_text}".lower()
                if any(kw in req_text for kw in DOMAIN_KEYWORDS[domain]):
                    source_reqs.append(req.id)
            if not source_reqs:
                source_reqs = [requirements[0].id] if requirements else []

            gap = IdentifiedGap(
                id=f"gap-{gap_counter:03d}",
                description=f"Pattern suggestion: {pattern_desc}",
                gap_type=GapType.PATTERN_SUGGESTION,
                severity=_assess_pattern_severity(domain, pattern_desc),
                source_requirements=tuple(source_reqs[:3]),  # Limit to 3 sources
                rationale=f"Domain: {domain}. {explanation}",
            )
            gaps.append(gap)
            gap_counter += 1

    return tuple(gaps)


def _assess_pattern_severity(domain: str, pattern_desc: str) -> Severity:
    """Assess severity of a pattern-based suggestion.

    Args:
        domain: The domain category (e.g., "authentication").
        pattern_desc: Description of the suggested pattern.

    Returns:
        Severity level for the pattern suggestion.
    """
    desc_lower = pattern_desc.lower()

    # High: Security-related patterns
    if domain in ("authentication", "authorization"):
        if any(
            term in desc_lower
            for term in ["lockout", "mfa", "multi-factor", "permission"]
        ):
            return Severity.HIGH
        return Severity.MEDIUM

    # Medium: Core CRUD and API patterns
    if domain in ("crud", "api"):
        if any(term in desc_lower for term in ["validation", "error", "rate"]):
            return Severity.MEDIUM
        return Severity.LOW

    # Low: UI and data patterns (nice-to-have)
    return Severity.LOW


@quality_gate("testability", blocking=True)
async def analyst_node(state: YoloState) -> dict[str, Any]:
    """Analyst agent node for requirement crystallization.

    Receives seed requirements from state messages and produces
    crystallized, categorized requirements with testability assessment.

    This function follows the LangGraph node pattern:
    - Receives full state as YoloState TypedDict
    - Returns only the state updates (not full state)
    - Never mutates the input state

    Args:
        state: Current orchestration state with accumulated messages.

    Returns:
        State update dict with:
        - messages: List of new messages to append
        - decisions: List of new decisions to append
        Never includes current_agent (handoff manages that).

    Example:
        >>> state: YoloState = {
        ...     "messages": [HumanMessage(content="Build an app")],
        ...     "current_agent": "analyst",
        ...     "handoff_context": None,
        ...     "decisions": [],
        ... }
        >>> result = await analyst_node(state)
        >>> "messages" in result
        True
    """
    logger.info(
        "analyst_node_start",
        current_agent=state.get("current_agent"),
        message_count=len(state.get("messages", [])),
    )

    # Extract seed content from messages
    seed_content = _extract_seed_from_messages(state.get("messages", []))

    # Process requirements using LLM
    output = await _crystallize_requirements(seed_content)

    # Count gaps by severity for decision record
    severity_counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for gap in output.structured_gaps:
        severity_counts[gap.severity.value] = severity_counts.get(gap.severity.value, 0) + 1

    # Create decision record with gap analysis details
    decision = Decision(
        agent="analyst",
        summary=f"Crystallized {len(output.requirements)} requirements, identified {len(output.structured_gaps)} gaps",
        rationale=(
            f"Analyzed seed content and extracted structured requirements. "
            f"Gap analysis found: {severity_counts['critical']} critical, "
            f"{severity_counts['high']} high, {severity_counts['medium']} medium, "
            f"{severity_counts['low']} low severity gaps. "
            f"Contradictions: {len(output.contradictions)}."
        ),
        related_artifacts=tuple(r.id for r in output.requirements) + tuple(
            g.id for g in output.structured_gaps
        ),
    )

    # Create output message with analyst attribution
    gap_summary = ""
    if output.structured_gaps:
        gap_summary = (
            f" Structured gaps: {len(output.structured_gaps)} "
            f"({severity_counts['critical']} critical, {severity_counts['high']} high, "
            f"{severity_counts['medium']} medium, {severity_counts['low']} low)."
        )

    message = create_agent_message(
        content=(
            f"Analysis complete: {len(output.requirements)} requirements crystallized."
            f"{gap_summary} "
            f"Contradictions: {len(output.contradictions)}."
        ),
        agent="analyst",
        metadata={"output": output.to_dict()},
    )

    logger.info(
        "analyst_node_complete",
        requirement_count=len(output.requirements),
        gaps_count=len(output.identified_gaps),
        structured_gaps_count=len(output.structured_gaps),
        gap_severity_counts=severity_counts,
        contradictions_count=len(output.contradictions),
    )

    # Return ONLY the updates, not full state
    return {
        "messages": [message],
        "decisions": [decision],
    }


def _extract_seed_from_messages(messages: list[BaseMessage]) -> str:
    """Extract seed content from accumulated messages.

    Looks through messages to find the seed document content that
    needs to be analyzed. Typically this is the first HumanMessage
    or content tagged as seed.

    Args:
        messages: List of accumulated messages in state.

    Returns:
        Concatenated seed content string.
    """
    if not messages:
        return ""

    # For now, concatenate all human message content as seed
    # In production, would look for specific seed tagging
    seed_parts: list[str] = []
    for msg in messages:
        # Check if it's a human message (has content attribute)
        if hasattr(msg, "content") and isinstance(msg.content, str):
            # Skip assistant/AI messages for seed extraction
            if msg.type == "human":
                seed_parts.append(msg.content)

    return "\n\n".join(seed_parts)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _call_llm(prompt: str, system: str) -> str:
    """Call LLM with retry logic.

    Uses LiteLLM's async API for LLM calls with automatic retries
    on transient failures. Uses the cheap_model from config for
    routine analysis tasks.

    Args:
        prompt: The user prompt to send to the LLM.
        system: The system prompt defining the LLM's role.

    Returns:
        The LLM's response content as a string.

    Raises:
        Exception: If all retry attempts fail.
    """
    from litellm import acompletion

    from yolo_developer.config import load_config

    config = load_config()
    model = config.llm.cheap_model

    logger.info("calling_llm", model=model, prompt_length=len(prompt))

    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    logger.debug("llm_response_received", response_length=len(content) if content else 0)

    return content or ""


def _parse_llm_response(response: str) -> AnalystOutput:
    """Parse LLM JSON response into AnalystOutput.

    Attempts to parse the LLM response as JSON and convert it to
    an AnalystOutput. Handles legacy format (Story 5.1), enhanced
    format (Story 5.2), and structured gaps format (Story 5.3).

    Args:
        response: The raw LLM response string (expected to be JSON).

    Returns:
        AnalystOutput parsed from the response.
    """
    try:
        data = json.loads(response)

        requirements = tuple(
            CrystallizedRequirement(
                id=req.get("id", f"req-{i:03d}"),
                original_text=req.get("original_text", ""),
                refined_text=req.get("refined_text", ""),
                category=req.get("category", "functional"),
                testable=req.get("testable", True),
                # New fields from Story 5.2 (with backward-compatible defaults)
                scope_notes=req.get("scope_notes"),
                implementation_hints=tuple(req.get("implementation_hints", [])),
                confidence=float(req.get("confidence", 1.0)),
            )
            for i, req in enumerate(data.get("requirements", []), start=1)
        )

        # Parse structured gaps from Story 5.3 (if present)
        structured_gaps: tuple[IdentifiedGap, ...] = ()
        if "structured_gaps" in data:
            parsed_gaps: list[IdentifiedGap] = []
            for i, gap_data in enumerate(data.get("structured_gaps", []), start=1):
                try:
                    gap = IdentifiedGap(
                        id=gap_data.get("id", f"gap-{i:03d}"),
                        description=gap_data.get("description", ""),
                        gap_type=GapType(gap_data.get("gap_type", "edge_case")),
                        severity=Severity(gap_data.get("severity", "medium")),
                        source_requirements=tuple(
                            gap_data.get("source_requirements", [])
                        ),
                        rationale=gap_data.get("rationale", ""),
                    )
                    parsed_gaps.append(gap)
                except (ValueError, KeyError) as gap_err:
                    logger.warning(
                        "gap_parse_failed", gap_index=i, error=str(gap_err)
                    )
            structured_gaps = tuple(parsed_gaps)

        return AnalystOutput(
            requirements=requirements,
            identified_gaps=tuple(data.get("identified_gaps", [])),
            contradictions=tuple(data.get("contradictions", [])),
            structured_gaps=structured_gaps,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("llm_response_parse_failed", error=str(e))
        return AnalystOutput(
            requirements=(),
            identified_gaps=("Failed to parse LLM response",),
            contradictions=(),
        )


async def _crystallize_requirements(seed_content: str) -> AnalystOutput:
    """Process seed content and extract crystallized requirements.

    When _USE_LLM is True, calls the LLM to analyze seed content
    and extract structured requirements. Otherwise, returns a
    placeholder output for testing that includes vague term detection.
    In both cases, runs local gap analysis to enhance results.

    Args:
        seed_content: The raw seed document content.

    Returns:
        AnalystOutput with requirements, gaps, contradictions, and structured_gaps.
    """
    logger.debug("crystallize_requirements_start", seed_length=len(seed_content))

    if not seed_content.strip():
        return AnalystOutput(
            requirements=(),
            identified_gaps=("No seed content provided for analysis",),
            contradictions=(),
        )

    # Detect vague terms for logging and placeholder generation
    vague_terms = _detect_vague_terms(seed_content)
    if vague_terms:
        logger.info(
            "vague_terms_detected",
            terms=list(vague_terms),
            count=len(vague_terms),
        )

    # Use LLM if enabled
    if _USE_LLM:
        from yolo_developer.agents.prompts.analyst import (
            ANALYST_SYSTEM_PROMPT,
            ANALYST_USER_PROMPT_TEMPLATE,
        )

        prompt = ANALYST_USER_PROMPT_TEMPLATE.format(seed_content=seed_content)
        response = await _call_llm(prompt, ANALYST_SYSTEM_PROMPT)
        output = _parse_llm_response(response)

        # Log transformation details for audit trail
        for req in output.requirements:
            logger.info(
                "requirement_crystallized",
                req_id=req.id,
                original_length=len(req.original_text),
                refined_length=len(req.refined_text),
                category=req.category,
                testable=req.testable,
                confidence=req.confidence,
                has_scope_notes=req.scope_notes is not None,
                hint_count=len(req.implementation_hints),
            )

        # Enhance with local gap analysis if LLM didn't produce structured gaps
        if not output.structured_gaps and output.requirements:
            output = _enhance_with_gap_analysis(output)

        return output

    # Placeholder for testing (when LLM is disabled)
    # Generate confidence based on vague term detection
    confidence = 1.0 - (len(vague_terms) * 0.1) if vague_terms else 1.0
    confidence = max(0.3, min(1.0, confidence))  # Clamp to [0.3, 1.0]

    # Generate scope notes if vague terms detected
    scope_notes: str | None = None
    if vague_terms:
        scope_notes = f"Vague terms detected: {', '.join(sorted(vague_terms))}. Scope needs clarification."

    # Generate implementation hints based on content
    hints: tuple[str, ...] = ()
    seed_lower = seed_content.lower()
    if "api" in seed_lower or "endpoint" in seed_lower:
        hints = ("Consider async handlers for I/O operations",)
    elif "ui" in seed_lower or "interface" in seed_lower:
        hints = ("Follow component-based architecture",)
    elif "data" in seed_lower or "database" in seed_lower:
        hints = ("Use repository pattern for data access",)

    placeholder_req = CrystallizedRequirement(
        id="req-001",
        original_text=seed_content[:200] if len(seed_content) > 200 else seed_content,
        refined_text=f"Implement: {seed_content[:100]}..." if len(seed_content) > 100 else seed_content,
        category="functional",
        testable=True,
        scope_notes=scope_notes,
        implementation_hints=hints,
        confidence=confidence,
    )

    # Create initial output with placeholder requirement
    initial_output = AnalystOutput(
        requirements=(placeholder_req,),
        identified_gaps=(),
        contradictions=(),
    )

    # Run gap analysis on placeholder output
    return _enhance_with_gap_analysis(initial_output)


def _enhance_with_gap_analysis(output: AnalystOutput) -> AnalystOutput:
    """Enhance AnalystOutput with gap analysis results.

    Runs edge case detection, implied requirement detection, and
    pattern-based suggestion on the crystallized requirements.

    Args:
        output: Initial AnalystOutput with requirements.

    Returns:
        Enhanced AnalystOutput with structured_gaps populated.
    """
    if not output.requirements:
        return output

    # Run all gap analysis functions
    edge_cases = _identify_edge_cases(output.requirements)
    implied_reqs = _identify_implied_requirements(output.requirements)
    pattern_suggestions = _suggest_from_patterns(output.requirements)

    # Combine all gaps
    all_gaps: list[IdentifiedGap] = []
    all_gaps.extend(edge_cases)
    all_gaps.extend(implied_reqs)
    all_gaps.extend(pattern_suggestions)

    # Sort by severity FIRST (critical -> high -> medium -> low)
    severity_order = {
        Severity.CRITICAL: 0,
        Severity.HIGH: 1,
        Severity.MEDIUM: 2,
        Severity.LOW: 3,
    }
    all_gaps.sort(key=lambda g: severity_order.get(g.severity, 4))

    # Re-number gap IDs AFTER sorting so IDs are sequential by severity
    renumbered_gaps: list[IdentifiedGap] = []
    for i, gap in enumerate(all_gaps, start=1):
        renumbered_gap = IdentifiedGap(
            id=f"gap-{i:03d}",
            description=gap.description,
            gap_type=gap.gap_type,
            severity=gap.severity,
            source_requirements=gap.source_requirements,
            rationale=gap.rationale,
        )
        renumbered_gaps.append(renumbered_gap)

    # Log gap analysis results
    logger.info(
        "gap_analysis_complete",
        edge_cases_count=len(edge_cases),
        implied_reqs_count=len(implied_reqs),
        pattern_suggestions_count=len(pattern_suggestions),
        total_gaps=len(renumbered_gaps),
    )

    # Create enhanced output with structured gaps
    return AnalystOutput(
        requirements=output.requirements,
        identified_gaps=output.identified_gaps,
        contradictions=output.contradictions,
        structured_gaps=tuple(renumbered_gaps),
    )
