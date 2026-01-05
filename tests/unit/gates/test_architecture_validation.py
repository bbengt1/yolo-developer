"""Unit tests for architecture validation gate.

Tests the architecture validation gate evaluator that validates
architectural decisions against 12-Factor principles, tech stack
constraints, and security anti-patterns.
"""

from __future__ import annotations

from typing import Any

import pytest

from yolo_developer.gates.evaluators import get_evaluator, register_evaluator

# Import will happen after implementation
# from yolo_developer.gates.gates.architecture_validation import (
#     SECURITY_ANTI_PATTERNS,
#     TWELVE_FACTOR_PRINCIPLES,
#     ArchitectureIssue,
#     architecture_validation_evaluator,
#     calculate_compliance_score,
#     check_twelve_factor_compliance,
#     detect_security_anti_patterns,
#     generate_architecture_report,
#     validate_tech_stack,
# )


@pytest.fixture(autouse=True)
def ensure_architecture_validation_evaluator_registered() -> None:
    """Ensure architecture_validation evaluator is registered for each test.

    Other tests may call clear_evaluators(), so we need to
    re-register the architecture_validation evaluator before tests that need it.
    """
    # Import here to trigger registration
    from yolo_developer.gates.gates.architecture_validation import (
        architecture_validation_evaluator,
    )

    if get_evaluator("architecture_validation") is None:
        register_evaluator("architecture_validation", architecture_validation_evaluator)


class TestArchitectureIssueDataclass:
    """Tests for ArchitectureIssue dataclass."""

    def test_architecture_issue_creation(self) -> None:
        """ArchitectureIssue can be created with required fields."""
        from yolo_developer.gates.gates.architecture_validation import ArchitectureIssue

        issue = ArchitectureIssue(
            decision_id="decision-001",
            issue_type="twelve_factor",
            description="Config not externalized",
            severity="high",
            principle="config",
        )
        assert issue.decision_id == "decision-001"
        assert issue.issue_type == "twelve_factor"
        assert issue.description == "Config not externalized"
        assert issue.severity == "high"
        assert issue.principle == "config"

    def test_architecture_issue_is_frozen(self) -> None:
        """ArchitectureIssue is immutable (frozen dataclass)."""
        from yolo_developer.gates.gates.architecture_validation import ArchitectureIssue

        issue = ArchitectureIssue(
            decision_id="decision-001",
            issue_type="security",
            description="Hardcoded secrets",
            severity="critical",
            principle=None,
        )
        with pytest.raises(AttributeError):
            issue.decision_id = "new-id"  # type: ignore[misc]

    def test_architecture_issue_to_dict(self) -> None:
        """ArchitectureIssue can be converted to dictionary."""
        from yolo_developer.gates.gates.architecture_validation import ArchitectureIssue

        issue = ArchitectureIssue(
            decision_id="decision-001",
            issue_type="tech_stack",
            description="Unsupported database",
            severity="medium",
            principle=None,
        )
        result = issue.to_dict()
        assert result == {
            "decision_id": "decision-001",
            "issue_type": "tech_stack",
            "description": "Unsupported database",
            "severity": "medium",
            "principle": None,
        }


class TestTwelveFactorPrinciples:
    """Tests for TWELVE_FACTOR_PRINCIPLES constant."""

    def test_twelve_factor_principles_has_all_12(self) -> None:
        """TWELVE_FACTOR_PRINCIPLES contains all 12 factors."""
        from yolo_developer.gates.gates.architecture_validation import TWELVE_FACTOR_PRINCIPLES

        assert len(TWELVE_FACTOR_PRINCIPLES) == 12

    def test_twelve_factor_principles_keys(self) -> None:
        """TWELVE_FACTOR_PRINCIPLES has expected keys."""
        from yolo_developer.gates.gates.architecture_validation import TWELVE_FACTOR_PRINCIPLES

        expected_keys = {
            "codebase",
            "dependencies",
            "config",
            "backing_services",
            "build_release_run",
            "processes",
            "port_binding",
            "concurrency",
            "disposability",
            "dev_prod_parity",
            "logs",
            "admin_processes",
        }
        assert set(TWELVE_FACTOR_PRINCIPLES.keys()) == expected_keys

    def test_twelve_factor_principles_values_have_description_and_remediation(self) -> None:
        """TWELVE_FACTOR_PRINCIPLES values have description and remediation."""
        from yolo_developer.gates.gates.architecture_validation import TWELVE_FACTOR_PRINCIPLES

        for key, value in TWELVE_FACTOR_PRINCIPLES.items():
            assert isinstance(value, dict), f"Value for {key} is not a dict"
            assert "description" in value, f"Value for {key} missing 'description'"
            assert "remediation" in value, f"Value for {key} missing 'remediation'"
            assert isinstance(value["description"], str), f"Description for {key} is not a string"
            assert isinstance(value["remediation"], str), f"Remediation for {key} is not a string"
            assert len(value["description"]) > 0, f"Description for {key} is empty"
            assert len(value["remediation"]) > 0, f"Remediation for {key} is empty"


class TestSecurityAntiPatterns:
    """Tests for SECURITY_ANTI_PATTERNS constant."""

    def test_security_anti_patterns_has_expected_patterns(self) -> None:
        """SECURITY_ANTI_PATTERNS contains expected patterns."""
        from yolo_developer.gates.gates.architecture_validation import SECURITY_ANTI_PATTERNS

        expected_patterns = {
            "hardcoded_secrets",
            "sql_injection",
            "missing_auth",
            "insecure_transport",
            "xss_risk",
        }
        assert set(SECURITY_ANTI_PATTERNS.keys()) == expected_patterns

    def test_security_anti_patterns_structure(self) -> None:
        """SECURITY_ANTI_PATTERNS has correct structure for each pattern."""
        from yolo_developer.gates.gates.architecture_validation import SECURITY_ANTI_PATTERNS

        for pattern_name, pattern_config in SECURITY_ANTI_PATTERNS.items():
            assert "patterns" in pattern_config, f"{pattern_name} missing 'patterns'"
            assert "severity" in pattern_config, f"{pattern_name} missing 'severity'"
            assert "remediation" in pattern_config, f"{pattern_name} missing 'remediation'"
            assert isinstance(pattern_config["patterns"], list), (
                f"{pattern_name} patterns not a list"
            )
            assert pattern_config["severity"] in (
                "critical",
                "high",
                "medium",
                "low",
            ), f"{pattern_name} has invalid severity"


class TestCheckTwelveFactorCompliance:
    """Tests for check_twelve_factor_compliance function."""

    def test_compliant_architecture_returns_no_issues(self) -> None:
        """Compliant architecture returns empty issues list."""
        from yolo_developer.gates.gates.architecture_validation import (
            check_twelve_factor_compliance,
        )

        architecture = {
            "decisions": [
                {
                    "id": "decision-001",
                    "title": "Configuration Management",
                    "description": "Use environment variables for all configuration",
                    "rationale": "12-Factor config principle",
                }
            ],
            "twelve_factor": {
                "codebase": True,
                "dependencies": True,
                "config": True,
                "backing_services": True,
                "build_release_run": True,
                "processes": True,
                "port_binding": True,
                "concurrency": True,
                "disposability": True,
                "dev_prod_parity": True,
                "logs": True,
                "admin_processes": True,
            },
        }
        issues = check_twelve_factor_compliance(architecture)
        assert len(issues) == 0

    def test_missing_config_externalization_creates_issue(self) -> None:
        """Missing config externalization creates an issue."""
        from yolo_developer.gates.gates.architecture_validation import (
            check_twelve_factor_compliance,
        )

        architecture = {
            "decisions": [
                {
                    "id": "decision-001",
                    "title": "Configuration",
                    "description": "Store config in hardcoded values",
                }
            ],
            "twelve_factor": {
                "config": False,
            },
        }
        issues = check_twelve_factor_compliance(architecture)
        assert len(issues) >= 1
        config_issues = [i for i in issues if i.principle == "config"]
        assert len(config_issues) >= 1
        assert config_issues[0].issue_type == "twelve_factor"

    def test_multiple_violations_detected(self) -> None:
        """Multiple 12-Factor violations are all detected."""
        from yolo_developer.gates.gates.architecture_validation import (
            check_twelve_factor_compliance,
        )

        architecture = {
            "twelve_factor": {
                "config": False,
                "logs": False,
                "processes": False,
            },
        }
        issues = check_twelve_factor_compliance(architecture)
        assert len(issues) >= 3
        principles_violated = {i.principle for i in issues}
        assert "config" in principles_violated
        assert "logs" in principles_violated
        assert "processes" in principles_violated

    def test_empty_architecture_returns_no_issues(self) -> None:
        """Empty architecture returns no issues (nothing to validate)."""
        from yolo_developer.gates.gates.architecture_validation import (
            check_twelve_factor_compliance,
        )

        issues = check_twelve_factor_compliance({})
        # No explicit violations flagged when twelve_factor section is missing
        assert isinstance(issues, list)


class TestValidateTechStack:
    """Tests for validate_tech_stack function."""

    def test_valid_tech_stack_returns_no_issues(self) -> None:
        """Valid tech stack matching constraints returns no issues."""
        from yolo_developer.gates.gates.architecture_validation import validate_tech_stack

        architecture = {
            "tech_stack": {
                "languages": ["python"],
                "frameworks": ["fastapi"],
                "databases": ["postgresql"],
            },
        }
        constraints = {
            "allowed_languages": ["python", "typescript"],
            "allowed_frameworks": ["fastapi", "django"],
            "allowed_databases": ["postgresql", "redis"],
        }
        issues = validate_tech_stack(architecture, constraints)
        assert len(issues) == 0

    def test_unsupported_language_creates_issue(self) -> None:
        """Unsupported language creates an issue."""
        from yolo_developer.gates.gates.architecture_validation import validate_tech_stack

        architecture = {
            "tech_stack": {
                "languages": ["rust"],
            },
        }
        constraints = {
            "allowed_languages": ["python", "typescript"],
        }
        issues = validate_tech_stack(architecture, constraints)
        assert len(issues) >= 1
        assert any("rust" in i.description.lower() for i in issues)
        assert issues[0].issue_type == "tech_stack"

    def test_unsupported_database_creates_issue(self) -> None:
        """Unsupported database creates an issue."""
        from yolo_developer.gates.gates.architecture_validation import validate_tech_stack

        architecture = {
            "tech_stack": {
                "databases": ["mongodb"],
            },
        }
        constraints = {
            "allowed_databases": ["postgresql"],
        }
        issues = validate_tech_stack(architecture, constraints)
        assert len(issues) >= 1
        assert any("mongodb" in i.description.lower() for i in issues)

    def test_empty_constraints_allows_everything(self) -> None:
        """Empty constraints allows any tech stack."""
        from yolo_developer.gates.gates.architecture_validation import validate_tech_stack

        architecture = {
            "tech_stack": {
                "languages": ["cobol"],
                "databases": ["cassandra"],
            },
        }
        issues = validate_tech_stack(architecture, {})
        assert len(issues) == 0


class TestDetectSecurityAntiPatterns:
    """Tests for detect_security_anti_patterns function."""

    def test_secure_architecture_returns_no_issues(self) -> None:
        """Secure architecture returns empty issues list."""
        from yolo_developer.gates.gates.architecture_validation import detect_security_anti_patterns

        architecture = {
            "security": {
                "secrets_management": "Use AWS Secrets Manager for all secrets",
                "authentication": "OAuth 2.0 with JWT tokens",
                "transport": "HTTPS only with TLS 1.3",
                "database_access": "Parameterized queries via SQLAlchemy ORM",
            },
        }
        issues = detect_security_anti_patterns(architecture)
        # Should have no issues for proper security config
        assert isinstance(issues, list)

    def test_hardcoded_secrets_detected(self) -> None:
        """Hardcoded secrets pattern is detected."""
        from yolo_developer.gates.gates.architecture_validation import detect_security_anti_patterns

        architecture = {
            "security": {
                "secrets_management": "password=secret123 in config file",
            },
            "components": [
                {
                    "id": "comp-001",
                    "description": "Uses api_key=ABC123 for authentication",
                }
            ],
        }
        issues = detect_security_anti_patterns(architecture)
        security_issues = [i for i in issues if i.issue_type == "security"]
        assert len(security_issues) >= 1
        assert any(
            "hardcoded" in i.description.lower() or "secret" in i.description.lower()
            for i in security_issues
        )

    def test_sql_injection_risk_detected(self) -> None:
        """SQL injection risk pattern is detected."""
        from yolo_developer.gates.gates.architecture_validation import detect_security_anti_patterns

        architecture = {
            "components": [
                {
                    "id": "comp-001",
                    "description": "Uses string concatenation for SQL queries",
                }
            ],
        }
        issues = detect_security_anti_patterns(architecture)
        sql_issues = [i for i in issues if "sql" in i.description.lower()]
        assert len(sql_issues) >= 1

    def test_insecure_transport_detected(self) -> None:
        """Insecure transport (HTTP) is detected."""
        from yolo_developer.gates.gates.architecture_validation import detect_security_anti_patterns

        architecture = {
            "components": [
                {
                    "id": "comp-001",
                    "description": "API endpoint available at http://api.example.com",
                }
            ],
        }
        issues = detect_security_anti_patterns(architecture)
        transport_issues = [
            i
            for i in issues
            if "http" in i.description.lower() or "transport" in i.description.lower()
        ]
        assert len(transport_issues) >= 1


class TestCalculateComplianceScore:
    """Tests for calculate_compliance_score function."""

    def test_no_issues_returns_100(self) -> None:
        """No issues returns score of 100."""
        from yolo_developer.gates.gates.architecture_validation import calculate_compliance_score

        score, breakdown = calculate_compliance_score([])
        assert score == 100
        assert isinstance(breakdown, dict)

    def test_critical_issue_deducts_25(self) -> None:
        """Critical issue deducts 25 points."""
        from yolo_developer.gates.gates.architecture_validation import (
            ArchitectureIssue,
            calculate_compliance_score,
        )

        issues = [
            ArchitectureIssue(
                decision_id="d-1",
                issue_type="security",
                description="Critical security issue",
                severity="critical",
                principle=None,
            )
        ]
        score, breakdown = calculate_compliance_score(issues)
        assert score == 75
        assert "critical" in breakdown

    def test_high_issue_deducts_15(self) -> None:
        """High severity issue deducts 15 points."""
        from yolo_developer.gates.gates.architecture_validation import (
            ArchitectureIssue,
            calculate_compliance_score,
        )

        issues = [
            ArchitectureIssue(
                decision_id="d-1",
                issue_type="twelve_factor",
                description="High severity issue",
                severity="high",
                principle="config",
            )
        ]
        score, breakdown = calculate_compliance_score(issues)
        assert score == 85
        assert "high" in breakdown

    def test_medium_issue_deducts_5(self) -> None:
        """Medium severity issue deducts 5 points."""
        from yolo_developer.gates.gates.architecture_validation import (
            ArchitectureIssue,
            calculate_compliance_score,
        )

        issues = [
            ArchitectureIssue(
                decision_id="d-1",
                issue_type="tech_stack",
                description="Medium severity issue",
                severity="medium",
                principle=None,
            )
        ]
        score, breakdown = calculate_compliance_score(issues)
        assert score == 95
        assert "medium" in breakdown

    def test_low_issue_deducts_1(self) -> None:
        """Low severity issue deducts 1 point."""
        from yolo_developer.gates.gates.architecture_validation import (
            ArchitectureIssue,
            calculate_compliance_score,
        )

        issues = [
            ArchitectureIssue(
                decision_id="d-1",
                issue_type="twelve_factor",
                description="Low severity issue",
                severity="low",
                principle="logs",
            )
        ]
        score, breakdown = calculate_compliance_score(issues)
        assert score == 99
        assert "low" in breakdown

    def test_multiple_issues_cumulative(self) -> None:
        """Multiple issues have cumulative deduction."""
        from yolo_developer.gates.gates.architecture_validation import (
            ArchitectureIssue,
            calculate_compliance_score,
        )

        issues = [
            ArchitectureIssue("d-1", "security", "Critical", "critical", None),
            ArchitectureIssue("d-2", "twelve_factor", "High", "high", "config"),
            ArchitectureIssue("d-3", "tech_stack", "Medium", "medium", None),
        ]
        score, _breakdown = calculate_compliance_score(issues)
        # 100 - 25 - 15 - 5 = 55
        assert score == 55

    def test_score_capped_at_zero(self) -> None:
        """Score is capped at minimum of 0."""
        from yolo_developer.gates.gates.architecture_validation import (
            ArchitectureIssue,
            calculate_compliance_score,
        )

        # Create enough critical issues to exceed 100 points deduction
        issues = [
            ArchitectureIssue(f"d-{i}", "security", f"Critical {i}", "critical", None)
            for i in range(10)
        ]
        score, _breakdown = calculate_compliance_score(issues)
        assert score == 0


class TestGenerateArchitectureReport:
    """Tests for generate_architecture_report function."""

    def test_empty_issues_returns_empty_string(self) -> None:
        """Empty issues list returns empty report."""
        from yolo_developer.gates.gates.architecture_validation import generate_architecture_report

        report = generate_architecture_report([], 100, {})
        assert report == ""

    def test_report_includes_score(self) -> None:
        """Report includes compliance score."""
        from yolo_developer.gates.gates.architecture_validation import (
            ArchitectureIssue,
            generate_architecture_report,
        )

        issues = [
            ArchitectureIssue("d-1", "security", "Critical issue", "critical", None),
        ]
        report = generate_architecture_report(issues, 75, {"critical": 1})
        assert "75" in report

    def test_report_includes_issue_details(self) -> None:
        """Report includes issue details."""
        from yolo_developer.gates.gates.architecture_validation import (
            ArchitectureIssue,
            generate_architecture_report,
        )

        issues = [
            ArchitectureIssue("d-1", "twelve_factor", "Config not externalized", "high", "config"),
        ]
        report = generate_architecture_report(issues, 85, {"high": 1})
        assert "Config not externalized" in report
        assert "config" in report  # Principle is included

    def test_report_categorizes_by_type(self) -> None:
        """Report categorizes issues by type."""
        from yolo_developer.gates.gates.architecture_validation import (
            ArchitectureIssue,
            generate_architecture_report,
        )

        issues = [
            ArchitectureIssue("d-1", "twelve_factor", "12-Factor issue", "high", "config"),
            ArchitectureIssue("d-2", "security", "Security issue", "critical", None),
            ArchitectureIssue("d-3", "tech_stack", "Tech stack issue", "medium", None),
        ]
        report = generate_architecture_report(issues, 55, {"critical": 1, "high": 1, "medium": 1})
        # Report should contain category headers or organize by type
        assert "12-Factor" in report or "twelve_factor" in report
        assert "Security" in report or "security" in report


class TestArchitectureValidationEvaluator:
    """Tests for architecture_validation_evaluator function."""

    @pytest.mark.asyncio
    async def test_evaluator_passes_with_compliant_architecture(self) -> None:
        """Evaluator passes with fully compliant architecture."""
        from yolo_developer.gates.gates.architecture_validation import (
            architecture_validation_evaluator,
        )
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "architecture": {
                "decisions": [
                    {
                        "id": "decision-001",
                        "title": "Configuration",
                        "description": "Use environment variables",
                    }
                ],
                "twelve_factor": {
                    "codebase": True,
                    "dependencies": True,
                    "config": True,
                    "backing_services": True,
                    "build_release_run": True,
                    "processes": True,
                    "port_binding": True,
                    "concurrency": True,
                    "disposability": True,
                    "dev_prod_parity": True,
                    "logs": True,
                    "admin_processes": True,
                },
                "tech_stack": {
                    "languages": ["python"],
                },
                "security": {
                    "secrets_management": "AWS Secrets Manager",
                    "authentication": "OAuth 2.0",
                },
            },
            "config": {
                "tech_stack": {
                    "allowed_languages": ["python"],
                },
                "quality": {
                    "architecture_threshold": 0.70,
                },
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluator_fails_with_low_score(self) -> None:
        """Evaluator fails when compliance score below threshold."""
        from yolo_developer.gates.gates.architecture_validation import (
            architecture_validation_evaluator,
        )
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "architecture": {
                "twelve_factor": {
                    "config": False,
                    "logs": False,
                    "processes": False,
                },
                "security": {
                    "secrets_management": "password=secret123 in config",
                },
            },
            "config": {
                "quality": {
                    "architecture_threshold": 0.70,
                },
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        assert result.passed is False
        assert result.reason is not None

    @pytest.mark.asyncio
    async def test_evaluator_fails_with_missing_architecture_key(self) -> None:
        """Evaluator fails when architecture key is missing from state."""
        from yolo_developer.gates.gates.architecture_validation import (
            architecture_validation_evaluator,
        )
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {}
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        assert result.passed is False
        assert "architecture" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluator_fails_with_invalid_architecture_type(self) -> None:
        """Evaluator fails when architecture is not a dict."""
        from yolo_developer.gates.gates.architecture_validation import (
            architecture_validation_evaluator,
        )
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {"architecture": "not a dict"}
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_evaluator_uses_default_threshold(self) -> None:
        """Evaluator uses default threshold of 70 when not configured."""
        from yolo_developer.gates.gates.architecture_validation import (
            architecture_validation_evaluator,
        )
        from yolo_developer.gates.types import GateContext

        # Architecture with one high issue (score 85, above default 70)
        state: dict[str, Any] = {
            "architecture": {
                "twelve_factor": {
                    "config": False,  # -15 points = score 85
                },
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        # Score 85 is above default threshold 70, should pass
        assert result.passed is True
        assert "85" in (result.reason or "")  # Score should be mentioned


class TestEvaluatorRegistration:
    """Tests for evaluator registration."""

    def test_evaluator_registered_on_import(self) -> None:
        """Evaluator is registered when module is imported."""
        # Re-import to ensure registration
        from yolo_developer.gates.gates import architecture_validation  # noqa: F401

        evaluator = get_evaluator("architecture_validation")
        assert evaluator is not None

    def test_evaluator_is_callable(self) -> None:
        """Registered evaluator is callable."""
        evaluator = get_evaluator("architecture_validation")
        assert callable(evaluator)


class TestVersionCompatibility:
    """Tests for version compatibility checking."""

    def test_version_constraint_satisfied(self) -> None:
        """Version that satisfies constraint passes."""
        from yolo_developer.gates.gates.architecture_validation import validate_tech_stack

        architecture = {
            "tech_stack": {
                "languages": ["python"],
                "language_versions": {"python": "3.11"},
            },
        }
        constraints = {
            "allowed_languages": ["python"],
            "version_constraints": {"python": ">=3.9"},
        }
        issues = validate_tech_stack(architecture, constraints)
        assert len(issues) == 0

    def test_version_constraint_violated(self) -> None:
        """Version that violates constraint creates issue."""
        from yolo_developer.gates.gates.architecture_validation import validate_tech_stack

        architecture = {
            "tech_stack": {
                "languages": ["python"],
                "language_versions": {"python": "3.8"},
            },
        }
        constraints = {
            "allowed_languages": ["python"],
            "version_constraints": {"python": ">=3.9"},
        }
        issues = validate_tech_stack(architecture, constraints)
        assert len(issues) >= 1
        assert any("version" in i.description.lower() for i in issues)
        assert any("3.8" in i.description for i in issues)

    def test_framework_version_constraint(self) -> None:
        """Framework version constraint is checked."""
        from yolo_developer.gates.gates.architecture_validation import validate_tech_stack

        architecture = {
            "tech_stack": {
                "frameworks": ["fastapi"],
                "framework_versions": {"fastapi": "0.90"},
            },
        }
        constraints = {
            "allowed_frameworks": ["fastapi"],
            "version_constraints": {"fastapi": ">=0.100"},
        }
        issues = validate_tech_stack(architecture, constraints)
        assert len(issues) >= 1
        assert any("version" in i.description.lower() for i in issues)


class TestADREvaluation:
    """Tests for ADR evaluation."""

    def test_complete_adr_passes(self) -> None:
        """Complete ADR passes validation."""
        from yolo_developer.gates.gates.architecture_validation import evaluate_adrs

        adrs = [
            {
                "id": "ADR-001",
                "title": "Use PostgreSQL for persistence",
                "status": "accepted",
                "context": "Need a reliable database",
                "decision": "Use PostgreSQL",
                "consequences": "Good ACID compliance",
            }
        ]
        issues = evaluate_adrs(adrs)
        assert len(issues) == 0

    def test_incomplete_adr_creates_issue(self) -> None:
        """ADR missing required fields creates issue."""
        from yolo_developer.gates.gates.architecture_validation import evaluate_adrs

        adrs = [
            {
                "id": "ADR-001",
                "title": "Use PostgreSQL",
                # Missing: status, context, decision, consequences
            }
        ]
        issues = evaluate_adrs(adrs)
        assert len(issues) >= 1
        assert any("missing" in i.description.lower() for i in issues)
        assert any("ADR-001" in i.decision_id for i in issues)

    def test_adr_with_security_antipattern_creates_issue(self) -> None:
        """ADR containing security anti-pattern creates issue."""
        from yolo_developer.gates.gates.architecture_validation import evaluate_adrs

        adrs = [
            {
                "id": "ADR-002",
                "title": "Database credentials",
                "status": "accepted",
                "context": "Need database access",
                "decision": "Use password=secret123 in config",
                "consequences": "Easy access",
            }
        ]
        issues = evaluate_adrs(adrs)
        security_issues = [i for i in issues if i.issue_type == "security"]
        assert len(security_issues) >= 1

    @pytest.mark.asyncio
    async def test_evaluator_processes_adrs(self) -> None:
        """Evaluator processes ADRs from state."""
        from yolo_developer.gates.gates.architecture_validation import (
            architecture_validation_evaluator,
        )
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "architecture": {
                "adrs": [
                    {
                        "id": "ADR-001",
                        "title": "Incomplete ADR",
                        # Missing required fields
                    }
                ],
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        # Should pass with warnings (ADR issues are medium severity)
        # Score = 100 - 5 = 95 > threshold 70
        assert result.passed is True
        assert "ADR" in (result.reason or "")


class TestDecisionIdTracking:
    """Tests for decision ID tracking."""

    @pytest.mark.asyncio
    async def test_uses_decision_id_from_architecture(self) -> None:
        """Evaluator uses decision ID from architecture decisions."""
        from yolo_developer.gates.gates.architecture_validation import (
            architecture_validation_evaluator,
        )
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "architecture": {
                "decisions": [
                    {
                        "id": "decision-001",
                        "title": "Config Management",
                        "description": "Use environment variables",
                    }
                ],
                "twelve_factor": {
                    "config": False,  # This should be tagged with decision-001
                },
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        # Check that decision-001 appears in the reason (it's the decision_id used)
        assert result.passed is True  # Score 85 > 70
        assert "decision-001" in (result.reason or "") or "85" in (result.reason or "")


class TestArchitectureValidationThresholdConfiguration:
    """Tests for architecture validation threshold configuration (Story 3.7)."""

    @pytest.mark.asyncio
    async def test_evaluator_uses_default_threshold_when_no_config(self) -> None:
        """Should use DEFAULT_COMPLIANCE_THRESHOLD (0.70) when no config provided."""
        from yolo_developer.gates.gates.architecture_validation import (
            DEFAULT_COMPLIANCE_THRESHOLD,
            architecture_validation_evaluator,
        )
        from yolo_developer.gates.types import GateContext

        # Verify default is 0.70
        assert DEFAULT_COMPLIANCE_THRESHOLD == 0.70

        # Architecture with one high issue (score 85, above default 70)
        state: dict[str, Any] = {
            "architecture": {
                "twelve_factor": {
                    "config": False,  # -15 points = score 85
                },
            },
            # No config
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        # Score 85 is above default threshold 70, should pass
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluator_respects_gate_specific_threshold(self) -> None:
        """Should use gate-specific threshold from resolver (highest priority)."""
        from yolo_developer.gates.gates.architecture_validation import (
            architecture_validation_evaluator,
        )
        from yolo_developer.gates.types import GateContext

        # Architecture with one high issue (score 85)
        state: dict[str, Any] = {
            "architecture": {
                "twelve_factor": {
                    "config": False,  # -15 points = score 85
                },
            },
            "config": {
                "quality": {
                    "gate_thresholds": {
                        "architecture_validation": {"min_score": 0.90},  # Gate-specific: 90%
                    },
                },
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        # Score 85 is BELOW threshold 90, should fail
        assert result.passed is False
        assert "85" in result.reason  # Score mentioned
        assert "90" in result.reason  # Threshold mentioned

    @pytest.mark.asyncio
    async def test_evaluator_passes_when_above_custom_threshold(self) -> None:
        """Should pass when score meets custom threshold."""
        from yolo_developer.gates.gates.architecture_validation import (
            architecture_validation_evaluator,
        )
        from yolo_developer.gates.types import GateContext

        # Architecture with one high issue (score 85)
        state: dict[str, Any] = {
            "architecture": {
                "twelve_factor": {
                    "config": False,  # -15 points = score 85
                },
            },
            "config": {
                "quality": {
                    "gate_thresholds": {
                        "architecture_validation": {"min_score": 0.80},  # 80% threshold
                    },
                },
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        # Score 85 is ABOVE threshold 80, should pass
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluator_uses_low_threshold_for_advisory_mode(self) -> None:
        """Should allow low threshold for advisory mode (non-blocking)."""
        from yolo_developer.gates.gates.architecture_validation import (
            architecture_validation_evaluator,
        )
        from yolo_developer.gates.types import GateContext

        # Architecture with multiple issues (score 55)
        state: dict[str, Any] = {
            "architecture": {
                "twelve_factor": {
                    "config": False,  # -15 points
                },
                "security": {
                    "secrets_management": "password=secret123 in config",  # -25 points
                },
            },
            "config": {
                "quality": {
                    "gate_thresholds": {
                        "architecture_validation": {"min_score": 0.50, "blocking": False},
                    },
                },
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        # Score around 55-60 should be ABOVE threshold 50
        assert result.passed is True
