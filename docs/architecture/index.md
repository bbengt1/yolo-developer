---
layout: default
title: Architecture
nav_order: 8
has_children: true
---

# Architecture
{: .no_toc }

Deep dive into YOLO Developer's multi-agent architecture, orchestration, memory, and quality systems.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## System Overview

YOLO Developer is built on a multi-agent architecture where specialized AI agents collaborate through a state machine orchestrated by LangGraph.

```
┌────────────────────────────────────────────────────────────────┐
│                        YOLO Developer                          │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  CLI     │  │  SDK     │  │  MCP     │  │  API     │       │
│  │ (Typer)  │  │ (Python) │  │(FastMCP) │  │ (REST)   │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │              │
│       └─────────────┴──────┬──────┴─────────────┘              │
│                            │                                   │
│  ┌─────────────────────────▼─────────────────────────────────┐ │
│  │                   Orchestrator                             │ │
│  │                  (LangGraph Engine)                        │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │                  State Machine                       │  │ │
│  │  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │  │ │
│  │  │  │Analyst│→│  PM  │→│Arch. │→│ Dev  │→│ TEA  │      │  │ │
│  │  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘      │  │ │
│  │  │                    ↑                                 │  │ │
│  │  │              ┌─────┴─────┐                           │  │ │
│  │  │              │    SM     │                           │  │ │
│  │  │              │(Orchestr.)│                           │  │ │
│  │  │              └───────────┘                           │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                   │
│       ┌────────────────────┼────────────────────┐              │
│       │                    │                    │              │
│  ┌────▼────┐         ┌─────▼─────┐        ┌─────▼─────┐       │
│  │ Memory  │         │  Quality  │        │   Audit   │       │
│  │(ChromaDB│         │   Gates   │        │   Trail   │       │
│  └─────────┘         └───────────┘        └───────────┘       │
└────────────────────────────────────────────────────────────────┘
```

---

## Agents

### Agent Overview

| Agent | Role | Primary Functions |
|:------|:-----|:------------------|
| **Analyst** | Requirements Engineering | Crystallization, ambiguity detection, contradiction flagging |
| **PM** | Product Management | Story generation, prioritization, dependency mapping |
| **Architect** | System Design | ADR generation, 12-Factor analysis, risk assessment |
| **Dev** | Development | Code generation, test writing, documentation |
| **TEA** | Test Engineering | Coverage validation, risk categorization, testability audit |
| **SM** | Scrum Master | Orchestration, conflict mediation, human escalation |

---

### Analyst Agent

**Purpose:** Transform vague requirements into precise, actionable specifications.

**Capabilities:**
- Requirement crystallization
- Ambiguity detection with clarification questions
- Contradiction flagging
- Requirement categorization (functional, non-functional, constraint)
- Implementability validation

**Input:**
```json
{
  "raw_requirements": "Build a fast user management system",
  "context": { "domain": "web", "language": "python" }
}
```

**Output:**
```json
{
  "crystallized_requirements": [
    {
      "id": "REQ-001",
      "original": "Build a fast user management system",
      "crystallized": "Create a REST API for user CRUD operations with response times < 200ms",
      "category": "functional",
      "confidence": 0.92
    }
  ],
  "ambiguities": [
    {
      "text": "fast",
      "question": "What response time threshold defines 'fast'?",
      "suggestions": ["< 100ms", "< 200ms", "< 500ms"]
    }
  ],
  "contradictions": []
}
```

---

### PM Agent

**Purpose:** Transform requirements into implementable user stories with acceptance criteria.

**Capabilities:**
- Story generation with acceptance criteria
- AC testability validation
- Priority assignment (MoSCoW)
- Dependency identification
- Epic breakdown

**Input:**
```json
{
  "requirements": [
    { "id": "REQ-001", "text": "User registration with email verification" }
  ]
}
```

**Output:**
```json
{
  "stories": [
    {
      "id": "US-001",
      "title": "User Registration",
      "description": "As a new user, I want to register with my email...",
      "acceptance_criteria": [
        "Given a valid email, when user submits registration, then account is created",
        "Given registration success, when system processes, then verification email is sent"
      ],
      "priority": "must-have",
      "story_points": 5,
      "dependencies": []
    }
  ],
  "epics": [
    { "id": "EPIC-001", "title": "User Management", "stories": ["US-001", "US-002"] }
  ]
}
```

---

### Architect Agent

**Purpose:** Design system architecture that meets requirements while following best practices.

**Capabilities:**
- 12-Factor App compliance analysis
- Architecture Decision Record (ADR) generation
- Quality attribute evaluation (performance, security, reliability, scalability, maintainability)
- Technical risk identification
- Tech stack constraint validation
- ATAM (Architecture Tradeoff Analysis Method) review
- Pattern matching against codebase conventions

**Input:**
```json
{
  "stories": [...],
  "requirements": [...],
  "tech_constraints": ["postgresql", "python", "fastapi"]
}
```

**Output:**
```json
{
  "adrs": [
    {
      "id": "ADR-001",
      "title": "JWT Authentication Strategy",
      "status": "accepted",
      "context": "Need stateless authentication for API",
      "decision": "Use JWT with short expiry and refresh tokens",
      "consequences": {
        "positive": ["Stateless", "Scalable", "Standard"],
        "negative": ["Cannot revoke before expiry"]
      }
    }
  ],
  "twelve_factor_analysis": {
    "compliant": ["codebase", "dependencies", "config", "backing_services"],
    "non_compliant": [],
    "recommendations": []
  },
  "quality_evaluation": {
    "performance": { "score": 0.85, "notes": "Expected < 200ms response" },
    "security": { "score": 0.90, "notes": "JWT + HTTPS + rate limiting" }
  },
  "risks": [
    {
      "id": "RISK-001",
      "description": "Token revocation complexity",
      "probability": "medium",
      "impact": "low",
      "mitigation": "Implement token blacklist for critical operations"
    }
  ]
}
```

---

### Dev Agent

**Purpose:** Generate production-quality code that implements user stories.

**Capabilities:**
- Code generation following learned patterns
- Unit test generation
- Integration test generation
- Documentation generation
- Definition of Done validation
- Communicative commit messages

**Input:**
```json
{
  "story": {
    "id": "US-001",
    "title": "User Registration",
    "acceptance_criteria": [...]
  },
  "architecture": { "adrs": [...], "patterns": [...] },
  "codebase_patterns": { "naming": "snake_case", "structure": "src/app/..." }
}
```

**Output:**
```json
{
  "files": [
    {
      "path": "src/app/models/user.py",
      "content": "...",
      "type": "source"
    },
    {
      "path": "tests/test_user_registration.py",
      "content": "...",
      "type": "test"
    }
  ],
  "test_coverage": 0.92,
  "dod_validation": {
    "passing": true,
    "criteria_met": [
      "Code reviewed",
      "Tests passing",
      "Documentation updated"
    ]
  },
  "commit_message": "feat(auth): implement user registration with email verification\n\nAdds POST /api/v1/users/register endpoint with:\n- Email validation\n- Password hashing with bcrypt\n- Verification email trigger\n\nCloses US-001"
}
```

---

### TEA Agent (Test Engineering Agent)

**Purpose:** Validate test coverage and identify quality risks.

**Capabilities:**
- Test coverage validation
- Test suite execution
- Confidence scoring
- Risk categorization
- Testability audit
- Deployment blocking decisions
- Gap analysis reports

**Input:**
```json
{
  "test_results": { "passed": 45, "failed": 0, "coverage": 0.87 },
  "stories": [...],
  "code_changes": [...]
}
```

**Output:**
```json
{
  "coverage_report": {
    "overall": 0.87,
    "by_module": {
      "auth": 0.92,
      "users": 0.85,
      "utils": 0.78
    }
  },
  "risk_assessment": {
    "high_risk": [],
    "medium_risk": ["utils module under-tested"],
    "low_risk": []
  },
  "testability_audit": {
    "score": 0.89,
    "issues": [
      { "file": "utils.py", "issue": "Complex function needs refactoring" }
    ]
  },
  "deployment_decision": {
    "approved": true,
    "confidence": 0.91,
    "notes": "All critical paths covered"
  }
}
```

---

### SM Agent (Scrum Master)

**Purpose:** Orchestrate the development sprint and manage agent collaboration.

**Capabilities:**
- Sprint planning and tracking
- Task delegation
- Health monitoring
- Circular logic detection
- Conflict mediation
- Agent handoff management
- Emergency protocols
- Human escalation
- Rollback coordination

**Runs continuously throughout the sprint, monitoring all agent activity.**

**Example Decisions:**
```json
{
  "decisions": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "type": "handoff",
      "from": "analyst",
      "to": "pm",
      "reason": "Requirements crystallized, ready for story generation"
    },
    {
      "timestamp": "2024-01-15T10:45:00Z",
      "type": "conflict_mediation",
      "agents": ["architect", "dev"],
      "issue": "Disagreement on database schema",
      "resolution": "Adopted architect recommendation per ADR-002"
    },
    {
      "timestamp": "2024-01-15T11:00:00Z",
      "type": "human_escalation",
      "reason": "Security requirement needs stakeholder approval",
      "status": "pending"
    }
  ]
}
```

---

## Orchestration

### LangGraph Workflow

The orchestrator uses LangGraph to define the agent workflow as a state machine:

```python
from langgraph.graph import StateGraph, END

# Define the workflow
workflow = StateGraph(SprintState)

# Add nodes (agents)
workflow.add_node("analyst", analyst_node)
workflow.add_node("pm", pm_node)
workflow.add_node("architect", architect_node)
workflow.add_node("dev", dev_node)
workflow.add_node("tea", tea_node)
workflow.add_node("sm", sm_node)

# Define edges (transitions)
workflow.add_edge("analyst", "pm")
workflow.add_edge("pm", "architect")
workflow.add_edge("architect", "dev")
workflow.add_edge("dev", "tea")

# Conditional edges
workflow.add_conditional_edges(
    "tea",
    should_continue,
    {
        "next_story": "dev",
        "complete": END,
        "escalate": "sm",
    }
)

# SM monitors throughout
workflow.add_edge("sm", "analyst")  # Can restart from any point
```

### State Schema

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class SprintState(TypedDict):
    # Seed input
    seed_id: str
    requirements: list[Requirement]

    # Agent outputs
    crystallized_requirements: list[CrystallizedRequirement]
    stories: list[UserStory]
    architecture: ArchitectureDecision
    generated_code: list[GeneratedFile]
    test_results: TestResults

    # Orchestration
    current_agent: str
    current_story: str | None
    iteration_count: int

    # Quality
    quality_gates: dict[str, GateResult]

    # Audit
    messages: Annotated[list, add_messages]
    decisions: list[Decision]
```

### Conditional Routing

```python
def should_continue(state: SprintState) -> str:
    """Determine next step after TEA validation."""
    if state["test_results"].coverage < threshold:
        return "escalate"  # Needs human review

    pending_stories = get_pending_stories(state)
    if pending_stories:
        return "next_story"  # Continue with next story

    return "complete"  # Sprint done
```

---

## Quality Gates

### Gate Framework

Quality gates are enforced via decorators on agent nodes:

```python
from yolo_developer.gates import quality_gate, GateResult

@quality_gate("testability", blocking=True)
async def pm_node(state: SprintState) -> SprintState:
    """PM agent with testability gate."""
    stories = await generate_stories(state)

    # Gate automatically validates output
    return {"stories": stories}
```

### Built-in Gates

| Gate | Description | Default Threshold |
|:-----|:------------|:------------------|
| `testability` | Stories have testable ACs | 0.7 |
| `ac_measurability` | ACs have measurable outcomes | 0.7 |
| `architecture_validation` | Design follows patterns | 0.7 |
| `definition_of_done` | Work meets DoD criteria | 0.7 |

### Gate Evaluation

```python
@dataclass
class GateResult:
    gate_name: str
    score: float  # 0.0 to 1.0
    threshold: float
    passing: bool
    details: dict
    remediation: str | None
```

### Blocking vs Warning Gates

**Blocking gates** halt progress if they fail:
```yaml
quality:
  blocking_gates:
    - testability
    - architecture_validation
```

**Warning gates** log issues but allow progress:
```yaml
quality:
  warning_gates:
    - code_complexity
```

---

## Memory System

### Vector Storage (ChromaDB)

Decisions and patterns are stored in ChromaDB for semantic retrieval:

```python
from yolo_developer.memory import create_memory_store

store = await create_memory_store(config)

# Store a decision
await store.store_decision(
    decision=decision,
    embedding=embedding,
    metadata={
        "agent": "architect",
        "type": "adr",
        "timestamp": datetime.now(),
    }
)

# Query similar decisions
similar = await store.query_similar(
    query="authentication strategy",
    limit=5,
    threshold=0.7,
)
```

### Pattern Learning

The system learns codebase patterns automatically:

```python
from yolo_developer.memory import PatternLearner

learner = PatternLearner(store)

# Learn from existing codebase
patterns = await learner.learn_patterns_from_codebase("/path/to/project")

# Patterns detected:
# - Naming conventions (snake_case, camelCase)
# - Directory structure
# - Test organization
# - Import patterns
# - Error handling patterns
```

### Session Persistence

Context is preserved across agent handoffs:

```python
# Save session state
await store.save_session(session_id, state)

# Restore session
state = await store.restore_session(session_id)
```

### Project Isolation

Multi-tenant support with data separation:

```python
# Each project has isolated storage
store_a = await create_memory_store(config, project_id="project-a")
store_b = await create_memory_store(config, project_id="project-b")
```

---

## Audit Trail

### Decision Logging

Every agent decision is logged:

```python
from yolo_developer.audit import AuditLogger

logger = AuditLogger(config)

await logger.log_decision(
    agent="architect",
    decision="Selected JWT for authentication",
    context={
        "alternatives_considered": ["session", "oauth"],
        "rationale": "Stateless requirement",
    },
    confidence=0.92,
    tokens_used=1234,
)
```

### Requirement Traceability

Track requirements through implementation:

```
REQ-001 (User registration)
    ↓
US-001 (User Registration story)
    ↓
ADR-001 (Auth strategy decision)
    ↓
src/auth/register.py (Implementation)
    ↓
tests/test_register.py (Tests)
```

### Token Cost Tracking

```python
audit = await client.get_audit_trail()

total_tokens = sum(e.tokens_used for e in audit.entries)
total_cost = sum(e.cost_usd for e in audit.entries)

print(f"Total tokens: {total_tokens}")
print(f"Total cost: ${total_cost:.2f}")
```

---

## File Structure

```
src/yolo_developer/
├── __init__.py              # Package exports
├── __version__.py           # Version info
│
├── agents/                  # Agent implementations
│   ├── __init__.py
│   ├── analyst/             # Analyst agent
│   │   ├── __init__.py
│   │   ├── node.py          # LangGraph node
│   │   ├── crystallizer.py  # Requirement crystallization
│   │   └── prompts.py       # LLM prompts
│   ├── pm/                  # PM agent
│   ├── architect/           # Architect agent
│   ├── dev/                 # Dev agent
│   ├── tea/                 # TEA agent
│   └── sm/                  # SM agent
│
├── orchestrator/            # LangGraph workflow
│   ├── __init__.py
│   ├── graph.py             # Workflow definition
│   ├── state.py             # State schema
│   └── routing.py           # Conditional routing
│
├── memory/                  # Memory system
│   ├── __init__.py
│   ├── store.py             # Memory store protocol
│   ├── chromadb_store.py    # ChromaDB implementation
│   ├── pattern_learner.py   # Pattern detection
│   └── session.py           # Session persistence
│
├── gates/                   # Quality gates
│   ├── __init__.py
│   ├── decorator.py         # @quality_gate decorator
│   ├── evaluator.py         # Gate evaluation
│   └── gates/               # Individual gates
│       ├── testability.py
│       ├── architecture.py
│       └── dod.py
│
├── seed/                    # Seed input
│   ├── __init__.py
│   ├── parser.py            # Document parsing
│   ├── validator.py         # Validation
│   └── ambiguity.py         # Ambiguity detection
│
├── audit/                   # Audit trail
│   ├── __init__.py
│   ├── logger.py            # Decision logging
│   ├── export.py            # Export formats
│   └── traceability.py      # Requirement tracing
│
├── config/                  # Configuration
│   ├── __init__.py
│   ├── schema.py            # Pydantic models
│   └── loader.py            # Config loading
│
├── cli/                     # CLI interface
│   ├── __init__.py
│   ├── main.py              # Typer app
│   └── commands/            # Individual commands
│
├── sdk/                     # Python SDK
│   ├── __init__.py
│   ├── client.py            # YoloClient
│   └── events.py            # Event types
│
└── mcp/                     # MCP server
    ├── __init__.py
    ├── server.py            # FastMCP server
    └── tools.py             # MCP tools
```

---

## Next Steps

- [CLI Reference](/yolo-developer/cli/) - Command-line interface
- [Python SDK](/yolo-developer/sdk/) - Programmatic API
- [Configuration](/yolo-developer/configuration/) - All options
