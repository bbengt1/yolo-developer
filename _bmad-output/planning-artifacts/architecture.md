---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
inputDocuments:
  - '_bmad-output/planning-artifacts/prd.md'
  - '_bmad-output/planning-artifacts/product-brief-yolo-developer-2026-01-04.md'
  - '_bmad-output/planning-artifacts/research/technical-multi-agent-orchestration-research-2026-01-03.md'
  - '_bmad-output/analysis/brainstorming-session-2026-01-03.md'
workflowType: 'architecture'
project_name: 'yolo-developer'
user_name: 'Brent'
date: '2026-01-04'
status: 'complete'
completedAt: '2026-01-04'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**

The PRD defines **117 functional requirements** organized into **14 capability areas**:

| Capability Area | FR Range | Architectural Implication |
|-----------------|----------|---------------------------|
| Seed Input & Validation | FR1-8 | Input processing layer, semantic validation |
| Agent Orchestration | FR9-17 | Core orchestration engine, SM control plane |
| Quality Gate Framework | FR18-27 | Gate evaluation system, blocking mechanism |
| Memory & Context | FR28-35 | Hybrid storage layer, context preservation |
| Analyst Agent | FR36-41 | Requirement crystallization module |
| PM Agent | FR42-48 | Story generation module |
| Architect Agent | FR49-56 | Design validation module |
| Dev Agent | FR57-64 | Code generation module |
| SM Agent | FR65-72 | Orchestration and health monitoring |
| TEA Agent | FR73-80 | Test and validation module |
| Audit Trail | FR81-88 | Logging and traceability system |
| Configuration | FR89-97 | Settings management layer |
| CLI Interface | FR98-105 | Command-line presentation layer |
| Python SDK | FR106-111 | Programmatic API layer |
| MCP Integration | FR112-117 | External integration protocol |

**Non-Functional Requirements:**

| Category | Key Constraints | Architectural Impact |
|----------|-----------------|----------------------|
| Performance | <5s handoff, <10s gate eval, <4hr sprint | Async processing, caching, optimization |
| Reliability | >95% completion, 3-retry, checkpoints | Transaction management, recovery patterns |
| Security | Project isolation, API key management | Tenant separation, secrets handling |
| Scalability | 5-10 stories MVP, 20-50 growth | Horizontal scaling design |
| Integration | MCP 1.0+, OpenTelemetry, multi-LLM | Abstraction layers, protocol compliance |
| Cost | 70% tiering, >50% cache, token optimization | Model routing, caching strategy |

### Scale & Complexity

**Complexity Assessment: HIGH**

| Indicator | Assessment | Rationale |
|-----------|------------|-----------|
| Real-time Features | Medium | Status updates, not live collaboration |
| Multi-tenancy | None (MVP) | Single-user, project isolation only |
| Regulatory Compliance | Low | No PII, no financial data |
| Integration Complexity | High | LLM providers, memory stores, MCP, observability |
| User Interaction | Medium | CLI-first, SDK, MCP - no rich UI |
| Data Complexity | High | Vector embeddings, graph relationships, audit logs |

**Project Classification:**
- **Primary Domain:** Developer Tool / AI Infrastructure
- **Technical Domain:** Backend-heavy with CLI interface
- **Complexity Level:** High
- **Estimated Architectural Components:** 15-20 major components

### Technical Constraints & Dependencies

**Hard Constraints:**
1. Python 3.10+ runtime (LangGraph ecosystem requirement)
2. LLM API dependency (OpenAI, Anthropic, or compatible)
3. MCP protocol compliance for external integration
4. Async-first design for agent orchestration

**Soft Constraints:**
1. ChromaDB preferred for vector storage (CrewAI ecosystem)
2. Neo4j preferred for graph storage (research validated)
3. LangSmith/Langfuse for observability (LangGraph integration)

**External Dependencies:**
- LangGraph/LangChain libraries
- LLM provider APIs
- Vector database (embedded or external)
- Graph database (optional for MVP)

### Cross-Cutting Concerns Identified

| Concern | Affected Components | Architectural Pattern |
|---------|--------------------|-----------------------|
| **Observability** | All agents, orchestrator, memory | OpenTelemetry integration, trace context |
| **State Management** | All agents, memory layer | Continuous memory, checkpoint/restore |
| **Quality Enforcement** | All agent boundaries | Gate evaluation, blocking mechanism |
| **Cost Control** | All LLM calls | Model router, token tracking, caching |
| **Error Recovery** | All agents, orchestrator | Retry with backoff, rollback coordination |
| **Configuration** | All components | Centralized config, schema validation |
| **Audit Trail** | All operations | Append-only logging, decision tracing |

## Starter Template & Technology Foundation

### Primary Technology Domain

**Python CLI Tool + Multi-Agent Framework** based on project requirements analysis.

This is not a web application - it's a developer tool with:
- CLI as primary interface
- Python SDK for programmatic access
- MCP server for external integration
- Multi-agent orchestration core

### Technology Stack Selection

| Layer | Selected Technology | Version | Rationale |
|-------|---------------------|---------|-----------|
| **Runtime** | Python | 3.10+ | LangGraph ecosystem requirement |
| **Orchestration** | LangGraph | 1.0.5 | Production-ready, graph-based state, research-validated |
| **CLI Framework** | Typer + Rich | Latest | Modern type-hint driven, beautiful output |
| **Vector Store** | ChromaDB | 1.2.x | Embedded option, LangChain ecosystem |
| **Graph Store** | Neo4j (optional) | 5.x | Relationship queries, defer to v1.1 if needed |
| **Observability** | LangSmith/Langfuse | Latest | Native LangGraph integration |
| **Configuration** | Pydantic + YAML | v2.x | Schema validation, type safety |
| **Testing** | pytest + pytest-asyncio | Latest | Async support for agent testing |

### Project Structure (LangGraph Best Practices)

```
yolo-developer/
├── src/
│   └── yolo_developer/
│       ├── __init__.py
│       ├── cli/                    # CLI Interface (Typer)
│       │   ├── __init__.py
│       │   ├── main.py             # Entry point
│       │   └── commands/           # CLI commands
│       ├── sdk/                    # Python SDK
│       │   ├── __init__.py
│       │   └── client.py
│       ├── mcp/                    # MCP Server
│       │   ├── __init__.py
│       │   └── server.py
│       ├── agents/                 # Individual Agent Modules
│       │   ├── __init__.py
│       │   ├── analyst.py
│       │   ├── pm.py
│       │   ├── architect.py
│       │   ├── dev.py
│       │   ├── sm.py               # Scrum Master (Control Plane)
│       │   └── tea.py              # Test Architect
│       ├── orchestrator/           # LangGraph Orchestration
│       │   ├── __init__.py
│       │   ├── graph.py            # Main graph definition
│       │   ├── state.py            # State schema (TypedDict)
│       │   └── nodes.py            # Node functions
│       ├── memory/                 # Memory Layer
│       │   ├── __init__.py
│       │   ├── vector.py           # ChromaDB integration
│       │   └── graph.py            # Neo4j integration (optional)
│       ├── gates/                  # Quality Gate Framework
│       │   ├── __init__.py
│       │   └── evaluators.py
│       ├── audit/                  # Audit Trail
│       │   ├── __init__.py
│       │   └── logger.py
│       └── config/                 # Configuration
│           ├── __init__.py
│           ├── schema.py           # Pydantic models
│           └── loader.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── pyproject.toml                  # Project metadata (PEP 621)
├── langgraph.json                  # LangGraph configuration
├── .env.example
└── README.md
```

### Initialization Commands

```bash
# Create project with uv (modern Python package manager)
mkdir yolo-developer && cd yolo-developer
uv init --lib

# Add core dependencies
uv add langgraph langchain-core langchain-anthropic langchain-openai
uv add chromadb
uv add typer rich
uv add pydantic pydantic-settings pyyaml
uv add python-dotenv

# Add development dependencies
uv add --dev pytest pytest-asyncio pytest-cov
uv add --dev ruff mypy
uv add --dev langsmith  # or langfuse

# Create project structure
mkdir -p src/yolo_developer/{cli,sdk,mcp,agents,orchestrator,memory,gates,audit,config}
mkdir -p tests/{unit,integration,e2e}
```

### Architectural Decisions Made by Foundation

**Language & Runtime:**
- Python 3.10+ with full type hints
- Async-first design for agent orchestration
- PEP 621 compliant pyproject.toml

**Package Management:**
- uv for fast, reliable dependency management
- Lock file for reproducible builds

**Code Quality:**
- Ruff for linting and formatting (replaces black, isort, flake8)
- mypy for static type checking
- pytest for testing with async support

**Observability:**
- LangSmith integration via environment variable
- Structured logging with Python logging + Rich
- OpenTelemetry-compatible tracing

**Note:** Project initialization using these commands should be the first implementation story.

### Sources

- [LangGraph Application Structure](https://docs.langchain.com/langgraph-platform/application-structure)
- [LangGraph Best Practices](https://www.swarnendu.de/blog/langgraph-best-practices/)
- [LangGraph 1.0 Announcement](https://blog.langchain.com/langchain-langgraph-1dot0/)
- [ChromaDB PyPI](https://pypi.org/project/chromadb/)
- [Typer Documentation](https://typer.tiangolo.com/)

## Core Architectural Decisions

### Decision Priority Analysis

**Critical Decisions (Block Implementation):**
1. State Management Pattern - TypedDict for graph state, Pydantic at boundaries
2. LLM Provider Abstraction - LiteLLM for multi-provider support
3. MCP Implementation - FastMCP 2.x for protocol compliance
4. Agent Communication - LangGraph message passing with typed state

**Important Decisions (Shape Architecture):**
5. Memory Persistence - ChromaDB embedded + optional Neo4j
6. Quality Gate Pattern - Decorator-based gate evaluation
7. Error Handling - Retry with exponential backoff, SM-coordinated rollback
8. Configuration - Pydantic Settings with YAML override

**Deferred Decisions (Post-MVP):**
- Self-regulation loops (Velocity Governor, Thermal Shutdown)
- Parallel agent execution
- Advanced rollback coordination
- SOP database evolution

### Data Architecture

#### ADR-001: State Management Pattern

**Decision:** Hybrid TypedDict (internal) + Pydantic (boundaries)

**Context:** LangGraph requires state schema. Options: pure TypedDict, pure Pydantic, or hybrid.

**Choice:** TypedDict for graph state, Pydantic for input/output validation

**Rationale:**
- TypedDict has zero runtime overhead, natural dict operations
- Pydantic validates at system boundaries (API inputs, user configs)
- LangGraph's partial update model works naturally with TypedDict
- Avoids Pydantic performance issues with recursive validation

**Implementation:**
```python
# Internal graph state (TypedDict)
class YoloState(TypedDict):
    seed: SeedInput
    requirements: list[Requirement]
    stories: list[Story]
    current_agent: str
    messages: Annotated[list[Message], add_messages]

# Boundary validation (Pydantic)
class SeedInput(BaseModel):
    content: str
    source: Literal["file", "text", "url"]
    model_config = ConfigDict(strict=True)
```

**Sources:** [Type Safety in LangGraph](https://shazaali.substack.com/p/type-safety-in-langgraph-when-to)

#### ADR-002: Memory Persistence Strategy

**Decision:** ChromaDB embedded (MVP) + optional Neo4j (v1.1)

**Context:** PRD requires vector + graph hybrid memory.

**Choice:**
- ChromaDB 1.2.x for vector storage (embedded mode)
- JSON-based graph for MVP relationships
- Neo4j as optional upgrade path for v1.1

**Rationale:**
- ChromaDB embeds cleanly, no external service required
- Rust-core rewrite (2025) provides 4x performance
- JSON graph sufficient for MVP relationship needs
- Neo4j adds complexity without clear MVP benefit

**Implementation:**
```python
# Memory abstraction layer
class MemoryStore(Protocol):
    async def store_embedding(self, key: str, content: str, metadata: dict) -> None: ...
    async def search_similar(self, query: str, k: int = 5) -> list[MemoryResult]: ...
    async def store_relationship(self, source: str, target: str, relation: str) -> None: ...

# ChromaDB implementation
class ChromaMemory(MemoryStore):
    def __init__(self, persist_directory: str):
        self.client = chromadb.PersistentClient(path=persist_directory)
```

### LLM Integration

#### ADR-003: LLM Provider Abstraction

**Decision:** LiteLLM for unified provider access

**Context:** PRD requires multi-LLM support with model tiering and cost tracking.

**Choice:** LiteLLM SDK for provider abstraction

**Rationale:**
- 100+ provider support in OpenAI-compatible format
- Built-in cost tracking and token counting
- Model fallback and load balancing
- 8ms P95 latency at 1k RPS (production proven)
- Native LangChain integration via ChatLiteLLM

**Implementation:**
```python
from litellm import completion

class LLMRouter:
    def __init__(self, config: LLMConfig):
        self.model_map = {
            "routine": config.cheap_model,      # e.g., "gpt-4o-mini"
            "complex": config.premium_model,    # e.g., "claude-sonnet-4-20250514"
            "critical": config.best_model,      # e.g., "claude-opus-4-5-20251101"
        }

    async def call(self, messages: list, tier: str = "routine") -> str:
        model = self.model_map[tier]
        response = await completion(model=model, messages=messages)
        return response.choices[0].message.content
```

**Sources:** [LiteLLM GitHub](https://github.com/BerriAI/litellm), [LiteLLM Docs](https://docs.litellm.ai/docs/)

### API & Communication Patterns

#### ADR-004: MCP Server Implementation

**Decision:** FastMCP 2.x for MCP protocol

**Context:** PRD requires MCP integration for external tool access.

**Choice:** FastMCP 2.x (production-ready, extends official SDK)

**Rationale:**
- FastMCP 1.0 incorporated into official SDK
- FastMCP 2.0 adds production features (auth, deployment, testing)
- Decorator-based tool definition matches project patterns
- Supports STDIO (local) and HTTP (production) transports

**Implementation:**
```python
from fastmcp import FastMCP

mcp = FastMCP("YOLO Developer", mask_error_details=True)

@mcp.tool
async def seed(content: str, source: str = "text") -> dict:
    """Provide seed requirements for autonomous development"""
    # Validate and process seed
    return {"status": "accepted", "seed_id": seed_id}

@mcp.tool
async def run(seed_id: str) -> dict:
    """Execute autonomous sprint on seeded requirements"""
    # Trigger orchestration
    return {"sprint_id": sprint_id, "status": "running"}

@mcp.tool
async def status(sprint_id: str) -> dict:
    """Get current sprint execution status"""
    return {"phase": "dev", "stories_complete": 3, "stories_total": 5}
```

**Sources:** [FastMCP GitHub](https://github.com/jlowin/fastmcp), [FastMCP Tutorial](https://www.firecrawl.dev/blog/fastmcp-tutorial-building-mcp-servers-python)

#### ADR-005: Inter-Agent Communication

**Decision:** LangGraph message passing with typed state transitions

**Context:** Agents need to communicate and hand off work.

**Choice:** LangGraph edges with state-based handoffs

**Rationale:**
- Graph edges define explicit handoff conditions
- State machine transitions are predictable and replayable
- Message accumulation via reducers for audit trail
- Matches SM as control plane pattern from brainstorming

**Implementation:**
```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(YoloState)

# Add agent nodes
workflow.add_node("analyst", analyst_node)
workflow.add_node("pm", pm_node)
workflow.add_node("architect", architect_node)
workflow.add_node("sm", sm_node)  # Control plane

# Define handoffs via edges
workflow.add_edge("analyst", "pm")
workflow.add_conditional_edges("pm", route_after_pm)
workflow.add_edge("architect", "sm")

# SM as orchestrator
def route_after_pm(state: YoloState) -> str:
    if state["needs_architecture"]:
        return "architect"
    return "sm"
```

### Quality Gate Framework

#### ADR-006: Quality Gate Pattern

**Decision:** Decorator-based gates with blocking capability

**Context:** Every agent boundary requires quality validation.

**Choice:** Python decorators wrapping node functions

**Rationale:**
- Non-invasive to agent logic
- Reusable across all agents
- Supports both blocking and advisory modes
- Integrates with audit trail

**Implementation:**
```python
from functools import wraps

def quality_gate(gate_name: str, blocking: bool = True):
    def decorator(func):
        @wraps(func)
        async def wrapper(state: YoloState) -> YoloState:
            # Run gate evaluation
            result = await evaluate_gate(gate_name, state)

            if not result.passed and blocking:
                state["gate_blocked"] = True
                state["gate_failure"] = result.reason
                return state  # Don't proceed

            # Log gate result to audit
            await log_gate_result(gate_name, result)

            return await func(state)
        return wrapper
    return decorator

@quality_gate("testability", blocking=True)
async def analyst_node(state: YoloState) -> YoloState:
    # Agent logic here
    ...
```

### Error Handling & Recovery

#### ADR-007: Error Handling Strategy

**Decision:** Retry with exponential backoff + SM-coordinated recovery

**Context:** LLM calls fail, agents can get stuck.

**Choice:**
- Tenacity for retry logic
- SM agent as recovery coordinator
- Checkpoint-based state recovery

**Rationale:**
- Matches PRD reliability requirements (3 retries)
- SM as control plane handles escalation
- LangGraph checkpointing enables resume from failure
- Aligns with "Keep Agents Working" brainstorming principle

**Implementation:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
async def call_llm_with_retry(prompt: str) -> str:
    return await llm_router.call(prompt)

# SM recovery coordination
async def sm_recovery_handler(state: YoloState, error: Exception) -> YoloState:
    if state["recovery_attempts"] >= 3:
        state["escalate_to_human"] = True
        return state

    # Attempt recovery based on error type
    recovery_action = determine_recovery(error)
    state["recovery_action"] = recovery_action
    state["recovery_attempts"] += 1
    return state
```

### Configuration

#### ADR-008: Configuration Management

**Decision:** Pydantic Settings with YAML override

**Context:** PRD requires configurable quality thresholds, LLM preferences, etc.

**Choice:** Pydantic Settings + YAML files + environment variables

**Rationale:**
- Schema validation catches config errors early
- YAML is human-readable for project config
- Environment variables for secrets
- Layered: defaults → YAML → env → CLI

**Implementation:**
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class YoloConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YOLO_",
        yaml_file="yolo.yaml",
    )

    # LLM settings
    cheap_model: str = "gpt-4o-mini"
    premium_model: str = "claude-sonnet-4-20250514"

    # Quality thresholds
    test_coverage_threshold: float = 0.80
    confidence_threshold: float = 0.90

    # Memory settings
    memory_persist_path: str = ".yolo/memory"
```

### Infrastructure & Deployment

#### ADR-009: Packaging & Distribution

**Decision:** PyPI package with CLI entry point

**Context:** PRD specifies CLI as primary interface.

**Choice:**
- PyPI distribution via `yolo-developer` package
- Typer CLI with `yolo` command entry point
- uv/pip installable

**Rationale:**
- Standard Python distribution model
- Single `pip install yolo-developer` for users
- Entry point creates `yolo` CLI command
- Works with pipx for isolated installation

**Implementation:**
```toml
# pyproject.toml
[project]
name = "yolo-developer"
version = "0.1.0"
requires-python = ">=3.10"

[project.scripts]
yolo = "yolo_developer.cli.main:app"
```

### Decision Impact Analysis

**Implementation Sequence:**
1. Project scaffolding with structure and dependencies
2. Configuration system (ADR-008)
3. LLM abstraction layer (ADR-003)
4. State schema and memory (ADR-001, ADR-002)
5. Quality gate framework (ADR-006)
6. Individual agent implementations
7. LangGraph orchestration (ADR-005)
8. MCP server (ADR-004)
9. CLI interface
10. SDK wrapper

**Cross-Component Dependencies:**

| Decision | Depends On | Affects |
|----------|------------|---------|
| State (ADR-001) | None | All agents, orchestrator |
| Memory (ADR-002) | State | All agents |
| LLM (ADR-003) | Config | All agents |
| MCP (ADR-004) | Config, LLM | External integrations |
| Communication (ADR-005) | State, Agents | Orchestrator, SM |
| Gates (ADR-006) | State | All agent boundaries |
| Errors (ADR-007) | LLM, State | All agents, SM |
| Config (ADR-008) | None | Everything |

## Implementation Patterns & Consistency Rules

### Pattern Categories Defined

**Critical Conflict Points Identified:** 8 areas where AI agents could make different choices

| Category | Risk Level | Conflict Example |
|----------|------------|------------------|
| State field naming | High | `current_agent` vs `currentAgent` in state dict |
| Agent output format | High | Different JSON structures from each agent |
| Error handling | Medium | Some agents raise, others return error state |
| Logging format | Medium | Inconsistent log message structure |
| Test organization | Medium | Tests in `/tests` vs co-located |
| Async patterns | High | Some agents blocking, others async |
| Type annotations | Medium | Optional types vs strict typing |
| Import ordering | Low | Random import organization |

### Naming Patterns

#### Python Module & Variable Naming

| Element | Convention | Example | Anti-Pattern |
|---------|------------|---------|--------------|
| Modules | snake_case | `agent_analyst.py` | `AgentAnalyst.py` |
| Classes | PascalCase | `AnalystAgent` | `analyst_agent` |
| Functions | snake_case | `validate_requirements` | `validateRequirements` |
| Constants | UPPER_SNAKE | `MAX_RETRIES = 3` | `maxRetries = 3` |
| Private | leading underscore | `_internal_state` | `internalState` |
| Type aliases | PascalCase | `AgentResult = dict[str, Any]` | `agent_result` |

#### State Field Naming

**Convention:** snake_case for all state dictionary keys

```python
# CORRECT
class YoloState(TypedDict):
    current_agent: str
    seed_input: SeedInput
    gate_results: list[GateResult]
    recovery_attempts: int

# WRONG
class YoloState(TypedDict):
    currentAgent: str          # camelCase
    SeedInput: SeedInput       # PascalCase
    gateResults: list          # camelCase
```

#### Agent Naming

| Agent | Module | Class | Node Function |
|-------|--------|-------|---------------|
| Analyst | `agents/analyst.py` | `AnalystAgent` | `analyst_node` |
| PM | `agents/pm.py` | `PMAgent` | `pm_node` |
| Architect | `agents/architect.py` | `ArchitectAgent` | `architect_node` |
| Dev | `agents/dev.py` | `DevAgent` | `dev_node` |
| SM | `agents/sm.py` | `SMAgent` | `sm_node` |
| TEA | `agents/tea.py` | `TEAAgent` | `tea_node` |

### Structure Patterns

#### Test Organization

**Convention:** Separate `/tests` directory mirroring source structure

```
tests/
├── unit/
│   ├── agents/
│   │   ├── test_analyst.py
│   │   ├── test_pm.py
│   │   └── ...
│   ├── gates/
│   │   └── test_evaluators.py
│   └── memory/
│       └── test_vector.py
├── integration/
│   ├── test_orchestrator.py
│   └── test_agent_handoffs.py
└── e2e/
    └── test_full_sprint.py
```

**Test Naming:**
- Test files: `test_<module>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<behavior>_<scenario>`

```python
# CORRECT
def test_analyst_validates_requirements_with_missing_fields():
    ...

# WRONG
def testAnalystValidation():  # camelCase, not descriptive
    ...
```

#### Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `pyproject.toml` | Root | Project metadata, dependencies |
| `langgraph.json` | Root | LangGraph configuration |
| `.env.example` | Root | Environment variable template |
| `yolo.yaml` | Root (optional) | User project configuration |
| `src/.../config/schema.py` | Package | Pydantic config models |

### Format Patterns

#### Agent Output Format

**Convention:** All agents return state updates as dicts with consistent structure

```python
# CORRECT - Agent node returns state update dict
async def analyst_node(state: YoloState) -> dict:
    # Process requirements
    requirements = await analyze(state["seed_input"])

    return {
        "requirements": requirements,
        "current_agent": "analyst",
        "messages": [AIMessage(content="Requirements analyzed")]
    }

# WRONG - Returning full state object
async def analyst_node(state: YoloState) -> YoloState:
    state["requirements"] = requirements  # Mutating state
    return state
```

#### Error Response Format

**Convention:** Structured error in state, not exceptions for flow control

```python
# Error state structure
class GateError(TypedDict):
    gate_name: str
    reason: str
    severity: Literal["blocking", "warning"]
    agent: str
    timestamp: str

# In state
class YoloState(TypedDict):
    ...
    gate_errors: list[GateError]
    escalate_to_human: bool
```

#### Log Message Format

**Convention:** Structured logging with consistent fields

```python
import structlog

logger = structlog.get_logger()

# CORRECT - Structured with context
logger.info(
    "agent_completed",
    agent="analyst",
    requirements_count=5,
    duration_ms=1234
)

# WRONG - Unstructured string
logger.info(f"Analyst done with 5 requirements in 1234ms")
```

### Communication Patterns

#### Message Accumulation

**Convention:** Use LangGraph's `add_messages` reducer for conversation history

```python
from langgraph.graph import add_messages
from typing import Annotated

class YoloState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

#### Agent-to-Agent Communication

**Convention:** Via state updates, never direct calls

```python
# CORRECT - Communicate through state
async def analyst_node(state: YoloState) -> dict:
    return {
        "requirements": analyzed_requirements,
        "handoff_notes": "3 requirements need clarification",
        "next_agent_context": {"priority_items": [1, 2, 3]}
    }

# WRONG - Direct agent invocation
async def analyst_node(state: YoloState) -> dict:
    pm_result = await pm_agent.process(state)  # No! Let graph handle handoff
    ...
```

### Process Patterns

#### Async Convention

**Convention:** All agent nodes and I/O operations are async

```python
# CORRECT - Async everywhere
async def analyst_node(state: YoloState) -> dict:
    requirements = await self.llm.agenerate(prompt)
    await self.memory.store(requirements)
    return {"requirements": requirements}

# WRONG - Blocking calls
def analyst_node(state: YoloState) -> dict:
    requirements = self.llm.generate(prompt)  # Blocking!
    ...
```

#### Retry Pattern

**Convention:** Use tenacity with consistent configuration

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Standard retry decorator for LLM calls
LLM_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIError))
)

@LLM_RETRY
async def call_llm(prompt: str) -> str:
    ...
```

#### Quality Gate Pattern

**Convention:** Decorator-based gates that return state updates

```python
# Gate decorator usage
@quality_gate("testability", blocking=True)
async def analyst_node(state: YoloState) -> dict:
    ...

# Gate always evaluated before node execution
# If blocking gate fails, state["gate_blocked"] = True
# Node can check and handle gracefully
```

### Type Annotation Patterns

**Convention:** Full type hints, use `from __future__ import annotations`

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yolo_developer.agents.base import BaseAgent

# Function signatures always typed
async def process_seed(
    seed: SeedInput,
    config: YoloConfig,
) -> tuple[list[Requirement], list[GateError]]:
    ...

# Use TypedDict for structured dicts
class Requirement(TypedDict):
    id: str
    content: str
    testable: bool
```

### Import Organization

**Convention:** isort-compatible, grouped by type

```python
# Standard library
from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING

# Third-party
from langgraph.graph import StateGraph
from pydantic import BaseModel
import structlog

# Local - absolute imports
from yolo_developer.config import YoloConfig
from yolo_developer.agents.base import BaseAgent

if TYPE_CHECKING:
    from yolo_developer.memory import MemoryStore
```

### Enforcement Guidelines

**All AI Agents MUST:**

1. Use snake_case for all state dictionary keys
2. Return dict updates from node functions, never mutate state
3. Use async/await for all I/O operations
4. Apply standard retry decorator for LLM calls
5. Log with structlog using structured format
6. Type annotate all function signatures
7. Follow test naming: `test_<behavior>_<scenario>`

**Pattern Enforcement:**

| Mechanism | Tool | When |
|-----------|------|------|
| Linting | Ruff | Pre-commit, CI |
| Type checking | mypy | Pre-commit, CI |
| Import sorting | Ruff (isort) | Pre-commit |
| Test naming | pytest-naming | CI |
| State schema | Pydantic validation | Runtime |

### Pattern Examples

**Good Example - Complete Agent Node:**

```python
from __future__ import annotations

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger()

@quality_gate("testability", blocking=True)
async def analyst_node(state: YoloState) -> dict:
    """Analyze seed input and produce structured requirements."""
    logger.info("agent_started", agent="analyst", seed_id=state["seed_input"]["id"])

    requirements = await _analyze_requirements(state["seed_input"])

    logger.info(
        "agent_completed",
        agent="analyst",
        requirements_count=len(requirements),
    )

    return {
        "requirements": requirements,
        "current_agent": "analyst",
    }
```

**Anti-Patterns to Avoid:**

```python
# DON'T: camelCase state keys
return {"currentAgent": "analyst"}

# DON'T: Mutate state directly
state["requirements"] = requirements
return state

# DON'T: Blocking calls
requirements = llm.generate(prompt)  # Should be await llm.agenerate()

# DON'T: Unstructured logging
print(f"Done with {len(requirements)} requirements")

# DON'T: Missing type hints
def analyst_node(state):  # Should be (state: YoloState) -> dict
```

## Project Structure & Boundaries

### Complete Project Directory Structure

```
yolo-developer/
├── README.md
├── LICENSE
├── pyproject.toml                      # Project metadata, dependencies (PEP 621)
├── uv.lock                             # Lock file for reproducible builds
├── langgraph.json                      # LangGraph platform configuration
├── .env.example                        # Environment variable template
├── .gitignore
├── .github/
│   └── workflows/
│       ├── ci.yml                      # Lint, type-check, test
│       └── release.yml                 # PyPI publishing
├── .pre-commit-config.yaml             # Pre-commit hooks config
├── ruff.toml                           # Ruff linter/formatter config
│
├── src/
│   └── yolo_developer/
│       ├── __init__.py                 # Package version, public API
│       ├── py.typed                    # PEP 561 marker
│       │
│       ├── cli/                        # CLI Interface (FR98-105)
│       │   ├── __init__.py
│       │   ├── main.py                 # Typer app, entry point
│       │   ├── commands/
│       │   │   ├── __init__.py
│       │   │   ├── init.py             # yolo init
│       │   │   ├── seed.py             # yolo seed
│       │   │   ├── run.py              # yolo run
│       │   │   ├── status.py           # yolo status
│       │   │   ├── logs.py             # yolo logs
│       │   │   ├── tune.py             # yolo tune
│       │   │   └── config.py           # yolo config
│       │   └── display.py              # Rich output formatting
│       │
│       ├── sdk/                        # Python SDK (FR106-111)
│       │   ├── __init__.py             # Public SDK API
│       │   ├── client.py               # YoloClient class
│       │   ├── types.py                # SDK-specific types
│       │   └── exceptions.py           # SDK exceptions
│       │
│       ├── mcp/                        # MCP Server (FR112-117)
│       │   ├── __init__.py
│       │   ├── server.py               # FastMCP server definition
│       │   └── tools.py                # MCP tool implementations
│       │
│       ├── orchestrator/               # LangGraph Orchestration (FR9-17)
│       │   ├── __init__.py
│       │   ├── graph.py                # Main StateGraph definition
│       │   ├── state.py                # YoloState TypedDict
│       │   ├── nodes.py                # Node function wrappers
│       │   ├── edges.py                # Conditional edge logic
│       │   └── checkpoints.py          # Checkpoint/recovery
│       │
│       ├── agents/                     # Agent Implementations
│       │   ├── __init__.py
│       │   ├── base.py                 # BaseAgent protocol
│       │   ├── analyst.py              # AnalystAgent (FR36-41)
│       │   ├── pm.py                   # PMAgent (FR42-48)
│       │   ├── architect.py            # ArchitectAgent (FR49-56)
│       │   ├── dev.py                  # DevAgent (FR57-64)
│       │   ├── sm.py                   # SMAgent (FR65-72)
│       │   ├── tea.py                  # TEAAgent (FR73-80)
│       │   └── prompts/                # Agent prompt templates
│       │       ├── __init__.py
│       │       ├── analyst.py
│       │       ├── pm.py
│       │       ├── architect.py
│       │       ├── dev.py
│       │       ├── sm.py
│       │       └── tea.py
│       │
│       ├── gates/                      # Quality Gate Framework (FR18-27)
│       │   ├── __init__.py
│       │   ├── decorator.py            # @quality_gate decorator
│       │   ├── evaluators.py           # Gate evaluation logic
│       │   ├── confidence.py           # Confidence scoring
│       │   └── gates/
│       │       ├── __init__.py
│       │       ├── testability.py      # Testability gate
│       │       ├── implementability.py # Implementability gate
│       │       └── dod.py              # Definition of Done gate
│       │
│       ├── memory/                     # Memory Layer (FR28-35)
│       │   ├── __init__.py
│       │   ├── protocol.py             # MemoryStore Protocol
│       │   ├── vector.py               # ChromaDB implementation
│       │   ├── graph.py                # JSON graph (Neo4j optional)
│       │   └── context.py              # Context management
│       │
│       ├── seed/                       # Seed Processing (FR1-8)
│       │   ├── __init__.py
│       │   ├── parser.py               # Seed document parsing
│       │   ├── validator.py            # Semantic validation
│       │   └── types.py                # SeedInput, etc.
│       │
│       ├── llm/                        # LLM Abstraction
│       │   ├── __init__.py
│       │   ├── router.py               # LLMRouter (model tiering)
│       │   ├── providers.py            # LiteLLM provider config
│       │   └── cache.py                # Semantic caching
│       │
│       ├── audit/                      # Audit Trail (FR81-88)
│       │   ├── __init__.py
│       │   ├── logger.py               # Structured audit logging
│       │   ├── trace.py                # Decision tracing
│       │   └── export.py               # Audit export formats
│       │
│       ├── config/                     # Configuration (FR89-97)
│       │   ├── __init__.py
│       │   ├── schema.py               # Pydantic Settings models
│       │   ├── loader.py               # Config loading logic
│       │   └── defaults.py             # Default configurations
│       │
│       └── utils/                      # Shared Utilities
│           ├── __init__.py
│           ├── retry.py                # Tenacity retry decorators
│           ├── logging.py              # Structlog configuration
│           └── async_utils.py          # Async helpers
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Pytest fixtures
│   ├── fixtures/                       # Test data and mocks
│   │   ├── __init__.py
│   │   ├── seeds/                      # Sample seed documents
│   │   ├── states/                     # Sample state snapshots
│   │   └── mocks.py                    # LLM mocks
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── test_analyst.py
│   │   │   ├── test_pm.py
│   │   │   ├── test_architect.py
│   │   │   ├── test_dev.py
│   │   │   ├── test_sm.py
│   │   │   └── test_tea.py
│   │   ├── gates/
│   │   │   └── test_evaluators.py
│   │   ├── memory/
│   │   │   └── test_vector.py
│   │   ├── seed/
│   │   │   └── test_validator.py
│   │   └── config/
│   │       └── test_schema.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_orchestrator.py
│   │   ├── test_agent_handoffs.py
│   │   └── test_memory_persistence.py
│   └── e2e/
│       ├── __init__.py
│       └── test_full_sprint.py
│
├── docs/                               # Documentation
│   ├── architecture.md                 # This document (copied from _bmad-output)
│   ├── configuration.md                # Config reference
│   ├── cli-reference.md                # CLI command reference
│   └── sdk-reference.md                # SDK API reference
│
└── examples/                           # Usage examples
    ├── basic_seed.txt                  # Simple seed example
    ├── complex_seed.md                 # Detailed seed example
    └── config_examples/
        ├── yolo.yaml                   # Example project config
        └── .env.example                # Environment setup
```

### Architectural Boundaries

#### API Boundaries

| Layer | Entry Point | Consumers | Protocol |
|-------|-------------|-----------|----------|
| **CLI** | `cli/main.py` | End users | Typer commands |
| **SDK** | `sdk/client.py` | Python programs | Python API |
| **MCP** | `mcp/server.py` | Claude Code, external tools | MCP 1.0+ |
| **Internal** | `orchestrator/graph.py` | CLI, SDK, MCP | Direct Python calls |

#### Component Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │   CLI   │    │   SDK   │    │   MCP   │                 │
│  └────┬────┘    └────┬────┘    └────┬────┘                 │
│       │              │              │                       │
│       └──────────────┼──────────────┘                       │
│                      │                                      │
├──────────────────────┼──────────────────────────────────────┤
│                      ▼                                      │
│              ┌───────────────┐                              │
│              │  Orchestrator │                              │
│              │  (LangGraph)  │                              │
│              └───────┬───────┘                              │
│                      │                                      │
├──────────────────────┼──────────────────────────────────────┤
│                      ▼                                      │
│    ┌─────────────────────────────────────────────────┐     │
│    │              Agent Layer                         │     │
│    │  ┌─────────┐ ┌────────┐ ┌──────────┐           │     │
│    │  │ Analyst │ │   PM   │ │ Architect│           │     │
│    │  └─────────┘ └────────┘ └──────────┘           │     │
│    │  ┌─────────┐ ┌────────┐ ┌──────────┐           │     │
│    │  │   Dev   │ │   SM   │ │   TEA    │           │     │
│    │  └─────────┘ └────────┘ └──────────┘           │     │
│    └─────────────────────────────────────────────────┘     │
│                      │                                      │
├──────────────────────┼──────────────────────────────────────┤
│                      ▼                                      │
│    ┌─────────────────────────────────────────────────┐     │
│    │           Cross-Cutting Services                 │     │
│    │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │     │
│    │  │ Memory │ │ Gates  │ │  LLM   │ │ Audit  │   │     │
│    │  └────────┘ └────────┘ └────────┘ └────────┘   │     │
│    └─────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Data Flow

```
Seed Input → Seed Parser → Validator → Analyst Node
                                            ↓
                                   Requirements
                                            ↓
                              PM Node → Stories
                                            ↓
                         Architect Node → Design
                                            ↓
                              Dev Node → Code
                                            ↓
                              TEA Node → Validation
                                            ↓
                              SM Node → Sprint Complete
                                            ↓
                                       Output
```

### Requirements to Structure Mapping

#### FR Categories to Modules

| FR Category | Primary Module | Supporting Modules |
|-------------|----------------|-------------------|
| Seed Input (FR1-8) | `seed/` | `gates/`, `memory/` |
| Orchestration (FR9-17) | `orchestrator/` | `agents/`, `gates/` |
| Quality Gates (FR18-27) | `gates/` | `audit/` |
| Memory (FR28-35) | `memory/` | `config/` |
| Analyst (FR36-41) | `agents/analyst.py` | `gates/`, `llm/` |
| PM (FR42-48) | `agents/pm.py` | `gates/`, `llm/` |
| Architect (FR49-56) | `agents/architect.py` | `gates/`, `llm/` |
| Dev (FR57-64) | `agents/dev.py` | `gates/`, `llm/` |
| SM (FR65-72) | `agents/sm.py` | `orchestrator/`, `audit/` |
| TEA (FR73-80) | `agents/tea.py` | `gates/`, `llm/` |
| Audit (FR81-88) | `audit/` | All modules |
| Config (FR89-97) | `config/` | All modules |
| CLI (FR98-105) | `cli/` | `sdk/` |
| SDK (FR106-111) | `sdk/` | `orchestrator/` |
| MCP (FR112-117) | `mcp/` | `orchestrator/` |

#### Cross-Cutting Concerns

| Concern | Location | Used By |
|---------|----------|---------|
| Logging | `utils/logging.py` | All modules |
| Retry | `utils/retry.py` | `llm/`, `agents/` |
| State | `orchestrator/state.py` | `orchestrator/`, `agents/` |
| Config | `config/schema.py` | All modules |
| Types | Module-specific `types.py` | Respective modules |

### Integration Points

#### Internal Communication

| From | To | Method |
|------|-----|--------|
| CLI → Orchestrator | Direct import | `from yolo_developer.orchestrator import run_sprint` |
| SDK → Orchestrator | Direct import | `from yolo_developer.orchestrator import run_sprint` |
| MCP → Orchestrator | Direct import | Same interface as SDK |
| Orchestrator → Agents | LangGraph nodes | State passed via graph |
| Agents → Memory | Dependency injection | Memory store injected |
| Agents → LLM | Dependency injection | LLM router injected |

#### External Integrations

| Service | Module | Configuration |
|---------|--------|---------------|
| OpenAI API | `llm/providers.py` | `OPENAI_API_KEY` env |
| Anthropic API | `llm/providers.py` | `ANTHROPIC_API_KEY` env |
| ChromaDB | `memory/vector.py` | Local path or URL |
| LangSmith | `utils/logging.py` | `LANGCHAIN_API_KEY` env |
| Neo4j (optional) | `memory/graph.py` | `NEO4J_URI`, `NEO4J_PASSWORD` |

### File Organization Patterns

#### Configuration Files Location

| File | Purpose | Owner |
|------|---------|-------|
| `pyproject.toml` | Package metadata | Developer |
| `langgraph.json` | LangGraph platform | LangGraph |
| `ruff.toml` | Linting rules | Developer |
| `.env.example` | Env template | Developer |
| `yolo.yaml` (user) | Project config | End user |

#### Source Organization Principles

1. **One responsibility per module** - Each `.py` file has a single purpose
2. **Explicit exports** - `__init__.py` defines public API
3. **No circular imports** - Dependencies flow downward
4. **Types near usage** - Type definitions in same module

#### Test Organization Principles

1. **Mirror source structure** - `tests/unit/agents/test_analyst.py` tests `agents/analyst.py`
2. **Shared fixtures in conftest.py** - Common mocks and setup
3. **Integration tests for boundaries** - Test component interactions
4. **E2E tests for full workflows** - Test complete sprints

### Development Workflow Integration

#### Development Server

```bash
# Run with hot reload
uv run python -m yolo_developer.cli.main

# Or via entry point
uv run yolo
```

#### Build Process

```bash
# Build package
uv build

# Creates:
# dist/yolo_developer-0.1.0.tar.gz
# dist/yolo_developer-0.1.0-py3-none-any.whl
```

#### Deployment Structure

```bash
# Install from PyPI
pip install yolo-developer

# Or with pipx for isolation
pipx install yolo-developer

# Creates 'yolo' command in PATH
```

## Architecture Validation Results

### Coherence Validation ✅

**Decision Compatibility:**

| Decision Pair | Status | Notes |
|---------------|--------|-------|
| LangGraph + Python 3.10+ | ✅ Compatible | LangGraph 1.0 requires Python 3.10+ |
| LangGraph + LiteLLM | ✅ Compatible | LiteLLM integrates via ChatLiteLLM |
| TypedDict + Pydantic | ✅ Compatible | Hybrid pattern validated by research |
| ChromaDB + LangGraph | ✅ Compatible | Both use async patterns |
| FastMCP + Typer | ✅ Compatible | Independent entry points |
| Ruff + mypy | ✅ Compatible | Complementary tooling |

**Pattern Consistency:**

| Pattern | Alignment |
|---------|-----------|
| snake_case state keys | ✅ Aligns with Python conventions and LangGraph examples |
| Async-first | ✅ Required by LangGraph, supported by all dependencies |
| Decorator-based gates | ✅ Natural Python pattern, non-invasive |
| Structured logging | ✅ Compatible with LangSmith observability |

**Structure Alignment:**

| Structure Element | Decision Supported |
|-------------------|-------------------|
| `orchestrator/state.py` | ADR-001 (TypedDict state) |
| `llm/router.py` | ADR-003 (LiteLLM abstraction) |
| `mcp/server.py` | ADR-004 (FastMCP) |
| `gates/decorator.py` | ADR-006 (Quality gates) |
| `config/schema.py` | ADR-008 (Pydantic Settings) |

### Requirements Coverage Validation ✅

**Functional Requirements Coverage:**

| FR Category | FRs | Architectural Support | Status |
|-------------|-----|----------------------|--------|
| Seed Input (FR1-8) | 8 | `seed/` module, validation gates | ✅ |
| Orchestration (FR9-17) | 9 | `orchestrator/`, SM control plane | ✅ |
| Quality Gates (FR18-27) | 10 | `gates/` framework | ✅ |
| Memory (FR28-35) | 8 | `memory/` with ChromaDB | ✅ |
| Analyst Agent (FR36-41) | 6 | `agents/analyst.py` | ✅ |
| PM Agent (FR42-48) | 7 | `agents/pm.py` | ✅ |
| Architect Agent (FR49-56) | 8 | `agents/architect.py` | ✅ |
| Dev Agent (FR57-64) | 8 | `agents/dev.py` | ✅ |
| SM Agent (FR65-72) | 8 | `agents/sm.py` | ✅ |
| TEA Agent (FR73-80) | 8 | `agents/tea.py` | ✅ |
| Audit Trail (FR81-88) | 8 | `audit/` module | ✅ |
| Configuration (FR89-97) | 9 | `config/` module | ✅ |
| CLI Interface (FR98-105) | 8 | `cli/` module | ✅ |
| Python SDK (FR106-111) | 6 | `sdk/` module | ✅ |
| MCP Integration (FR112-117) | 6 | `mcp/` module | ✅ |

**Total: 117 FRs covered by architecture**

**Non-Functional Requirements Coverage:**

| NFR Category | Key Requirements | Architectural Support |
|--------------|------------------|----------------------|
| Performance | <5s handoff, <10s gate | ✅ Async design, caching layer |
| Reliability | >95% completion, 3-retry | ✅ Tenacity retry, checkpointing |
| Security | API keys, isolation | ✅ Env vars, project scoping |
| Scalability | 5-10 stories MVP | ✅ LangGraph handles scale |
| Integration | MCP 1.0+, OpenTelemetry | ✅ FastMCP, LangSmith |
| Cost | 70% model tiering | ✅ LLMRouter with tiers |

### Implementation Readiness Validation ✅

**Decision Completeness:**

| Criterion | Status |
|-----------|--------|
| All ADRs have version numbers | ✅ 9 ADRs with specific versions |
| Implementation code examples | ✅ Code snippets for all patterns |
| Technology rationale documented | ✅ Each decision includes why |
| Sources cited | ✅ Research references included |

**Structure Completeness:**

| Criterion | Status |
|-----------|--------|
| All directories defined | ✅ Complete tree with 70+ files |
| File purposes documented | ✅ Comments in tree structure |
| Entry points specified | ✅ CLI, SDK, MCP entry points |
| Test structure defined | ✅ unit/, integration/, e2e/ |

**Pattern Completeness:**

| Criterion | Status |
|-----------|--------|
| Naming conventions | ✅ 6 element types defined |
| State field naming | ✅ snake_case with examples |
| Agent naming | ✅ Table with all 6 agents |
| Test naming | ✅ test_<behavior>_<scenario> |
| Error handling | ✅ Structured errors in state |
| Async patterns | ✅ All I/O is async |

### Gap Analysis Results

**Critical Gaps:** None identified

**Important Gaps (addressable during implementation):**

| Gap | Priority | Mitigation |
|-----|----------|------------|
| Agent prompts not defined | Medium | Create during agent implementation |
| Specific gate evaluators | Medium | Implement with agent stories |
| Observability dashboard | Low | Add after core functionality |

**Deferred Decisions (documented for v1.1+):**

| Decision | Rationale |
|----------|-----------|
| Neo4j integration | JSON graph sufficient for MVP |
| Self-regulation loops | Velocity Governor, Thermal Shutdown for v1.1 |
| Parallel agent execution | Sequential sufficient for MVP |
| SOP database evolution | Learning system for v1.2+ |

### Validation Issues Addressed

No blocking issues found. All architectural decisions are coherent and support the PRD requirements.

### Architecture Completeness Checklist

**✅ Requirements Analysis**

- [x] Project context thoroughly analyzed (117 FRs, 14 capability areas)
- [x] Scale and complexity assessed (High complexity)
- [x] Technical constraints identified (Python 3.10+, LLM API dependency)
- [x] Cross-cutting concerns mapped (7 concerns identified)

**✅ Architectural Decisions**

- [x] Critical decisions documented with versions (9 ADRs)
- [x] Technology stack fully specified (LangGraph 1.0.5, ChromaDB 1.2.x, etc.)
- [x] Integration patterns defined (MCP, LLM providers, memory stores)
- [x] Performance considerations addressed (async, caching, tiering)

**✅ Implementation Patterns**

- [x] Naming conventions established (6 element types)
- [x] Structure patterns defined (test org, config files)
- [x] Communication patterns specified (state updates, message accumulation)
- [x] Process patterns documented (async, retry, gates)

**✅ Project Structure**

- [x] Complete directory structure defined (70+ files)
- [x] Component boundaries established (4 layers)
- [x] Integration points mapped (internal + external)
- [x] Requirements to structure mapping complete (15 FR categories → modules)

### Architecture Readiness Assessment

**Overall Status:** ✅ READY FOR IMPLEMENTATION

**Confidence Level:** HIGH

**Key Strengths:**

1. **Research-validated stack** - LangGraph selected based on technical research
2. **Production-ready versions** - LangGraph 1.0, FastMCP 2.x, ChromaDB 1.2.x
3. **Clear separation of concerns** - 4-layer architecture with defined boundaries
4. **Comprehensive patterns** - 8 conflict points identified and addressed
5. **Full FR coverage** - All 117 functional requirements mapped to modules
6. **Explicit decision framework** - 9 ADRs with rationale and code examples

**Areas for Future Enhancement:**

1. Parallel agent execution (v1.1)
2. Self-regulation feedback loops (v1.1)
3. Neo4j graph integration (v1.1)
4. SOP database evolution (v1.2)
5. Team/multi-tenant features (v2.0)

### Implementation Handoff

**AI Agent Guidelines:**

1. Follow all architectural decisions exactly as documented in ADRs
2. Use implementation patterns consistently across all components
3. Respect project structure and boundaries
4. Use snake_case for all state dictionary keys
5. Return dict updates from node functions, never mutate state
6. Use async/await for all I/O operations
7. Apply standard retry decorator for LLM calls
8. Log with structlog using structured format
9. Type annotate all function signatures
10. Follow test naming: `test_<behavior>_<scenario>`

**First Implementation Priority:**

```bash
# Project initialization
mkdir yolo-developer && cd yolo-developer
uv init --lib

# Add core dependencies
uv add langgraph langchain-core langchain-anthropic langchain-openai
uv add chromadb typer rich pydantic pydantic-settings pyyaml python-dotenv tenacity structlog

# Add development dependencies
uv add --dev pytest pytest-asyncio pytest-cov ruff mypy langsmith

# Create project structure
mkdir -p src/yolo_developer/{cli/commands,sdk,mcp,agents/prompts,orchestrator,memory,gates/gates,seed,llm,audit,config,utils}
mkdir -p tests/{unit/{agents,gates,memory,seed,config},integration,e2e,fixtures/seeds,fixtures/states}

# Initialize Python package
touch src/yolo_developer/__init__.py src/yolo_developer/py.typed
```

## Architecture Completion Summary

### Workflow Completion

**Architecture Decision Workflow:** COMPLETED ✅
**Total Steps Completed:** 8
**Date Completed:** 2026-01-04
**Document Location:** `_bmad-output/planning-artifacts/architecture.md`

### Final Architecture Deliverables

**📋 Complete Architecture Document**

- All architectural decisions documented with specific versions
- Implementation patterns ensuring AI agent consistency
- Complete project structure with all files and directories
- Requirements to architecture mapping
- Validation confirming coherence and completeness

**🏗️ Implementation Ready Foundation**

- 9 architectural decisions made (ADR-001 through ADR-009)
- 8 implementation pattern categories defined
- 15 architectural components specified
- 117 functional requirements fully supported

**📚 AI Agent Implementation Guide**

- Technology stack with verified versions (LangGraph 1.0.5, ChromaDB 1.2.x, FastMCP 2.x)
- Consistency rules that prevent implementation conflicts
- Project structure with clear boundaries (4-layer architecture)
- Integration patterns and communication standards

### Implementation Handoff

**For AI Agents:**
This architecture document is your complete guide for implementing yolo-developer. Follow all decisions, patterns, and structures exactly as documented.

**First Implementation Priority:**
```bash
mkdir yolo-developer && cd yolo-developer
uv init --lib
uv add langgraph langchain-core langchain-anthropic langchain-openai chromadb typer rich pydantic pydantic-settings pyyaml python-dotenv tenacity structlog
```

**Development Sequence:**

1. Initialize project using documented starter template
2. Set up development environment per architecture
3. Implement core architectural foundations (Config, LLM Router, State)
4. Build agent implementations following established patterns
5. Maintain consistency with documented rules

### Quality Assurance Checklist

**✅ Architecture Coherence**

- [x] All decisions work together without conflicts
- [x] Technology choices are compatible
- [x] Patterns support the architectural decisions
- [x] Structure aligns with all choices

**✅ Requirements Coverage**

- [x] All 117 functional requirements are supported
- [x] All non-functional requirements are addressed
- [x] Cross-cutting concerns are handled (7 identified)
- [x] Integration points are defined

**✅ Implementation Readiness**

- [x] Decisions are specific and actionable
- [x] Patterns prevent agent conflicts (8 conflict points addressed)
- [x] Structure is complete and unambiguous (70+ files defined)
- [x] Examples are provided for clarity

### Project Success Factors

**🎯 Clear Decision Framework**
Every technology choice was made with clear rationale, ensuring all stakeholders understand the architectural direction.

**🔧 Consistency Guarantee**
Implementation patterns and rules ensure that multiple AI agents will produce compatible, consistent code that works together seamlessly.

**📋 Complete Coverage**
All project requirements are architecturally supported, with clear mapping from business needs to technical implementation.

**🏗️ Solid Foundation**
The chosen technology stack and architectural patterns provide a production-ready foundation following current best practices.

---

**Architecture Status:** READY FOR IMPLEMENTATION ✅

**Next Phase:** Begin implementation using the architectural decisions and patterns documented herein.

**Document Maintenance:** Update this architecture when major technical decisions are made during implementation.

