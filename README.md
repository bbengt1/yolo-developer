# YOLO Developer

Autonomous multi-agent AI development system using the BMad Method. YOLO Developer orchestrates specialized AI agents through a LangGraph-based engine to autonomously handle software development tasks from requirements to implementation.

## Project Status

**Current Phase:** Core Infrastructure Complete (Epics 1-7)

| Epic | Status | Description |
|------|--------|-------------|
| 1. Project Initialization | Done | Configuration system with YAML + env vars |
| 2. Memory & Context Layer | Done | ChromaDB vector storage, pattern learning |
| 3. Quality Gate Framework | Done | Blocking gates, confidence scoring, metrics |
| 4. Seed Input & Validation | Done | NL parsing, ambiguity detection, SOP validation |
| 5. Analyst Agent | Done | Requirement crystallization, contradiction flagging |
| 6. PM Agent | Done | Story generation, prioritization, dependencies |
| 7. Architect Agent | Done | ADRs, 12-Factor, ATAM review, pattern matching |
| 8. Dev Agent | Backlog | Code generation, tests, documentation |
| 9. TEA Agent | Backlog | Test coverage, risk categorization |
| 10. Orchestration & SM | Backlog | LangGraph workflow, sprint management |
| 11. Audit Trail | Backlog | Decision logging, traceability |
| 12. CLI Interface | Backlog | Full Typer CLI commands |
| 13. Python SDK | Backlog | Programmatic API access |
| 14. MCP Integration | Backlog | Claude Code compatibility |

**Test Coverage:** 2,884 tests

## Architecture

```
src/yolo_developer/
├── agents/              # AI Agent modules
│   ├── analyst/         # Requirement analysis (Epic 5)
│   ├── pm/              # Story generation (Epic 6)
│   ├── architect/       # Design decisions (Epic 7)
│   └── prompts/         # Shared prompt templates
├── config/              # Pydantic configuration (Epic 1)
├── gates/               # Quality gate framework (Epic 3)
│   └── gates/           # Individual gate implementations
├── memory/              # Memory & context layer (Epic 2)
│   └── analyzers/       # Pattern analyzers
├── seed/                # Seed input validation (Epic 4)
├── orchestrator/        # LangGraph orchestration (Epic 10 - partial)
├── cli/                 # CLI interface (Epic 12 - partial)
├── sdk/                 # Python SDK (Epic 13 - planned)
├── mcp/                 # MCP server (Epic 14 - planned)
└── audit/               # Audit trail (Epic 11 - planned)
```

## Implemented Features

### Configuration System (Epic 1)
- Pydantic v2 schema with full type validation
- Three-layer priority: defaults -> YAML -> environment variables
- Secret management via `SecretStr` (API keys never exported)
- Configuration export/import for project portability

```python
from yolo_developer.config import load_config

config = load_config()  # Loads from ./yolo.yaml with env overrides
```

### Memory & Context Layer (Epic 2)
- **ChromaDB Vector Storage:** Semantic search for decisions and patterns
- **JSON Graph Storage:** Relationship tracking between artifacts
- **Pattern Learning:** Automatic detection of codebase naming/structure patterns
- **Session Persistence:** Context preservation across agent handoffs
- **Project Isolation:** Multi-project support with tenant separation

```python
from yolo_developer.memory import create_memory_store, PatternLearner

store = await create_memory_store(config)
learner = PatternLearner(store)
patterns = await learner.learn_patterns_from_codebase("/path/to/project")
```

### Quality Gate Framework (Epic 3)
- **Decorator-based gates:** `@quality_gate("testability", blocking=True)`
- **Built-in gates:** Testability, AC Measurability, Architecture Validation, Definition of Done
- **Confidence scoring:** 0.0-1.0 scores with configurable thresholds
- **Metrics tracking:** Historical gate performance over time
- **Failure reports:** Detailed remediation guidance

```python
from yolo_developer.gates import quality_gate, GateResult

@quality_gate("architecture_validation", blocking=True)
async def architect_node(state):
    # Gate automatically validates output
    return state_update
```

### Seed Input & Validation (Epic 4)
- Natural language document parsing (Markdown, plain text)
- Ambiguity detection with clarification question generation
- SOP constraint validation
- Semantic validation reports
- Quality threshold rejection

```python
from yolo_developer.seed import parse_seed_document, detect_ambiguities

parsed = await parse_seed_document(content)
ambiguities = await detect_ambiguities(parsed.requirements)
```

### Analyst Agent (Epic 5)
- Requirement crystallization from vague inputs
- Missing requirement identification
- Requirement categorization (functional, non-functional, constraint)
- Implementability validation
- Contradiction flagging between requirements
- Escalation to PM for clarification

### PM Agent (Epic 6)
- Transform requirements to user stories
- Acceptance criteria testability validation
- Story prioritization (MoSCoW, value-based)
- Dependency identification between stories
- Epic breakdown for large features
- Escalation to Analyst for requirement gaps

### Architect Agent (Epic 7)
- **12-Factor Analysis:** Compliance checking against 12-Factor App principles
- **ADR Generation:** Architecture Decision Records with context and consequences
- **Quality Attribute Evaluation:** Performance, security, reliability, scalability, maintainability
- **Technical Risk Identification:** Risk categorization with mitigation strategies
- **Tech Stack Validation:** Constraint checking against configured technology stack
- **ATAM Review:** Architecture Tradeoff Analysis Method evaluation
- **Pattern Matching:** Design validation against learned codebase patterns

```python
from yolo_developer.agents.architect import (
    architect_node,
    analyze_twelve_factor,
    generate_adr,
    run_pattern_matching,
)

result = await architect_node(state)
# Returns: design_decisions, ADRs, quality evaluations, risk reports, ATAM reviews
```

## Installation

```bash
# Clone the repository
git clone https://github.com/bbengt1/yolo-developer.git
cd yolo-developer

# Install dependencies with uv
uv sync
```

## Configuration

Create a `yolo.yaml` in your project root:

```yaml
project_name: my-project

llm:
  smart_model: gpt-4o
  routine_model: gpt-4o-mini
  # API keys via environment variables only

quality:
  test_coverage_threshold: 0.8
  gate_pass_threshold: 0.7

memory:
  vector_store: chromadb
  persist_directory: .yolo/memory
```

Environment variables use `YOLO_` prefix:
```bash
export YOLO_LLM__OPENAI_API_KEY=sk-...
export YOLO_LLM__ANTHROPIC_API_KEY=sk-ant-...
export YOLO_QUALITY__TEST_COVERAGE_THRESHOLD=0.9
```

## Usage

```bash
# Show available commands
yolo --help

# Initialize a new project
yolo init

# Seed with requirements document
yolo seed requirements.md
```

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/yolo_developer --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/agents/architect/test_pattern_matcher.py -v

# Type checking
uv run mypy src/yolo_developer

# Linting and formatting
uv run ruff check src tests
uv run ruff format src tests
```

## Roadmap

### Next Up: Epic 8 - Dev Agent
- Code generation following learned patterns
- Unit and integration test generation
- Documentation generation
- Definition of Done validation

### Future Epics
- **Epic 9:** TEA (Test Engineering Agent) for coverage validation
- **Epic 10:** Orchestration with SM Agent for sprint management
- **Epic 11:** Audit trail and observability
- **Epic 12:** Complete CLI interface
- **Epic 13:** Python SDK for programmatic access
- **Epic 14:** MCP integration for Claude Code

## Technology Stack

| Component | Technology |
|-----------|------------|
| Runtime | Python 3.10+ |
| Orchestration | LangGraph |
| Vector Store | ChromaDB |
| Configuration | Pydantic v2 + YAML |
| CLI | Typer + Rich |
| Testing | pytest + pytest-asyncio |
| Linting | ruff |
| Type Checking | mypy (strict) |

## License

MIT
