# YOLO Developer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-6500+-brightgreen.svg)](#testing)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Autonomous multi-agent AI development system** that orchestrates specialized AI agents through LangGraph to handle software development tasks from requirements to implementation.

Built on the [BMad Method](https://github.com/bmadcode/BMAD-METHOD) for AI-assisted development.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
- [Python SDK](#python-sdk)
- [MCP Integration](#mcp-integration)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Development](#development)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

### Multi-Agent Orchestration
- **6 Specialized Agents**: Analyst, PM, Architect, Dev, TEA (Test Engineering), and SM (Scrum Master)
- **LangGraph Workflow**: State-machine orchestration with conditional routing and agent handoffs
- **Conflict Mediation**: Automatic resolution of inter-agent disagreements
- **Human Escalation**: Configurable breakpoints for human review

### Intelligent Analysis
- **Requirement Crystallization**: Transform vague inputs into actionable requirements
- **Ambiguity Detection**: Identify unclear specifications with clarification questions
- **Contradiction Flagging**: Detect conflicting requirements automatically
- **12-Factor Compliance**: Architecture validation against cloud-native principles

### Quality Assurance
- **Quality Gate Framework**: Blocking gates with confidence scoring (0.0-1.0)
- **ADR Generation**: Automatic Architecture Decision Records
- **Test Coverage Validation**: Configurable thresholds with gap analysis
- **Definition of Done**: Automated DoD validation per story

### Memory & Context
- **ChromaDB Vector Storage**: Semantic search for decisions and patterns
- **Pattern Learning**: Automatic detection of codebase naming/structure conventions
- **Session Persistence**: Context preservation across agent handoffs
- **Project Isolation**: Multi-tenant support with data separation

### Observability
- **Audit Trail**: Complete decision logging with traceability
- **Token Cost Tracking**: LLM usage monitoring per agent
- **Real-time Activity Display**: Live progress visualization
- **Export Formats**: JSON, Markdown, and structured reports

---

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/bbengt1/yolo-developer.git
cd yolo-developer

# Install dependencies
uv sync

# Verify installation
uv run yolo --help
```

### Install with Development Dependencies

```bash
uv sync --all-extras
```

---

## Quick Start

### 1. Initialize a Project

```bash
# Create a new YOLO project in the current directory
yolo init

# Or specify a project name
yolo init --name my-project
```

### 2. Configure API Keys

```bash
# Set your LLM API keys (never stored in config files)
export YOLO_LLM__OPENAI_API_KEY=sk-...
export YOLO_LLM__ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Seed Requirements

```bash
# From a requirements document
yolo seed requirements.md

# Or create a quick requirements file
cat <<'EOF' > requirements.md
Build a REST API for user management with JWT authentication.
EOF
yolo seed requirements.md
```

### 4. Run Autonomous Development

```bash
# Start the autonomous development sprint
yolo run

# Check status
yolo status

# View logs
yolo logs
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `yolo init` | Initialize a new YOLO project |
| `yolo seed <file>` | Seed requirements from a document |
| `yolo run` | Execute autonomous development sprint |
| `yolo status` | Display current sprint status |
| `yolo logs` | View agent activity logs |
| `yolo config` | Manage project configuration |
| `yolo tune` | Adjust quality thresholds |
| `yolo mcp` | Start MCP server for Claude Code |

Run `yolo <command> --help` for detailed usage.

---

## Python SDK

```python
from yolo_developer import YoloClient

# Initialize client
client = YoloClient()

# Seed requirements programmatically
result = await client.seed("Build a user authentication system")

# Run autonomous sprint
sprint = await client.run()

# Access audit trail
audit = await client.get_audit_trail()
for entry in audit.entries:
    print(f"{entry.agent}: {entry.decision}")

# Register event hooks
@client.on("agent.completed")
async def on_agent_done(event):
    print(f"Agent {event.agent} completed: {event.summary}")
```

---

## MCP Integration

YOLO Developer exposes an MCP (Model Context Protocol) server for integration with Claude Code and other MCP-compatible clients.

### Start MCP Server

```bash
# STDIO transport (default, for Claude Desktop)
yolo mcp

# HTTP transport for remote access
yolo mcp --transport http --port 8080
```

### Quick MCP Tool Verification (CLI)

This uses an in-process FastMCP client to list tool names. Use this for local sanity checks; for real clients (Claude Desktop or HTTP) you still need a running MCP server.

```bash
uv run python - <<'PY'
import asyncio
from fastmcp import Client
from yolo_developer.mcp import mcp

async def main() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print([t.name for t in tools])

asyncio.run(main())
PY
```

### Claude Desktop Configuration

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "yolo-developer": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/yolo-developer", "yolo", "mcp"]
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `yolo_seed` | Provide seed requirements (text or file) |
| `yolo_run` | Execute autonomous sprint |
| `yolo_status` | Query sprint status |
| `yolo_audit` | Access audit trail (coming soon) |

### yolo_status Walkthrough

1) Seed requirements
```bash
cat <<'EOF' > requirements.md
Build a REST API for user management with JWT authentication.
EOF
yolo seed requirements.md
```

2) Start a sprint (returns `sprint_id`)
```json
{
  "status": "started",
  "sprint_id": "sprint-abcdef12",
  "seed_id": "550e8400-e29b-41d4-a716-446655440000",
  "thread_id": "thread-1234abcd",
  "started_at": "2026-01-21T09:12:34.567890+00:00"
}
```

3) Query sprint status
```json
{
  "status": "running",
  "sprint_id": "sprint-abcdef12",
  "seed_id": "550e8400-e29b-41d4-a716-446655440000",
  "thread_id": "thread-1234abcd",
  "started_at": "2026-01-21T09:12:34.567890+00:00",
  "completed_at": null,
  "error": null
}
```

4) Unknown sprint_id error
```json
{
  "status": "error",
  "error": "Sprint not found"
}
```

---

## Architecture

```
src/yolo_developer/
├── agents/              # AI Agent modules
│   ├── analyst/         # Requirement analysis & crystallization
│   ├── pm/              # Story generation & prioritization
│   ├── architect/       # Design decisions, ADRs, ATAM review
│   ├── dev/             # Code generation & testing
│   ├── tea/             # Test coverage & risk assessment
│   └── sm/              # Sprint management & orchestration
├── orchestrator/        # LangGraph workflow engine
├── memory/              # ChromaDB vector storage & patterns
├── gates/               # Quality gate framework
├── seed/                # Input validation & ambiguity detection
├── audit/               # Decision logging & traceability
├── cli/                 # Typer CLI interface
├── sdk/                 # Python SDK client
├── mcp/                 # FastMCP server
└── config/              # Pydantic configuration
```

### Agent Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Analyst   │────▶│     PM      │────▶│  Architect  │
│ (Req. Eng.) │     │  (Stories)  │     │  (Design)   │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                    ┌─────────────┐     ┌─────▼───────┐
                    │     TEA     │◀────│    Dev      │
                    │  (Testing)  │     │   (Code)    │
                    └─────────────┘     └─────────────┘
                           │
                    ┌──────▼──────┐
                    │     SM      │
                    │ (Orchestr.) │
                    └─────────────┘
```

---

## Configuration

### Configuration File

Create `yolo.yaml` in your project root:

```yaml
project_name: my-project

llm:
  smart_model: gpt-4o           # For complex reasoning
  routine_model: gpt-4o-mini    # For routine tasks
  # API keys must be set via environment variables

quality:
  test_coverage_threshold: 0.8
  gate_pass_threshold: 0.7
  blocking_gates:
    - testability
    - architecture_validation

memory:
  vector_store: chromadb
  persist_directory: .yolo/memory

agents:
  max_iterations: 10
  timeout_seconds: 300
```

### Environment Variables

Environment variables use `YOLO_` prefix with `__` as nested delimiter:

```bash
# API Keys (required)
export YOLO_LLM__OPENAI_API_KEY=sk-...
export YOLO_LLM__ANTHROPIC_API_KEY=sk-ant-...

# Override configuration values
export YOLO_PROJECT_NAME=my-project
export YOLO_QUALITY__TEST_COVERAGE_THRESHOLD=0.9
export YOLO_MEMORY__PERSIST_DIRECTORY=/custom/path
```

### Configuration Priority

1. **Defaults** (built-in)
2. **YAML file** (`yolo.yaml`)
3. **Environment variables** (highest priority)

---

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src/yolo_developer --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/agents/architect/test_pattern_matcher.py -v

# Run only unit tests
uv run pytest tests/unit -v
```

### Code Quality

```bash
# Type checking (strict mode)
uv run mypy src/yolo_developer

# Linting
uv run ruff check src tests

# Formatting
uv run ruff format src tests
```

### Project Structure

```
tests/
├── unit/           # Unit tests (~2700 tests)
├── integration/    # Integration tests (~3800 tests)
├── e2e/            # End-to-end tests
└── fixtures/       # Shared test fixtures
```

---

## Roadmap

### Current Status

| Epic | Status | Description |
|------|--------|-------------|
| 1-13 | ✅ **Complete** | Core infrastructure, all agents, CLI, SDK |
| 14 | 🚧 **In Progress** | MCP Integration (2/6 stories done) |

### Planned Features

#### LLM Providers
| Issue | Feature | Labels |
|-------|---------|--------|
| [#1](https://github.com/bbengt1/yolo-developer/issues/1) | Local LLM Support (Ollama, LM Studio, vLLM) | `enhancement` `epic` |
| [#8](https://github.com/bbengt1/yolo-developer/issues/8) | ChatGPT Codex Support | `enhancement` `epic` |

#### IDE Integrations
| Issue | Feature | Labels |
|-------|---------|--------|
| [#9](https://github.com/bbengt1/yolo-developer/issues/9) | Cursor IDE Support | `enhancement` `epic` `ide-integration` |
| [#10](https://github.com/bbengt1/yolo-developer/issues/10) | GitHub Copilot Support | `enhancement` `epic` `ide-integration` |

#### User Interfaces
| Issue | Feature | Labels |
|-------|---------|--------|
| [#3](https://github.com/bbengt1/yolo-developer/issues/3) | Web Interface | `enhancement` `epic` `frontend` |
| [#7](https://github.com/bbengt1/yolo-developer/issues/7) | Web Dashboard for Sprint Visualization | `enhancement` `frontend` |

#### Core Enhancements
| Issue | Feature | Labels |
|-------|---------|--------|
| [#2](https://github.com/bbengt1/yolo-developer/issues/2) | Brownfield Project Support | `enhancement` `epic` |
| [#6](https://github.com/bbengt1/yolo-developer/issues/6) | Plugin System for Custom Agents | `enhancement` `epic` |
| [#11](https://github.com/bbengt1/yolo-developer/issues/11) | Course Correction for Requirement Changes | `enhancement` `epic` |
| [#12](https://github.com/bbengt1/yolo-developer/issues/12) | GitHub Repository Management | `enhancement` `epic` `github-integration` |
| [#13](https://github.com/bbengt1/yolo-developer/issues/13) | GitHub Issue to User Story Conversion | `enhancement` `epic` `github-integration` |
| [#14](https://github.com/bbengt1/yolo-developer/issues/14) | Interactive Requirements Gathering | `enhancement` `epic` |

#### Performance
| Issue | Feature | Labels |
|-------|---------|--------|
| [#4](https://github.com/bbengt1/yolo-developer/issues/4) | Context / Token Efficiencies | `enhancement` `performance` |
| [#5](https://github.com/bbengt1/yolo-developer/issues/5) | Performance Optimization for Large Codebases | `enhancement` `performance` |
| [#15](https://github.com/bbengt1/yolo-developer/issues/15) | Token Limit Scheduler | `enhancement` `epic` `performance` |

See all issues: [GitHub Issues](https://github.com/bbengt1/yolo-developer/issues)

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Runtime | Python 3.10+ |
| Package Manager | uv |
| Orchestration | LangGraph |
| Vector Store | ChromaDB |
| Configuration | Pydantic v2 + YAML |
| CLI | Typer + Rich |
| MCP Server | FastMCP 2.x |
| Testing | pytest + pytest-asyncio |
| Linting | ruff |
| Type Checking | mypy (strict) |

---

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`uv run ruff check && uv run mypy src`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with the [BMad Method](https://github.com/bmadcode/BMAD-METHOD) for AI-assisted development
- Powered by [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- Uses [FastMCP](https://gofastmcp.com) for MCP protocol support
