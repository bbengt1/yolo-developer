---
layout: default
title: CLI Reference
nav_order: 4
has_children: true
---

# CLI Reference
{: .no_toc }

Complete documentation for all YOLO Developer command-line interface commands.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The YOLO Developer CLI (`yolo`) provides commands for initializing projects, seeding requirements, running autonomous development sprints, and monitoring progress.

### Basic Usage

```bash
yolo [OPTIONS] COMMAND [ARGS]...
```

### Global Options

| Option | Description |
|:-------|:------------|
| `--version` | Show version and exit |
| `--help` | Show help message and exit |
| `-v, --verbose` | Enable verbose output |
| `-q, --quiet` | Suppress non-essential output |
| `--config PATH` | Use specific config file |

---

## Command Summary

| Command | Description |
|:--------|:------------|
| [`yolo init`](#yolo-init) | Initialize a new YOLO project |
| [`yolo seed`](#yolo-seed) | Seed requirements for development |
| [`yolo run`](#yolo-run) | Execute autonomous development sprint |
| [`yolo status`](#yolo-status) | Display current sprint status |
| [`yolo logs`](#yolo-logs) | View agent activity logs |
| [`yolo config`](#yolo-config) | Manage project configuration |
| [`yolo tune`](#yolo-tune) | Adjust quality thresholds |
| [`yolo mcp`](#yolo-mcp) | Start MCP server for Claude Code |
| [`yolo scan`](#yolo-scan) | Scan existing project for brownfield context |
| [`yolo git`](#yolo-git) | Local git operations |
| [`yolo pr`](#yolo-pr) | Pull request operations |
| [`yolo issue`](#yolo-issue) | Issue operations |
| [`yolo release`](#yolo-release) | Release operations |
| [`yolo workflow`](#yolo-workflow) | GitHub workflow automation |
| [`yolo import`](#yolo-import) | Import GitHub issues |
| [`yolo gather`](#yolo-gather) | Interactive requirements gathering |

---

## yolo init

Initialize a new YOLO Developer project in the current directory.

### Synopsis

```bash
yolo init [OPTIONS] [PATH]
```

### Options

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `--name, -n` | TEXT | Directory name | Project name |
| `--author, -a` | TEXT | git config | Author name for pyproject.toml |
| `--email, -e` | TEXT | git config | Author email for pyproject.toml |
| `--interactive, -i` | FLAG | False | Prompt for project details |
| `--no-input` | FLAG | False | Use defaults without prompting |
| `--existing, --brownfield` | FLAG | False | Add YOLO to existing project |
| `--scan-only` | FLAG | False | Scan existing project without changes |
| `--non-interactive` | FLAG | False | Skip brownfield prompts |
| `--hint` | TEXT | None | Hint about project type |

### Examples

**Basic initialization:**
```bash
yolo init
```

**Output:**
```
Initializing YOLO Developer project...

? Project name: my-api
? Author name: Developer
? Author email: dev@example.com

Creating configuration...
  ✓ Created yolo.yaml
  ✓ Created .yolo/ directory
  ✓ Initialized memory store

Project initialized successfully!
```

**Brownfield initialization:**
```bash
yolo init --brownfield
```

**Scan only (no changes):**
```bash
yolo init --brownfield --scan-only
```

### Files Created

| File | Description |
|:-----|:------------|
| `yolo.yaml` | Project configuration |
| `.yolo/` | YOLO data directory |
| `.yolo/memory/` | ChromaDB vector store |
| `.yolo/cache/` | Agent response cache |

---

## yolo seed

Seed requirements for autonomous development.

### Synopsis

```bash
yolo seed [OPTIONS] [FILE]
```

### Arguments

| Argument | Type | Required | Description |
|:---------|:-----|:---------|:------------|
| `FILE` | PATH | No* | Path to requirements document |

*Required unless `--text` is provided.

### Options

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `--text, -t` | TEXT | None | Provide requirements as inline text |
| `--format, -f` | CHOICE | auto | Input format: auto, markdown, text |
| `--validate-only` | FLAG | False | Validate without storing seed |
| `--skip-validation` | FLAG | False | Skip ambiguity/contradiction checks |
| `--output, -o` | PATH | None | Export parsed requirements to file |

### Examples

**Seed from file:**
```bash
yolo seed requirements.md
```

**Output:**
```
Parsing requirements document...
  ✓ Parsed 12 requirements

Running validation...
  ✓ No ambiguities detected
  ✓ No contradictions found
  ✓ SOP constraints validated

Seed Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Requirements: 12 total
    - Functional: 8
    - Non-Functional: 3
    - Constraints: 1
  Quality Score: 0.94

Seed ID: seed_a1b2c3d4-e5f6-7890-abcd-ef1234567890

Ready to run: yolo run
```

**Seed from inline text:**
```bash
yolo seed --text "Build a REST API with user authentication using JWT tokens"
```

**Validate without seeding:**
```bash
yolo seed requirements.md --validate-only
```

**Output with ambiguities detected:**
```
Parsing requirements document...
  ✓ Parsed 8 requirements

Running validation...
  ⚠ 2 ambiguities detected:
    1. "fast response times" - What is considered fast? (< 100ms? < 500ms?)
    2. "support many users" - Specific concurrency target needed

  ✓ No contradictions found
  ✓ SOP constraints validated

Clarification questions generated:
  Q1: What response time threshold defines "fast"?
      Suggestions: < 100ms, < 200ms, < 500ms
  Q2: How many concurrent users should be supported?
      Suggestions: 100, 1000, 10000

Fix ambiguities and re-run, or use --skip-validation to proceed.
```

### Input Formats

**Markdown (recommended):**
```markdown
# Project Requirements

## Functional Requirements
- User can register with email
- User can login with password

## Non-Functional Requirements
- Response time < 200ms
- 99.9% uptime

## Constraints
- Must use PostgreSQL
```

**Plain text:**
```
User Management System

The system must allow users to register with their email address.
Users must be able to login with a password.
Response times should be under 200ms.
The system must use PostgreSQL for data storage.
```

---

## yolo run

Execute an autonomous development sprint.

### Synopsis

```bash
yolo run [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `--seed-id` | TEXT | Latest | Specific seed ID to use |
| `--agents` | TEXT | All | Comma-separated list of agents to run |
| `--max-iterations` | INT | 10 | Maximum iterations per agent |
| `--timeout` | INT | 300 | Timeout in seconds per agent |
| `--dry-run` | FLAG | False | Simulate without making changes |
| `--continue` | FLAG | False | Continue from last checkpoint |
| `--watch` | FLAG | False | Watch mode with live updates |
| `--output-dir` | PATH | . | Output directory for artifacts |

### Examples

**Basic run:**
```bash
yolo run
```

**Output:**
```
Starting autonomous development sprint...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ANALYST] Analyzing requirements...
  → Crystallizing 12 requirements
  ✓ Completed in 2m 34s

[PM] Generating user stories...
  → Creating stories from requirements
  ✓ Generated 8 stories in 1m 12s

[ARCHITECT] Designing system...
  → Validating 12-Factor compliance
  → Generating ADRs
  ✓ Completed in 3m 45s

[DEV] Implementing stories...
  → US-001: User Registration
  → US-002: User Authentication
  ...
  ✓ Completed 8 stories in 15m 22s

[TEA] Validating tests...
  → Running test suite
  → Analyzing coverage
  ✓ Coverage: 87% in 2m 10s

[SM] Finalizing sprint...
  ✓ Sprint complete!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sprint Summary:
  Duration: 25m 03s
  Stories: 8/8 completed
  Coverage: 87%
  Quality Gates: 4/4 passing
  Tokens: 145,230 ($1.42)
```

**Run specific agents only:**
```bash
yolo run --agents analyst,pm,architect
```

**Dry run (no changes):**
```bash
yolo run --dry-run
```

**Continue from checkpoint:**
```bash
yolo run --continue
```

**Watch mode with live updates:**
```bash
yolo run --watch
```

### Agent Execution Order

1. **Analyst** - Requirement crystallization
2. **PM** - Story generation
3. **Architect** - System design
4. **Dev** - Implementation
5. **TEA** - Test validation
6. **SM** - Orchestration (runs throughout)

---

## yolo status

Display current sprint status and progress.

### Synopsis

```bash
yolo status [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `--format, -f` | CHOICE | table | Output format: table, json, yaml |
| `--agents` | FLAG | False | Show detailed agent status |
| `--gates` | FLAG | False | Show quality gate details |
| `--stories` | FLAG | False | Show story breakdown |
| `--watch, -w` | FLAG | False | Live updating status |
| `--refresh` | INT | 5 | Refresh interval for watch mode |

### Examples

**Basic status:**
```bash
yolo status
```

**Output:**
```
Sprint Status: IN_PROGRESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Progress: ████████████░░░░░░░░ 60%

Stories: 5/8 completed
  ✓ US-001: User Registration
  ✓ US-002: User Authentication
  ✓ US-003: Profile View
  ✓ US-004: Profile Update
  ✓ US-005: Password Reset
  → US-006: Admin User List (in progress)
  ○ US-007: Admin User Edit
  ○ US-008: Admin User Delete

Current Agent: DEV
  Task: Implementing US-006 Admin User List
  Duration: 3m 42s

Quality Gates: 4/4 passing
Token Usage: 89,450 tokens ($0.87)
```

**Detailed agent status:**
```bash
yolo status --agents
```

**Output:**
```
Agent Status:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANALYST
  Status: COMPLETED
  Duration: 2m 34s
  Output: 12 crystallized requirements
  Tokens: 12,340

PM
  Status: COMPLETED
  Duration: 1m 12s
  Output: 8 user stories
  Tokens: 8,920

ARCHITECT
  Status: COMPLETED
  Duration: 3m 45s
  Output: 3 ADRs, 2 risk reports
  Tokens: 15,670

DEV
  Status: IN_PROGRESS
  Duration: 8m 22s (ongoing)
  Current: US-006 Admin User List
  Completed: 5/8 stories
  Tokens: 45,230

TEA
  Status: PENDING
  Waiting for: DEV completion

SM
  Status: MONITORING
  Health: HEALTHY
  Iterations: 12
  Escalations: 0
```

**Quality gate details:**
```bash
yolo status --gates
```

**Output:**
```
Quality Gates:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ TESTABILITY
  Score: 0.89 (threshold: 0.70)
  Details: All stories have testable acceptance criteria

✓ AC_MEASURABILITY
  Score: 0.92 (threshold: 0.70)
  Details: 95% of ACs have measurable outcomes

✓ ARCHITECTURE_VALIDATION
  Score: 0.85 (threshold: 0.70)
  Details: 12-Factor compliant, no circular dependencies

✓ DEFINITION_OF_DONE
  Score: 0.88 (threshold: 0.70)
  Details: All completed stories meet DoD criteria
```

**JSON output for scripting:**
```bash
yolo status --format json
```

**Output:**
```json
{
  "status": "IN_PROGRESS",
  "progress": 0.6,
  "stories": {
    "total": 8,
    "completed": 5,
    "in_progress": 1,
    "pending": 2
  },
  "current_agent": "DEV",
  "quality_gates": {
    "testability": {"score": 0.89, "passing": true},
    "ac_measurability": {"score": 0.92, "passing": true},
    "architecture_validation": {"score": 0.85, "passing": true},
    "definition_of_done": {"score": 0.88, "passing": true}
  },
  "tokens": {
    "total": 89450,
    "cost_usd": 0.87
  }
}
```

---

## yolo logs

View agent activity logs and audit trail.

### Synopsis

```bash
yolo logs [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `--agent, -a` | TEXT | All | Filter by agent name |
| `--level, -l` | CHOICE | info | Log level: debug, info, warn, error |
| `--since` | TEXT | None | Show logs since time (e.g., 1h, 30m) |
| `--until` | TEXT | None | Show logs until time |
| `--limit, -n` | INT | 100 | Maximum number of entries |
| `--follow, -f` | FLAG | False | Follow log output in real-time |
| `--export` | PATH | None | Export logs to file |
| `--format` | CHOICE | text | Output format: text, json |

### Examples

**View recent logs:**
```bash
yolo logs
```

**Output:**
```
2024-01-15 10:23:45 [INFO] [ANALYST] Starting requirement analysis
2024-01-15 10:23:47 [INFO] [ANALYST] Parsing 12 requirements
2024-01-15 10:24:12 [INFO] [ANALYST] Crystallized: "User registration" → 3 specs
2024-01-15 10:24:35 [INFO] [ANALYST] Crystallized: "User authentication" → 2 specs
2024-01-15 10:25:02 [WARN] [ANALYST] Ambiguity detected: "fast response"
2024-01-15 10:25:03 [INFO] [ANALYST] Generated clarification question
2024-01-15 10:26:19 [INFO] [ANALYST] Analysis complete: 12 requirements → 18 specs
2024-01-15 10:26:20 [INFO] [PM] Starting story generation
2024-01-15 10:26:45 [INFO] [PM] Created story: US-001 User Registration
2024-01-15 10:27:02 [INFO] [PM] Created story: US-002 User Authentication
```

**Filter by agent:**
```bash
yolo logs --agent dev
```

**Follow logs in real-time:**
```bash
yolo logs --follow
```

**Export to JSON:**
```bash
yolo logs --export audit.json --format json
```

**Output (audit.json):**
```json
{
  "entries": [
    {
      "timestamp": "2024-01-15T10:23:45Z",
      "level": "INFO",
      "agent": "ANALYST",
      "message": "Starting requirement analysis",
      "context": {
        "seed_id": "seed_abc123",
        "requirement_count": 12
      }
    },
    {
      "timestamp": "2024-01-15T10:24:12Z",
      "level": "INFO",
      "agent": "ANALYST",
      "message": "Crystallized requirement",
      "context": {
        "original": "User registration",
        "specs_generated": 3,
        "confidence": 0.92
      }
    }
  ]
}
```

---

## yolo config

Manage project configuration.

### Synopsis

```bash
yolo config [COMMAND] [OPTIONS]
```

### Commands

| Command | Description |
|:--------|:------------|
| `show` | Display current configuration |
| `set` | Set a configuration value |
| `get` | Get a specific configuration value |
| `reset` | Reset to default configuration |
| `validate` | Validate configuration file |

### Examples

**Show configuration:**
```bash
yolo config show
```

**Output:**
```yaml
project_name: my-api
llm:
  provider: auto
  cheap_model: gpt-4o-mini
  premium_model: claude-sonnet-4-20250514
  best_model: claude-opus-4-5-20251101
  openai_api_key: "**********" (configured)
  anthropic_api_key: "**********" (configured)
  openai:
    code_model: gpt-4o
  hybrid:
    enabled: false
quality:
  test_coverage_threshold: 0.8
  gate_pass_threshold: 0.7
memory:
  persist_path: .yolo/memory
  vector_store_type: chromadb
  graph_store_type: json
agents:
  max_iterations: 10
  timeout_seconds: 300
```

**Set a value:**
```bash
yolo config set quality.test_coverage_threshold 0.9
```

**Get a specific value:**
```bash
yolo config get llm.premium_model
```

**Output:**
```
claude-sonnet-4-20250514
```

**Validate configuration:**
```bash
yolo config validate
```

**Output:**
```
Validating yolo.yaml...
  ✓ Schema valid
  ✓ API keys configured
  ✓ Memory directory writable
  ✓ All settings valid

Configuration is valid.
```

---

## yolo tune

Adjust quality thresholds interactively or via options.

### Synopsis

```bash
yolo tune [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `--coverage` | FLOAT | None | Set test coverage threshold (0.0-1.0) |
| `--gate-pass` | FLOAT | None | Set gate pass threshold (0.0-1.0) |
| `--interactive, -i` | FLAG | True | Interactive tuning mode |
| `--preset` | CHOICE | None | Use preset: strict, balanced, relaxed |

### Examples

**Interactive tuning:**
```bash
yolo tune
```

**Output:**
```
Quality Threshold Tuning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Current thresholds:
  Test Coverage: 0.80 (80%)
  Gate Pass: 0.70 (70%)

? Adjust test coverage threshold?
  Current: 0.80
  [←/→] to adjust, [Enter] to confirm
  ████████████████████░░░░░ 0.80

? Adjust gate pass threshold?
  Current: 0.70
  [←/→] to adjust, [Enter] to confirm
  ██████████████░░░░░░░░░░░ 0.70

Updated thresholds saved to yolo.yaml
```

**Direct threshold setting:**
```bash
yolo tune --coverage 0.9 --gate-pass 0.8
```

**Use preset:**
```bash
yolo tune --preset strict
```

**Presets:**
| Preset | Coverage | Gate Pass |
|:-------|:---------|:----------|
| strict | 0.95 | 0.90 |
| balanced | 0.80 | 0.70 |
| relaxed | 0.60 | 0.50 |

---

## yolo mcp

Start MCP server for Claude Code integration.

### Synopsis

```bash
yolo mcp [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `--transport, -t` | CHOICE | stdio | Transport: stdio, http |
| `--port, -p` | INT | 8080 | Port for HTTP transport |
| `--host` | TEXT | 127.0.0.1 | Host for HTTP transport |

### Examples

**Start with STDIO (for Claude Desktop):**
```bash
yolo mcp
```

**Start with HTTP transport:**
```bash
yolo mcp --transport http --port 8080
```

**Output:**
```
Starting YOLO Developer MCP Server...
  Transport: HTTP
  Address: http://127.0.0.1:8080

Available tools:
  - yolo_seed: Provide seed requirements
  - yolo_run: Execute sprint
  - yolo_status: Get status
  - yolo_audit: Access audit

Server ready. Press Ctrl+C to stop.
```

See [MCP Integration](/yolo-developer/mcp/) for detailed usage.

---

## yolo scan

Scan an existing repository and optionally write `.yolo/project-context.yaml`.

### Synopsis

```bash
yolo scan [OPTIONS] [PATH]
```

### Options

| Option | Type | Default | Description |
|:-------|:-----|:--------|:------------|
| `--scan-depth` | INT | 3 | Directory depth to scan |
| `--max-files` | INT | 1000 | Maximum files to analyze |
| `--git-history/--no-git-history` | FLAG | from config | Include git history analysis |
| `--interactive, -i` | FLAG | False | Prompt for ambiguous findings |
| `--hint` | TEXT | None | Hint about project type |
| `--refresh` | FLAG | False | Overwrite existing project-context.yaml |
| `--write-context/--no-write-context` | FLAG | True | Write project context file |

### Examples

```bash
yolo scan
yolo scan --refresh
yolo scan --max-files 200 --scan-depth 4
```

---

## yolo git

Manage local Git operations.

```bash
yolo git status
yolo git commit -m "feat: update"
yolo git push
```

---

## yolo pr

Manage pull requests.

```bash
yolo pr create --title "Title" --body "Body"
yolo pr merge 123 --method squash
```

---

## yolo issue

Manage GitHub issues.

```bash
yolo issue create --title "Bug" --body "Details"
yolo issue close 123 --comment "Fixed in PR #456"
```

---

## yolo release

Create GitHub releases.

```bash
yolo release create --tag v1.2.0 --name "Release 1.2.0" --body "Notes"
```

---

## yolo workflow

Automate story workflows.

```bash
yolo workflow start US-001 --title "Add endpoint" --description "..."
yolo workflow complete US-001 --title "Add endpoint" --description "..." --commit "feat: add endpoint"
```

---

## yolo import

Import GitHub issues and convert them into user stories.

```bash
yolo import issue 42
yolo import preview 42
yolo import issues --label "ready" --auto-seed
```

Common options:

- `--repo owner/repo`: override repository
- `--auto-seed`: write seed file to `.yolo/imported-issues/`
- `--preview`: preview only
- `--output`: export story to file

---

## yolo gather

Interactive requirements gathering with the Analyst agent.

```bash
yolo gather start my-project --description "Build a task manager"
yolo gather list
yolo gather export 20250122093000 --format markdown --output requirements.md
```

---

## Exit Codes
## Exit Codes

| Code | Description |
|:-----|:------------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Validation error |
| 4 | API error (LLM) |
| 5 | Quality gate failure |
| 130 | Interrupted (Ctrl+C) |

---

## Environment Variables

All configuration can be overridden via environment variables using the `YOLO_` prefix:

| Variable | Description |
|:---------|:------------|
| `YOLO_PROJECT_NAME` | Project name |
| `YOLO_LLM__PROVIDER` | Primary LLM provider (auto/openai/anthropic/hybrid) |
| `YOLO_LLM__CHEAP_MODEL` | Model for routine tasks |
| `YOLO_LLM__PREMIUM_MODEL` | Model for complex tasks |
| `YOLO_LLM__BEST_MODEL` | Model for critical tasks |
| `YOLO_LLM__OPENAI__API_KEY` | OpenAI API key (preferred) |
| `YOLO_LLM__OPENAI_API_KEY` | OpenAI API key (legacy) |
| `YOLO_LLM__ANTHROPIC_API_KEY` | Anthropic API key |
| `YOLO_LLM__OPENAI__CODE_MODEL` | OpenAI model for code tasks |
| `YOLO_LLM__HYBRID__ENABLED` | Enable hybrid routing |
| `YOLO_QUALITY__TEST_COVERAGE_THRESHOLD` | Coverage threshold |
| `YOLO_QUALITY__GATE_PASS_THRESHOLD` | Gate pass threshold |
| `YOLO_MEMORY__PERSIST_PATH` | Memory storage path |
| `YOLO_MEMORY__VECTOR_STORE_TYPE` | Vector store backend |
| `YOLO_MEMORY__GRAPH_STORE_TYPE` | Graph store backend |
| `YOLO_BROWNFIELD__SCAN_DEPTH` | Brownfield scan depth |
| `YOLO_BROWNFIELD__MAX_FILES_TO_ANALYZE` | Brownfield scan file limit |
| `YOLO_GITHUB__TOKEN` | GitHub token |
| `YOLO_GITHUB__REPOSITORY` | GitHub repo slug (owner/repo) |

---

## Next Steps

- [MCP Integration](/yolo-developer/mcp/) - Use with Claude Code
- [Python SDK](/yolo-developer/sdk/) - Programmatic API
- [Configuration](/yolo-developer/configuration/) - All options
