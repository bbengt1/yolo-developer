---
layout: default
title: Configuration
nav_order: 7
has_children: true
---

# Configuration
{: .no_toc }

Complete reference for all YOLO Developer configuration options.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

YOLO Developer uses a three-layer configuration system:

1. **Defaults** - Built-in sensible defaults
2. **YAML file** - Project-specific `yolo.yaml`
3. **Environment variables** - Runtime overrides (highest priority)

---

## Configuration File

Create `yolo.yaml` in your project root:

```yaml
# Project identification
project_name: my-awesome-api

# LLM configuration
llm:
  provider: auto
  cheap_model: gpt-4o-mini
  premium_model: claude-sonnet-4-20250514
  best_model: claude-opus-4-5-20251101
  openai:
    code_model: gpt-4o
  # API keys should be set via environment variables

# Quality thresholds
quality:
  test_coverage_threshold: 0.8
  gate_pass_threshold: 0.7
  blocking_gates:
    - testability
    - architecture_validation
    - definition_of_done

# Memory and storage
memory:
  vector_store_type: chromadb
  graph_store_type: json
  persist_path: .yolo/memory

# Agent configuration
agents:
  max_iterations: 10
  timeout_seconds: 300
  enable_caching: true

# Orchestration
orchestrator:
  max_agent_retries: 3
  circular_detection_threshold: 5
  human_escalation_enabled: true

# Audit and logging
audit:
  enabled: true
  log_level: info
  export_format: json
  retention_days: 30

# MCP server
mcp:
  transport: stdio
  http:
    port: 8080
    host: "127.0.0.1"
```

---

## Configuration Sections

### project_name

**Type:** `string`
**Required:** No (defaults to directory name)
**Environment:** `YOLO_PROJECT_NAME`

The name of your project. Used in logs, audit trail, and generated artifacts.

```yaml
project_name: my-awesome-api
```

---

### llm

LLM (Large Language Model) configuration for agent reasoning.

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `provider` | string | auto | `YOLO_LLM__PROVIDER` | Primary provider (auto/openai/anthropic/hybrid) |
| `cheap_model` | string | gpt-4o-mini | `YOLO_LLM__CHEAP_MODEL` | Model for routine tasks |
| `premium_model` | string | claude-sonnet-4-20250514 | `YOLO_LLM__PREMIUM_MODEL` | Model for complex reasoning |
| `best_model` | string | claude-opus-4-5-20251101 | `YOLO_LLM__BEST_MODEL` | Model for critical decisions |
| `openai_api_key` | string | None | `YOLO_LLM__OPENAI_API_KEY` | OpenAI API key (legacy) |
| `anthropic_api_key` | string | None | `YOLO_LLM__ANTHROPIC_API_KEY` | Anthropic API key |

{: .warning }
> Never put API keys in `yolo.yaml`. Always use environment variables.

```yaml
llm:
  provider: auto
  cheap_model: gpt-4o-mini
  premium_model: claude-sonnet-4-20250514
  best_model: claude-opus-4-5-20251101
  openai:
    cheap_model: gpt-4o-mini
    premium_model: gpt-4o
    code_model: gpt-4o
    reasoning_model: null
  hybrid:
    enabled: false
    routing:
      code_generation: openai
      code_review: openai
      architecture: anthropic
      analysis: anthropic
      documentation: openai
      testing: openai
```

**Environment Variables:**
```bash
export YOLO_LLM__OPENAI__API_KEY=sk-proj-...
export YOLO_LLM__OPENAI_API_KEY=sk-proj-...  # Legacy
export YOLO_LLM__ANTHROPIC_API_KEY=sk-ant-...
export YOLO_LLM__PREMIUM_MODEL=claude-sonnet-4-20250514
```

#### llm.openai

OpenAI/Codex configuration for code-optimized models.

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `api_key` | string | None | `YOLO_LLM__OPENAI__API_KEY` | OpenAI API key (preferred) |
| `cheap_model` | string | gpt-4o-mini | `YOLO_LLM__OPENAI__CHEAP_MODEL` | OpenAI model for routine tasks |
| `premium_model` | string | gpt-4o | `YOLO_LLM__OPENAI__PREMIUM_MODEL` | OpenAI model for complex reasoning |
| `code_model` | string | gpt-4o | `YOLO_LLM__OPENAI__CODE_MODEL` | OpenAI model for code tasks |
| `reasoning_model` | string | None | `YOLO_LLM__OPENAI__REASONING_MODEL` | OpenAI model for deep reasoning |

#### llm.hybrid

Hybrid routing configuration for task-based provider selection.

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `enabled` | bool | false | `YOLO_LLM__HYBRID__ENABLED` | Enable hybrid routing |
| `routing.code_generation` | string | openai | `YOLO_LLM__HYBRID__ROUTING__CODE_GENERATION` | Provider for code generation |
| `routing.code_review` | string | openai | `YOLO_LLM__HYBRID__ROUTING__CODE_REVIEW` | Provider for code review |
| `routing.architecture` | string | anthropic | `YOLO_LLM__HYBRID__ROUTING__ARCHITECTURE` | Provider for architecture |
| `routing.analysis` | string | anthropic | `YOLO_LLM__HYBRID__ROUTING__ANALYSIS` | Provider for analysis |
| `routing.documentation` | string | openai | `YOLO_LLM__HYBRID__ROUTING__DOCUMENTATION` | Provider for documentation |
| `routing.testing` | string | openai | `YOLO_LLM__HYBRID__ROUTING__TESTING` | Provider for testing |

#### Supported Models

**OpenAI:**
- `gpt-4o` (recommended for premium/code)
- `gpt-4o-mini` (recommended for cheap)
- `gpt-4-turbo`
- `gpt-3.5-turbo`
 - `o1-preview` (deep reasoning)
 - `o1-mini`

**Anthropic:**
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

---

### quality

Quality gate thresholds and configuration.

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `test_coverage_threshold` | float | 0.8 | `YOLO_QUALITY__TEST_COVERAGE_THRESHOLD` | Minimum test coverage (0.0-1.0) |
| `gate_pass_threshold` | float | 0.7 | `YOLO_QUALITY__GATE_PASS_THRESHOLD` | Minimum gate score (0.0-1.0) |
| `blocking_gates` | list | [testability, architecture_validation] | `YOLO_QUALITY__BLOCKING_GATES` | Gates that block progress |
| `warning_gates` | list | [] | `YOLO_QUALITY__WARNING_GATES` | Gates that only warn |

```yaml
quality:
  test_coverage_threshold: 0.8
  gate_pass_threshold: 0.7
  blocking_gates:
    - testability
    - ac_measurability
    - architecture_validation
    - definition_of_done
  warning_gates:
    - code_complexity
```

#### Available Gates

| Gate | Description |
|:-----|:------------|
| `testability` | Validates stories have testable acceptance criteria |
| `ac_measurability` | Checks acceptance criteria are measurable |
| `architecture_validation` | Validates design against patterns and constraints |
| `definition_of_done` | Checks completed work meets DoD criteria |
| `code_complexity` | Analyzes cyclomatic complexity |
| `security_scan` | Runs security checks on generated code |

---

### memory

Memory and vector storage configuration.

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `persist_path` | string | .yolo/memory | `YOLO_MEMORY__PERSIST_PATH` | Storage location |
| `vector_store_type` | string | chromadb | `YOLO_MEMORY__VECTOR_STORE_TYPE` | Vector store backend |
| `graph_store_type` | string | json | `YOLO_MEMORY__GRAPH_STORE_TYPE` | Graph store backend |

```yaml
memory:
  persist_path: .yolo/memory
  vector_store_type: chromadb
  graph_store_type: json
```

#### Vector Store Types

| Backend | Description |
|:--------|:------------|
| `chromadb` | Default, local ChromaDB instance |

#### Graph Store Types

| Backend | Description |
|:--------|:------------|
| `json` | Default, local JSON graph store |
| `neo4j` | Optional Neo4j backend |

---

### brownfield

Brownfield scanning configuration for existing projects.

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `scan_depth` | int | 3 | `YOLO_BROWNFIELD__SCAN_DEPTH` | Directory depth to scan |
| `exclude_patterns` | list | node_modules, .git, __pycache__, .venv, venv | `YOLO_BROWNFIELD__EXCLUDE_PATTERNS` | Patterns to skip during scan |
| `include_git_history` | bool | true | `YOLO_BROWNFIELD__INCLUDE_GIT_HISTORY` | Include git metadata |
| `max_files_to_analyze` | int | 1000 | `YOLO_BROWNFIELD__MAX_FILES_TO_ANALYZE` | File limit for scanning |
| `interactive` | bool | true | `YOLO_BROWNFIELD__INTERACTIVE` | Prompt for ambiguous findings |

```yaml
brownfield:
  scan_depth: 3
  exclude_patterns:
    - node_modules
    - .git
    - __pycache__
    - .venv
    - venv
  include_git_history: true
  max_files_to_analyze: 1000
  interactive: true
```

---

### analyst

Interactive requirements gathering configuration.

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `gathering.enabled` | bool | true | `YOLO_ANALYST__GATHERING__ENABLED` | Enable gathering sessions |
| `gathering.storage_path` | string | .yolo/sessions | `YOLO_ANALYST__GATHERING__STORAGE_PATH` | Session storage |
| `gathering.max_questions_per_phase` | int | 5 | `YOLO_ANALYST__GATHERING__MAX_QUESTIONS_PER_PHASE` | Question cap |

```yaml
analyst:
  gathering:
    enabled: true
    storage_path: .yolo/sessions
    max_questions_per_phase: 5
```

---

### web

Local web UI configuration.

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `enabled` | bool | true | `YOLO_WEB__ENABLED` | Enable web UI |
| `host` | string | 127.0.0.1 | `YOLO_WEB__HOST` | Host to bind |
| `port` | int | 3000 | `YOLO_WEB__PORT` | Port to bind |
| `api_only` | bool | false | `YOLO_WEB__API_ONLY` | Run API only |
| `uploads.enabled` | bool | true | `YOLO_WEB__UPLOADS__ENABLED` | Enable uploads |
| `uploads.max_size_mb` | int | 10 | `YOLO_WEB__UPLOADS__MAX_SIZE_MB` | Max upload size |
| `uploads.storage_path` | string | .yolo/uploads | `YOLO_WEB__UPLOADS__STORAGE_PATH` | Upload storage |

```yaml
web:
  enabled: true
  host: 127.0.0.1
  port: 3000
  api_only: false
  uploads:
    enabled: true
    max_size_mb: 10
    storage_path: .yolo/uploads
```

---

### github

GitHub automation configuration for repository management.

Prerequisite: install the GitHub CLI (`gh`) and authenticate (`gh auth login`).

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `enabled` | bool | false | `YOLO_GITHUB__ENABLED` | Enable GitHub automation |
| `token` | string | None | `YOLO_GITHUB__TOKEN` | GitHub token (env only) |
| `repository` | string | None | `YOLO_GITHUB__REPOSITORY` | owner/repo override |
| `default_branch` | string | main | `YOLO_GITHUB__DEFAULT_BRANCH` | Default branch |
| `branch_prefix` | string | feature/ | `YOLO_GITHUB__BRANCH_PREFIX` | Branch prefix |

```yaml
github:
  enabled: true
  repository: bbengt1/yolo-developer
  default_branch: main
  branch_prefix: feature/
  automation:
    auto_commit: true
    auto_push: true
    auto_pr: true
  pull_requests:
    draft_by_default: false
    merge_method: squash
  issues:
    create_from_stories: true
  releases:
    generate_notes: true
  commits:
    conventional: true
  import_config:
    enabled: true
    update_issues: true
    add_label: yolo-imported
    story:
      id_prefix: US
      include_technical_notes: true
      estimate_points: true
```

#### github.import_config

GitHub issue import configuration.

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `enabled` | bool | true | `YOLO_GITHUB__IMPORT_CONFIG__ENABLED` | Enable issue import |
| `default_repo` | string | None | `YOLO_GITHUB__IMPORT_CONFIG__DEFAULT_REPO` | Default repo for imports |
| `update_issues` | bool | true | `YOLO_GITHUB__IMPORT_CONFIG__UPDATE_ISSUES` | Post back to issues |
| `add_label` | string | yolo-imported | `YOLO_GITHUB__IMPORT_CONFIG__ADD_LABEL` | Label to apply |
| `add_comment` | bool | true | `YOLO_GITHUB__IMPORT_CONFIG__ADD_COMMENT` | Comment with story summary |

#### github.import_config.story

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `id_prefix` | string | US | `YOLO_GITHUB__IMPORT_CONFIG__STORY__ID_PREFIX` | Story ID prefix |
| `include_technical_notes` | bool | true | `YOLO_GITHUB__IMPORT_CONFIG__STORY__INCLUDE_TECHNICAL_NOTES` | Include technical notes |
| `estimate_points` | bool | true | `YOLO_GITHUB__IMPORT_CONFIG__STORY__ESTIMATE_POINTS` | Estimate points |

---

### agents

Agent execution configuration.

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `max_iterations` | int | 10 | `YOLO_AGENTS__MAX_ITERATIONS` | Max iterations per agent |
| `timeout_seconds` | int | 300 | `YOLO_AGENTS__TIMEOUT_SECONDS` | Agent timeout |
| `enable_caching` | bool | true | `YOLO_AGENTS__ENABLE_CACHING` | Cache agent responses |
| `cache_ttl_hours` | int | 24 | `YOLO_AGENTS__CACHE_TTL_HOURS` | Cache expiry time |
| `parallel_stories` | int | 1 | `YOLO_AGENTS__PARALLEL_STORIES` | Stories to process in parallel |

```yaml
agents:
  max_iterations: 10
  timeout_seconds: 300
  enable_caching: true
  cache_ttl_hours: 24
  parallel_stories: 1
```

#### Per-Agent Configuration

```yaml
agents:
  max_iterations: 10

  analyst:
    max_iterations: 5
    timeout_seconds: 120

  dev:
    max_iterations: 15
    timeout_seconds: 600
    parallel_stories: 2

  tea:
    timeout_seconds: 180
```

---

### orchestrator

Orchestration and workflow configuration.

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `max_agent_retries` | int | 3 | `YOLO_ORCHESTRATOR__MAX_AGENT_RETRIES` | Retries on agent failure |
| `circular_detection_threshold` | int | 5 | `YOLO_ORCHESTRATOR__CIRCULAR_DETECTION_THRESHOLD` | Detect circular logic after N loops |
| `human_escalation_enabled` | bool | true | `YOLO_ORCHESTRATOR__HUMAN_ESCALATION_ENABLED` | Allow human escalation |
| `checkpoint_enabled` | bool | true | `YOLO_ORCHESTRATOR__CHECKPOINT_ENABLED` | Enable checkpoints |
| `checkpoint_interval` | int | 5 | `YOLO_ORCHESTRATOR__CHECKPOINT_INTERVAL` | Checkpoint every N operations |

```yaml
orchestrator:
  max_agent_retries: 3
  circular_detection_threshold: 5
  human_escalation_enabled: true
  checkpoint_enabled: true
  checkpoint_interval: 5
```

---

### audit

Audit trail and logging configuration.

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `enabled` | bool | true | `YOLO_AUDIT__ENABLED` | Enable audit logging |
| `log_level` | string | info | `YOLO_AUDIT__LOG_LEVEL` | Log level (debug, info, warn, error) |
| `export_format` | string | json | `YOLO_AUDIT__EXPORT_FORMAT` | Export format (json, markdown) |
| `retention_days` | int | 30 | `YOLO_AUDIT__RETENTION_DAYS` | Days to retain logs |
| `include_tokens` | bool | true | `YOLO_AUDIT__INCLUDE_TOKENS` | Track token usage |
| `include_costs` | bool | true | `YOLO_AUDIT__INCLUDE_COSTS` | Track API costs |

```yaml
audit:
  enabled: true
  log_level: info
  export_format: json
  retention_days: 30
  include_tokens: true
  include_costs: true
```

---

### mcp

MCP server configuration.

| Option | Type | Default | Env Var | Description |
|:-------|:-----|:--------|:--------|:------------|
| `transport` | string | stdio | `YOLO_MCP__TRANSPORT` | Transport type (stdio, http) |
| `http.port` | int | 8080 | `YOLO_MCP__HTTP__PORT` | HTTP port |
| `http.host` | string | 127.0.0.1 | `YOLO_MCP__HTTP__HOST` | HTTP host |
| `mask_errors` | bool | true | `YOLO_MCP__MASK_ERRORS` | Hide internal errors |

```yaml
mcp:
  transport: stdio
  http:
    port: 8080
    host: "127.0.0.1"
  mask_errors: true
```

---

## Environment Variables

All configuration can be overridden via environment variables using the `YOLO_` prefix with `__` as the nested delimiter.

### Syntax

```
YOLO_<SECTION>__<KEY>=<VALUE>
```

### Examples

```bash
# Project name
export YOLO_PROJECT_NAME=my-api

# LLM settings
export YOLO_LLM__PROVIDER=hybrid
export YOLO_LLM__CHEAP_MODEL=gpt-4o-mini
export YOLO_LLM__PREMIUM_MODEL=claude-sonnet-4-20250514
export YOLO_LLM__BEST_MODEL=claude-opus-4-5-20251101
export YOLO_LLM__OPENAI__API_KEY=sk-proj-...
export YOLO_LLM__HYBRID__ENABLED=true

# Quality thresholds
export YOLO_QUALITY__TEST_COVERAGE_THRESHOLD=0.9
export YOLO_QUALITY__GATE_PASS_THRESHOLD=0.8

# Memory
export YOLO_MEMORY__PERSIST_PATH=/custom/path

# Agent settings
export YOLO_AGENTS__MAX_ITERATIONS=15
export YOLO_AGENTS__TIMEOUT_SECONDS=600

# Orchestrator
export YOLO_ORCHESTRATOR__HUMAN_ESCALATION_ENABLED=false

# MCP
export YOLO_MCP__TRANSPORT=http
export YOLO_MCP__HTTP__PORT=9000

# Brownfield scanning
export YOLO_BROWNFIELD__SCAN_DEPTH=4
export YOLO_BROWNFIELD__INTERACTIVE=false

# GitHub automation
export YOLO_GITHUB__TOKEN=ghp_...
export YOLO_GITHUB__REPOSITORY=bbengt1/yolo-developer
```

### List Values

For list configuration (like `blocking_gates`), use comma-separated values:

```bash
export YOLO_QUALITY__BLOCKING_GATES=testability,architecture_validation,definition_of_done
```

---

## Configuration Priority

When the same setting is defined in multiple places, the priority is:

1. **Environment variables** (highest)
2. **yolo.yaml**
3. **Built-in defaults** (lowest)

### Example

```yaml
# yolo.yaml
quality:
  test_coverage_threshold: 0.8
```

```bash
# Environment
export YOLO_QUALITY__TEST_COVERAGE_THRESHOLD=0.9
```

**Result:** `test_coverage_threshold = 0.9` (environment wins)

---

## Configuration Validation

### Validate Configuration

```bash
yolo config validate
```

**Output:**
```
Validating yolo.yaml...
  ✓ Schema valid
  ✓ API keys configured
  ✓ Memory directory writable
  ✓ Models available
  ✓ All settings valid

Configuration is valid.
```

### Common Validation Errors

**Invalid threshold:**
```
Error: quality.test_coverage_threshold must be between 0.0 and 1.0
  Current value: 1.5
  Valid range: 0.0 - 1.0
```

**Missing API key:**
```
Error: No LLM API key configured
  Set one of:
    - YOLO_LLM__OPENAI__API_KEY
    - YOLO_LLM__ANTHROPIC_API_KEY
```

**Invalid model:**
```
Error: Unknown model: gpt-5-turbo
  Valid OpenAI models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
  Valid Anthropic models: claude-3-opus-*, claude-3-sonnet-*, claude-3-haiku-*
```

---

## Configuration Profiles

### Development Profile

```yaml
# yolo.dev.yaml
project_name: my-api-dev

llm:
  cheap_model: gpt-4o-mini  # Cheaper for dev
  premium_model: gpt-4o-mini
  best_model: gpt-4o

quality:
  test_coverage_threshold: 0.6  # Relaxed for iteration
  gate_pass_threshold: 0.5

agents:
  max_iterations: 5  # Faster iterations
  timeout_seconds: 120

audit:
  log_level: debug  # Verbose logging
```

### Production Profile

```yaml
# yolo.prod.yaml
project_name: my-api-prod

llm:
  cheap_model: gpt-4o-mini
  premium_model: claude-sonnet-4-20250514
  best_model: claude-opus-4-5-20251101

quality:
  test_coverage_threshold: 0.95  # Strict
  gate_pass_threshold: 0.9

agents:
  max_iterations: 15
  timeout_seconds: 600

audit:
  log_level: info
  retention_days: 90  # Longer retention
```

### Using Profiles

```bash
# Use specific config file
yolo --config yolo.dev.yaml run

# Or via environment
export YOLO_CONFIG_PATH=yolo.prod.yaml
yolo run
```

---

## Complete Example

```yaml
# yolo.yaml - Complete configuration example

# Project identification
project_name: user-management-api

# LLM configuration
llm:
  provider: auto
  cheap_model: gpt-4o-mini
  premium_model: claude-sonnet-4-20250514
  best_model: claude-opus-4-5-20251101
  # API keys via environment variables
  openai:
    cheap_model: gpt-4o-mini
    premium_model: gpt-4o
    code_model: gpt-4o
    reasoning_model: null
  hybrid:
    enabled: false
    routing:
      code_generation: openai
      code_review: openai
      architecture: anthropic
      analysis: anthropic
      documentation: openai
      testing: openai

# Quality gates
quality:
  test_coverage_threshold: 0.85
  gate_pass_threshold: 0.75
  blocking_gates:
    - testability
    - ac_measurability
    - architecture_validation
    - definition_of_done
  warning_gates:
    - code_complexity

# Memory configuration
memory:
  persist_path: .yolo/memory
  vector_store_type: chromadb
  graph_store_type: json

# Agent configuration
agents:
  max_iterations: 10
  timeout_seconds: 300
  enable_caching: true
  cache_ttl_hours: 24
  parallel_stories: 1

  # Per-agent overrides
  dev:
    max_iterations: 15
    timeout_seconds: 600

  tea:
    timeout_seconds: 180

# Orchestration
orchestrator:
  max_agent_retries: 3
  circular_detection_threshold: 5
  human_escalation_enabled: true
  checkpoint_enabled: true
  checkpoint_interval: 5

# Audit and logging
audit:
  enabled: true
  log_level: info
  export_format: json
  retention_days: 30
  include_tokens: true
  include_costs: true

# MCP server
mcp:
  transport: stdio
  mask_errors: true
```

---

## Next Steps

- [CLI Reference](/yolo-developer/cli/) - Command-line usage
- [Architecture](/yolo-developer/architecture/) - System design
- [MCP Integration](/yolo-developer/mcp/) - External tool integration
