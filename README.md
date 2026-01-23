# yolo-developer

Autonomous multi-agent AI development system using BMad Method.

## Installation

```bash
uv sync
```

## Usage

```bash
yolo
```

Starts an interactive chat session. For one-shot prompts:

```bash
yolo "Summarize the current sprint status"
```

### External Tool Integration

YOLO can delegate tasks to external CLI tools like Claude Code and Aider for enhanced capabilities:

```bash
# Check tool availability
yolo tools

# View detailed status as JSON
yolo tools status --json
```

Configure tools in `yolo.yaml`:

```yaml
tools:
  claude_code:
    enabled: true
    timeout: 300
  aider:
    enabled: false
```

Or via environment variables:

```bash
export YOLO_TOOLS__CLAUDE_CODE__ENABLED=true
```

### MCP Integration

Integrate with MCP clients (Claude Code, Cursor, VS Code):

```bash
yolo integrate claude-code
```

Full command list:

```bash
yolo --help
```

## Development

```bash
uv sync --all-extras
uv run pytest
```
