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

Integrate with external AI CLIs (Codex, Claude Code, Cursor, VS Code):

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
