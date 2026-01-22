---
layout: default
title: Installation
nav_order: 3
---

# Installation Guide
{: .no_toc }

Complete installation instructions for all platforms and environments.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|:----------|:------------|
| Python | 3.10 or higher |
| Memory | 4 GB RAM |
| Disk Space | 500 MB |
| OS | macOS 12+, Ubuntu 20.04+, Windows 10+ (WSL2) |

### Recommended Requirements

| Component | Requirement |
|:----------|:------------|
| Python | 3.12 |
| Memory | 8 GB RAM |
| Disk Space | 2 GB (for memory persistence) |
| OS | macOS 14+, Ubuntu 22.04+ |

---

## Installation Methods

### Method 1: Using uv (Recommended)

[uv](https://astral.sh/uv) is a fast Python package manager that handles dependencies efficiently.

#### Install uv

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Homebrew (macOS):**
```bash
brew install uv
```

#### Install YOLO Developer

```bash
# Clone repository
git clone https://github.com/bbengt1/yolo-developer.git
cd yolo-developer

# Install dependencies
uv sync

# Verify installation
uv run yolo --version
```

**Expected output:**
```
YOLO Developer v0.1.0
```

---

### Method 2: Using pip

If you prefer traditional pip installation:

```bash
# Clone repository
git clone https://github.com/bbengt1/yolo-developer.git
cd yolo-developer

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .

# Verify installation
yolo --version
```

---

### Method 3: Development Installation

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/bbengt1/yolo-developer.git
cd yolo-developer

# Install with all development dependencies
uv sync --all-extras

# Install pre-commit hooks (optional)
uv run pre-commit install

# Verify development tools
uv run pytest --version
uv run mypy --version
uv run ruff --version
```

---

### Method 4: Docker Installation

{: .note }
> Docker support is coming soon. This section will be updated when available.

```bash
# Pull the image
docker pull ghcr.io/bbengt1/yolo-developer:latest

# Run with mounted project
docker run -v $(pwd):/project -e YOLO_LLM__OPENAI__API_KEY=$OPENAI_API_KEY \
  ghcr.io/bbengt1/yolo-developer yolo init
```

---

## Platform-Specific Instructions

### macOS

#### Prerequisites

1. **Install Xcode Command Line Tools:**
   ```bash
   xcode-select --install
   ```

2. **Install Python 3.10+ (if not already installed):**
   ```bash
   # Using Homebrew
   brew install python@3.12

   # Verify
   python3 --version
   ```

3. **Install uv:**
   ```bash
   brew install uv
   ```

#### Installation

```bash
git clone https://github.com/bbengt1/yolo-developer.git
cd yolo-developer
uv sync
```

#### Shell Configuration

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
# YOLO Developer API Keys
export YOLO_LLM__OPENAI__API_KEY="sk-proj-..."
export YOLO_LLM__ANTHROPIC_API_KEY="sk-ant-..."

# Optional: Add yolo to PATH if using pip install
export PATH="$HOME/yolo-developer/.venv/bin:$PATH"
```

Reload shell:
```bash
source ~/.zshrc
```

---

### Linux (Ubuntu/Debian)

#### Prerequisites

1. **Update system packages:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install Python 3.10+:**
   ```bash
   sudo apt install python3.12 python3.12-venv python3-pip -y

   # Verify
   python3 --version
   ```

3. **Install uv:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc
   ```

#### Installation

```bash
git clone https://github.com/bbengt1/yolo-developer.git
cd yolo-developer
uv sync
```

#### Shell Configuration

Add to your `~/.bashrc`:

```bash
# YOLO Developer API Keys
export YOLO_LLM__OPENAI__API_KEY="sk-proj-..."
export YOLO_LLM__ANTHROPIC_API_KEY="sk-ant-..."
```

Reload:
```bash
source ~/.bashrc
```

---

### Windows (WSL2)

{: .important }
> YOLO Developer requires WSL2 on Windows. Native Windows support is not available.

#### Prerequisites

1. **Enable WSL2:**
   ```powershell
   # Run in PowerShell as Administrator
   wsl --install
   ```

2. **Install Ubuntu from Microsoft Store**

3. **Open Ubuntu terminal and update:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

4. **Install Python:**
   ```bash
   sudo apt install python3.12 python3.12-venv -y
   ```

5. **Install uv:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc
   ```

#### Installation

```bash
git clone https://github.com/bbengt1/yolo-developer.git
cd yolo-developer
uv sync
```

#### Accessing from Windows

Your WSL2 files are accessible at:
```
\\wsl$\Ubuntu\home\<username>\yolo-developer
```

---

## Post-Installation Setup

### 1. Configure API Keys

{: .warning }
> Never commit API keys to version control. Always use environment variables.

**Option A: Environment Variables (Recommended)**

```bash
export YOLO_LLM__OPENAI__API_KEY="sk-proj-..."
```

**Option B: .env File (Local Development)**

Create `.env` in your project root:
```bash
YOLO_LLM__OPENAI__API_KEY=sk-proj-...
YOLO_LLM__ANTHROPIC_API_KEY=sk-ant-...
```

{: .note }
> Add `.env` to your `.gitignore` file.

### 2. Verify Installation

Run the verification command:

```bash
uv run yolo --version
uv run yolo config show
```

**Expected output:**
```
YOLO Developer v0.1.0

project_name: null
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
```

### 3. Initialize Your First Project

```bash
cd /path/to/your/project
uv run yolo init --name my-project
```

---

## Upgrading

### Upgrade to Latest Version

```bash
cd yolo-developer
git pull origin main
uv sync
```

### Check for Updates

```bash
git fetch origin
git log HEAD..origin/main --oneline
```

---

## Uninstallation

### Remove YOLO Developer

```bash
# Remove the repository
rm -rf /path/to/yolo-developer

# Remove project data (in each project)
rm -rf .yolo/
rm yolo.yaml
```

### Remove uv (Optional)

```bash
rm -rf ~/.cargo/bin/uv
```

---

## Troubleshooting Installation

### Python Version Issues

**Problem:** `Python 3.10+ required`

**Solution:**
```bash
# Check installed versions
python3 --version
which python3.12

# Use specific version with uv
uv python install 3.12
uv sync
```

### Permission Denied

**Problem:** `Permission denied` when installing

**Solution:**
```bash
# Don't use sudo with uv
# Instead, ensure ~/.local/bin is in PATH
export PATH="$HOME/.local/bin:$PATH"
```

### ChromaDB Build Errors

**Problem:** `Failed to build chromadb`

**Solution (Ubuntu):**
```bash
sudo apt install build-essential python3-dev
uv sync
```

**Solution (macOS):**
```bash
xcode-select --install
uv sync
```

### SSL Certificate Errors

**Problem:** `SSL: CERTIFICATE_VERIFY_FAILED`

**Solution (macOS):**
```bash
# Run the certificate installer
/Applications/Python\ 3.12/Install\ Certificates.command
```

### uv Not Found

**Problem:** `command not found: uv`

**Solution:**
```bash
# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Or reinstall
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

---

## Next Steps

After successful installation:

1. **[Getting Started](/yolo-developer/getting-started)** - Quick start guide
2. **[Configuration](/yolo-developer/configuration/)** - Customize your setup
3. **[CLI Reference](/yolo-developer/cli/)** - Learn all commands
