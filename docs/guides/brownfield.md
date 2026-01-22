---
layout: default
title: Brownfield Guide
nav_order: 8
parent: null
---

# Brownfield Guide
{: .no_toc }

Add YOLO Developer to an existing repository by scanning for language, frameworks, and conventions.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## When to Use Brownfield Mode

Use brownfield mode when you are integrating YOLO Developer into a project with existing code, tests, and conventions.

## Run the Scan

```bash
cd /path/to/existing/project
uv run yolo init --brownfield
```

YOLO will generate `.yolo/project-context.yaml` with detected structure, frameworks, and conventions.

## Scan Without Changes

```bash
uv run yolo init --brownfield --scan-only
```

## Re-Scan Later

```bash
yolo scan --refresh
```

## Tips

- Use `--hint` to guide detection for atypical projects.
- Use `--non-interactive` for CI or scripted scans.
- Review `.yolo/project-context.yaml` and refine any conventions as needed.
