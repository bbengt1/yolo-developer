---
layout: default
title: Web Dashboard
nav_order: 12
parent: null
---

# Web Dashboard
{: .no_toc }

Start the local web UI for sprint monitoring and task seeding.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Start the Web UI

```bash
yolo web start
```

The dashboard is available at `http://127.0.0.1:3000`.

![Web Dashboard Preview](/yolo-developer/assets/images/dashboard.svg)

---

## Run API-Only

```bash
yolo web start --api-only
```

---

## Upload Requirements

Use the API upload endpoint for requirements documents:

```bash
curl -F "file=@requirements.md" http://127.0.0.1:3000/api/v1/uploads
```
