# Sentivis AI — User Guide

**Version:** 1.0

## Overview

Sentivis AI analyzes images through multiple understanding stages before generating a natural language caption describing scene meaning, context, and story.

## Quick Start

1. Launch Sentivis AI
2. Click **Open Image** and select a photo
3. Click **Analyze**
4. Review caption and scene context panels
5. Export results as JSON, TXT, or PDF

## Interface

| Area | Purpose |
|------|---------|
| Sidebar | Open, analyze, cancel, export |
| Image viewer | Selected image preview |
| Progress | Current pipeline stage |
| Caption | Final refined caption |
| Scene Context | Objects, environment, activities |
| History | Previous analyses in session |

## Tips

- Supported formats: PNG, JPEG, WEBP, BMP
- Maximum image size: 4096 px, 32 MB
- Analysis runs in background; UI stays responsive
- Use **Cancel** to stop before next stage completes

## Export Formats

| Format | Contents |
|--------|----------|
| JSON | Structured analysis data |
| TXT | Caption and summary text |
| PDF | Formatted report |

Exports save to the `exports/` directory by default.

## Errors

User-friendly messages appear in dialogs. The application continues running after errors. Retry with a smaller image if memory errors persist.
