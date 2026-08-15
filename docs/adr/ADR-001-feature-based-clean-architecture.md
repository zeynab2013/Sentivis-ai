# ADR-001: Feature-Based Clean Architecture

**Status:** Accepted  
**Date:** 2026-07-30  
**Version:** 1.0

## Decision

Organize Sentivis AI as **Feature-Based Clean Architecture** with layers: `ui` → `services` → `{vision, language, analysis}` → `core`. Cross-feature communication uses immutable DTOs in `core/contracts/` passed by `PipelineOrchestrator`.

## Context

Sentivis AI combines computer vision, NLP, scene analysis, and desktop UI. Requirements demand maintainability, independent module evolution, testability, and clean separation of business logic from UI. Multiple AI models must be swappable without rewriting the pipeline.

## Alternatives

1. **Monolithic package** — all code in flat modules.
2. **Layer-only Clean Architecture** — domain/application/infrastructure without feature boundaries.
3. **Microkernel** — plugins for every stage.

## Advantages

- Clear ownership per feature folder.
- Acyclic imports enforceable by structure.
- New models added via interface + registry.
- UI never touches torch or model code.
- Aligns with Master Prompt Book Part 2.

## Disadvantages

- More folders and boilerplate than a monolith.
- DTO definitions require upfront design.
- Orchestrator becomes a central coordination point.

## Justification

A monolith would violate single-responsibility and prevent independent module evolution required for competition judging and future model additions. Feature boundaries map directly to the canonical pipeline stages while keeping dependencies acyclic.
