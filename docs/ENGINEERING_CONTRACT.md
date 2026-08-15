# SENTIVIS AI — Engineering Contract

**Version:** 1.2  
**Part:** 1 / 4 — Project Foundation (§1 + §2 + §3)  
**Status:** Active — binding for all engineering work  
**Project:** Sentivis AI · *SEE. UNDERSTAND. INSPIRE.*

---

## 1. Identity

Sentivis AI is a **complete visual understanding platform**, not an object detector and not a simple caption generator. The system analyzes an image through multiple understanding stages before producing natural language output.

### Required Understanding Dimensions

The software must fully process each dimension before final caption generation:

| Dimension | Pipeline Responsibility |
|-----------|------------------------|
| Objects | YOLO detection → `DetectionResult` |
| Object Attributes | `AttributeExtractor` → `AttributeSet` |
| Object Relationships | `RelationshipAnalyzer` → `Relation[]` |
| Spatial Layout | `SceneGraphBuilder` + spatial metadata in graph nodes |
| Human Activities | `ActivityAnalyzer` → `ActivityHints` |
| Environment | `ContextBuilder` (scene type, setting, conditions) |
| Scene Context | `ContextBuilder` → `SceneContext` |
| Visual Semantics | BLIP vision-language understanding |
| Image Intent | Gemma reasoning over structured context + prompt |
| Possible Story | `CaptionRefiner` + Gemma narrative synthesis |

Final caption generation occurs **only after** stages 1–12 complete (see `docs/ARCHITECTURE.md` §6).

---

## 2. Primary Goal

Deliver a **commercial-grade desktop AI application** that is:

- Elegant
- Reliable
- Fast
- Maintainable
- Scalable
- Professional

---

## 3. Secondary Goals

| Area | Standard |
|------|----------|
| Architecture | Feature-Based Clean Architecture; acyclic dependencies |
| Code Quality | SOLID, DRY, KISS, YAGNI, PEP 8, full type hints |
| User Experience | Premium, responsive, fault-tolerant |
| Documentation | Architecture, developer guide, user guide |
| Performance | Responsive UI; background workers; 2 GB VRAM sequential model policy |
| Maintainability | Single responsibility per class; injectable dependencies |
| Testing | Unit, integration, performance coverage |
| Error Handling | Recoverable degradation; never crash the application |
| Logging | Structured, multi-level, auditable |

---

## 4. Development Philosophy

```
Understand → Design → Review → Implement → Validate → Refactor → Repeat
```

### Absolute Rules

1. **Never write code before designing.**
2. **Never design before understanding requirements.**
3. **Never implement before architecture has been validated.**
4. Accumulate every Master Prompt Book part; never overwrite prior rules.
5. On conflict, prefer the rule that improves long-term software quality.

---

## 5. Project Classification

This is **production software**. It is not:

- A prototype
- A university assignment
- A hackathon demo
- An MVP
- A proof of concept
- A quick script
- A toy application

---

## 6. Engineering Principles (Mandatory)

- Clean Architecture
- SOLID
- DRY
- KISS
- YAGNI
- Dependency Injection
- Composition over Inheritance
- High Cohesion
- Loose Coupling
- Meaningful Naming
- PEP 8
- Type Hinting
- Documentation on every public method
- Modularity

---

## 7. Quality Standard

| Unit | Requirement |
|------|-------------|
| File | One clear purpose |
| Folder | One clear responsibility |
| Class | Exactly one responsibility |
| Function | Solves one problem |
| Dependency | Intentional and injectable |
| Engineering decision | Explainable and documented |

---

## 8. Prohibited Artifacts

The following must **never** appear in the codebase:

- Temporary implementations
- Placeholder functions
- Fake implementations
- `TODO` comments
- Mock production logic
- Experimental shortcuts
- Duplicate code
- Dead code
- Unused imports or variables
- Empty classes or methods
- Magic numbers
- Hardcoded paths
- Hardcoded configuration

Configuration belongs in `config/` and user app-data overrides. Paths resolve through `core.utils.paths`.

---

## 9. Required Code Characteristics

Readable · Professional · Self-documenting · Maintainable · Testable · Reusable · Extensible · Consistent · Predictable

---

## 10. Engineering Team Roles

Every decision represents the combined judgment of:

- Principal Software Architect
- Principal AI Research Scientist
- Senior Computer Vision Engineer
- Senior Machine Learning Engineer
- Senior Python Engineer
- Desktop Application Engineer
- UI/UX Designer
- QA Engineer
- Performance Engineer
- Security Engineer
- DevOps Engineer
- Technical Writer

---

## 11. Engineering Mindset

| Prefer | Over |
|--------|------|
| Quality | Speed |
| Architecture | Shortcuts |
| Maintainability | Clever code |
| Readability | Compactness |
| Robustness | Unnecessary complexity |

Never optimize prematurely. Never sacrifice long-term quality for short-term convenience.

---

## 12. Success Definition

The project succeeds only when:

- [ ] The software is stable under normal and degraded operation
- [ ] The architecture remains clean with no circular dependencies
- [ ] The user experience feels premium
- [ ] A senior engineer can navigate the codebase without guidance
- [ ] Every module can evolve independently
- [ ] New AI models integrate without rewriting existing modules
- [ ] The application can be demonstrated confidently before international judges

---

## 13. Target Platform (Part 1 §2)

| Resource | Constraint |
|----------|------------|
| OS | Windows 11 |
| Python | 3.10.11 |
| GPU | NVIDIA · **2 GB VRAM** |
| RAM | 8 GB min · 16 GB recommended |
| CPU | Mid-range Intel/AMD |
| Storage | SSD preferred |

Full policy: `docs/HARDWARE_PERFORMANCE_POLICY.md`

---

## 14. Quality Gates (Part 1 §2 + §3)

A module is not complete until it passes:

- Responsibility and public API documentation (class + function docstrings per `DEVELOPER_GUIDE.md` §4)
- Architecture compliance (no forbidden imports)
- Self-review checklist (`DEVELOPER_GUIDE.md` §9)
- `ruff format --check` · `ruff check` · `mypy --strict`
- Unit tests · integration with dependents
- Definition of Done (`DEVELOPER_GUIDE.md` §11)

Failed gate → fix before continuing downstream work.

---

## 15. Coding Standards (Part 1 §3)

Full specification: `docs/DEVELOPER_GUIDE.md` v1.0

Summary: Python 3.10.11 · PEP 8 · type hints everywhere · frozen dataclass DTOs · no hardcoded config · no bare except · DI for testability · QA workflow before each next module.

---

## 16. Governance (Part 1 §2)

- **ADRs** required for major decisions → `docs/adr/`
- **Document versioning** on every engineering artifact
- **Module documentation:** purpose, responsibilities, dependencies, interfaces, flow, extensions, limitations, future work
- **Pre-implementation reviews** → `docs/ENGINEERING_REVIEW.md`

---

## 17. Implementation Gate

Implementation begins **only when**:

1. All Master Prompt Book parts (1–4) received and integrated
2. `docs/ARCHITECTURE.md` validated (Part 2)
3. `docs/ENGINEERING_REVIEW.md` passed (design phase ✓)
4. `docs/DEVELOPER_GUIDE.md` published (Part 1 §3 ✓)

**Current gate status:** Parts 1.1 + 1.2 + 1.3 + Part 2 integrated. Design reviews passed. Awaiting Part 1 §4.

---

## 18. Related Documents

| Document | Version | Purpose |
|----------|---------|---------|
| `docs/MASTER_SPEC_INDEX.md` | — | Accumulated specification index |
| `docs/ARCHITECTURE.md` | 1.2 | Software architecture (Part 2) |
| `docs/HARDWARE_PERFORMANCE_POLICY.md` | 1.0 | Hardware, memory, threading (Part 1 §2) |
| `docs/DEVELOPER_GUIDE.md` | 1.0 | Coding standards, QA, DoD (Part 1 §3) |
| `docs/ENGINEERING_REVIEW.md` | 1.0 | Pre-implementation reviews |
| `docs/adr/` | — | Architecture Decision Records |
| `docs/USER_GUIDE.md` | — | End-user documentation (pending) |

---

*This contract remains active until the entire Sentivis AI project is completed.*
