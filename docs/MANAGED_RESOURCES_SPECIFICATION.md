# SENTIVIS AI — Managed Resources Specification

**Version:** 2.1  
**Part:** 2 / 4 (section 2)

---

## 1. Purpose

Uniform lifecycle for every heavy or temporary resource. Prevents leaks across pipeline stages and worker threads.

---

## 2. Interface — `IManagedResource`

```python
class IManagedResource(Protocol):
    def initialize(self) -> None: ...
    def acquire(self) -> None: ...
    def release(self) -> None: ...
    def dispose(self) -> None: ...

    @property
    def memory_budget_mb(self) -> float: ...

    @property
    def is_acquired(self) -> bool: ...
```

### Lifecycle States

```
UNINITIALIZED → INITIALIZED → ACQUIRED → IN_USE → RELEASED → DISPOSED
```

| Phase | Action |
|-------|--------|
| **Initialize** | Allocate handles, validate config |
| **Acquire** | Reserve memory / open handles |
| **Use** | Active operation |
| **Release** | Drop heavy refs; keep shell |
| **Dispose** | Final cleanup; idempotent |

---

## 3. Resource Categories

| Resource | Implementation | Manager |
|----------|----------------|---------|
| AI model weights | `IModelEngine` | `ModelManager` |
| Image tensor buffer | `ImageBufferResource` | `IResourceScope` |
| PIL/numpy buffer | `ImageBufferResource` | `IResourceScope` |
| Pipeline worker | `PipelineWorker` | Qt thread lifecycle |
| Export file stream | `ExportStreamResource` | `ExportService` |
| Cache entry | `CacheEntryResource` | `CacheManager` |

---

## 4. `IResourceScope`

```python
class IResourceScope(Protocol):
    def register(self, resource: IManagedResource) -> None: ...
    def dispose_all(self) -> None: ...
```

`StageRunner` creates one scope per stage. All stage temporaries register here. `dispose_all()` in `finally`.

---

## 5. `ManagedResourceManager` (services/memory/)

| Method | Purpose |
|--------|---------|
| `track(resource)` | Register for peak logging |
| `force_dispose_all()` | OOM recovery |
| `context()` | Context manager yielding `IResourceScope` |

Location: `services/memory/managed_resources.py`

---

## 6. Memory Expectations

| Resource | Budget | Release trigger |
|----------|--------|-----------------|
| Model engine | ≤ 1100 MB VRAM | After stage infer |
| Image buffer | ≤ 48 MB (4K RGB) | End of stage |
| Export stream | ≤ 10 MB | Close file in dispose |

---

## 7. Thread Safety

- Acquire/release on pipeline worker only for model resources
- Export streams on export worker thread
- `dispose()` must be safe to call from owning thread in `finally`

---

## 8. Acceptance Criteria

- [ ] No bare tensors survive stage boundary
- [ ] OOM recovery calls `force_dispose_all()`
- [ ] Peak memory logged per run includes managed resources

---

*Architecture Phase — implementation pending*
