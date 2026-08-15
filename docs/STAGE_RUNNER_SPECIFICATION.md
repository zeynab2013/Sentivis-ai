# SENTIVIS AI — StageRunner Specification

**Version:** 2.1  
**Part:** 2 / 4 (section 2)

---

## 1. Separation of Concerns

| Component | Role |
|-----------|------|
| `PipelineOrchestrator` | Coordinates stage order, DTO threading, recovery policy |
| `StageRunner` | Executes one stage lifecycle |

Orchestrator **never** calls feature implementations directly. It delegates each stage to `StageRunner.run(stage, callable, context)`.

---

## 2. Interface — `IStageRunner`

```python
class StageContext(TypedDict):
    run_id: str
    stage: PipelineStage
    device: str

class IStageRunner(Protocol):
    def run(
        self,
        stage: PipelineStage,
        action: Callable[[], TOut],
        *,
        recoverable: bool = False,
        fallback: Callable[[], TOut] | None = None,
    ) -> TOut: ...
```

---

## 3. Responsibilities

| Responsibility | Detail |
|----------------|--------|
| Stage lifecycle | Enter → execute → exit with cleanup |
| Timing | Record `duration_ms`; emit metric |
| Progress reporting | Notify `IProgressReporter` at start/end |
| Cancellation | Check `ICancellationToken` before and after |
| Retry policy | One GPU retry → CPU fallback for model stages |
| Failure recovery | Invoke `fallback` when `recoverable=True` |
| Metrics collection | Stage timing, memory delta |
| Logging | INFO start/end; ERROR on failure; DEBUG timing |
| Resource cleanup | `finally`: release stage temporaries via `IResourceScope` |

---

## 4. Execution Flow

```
StageRunner.run(stage, action)
  ├─ cancellation.raise_if_cancelled()
  ├─ progress.emit(stage, start_percent, "Starting…")
  ├─ memory.snapshot("before")
  ├─ log INFO "Stage {stage} started"
  ├─ try:
  │    result = action()
  ├─ except SentivisError as e:
  │    if recoverable and fallback: return fallback()
  │    raise
  ├─ except Exception as e:
  │    wrap as OrchestrationError
  ├─ finally:
  │    resource_scope.dispose_all()
  │    memory.snapshot("after")
  ├─ progress.emit(stage, end_percent, "Complete")
  └─ log INFO "Stage {stage} completed in {ms}ms"
```

---

## 5. Model Stages

For stages requiring `IModelEngine`, action closure is provided by orchestrator but **model acquire/release** occurs inside the closure via `IModelManager` — StageRunner ensures `finally` cleanup if action raises.

---

## 6. Retry Policy

| Condition | Action |
|-----------|--------|
| GPU OOM | Log WARNING → `IModelManager.handle_oom()` → retry once |
| Recoverable inference error | Execute `fallback` callable |
| Non-recoverable | Propagate to orchestrator |
| Cancel requested | Raise `CancelledError`; no retry |

---

## 7. Performance Expectations

| Metric | Target |
|--------|--------|
| StageRunner overhead | < 5 ms excluding action |
| Progress emit latency | < 10 ms to UI signal |

---

## 8. Testing Strategy

- Unit: mock action raising each exception type
- Unit: verify fallback invoked when recoverable
- Unit: verify cancellation before action
- Integration: full pipeline with fake stage actions

---

## 9. Acceptance Criteria

- [ ] Orchestrator contains zero try/except per stage (delegated to StageRunner)
- [ ] All stages logged with duration
- [ ] Cancellation honoured between stages
- [ ] Resource scope disposed in `finally`

---

*Architecture Phase — implementation pending*
