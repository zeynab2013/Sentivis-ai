# Sentivis AI — End-to-End Acceptance Test Checklist

**Version:** 1.0.0  
**Purpose:** Manual verification that Sentivis AI works as a desktop application  
**Automated suite:** `python -m acceptance` or `pytest tests/acceptance -v`

---

## Pre-Test Setup

- [ ] Windows 11 (64-bit) with Python 3.10.11 installed
- [ ] Virtual environment created and activated
- [ ] `pip install -e ".[dev]"` completed without errors
- [ ] GPU drivers installed (optional; CPU fallback acceptable)
- [ ] Network available for first-time Hugging Face model download
- [ ] Clean `logs/`, `cache/`, and `exports/` directories (or note existing state)

---

## 1. Application Launch

- [ ] Run `sentivis-ai` — application window opens without crash
- [ ] Window title shows **Sentivis AI — SEE. UNDERSTAND. INSPIRE.**
- [ ] Sidebar visible with brand, session info, navigation buttons
- [ ] Image viewer shows placeholder: *Drop an image here or use Open Image*
- [ ] Dashboard shows stage progress, results panel, export panel
- [ ] Status badge shows **Ready** or equivalent idle state
- [ ] Press **F1** — About dialog opens with version, build, architecture info
- [ ] Close About dialog — returns to main window

---

## 2. Environment & Diagnostics

- [ ] Check `logs/startup-diagnostics.txt` exists after launch
- [ ] Diagnostics report Python version, OS, GPU/CUDA status
- [ ] Diagnostics lists 8 startup stages completed
- [ ] Diagnostics lists 3 registered plugins (YOLO, BLIP, Gemma)
- [ ] Open **Settings → Diagnostics** tab — fields populated
- [ ] No fatal errors in `logs/error.log`

---

## 3. Image Loading

- [ ] Click **Open Image** — file dialog opens
- [ ] Select a valid PNG/JPG image — image displays in viewer
- [ ] Notification shows *Loaded {filename}*
- [ ] Session panel updates with image name
- [ ] **Run Analysis** button becomes enabled
- [ ] Drag-and-drop an image onto viewer — image loads
- [ ] **Ctrl+O** shortcut opens file dialog

---

## 4. Image Viewer Controls

- [ ] **Fit** button scales image to window
- [ ] **100%** button shows original size
- [ ] **Zoom +** increases magnification
- [ ] **Zoom −** decreases magnification
- [ ] **Ctrl+0** fits to window
- [ ] **Ctrl++** / **Ctrl+=** zooms in
- [ ] **Ctrl+-** zooms out
- [ ] Mouse wheel zoom works (if enabled)
- [ ] **Detections** checkbox toggles bounding box overlays (after analysis)
- [ ] **Relationships** checkbox toggles relationship lines (after analysis)

---

## 5. Pipeline Execution

- [ ] Click **Run Analysis** — progress bar advances
- [ ] Stage progress shows: Validation → Preprocessing → Detection → … → Export
- [ ] Status updates during analysis (device label visible in dev mode)
- [ ] Analysis completes without error dialog
- [ ] Success notification shows duration and filename
- [ ] **Ctrl+R** starts analysis when image loaded
- [ ] **Escape** cancels in-progress analysis (optional test)

---

## 6. Results Presentation

- [ ] **Final Caption** section shows generated caption text
- [ ] **Scene Summary** section populated
- [ ] **Detected Objects** lists detected items with attributes
- [ ] **Relationships** section shows spatial/social relationships
- [ ] **Activities** section populated (may be empty for simple scenes)
- [ ] **Environment** section shows scene context
- [ ] **Quality Report** shows scores and hallucination risk
- [ ] **Execution Metrics** shows timing, RAM, object counts
- [ ] **Copy All** copies results to clipboard
- [ ] **Expand All** / **Collapse All** work on sections
- [ ] **Ctrl+F** focuses results search field
- [ ] Search filter hides non-matching sections

---

## 7. Export Workflow

- [ ] Export panel enabled after successful analysis
- [ ] Destination preview shows expected filename
- [ ] **Export JSON** — file created in `exports/` directory
- [ ] **Export TXT** — readable plain-text report
- [ ] **Export Markdown** — formatted markdown report
- [ ] **Export PDF** — printable PDF opens without error
- [ ] Re-export prompts overwrite confirmation when file exists
- [ ] Success notification shows exported filename
- [ ] Exported JSON contains caption, objects, metrics fields
- [ ] Exported files contain the same caption as UI

---

## 8. Presentation Mode

- [ ] Click **Presentation** or press **F11** — sidebar hides
- [ ] Dashboard switches to presentation layout
- [ ] Dev details (stage labels, model hints) hidden
- [ ] Press **F11** again — normal layout restored
- [ ] Settings dialog hides Diagnostics/Advanced tabs in presentation mode

---

## 9. Settings

- [ ] **Ctrl+,** or **Settings** button opens settings dialog
- [ ] All tabs visible: General, AI Models, Performance, Appearance, Exports, Diagnostics, Advanced
- [ ] **Restore Defaults** resets displayed values
- [ ] Change theme (Dark/Light) and save — UI theme updates
- [ ] Cancel closes without applying changes

---

## 10. Session History

- [ ] After analysis, entry appears in **Recent Analyses** list
- [ ] Entry shows image name, caption preview, timestamp, duration
- [ ] Multiple analyses accumulate in session history

---

## 11. Stress & Edge Cases (Manual)

- [ ] Run 3+ sequential analyses on different images — no memory leak or crash
- [ ] Load large image (2048×1536) — completes or shows clear error
- [ ] Load invalid file (`.gif`, tiny image, corrupted PNG) — user-friendly error
- [ ] Launch with empty `models/` folder — app starts, warns about missing weights
- [ ] Launch without GPU — CPU fallback message in logs, analysis still runs
- [ ] Close application — no hang, models released, clean exit

---

## 12. Performance (Manual Observation)

- [ ] Startup completes within 2 minutes (first launch may be slower)
- [ ] Stub/test analysis completes within 60 seconds
- [ ] Real-model analysis completes within reasonable time for hardware
- [ ] RAM usage stable after multiple runs
- [ ] GPU memory released after analysis (check Task Manager / nvidia-smi)

---

## 13. Automated Test Execution

Run the full automated acceptance suite:

```bash
python -m acceptance
```

Or individual categories:

```bash
pytest tests/acceptance/test_smoke.py -v          # Smoke tests
pytest tests/acceptance/test_e2e_pipeline.py -v   # Headless E2E
pytest tests/acceptance/test_e2e_desktop.py -v    # Desktop E2E
pytest tests/acceptance/test_stress.py -v         # Stress tests
pytest tests/acceptance/test_performance.py -v    # Performance tests
pytest tests/acceptance/test_ui.py -v             # UI tests
```

- [ ] All automated tests pass
- [ ] `docs/TEST_REPORT.md` generated with PASSED status

---

## Sign-Off

| Field | Value |
|-------|-------|
| Tester | |
| Date | |
| Environment (OS / Python / GPU) | |
| Automated suite result | PASS / FAIL |
| Manual checklist result | PASS / FAIL |
| Notes | |
