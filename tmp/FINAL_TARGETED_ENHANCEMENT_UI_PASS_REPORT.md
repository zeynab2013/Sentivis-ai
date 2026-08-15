# FINAL TARGETED PASS — Enhancement + Streamlit UI

**Date:** 2026-08-13  
**Scope only:** (A) image enhancement execution/validation honesty (B) Streamlit visual polish  
**Frozen:** detection, counts, relationships, activities, color, environment, caption, QA, OCR, translation, TTS, exports, metric *calculations*

---

## 1. Enhancement root cause

Two compounding failures caused LOW/MEDIUM images to report “Not applied” even when classical enhancement would help:

1. **SR all-or-nothing discard:** When super-resolution was enabled (default / UI), sharpen was deferred. SR often failed (missing weights / gate). The pipeline then discarded classical pre-SR work and kept the raw original → honest but ineffective.
2. **Regressive JPEG cleanup:** `reduce_jpeg_artifacts` could tank estimated quality (e.g. −0.135) on soft/high-artifact-score images. That polluted the candidate so the multi-signal gate rejected *everything*, including good contrast/CLAHE work already done.

Secondary honesty issue: disabled enhancement on non-HIGH images was labeled `ENHANCEMENT_NOT_REQUIRED` (“Already clear”).

---

## 2. Enhancement fix

- Snapshot classical `pre_sr_*`; if SR fails or SR gate fails, restore classical + deferred sharpen and re-gate (do not wipe to original unless classical also fails).
- Per-op quality gate (`_accept_op`) for white-balance, CLAHE, gamma, denoise, deblur, JPEG, sharpen — skip ops that regress quality.
- Gate: oversharpen reject only when noise rises *and* quality gain is weak (`q_delta < 0.03`).
- `ImageEnhancer(enabled=False)`: HIGH → `NOT_REQUIRED`; otherwise `FAILED` + “disabled in settings” (UI no longer says “Already clear”).
- Cache key bumped to `enhance_v5`.

---

## 3. How enhancement was validated

Synthetic behavioral checks:

| Case | Result |
|------|--------|
| HIGH checker | `NOT_REQUIRED`, original preserved |
| Soft photo MED/LOW | `APPLIED`, quality ↑ (~+0.06) |
| Blurry LOW | `APPLIED` (clarity) |
| Soft checker + jpeg trap | `APPLIED` after jpeg skip (contrast/CLAHE kept) |
| Disabled enhancer | original kept; reason contains “disabled” |

Unit: `tests/unit/vision/test_enhancement_targeted_pass.py` + existing enhancement suites — **37 passed** in focused enhancement run.

---

## 4. UI improvements (presentation only)

- Refined `theme.py`: quieter glow, coherent purple/pink tokens, section headings, VA suggestion buttons, export panel, expander cards.
- Caption panel typography/spacing; technical details in glass cards.
- Vision Assistant turns / suggested questions styling.
- Export section heading + format strip.
- Enhancement status mapping: disabled ≠ “Already clear”.

No Settings page; no metric/QA/export logic changes.

---

## 5. Tests passed

- Full `tests/unit`: **630 passed**
- Streamlit import / bootstrap: OK
- Streamlit `streamlit run … --server.port 8765`: started (“You can now view…”) then stopped

Real horse/soccer/moto/bike full pipeline freeze matrix: **Not verified** in this pass (out of scope; frozen AI not re-run end-to-end).

---

## 6. Remaining limitations

- True Real-ESRGAN still requires local model weights; without them SR fails safely and classical path is used when it verifies.
- Some pathological synthetics may still reject enhancement honestly when no op improves multi-signal metrics.
- Competition demo visual QA on a real laptop resolution was not pixel-audited in a browser: **Not verified** beyond app start + CSS unit tokens.
