# Known AI Limitations

**Part 3 complete — documented constraints for competition and production use**

---

## Hardware

- **2 GB VRAM:** Gemma 2B with INT4 quantization is required for GPU inference; BLIP and YOLO must not overlap on GPU.
- **CPU fallback:** Supported but significantly slower; intended for recovery, not primary use.
- **No multi-GPU:** Single device selection only.

---

## Model Behavior

- **Gemma determinism:** Competition mode sets seed and temperature 0, but full bitwise reproducibility is not guaranteed across CUDA/driver versions.
- **BLIP observations:** Visual descriptions may omit fine details not salient to the base model.
- **YOLO coverage:** Limited to COCO-trained classes; novel objects are not detected.
- **Activity inference:** Heuristic and evidence-based; not action recognition from video.

---

## Quality Assurance

- **QA heuristics:** Token and coverage checks are not semantic entailment; edge-case captions may pass or fail incorrectly.
- **Strict mode:** Competition QA may reject valid creative phrasing that lacks exact graph tokens.
- **Fallback captions:** Recovery captions are conservative and may be less descriptive.

---

## Performance

- **Cold start:** First model load includes Hugging Face / Ultralytics download and initialization latency.
- **Stub benchmarks:** Automated tests use stub models; real-model timings require on-device benchmarking.
- **VRAM threshold:** Release verification uses allocated-bytes threshold, not driver-level guarantees.

---

## Environment

- **Python 3.10.11 is the official target per `pyproject.toml`.
- **Windows primary:** Linux/macOS may work but are not the primary validation target.

---

## UI (Part 4 scope)

- Metrics and competition mode are wired in the pipeline but not yet exposed in the desktop UI.
- Benchmark runner is an internal facility; no menu entry until Part 4.
