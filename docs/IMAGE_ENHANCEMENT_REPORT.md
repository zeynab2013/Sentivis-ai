# Sentivis AI — Image Enhancement Report

## Scope

Adaptive enhancement pipeline (`vision/enhancement/`) with quality estimation, CLAHE, gamma, denoise, sharpen, and super-resolution fallbacks (RealESRGAN → OpenCV DNN → bicubic).

## Measured Enhancement Benchmark

- Sample images evaluated: **10**
- Images enhanced adaptively: **1**
- Average quality before: **80.3%**
- Average quality after: **79.7%**
- Average improvement (enhanced subset): **0.0%**

## Per-Image Samples

| Image | Before | After | Improved | Operations |
|-------|--------|-------|----------|------------|
| `000000562581.jpg` | 70.9% | 64.5% | Yes | clahe, jpeg_artifact_removal |
| `000000559547.jpg` | 77.7% | 77.7% | No | none |
| `000000319935.jpg` | 80.9% | 80.9% | No | none |
| `000000014038.jpg` | 84.7% | 84.7% | No | none |
| `000000548780.jpg` | 85.1% | 85.1% | No | none |
| `000000120420.jpg` | 82.9% | 82.9% | No | none |
| `000000542625.jpg` | 80.2% | 80.2% | No | none |
| `000000480944.jpg` | 79.7% | 79.7% | No | none |
| `000000254814.jpg` | 77.9% | 77.9% | No | none |
| `000000568290.jpg` | 83.0% | 83.0% | No | none |

## Configuration

- Normal mode: adaptive enhancement when estimated quality is below threshold.
- Competition mode: always applies highest-quality enhancement path.
- Super resolution: optional; RealESRGAN when weights are present, otherwise OpenCV DNN or bicubic.
