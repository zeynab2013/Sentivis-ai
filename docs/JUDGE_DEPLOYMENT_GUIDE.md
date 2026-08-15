# Judge Deployment Guide

## Requirements

- Windows 10 or 11
- Python 3.10.11
- Copy the entire project folder to any drive or user profile

## One-command launch

```bash
pip install -e .
sentivis-ai
```

Alternative:

```bash
streamlit run streamlit_app/main.py
```

## First launch

The application automatically creates runtime folders and runs a startup self-test.
Open the sidebar **Startup diagnostics** panel for **System Ready** or **System Not Ready**.

## Models

Place weights in the portable `models/` directory. YOLO, BLIP, Gemma, SAM2, and RealESRGAN
are discovered automatically from `models/` and configured search paths.

## Offline use

The UI starts without internet. Features that require downloads are disabled with warnings.

## Measured status on validation machine

- System Ready: Core requirements satisfied. Review optional notes below.
- Pytest: PASS
