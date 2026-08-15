# Model Recovery Guide

## Network Failures

- Downloads retry up to 3 times with exponential backoff
- Partial files use `.partial` suffix and resume on retry
- Run `sentivis-models download` to retry manually

## Disk Full

- Check free space in the Model Setup dialog
- Clear partial downloads: `sentivis-models cache` then remove `models/*.partial`

## Checksum Mismatch

- Invalid YOLO weights are deleted automatically
- Re-run download: `sentivis-models download`

## Hugging Face Authentication

```bash
set HF_TOKEN=your_token_here
sentivis-models download
```

Or enter the token in the first-launch dialog (stored securely in user config).

## Repair / Reinstall

```python
# Via service API after startup
context.model_management.repair_model(ModelKind.YOLO)
context.model_management.uninstall_model(ModelKind.YOLO)
```

## Ollama (Optional Gemma Path)

If using Ollama for Gemma: install from https://ollama.com/download, then:

```bash
ollama pull gemma2:2b
```

## Offline Errors

When offline and models are missing, the application lists unavailable models and disables analysis until installation completes online.
