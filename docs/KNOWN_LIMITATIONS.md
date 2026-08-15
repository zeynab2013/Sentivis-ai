# Known Limitations

**Version:** 1.0

- Gemma-2B requires Hugging Face authentication for some environments
- INT4 quantization on Windows depends on bitsandbytes availability
- Video input not supported in v1
- Object tracking is pass-through (NoOpTracker)
- Single concurrent pipeline execution
- Time of day and weather inference not implemented (returns "unknown")

See `docs/adr/` for architectural trade-offs.
