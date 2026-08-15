# Model Registry Reference

## ModelRecord Fields

| Field | Description |
|-------|-------------|
| `kind` | YOLO, BLIP, or GEMMA |
| `identifier` | Production model ID |
| `display_name` | User-facing name |
| `version` | Model version string |
| `provider` | Ultralytics / Hugging Face |
| `download_source` | ultralytics / huggingface / ollama |
| `local path` | `file_location` for YOLO weights |
| `expected_size_bytes` | Approximate download size |
| `checksum` | SHA-256 when validated locally |
| `license_name` | Model license |
| `installation_status` | not_installed / installed / validated / corrupted |
| `runtime_status` | missing / ready / validation_failed / … |
| `quantization` | e.g. int4 for Gemma |
| `mandatory` | Required before analysis |

## Installation Status Values

- `not_installed` — model not present locally
- `downloading` — background download in progress
- `installed` — files present, pending validation
- `validated` — passed post-download checks
- `corrupted` — failed validation, files removed

## API

- `CentralModelRegistry` — core registry (services/runtime)
- `ModelManagementService` — download, validation, cache orchestration
- `RegistryEnricher` — applies production catalog metadata
