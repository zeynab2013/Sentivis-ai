# Sentivis AI — Directory Structure

**Version:** 1.0.0

```
SENTIVIS AI/
├── app/                    # Application entry, bootstrap, DI container, startup
├── analysis/               # Scene analysis engines (frozen)
├── assets/                 # Bundled icons, samples, export templates
│   ├── icons/
│   ├── samples/
│   └── export_templates/
├── config/                 # Default TOML configuration
├── core/                   # Config, contracts, logging, utilities
├── docs/                   # User and engineering documentation
├── language/               # BLIP and Gemma engines (frozen)
├── logs/                   # Runtime logs (gitignored)
├── models/                 # Local model weights (optional)
├── release/                # Release engineering, build, validation
│   └── resources/          # Installer-ready license, manifest, config bundle
├── services/               # Pipeline, cache, runtime asset management
├── tests/                  # Unit and integration tests
├── ui/                     # Desktop presentation layer (frozen)
├── vision/                 # YOLO detection (frozen)
├── cache/                  # Pipeline cache (runtime, gitignored)
├── exports/                # User exports (runtime, gitignored)
├── dist/                   # Build output (generated, gitignored)
├── pyproject.toml          # Package metadata and tooling
├── LICENSE                 # Proprietary license
└── README.md               # Project overview
```

## User Data (Windows)

```
%APPDATA%/SentivisAI/config/   # User configuration overrides
```

## Build Output

```
dist/
├── development/
├── production/
├── portable/
└── release/
    └── sentivis-ai-{version}-build{n}/
        ├── app/ ... ui/
        ├── config/
        ├── assets/
        ├── docs/
        ├── installer/         # License, notices, manifest, default config
        └── build_manifest.json
```

## Key Files

| Path | Purpose |
|------|---------|
| `app/main.py` | Console entry point |
| `config/app.default.toml` | Application settings |
| `config/models.default.toml` | Model and plugin configuration |
| `logs/startup-diagnostics.json` | Startup health report |
