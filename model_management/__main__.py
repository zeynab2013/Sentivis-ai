"""Model management CLI."""

from __future__ import annotations

import argparse

from app.container import DependencyContainer
from core.config.loader import load_analysis_config, load_app_config, load_model_config, load_theme_config
from model_management.service import ModelManagementService


def main() -> int:
    parser = argparse.ArgumentParser(description="Sentivis AI model management")
    parser.add_argument("command", choices=["status", "download", "validate", "cache"])
    args = parser.parse_args()

    context = DependencyContainer().build(
        load_app_config(),
        load_model_config(),
        load_theme_config(),
        load_analysis_config(),
    )
    service = ModelManagementService.create(
        context.model_registry,
        context.main_controller.app_config.paths.models_dir,
    )

    if args.command == "status":
        for record in service.records:
            print(
                f"{record.display_name}: {record.installation_status.value} "
                f"({record.runtime_status.value}) — {record.validation_detail}"
            )
        return 0

    if args.command == "download":
        service.download_all(on_progress=lambda progress: print(progress.message or progress.state.value))
        service.wait_for_downloads()
        return 0 if service.all_mandatory_ready() else 1

    if args.command == "validate":
        for record in service.validate_installed():
            print(f"{record.display_name}: {record.installation_status.value}")
        return 0

    report = service.cache.report()
    print(f"Models dir: {report.models_dir}")
    print(f"Model bytes: {report.models_bytes}")
    print(f"Partial bytes: {report.partial_bytes}")
    print(f"Files: {', '.join(report.model_files) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
