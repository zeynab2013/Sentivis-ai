"""Fill German catalog strings via local Ollama gemma3:4b (text-only)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from analysis.activity.ollama_client import OllamaClient

ROOT = Path(__file__).resolve().parents[1]
EN_PATH = ROOT / "translations" / "en.json"
DE_PATH = ROOT / "translations" / "de.json"


def main() -> None:
    en = json.loads(EN_PATH.read_text(encoding="utf-8"))
    de = json.loads(DE_PATH.read_text(encoding="utf-8")) if DE_PATH.is_file() else {}
    client = OllamaClient(model="gemma3:4b", timeout_seconds=120.0)
    pending = [k for k, v in en.items() if de.get(k, v) == v]
    print(f"Pending German translations: {len(pending)} / {len(en)}")
    batch_size = 20
    for i in range(0, len(pending), batch_size):
        chunk_keys = pending[i : i + batch_size]
        payload = {k: en[k] for k in chunk_keys}
        system = (
            "Translate UI strings from English to German. "
            "Return ONLY valid JSON object mapping the same keys to German values. "
            "Keep placeholders like {device}, {name}, {path} unchanged."
        )
        user = json.dumps(payload, ensure_ascii=False)
        try:
            response = client.generate_text(system=system, user=user, max_tokens=1800)
            text = response.text.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                print(f"batch {i}: no JSON")
                continue
            parsed = json.loads(text[start : end + 1])
            if not isinstance(parsed, dict):
                continue
            for key in chunk_keys:
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    de[key] = value.strip()
            DE_PATH.write_text(json.dumps({**en, **de}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"batch {i}-{i+len(chunk_keys)} ok")
        except Exception as exc:  # noqa: BLE001
            print(f"batch {i} failed: {exc}")
        time.sleep(0.2)
    # Ensure every English key exists.
    merged = {**en, **de}
    DE_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    translated = sum(1 for k, v in en.items() if merged.get(k, v) != v)
    print(f"Done. Distinct German strings: {translated}/{len(en)}")


if __name__ == "__main__":
    main()
