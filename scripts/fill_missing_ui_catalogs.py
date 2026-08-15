"""Fill missing fa/es/zh UI catalog keys via Ollama gemma3:4b (batched)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from analysis.activity.ollama_client import OllamaClient

ROOT = Path(__file__).resolve().parents[1]
LANG_NAMES = {"fa": "Persian", "es": "Spanish", "zh": "Simplified Chinese"}


def _batch_translate(client: OllamaClient, lang: str, items: dict[str, str]) -> dict[str, str]:
    payload = json.dumps(items, ensure_ascii=False)
    system = (
        f"You translate UI strings into {LANG_NAMES[lang]}. "
        "Return ONLY a JSON object mapping the same keys to translated values. "
        "Keep placeholders like {device} unchanged. Do not translate brand name Sentivis AI. "
        "Do not add commentary."
    )
    user = f"Translate these UI strings:\n{payload}"
    response = client.generate_text(system=system, user=user, max_tokens=1800, purpose="translation")
    text = response.text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object in response: {text[:200]}")
    parsed = json.loads(text[start : end + 1])
    out: dict[str, str] = {}
    for key, value in items.items():
        translated = str(parsed.get(key, "")).strip()
        out[key] = translated if translated else value
    return out


def fill_language(lang: str) -> dict[str, object]:
    en = json.loads((ROOT / "translations" / "en.json").read_text(encoding="utf-8"))
    path = ROOT / "translations" / f"{lang}.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    missing = {k: en[k] for k in en if k not in current or not str(current.get(k, "")).strip()}
    client = OllamaClient(model="gemma3:4b", keep_alive="30m", timeout_seconds=180.0)
    keys = sorted(missing)
    filled = 0
    batch_size = 18
    for i in range(0, len(keys), batch_size):
        chunk_keys = keys[i : i + batch_size]
        chunk = {k: missing[k] for k in chunk_keys}
        try:
            translated = _batch_translate(client, lang, chunk)
            current.update(translated)
            filled += len(translated)
        except Exception as exc:  # noqa: BLE001
            print(f"[{lang}] batch {i} failed: {exc}; keeping English fallbacks")
            current.update(chunk)
            filled += len(chunk)
        path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{lang}] progress {min(i + batch_size, len(keys))}/{len(keys)}")
        time.sleep(0.2)
    # Ensure exact key set matches English.
    ordered = {k: current.get(k, en[k]) for k in en}
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"lang": lang, "filled": filled, "total": len(en), "path": str(path)}


def main() -> None:
    reports = []
    for lang in ("fa", "es", "zh"):
        reports.append(fill_language(lang))
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
