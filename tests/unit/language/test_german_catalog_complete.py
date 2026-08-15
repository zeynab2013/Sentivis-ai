"""German UI catalog must remain complete (470 keys)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_german_catalog_has_all_english_keys() -> None:
    en = json.loads((ROOT / "translations" / "en.json").read_text(encoding="utf-8"))
    de = json.loads((ROOT / "translations" / "de.json").read_text(encoding="utf-8"))
    assert isinstance(en, dict) and isinstance(de, dict)
    missing = sorted(set(en) - set(de))
    extra = sorted(set(de) - set(en))
    assert not missing, f"Missing German keys: {missing[:20]}"
    assert not extra, f"Extra German keys: {extra[:20]}"
    assert len(en) == len(de)
    # Must remain complete versus English (grows when new UI keys are added).
    assert len(de) >= 470
