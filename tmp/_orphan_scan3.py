#!/usr/bin/env python3
"""Fixed orphan scan: module match must not be a prefix of a longer dotted path."""
from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

root = Path(r"D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI")
focus = ["language", "analysis", "vision", "services", "core"]
exclude_candidates = {
    "models",
    "translations",
    "streamlit_app",
    "ui",
    "tests",
    "tmp",
    "logs",
    "site-packages",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
}
search_exclude = {
    "site-packages",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
    "tmp",
}

# After module name: end, or non-identifier non-dot (space, comma, etc.) — NOT another dotted segment
MOD_END = r"(?![.\w])"


def collect_py(for_candidates: bool) -> list[Path]:
    exclude_names = exclude_candidates if for_candidates else search_exclude
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames if d not in exclude_names and not d.startswith(".")
        ]
        if any(x in str(dp) for x in ("site-packages", "venv", ".venv")):
            dirnames[:] = []
            continue
        if for_candidates:
            try:
                parts = set(dp.relative_to(root).parts)
            except Exception:
                parts = set()
            if parts & {
                "models",
                "translations",
                "streamlit_app",
                "ui",
                "tests",
                "tmp",
                "logs",
            }:
                dirnames[:] = []
                continue
        for f in filenames:
            if f.endswith(".py"):
                out.append(dp / f)
    return out


search_py = collect_py(False)
candidates: list[Path] = []
for p in collect_py(True):
    rel = p.relative_to(root)
    if rel.parts[0] not in focus:
        continue
    if p.name == "__init__.py":
        continue
    if "translations" in rel.parts:
        continue
    candidates.append(p)

config_exts = {".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".md", ".bat", ".ps1", ".sh", ".txt"}
search_cfg: list[Path] = []
for dirpath, dirnames, filenames in os.walk(root):
    dp = Path(dirpath)
    dirnames[:] = [d for d in dirnames if d not in search_exclude and not d.startswith(".")]
    if any(x in str(dp) for x in ("site-packages", "venv", ".venv")):
        dirnames[:] = []
        continue
    for f in filenames:
        ext = Path(f).suffix.lower()
        if ext in config_exts or f in {
            "Makefile",
            "Dockerfile",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
        }:
            search_cfg.append(dp / f)

print(f"search_py={len(search_py)} candidates={len(candidates)} cfg={len(search_cfg)}")

reasons = Counter()
unused: list[tuple[str, str]] = []
# Also track modules only hit via their package __init__.py — still "used"
# but report modules with only __init__ hits separately for curiosity
init_only: list[tuple[str, str]] = []

for cand in sorted(candidates):
    rel = cand.relative_to(root)
    full_mod = ".".join(rel.with_suffix("").parts)
    stem = rel.stem
    parent = ".".join(rel.with_suffix("").parts[:-1])
    posix = rel.as_posix()

    hits: list[tuple[str, str]] = []

    for src in search_py:
        if src.resolve() == cand.resolve():
            continue
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        srel = src.relative_to(root).as_posix()

        if re.search(rf"(?:from|import)\s+{re.escape(full_mod)}{MOD_END}", text):
            hits.append(("import_full", srel))
            continue
        if parent and re.search(
            rf"from\s+{re.escape(parent)}{MOD_END}\s+import\s+[^\n]*\b{re.escape(stem)}\b",
            text,
        ):
            hits.append(("from_parent", srel))
            continue
        if src.parent == cand.parent:
            if re.search(rf"from\s+\.{re.escape(stem)}{MOD_END}", text):
                hits.append(("rel_from_dot_stem", srel))
                continue
            if re.search(
                rf"from\s+\.\s*import\s+[^\n]*\b{re.escape(stem)}\b",
                text,
            ):
                hits.append(("rel_from_dot_import", srel))
                continue
        # multi-dot relative that resolves to this module: from ..pkg.stem
        # only if exact path segments match — skip for conservatism unless same tree
        if re.search(rf"[\"']{re.escape(full_mod)}[\"']", text):
            hits.append(("string_mod", srel))
            continue

    if not hits:
        for cfg in search_cfg:
            try:
                text = cfg.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            crel = cfg.relative_to(root).as_posix()
            if re.search(rf"(?<![.\w]){re.escape(full_mod)}{MOD_END}", text) or posix in text.replace(
                "\\", "/"
            ):
                hits.append(("config", crel))
                break
            if len(stem) >= 12 and (stem + ".py") in text:
                hits.append(("config_fname", crel))
                break

    if not hits:
        unused.append((posix, full_mod))
        reasons["UNUSED"] += 1
        continue

    # classify
    non_init = [h for h in hits if not h[1].endswith("__init__.py")]
    if not non_init:
        init_only.append((posix, hits[0][1]))
        reasons["init_only"] += 1
    else:
        reasons[hits[0][0]] += 1

print("---STRICT UNUSED---")
for u, mod in unused:
    print(f"{u}\t{mod}")
print(f"UNUSED_COUNT={len(unused)}")
print("---INIT_ONLY (package re-export; still referenced)---")
for u, via in init_only:
    print(f"{u}\tonly via {via}")
print(f"INIT_ONLY_COUNT={len(init_only)}")
print("---REASON COUNTS---")
for k, v in reasons.most_common():
    print(f"{k}: {v}")
