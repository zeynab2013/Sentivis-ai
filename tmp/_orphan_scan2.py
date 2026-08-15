#!/usr/bin/env python3
"""Stricter orphan scan + hit-reason dump for weak matches."""
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


def collect_py(exclude_names: set[str]) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames if d not in exclude_names and not d.startswith(".")
        ]
        if any(x in str(dp) for x in ("site-packages", "venv", ".venv")):
            dirnames[:] = []
            continue
        # for candidate collection, also skip excluded package trees
        if exclude_names is exclude_candidates:
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


search_py = collect_py(search_exclude)
# candidates only in focus, not in excluded trees
candidates: list[Path] = []
for p in collect_py(exclude_candidates):
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
weak: list[tuple[str, str, str]] = []
strong_unused: list[str] = []

for cand in sorted(candidates):
    rel = cand.relative_to(root)
    full_mod = ".".join(rel.with_suffix("").parts)
    stem = rel.stem
    parent = ".".join(rel.with_suffix("").parts[:-1])
    posix = rel.as_posix()

    strong = None
    weak_hit = None

    for src in search_py:
        if src.resolve() == cand.resolve():
            continue
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        srel = src.relative_to(root).as_posix()

        if re.search(rf"(?:from|import)\s+{re.escape(full_mod)}\b", text):
            strong = ("import_full", srel)
            break
        if parent and re.search(
            rf"from\s+{re.escape(parent)}\s+import\s+[^\n]*\b{re.escape(stem)}\b",
            text,
        ):
            strong = ("from_parent", srel)
            break
        # relative same-dir only (strict)
        if src.parent == cand.parent:
            if re.search(rf"from\s+\.{re.escape(stem)}\b", text):
                strong = ("rel_from_dot_stem", srel)
                break
            if re.search(rf"from\s+\.\s*import\s+[^\n]*\b{re.escape(stem)}\b", text):
                strong = ("rel_from_dot_import", srel)
                break
        # string literal of full module
        if re.search(rf"[\"']{re.escape(full_mod)}[\"']", text):
            strong = ("string_mod", srel)
            break

        # WEAK: broader relative / partial
        if re.search(rf"from\s+\.+{re.escape(stem)}\b", text):
            weak_hit = weak_hit or ("weak_rel_stem", srel)
        if re.search(rf"from\s+\.+\s*import\s+[^\n]*\b{re.escape(stem)}\b", text):
            weak_hit = weak_hit or ("weak_rel_import", srel)

    if strong:
        reasons[strong[0]] += 1
        continue

    # config
    cfg_hit = None
    for cfg in search_cfg:
        try:
            text = cfg.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        crel = cfg.relative_to(root).as_posix()
        if full_mod in text or posix in text.replace("\\", "/"):
            cfg_hit = ("config", crel)
            break
        if len(stem) >= 12 and (stem + ".py") in text:
            cfg_hit = ("config_fname", crel)
            break

    if cfg_hit:
        reasons[cfg_hit[0]] += 1
        continue

    if weak_hit:
        reasons[weak_hit[0]] += 1
        weak.append((posix, weak_hit[0], weak_hit[1]))
        continue

    strong_unused.append(posix)
    reasons["UNUSED"] += 1

print("---STRICT UNUSED---")
for u in strong_unused:
    print(u)
print(f"UNUSED_COUNT={len(strong_unused)}")
print("---WEAK ONLY (may be false positive used)---")
for w in weak:
    print(f"{w[0]}\t{w[1]}\tvia {w[2]}")
print("---REASON COUNTS---")
for k, v in reasons.most_common():
    print(f"{k}: {v}")
