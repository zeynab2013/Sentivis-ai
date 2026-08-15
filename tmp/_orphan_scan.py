#!/usr/bin/env python3
"""Conservative orphan-module scan for focus packages."""
from __future__ import annotations

import os
import re
from pathlib import Path

root = Path(r"D:\comptitions\America competition(innoverse) 2026\SENTIVIS AI")
focus = ["language", "analysis", "vision", "services", "core"]
exclude_dir_names = {
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


def should_skip_dir(dirpath: Path) -> bool:
    parts = set(dirpath.relative_to(root).parts) if dirpath != root else set()
    if parts & exclude_dir_names:
        return True
    if "site-packages" in str(dirpath):
        return True
    return False


all_py: list[Path] = []
for dirpath, dirnames, filenames in os.walk(root):
    dp = Path(dirpath)
    dirnames[:] = [
        d
        for d in dirnames
        if d not in exclude_dir_names and not d.startswith(".")
    ]
    if should_skip_dir(dp):
        dirnames[:] = []
        continue
    for f in filenames:
        if f.endswith(".py"):
            all_py.append(dp / f)

candidates: list[Path] = []
for p in all_py:
    rel = p.relative_to(root)
    if rel.parts[0] not in focus:
        continue
    if p.name == "__init__.py":
        continue
    if "translations" in rel.parts:
        continue
    candidates.append(p)

config_exts = {
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".ini",
    ".cfg",
    ".md",
    ".bat",
    ".ps1",
    ".sh",
    ".txt",
}
config_files: list[Path] = []
for dirpath, dirnames, filenames in os.walk(root):
    dp = Path(dirpath)
    dirnames[:] = [
        d
        for d in dirnames
        if d not in exclude_dir_names and not d.startswith(".")
    ]
    if should_skip_dir(dp):
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
            config_files.append(dp / f)

# Include tests AND excluded dirs for reference search (except site-packages/venv)
# User asked: unused = no imports from any other project file, not referenced in
# entry points, tests, or configs. So search corpus must INCLUDE tests/ui/etc.
search_py: list[Path] = []
search_exclude = {"site-packages", "__pycache__", "venv", ".venv", "node_modules", "tmp"}
for dirpath, dirnames, filenames in os.walk(root):
    dp = Path(dirpath)
    dirnames[:] = [
        d
        for d in dirnames
        if d not in search_exclude and not d.startswith(".")
    ]
    if any(x in str(dp) for x in ("site-packages", "venv", ".venv")):
        dirnames[:] = []
        continue
    for f in filenames:
        if f.endswith(".py"):
            search_py.append(dp / f)

search_cfg: list[Path] = []
for dirpath, dirnames, filenames in os.walk(root):
    dp = Path(dirpath)
    dirnames[:] = [
        d
        for d in dirnames
        if d not in search_exclude and not d.startswith(".")
    ]
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


def file_mentions_module(text: str, full_mod: str, parent: str, stem: str) -> str | None:
    if re.search(rf"(?:from|import)\s+{re.escape(full_mod)}\b", text):
        return "import_full"
    if parent and re.search(
        rf"from\s+{re.escape(parent)}\s+import\s+[^\n]*\b{re.escape(stem)}\b",
        text,
    ):
        return "from_parent"
    if re.search(rf"[\"']{re.escape(full_mod)}[\"']", text):
        return "string_mod"
    return None


def relative_hit(src: Path, cand: Path, text: str, stem: str) -> str | None:
    # from .stem / from ..pkg.stem etc.
    if re.search(rf"from\s+\.+{re.escape(stem)}\b", text):
        # same package tree only — conservative: require shared parent package dir
        try:
            # resolve: if src can see cand via relative import of that stem name
            if src.parent == cand.parent:
                return "rel_same_dir"
            # from ..stem when cand is parent package sibling module? uncommon
            if cand.parent in src.parents and cand.stem == stem:
                # e.g. from ..foo import ... where foo.py is parent package sibling
                # only if the dotted relative path length matches
                return "rel_ancestor_pkg"
        except Exception:
            pass
    if re.search(rf"from\s+\.+\s*import\s+[^\n]*\b{re.escape(stem)}\b", text):
        if src.parent == cand.parent:
            return "rel_import_same_dir"
    return None


orphans: list[tuple[str, str]] = []
for cand in sorted(candidates):
    rel = cand.relative_to(root)
    full_mod = ".".join(rel.with_suffix("").parts)
    stem = rel.stem
    parent = ".".join(rel.with_suffix("").parts[:-1])

    hit_info = None
    for src in search_py:
        if src.resolve() == cand.resolve():
            continue
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        kind = file_mentions_module(text, full_mod, parent, stem)
        if kind:
            hit_info = (kind, src.relative_to(root).as_posix())
            break
        kind = relative_hit(src, cand, text, stem)
        if kind:
            hit_info = (kind, src.relative_to(root).as_posix())
            break

    if hit_info:
        continue

    # config / docs string refs
    posix = rel.as_posix()
    for cfg in search_cfg:
        try:
            text = cfg.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if full_mod in text or posix in text.replace("\\", "/"):
            hit_info = ("config", cfg.relative_to(root).as_posix())
            break
        # filename only if distinctive
        if len(stem) >= 10 and (stem + ".py") in text:
            hit_info = ("config_fname", cfg.relative_to(root).as_posix())
            break

    if hit_info:
        continue

    orphans.append((posix, full_mod))

print("---ORPHANS---")
for posix, full_mod in orphans:
    print(f"{posix}\t{full_mod}")
print(f"COUNT={len(orphans)}")
