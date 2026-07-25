#!/usr/bin/env python3
"""
gen_stats.py — single source of truth для числовых характеристик PromeTorch.

Считает реальные счётчики из исходников (backward functions, CUDA kernels,
optimizers, LR schedulers, tests, строки кода) и пишет STATS.md + STATS.json.

Зачем: README/CLAUDE/JOURNAL накопили противоречия (backward 112/119/121,
CUDA 92/99/149, optimizers 10/16) — audit 2026-06-02 _claims_drift нашёл
30 находок. Этот скрипт — единственный авторитетный источник. CI-гейт
(`python scripts/gen_stats.py --check`) валит сборку если STATS.md
разошёлся с реальностью.

Usage:
    python scripts/gen_stats.py            # перегенерировать STATS.md/json
    python scripts/gen_stats.py --check    # CI: упасть если есть drift
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def _tracked_files():
    """git ls-files — только tracked (детерминированно, без build-мусора и
    untracked dev-папок типа nm_quad_qwen/)."""
    try:
        out = subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True)
    except Exception:
        return []
    return [REPO / line for line in out.splitlines() if line]


_ALL_TRACKED = None


def _iter_files(exts):
    global _ALL_TRACKED
    if _ALL_TRACKED is None:
        _ALL_TRACKED = _tracked_files()
    for p in _ALL_TRACKED:
        if p.suffix in exts and p.is_file():
            yield p


def count_backward_functions():
    """struct/class *Backward в torch/csrc/autograd/functions/.
    По git-tracked (rglob по ФС ловил untracked/локальные файлы → дрейф
    между dev-деревом и чистым CI-checkout'ом)."""
    sub = "torch/csrc/autograd/functions/"
    pat = re.compile(r"^\s*(?:struct|class)\s+([A-Za-z_]\w*Backward)\b", re.M)
    names = set()
    for p in _iter_files({".h"}):
        if p.relative_to(REPO).as_posix().startswith(sub):
            names.update(pat.findall(p.read_text(encoding="utf-8", errors="replace")))
    return len(names)


def count_cuda_kernels():
    """__global__ kernel definitions в aten/src/ATen/cuda/ (git-tracked)."""
    sub = "aten/src/ATen/cuda/"
    n = 0
    for p in _iter_files({".cu"}):
        if p.relative_to(REPO).as_posix().startswith(sub):
            txt = p.read_text(encoding="utf-8", errors="replace")
            n += len(re.findall(r"__global__\s+\w", txt))
    return n


def count_optimizers():
    """Уникальные оптимизаторы в torch/optim/ (исключая utility-файлы)."""
    d = REPO / "torch" / "optim"
    util = {"optim", "optimizer", "lr_scheduler", "ema"}
    return sum(1 for p in d.glob("*.h") if p.stem not in util)


def count_lr_schedulers():
    """class *LR / *Scheduler / WarmRestarts в torch/optim/lr_scheduler.h."""
    f = REPO / "torch" / "optim" / "lr_scheduler.h"
    if not f.exists():
        return 0
    txt = f.read_text(encoding="utf-8", errors="replace")
    names = set(re.findall(r"\bclass\s+([A-Za-z_]\w+)", txt))
    sched = {n for n in names if re.search(r"(LR|Scheduler|WarmRestarts|Cyclic|OneCycle)$", n)}
    return len(sched)


def count_tests():
    """TEST(...) макросы + gtest TEST_F во всех test-файлах."""
    n = 0
    for p in _iter_files({".cpp", ".cc"}):
        rel = p.relative_to(REPO).as_posix()
        if "test" not in rel.lower():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        n += len(re.findall(r"\bTEST(?:_F)?\s*\(", txt))
    return n


def count_lines():
    """Строки кода по категориям через wc-эквивалент (чистый Python)."""
    cats = {"cpp_h": {".cpp", ".h", ".hpp"}, "cuda": {".cu", ".cuh"},
            "python": {".py"}}
    out = {}
    for name, exts in cats.items():
        total = 0
        for p in _iter_files(exts):
            try:
                total += sum(1 for _ in p.open("rb"))
            except OSError:
                pass
        out[name] = total
    return out


def git_short_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            text=True).strip()
    except Exception:
        return "unknown"


def collect():
    lines = count_lines()
    return {
        "head": git_short_head(),
        "backward_functions": count_backward_functions(),
        "cuda_kernels": count_cuda_kernels(),
        "optimizers": count_optimizers(),
        "lr_schedulers": count_lr_schedulers(),
        "tests": count_tests(),
        "lines_cpp_h": lines["cpp_h"],
        "lines_cuda": lines["cuda"],
        "lines_python": lines["python"],
    }


def render_md(s):
    return f"""# PromeTorch — STATS (auto-generated)

**НЕ редактировать вручную.** Сгенерировано `scripts/gen_stats.py` из
исходников. README/docs должны ссылаться на эти числа, не хардкодить их.
CI-гейт: `python scripts/gen_stats.py --check`.

HEAD: `{s['head']}`

| Метрика | Значение |
|---------|---------:|
| Backward functions | {s['backward_functions']} |
| CUDA kernels (`__global__`) | {s['cuda_kernels']} |
| Optimizers | {s['optimizers']} |
| LR schedulers | {s['lr_schedulers']} |
| Tests (`TEST`/`TEST_F`) | {s['tests']} |
| Строк C++/headers | {s['lines_cpp_h']:,} |
| Строк CUDA | {s['lines_cuda']:,} |
| Строк Python | {s['lines_python']:,} |
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="CI mode: exit 1 if STATS.md/json out of date")
    args = ap.parse_args()

    stats = collect()
    md_path = REPO / "STATS.md"
    json_path = REPO / "STATS.json"
    new_md = render_md(stats)
    new_json = json.dumps(stats, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        old_md = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        old_json = json_path.read_text(encoding="utf-8") if json_path.exists() else ""
        if old_md != new_md or old_json != new_json:
            print("STATS drift detected. Run: python scripts/gen_stats.py",
                  file=sys.stderr)
            print("--- current ---", file=sys.stderr)
            print(new_json, file=sys.stderr)
            return 1
        print("STATS up to date.")
        return 0

    md_path.write_text(new_md, encoding="utf-8")
    json_path.write_text(new_json, encoding="utf-8")
    print(f"Wrote {md_path.name} + {json_path.name}")
    print(new_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
