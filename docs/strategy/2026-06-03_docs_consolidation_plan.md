# План консолидации документации PromeTorch

**Дата:** 2026-06-03
**HEAD:** `59822c7`
**Контекст:** внешний консультант (DeepSeek) предложил "Docs as Code" (MkDocs/Docusaurus + GitHub Pages, версионирование по тегам). Этот документ — оценка под реальность репо и конкретный план.
**Метрика проблемы:** 33 .md в корне (12 847 строк), 60 .md в `docs/` (≈40K строк). README — 1422 строки. Числовые противоречия документированы в `docs/audit/2026-06-02_claims_drift.md` (30 находок) и `_SUMMARY.md` (#13).

---

## 0. Вердикт по DeepSeek-предложению

| Предложение | Решение | Обоснование |
|---|---|---|
| MkDocs + Material theme | **ДА** | Markdown уже есть, Python в стеке, нулевой JS-tooling, `mkdocs gh-deploy` в одну команду. |
| Docusaurus | **НЕТ (overkill)** | React/Node toolchain, MDX, i18n-машинерия — несоразмерно для solo-репо. MkDocs покрывает 100% нужд. |
| GitHub Pages деплой | **ДА** | Бесплатно, нужен 1 workflow (`.github/workflows/` уже есть 5 шт, добавить `docs.yml`). |
| Версионирование по тегам (mike) | **ОТЛОЖИТЬ** | `mike` оправдан после стабильного 1.0. Сейчас 0.1.0a1 — версионировать нечего, only `latest`. |

Итог: MkDocs Material + один GH Pages workflow. Не Docusaurus, не mike (пока).

---

## 1. Инвентарь: аудитория → судьба

Аудитории: **USER** (внешний пользователь/reviewer), **DEV** (контрибьютор), **AI** (инструкции ассистенту — в публичный сайт НЕ идут: academic reviewers воспримут негативно, уже обсуждали).

### Корень — AI-internal (НЕ в сайт, оставить в репо как есть)
| Файл | Аудитория | Судьба |
|---|---|---|
| CLAUDE.md (gitignored) | AI | keep, **полный update Phase-table** (см. §3) |
| AVOIDRECURSION.md (gitignored) | AI | keep |
| JOURNAL.md (3830) | AI/DEV | keep в репо, НЕ в сайт |
| JOURNAL_BREAKDOWNS.md (439) | AI | keep |

### Корень — USER-facing (источник для сайта)
| Файл | Судьба |
|---|---|
| README.md (1422) | **split** → короткий README (≤250 строк) + контент в сайт (§4) |
| README.en.md (251) | keep как зеркало; сайт пока RU-only |
| RESULTS.md (229) | **single source benchmarks** → сайт `benchmarks/`, на него ссылаются все BENCH_*. |
| RESUME.md (151) | merge в landing сайта |
| CHANGELOG.md, RELEASE_NOTES_v0.1.0.md | сайт `changelog/` |
| LICENSE-смежные: SECURITY, CODE_OF_CONDUCT, CONTRIBUTING, TRADEMARKS, THIRD_PARTY_NOTICES | keep в корне (GitHub-конвенция); CONTRIBUTING+CoC дублировать в сайт `contributing/` |

### Корень — BENCH_*.md (9 файлов) → DEV/USER
- **Проблема:** разрозненные числа, дублируют RESULTS.md. **Судьба:** RESULTS.md = single source; BENCH_* → `docs/benchmarks/` как детальные приложения, верхние числа из RESULTS.md.

### Корень — отчёты/архив → archive
| Файл | Судьба |
|---|---|
| TECHNICAL_SPECIFICATION.md (1327) | **archive** — план 2026-01 с пустыми чекбоксами, противоречит реальности. Пометить "ARCHIVED — see README". |
| INFRASTRUCTURE_AUDIT.md (1039), GAP_ANALYSIS_VS_PYTORCH.md, CUDA_CRASH_INVESTIGATION.md, EXAMPLES_VERIFIED.md, TEST_PLAN.md | DEV-internal → `docs/dev/` или archive, НЕ в публичный сайт |
| ELBRUS_REPORT.md, REPORT_ELBRUS_LLM_INFERENCE_2026-05-02.md | merge в `platforms/elbrus` сайта |
| PROMEPEDIA.md (754) | глоссарий → сайт `reference/glossary` (хороший USER-контент) |

### `docs/` — публичные (в сайт)
BUILD_ELBRUS, BUILD_WINDOWS, PACKAGING, ROCM, MCP_INTEGRATION, MULTIHOST_SGD, COMPARISON_VS_PYTORCH_RU, PROMETHORCH_RU, ADAM_KILLER_SPEC, elbrus/*, elbrus_isa/{README,PERFORMANCE_BY_ISA,ROADMAP}, nmcard/*, nmquad/README_NMQUAD_RU.

### `docs/` — internal (НЕ в сайт)
- `docs/audit/*` (21 файл) — internal dev. **archive/keep**, не публиковать.
- `docs/research/*` (F1/F2/AGENT_STACK_R2/AI_CHAT_FEATURES) — DEV-планы, keep.
- `docs/strategy/*` — этот план, internal.
- `docs/GEMINI_AMD_INTEL_PLAN.md` — internal план.
- `docs/elbrus_isa/MANUAL.md` (11809) — справочник ISA; ссылка из сайта, но не рендерить (огромный).

---

## 2. Структура MkDocs site

```
nav:
  Главная            (из README intro + RESUME)
  Установка          (BUILD_WINDOWS, BUILD_ELBRUS, PACKAGING, docker, ROCm)
  Quickstart         (MNIST C++, Python API, PromeServe)
  Платформы/
    Эльбрус          (elbrus/*, PERFORMANCE_BY_ISA, ELBRUS_REPORT)
    NM Card / NM Quad (nmcard, nmquad)
    NVIDIA / CUDA
  API Reference      (тензоры, nn, optim, autograd, data, AMP)
  Бенчмарки          (RESULTS.md = canonical + BENCH_* приложения)
  Сравнение с PyTorch (COMPARISON_VS_PYTORCH_RU, GAP_ANALYSIS, Known Limitations)
  Глоссарий          (PROMEPEDIA)
  Contributing       (CONTRIBUTING, CoC)
  Changelog
```

**НЕ в сайт:** CLAUDE/AVOIDRECURSION/JOURNAL/JOURNAL_BREAKDOWNS, docs/audit/*, docs/strategy/*, docs/research/*, GEMINI_AMD_INTEL_PLAN, INFRASTRUCTURE_AUDIT, TECHNICAL_SPECIFICATION (archived), ISA MANUAL (link-only).

`mkdocs.yml`: `theme: material`, `docs_dir: site_src/` (символлинки или `include`-плагин на существующие .md, чтобы НЕ дублировать файлы), `markdownextensions: admonition, pymdownx.*`.

---

## 3. Устранение числовых противоречий — single source

**Проблема (audit #11):** backward 112/119/121 (реально **121**), CUDA "65+18+9=92" vs 149, line counts 114K vs 122 978, NMCard 93.64% vs **88.94%**, optimizers 4 vs 16, tests 720 vs 907.

**Решение — `STATS.md` генерится скриптом `scripts/gen_stats.py`:**
- `git grep -c` по паттернам → backward functions, CUDA `__global__`, optimizer/scheduler классы, `TEST(`, `wc -l` по torch/aten/c10/python/examples.
- Скрипт пишет `STATS.md` + JSON. MkDocs-макрос/CI подставляет числа.
- CI-гейт (`docs.yml`): regenerate STATS, `git diff --exit-code` → красный билд если drift.
- **README/сайт ссылаются на STATS, не хардкодят.** Один источник правды навсегда.

**Разовая чистка перед автоматизацией** (по таблице audit #11 §"Конкретные строки"): 121 backward везде, 149 CUDA с правильным breakdown, 88.94% NMCard, line counts 122 978/18 622/4 637, 16 opt/16 sched в L983/984, tests 907.

---

## 4. README split

**Оставить в README (≤250 строк):** заголовок+бейджи, 1-абзац pitch, ключевые числа (из STATS), 3-5 headline benchmark (ссылка RESULTS.md), быстрый старт (build one-liner + MNIST + PromeServe), ссылки на сайт, лицензия (коротко), авторы.

**Вынести в сайт:** Coverage-таблица, полные Результаты (Эльбрус/CUDA/NMQuad), "Что нового" log'и (→ changelog), Known Limitations (→ Сравнение), весь API Reference, Архитектура/Компоненты, CMake опции, Troubleshooting, Структура проекта, Roadmap.

---

## 5. CLAUDE/JOURNAL — оставить в репо

- CLAUDE.md, AVOIDRECURSION.md уже **gitignored** (подтверждено в `.gitignore`). Остаются локально, в публичный сайт не идут — корректно.
- JOURNAL.md — committed, оставить в репо (ценная история для DEV), но **исключить из nav сайта**.
- Перемещать никуда не надо: gitignore + nav-exclusion решают задачу. `.bak` уже убраны (из памяти сессии).

---

## 6. Effort estimate

| Этап | Объём |
|---|---|
| `scripts/gen_stats.py` + STATS.md + CI-гейт | ~0.5 дня |
| Разовая чистка чисел (audit #11 таблица, ~40 правок) | ~0.5 дня |
| README split (короткий + перенос в сайт) | ~0.5 дня |
| `mkdocs.yml` + nav + include-плагин (без дублей) | ~0.5 дня |
| `docs.yml` GH Pages workflow | ~2 ч |
| Архивация TECH_SPEC + раскладка docs/dev vs public | ~2 ч |
| CLAUDE.md Phase-table полный update (audit #11 §CLAUDE) | ~2 ч |
| **Итого** | **~3 дня** |

Версионирование `mike` — отдельно после 1.0.
