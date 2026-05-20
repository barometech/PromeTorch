# Packaging — `pip install prometorch`

## Текущее состояние

| Файл | Назначение |
|------|-----------|
| `pyproject.toml` | scikit-build-core build backend + project metadata |
| `LICENSE` | Apache 2.0 |
| `NOTICE` | Copyright + third-party attribution |
| `TRADEMARKS.md` | Trademark policy |
| `.github/workflows/wheels.yml` | CI: builds wheels for Linux/Win/Mac × Python 3.9-3.13 |

## Локальная сборка wheel

```bash
# Поставить build dependencies
pip install build scikit-build-core pybind11 cmake

# Собрать wheel
python -m build --wheel

# Результат — в dist/
ls -la dist/
# prometorch-0.1.0a1-cp312-cp312-win_amd64.whl
```

## Локальный install + test

```bash
pip install dist/prometorch-0.1.0a1-cp312-cp312-win_amd64.whl
python -c "import prometorch; print(prometorch.__version__)"
```

## Multi-platform через cibuildwheel (локально)

```bash
pip install cibuildwheel
cibuildwheel --platform linux    # требует Docker
cibuildwheel --platform windows  # на Win-машине
cibuildwheel --platform macos    # на macOS
```

Результат — куча wheels в `wheelhouse/`.

## Публикация на PyPI

### Через GitHub Actions (рекомендуется)

1. Настрой **PyPI Trusted Publishing** на pypi.org:
   - https://pypi.org/manage/account/publishing/
   - Add a new pending publisher:
     - Project name: `prometorch`
     - Owner: `barometech`
     - Repository name: `PromeTorch`
     - Workflow name: `wheels.yml`
     - Environment name: `pypi`

2. То же для **TestPyPI** на test.pypi.org → environment `test-pypi`.

3. **Smoke-тест на TestPyPI:**
   - GitHub → Actions → "Build & Publish wheels" → Run workflow
   - publish_target = `test-pypi`
   - После прохождения: `pip install -i https://test.pypi.org/simple/ prometorch`

4. **Релиз на PyPI:**
   - `git tag v0.1.0a1 && git push --tags`
   - Workflow автоматически запустится, соберёт wheels, опубликует на PyPI
   - Через 5 минут: `pip install prometorch` работает у всех

### Вручную через twine (если CI сломан)

```bash
pip install twine
python -m build --sdist --wheel
twine upload --repository testpypi dist/*  # сначала test
twine upload dist/*                         # потом main PyPI
```

Требует API token из pypi.org → Account Settings → API tokens.

## Что НЕ включено в этот baseline wheel

| Что | Почему | Где брать |
|-----|--------|-----------|
| CUDA backend | Wheel >100MB (PyPI limit) + binding к конкретной CUDA версии | Отдельный wheel `prometorch-cu121` через external index URL (как `torch+cu121`) |
| NMCard / NMQuad | Нужен SDK который не open-source | Сборка из исходников с `-DPT_USE_NMQUAD=ON` |
| Elbrus E2K | LCC компилятор not on PyPI workers | Сборка на самой машине через `./scripts/build-elbrus.sh` |
| LinQ H1M | Не stable backend | TODO |

В будущем — отдельные wheels через external PEP 503 index. Сейчас — только CPU
(x86_64 + arm64).

## Версионирование

Используется **SemVer 2.0** + PEP 440:

- `0.1.0a1` — alpha, ABI может меняться без deprecation
- `0.1.0b1` — beta, freeze API
- `0.1.0rc1` — release candidate
- `0.1.0` — stable
- `0.2.0` — minor (new features, backward-compatible)
- `1.0.0` — major (stability promise)

Сейчас на `0.1.0a1`. Можно бамп через CLI:

```bash
# Bump version
sed -i 's/version = "0.1.0a1"/version = "0.1.0a2"/' pyproject.toml
git add pyproject.toml && git commit -m "release: v0.1.0a2"
git tag v0.1.0a2 && git push --tags  # триггерит CI publish
```

## Известные ограничения

1. **Размер wheel** — CPU-only wheel ≈ 5-10 MB (зависит от shared libs).
   CUDA wheel будет 200-500 MB → external index.
2. **Cross-compile** — cibuildwheel на Linux собирает manylinux2014_x86_64 +
   manylinux2014_aarch64 (если доступен QEMU). Windows/Mac — native runners.
3. **Elbrus / NMCard / Baikal** — пользователи строят из source через
   `git clone + ./scripts/build-elbrus.sh`. PyPI wheel для них не имеет смысла
   (нужен LCC compiler, EML lib, проприетарные dependencies).
