"""
PromeTorch build shim.

Сборка идёт через scikit-build-core (см. pyproject.toml:
`build-backend = "scikit_build_core.build"`). Этот файл — НЕ backend и
больше не собирает расширение; он оставлен только чтобы legacy-вызов
`python setup.py ...` падал громко, а не «успешно» без C++-расширения
(аудит P2-1: тихий fallback давал установленный, но нерабочий пакет).
"""
import sys

sys.exit(
    "setup.py не используется: PromeTorch собирается через scikit-build-core.\n"
    "Ставь так:   pip install .        (или  pip install -e .  для dev)\n"
    "Сборка C++:  cmake --build build  (см. docs/BUILD.md)."
)
