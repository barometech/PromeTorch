"""
PromeServe integration tests — shared fixtures.

Стартует реальный promeserve экзешник в фоне для test-сессии. Тесты
говорят с ним по HTTP на TEST_PORT. После сессии — graceful shutdown.

Если переменная PROMESERVE_BIN не задана — авто-ищем в build_cpu_gguf/
и build_promeserve/.
"""
import os
import subprocess
import time
import socket
import shutil
from pathlib import Path
import pytest
import urllib.request
import urllib.error

REPO_ROOT = Path(__file__).resolve().parents[2]

# Sandbox-корень tool-инструментов (list_dir/read_file/write_file). Сервер
# джойнит user_path к нему и запрещает выход наружу (path traversal). Тесты
# пишут файлы СЮДА и обращаются относительным путём — иначе list_dir на
# внешний pytest-tmp (вне sandbox) возвращает {"files":[]} и тест падал.
TOOL_ROOT = REPO_ROOT / "tests" / "promeserve" / "_toolroot"

# По умолчанию слушаем на 18434 (Ollama 11434 + offset 7000) чтобы не конфликтовать
# с реальным Ollama если он запущен у разработчика.
TEST_PORT = int(os.environ.get("PROMESERVE_TEST_PORT", "18434"))
HOST = "127.0.0.1"
BASE_URL = f"http://{HOST}:{TEST_PORT}"


def find_promeserve_bin() -> Path:
    env = os.environ.get("PROMESERVE_BIN")
    if env:
        p = Path(env)
        if p.exists():
            return p
        raise FileNotFoundError(f"PROMESERVE_BIN points to non-existent {env}")

    # Авто-поиск
    candidates = [
        REPO_ROOT / "build_cpu_gguf" / "promeserve" / "promeserve.exe",
        REPO_ROOT / "build_cpu_gguf" / "promeserve" / "promeserve",
        REPO_ROOT / "build_promeserve" / "promeserve" / "promeserve.exe",
        REPO_ROOT / "build_promeserve" / "promeserve" / "promeserve",
        REPO_ROOT / "build_elbrus" / "promeserve" / "promeserve",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "promeserve binary not found. Set PROMESERVE_BIN env or build via "
        "scripts/build-elbrus.sh / Windows nmake."
    )


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            return s.connect_ex((host, port)) == 0
        except OSError:
            return False


def wait_for_server(timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/api/version", timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def promeserve_url() -> str:
    """Стартует promeserve один раз для всей сессии. Возвращает base URL."""
    if port_in_use(HOST, TEST_PORT):
        pytest.skip(f"port {TEST_PORT} занят — отдельный promeserve уже работает?")

    bin_path = find_promeserve_bin()
    log_dir = REPO_ROOT / "tests" / "promeserve" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "promeserve_test.log"

    # CPU-only для tests — мы тестируем API, не GPU инференс.
    TOOL_ROOT.mkdir(parents=True, exist_ok=True)
    server_env = dict(os.environ, PROMESERVE_TOOL_ROOT=str(TOOL_ROOT))
    args = [str(bin_path), "--port", str(TEST_PORT), "--device", "cpu"]
    proc = subprocess.Popen(
        args,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=server_env,
    )

    try:
        if not wait_for_server(timeout=30.0):
            proc.terminate()
            log = log_file.read_text(errors="replace")[-2000:]
            pytest.fail(f"promeserve не поднялся за 30s. Tail log:\n{log}")
        yield BASE_URL
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def tool_root(tmp_path_factory):
    """Папка ВНУТРИ sandbox-корня сервера, куда тест кладёт файлы и обращается
    относительным путём. Уникальная на тест, чистится после."""
    import uuid
    d = TOOL_ROOT / f"t_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    yield d, d.relative_to(TOOL_ROOT).as_posix()  # (абсолютный путь, относительный к root)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def has_model(promeserve_url) -> bool:
    """True если хотя бы одна модель доступна — гейтит inference-tests."""
    import json
    with urllib.request.urlopen(f"{promeserve_url}/api/tags", timeout=5) as resp:
        body = json.loads(resp.read().decode())
    return len(body.get("models", [])) > 0
