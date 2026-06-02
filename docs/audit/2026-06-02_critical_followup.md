# Critical Bugs Follow-up — 2026-06-02

HEAD: `85c0fb5` (главная ветка `main`).

Аудит 2026-05-20 нашёл 30+ багов, из них 4 помечены CRITICAL. Между двумя
аудитами приоритет был на E16C / multi-arch Elbrus — критические так и не
закрыты. Этот документ — regression-чек: что осталось, минимальный fix,
тесты-доказательства, оценка трудоёмкости. **Кода не трогали.**

---

## #A. Engine race condition — singleton мутирует Node-state без mutex

### status_in_code
**ВСЁ ЕЩЁ В КОДЕ.**

- `torch/csrc/autograd/engine.h:82-85` — `Engine::get_default_engine()`
  возвращает один `static Engine`.
- `torch/csrc/autograd/node.h:246-248` — `Node::dependency_count_` (int),
  `Node::accumulated_grad_` (variable_list), `Node::visited_` (bool) — обычные
  non-atomic члены. Mutex `Node::mutex_` объявлен в `node.h:239`, но **engine
  его никогда не лочит**.
- Запись без защиты:
  - `engine.h:170-172, 186, 189-190` — `compute_dependencies` ставит
    `visited_=true`, `dependency_count_++`.
  - `engine.h:316` — `next_node->dependency_count_--`.
  - `engine.h:319-321` — read-modify-write `accumulated_grad_`.
  - `engine.h:226-254` — `accumulate_grad` мутирует `node->accumulated_grad_`.
- Стек-локальный `GraphTask` (engine.h:342) есть, но он указывает на тот же
  Node — два параллельных `backward()` на пересекающихся графах (общие leaf
  тензоры → один и тот же AccumulateGrad-Node) дают **silent wrong gradients**:
  `dependency_count_` обнуляется из двух BFS, грады мерджатся.
- Тестов на конкуррент backward в репо нет (`Grep` по `thread.*backward` пусто).

### minimal_fix
Три варианта, ранжированы по затратам:

1. **Per-graph state, никакого мутекса** (PyTorch-style). Снять
   `dependency_count_/visited_/accumulated_grad_` с Node и держать в
   `unordered_map<Node*, NodeExecState>` внутри `GraphTask`. Дороже (хеш на
   каждый node), но **тривиально parallel-safe** — поля стек-локальны.
2. **Глобальный `std::mutex` в `Engine`** — wrap'нуть execute() целиком.
   Сериализует backward, но 1-строчный фикс и закрывает race гарантированно.
   ОК как hotfix до варианта 1.
3. **`thread_local Engine`** — каждый поток получает свой singleton + свой
   набор графов. Не закрывает кейс с shared leaf-AccumulateGrad, поэтому
   слабее (1) и (2). Не рекомендую.

### tests_proving_fix
```cpp
// tests/autograd/test_concurrent_backward.cpp (новый)
TEST(EngineRace, ConcurrentBackwardSameLeaf) {
    auto W = at::randn({64,64}).requires_grad_(true);
    auto loss1 = (at::randn({64,64}) @ W).sum();
    auto loss2 = (at::randn({64,64}) @ W).sum();
    Tensor g_serial;
    { loss1.backward(); loss2.backward(); g_serial = W.grad().clone(); W.grad().zero_(); }

    std::thread t1([&]{ loss1.backward(); });
    std::thread t2([&]{ loss2.backward(); });
    t1.join(); t2.join();

    EXPECT_TRUE(at::allclose(W.grad(), g_serial, 1e-6));
}
// До fix: rate failures > 0 на 100 итерациях с TSan.
// + второй stress: 8 потоков, AccumulateGrad счётчик === ожидаемому.
```
Дополнительно — прогнать `tsan`-сборку (`-fsanitize=thread`) на любом существующем
MNIST-тренинге с `OMP_NUM_THREADS=4`.

### effort_estimate
Вариант 2 (mutex hotfix): **0.5 чел-дня**.
Вариант 1 (per-graph map): **2 чел-дня** + регрессия perf (ожидаю −5% на single-thread).

---

## #B. PromeServe path traversal → RCE chain

### status_in_code
**ВСЁ ЕЩЁ В КОДЕ.**

- `promeserve/tool_call.h:518-526` — `sandbox_path()` проверяет только
  `p.find("..") != npos`. Нет `realpath()`/`canonicalize`, нет проверки
  итогового абс. пути на префикс `root`. Имя файла `foo..bar` тоже триггерит
  false-positive, но `..` или unicode-симлинки внутри `root`
  пройдут.
- `promeserve/http_server.h:303-309` — bind на `INADDR_ANY` (0.0.0.0), **без
  env override** (`PROMESERVE_HOST` отсутствует в коде, есть только в
  `scripts/test_promeserve_tools.sh`).
- Auth: `Grep` `auth|token|Authorization` в `http_server.h` — пусто. CORS
  header настроен, но **никакой проверки токена нет**. Любой в LAN/Wi-Fi
  пишет POST на `/api/mcp/call` → тулзу выполняет.
- `promeserve/api_handlers.h:354-362` — стартует MCP-серверы из
  `~/.promeserve/mcp.json` в фоне без верификации. Если файл подменён —
  spawn arbitrary command. На запись этого файла из write_file нужен либо
  трюк с `PROMESERVE_TOOL_ROOT=~/.promeserve`, либо обход `..`-фильтра
  (например симлинк внутри sandbox-root → реальный home). Не "1 запрос", но
  реалистично.

### minimal_fix
1. **`sandbox_path()`** — после `root + "/" + p` сделать
   `std::filesystem::weakly_canonical(full)` и проверить, что результат
   начинается на `weakly_canonical(root) + "/"`. ≈ 8 строк, C++17 std::fs.
2. **`http_server.h`** — читать `PROMESERVE_HOST` (default `"127.0.0.1"`),
   парсить через `inet_pton`. Минимально — заменить
   `addr.sin_addr.s_addr = INADDR_ANY;` на:
   ```cpp
   const char* host = std::getenv("PROMESERVE_HOST");
   addr.sin_addr.s_addr = host ? inet_addr(host) : htonl(INADDR_LOOPBACK);
   ```
3. **Tool whitelist на MCP-call endpoint** — `/api/mcp/call` должен
   проверять `PROMESERVE_TOOL_ALLOW` (CSV) и/или `PROMESERVE_AUTH_TOKEN` в
   header `Authorization: Bearer …`. Любой non-localhost запрос без токена
   → 401.
4. Документация — отметить что binding на public IP требует явного
   `PROMESERVE_HOST=0.0.0.0 PROMESERVE_AUTH_TOKEN=…`.

### tests_proving_fix
- `tests/promeserve/test_sandbox_escape.cpp` — массив 20 враждебных путей:
  `../../etc/passwd`, `foo/./../../bar`, NUL-truncate `safe\x00../etc`,
  unicode `../x`, симлинк `sandbox/evil -> /tmp/owned`. Все
  должны бросить `path traversal denied`.
- `tests/promeserve/test_bind_default.sh` — стартуем сервер без env,
  `ss -ltnp | grep :8080` показывает `127.0.0.1:8080`, **не** `0.0.0.0:8080`.
  Затем `PROMESERVE_HOST=0.0.0.0` — bind на `0.0.0.0`.
- `tests/promeserve/test_unauth_block.py` — без `Authorization` хедера POST
  на `/api/mcp/call` возвращает 401. С токеном — 200.

### effort_estimate
**1 чел-день** (включая тесты + правку docs).

---

## #C. PT_ASSERT → `assert()`: -DNDEBUG выпиливает все runtime invariants

### status_in_code
**В КОДЕ, но severity МАЛАЯ.** `c10/macros/Macros.h:98`:
```cpp
#include <cassert>
#define PT_ASSERT(cond) assert(cond)
#define PT_ASSERT_MSG(cond, msg) assert((cond) && (msg))
```

**Главное открытие:** `Grep "\bPT_ASSERT\b"` по всему коду нашёл **только 1
callsite** — собственный define в `Macros.h`. В рабочем коде PT_ASSERT
**не используется**. Прочие 3 хита — `JOURNAL.md` и audit-md.

Это значит:
- Прямых сейчас потерь нет — но макрос-ловушка остаётся: первый кто
  напишет `PT_ASSERT(invariant)` в hot-path получит беззвучный no-op в
  Release.
- Все runtime invariants в коде сейчас используют либо `PT_CHECK` (бросает
  exception, `Macros.h:109-117`), либо голый `assert()` (тоже отключается
  -DNDEBUG, но это уже не наш макрос).

### minimal_fix
- В `Macros.h:97-99`: переписать на bullet-proof форму с трёх режимов:
  ```cpp
  #ifdef NDEBUG
  #define PT_ASSERT(cond) ((void)0)            // строго debug-only — рекомендация в комменте
  #define PT_INVARIANT(cond) PT_CHECK(cond)    // always-on alias для PT_CHECK
  #else
  #define PT_ASSERT(cond) assert(cond)
  #define PT_INVARIANT(cond) assert(cond)
  #endif
  ```
- Документировать: PT_ASSERT == debug-only sanity, PT_INVARIANT/PT_CHECK ==
  runtime invariant.
- Прогнать `Grep "\bassert\("` по `c10/`, `aten/`, `torch/csrc/` и
  заменить голые `assert(` (которые проверяют correctness, не just
  internal sanity) на `PT_CHECK`/`PT_INVARIANT`.

### tests_proving_fix
- Сборка с `-DNDEBUG` (Release) **+** namespaced test
  `tests/core/test_pt_check_release.cpp`:
  ```cpp
  TEST(Release, PtCheckStillThrows) {
      EXPECT_THROW(PT_CHECK(false), std::runtime_error);
      EXPECT_THROW(PT_INVARIANT(false), std::runtime_error);
      // PT_ASSERT(false) — silent in Release (документировано)
  }
  ```
- CI matrix: Debug + Release запускают полный test suite.

### effort_estimate
**0.5 чел-дня** (макрос + миграция десятка `assert(` callsites).

---

## #D. GGUF loader — unbounded counts → OOM

### status_in_code
**ВСЁ ЕЩЁ В КОДЕ.**

- `torch/io/gguf_loader.h:734-735` — `tensor_count = read_val<uint64_t>(f);`
  и `metadata_kv_count = read_val<uint64_t>(f);` без верхней границы.
- `gguf_loader.h:813` — `tensors.resize(tensor_count);` — `uint64_t` может
  быть `0xFFFFFFFFFFFFFFFF`, попытка `resize()` на 18 EB → `std::bad_alloc`
  / OOM-killer.
- `gguf_loader.h:818-822` — `n_dims = read_val<uint32_t>(f);` затем
  `t.dims.resize(n_dims);` — без upper bound. `n_dims=2^32-1` → 32 GB
  resize.
- `gguf_loader.h:783-790` — ARRAY: `count = read_val<uint64_t>(f);`,
  затем reserve `min(count, 1000000)` (good!), но **цикл `for (i = 0;
  i < count; ++i)` всё равно идёт по полному `count`** → бесконечная
  читалка из файла (или сильно длинная если файл большой). Эффективная
  защита потерялась.
- `metadata_kv_count` (`parse_metadata`, line 800) — аналогично.

### minimal_fix
Добавить константы в начало struct (после `GGUF_MAGIC`):
```cpp
static constexpr uint64_t GGUF_MAX_TENSORS = 32 * 1024;     // 32K
static constexpr uint64_t GGUF_MAX_METADATA = 8 * 1024;     // 8K
static constexpr uint32_t GGUF_MAX_DIMS = 8;                // GGML maxdim
static constexpr uint64_t GGUF_MAX_ARRAY = 1 * 1000 * 1000; // 1M
```
И на каждом из 5 read-сайтов:
```cpp
if (count > GGUF_MAX_*) throw std::runtime_error("GGUF: ... exceeds limit");
```
В ARRAY-кейсе (line 787) — поменять `for (i < count)` на
`for (i < std::min(count, GGUF_MAX_ARRAY))` + после цикла skip остатка
через `f.seekg`. Либо просто throw — все легитимные GGUF под лимитом.

### tests_proving_fix
- `tests/io/test_gguf_malformed.cpp` (директория уже создана, untracked в
  git status): 5 fixture-файлов с raw-байтами:
  - `evil_tensor_count.gguf` — `tensor_count = 0xFFFFFFFFFFFFFFFF`
  - `evil_metadata_count.gguf` — то же для metadata
  - `evil_n_dims.gguf` — `n_dims = 0xFFFFFFFF`
  - `evil_array_count.gguf` — ARRAY value с `count = 1<<40`
  - `evil_combo.gguf` — все четыре сразу
- Каждый: `EXPECT_THROW(GGUFLoader::load(path), std::runtime_error);`
  Сборка с `-fsanitize=address,undefined` — bad_alloc/RSS-spike до 1GB →
  fail.

### effort_estimate
**0.5 чел-дня** — 5 строк констант + 5 проверок + 5 fixture-файлов.

---

## Сводная таблица

| Bug | Файл | Status | Severity | Effort |
|-----|------|--------|----------|--------|
| #A engine race | `torch/csrc/autograd/engine.h:82,186,316`, `node.h:246-248` | в коде | CRITICAL (silent wrong grads) | 0.5–2 дня |
| #B promeserve RCE | `promeserve/tool_call.h:518-526`, `http_server.h:303-309`, `api_handlers.h:354-362` | в коде | CRITICAL (LAN-exposed) | 1 день |
| #C PT_ASSERT NDEBUG | `c10/macros/Macros.h:98` | в коде, **0 callsites** | LOW (потенциальная ловушка) | 0.5 дня |
| #D GGUF unbounded | `torch/io/gguf_loader.h:734-735,787-790,813,818-822` | в коде | HIGH (OOM на malformed) | 0.5 дня |

**Итого: 2.5–4 чел-дня закроют все 4 CRITICAL.**

Порядок рекомендую: #D (быстрее всего, защищает реальных пользователей) →
#B (продакт-риск, открытый сервер) → #A (фундаментальная корректность) →
#C (косметика, нет regression-риска).
