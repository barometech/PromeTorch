# Multi-host SGD между гетерогенными Эльбрусами

## Идея

Тысячи Эльбрусов разных поколений (v3/v4/v5/v6) у юзеров. Запустим распределённое
обучение поверх **существующего** intra-host 4-proc Local SGD, а между хостами
синхронизируемся через **SCP-AllReduce** раз в N шагов.

## Архитектура

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  E2S/4C (v3)    │    │  E16C (v6)      │    │  E8C2 master    │
│  16 cores, 750M │    │  8 cores, 2GHz  │    │  32 cores, 1.5G │
│  weight 0.3×    │    │  weight 1.0×    │    │  weight 2.4×    │
│                 │    │                 │    │                 │
│  4-proc DDP ───┐│    │  4-proc DDP ───┐│    │  4-proc DDP ───┐│
│  intra Local   ││    │  intra Local   ││    │  intra Local   ││
│  SGD K_intra=10││    │  SGD K_intra=10││    │  SGD K_intra=10││
└────────────────┼┘    └────────────────┼┘    └────────────────┼┘
                 │ INTERVAL=500 steps   │                      │
                 ▼                      ▼                      ▼
                 SCP /tmp/pt_mh/local.bin → master:peer_*.bin
                                                                │
                          ┌─────────────────────────────────────┘
                          ▼
                 multihost_avg.py (weighted average)
                          │
                          ▼
                 master:avg.bin → SCP back to workers
                          │
                          ▼
                 все 3 хоста загружают averaged weights
                          │
                          ▼
                 продолжают следующий INTERVAL шагов
```

## Алгоритм (Hierarchical Local SGD)

```
for round = 1..MAX_ROUNDS:
    # Phase 1: Intra-host training INTERVAL steps от текущих weights
    каждый host запускает 4-proc Local SGD на своих INTERVAL шагов
    (внутри уже работает file-based gradient averaging между proc 0..3)

    # Phase 2: Inter-host averaging (SCP-AllReduce)
    worker:
        scp /tmp/pt_mh/local.bin → master:/tmp/pt_mh/peer_$id.bin
        ждать /tmp/pt_mh/avg_round_$N.bin на мастере
        scp master:avg.bin → /tmp/pt_mh/local.bin
    master:
        ждать всех peer_*.bin (timeout 30 мин)
        python3 multihost_avg.py avg.bin local.bin peer1.bin peer2.bin \
            --weighted 2.4,1.0,0.3
        avg.bin → доступен workers через SCP

    # Phase 3: Все 3 хоста загружают averaged weights и стартуют next round
```

## Веса хостов

Compute capacity:
- **4C/E2S v3** (16 × 750MHz) ≈ **0.3**
- **16C v6** (8 × 2GHz)         ≈ **1.0**
- **8C2 v5** (32 × 1.5GHz)      ≈ **2.4**

Использовать через `PT_MH_WEIGHTS="2.4,1.0,0.3"`. Порядок:
`master_self,peer1,peer2,...` (как в `PT_MH_PEERS`).

**Зачем веса:** более мощный хост увидел больше batch'ей за тот же
wall-time, его weights ближе к "правильным" для каждой итерации.

## Trade-offs

| Параметр | Малое значение | Большое значение |
|----------|----------------|-------------------|
| `INTERVAL` (intra-steps per sync) | Чаще sync → меньше drift, больше Internet trafиc | Меньше overhead, больше drift между хостами |
| `MAX_ROUNDS` | Меньше convergence | Больше времени |

Рекомендация: `INTERVAL=500` (≈ 6 минут на 8C2), `MAX_ROUNDS=100` →
50000 effective steps × 3 хоста ≈ 150000 worker-steps.

## Bottleneck — SCP bandwidth

Размер модели = N_params × 4 (fp32). Для PIR 189M это **756 MB**.
- При 100 Mbps Internet: 60 секунд на upload + 60 на download = **2 мин/sync round**
- При 1 Gbps: 6 + 6 = **12 секунд/sync round**

Если INTERVAL = 500 steps × 60s/step = 30 минут intra-compute, sync overhead
= 4-10% — приемлемо. Если INTERVAL = 50 steps → sync overhead ~40% — плохо.

## Setup

### 1. SSH keys cross-server

На каждом worker генерируем pubkey и заливаем на master:
```bash
# на worker
ssh-keygen -t ed25519 -f ~/.ssh/mh_key -N ""
ssh-copy-id -i ~/.ssh/mh_key.pub master_user@master_host
```

И наоборот — master нужно тоже залить ключ workers (для broadcast back).

### 2. Зависимости

На КАЖДОМ хосте:
```bash
apt-get install -y python3 python3-numpy openssh-client
```

### 3. Сборка PromeTorch (одинаковый commit на ВСЕХ хостах)

```bash
git pull && rm -rf build_elbrus && ./scripts/build-elbrus.sh
```

### 4. Broadcast стартового checkpoint'a

```bash
# на master
scp ~/nanogpt_tinystories/checkpoints/pir_fused_step_200.bin \
    user@worker1:~/nanogpt_tinystories/checkpoints/
scp ~/nanogpt_tinystories/checkpoints/pir_fused_step_200.bin \
    user@worker2:~/nanogpt_tinystories/checkpoints/
```

### 5. Запуск

**На master (8C2):**
```bash
PT_MH_ROLE=master \
PT_MH_PEERS="paperclipdnb@<host16c>,paperclipdnb@<host4c>" \
PT_MH_WEIGHTS="2.4,1.0,0.3" \
PT_MH_SSH_OPTS="-i ~/.ssh/mh_key -o StrictHostKeyChecking=no" \
PT_MH_INTERVAL=500 \
PT_MH_MAX_ROUNDS=100 \
./scripts/run_multihost.sh > mh_master.log 2>&1 &
```

**На каждом worker:**
```bash
PT_MH_ROLE=worker \
PT_MH_MASTER=user@<host8c2> \
PT_MH_SSH_OPTS="-i ~/.ssh/mh_key -o StrictHostKeyChecking=no" \
PT_MH_INTERVAL=500 \
PT_MH_MAX_ROUNDS=100 \
./scripts/run_multihost.sh > mh_worker.log 2>&1 &
```

### 6. Мониторинг

```bash
tail -f mh_master.log    # master
tail -f mh_worker.log    # worker
```

Каждый `round N` логируется с timestamp'ом. Sync round длительность видна
в строке `[run_mh] sync OK — current.bin обновлён`.

## Convergence

Hierarchical Local SGD theoretically достигает того же качества что и
синхронный SGD при больших batch'ах, если:
1. `INTERVAL` ≪ дистанция до convergence
2. `lr` понижен compared к local SGD (на N_workers фактор)
3. **Веса нормализованы** по compute (наше weighted averaging)

См. [Yu et al. 2019 "Parallel Restarted SGD with Faster Convergence and
Less Communication"](https://arxiv.org/abs/1807.06629).

## Failure modes

| Симптом | Причина | Решение |
|---------|---------|---------|
| `Timeout waiting for peer` | Worker упал/завис | Master продолжит с тем что есть (graceful degrade) |
| `SCP failed` | SSH connectivity | Проверить ключи + `nc -z host port` |
| Loss растёт после sync | weighted коэффициенты или drift между хостами | Уменьшить `INTERVAL` или `lr` |
| Bit divergence между host'ами | Разные ISA (v3 vs v6) | Это нормально для fp32, average сглаживает |

## Что НЕ реализовано (future work)

- TCP socket-based sync (быстрее SCP, ~10× меньше latency)
- Quantized gradient sync (INT8 → 4× меньше трафик)
- Async pipeline (workers не ждут master, продолжают, master догоняет)
- Hot restart при потере worker'а (сейчас нужен restart всей кампании)
- Multi-master (ring all-reduce)
