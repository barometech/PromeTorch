#!/bin/bash
# run_multihost.sh — top-level launcher для Hierarchical Local SGD между
# гетерогенными Эльбрус-хостами (4C/8C/8C2/16C/...).
#
# АРХИТЕКТУРА:
#   Каждый хост запускает 4-proc Local SGD INTRA-host (как run_tp_elbrus.sh).
#   Каждые K_GLOBAL шагов делает SCP-AllReduce между хостами через
#   multihost_sync.sh. Веса усредняются взвешенно по compute capacity.
#
# ЗАПУСК:
#
#   На master хосте (наш мощный 8C2):
#     PT_MH_ROLE=master \
#     PT_MH_PEERS="user@host16c,user@host4c" \
#     PT_MH_WEIGHTS="2.4,1.0,0.3" \
#     PT_MH_INTERVAL=500 \
#     PT_MH_MAX_ROUNDS=10 \
#     ./scripts/run_multihost.sh
#
#   На каждом worker:
#     PT_MH_ROLE=worker \
#     PT_MH_MASTER=user@host8c2 \
#     PT_MH_INTERVAL=500 \
#     ./scripts/run_multihost.sh
#
# Все хосты должны иметь:
#   - Одинаковый PromeTorch checkout (один commit hash, иначе ABI mismatch)
#   - Одинаковый model config (vocab, dim, layers)
#   - Стартовый checkpoint (PT_MH_INIT_CKPT — broadcast'нутый master'ом ДО запуска)
#   - SSH key-based connectivity master ↔ workers (key прописан в PT_MH_SSH_OPTS)
#   - python3 + numpy (для multihost_avg.py)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MH_DIR="${PT_MH_DIR:-/tmp/pt_mh}"

ROLE="${PT_MH_ROLE:?PT_MH_ROLE не задан (master|worker)}"
INTERVAL="${PT_MH_INTERVAL:-500}"
MAX_ROUNDS="${PT_MH_MAX_ROUNDS:-100}"
INIT_CKPT="${PT_MH_INIT_CKPT:-$HOME/nanogpt_tinystories/checkpoints/pir_fused_step_200.bin}"
TRAIN_BIN="${PT_MH_TRAIN_BIN:-$HOME/PromeTorch/build_elbrus/examples/pir/train_pir_elbrus}"
DATA="${PT_MH_DATA:-$HOME/nanogpt_tinystories/tinystories.txt}"

# Intra-host config (передаётся в train_pir_elbrus)
INTRA_NPROCS="${PT_MH_INTRA_NPROCS:-4}"
INTRA_OMP="${PT_MH_INTRA_OMP:-$(( $(nproc) / INTRA_NPROCS ))}"
BATCH_SIZE="${PT_MH_BATCH_SIZE:-4}"
GRAD_ACCUM="${PT_MH_GRAD_ACCUM:-10}"
LR="${PT_MH_LR:-6e-4}"

mkdir -p "$MH_DIR"

log() { echo "[run_mh role=$ROLE] $(date '+%H:%M:%S') $*" >&2; }

log "==== Multi-host Local SGD ===="
log "role=$ROLE, interval=$INTERVAL steps/round, max_rounds=$MAX_ROUNDS"
log "intra-host: $INTRA_NPROCS procs × $INTRA_OMP threads, batch=$BATCH_SIZE × grad_accum=$GRAD_ACCUM"
log "init_ckpt=$INIT_CKPT"

# Проверки.
[ -f "$INIT_CKPT" ] || { log "ERROR: $INIT_CKPT не существует"; exit 1; }
[ -x "$TRAIN_BIN" ] || { log "ERROR: $TRAIN_BIN не исполняемый"; exit 1; }
[ -f "$DATA" ]      || { log "ERROR: $DATA не существует"; exit 1; }

# Стартовый ckpt → local working copy.
CURRENT_CKPT="$MH_DIR/current.bin"
cp "$INIT_CKPT" "$CURRENT_CKPT"

# Linger чтобы тренировка пережила SSH disconnect.
loginctl enable-linger "$USER" 2>/dev/null || true

ROUND=0
while [ "$ROUND" -lt "$MAX_ROUNDS" ]; do
    ROUND=$((ROUND + 1))
    log "==== round $ROUND/$MAX_ROUNDS START ===="

    # 1. Intra-host тренировка INTERVAL steps от current.bin
    cd "$(dirname "$TRAIN_BIN")"
    export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PT_NO_NUMA_POOL=1
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    export OMP_NUM_THREADS=$INTRA_OMP

    PIDS=()
    for (( rank=0; rank<INTRA_NPROCS; rank++ )); do
        LOG_FILE="$MH_DIR/round${ROUND}_proc${rank}.log"
        numactl --cpunodebind=$rank --preferred=$rank \
            "$TRAIN_BIN" \
            --fused --full --batch_size $BATCH_SIZE \
            --rank $rank --nprocs $INTRA_NPROCS \
            --max_steps $INTERVAL --log_interval 50 \
            --save_interval $INTERVAL \
            --save_dir "$MH_DIR/round${ROUND}_ckpts" \
            --grad_accum $GRAD_ACCUM --lr $LR \
            --load "$CURRENT_CKPT" \
            --data "$DATA" \
            > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
    done
    log "запущено $INTRA_NPROCS proc, PIDs: ${PIDS[*]}"

    # Ждать всех.
    for pid in "${PIDS[@]}"; do
        wait "$pid" || log "WARNING: pid $pid вышел с не-0 кодом"
    done
    log "intra-host round $ROUND завершён за $(grep -E 'tok/s' "$MH_DIR/round${ROUND}_proc0.log" | tail -1)"

    # 2. Saved ckpt → local.bin для sync.
    SAVED="$MH_DIR/round${ROUND}_ckpts/pir_fused_step_${INTERVAL}.bin"
    if [ ! -f "$SAVED" ]; then
        log "ERROR: training не сохранил $SAVED — пропускаю sync, продолжаю с current"
        continue
    fi
    cp "$SAVED" "$CURRENT_CKPT"

    # 3. Sync round — SCP AllReduce.
    log "запускаю multihost_sync.sh round $ROUND..."
    if "$SCRIPT_DIR/multihost_sync.sh" "$ROUND" "$CURRENT_CKPT"; then
        log "sync OK — current.bin обновлён до averaged weights"
    else
        log "WARNING: sync FAILED — продолжаю с локальными weights (no average)"
    fi

    log "==== round $ROUND/$MAX_ROUNDS DONE ===="
done

log "==== ВСЕ $MAX_ROUNDS round завершено ===="
log "Финальный checkpoint: $CURRENT_CKPT"
