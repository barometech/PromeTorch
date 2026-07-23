#!/bin/bash
# ============================================================================
# train_local_sgd.sh — 4-процессный Local SGD PIR на Эльбрус E8C2 (4 NUMA-узла)
# ============================================================================
# Проверенная схема: 1 процесс на NUMA-узел (8 ядер), node-local память,
# EML_MT — 8 OMP-потоков внутри процесса, веса усредняются через /dev/shm
# каждые GRAD_ACCUM шагов (Local SGD, НЕ gradient AllReduce — дешевле на
# порядки по трафику через межчиповый интерконнект).
#
# Использование:
#   ./train_local_sgd.sh                      # старт с нуля
#   RESUME=1 ./train_local_sgd.sh             # продолжить с последнего чекпоинта
#   DATA=data/my.tokens VOCAB=16000 ./train_local_sgd.sh
#
# ВАЖНО перед запуском:
#   1. loginctl enable-linger $USER   — иначе systemd убьёт процессы при
#      SSH-disconnect (скрипт делает это сам, но после reboot сервера —
#      выполнить заново!). От reboot спасает только RESUME=1.
#   2. ldd $BIN | grep eml_mt         — если пусто, сборка ушла в сериальный
#      TudaBLAS-фолбэк (~10-25x медленнее). См. docs/BUILD_ELBRUS.md,
#      секция «EML не линкуется».
# ============================================================================
set -u

# --- параметры (переопределяются окружением) ---
BIN=${BIN:-./build_elbrus/examples/pir/train_pir_elbrus}
DATA=${DATA:-data/russian_16k.tokens}
VOCAB=${VOCAB:-16000}
N_EMBD=${N_EMBD:-256}
N_LAYERS=${N_LAYERS:-4}
BLOCK=${BLOCK:-256}
BATCH=${BATCH:-2}
MAX_STEPS=${MAX_STEPS:-10000}
GRAD_ACCUM=${GRAD_ACCUM:-10}
LR=${LR:-0.0006}
SAVE_DIR=${SAVE_DIR:-checkpoints_pir}
LOG_DIR=${LOG_DIR:-logs}
NPROCS=${NPROCS:-4}          # = числу NUMA-узлов (numactl --hardware)
THREADS=${THREADS:-8}        # ядер на узел
RESUME=${RESUME:-0}

# --- проверки ---
if [ ! -x "$BIN" ]; then echo "нет бинарника: $BIN (собери train_pir_elbrus)"; exit 1; fi
if [ ! -f "$DATA" ]; then echo "нет данных: $DATA (см. prepare_bpe.py)"; exit 1; fi
if ! ldd "$BIN" 2>/dev/null | grep -q eml; then
    echo "ПРЕДУПРЕЖДЕНИЕ: в ldd нет libeml — GEMM будет сериальным (~10-25x медленнее)."
    echo "Фикс: cmake -U '_PT_EML*' -DPT_EML_RUNTIME_OK=ON . && make train_pir_elbrus"
fi
loginctl enable-linger "$USER" 2>/dev/null || true

mkdir -p "$SAVE_DIR" "$LOG_DIR"

# --- resume: подхватить последний чекпоинт ---
LOAD_ARGS=""
if [ "$RESUME" = "1" ]; then
    CKPT=$(ls -v "$SAVE_DIR"/pir_fused_step_*.bin 2>/dev/null | tail -1)
    if [ -n "$CKPT" ]; then
        STEP=$(basename "$CKPT" | grep -oE '[0-9]+')
        LOAD_ARGS="--load $CKPT --start_step $STEP"
        echo "resume: $CKPT (step $STEP; LR-warmup первые 200 шагов после резюма)"
    else
        echo "RESUME=1, но чекпоинтов в $SAVE_DIR нет — старт с нуля"
    fi
fi

# --- чистим стейл shared-memory от прошлых запусков GradSync ---
rm -f /dev/shm/pir_* 2>/dev/null

# --- запуск: по процессу на NUMA-узел ---
for node in $(seq 0 $((NPROCS-1))); do
    PT_NO_NUMA_POOL=1 OMP_NUM_THREADS=$THREADS OMP_PLACES=cores OMP_PROC_BIND=close \
    numactl --cpunodebind=$node --preferred=$node \
    "$BIN" \
        --fused --full --nprocs $NPROCS --rank $node \
        --vocab_size $VOCAB --data "$DATA" \
        --n_embd $N_EMBD --n_layers $N_LAYERS --block_size $BLOCK --batch_size $BATCH \
        --max_steps $MAX_STEPS --grad_accum $GRAD_ACCUM --lr $LR \
        --log_interval 50 --gen_interval 99999 \
        --save_interval 500 --save_dir "$SAVE_DIR" \
        --seed $((42 + node)) \
        $LOAD_ARGS \
        > "$LOG_DIR/pir_rank$node.log" 2>&1 &
    sleep 8   # стаггер: GradSync init по очереди
done

echo "запущено $NPROCS процессов; логи: $LOG_DIR/pir_rank*.log"
echo "прогресс: tail -f $LOG_DIR/pir_rank0.log | grep '^step'"
wait
