#!/bin/bash
# Run GGUF inference on Эльбрус через PromeTorch — Tensor Parallel.
#
# Адаптивный: автоматически определяет количество NUMA-нод (чипов) и
# распределяет ранки по ним. На 4-чиповом 8C2 запустит TP-4, на 1-чиповом
# 8СВ или однопроцессорном dev-сервере fallback на 1-proc через
# run_1proc_elbrus.sh.
#
# Usage:
#   ./run_tp_elbrus.sh [--greedy|--sample] [prompt]
#
# Env:
#   PT_MODEL          путь до GGUF (default: ~/gguf_models/qwen3-4b-Q4_K_M.gguf)
#   PT_NPROCS         override автоопределения (1, 2, 4, ...)
#   PT_OMP_PER_RANK   override threads/rank (default: cores_total / nprocs)
#   PT_MASTER_PORT    SHM AllReduce master port (default: 29500)
#   PT_MAX_TOK        max generation tokens (default: 100)
#   PT_CHAT=0         выключить chat template (raw completion)
#   PT_Q8_SOA         override (default 1; 0 = legacy decode-only path)

set -u
cd ~/promethorch

# Safety: systemd must not kill processes on SSH disconnect.
loginctl enable-linger "$USER" 2>/dev/null || true

# Guard: PT_PIN_THREADS=1 в TP-режиме катастрофически ломает производительность.
# ThreadPool маппит worker_id на абсолютные CPU ID, а ранки 1+ уже numactl
# --cpunodebind'ы на разные NUMA-ноды. Падение до 1.4 tok/s (bisect 2026-04-30).
if [ "${PT_PIN_THREADS:-}" = "1" ]; then
    echo "ERROR: PT_PIN_THREADS=1 в TP-режиме ломает NUMA-binding и режет tok/s в ~7 раз." >&2
    echo "       Сними этот env и запусти снова. См. BENCH_ELBRUS.md." >&2
    exit 1
fi

# ============================================================================
# Auto-detect NUMA topology
# ============================================================================
detect_numa_nodes() {
    if command -v numactl >/dev/null 2>&1; then
        # numactl --hardware пишет "available: N nodes (0-...)"
        local n=$(numactl --hardware 2>/dev/null | awk '/available:/ {print $2}')
        if [ -n "$n" ] && [ "$n" -gt 0 ]; then
            echo "$n"; return
        fi
    fi
    # Fallback: посчитать /sys/devices/system/node/node*
    local n=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "$n"; return
    fi
    echo "1"
}

detect_total_cores() {
    if command -v nproc >/dev/null 2>&1; then
        nproc; return
    fi
    grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "1"
}

NUMA_NODES=$(detect_numa_nodes)
TOTAL_CORES=$(detect_total_cores)
NPROCS="${PT_NPROCS:-$NUMA_NODES}"
OMP_PER_RANK="${PT_OMP_PER_RANK:-$(( TOTAL_CORES / NPROCS ))}"
[ "$OMP_PER_RANK" -lt 1 ] && OMP_PER_RANK=1

# Sanity: TP requires ≥2 ranks; fallback to 1-proc on single-chip systems.
if [ "$NPROCS" -lt 2 ]; then
    echo "[run_tp_elbrus] Detected $NUMA_NODES NUMA node(s), $TOTAL_CORES cores total."
    echo "[run_tp_elbrus] Single-chip система — fallback на run_1proc_elbrus.sh."
    echo "[run_tp_elbrus] Если хочешь принудительно TP — укажи PT_NPROCS=2 или больше."
    exec "$(dirname "$0")/run_1proc_elbrus.sh" "$@"
fi

MODEL="${PT_MODEL:-$HOME/gguf_models/qwen3-4b-Q4_K_M.gguf}"
PROMPT="${2:-Write a short haiku about artificial intelligence}"
MODE="${1:---greedy}"
MAX_TOK="${PT_MAX_TOK:-100}"
MASTER_PORT="${PT_MASTER_PORT:-29500}"

BIN="./build_elbrus/examples/gguf/test_gguf_inference"
if [ ! -x "$BIN" ]; then
    echo "ERR: $BIN not built. Run: ./scripts/build-elbrus.sh"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERR: $MODEL not found. Установи через PT_MODEL=/path/to/model.gguf"
    exit 1
fi

mkdir -p run_logs

echo "=== PromeTorch TP-$NPROCS inference на Эльбрусе ==="
echo "    NUMA nodes: $NUMA_NODES · total cores: $TOTAL_CORES · OMP/rank: $OMP_PER_RANK"
echo "    Model:  $MODEL"
echo "    Prompt: $PROMPT"
echo "    Mode:   $MODE · max_tokens: $MAX_TOK · master_port: $MASTER_PORT"
date +"Start: %F %T"

# Очистка возможных stale SHM-слотов от прерванных прошлых запусков.
rm -f /dev/shm/prometorch_ddp_${MASTER_PORT} 2>/dev/null

# BUG-1 fix: --chat применяет apply_chat_template. PT_CHAT=0 для raw completion.
CHAT_FLAG=""
if [ "${PT_CHAT:-1}" = "1" ]; then CHAT_FLAG="--chat"; fi

# Launch ranks 0..N-1, по одному на NUMA-ноду.
# --cpunodebind + --membind: per-rank local DDR bandwidth = N× aggregate vs 1 NUMA.
# --membind и --preferred конфликтуют ("Conflicting policies") — только --membind.
# PT_NUMA_REPLICATE=0: на >2 чипах работает; на 1-чипе нет смысла (но мы тут уже TP).
PIDS=()
for (( rank=0; rank < NPROCS; rank++ )); do
    PT_NO_NUMA_POOL=1 \
    OMP_NUM_THREADS=$OMP_PER_RANK \
    PT_NUMA_REPLICATE=0 \
    PT_DDP_SHM=1 \
    PT_Q8_SOA="${PT_Q8_SOA:-1}" \
    numactl --cpunodebind=$rank --membind=$rank \
        "$BIN" "$MODEL" \
        --nprocs $NPROCS --rank $rank \
        --master-addr 127.0.0.1 --master-port $MASTER_PORT \
        --max-tokens $MAX_TOK $MODE $CHAT_FLAG \
        "$PROMPT" \
        > run_logs/tp${NPROCS}_rank${rank}.log 2>&1 &
    PIDS+=($!)
    echo "Rank $rank launched PID=${PIDS[-1]} (cpunodebind=$rank, OMP=$OMP_PER_RANK)"
done

wait "${PIDS[@]}"
date +"End: %F %T"
echo
echo "=== Rank 0 output ==="
cat run_logs/tp${NPROCS}_rank0.log
