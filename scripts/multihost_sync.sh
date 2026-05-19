#!/bin/bash
# multihost_sync.sh — один sync round Hierarchical Local SGD между Эльбрус-хостами.
#
# КАК ЭТО РАБОТАЕТ:
#   Каждый хост запускает train_pir_elbrus с --multihost-sync-cmd указывающим
#   на этот скрипт. Тренировка делает 4-proc intra-host Local SGD как обычно,
#   а каждые K_GLOBAL шагов зовёт этот скрипт. Скрипт делает SCP-AllReduce:
#
#   1. (Worker) Дампит свои weights в /tmp/pt_mh/local.bin
#   2. (Worker) SCP /tmp/pt_mh/local.bin → master:/tmp/pt_mh/peer_$HOSTNAME.bin
#   3. (Master) Ждёт всех peer'ов (peer_*.bin), запускает multihost_avg.py
#   4. (Master) Пишет /tmp/pt_mh/avg.bin
#   5. (Worker) SCP master:/tmp/pt_mh/avg.bin → /tmp/pt_mh/local.bin
#   6. (Both) train_pir_elbrus загружает /tmp/pt_mh/local.bin как новый стартовый ckpt
#
# ENV переменные:
#   PT_MH_ROLE         master | worker
#   PT_MH_MASTER       hostname/IP мастера (только для worker)
#   PT_MH_PEERS        comma-separated hostname/IP всех воркеров (только master).
#                      Пример: PT_MH_PEERS="user1@h1,user2@h2"
#   PT_MH_WEIGHTS      comma-separated веса (по compute capacity).
#                      Порядок: master_self,peer1,peer2,... Пример: "2.4,1.0,0.3"
#   PT_MH_DIR          рабочая директория (default: /tmp/pt_mh)
#   PT_MH_TIMEOUT      seconds wait для peer upload (default 1800 = 30 мин)
#   PT_MH_SSH_OPTS     доп. опции SSH (например '-i key.pem -p 2132')
#   PT_MH_KEEP         '1' = не удалять файлы после round (для дебага)
#
# Usage (вызывается из train_pir_elbrus):
#   ./multihost_sync.sh <round_num> <local_weights.bin>
#
# Exit 0 = OK, weights в /tmp/pt_mh/local.bin обновлены.
# Exit != 0 = ошибка sync; вызывающий код может игнорировать и продолжать с old weights.

set -uo pipefail

ROUND="${1:-0}"
LOCAL_CKPT="${2:-}"

ROLE="${PT_MH_ROLE:-worker}"
MH_DIR="${PT_MH_DIR:-/tmp/pt_mh}"
TIMEOUT="${PT_MH_TIMEOUT:-1800}"
SSH_OPTS="${PT_MH_SSH_OPTS:-}"

mkdir -p "$MH_DIR"

log() { echo "[mh_sync round=$ROUND role=$ROLE] $*" >&2; }

if [ -z "$LOCAL_CKPT" ] || [ ! -f "$LOCAL_CKPT" ]; then
    log "ERROR: local checkpoint '$LOCAL_CKPT' не существует"
    exit 1
fi

# Размер для sanity. Все хосты должны давать одинаковый.
LOCAL_SIZE=$(stat -c%s "$LOCAL_CKPT")
log "local size=$LOCAL_SIZE bytes ($(( LOCAL_SIZE / 1048576 )) MB)"

cp "$LOCAL_CKPT" "$MH_DIR/local.bin"

# ============================================================================
# WORKER role: SCP local → master, wait for avg, SCP back.
# ============================================================================
if [ "$ROLE" = "worker" ]; then
    MASTER="${PT_MH_MASTER:?PT_MH_MASTER не задан для worker}"
    HOST_TAG="$(hostname)_${RANDOM}_$$"

    log "uploading to master ($MASTER)..."
    t0=$(date +%s)
    if ! scp $SSH_OPTS "$MH_DIR/local.bin" "$MASTER:$MH_DIR/peer_${HOST_TAG}.bin"; then
        log "ERROR: SCP upload failed"
        exit 2
    fi
    # Маркер готовности (атомарный rename на master'е).
    ssh $SSH_OPTS "$MASTER" "mv $MH_DIR/peer_${HOST_TAG}.bin $MH_DIR/peer_${HOST_TAG}_ready.bin" || true
    log "uploaded in $(( $(date +%s) - t0 ))s. waiting for avg.bin..."

    # Polling: ждём пока master выложит /tmp/pt_mh/round_$ROUND/avg.bin
    AVG_REMOTE="$MH_DIR/avg_round_${ROUND}.bin"
    t0=$(date +%s)
    while true; do
        if ssh $SSH_OPTS "$MASTER" "test -f $AVG_REMOTE" 2>/dev/null; then
            break
        fi
        now=$(date +%s)
        if [ $(( now - t0 )) -gt "$TIMEOUT" ]; then
            log "ERROR: timeout $TIMEOUT с — master не выложил avg.bin"
            exit 3
        fi
        sleep 5
    done

    log "downloading avg.bin..."
    if ! scp $SSH_OPTS "$MASTER:$AVG_REMOTE" "$MH_DIR/local.bin"; then
        log "ERROR: SCP download failed"
        exit 4
    fi
    log "OK — averaged weights ($(stat -c%s "$MH_DIR/local.bin") bytes) в $MH_DIR/local.bin"
    cp "$MH_DIR/local.bin" "$LOCAL_CKPT"
    exit 0
fi

# ============================================================================
# MASTER role: ждать всех peer_*_ready.bin, усреднить, разложить avg.bin.
# ============================================================================
if [ "$ROLE" = "master" ]; then
    PEERS="${PT_MH_PEERS:?PT_MH_PEERS не задан для master}"
    IFS=',' read -ra PEER_LIST <<< "$PEERS"
    N_PEERS=${#PEER_LIST[@]}

    log "ожидаю $N_PEERS peer'ов upload..."
    t0=$(date +%s)
    while true; do
        ready=$(ls "$MH_DIR"/peer_*_ready.bin 2>/dev/null | wc -l)
        if [ "$ready" -ge "$N_PEERS" ]; then
            log "все $ready peer(ов) готовы за $(( $(date +%s) - t0 ))s"
            break
        fi
        now=$(date +%s)
        if [ $(( now - t0 )) -gt "$TIMEOUT" ]; then
            log "WARNING: timeout — только $ready/$N_PEERS peer(ов). продолжаю с тем что есть."
            break
        fi
        sleep 5
    done

    # Собираем inputs: master local + все peer_*_ready
    INPUTS=("$MH_DIR/local.bin")
    for f in "$MH_DIR"/peer_*_ready.bin; do
        [ -f "$f" ] && INPUTS+=("$f")
    done

    if [ ${#INPUTS[@]} -lt 2 ]; then
        log "ERROR: <2 inputs, нечего усреднять"
        exit 5
    fi

    log "averaging ${#INPUTS[@]} files..."
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    AVG_LOCAL="$MH_DIR/avg_round_${ROUND}.bin"

    WEIGHTED_ARG=""
    if [ -n "${PT_MH_WEIGHTS:-}" ]; then
        WEIGHTED_ARG="--weighted $PT_MH_WEIGHTS"
    fi

    if ! python3 "$SCRIPT_DIR/multihost_avg.py" $WEIGHTED_ARG "$AVG_LOCAL" "${INPUTS[@]}"; then
        log "ERROR: averaging failed"
        exit 6
    fi

    log "avg ready: $AVG_LOCAL ($(stat -c%s "$AVG_LOCAL") bytes)"
    # Master тоже использует усреднённые weights
    cp "$AVG_LOCAL" "$LOCAL_CKPT"

    # Очистка peer_*.bin (master'у больше не нужны)
    if [ "${PT_MH_KEEP:-0}" != "1" ]; then
        rm -f "$MH_DIR"/peer_*_ready.bin
    fi

    log "OK — master завершил sync round $ROUND"
    exit 0
fi

log "ERROR: неизвестная роль '$ROLE' (master|worker)"
exit 1
