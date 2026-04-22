#!/bin/bash
# Launch PromeServe on Elbrus — single-process, all 24 threads, NUMA interleave.
# Model: qwen3-4b Q4_K_M. Port 11434 (Ollama default). CPU device.
#
# After this exits, server is running in background; logs in /tmp/promeserve.log.

set -eu
cd "$HOME/promethorch"

# Kill any prior instance
pkill -f promeserve 2>/dev/null || true
sleep 2

loginctl enable-linger user 2>/dev/null || true

# Use tmux detached session — survives SSH disconnect reliably on Elbrus.
tmux kill-session -t promeserve 2>/dev/null || true
tmux new-session -d -s promeserve \
    "cd ~/promethorch && \
     OMP_NUM_THREADS=24 OMP_PLACES=cores OMP_PROC_BIND=close \
     numactl --interleave=all \
     ./build_elbrus/promeserve/promeserve \
         --port 11434 --device cpu \
         --model $HOME/gguf_models/qwen3-4b-Q4_K_M.gguf \
     2>&1 | tee /tmp/promeserve.log"

echo "Started tmux session 'promeserve'."
echo "Attach:    tmux attach -t promeserve"
echo "Logs:      tail -f /tmp/promeserve.log"
echo "Check port: ss -tln | grep 11434  (wait ~20s for model load)"
