#!/bin/bash
# Start PromeServe with mistral-7b in foreground (caller wraps with nohup setsid).
cd /home/paperclipdnb/promethorch
exec ./build_elbrus/promeserve/promeserve --port 11500 --device cpu \
    --model /home/paperclipdnb/gguf_models/mistral-7b-Q4_K_M.gguf
