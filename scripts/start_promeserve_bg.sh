#!/bin/bash
# Start PromeServe в background, выживает после ssh disconnect.
cd /home/paperclipdnb/promethorch
pkill -9 -f promeserve 2>/dev/null
pkill -9 -f test_gguf 2>/dev/null
sleep 3
rm -f /dev/shm/prometorch_ddp_*

nohup setsid ./build_elbrus/promeserve/promeserve \
    --port 11500 --device cpu \
    --model /home/paperclipdnb/gguf_models/mistral-7b-Q4_K_M.gguf \
    </dev/null >/tmp/pserve.log 2>&1 & disown

# Block until /api/version responds
for i in {1..60}; do
    sleep 5
    if curl -s -m 2 http://127.0.0.1:11500/api/version 2>/dev/null | grep -q version; then
        echo "PromeServe ready after $((i*5))s"
        echo READY > /tmp/pserve.ready
        exit 0
    fi
done
echo "PromeServe failed to start in 5min"
exit 1
