#!/bin/bash
exec /home/paperclipdnb/llama.cpp/build/bin/llama-server \
    -m /home/paperclipdnb/gguf_models/qwen3-4b-Q4_K_M.gguf \
    -t 32 --host 127.0.0.1 --port 18099 -c 2048
