#!/bin/bash
cd /home/paperclipdnb/promethorch
pkill -9 -f test_gguf_inference 2>/dev/null
sleep 2
touch examples/gguf/test_gguf_inference.cpp
rm -f /tmp/build3.done
cmake --build build_elbrus --target test_gguf_inference -j 16 > /tmp/build3.log 2>&1
echo DONE > /tmp/build3.done
