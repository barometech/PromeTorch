#!/bin/bash
cd /home/paperclipdnb/promethorch
touch examples/gguf/test_gguf_inference.cpp torch/io/gguf_model.h
cmake --build build_elbrus --target test_gguf_inference -j 16
echo BUILD_EXIT=$?
stat -c "%y" build_elbrus/examples/gguf/test_gguf_inference
