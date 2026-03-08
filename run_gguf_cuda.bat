@echo off
set "PATH=C:\ProgramData\anaconda3\Library\bin;C:\Users\paper\Desktop\promethorch\build_gguf_cuda;%PATH%"
cd /d C:\Users\paper\Desktop\promethorch
echo RUNNING... > run_gguf_cuda_log.txt
build_gguf_cuda\examples\gguf\test_gguf_inference.exe qwen3:4b --device cuda --greedy --max-tokens 50 "Once upon a time" >> run_gguf_cuda_log.txt 2>&1
echo EXIT=%ERRORLEVEL% >> run_gguf_cuda_log.txt
echo DONE >> run_gguf_cuda_log.txt
