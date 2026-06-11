# PromeTorch Windows AMD CPU smoke status

## Current branch

windows-amd-cpu-build

## Repository

https://github.com/egorKara/PromeTorch

## Confirmed platform

- Windows
- MSVC
- AMD CPU
- external smoke-tests built with /MD

## Confirmed libraries

- build/c10.lib
- build/aten_cpu.lib
- build/torch_autograd.lib

## Confirmed smoke levels

| Tag | Meaning |
|---|---|
| windows-amd-cpu-build-ok | Windows AMD CPU build works |
| windows-amd-cpu-api-smoke-ok | external API smoke works |
| windows-amd-cpu-api-smoke-md-ok | external API smoke works with /MD |
| windows-amd-cpu-sum-smoke-ok | sum/reduction works |
| windows-amd-cpu-elementwise-smoke-ok | elementwise arithmetic works |
| windows-amd-cpu-matmul-smoke-ok | matmul works |
| windows-amd-cpu-autograd-smoke-ok | autograd/backward/grad works |
| windows-amd-cpu-training-loop-smoke-ok | minimal training loop works |
| windows-amd-cpu-trained-weight-file-persistence-smoke-ok | trained scalar can be saved/loaded as file |
| windows-amd-cpu-prometorch-trained-weight-persistence-smoke-ok | PromeTorch-trained scalar can be saved/loaded |
| windows-amd-cpu-linear-model-smoke-ok | linear model y = x*w + b trains |
| windows-amd-cpu-linear-model-params-persistence-smoke-ok | trained w,b can be saved/loaded |
| windows-amd-cpu-linear-model-inference-smoke-ok | loaded w,b can run single inference |
| windows-amd-cpu-linear-model-e2e-smoke-ok | train/save/load/inference works in one exe |
| windows-amd-cpu-linear-model-batch-inference-smoke-ok | loaded model works on batch inputs |
| windows-amd-cpu-linear-model-smoke-runner-ok | linear smoke runner exists |

## Linear smoke runner

Run from PowerShell:

    Set-Location C:\PromeTorch_Local
    powershell -ExecutionPolicy Bypass -File .\smoke\run_linear_model_smokes.ps1

Expected final line:

    OK: all linear model smoke tests passed

## Current confirmed ML chain

Tensor API
-> autograd/backward/grad
-> training loop
-> train linear model
-> save w,b
-> load w,b
-> inference
-> batch inference
