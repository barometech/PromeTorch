Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found"
}

$vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) {
    throw "Visual Studio C++ Build Tools not found"
}

$vcvars = Join-Path $vsInstall "VC\Auxiliary\Build\vcvarsall.bat"
if (-not (Test-Path $vcvars)) {
    throw "vcvarsall.bat not found: $vcvars"
}

New-Item -ItemType Directory -Force -Path ".\build\smoke" | Out-Null

$compile = @"
call "$vcvars" x64 ^
&& cl /nologo /EHsc /std:c++17 /MD /I. smoke\prometorch_linear_model_smoke.cpp build\c10.lib build\aten_cpu.lib build\torch_autograd.lib /Fe:build\smoke\prometorch_linear_model_smoke.exe ^
&& cl /nologo /EHsc /std:c++17 /MD /I. smoke\prometorch_linear_model_params_persistence_smoke.cpp build\c10.lib build\aten_cpu.lib build\torch_autograd.lib /Fe:build\smoke\prometorch_linear_model_params_persistence_smoke.exe ^
&& cl /nologo /EHsc /std:c++17 /MD smoke\load_linear_model_params_smoke.cpp /Fe:build\smoke\load_linear_model_params_smoke.exe ^
&& cl /nologo /EHsc /std:c++17 /MD smoke\linear_model_inference_smoke.cpp /Fe:build\smoke\linear_model_inference_smoke.exe ^
&& cl /nologo /EHsc /std:c++17 /MD /I. smoke\prometorch_linear_model_e2e_smoke.cpp build\c10.lib build\aten_cpu.lib build\torch_autograd.lib /Fe:build\smoke\prometorch_linear_model_e2e_smoke.exe ^
&& cl /nologo /EHsc /std:c++17 /MD smoke\linear_model_batch_inference_smoke.cpp /Fe:build\smoke\linear_model_batch_inference_smoke.exe
"@

cmd /d /s /c $compile
if ($LASTEXITCODE -ne 0) {
    throw "compile failed"
}

$tests = @(
    ".\build\smoke\prometorch_linear_model_smoke.exe",
    ".\build\smoke\prometorch_linear_model_params_persistence_smoke.exe",
    ".\build\smoke\load_linear_model_params_smoke.exe",
    ".\build\smoke\linear_model_inference_smoke.exe",
    ".\build\smoke\prometorch_linear_model_e2e_smoke.exe",
    ".\build\smoke\linear_model_batch_inference_smoke.exe"
)

foreach ($test in $tests) {
    Write-Host ""
    Write-Host "=== RUN $test ==="
    & $test

    if ($LASTEXITCODE -ne 0) {
        throw "FAILED: $test"
    }
}

Write-Host ""
Write-Host "OK: all linear model smoke tests passed"
