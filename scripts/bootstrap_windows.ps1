$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

if (!(Test-Path ".venv")) {
    py -3 -m venv .venv
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

$llamaServer = Join-Path $projectRoot "vendor\llama.cpp\build\bin\Release\llama-server.exe"
$llamaOnPath = Get-Command "llama-server.exe" -ErrorAction SilentlyContinue
if (!(Test-Path $llamaServer) -and $null -eq $llamaOnPath) {
    New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "vendor") | Out-Null
    if (!(Test-Path (Join-Path $projectRoot "vendor\llama.cpp\.git"))) {
        git clone https://github.com/ggml-org/llama.cpp.git (Join-Path $projectRoot "vendor\llama.cpp")
    }
    cmake -S (Join-Path $projectRoot "vendor\llama.cpp") -B (Join-Path $projectRoot "vendor\llama.cpp\build") -DGGML_CUDA=ON -DLLAMA_CURL=OFF
    if ($LASTEXITCODE -ne 0) {
        cmake -S (Join-Path $projectRoot "vendor\llama.cpp") -B (Join-Path $projectRoot "vendor\llama.cpp\build") -DLLAMA_CURL=OFF
    }
    cmake --build (Join-Path $projectRoot "vendor\llama.cpp\build") --config Release --parallel
}

$configPath = Join-Path $projectRoot "config.yaml"
if (!(Test-Path $configPath)) {
    & $python (Join-Path $projectRoot "setup_wizard.py") --config $configPath
}

& $python (Join-Path $projectRoot "validate_environment.py") --config $configPath
& $python (Join-Path $projectRoot "desktop_app.py")
