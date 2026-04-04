$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Args
    )
    & $FilePath @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $Args"
    }
}

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $projectRoot

if (!(Test-Path ".venv")) {
    Invoke-Checked py -3 -m venv .venv
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
Invoke-Checked $python -m pip install --upgrade pip
Invoke-Checked $python -m pip install -r requirements-build.txt

$llamaBuildBinDir = Join-Path $projectRoot "vendor\llama.cpp\build\bin\Release"
$llamaServer = Join-Path $llamaBuildBinDir "llama-server.exe"
if (!(Test-Path $llamaServer)) {
    throw "Missing $llamaServer. Build llama.cpp first, then rerun this packager."
}

if (Test-Path (Join-Path $projectRoot "dist\Celeste")) {
    Remove-Item -Recurse -Force (Join-Path $projectRoot "dist\Celeste")
}

Invoke-Checked $python -m PyInstaller packaging\pyinstaller\celeste.spec --noconfirm --clean

$bundleVendorDir = Join-Path $projectRoot "dist\Celeste\vendor\llama.cpp\build\bin\Release"
New-Item -ItemType Directory -Force -Path $bundleVendorDir | Out-Null
Copy-Item -Force (Join-Path $llamaBuildBinDir "*") $bundleVendorDir

$modelsDir = Join-Path $projectRoot "models"
if (Test-Path $modelsDir) {
    Copy-Item -Recurse -Force $modelsDir (Join-Path $projectRoot "dist\Celeste\models")
}

$embeddingsDir = Join-Path $projectRoot "embeddings"
if (Test-Path $embeddingsDir) {
    Copy-Item -Recurse -Force $embeddingsDir (Join-Path $projectRoot "dist\Celeste\embeddings")
}

$isccCommand = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
$isccPath = $null
if ($null -ne $isccCommand) {
    $isccPath = $isccCommand.Path
} else {
    foreach ($candidate in @(
        "C:\Program Files\Inno Setup 7\ISCC.exe",
        "C:\Program Files (x86)\Inno Setup 7\ISCC.exe"
    )) {
        if (Test-Path $candidate) {
            $isccPath = $candidate
            break
        }
    }
}

if ($null -ne $isccPath) {
    Invoke-Checked $isccPath (Join-Path $projectRoot "packaging\windows\celeste.iss")
} else {
    Write-Host "ISCC.exe not found. Generated dist\Celeste onedir bundle, but skipped .exe installer creation."
}
