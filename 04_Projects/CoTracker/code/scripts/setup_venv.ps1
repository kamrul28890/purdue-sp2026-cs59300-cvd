<#
Setup script for Windows PowerShell to create a Python 3.10 virtualenv
and install project dependencies.

Usage (from repo root):
  .\scripts\setup_venv.ps1 [-Recreate] [-SkipRequirements] [-UseLockfile]

Options:
  -Recreate          Remove existing .venv and recreate it.
  -SkipRequirements  Create venv but skip installing requirements.
  -UseLockfile       Install from requirements-lock.txt instead of
                     requirements-repro.txt.

Notes:
  - requirements-repro.txt is the recommended path for fresh local setup.
  - requirements-lock.txt is an exact local environment snapshot and may be
    platform-specific.
#>

param(
    [switch]$Recreate = $false,
    [switch]$SkipRequirements = $false,
    [switch]$UseLockfile = $false
)

function New-Venv {
    param([string]$TargetDir)

    if (Get-Command python3.10 -ErrorAction SilentlyContinue) {
        Write-Host "Creating venv using python3.10"
        & python3.10 -m venv $TargetDir
        return
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        Write-Host "Creating venv using py -3.10"
        & py -3.10 -m venv $TargetDir
        return
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        $version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($version -eq "3.10") {
            Write-Host "Creating venv using python"
            & python -m venv $TargetDir
            return
        }
    }

    Write-Error "Python 3.10 not found on PATH. Please install Python 3.10 and re-run."
    exit 1
}

if ($Recreate -and (Test-Path .\.venv)) {
    Write-Host "Removing existing .venv..."
    Remove-Item -Recurse -Force .\.venv
}

if (-not (Test-Path .\.venv)) {
    New-Venv -TargetDir ".venv"
}

Write-Host "To activate: . .\\.venv\\Scripts\\Activate.ps1"

if (-not $SkipRequirements) {
    $requirementsFile = if ($UseLockfile) { ".\requirements-lock.txt" } else { ".\requirements-repro.txt" }

    Write-Host "Upgrading pip and wheel"
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools

    Write-Host "Installing project requirements from $requirementsFile"
    & .\.venv\Scripts\pip.exe install -r $requirementsFile

    Write-Host "Installing package in editable mode"
    & .\.venv\Scripts\pip.exe install -e .
}

Write-Host "Setup complete. Activate with: . .\\.venv\\Scripts\\Activate.ps1"
