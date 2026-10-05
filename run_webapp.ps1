param (
    [switch]$Fast = $false
)

# The webapp needs an HDF5 dataset (not included) and a trained run under runs/.
# See src/speed_profile_generator/webapp/README_WEBAPP.md.

# Prefer the repo virtualenv if present
$venvPython = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
if (Test-Path $venvPython) {
    $py = $venvPython
}
else {
    $py = "python"
}

& $py --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python is not installed or not in PATH."
    exit 1
}

if (-not $Fast) {
    # Install Python Requirements
    Write-Host "Installing/Updating Python dependencies..."
    & $py -m pip install -r requirements-webapp.txt

    # Check Node (only needed for frontend build/dev)
    node --version
    if ($LASTEXITCODE -eq 0) {
        # Build for production if the bundle is missing
        if (-not (Test-Path "src/speed_profile_generator/webapp/static/index.html")) {
            Write-Host "Building Frontend for Production..."
            & (Join-Path $PSScriptRoot "scripts/build_frontend.ps1")
        }
    }
    else {
        Write-Warning "Node.js not found. Frontend build skipped. Ensure 'src/speed_profile_generator/webapp/static' exists or install Node."
    }
}
else {
    Write-Host "Skipping dependency checks (-Fast mode)..."
}

# Run Backend
Write-Host "Starting Web App Backend..."
$env:PYTHONPATH = "src"
& $py -m speed_profile_generator.webapp.main
