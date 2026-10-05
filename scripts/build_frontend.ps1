# Build the React frontend from source. Output goes to
# src/speed_profile_generator/webapp/static (git-ignored build output).
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $root "src/speed_profile_generator/webapp/frontend")
try {
    npm ci
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
}
finally {
    Pop-Location
}
