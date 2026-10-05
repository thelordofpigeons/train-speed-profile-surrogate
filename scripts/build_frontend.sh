#!/usr/bin/env bash
# Build the React frontend from source. Output goes to
# src/speed_profile_generator/webapp/static (git-ignored build output).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src/speed_profile_generator/webapp/frontend"

npm ci
npm run build
