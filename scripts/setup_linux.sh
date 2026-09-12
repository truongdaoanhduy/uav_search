#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
cd "${ROOT}"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required to install the pinned UavNetSim source." >&2
  exit 2
fi

"${PYTHON_BIN}" -m pip install -e ".[dev]"
"${ROOT}/scripts/install_uavnetsim.sh"
"${PYTHON_BIN}" "${ROOT}/scripts/check_system.py" --device auto --require-uavnetsim

echo "Linux setup complete. Run: bash scripts/verify_repo.sh"
