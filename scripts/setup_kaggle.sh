#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
cd "${ROOT}"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required to install the pinned UavNetSim source." >&2
  exit 2
fi

"${PYTHON_BIN}" -m pip install -r requirements.txt
"${PYTHON_BIN}" -m pip install -e ".[dev]"
"${ROOT}/scripts/install_uavnetsim.sh"
"${PYTHON_BIN}" "${ROOT}/scripts/check_system.py" --device auto --require-uavnetsim

cat <<'EOF'
Kaggle setup complete.
Recommended smoke check:
  python scripts/train.py --algorithm masac --scenario u6 --episodes 1 --steps 4 --device auto --amp auto --local-only
EOF
