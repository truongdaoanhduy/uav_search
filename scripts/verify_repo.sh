#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
VERIFY_DEVICE="${VERIFY_DEVICE:-auto}"
cd "${ROOT}"

printf '[1/6] Python compile check\n'
"${PYTHON_BIN}" -m compileall -q src scripts tests

printf '[2/6] Shell syntax check\n'
bash -n scripts/install_uavnetsim.sh scripts/setup_kaggle.sh scripts/verify_repo.sh

printf '[3/6] Git whitespace check\n'
git diff --check

printf '[4/6] Full static lint check\n'
"${PYTHON_BIN}" -m ruff check src scripts tests

printf '[5/6] Strict runtime/UavNetSim preflight\n'
"${PYTHON_BIN}" scripts/check_system.py --device "${VERIFY_DEVICE}" --require-uavnetsim

printf '[6/6] Full regression suite\n'
"${PYTHON_BIN}" -m pytest -q -rs

printf 'Repository verification passed.\n'
