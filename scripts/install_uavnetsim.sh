#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
UAVNETSIM_COMMIT="04daafb815eb377409b40b285574eeb62b9a8d58"
UAVNETSIM_REPO="https://github.com/Zihao-Felix-Zhou/UavNetSim.git"

"${PYTHON_BIN}" -m pip install --upgrade "simpy==4.1.1"
# The MARL training backend intentionally uses UavNetSim CHANNEL_MODE=a2a, so
# Sionna RT and the other heavy visualization/server dependencies are not
# required. --no-deps keeps the training environment lightweight and stable.
"${PYTHON_BIN}" -m pip install --upgrade --no-deps \
  "git+${UAVNETSIM_REPO}@${UAVNETSIM_COMMIT}"

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import logging
import simpy

# UavNetSim upstream configures running_log.log at import time when the root
# logger has no handler. Guard this installer self-check just like the adapter.
root = logging.getLogger()
guard = logging.NullHandler()
added = not root.handlers
if added:
    root.addHandler(guard)
try:
    from phy.channel import Channel
    from mac.csma_ca import CsmaCa
    from entities.packet import DataPacket
finally:
    if added:
        root.removeHandler(guard)

print("simpy", simpy.__version__)
print("uavnetsim", importlib.metadata.version("uavnetsim"))
print("UavNetSim MAC/PHY imports OK:", Channel.__name__, CsmaCa.__name__, DataPacket.__name__)
PY
