from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from typing import Any

UAVNETSIM_REPOSITORY = "https://github.com/Zihao-Felix-Zhou/UavNetSim.git"
UAVNETSIM_COMMIT = "04daafb815eb377409b40b285574eeb62b9a8d58"
UAVNETSIM_VERSION = "2.0.0"
SIMPY_VERSION = "4.1.1"
UAVNETSIM_REQUIRED_MODULES = (
    "simpy",
    "utils.config",
    "phy.channel",
    "phy.a2a",
    "phy.sionna_rt",
    "mac.csma_ca",
    "entities.packet",
)


def _distribution_direct_url(distribution_name: str) -> dict[str, Any] | None:
    try:
        dist = importlib.metadata.distribution(distribution_name)
        raw = dist.read_text("direct_url.json")
    except (importlib.metadata.PackageNotFoundError, FileNotFoundError):
        return None
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def uavnetsim_status() -> dict[str, Any]:
    """Report only the lightweight MAC/PHY dependency surface used by this repo."""
    modules: dict[str, bool] = {}
    for name in UAVNETSIM_REQUIRED_MODULES:
        try:
            modules[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            modules[name] = False

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    direct_url = _distribution_direct_url("uavnetsim") or {}
    vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
    installed_commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    installed_version = version("uavnetsim")
    simpy_version = version("simpy")
    available = all(modules.values()) and installed_version is not None
    commit_match = installed_commit == UAVNETSIM_COMMIT if installed_commit else False
    version_match = installed_version == UAVNETSIM_VERSION if installed_version else False
    simpy_match = simpy_version == SIMPY_VERSION if simpy_version else False
    return {
        "uavnetsim_available": bool(available),
        "uavnetsim_version": installed_version,
        "uavnetsim_expected_version": UAVNETSIM_VERSION,
        "uavnetsim_version_match": bool(version_match),
        "uavnetsim_installed_commit": installed_commit,
        "uavnetsim_expected_commit": UAVNETSIM_COMMIT,
        "uavnetsim_commit_match": bool(commit_match),
        "simpy_version": simpy_version,
        "simpy_expected_version": SIMPY_VERSION,
        "simpy_version_match": bool(simpy_match),
        "required_modules": modules,
        "lightweight_runtime_ready": bool(available and version_match and commit_match and simpy_match),
    }


def require_pinned_uavnetsim() -> dict[str, Any]:
    status = uavnetsim_status()
    if not status["lightweight_runtime_ready"]:
        raise RuntimeError(
            "Pinned lightweight UavNetSim runtime is not ready. "
            "Run scripts/install_uavnetsim.sh (or scripts/setup_kaggle.sh on Kaggle). "
            f"Expected UavNetSim {UAVNETSIM_VERSION} at commit {UAVNETSIM_COMMIT} "
            f"with SimPy {SIMPY_VERSION}. Status: {status}"
        )
    return status
