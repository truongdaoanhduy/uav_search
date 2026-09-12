from uav_search.dependencies import (
    SIMPY_VERSION,
    UAVNETSIM_COMMIT,
    UAVNETSIM_REQUIRED_MODULES,
    UAVNETSIM_VERSION,
    uavnetsim_status,
)


def test_uavnetsim_status_has_reproducibility_contract():
    status = uavnetsim_status()
    assert status["uavnetsim_expected_version"] == UAVNETSIM_VERSION
    assert status["uavnetsim_expected_commit"] == UAVNETSIM_COMMIT
    assert status["simpy_expected_version"] == SIMPY_VERSION
    assert set(status["required_modules"]) == set(UAVNETSIM_REQUIRED_MODULES)
    assert isinstance(status["lightweight_runtime_ready"], bool)
