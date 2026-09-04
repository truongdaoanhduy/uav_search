import torch
import pytest

from uav_search.runtime import configure_runtime, make_adam, resolve_device, system_report
from uav_search.runner.train import train_experiment



def test_system_report_contains_training_backend_information():
    report = system_report("auto")
    for key in ["python", "torch", "device", "device_name", "cuda_available", "amp_enabled"]:
        assert key in report
    assert report["device"] in {"cpu", "cuda"} or str(report["device"]).startswith("cuda:")

def test_resolve_device_auto_returns_available_backend():
    device = resolve_device("auto")
    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_cpu_auto_amp_is_disabled():
    profile = configure_runtime(torch.device("cpu"), deterministic=False, amp_mode="auto")
    assert profile.amp_enabled is False
    assert profile.device.type == "cpu"
    assert profile.deterministic is False


def test_amp_on_requires_cuda():
    with pytest.raises(RuntimeError, match="AMP.*CUDA"):
        configure_runtime(torch.device("cpu"), deterministic=False, amp_mode="on")


def test_deterministic_flag_is_explicit():
    profile = configure_runtime(torch.device("cpu"), deterministic=True, amp_mode="off")
    assert profile.deterministic is True
    assert torch.are_deterministic_algorithms_enabled()

    profile = configure_runtime(torch.device("cpu"), deterministic=False, amp_mode="off")
    assert profile.deterministic is False
    assert not torch.are_deterministic_algorithms_enabled()


def test_make_adam_cpu_executes_training_step():
    model = torch.nn.Linear(3, 2)
    optimizer = make_adam(model.parameters(), lr=1e-3, device=torch.device("cpu"))
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    assert isinstance(optimizer, torch.optim.Adam)


def test_train_experiment_accepts_runtime_profile_flags(tmp_path):
    run_dir = train_experiment(
        algorithm="maddpg",
        scenario="f1_m5",
        episodes=1,
        steps=2,
        seed=3,
        device="cpu",
        output_root=tmp_path,
        deterministic=False,
        amp_mode="off",
        runtime_overrides={
            "batch_size": 2,
            "replay_size": 8,
            "hidden_sizes": [8, 8],
            "warmup_steps": 2,
            "update_after": 99,
        },
    )
    assert (run_dir / "config.yaml").exists()
