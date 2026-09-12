import os

import pandas as pd
import pytest
import torch
import yaml

import uav_search.runner.train as train_module
from uav_search.runner.train import train_experiment
from uav_search.runtime import (
    configure_runtime,
    make_adam,
    resolve_device,
    system_report,
)


def test_system_report_contains_training_backend_information():
    report = system_report("auto")
    for key in ["python", "torch", "device", "device_name", "cuda_available", "amp_enabled", "deterministic_enabled"]:
        assert key in report
    assert report["device"] in {"cpu", "cuda"} or str(report["device"]).startswith("cuda:")
    assert report["deterministic_default"] is True
    assert report["deterministic_enabled"] is True

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


@pytest.mark.parametrize(
    ("episodes", "steps", "message"),
    [(0, None, "episodes must be >= 1"), (1, 0, "steps must be >= 1")],
)
def test_train_experiment_rejects_nonpositive_lengths_before_setup(
    monkeypatch, tmp_path, episodes, steps, message
):
    def fail_if_setup_starts(*_args, **_kwargs):
        raise AssertionError("configuration loading must not start for invalid lengths")

    monkeypatch.setattr(train_module, "load_config", fail_if_setup_starts)

    with pytest.raises(ValueError, match=message):
        train_experiment(
            algorithm="maddpg",
            scenario="f1_m5",
            episodes=episodes,
            steps=steps,
            output_root=tmp_path,
        )


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


def test_deterministic_runtime_sets_cublas_workspace_config(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    configure_runtime(torch.device("cpu"), deterministic=True, amp_mode="off")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in {":4096:8", ":16:8"}


@pytest.mark.parametrize("algorithm", ["maddpg", "matd3", "masac"])
def test_paper_training_defaults_to_deterministic_and_same_seed_repeats_metrics(algorithm, tmp_path):
    kwargs = {
        "algorithm":algorithm,
        "scenario":"f1_m5",
        "episodes":2,
        "steps":4,
        "seed":123,
        "device":"cpu",
        "amp_mode":"off",
        "runtime_overrides":{
            "batch_size": 4,
            "replay_size": 32,
            "hidden_sizes": [8, 8],
            "warmup_steps": 0,
            "update_after": 4,
            "checkpoint_every_episodes": 0,
        },
    }
    run_a = train_experiment(output_root=tmp_path / f"a-{algorithm}", **kwargs)
    run_b = train_experiment(output_root=tmp_path / f"b-{algorithm}", **kwargs)
    cfg_a = yaml.safe_load((run_a / "config.yaml").read_text())
    cfg_b = yaml.safe_load((run_b / "config.yaml").read_text())
    assert cfg_a["runtime"]["deterministic"] is True
    assert cfg_b["runtime"]["deterministic"] is True
    a = pd.read_csv(run_a / "metrics" / "episodes.csv")
    b = pd.read_csv(run_b / "metrics" / "episodes.csv")
    nondeterministic_columns = {
        "episode_sec", "wall_time_sec", "env_steps_per_sec", "updates_per_sec",
        "gpu_allocated_mb", "gpu_reserved_mb", "gpu_peak_allocated_mb",
    }
    comparable = [c for c in a.columns if c not in nondeterministic_columns]
    pd.testing.assert_frame_equal(a[comparable], b[comparable], check_exact=True)
    eval_a = yaml.safe_load((run_a / "evaluation.json").read_text())
    eval_b = yaml.safe_load((run_b / "evaluation.json").read_text())
    assert eval_a == eval_b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available in this test environment")
@pytest.mark.parametrize("algorithm", ["maddpg", "matd3", "masac"])
def test_cuda_same_seed_repeatability_when_gpu_is_available(algorithm, tmp_path):
    kwargs = {
        "algorithm": algorithm,
        "scenario": "f1_m5",
        "episodes": 1,
        "steps": 4,
        "seed": 321,
        "device": "cuda:0",
        "amp_mode": "auto",
        "runtime_overrides": {
            "batch_size": 4, "replay_size": 32, "hidden_sizes": [8, 8],
            "warmup_steps": 0, "update_after": 4, "checkpoint_every_episodes": 0,
        },
    }
    run_a = train_experiment(output_root=tmp_path / f"cuda-a-{algorithm}", **kwargs)
    run_b = train_experiment(output_root=tmp_path / f"cuda-b-{algorithm}", **kwargs)
    a = pd.read_csv(run_a / "metrics" / "episodes.csv")
    b = pd.read_csv(run_b / "metrics" / "episodes.csv")
    timing = {
        "episode_sec", "wall_time_sec", "env_steps_per_sec", "updates_per_sec",
        "gpu_allocated_mb", "gpu_reserved_mb", "gpu_peak_allocated_mb",
    }
    comparable = [c for c in a.columns if c not in timing]
    pd.testing.assert_frame_equal(a[comparable], b[comparable], check_exact=True)
