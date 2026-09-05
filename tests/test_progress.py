from uav_search.runner.progress import (
    format_episode_progress,
    format_startup_summary,
    should_report_episode,
)


def test_should_report_first_interval_and_final_episode():
    assert should_report_episode(1, 50000, 10)
    assert not should_report_episode(2, 50000, 10)
    assert should_report_episode(10, 50000, 10)
    assert should_report_episode(50000, 50000, 10)


def test_progress_interval_must_be_positive():
    try:
        should_report_episode(1, 10, 0)
    except ValueError as exc:
        assert ">= 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_format_episode_progress_contains_useful_training_state():
    text = format_episode_progress(
        {
            "episode": 20,
            "return_mean": -12.5,
            "search_rate": 0.4,
            "targets_found": 2,
            "episode_sec": 1.25,
            "env_steps_per_sec": 40.0,
        },
        total_episodes=50000,
    )
    assert "20/50000" in text
    assert "return=-12.500" in text
    assert "search=40.0%" in text
    assert "targets=2" in text
    assert "1.25s" in text
    assert "40.0 step/s" in text


def test_format_startup_summary_shows_device_and_wandb_destination():
    text = format_startup_summary(
        algorithm="masac",
        scenario="f1_m5",
        episodes=50000,
        steps=50,
        device="cuda:0",
        device_name="Tesla T4",
        amp_enabled=True,
        deterministic=False,
        tracking_status="wandb-online",
        project_url="https://wandb.ai/uav_search_paper/uav_search_target",
        run_url="https://wandb.ai/uav_search_paper/uav_search_target/runs/abc",
        run_dir="runs/masac/f1_m5/abc",
    )
    assert "MASAC" in text
    assert "f1_m5" in text
    assert "50000" in text
    assert "cuda:0" in text
    assert "Tesla T4" in text
    assert "AMP=on" in text
    assert "wandb-online" in text
    assert "/runs/abc" in text
