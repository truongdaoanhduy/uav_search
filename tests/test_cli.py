from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_train_cli_exposes_generic_gpu_and_wandb_modes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    help_text = proc.stdout
    for flag in ["--device", "--amp", "--deterministic", "--wandb-api-key", "--wandb-mode", "--local-only", "--progress-every"]:
        assert flag in help_text
    assert "--wandb-project" not in help_text
    assert "--wandb-entity" not in help_text


def test_run_all_cli_exposes_same_runtime_controls():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_all.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    help_text = proc.stdout
    for flag in ["--device", "--amp", "--deterministic", "--wandb-api-key", "--wandb-mode", "--local-only", "--progress-every"]:
        assert flag in help_text
    assert "--wandb-project" not in help_text
    assert "--wandb-entity" not in help_text


def test_run_all_help_defaults_to_fig7_scope_and_exposes_eval_episodes():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_all.py"), "--help"], cwd=ROOT, text=True, capture_output=True, check=True)
    assert "--eval-episodes" in proc.stdout
    assert "f1_m5/f1_m9" in proc.stdout


def test_cli_defaults_to_deterministic_and_progress_100():
    for script in ["train.py", "run_all.py"]:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        text = proc.stdout
        assert "--no-deterministic" in text
        assert "100 episodes" in text


def test_check_system_describes_deterministic_paper_default():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_system.py"), "--device", "cpu"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert "Deterministic paper runs are the default" in proc.stdout
    assert "Use --deterministic only" not in proc.stdout
