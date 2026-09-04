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
    for flag in ["--device", "--amp", "--deterministic", "--wandb-api-key", "--wandb-mode", "--wandb-project", "--local-only"]:
        assert flag in help_text


def test_run_all_cli_exposes_same_runtime_controls():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_all.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    help_text = proc.stdout
    for flag in ["--device", "--amp", "--deterministic", "--wandb-api-key", "--wandb-mode", "--wandb-project", "--local-only"]:
        assert flag in help_text
