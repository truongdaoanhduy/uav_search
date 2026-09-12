import subprocess
import sys
from pathlib import Path

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


def test_run_all_help_defaults_to_u6_u9_scope_and_exposes_eval_episodes():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_all.py"), "--help"], cwd=ROOT, text=True, capture_output=True, check=True)
    assert "--eval-episodes" in proc.stdout
    assert "u6/u9" in proc.stdout


def test_cli_defaults_to_deterministic_and_progress_100():
    for script in ["train.py", "run_all.py"]:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        text = proc.stdout
        assert "--no-deterministic" in text
        assert "100 episodes" in text


def test_network_calibration_help_describes_power_scaled_contact_range():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "calibrate_u6_network.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "scales the effective radius" in proc.stdout
    assert "does not enlarge these radii" not in proc.stdout


def test_check_system_describes_deterministic_paper_default():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_system.py"), "--device", "cpu"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert "Deterministic paper runs are the default" in proc.stdout
    assert "Use --deterministic only" not in proc.stdout


def test_seed_defaults_are_44_across_generic_training_entrypoints():
    import importlib.util
    import inspect

    import uav_search.runner.train as train_module

    def load_script(name):
        spec = importlib.util.spec_from_file_location(f"test_{name}", ROOT / "scripts" / name)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    train_script = load_script("train.py")
    run_all_script = load_script("run_all.py")
    train_args = train_script.build_parser().parse_args(["--algorithm", "masac", "--scenario", "u6"])
    run_all_args = run_all_script.build_parser().parse_args([])

    assert train_args.seed == 44
    assert run_all_args.seed == 44
    assert inspect.signature(train_module.train_experiment).parameters["seed"].default == 44


def test_check_system_exposes_strict_uavnetsim_preflight_flag():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_system.py"), "--help"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert "--require-uavnetsim" in proc.stdout


def test_kaggle_setup_script_is_present_and_shell_valid():
    script = ROOT / "scripts" / "setup_kaggle.sh"
    assert script.exists()
    subprocess.run(["bash", "-n", str(script)], cwd=ROOT, check=True)
