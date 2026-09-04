import sys
from types import SimpleNamespace

from uav_search.runner.wandb_logger import tracking_mode, WandbLogger


class FakeTable:
    def __init__(self, columns=None, data=None, dataframe=None):
        self.columns = columns or ([] if dataframe is None else list(dataframe.columns))
        self.data = data or ([] if dataframe is None else dataframe.values.tolist())


class FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []
        self.dirs = []

    def add_file(self, path, name=None):
        self.files.append((path, name))

    def add_dir(self, path):
        self.dirs.append(path)

    def add(self, obj, name):
        self.files.append((name, obj))


class FakeRun:
    def __init__(self):
        self.logs = []
        self.artifacts = []
        self.finished = False
        self.summary = {}

    def log(self, payload, step=None):
        self.logs.append((payload, step))

    def log_artifact(self, artifact, aliases=None):
        self.artifacts.append((artifact, list(aliases or [])))

    def finish(self):
        self.finished = True


class FakeSettings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def fake_wandb_module(run):
    return SimpleNamespace(
        login=lambda **kwargs: True,
        init=lambda **kwargs: run,
        Settings=FakeSettings,
        Table=FakeTable,
        Artifact=FakeArtifact,
        Image=lambda path: ("image", path),
    )


def test_tracking_mode_without_api_key_is_local(tmp_path):
    assert tracking_mode() == "local"
    logger = WandbLogger(project="test", run_name="no-key", config={}, run_dir=tmp_path, mode="auto")
    assert logger.mode == "local"
    assert logger.online is False
    logger.log({"train/reward": 1.0}, step=1)
    logger.finish()


def test_tracking_mode_with_api_key_prefers_online():
    assert tracking_mode(api_key="test-key") == "online"

def test_environment_api_key_is_ignored(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "environment-key-must-not-be-used")
    assert tracking_mode() == "local"



def test_online_logger_streams_diagnostics_and_artifacts(monkeypatch, tmp_path):
    run = FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb_module(run))
    logger = WandbLogger(project="test", run_name="online", config={"x": 1}, run_dir=tmp_path, api_key="test-key", mode="auto")
    assert logger.online is True

    logger.log({"train/return_mean": 3.0}, step=2)
    logger.log_low_episode({"episode": 2, "search_rate": 0.0, "signals": ["search_low"], "severity": "critical"})
    logger.log_error({"episode": 2, "exception_type": "RuntimeError", "message": "boom", "traceback": "trace"})

    plot = tmp_path / "plot.png"
    plot.write_bytes(b"png")
    model = tmp_path / "final.pt"
    model.write_bytes(b"model")
    logger.log_image(plot, "plots/reward")
    logger.log_model(model, "test-model", aliases=["final"])
    logger.log_run_artifact("test-run-output")
    logger.finish()

    logged_keys = {k for payload, _ in run.logs for k in payload}
    assert "train/return_mean" in logged_keys
    assert "diagnostics/low_episode" in logged_keys
    assert "diagnostics/error" in logged_keys
    assert "diagnostics/low_metric_episodes" in logged_keys
    assert "diagnostics/errors" in logged_keys
    assert "plots/reward" in logged_keys
    assert len(run.artifacts) >= 2
    assert run.finished is True


def test_wandb_log_failure_falls_back_without_raising(monkeypatch, tmp_path):
    class BrokenRun(FakeRun):
        def log(self, payload, step=None):
            raise RuntimeError("network down")

    run = BrokenRun()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb_module(run))
    logger = WandbLogger(project="test", run_name="broken", config={}, run_dir=tmp_path, api_key="test-key", mode="auto")
    logger.log({"train/reward": 1.0}, step=1)
    assert logger.online is False
    assert logger.mode == "local"
    assert "network down" in (tmp_path / "logs" / "tracking_fallback.log").read_text()
