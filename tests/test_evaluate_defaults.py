import inspect

import pytest

from uav_search.runner.evaluate import evaluate_checkpoint


def test_eval_default_is_paper_5000_cases():
    assert inspect.signature(evaluate_checkpoint).parameters["episodes"].default == 5000


def test_evaluation_rejects_zero_cases_before_loading_checkpoint():
    with pytest.raises(ValueError, match="episodes must be >= 1"):
        evaluate_checkpoint("missing-checkpoint.pt", episodes=0)
