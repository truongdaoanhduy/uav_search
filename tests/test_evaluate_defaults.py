import inspect
from uav_search.runner.evaluate import evaluate_checkpoint

def test_eval_default_is_paper_5000_cases():
    assert inspect.signature(evaluate_checkpoint).parameters["episodes"].default == 5000
