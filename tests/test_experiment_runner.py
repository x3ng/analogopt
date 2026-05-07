import json
import tempfile
import numpy as np
from pathlib import Path
from runner.experiment import ExperimentRunner, RunConfig


def test_run_config():
    config = RunConfig(method="bo", circuit="NMCF", n_iterations=10, n_initial=5, output_path="/tmp/test.json")
    assert config.method == "bo"
    assert config.n_iterations == 10


def test_basic_run():
    np.random.seed(42)

    def objective_fn(x):
        return -float(np.sum(x ** 2))

    def suggest_fn():
        return np.random.uniform(-1, 1, 2)

    config = RunConfig(method="test", circuit="test", n_iterations=5, output_path="/tmp/test_results.json")
    results = ExperimentRunner.run(config, objective_fn, suggest_fn)
    assert results["method"] == "test"
    assert len(results["iterations"]) == 5
    assert isinstance(results["best_fom"], float)


def test_save_and_load():
    results = {
        "method": "bo",
        "iterations": [
            {"iter": 0, "params": [0.5, -0.3], "fom": 0.85},
            {"iter": 1, "params": [0.2, 0.1], "fom": 0.92},
        ],
        "best_fom": 0.92,
        "best_params": [0.2, 0.1],
        "total_time": 12.5,
    }
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(b"dummy")
        tmp = f.name
    ExperimentRunner.save_results(results, tmp)
    loaded = ExperimentRunner.load_results(tmp)
    assert loaded["best_fom"] == 0.92
    assert len(loaded["iterations"]) == 2


def test_compare():
    results = [
        {"method": "bo", "best_fom": 0.92, "total_time": 10.0, "iterations": [{}] * 10},
        {"method": "llmbo", "best_fom": 0.95, "total_time": 30.0, "iterations": [{}] * 10},
    ]
    table = ExperimentRunner.compare(results)
    assert "bo" in table
    assert "llmbo" in table
    assert "0.92" in table
