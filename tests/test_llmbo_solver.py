import numpy as np
from unittest.mock import MagicMock
from llmbo.llmbo_solver import LLMBOSolver


def test_llmbo_solver_init():
    solver = LLMBOSolver(
        param_bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        llm_interface=MagicMock(),
        n_initial=5,
        random_state=42,
    )
    assert solver.n_dims == 2
    assert len(solver.X_observed) == 0


def test_llmbo_initial_sampling_delegates_to_bo():
    solver = LLMBOSolver(
        param_bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        llm_interface=MagicMock(),
        n_initial=5,
        random_state=42,
    )
    X = solver.sample_initial_points()
    assert X.shape == (5, 2)
    assert np.all(X >= -2.0)
    assert np.all(X <= 2.0)


def test_llmbo_combines_bo_and_llm_candidates():
    llm_mock = MagicMock()
    llm_mock.generate_candidates.return_value = [[0.5, 0.5], [-0.3, -0.3]]

    solver = LLMBOSolver(
        param_bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        llm_interface=llm_mock,
        n_initial=5,
        random_state=42,
    )
    solver.bo_solver.suggest = MagicMock(return_value=np.array([0.1, 0.1]))

    for _ in range(6):
        solver.add_observation(np.random.uniform(-2, 2, 2), np.random.random())

    candidates = solver.suggest_candidates(n_bo=1, n_llm=2)
    assert len(candidates) == 3


def test_best_observation():
    solver = LLMBOSolver(
        param_bounds=np.array([[-1.0, 1.0]]),
        llm_interface=MagicMock(),
        n_initial=3,
        random_state=42,
    )
    solver.add_observation(np.array([0.5]), 0.8)
    solver.add_observation(np.array([-0.3]), 0.9)
    solver.add_observation(np.array([0.1]), 0.3)
    best_x, best_y = solver.best_observation
    assert best_y == 0.9
