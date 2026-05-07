import numpy as np
from bo_baseline.bo_solver import BOSolver


def test_bo_solver_init():
    solver = BOSolver(param_bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]), n_initial=5)
    assert solver.param_bounds.shape == (2, 2)
    assert solver.n_initial == 5
    assert len(solver.X_observed) == 0


def test_bo_solver_initial_sampling():
    solver = BOSolver(param_bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]), n_initial=5, random_state=42)
    X_init = solver.sample_initial_points()
    assert X_init.shape == (5, 2)
    assert np.all(X_init >= -2.0)
    assert np.all(X_init <= 2.0)


def test_bo_solver_suggest_and_update():
    solver = BOSolver(param_bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]), n_initial=5, random_state=42)
    X_init = solver.sample_initial_points()
    for x in X_init:
        y = -float(np.sum(x ** 2))
        solver.add_observation(x, y)
    assert len(solver.X_observed) == 5
    next_point = solver.suggest()
    assert next_point.shape == (2,)
    assert np.all(next_point >= -2.0)
    assert np.all(next_point <= 2.0)


def test_best_observation():
    solver = BOSolver(param_bounds=np.array([[-1.0, 1.0]]), n_initial=3, random_state=42)
    solver.add_observation(np.array([0.5]), 0.8)
    solver.add_observation(np.array([-0.3]), 0.9)
    solver.add_observation(np.array([0.1]), 0.3)
    best_x, best_y = solver.best_observation
    assert best_y == 0.9
    assert best_x[0] == -0.3
