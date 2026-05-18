import numpy as np
from typing import List, Optional

from solver.bo_solver import BOSolver
from solver.llm_interface import LLMInterface


class LLMBOSolver:
    """LLM-enhanced Bayesian Optimization. Combines BO-proposed points
    with LLM-generated candidates based on optimization history."""

    def __init__(
        self,
        param_bounds: np.ndarray,
        llm_interface: LLMInterface,
        n_initial: int = 10,
        random_state: Optional[int] = None,
    ):
        self.param_bounds = np.asarray(param_bounds, dtype=np.float64)
        self.n_dims = param_bounds.shape[0]
        self.llm = llm_interface
        self.bo_solver = BOSolver(param_bounds, n_initial, random_state)
        self.X_observed = []
        self.y_observed = []

    def sample_initial_points(self) -> np.ndarray:
        return self.bo_solver.sample_initial_points()

    def add_observation(self, x: np.ndarray, y: float):
        self.X_observed.append(np.asarray(x, dtype=np.float64))
        self.y_observed.append(float(y))
        self.bo_solver.add_observation(x, y)

    def suggest_candidates(self, n_bo: int = 1, n_llm: int = 2) -> List[np.ndarray]:
        """Generate mixed candidates: BO proposals + LLM proposals."""
        candidates = []

        for _ in range(n_bo):
            try:
                bo_point = self.bo_solver.suggest()
                candidates.append(
                    np.clip(bo_point, self.param_bounds[:, 0], self.param_bounds[:, 1])
                )
            except Exception:
                candidates.append(self._random_point())

        if n_llm > 0 and len(self.X_observed) >= self.bo_solver.n_initial:
            history = list(zip([x.tolist() for x in self.X_observed], self.y_observed))
            try:
                llm_points = self.llm.generate_candidates(
                    history=history,
                    param_bounds=self.param_bounds.tolist(),
                    n_candidates=n_llm,
                )
                for p in llm_points:
                    arr = np.clip(
                        np.array(p, dtype=np.float64),
                        self.param_bounds[:, 0],
                        self.param_bounds[:, 1],
                    )
                    candidates.append(arr)
            except Exception:
                for _ in range(n_llm):
                    candidates.append(self._random_point())

        return candidates

    def _random_point(self) -> np.ndarray:
        lo = self.param_bounds[:, 0]
        hi = self.param_bounds[:, 1]
        return lo + np.random.random(self.n_dims) * (hi - lo)

    @property
    def best_observation(self):
        if not self.y_observed:
            return None, None
        idx = np.argmax(self.y_observed)
        return np.array(self.X_observed[idx]), self.y_observed[idx]
