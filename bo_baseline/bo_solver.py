import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood


class BOSolver:
    """Bayesian Optimization solver using BoTorch with GP + EI."""

    def __init__(self, param_bounds: np.ndarray, n_initial: int = 10, random_state: int = None):
        self.param_bounds = np.asarray(param_bounds, dtype=np.float64)
        self.n_initial = n_initial
        self.n_dims = param_bounds.shape[0]
        self.X_observed = []
        self.y_observed = []
        self._best_y = -float("inf")
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

    def sample_initial_points(self) -> np.ndarray:
        from botorch.utils.sampling import draw_sobol_samples
        bounds_tensor = torch.tensor(self.param_bounds.T, dtype=torch.float64)
        X = draw_sobol_samples(bounds=bounds_tensor, n=self.n_initial, q=1).squeeze(1).numpy()
        return X

    def add_observation(self, x: np.ndarray, y: float):
        self.X_observed.append(np.asarray(x, dtype=np.float64))
        self.y_observed.append(float(y))
        self._best_y = max(self._best_y, float(y))

    def _build_model(self):
        train_X = torch.tensor(np.stack(self.X_observed), dtype=torch.float64)
        train_Y = torch.tensor(self.y_observed, dtype=torch.float64).unsqueeze(-1)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def suggest(self) -> np.ndarray:
        model = self._build_model()
        best_f = torch.tensor(self._best_y, dtype=torch.float64)
        ei = ExpectedImprovement(model=model, best_f=best_f)
        bounds = torch.tensor(self.param_bounds.T, dtype=torch.float64)
        from botorch.optim import optimize_acqf
        candidate, _ = optimize_acqf(
            acq_function=ei, bounds=bounds, q=1, num_restarts=10, raw_samples=256
        )
        return candidate.squeeze(0).detach().numpy()

    @property
    def best_observation(self):
        if not self.y_observed:
            return None, None
        idx = np.argmax(self.y_observed)
        return np.array(self.X_observed[idx]), self.y_observed[idx]
