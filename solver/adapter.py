import os
import sys
import numpy as np
from pathlib import Path

AG_DIR = Path(__file__).parent.parent / "analoggym" / "RGNN_RL"


class AnalogGymInterface:
    """Thin wrapper around AnalogGym's NMCF environment for optimization."""

    def __init__(self):
        cwd = os.getcwd()
        os.chdir(str(AG_DIR))
        sys.path.insert(0, str(AG_DIR))
        from AMP_NMCF import AMPNMCFEnv
        self._env = AMPNMCFEnv()
        os.chdir(cwd)
        self.action_dim = self._env.action_space.shape[0]
        self.bounds = np.array([[-1.0, 1.0]] * self.action_dim)

    def evaluate(self, params: np.ndarray) -> float:
        """Run one simulation and return Figure of Merit (reward)."""
        cwd = os.getcwd()
        os.chdir(str(AG_DIR))
        action = np.asarray(params, dtype=np.float64)
        obs, reward, terminated, _, info = self._env.step(action)
        os.chdir(cwd)
        return float(reward)

    def get_info(self, params: np.ndarray) -> dict:
        """Run simulation and return detailed performance info."""
        cwd = os.getcwd()
        os.chdir(str(AG_DIR))
        action = np.asarray(params, dtype=np.float64)
        obs, reward, terminated, _, info = self._env.step(action)
        os.chdir(cwd)
        return {"reward": float(reward), **info}

    def close(self):
        self._env.close()
