# LLM辅助模拟电路参数优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete pipeline to compare RL, BO, and LLM+BO for analog circuit parameter optimization on AnalogGym's NMCF circuit

**Architecture:** Three independent experiment runners (RL, BO, LLM+BO) share a common AnalogGym environment interface. Each runner queries the env, records results, and outputs structured JSON. A final report script generates comparison tables and plots.

**Tech Stack:** Python 3.10+, AnalogGym (Ngspice+SKY130 in Docker), BoTorch, Anthropic SDK (DeepSeek/Qwen via Anthropic-compatible endpoints)

---

### Task 1: Initialize git repo and project structure

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`

- [ ] **Step 1: Initialize git repository**

Run:
```bash
cd /home/xen/Project/llmeda/analogopt && git init
```
Expected: "Initialized empty Git repository"

- [ ] **Step 2: Write .gitignore**

```gitignore
__pycache__/
*.pyc
.venv/
.env
*.swp
*~
analoggym/
results/
llambo/
llana/
```

- [ ] **Step 3: Create initial requirements.txt**

```
numpy>=1.24.0
scipy>=1.10.0
botorch>=0.9.0
gpytorch>=1.10.0
anthropic>=0.39.0
matplotlib>=3.7.0
pandas>=2.0.0
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore requirements.txt
git commit -m "chore: init project structure"
```

---

### Task 2: Clone AnalogGym and set up environment interface

**Files:**
- Create: `env_interface/__init__.py`
- Create: `env_interface/base.py`

- [ ] **Step 1: Clone AnalogGym**

```bash
cd /home/xen/Project/llmeda/analogopt && git clone https://github.com/CODA-Team/AnalogGym.git analoggym
```
Expected: AnalogGym repo cloned. Verify: `ls analoggym/RGNN_RL/main_AMP.py`

- [ ] **Step 2: Replace dev_params.py with our version**

```bash
cp /home/xen/Project/llmeda/analogopt/dev_params.py /home/xen/Project/llmeda/analogopt/analoggym/RGNN_RL/dev_params.py
```

- [ ] **Step 3: Explore AnalogGym env API**

Read `analoggym/RGNN_RL/main_AMP.py` and understand the Gym env interface: how is the env created, what does `step()` return, what is the action/observation space.

- [ ] **Step 4: Write abstract environment interface**

Create `env_interface/base.py`:

```python
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class EnvConfig:
    """Configuration for an AnalogGym environment."""
    circuit_name: str       # e.g. "NMCF"
    num_params: int         # number of tunable parameters
    param_bounds: np.ndarray  # shape (num_params, 2), low/high per param
    max_steps: int          # max simulation steps per episode

class BaseEnv(ABC):
    """Abstract interface for circuit optimization environments."""

    def __init__(self, config: EnvConfig):
        self.config = config

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset env and return initial observation."""
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Take action, return (obs, reward, done, info)."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        ...
```

- [ ] **Step 5: Commit**

```bash
git add env_interface/ .gitignore
git commit -m "feat: add abstract environment interface"
```

---

### Task 3: Write AnalogGym environment adapter

**Files:**
- Create: `env_interface/analoggym_adapter.py`
- Modify: `analoggym/RGNN_RL/dev_params.py` (already copied)

- [ ] **Step 1: Read main_AMP.py to understand the exact env interface**

Read `analoggym/RGNN_RL/main_AMP.py` — note the exact Gym registration, environment creation, action/observation format.

- [ ] **Step 2: Write the adapter**

Create `env_interface/analoggym_adapter.py`:

```python
import sys
import numpy as np
from pathlib import Path
from typing import Tuple

# Add AnalogGym to path
ANALOGGYM_PATH = Path(__file__).parent.parent / "analoggym" / "RGNN_RL"
sys.path.insert(0, str(ANALOGGYM_PATH))

from env_interface.base import BaseEnv, EnvConfig


class AnalogGymAdapter(BaseEnv):
    """Adapter wrapping AnalogGym's NMCF environment."""

    def __init__(self, config: EnvConfig):
        super().__init__(config)
        self._setup_gym()

    def _setup_gym(self):
        import gym
        import gym.envs.registration as reg

        # Register the NMCF environment
        reg.register(
            id='gymNMCf-v0',
            entry_point='gymNMCf:NMCfEnv',
        )
        self._env = gym.make('gymNMCf-v0')

    def reset(self) -> np.ndarray:
        obs = self._env.reset()
        return np.array(obs, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self._env.step(action)
        return np.array(obs, dtype=np.float32), float(reward), bool(done), info

    def close(self) -> None:
        self._env.close()
```

Note: exact gym registration details depend on what `main_AMP.py` reveals.

- [ ] **Step 3: Write smoke test — verify adapter can be imported and check expected structure**

Create `tests/test_adapter_structure.py`:

```python
import numpy as np
from env_interface.base import EnvConfig


def test_env_config_creation():
    config = EnvConfig(
        circuit_name="NMCF",
        num_params=10,
        param_bounds=np.array([[0.1, 10.0]] * 10),
        max_steps=100,
    )
    assert config.circuit_name == "NMCF"
    assert config.param_bounds.shape == (10, 2)
```

Run:
```bash
cd /home/xen/Project/llmeda/analogopt && python -m pytest tests/test_adapter_structure.py -v
```
Expected: PASS (1 test)

- [ ] **Step 4: Commit**

```bash
git add env_interface/ tests/
git commit -m "feat: add AnalogGym adapter and base env interface"
```

---

### Task 4: Build BO baseline solver

**Files:**
- Create: `bo_baseline/__init__.py`
- Create: `bo_baseline/bo_solver.py`
- Test: `tests/test_bo_solver.py`

- [ ] **Step 1: Write failing test for BO solver**

Create `tests/test_bo_solver.py`:

```python
import numpy as np
from bo_baseline.bo_solver import BOSolver


def dummy_objective(x: np.ndarray) -> float:
    """Simple 2D quadratic: f(x,y) = -(x^2 + y^2), max at (0,0)."""
    return -float(np.sum(x ** 2))


def test_bo_solver_init():
    solver = BOSolver(
        param_bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        n_initial=5,
    )
    assert solver.param_bounds.shape == (2, 2)
    assert solver.n_initial == 5
    assert len(solver.X_observed) == 0
    assert len(solver.y_observed) == 0


def test_bo_solver_initial_sampling():
    solver = BOSolver(
        param_bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        n_initial=5,
        random_state=42,
    )
    X_init = solver.sample_initial_points()
    assert X_init.shape == (5, 2)
    assert np.all(X_init >= -2.0)
    assert np.all(X_init <= 2.0)


def test_bo_solver_suggest_and_update():
    solver = BOSolver(
        param_bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        n_initial=5,
        random_state=42,
    )
    # Seed with initial points
    X_init = solver.sample_initial_points()
    for x in X_init:
        y = dummy_objective(x)
        solver.add_observation(x, y)

    # Fit model and suggest next point
    next_point = solver.suggest()
    assert next_point.shape == (2,)
    assert np.all(next_point >= -2.0)
    assert np.all(next_point <= 2.0)
```

Run:
```bash
cd /home/xen/Project/llmeda/analogopt && python -m pytest tests/test_bo_solver.py -v
```
Expected: FAIL — module not found

- [ ] **Step 2: Implement BOSolver**

Create `bo_baseline/bo_solver.py`:

```python
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood


class BOSolver:
    """Bayesian Optimization solver using BoTorch."""

    def __init__(self, param_bounds: np.ndarray, n_initial: int = 10, random_state: int = None):
        self.param_bounds = np.asarray(param_bounds)
        self.n_initial = n_initial
        self.n_dims = param_bounds.shape[0]
        self.X_observed = []
        self.y_observed = []
        self._best_y = -float("inf")
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

    def sample_initial_points(self) -> np.ndarray:
        """Generate Sobol/LHS initial points within bounds."""
        from botorch.utils.sampling import draw_sobol_samples

        X = draw_sobol_samples(
            bounds=torch.tensor(self.param_bounds.T, dtype=torch.float64),
            n=self.n_initial,
            q=1,
        ).squeeze(1).numpy()
        return X

    def add_observation(self, x: np.ndarray, y: float):
        """Record an (x, y) observation."""
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
        """Fit GP and propose next point via EI optimization."""
        model = self._build_model()
        best_f = torch.tensor(self._best_y, dtype=torch.float64)
        ei = ExpectedImprovement(model=model, best_f=best_f)

        bounds = torch.tensor(self.param_bounds.T, dtype=torch.float64)
        from botorch.optim import optimize_acqf
        candidate, _ = optimize_acqf(
            acq_function=ei,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )
        return candidate.squeeze(0).detach().numpy()

    @property
    def best_observation(self):
        if not self.y_observed:
            return None
        idx = np.argmax(self.y_observed)
        return self.X_observed[idx], self.y_observed[idx]
```

- [ ] **Step 3: Run tests**

```bash
cd /home/xen/Project/llmeda/analogopt && python -m pytest tests/test_bo_solver.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 4: Commit**

```bash
git add bo_baseline/ tests/test_bo_solver.py
git commit -m "feat: implement BO solver with BoTorch"
```

---

### Task 5: Build LLM interface

**Files:**
- Create: `llmbo/__init__.py`
- Create: `llmbo/llm_interface.py`
- Test: `tests/test_llm_interface.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_llm_interface.py`:

```python
import os
import json
from llmbo.llm_interface import LLMInterface


def test_llm_interface_init_with_api_key():
    iface = LLMInterface(
        api_key="test-key",
        base_url="https://test.api.com/anthropic",
        model="test-model",
    )
    assert iface.model == "test-model"


def test_build_candidate_prompt():
    iface = LLMInterface(api_key="test-key", base_url="http://test", model="test")
    history = [
        ([-0.5, 1.2], 0.85),
        ([0.3, -0.8], 0.72),
    ]
    param_bounds = [[-2.0, 2.0], [-2.0, 2.0]]
    prompt = iface._build_candidate_prompt(history, param_bounds)
    assert "[-0.5, 1.2]" in prompt or "-0.5" in prompt
    assert "0.85" in prompt


def test_parse_candidate_response():
    iface = LLMInterface(api_key="test-key", base_url="http://test", model="test")
    response = '{"candidates": [[0.11, -0.32], [0.87, 0.45]], "reasoning": "exploring upper right"}'
    candidates = iface._parse_candidates(response, n_dims=2)
    assert len(candidates) == 2
    assert len(candidates[0]) == 2
```

Run:
```bash
cd /home/xen/Project/llmeda/analogopt && python -m pytest tests/test_llm_interface.py -v
```
Expected: FAIL — module not found

- [ ] **Step 2: Implement LLMInterface**

Create `llmbo/llm_interface.py`:

```python
import json
import re
from typing import List, Tuple, Optional
try:
    import anthropic
except ImportError:
    anthropic = None


class LLMInterface:
    """LLM API wrapper using Anthropic-compatible SDK (DeepSeek / Qwen)."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        if anthropic is None:
            raise ImportError("anthropic package required: pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def chat(self, system: str, user_message: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Send a chat completion request."""
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return message.content[0].text

    def generate_candidates(
        self,
        history: List[Tuple[List[float], float]],
        param_bounds: List[List[float]],
        n_candidates: int = 3,
    ) -> List[List[float]]:
        """Use LLM to generate promising candidate parameter points."""
        prompt = self._build_candidate_prompt(history, param_bounds, n_candidates)
        system = (
            "You are an expert in analog circuit design and Bayesian optimization. "
            "Given the optimization history, propose promising parameter values to try next. "
            "Exploit areas of good performance while exploring uncertain regions. "
            "Respond ONLY with valid JSON: {\"candidates\": [[...], [...], ...], \"reasoning\": \"...\"}"
        )
        response = self.chat(system, prompt, temperature=0.7)
        return self._parse_candidates(response, len(param_bounds))

    def _build_candidate_prompt(
        self,
        history: List[Tuple[List[float], float]],
        param_bounds: List[List[float]],
        n_candidates: int,
    ) -> str:
        lines = ["Optimization history (parameters → Figure of Merit):"]
        for i, (params, fom) in enumerate(history):
            lines.append(f"  Iter {i}: {params} → FOM={fom:.6f}")

        sorted_hist = sorted(history, key=lambda x: x[1], reverse=True)
        best_params, best_fom = sorted_hist[0]
        lines.append(f"\nBest so far: {best_params} → FOM={best_fom:.6f}")
        lines.append(f"\nParameter bounds: {param_bounds}")
        lines.append(
            f"\nPropose {n_candidates} new parameter vectors to try next. "
            f"Consider both exploitation (near best point) and exploration (uncertain regions). "
            f"Return as JSON array."
        )
        return "\n".join(lines)

    def _parse_candidates(self, response: str, n_dims: int) -> List[List[float]]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Cannot parse LLM response: {response[:200]}...")
        candidates = data.get("candidates", data.get("parameters", []))
        if not candidates:
            raise ValueError(f"No candidates found in response: {response[:200]}...")
        return [[float(v) for v in c[:n_dims]] for c in candidates[:n_dims]]
```

- [ ] **Step 3: Run tests**

```bash
cd /home/xen/Project/llmeda/analogopt && python -m pytest tests/test_llm_interface.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 4: Commit**

```bash
git add llmbo/ tests/test_llm_interface.py
git commit -m "feat: implement LLM interface for candidate generation"
```

---

### Task 6: Build LLM+BO solver

**Files:**
- Create: `llmbo/llmbo_solver.py`
- Test: `tests/test_llmbo_solver.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_llmbo_solver.py`:

```python
import numpy as np
from unittest.mock import MagicMock, patch
from llmbo.llmbo_solver import LLMBOSolver


def dummy_objective(x):
    return -float(np.sum(x ** 2))


def test_llmbo_solver_init():
    solver = LLMBOSolver(
        param_bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        llm_interface=MagicMock(),
        n_initial=5,
        random_state=42,
    )
    assert solver.n_dims == 2
    assert len(solver.X_observed) == 0


@patch("llmbo.llmbo_solver.BOSolver")
def test_llmbo_solver_delegates_initial_sampling(mock_bo):
    """LLMBO should use BO for initial sampling, then LLM-enhanced thereafter."""
    solver = LLMBOSolver(
        param_bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        llm_interface=MagicMock(),
        n_initial=5,
    )
    solver.bo_solver.sample_initial_points = MagicMock(
        return_value=np.random.uniform(-2, 2, (5, 2))
    )
    X = solver.sample_initial_points()
    assert X.shape == (5, 2)


def test_llmbo_combines_candidates():
    """LLMBO should combine BO-proposed and LLM-proposed candidates."""
    llm_mock = MagicMock()
    llm_mock.generate_candidates.return_value = [[0.5, 0.5], [-0.3, -0.3]]

    solver = LLMBOSolver(
        param_bounds=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        llm_interface=llm_mock,
        n_initial=5,
        random_state=42,
    )
    solver.bo_solver.suggest = MagicMock(return_value=np.array([0.1, 0.1]))

    # Seed enough observations
    for _ in range(6):
        solver.add_observation(np.random.uniform(-2, 2, 2), np.random.random())

    candidates = solver.suggest_candidates(n_bo=1, n_llm=2)
    assert len(candidates) == 3  # 1 BO + 2 LLM
```

Run:
```bash
cd /home/xen/Project/llmeda/analogopt && python -m pytest tests/test_llmbo_solver.py -v
```
Expected: FAIL — module not found

- [ ] **Step 2: Implement LLMBOSolver**

Create `llmbo/llmbo_solver.py`:

```python
import numpy as np
from typing import List, Optional

from bo_baseline.bo_solver import BOSolver
from llmbo.llm_interface import LLMInterface


class LLMBOSolver:
    """LLM-enhanced Bayesian Optimization. Uses BO for systematic exploration
    and LLM for generating informed candidate points based on history."""

    def __init__(
        self,
        param_bounds: np.ndarray,
        llm_interface: LLMInterface,
        n_initial: int = 10,
        random_state: Optional[int] = None,
    ):
        self.param_bounds = np.asarray(param_bounds)
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

        # BO proposals
        for _ in range(n_bo):
            try:
                bo_point = self.bo_solver.suggest()
                candidates.append(np.clip(bo_point, self.param_bounds[:, 0], self.param_bounds[:, 1]))
            except Exception:
                candidates.append(self._random_point())

        # LLM proposals
        if n_llm > 0 and len(self.X_observed) >= self.bo_solver.n_initial:
            history = list(zip([x.tolist() for x in self.X_observed], self.y_observed))
            try:
                llm_points = self.llm.generate_candidates(
                    history=history,
                    param_bounds=self.param_bounds.tolist(),
                    n_candidates=n_llm,
                )
                for p in llm_points:
                    arr = np.clip(np.array(p), self.param_bounds[:, 0], self.param_bounds[:, 1])
                    candidates.append(arr)
            except Exception as e:
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
        return self.X_observed[idx], self.y_observed[idx]
```

- [ ] **Step 3: Run tests**

```bash
cd /home/xen/Project/llmeda/analogopt && python -m pytest tests/test_llmbo_solver.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 4: Commit**

```bash
git add llmbo/ tests/test_llmbo_solver.py
git commit -m "feat: implement LLMBO solver combining BO and LLM proposals"
```

---

### Task 7: Write experiment runner

**Files:**
- Create: `runner/__init__.py`
- Create: `runner/experiment.py`
- Test: `tests/test_experiment_runner.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_experiment_runner.py`:

```python
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from runner.experiment import ExperimentRunner, RunConfig


def test_run_config():
    config = RunConfig(
        method="bo",
        circuit="NMCF",
        n_iterations=50,
        n_initial=10,
        output_path="/tmp/test_results.json",
    )
    assert config.method == "bo"
    assert config.n_iterations == 50


def test_save_results():
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
        ExperimentRunner.save_results(results, f.name)
        saved = json.loads(Path(f.name).read_text())
    assert saved["best_fom"] == 0.92
    assert len(saved["iterations"]) == 2
```

Run:
```bash
cd /home/xen/Project/llmeda/analogopt && python -m pytest tests/test_experiment_runner.py -v
```
Expected: FAIL — module not found

- [ ] **Step 2: Implement ExperimentRunner**

Create `runner/experiment.py`:

```python
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Callable


@dataclass
class RunConfig:
    method: str            # "rl", "bo", "llmbo"
    circuit: str           # "NMCF"
    n_iterations: int      # total optimization steps
    n_initial: int         # initial random samples
    output_path: str       # where to save results
    seed: int = 42


class ExperimentRunner:
    """Runs an optimization experiment and records results."""

    @staticmethod
    def run(
        config: RunConfig,
        objective_fn: Callable[[np.ndarray], float],
        suggest_fn: Callable[[], np.ndarray],
        param_bounds: np.ndarray,
    ) -> dict:
        """Generic experiment loop."""
        np.random.seed(config.seed)
        results = {"method": config.method, "circuit": config.circuit,
                   "n_iterations": config.n_iterations, "iterations": []}

        start_time = time.time()
        best_fom = -float("inf")
        best_params = None

        for i in range(config.n_iterations):
            params = suggest_fn()
            fom = objective_fn(params)

            results["iterations"].append({
                "iter": i,
                "params": params.tolist(),
                "fom": float(fom),
            })

            if fom > best_fom:
                best_fom = fom
                best_params = params.tolist()

        results["best_fom"] = float(best_fom)
        results["best_params"] = best_params
        results["total_time"] = time.time() - start_time
        return results

    @staticmethod
    def save_results(results: dict, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    @staticmethod
    def load_results(path: str) -> dict:
        with open(path) as f:
            return json.load(f)
```

- [ ] **Step 3: Run tests**

```bash
cd /home/xen/Project/llmeda/analogopt && python -m pytest tests/test_experiment_runner.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 4: Commit**

```bash
git add runner/ tests/test_experiment_runner.py
git commit -m "feat: add experiment runner with result recording"
```

---

### Task 8: Wire up end-to-end experiments with mock objective

**Files:**
- Create: `run_experiments.py`

- [ ] **Step 1: Write the master experiment script**

Create `run_experiments.py`:

```python
"""Master script to run all optimization experiments and compare results."""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from runner.experiment import ExperimentRunner, RunConfig
from bo_baseline.bo_solver import BOSolver
from llmbo.llm_interface import LLMInterface
from llmbo.llmbo_solver import LLMBOSolver

RESULTS_DIR = Path(__file__).parent / "results"


def get_objective_fn(circuit: str):
    """Get the objective function for the given circuit.
    Uses mock quadratic for testing; replace with AnalogGym adapter when Docker is ready.
    """
    if os.environ.get("USE_ANALOGGYM"):
        from env_interface.analoggym_adapter import AnalogGymAdapter
        from env_interface.base import EnvConfig

        config = EnvConfig(
            circuit_name=circuit,
            num_params=10,
            param_bounds=np.array([[-1.0, 1.0]] * 10),
            max_steps=1,
        )
        env = AnalogGymAdapter(config)

        def f(params):
            obs, reward, done, info = env.step(params)
            return float(reward)
        return f, config.param_bounds
    else:
        # Mock objective: maximize negative quadratic (optimum at all-zeros)
        # scaled to [0, 1] range for realistic FoM behavior
        def mock_objective(params):
            x = np.asarray(params)
            # A more interesting landscape: Ackley-like with noise
            n = len(x)
            sum_sq = np.sum(x ** 2)
            sum_cos = np.sum(np.cos(2 * np.pi * x))
            fom = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e
            return float(fom)

        bounds = np.array([[-1.0, 1.0]] * 10)
        return mock_objective, bounds


def run_bo_experiment(circuit: str, n_iterations: int):
    objective_fn, bounds = get_objective_fn(circuit)
    solver = BOSolver(param_bounds=bounds, n_initial=10, random_state=42)

    # Initial sampling
    X_init = solver.sample_initial_points()
    for x in X_init:
        solver.add_observation(x, objective_fn(x))

    def suggest():
        x = solver.suggest()
        y = objective_fn(x)
        solver.add_observation(x, y)
        return x

    remaining = n_iterations - len(X_init)
    config = RunConfig(
        method="bo",
        circuit=circuit,
        n_iterations=remaining,
        n_initial=10,
        output_path=str(RESULTS_DIR / "bo_results.json"),
        seed=42,
    )
    results = ExperimentRunner.run(config, objective_fn, suggest, bounds)
    results["n_total"] = n_iterations
    ExperimentRunner.save_results(results, config.output_path)
    print(f"BO best FOM: {results['best_fom']:.4f}, time: {results['total_time']:.1f}s")
    return results


def run_llmbo_experiment(circuit: str, n_iterations: int):
    objective_fn, bounds = get_objective_fn(circuit)

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    model = os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]")

    llm = LLMInterface(api_key=api_key, base_url=base_url, model=model)
    solver = LLMBOSolver(param_bounds=bounds, llm_interface=llm, n_initial=10, random_state=42)

    X_init = solver.sample_initial_points()
    for x in X_init:
        solver.add_observation(x, objective_fn(x))

    i = len(X_init)

    def suggest():
        nonlocal i
        candidates = solver.suggest_candidates(n_bo=1, n_llm=2)
        best_x, best_y = None, -float("inf")
        for x in candidates:
            y = objective_fn(x)
            solver.add_observation(x, y)
            if y > best_y:
                best_y = y
                best_x = x
        i += len(candidates)
        return best_x

    remaining = n_iterations - len(X_init)
    config = RunConfig(
        method="llmbo",
        circuit=circuit,
        n_iterations=remaining,
        n_initial=10,
        output_path=str(RESULTS_DIR / "llmbo_results.json"),
        seed=42,
    )
    results = ExperimentRunner.run(config, objective_fn, suggest, bounds)
    results["n_total"] = n_iterations
    ExperimentRunner.save_results(results, config.output_path)
    print(f"LLMBO best FOM: {results['best_fom']:.4f}, time: {results['total_time']:.1f}s")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", default="NMCF")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--methods", nargs="+", default=["bo", "llmbo"])
    parser.add_argument("--use-analoggym", action="store_true")
    args = parser.parse_args()

    if args.use_analoggym:
        os.environ["USE_ANALOGGYM"] = "1"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    if "bo" in args.methods:
        print("=== Running BO baseline ===")
        results["bo"] = run_bo_experiment(args.circuit, args.iterations)

    if "llmbo" in args.methods:
        print("=== Running LLM+BO ===")
        results["llmbo"] = run_llmbo_experiment(args.circuit, args.iterations)

    # Print summary comparison
    print("\n=== Summary ===")
    for method, r in results.items():
        print(f"{method}: best FOM={r['best_fom']:.4f}, time={r['total_time']:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run with mock objective to verify end-to-end flow**

First, test just BO (no LLM API needed):
```bash
cd /home/xen/Project/llmeda/analogopt && python run_experiments.py --methods bo --iterations 20
```
Expected: Runs 20 BO iterations on mock objective, prints best FOM, saves `results/bo_results.json`.

- [ ] **Step 3: Verify results file**

Run:
```bash
cat /home/xen/Project/llmeda/analogopt/results/bo_results.json | python -c "import json,sys; d=json.load(sys.stdin); print(f'Method: {d[\"method\"]}, Best FOM: {d[\"best_fom\"]:.4f}, Iterations: {len(d[\"iterations\"])}')"
```
Expected: `Method: bo, Best FOM: <value>, Iterations: 10` (20 - 10 initial)

- [ ] **Step 4: Commit**

```bash
git add run_experiments.py results/
git commit -m "feat: wire up end-to-end experiment pipeline"
```

---

### Task 9: Run LLM+BO experiment with real API

- [ ] **Step 1: Start nix-shell with DeepSeek API**

Run:
```bash
cd /home/xen/Project/llmeda/analogopt && nix-shell ~/nix-shell/cc-ds.nix --run "python run_experiments.py --methods llmbo --iterations 20"
```
Expected: Runs LLM+BO on mock objective. Watch LLM API calls in output.

- [ ] **Step 2: Run with Qwen API as alternative**

```bash
cd /home/xen/Project/llmeda/analogopt && nix-shell ~/nix-shell/cc-qw.nix --run "python run_experiments.py --methods llmbo --iterations 20"
```

- [ ] **Step 3: Compare BO vs LLM+BO results**

```bash
python -c "
import json
for f in ['results/bo_results.json', 'results/llmbo_results.json']:
    d = json.load(open(f))
    print(f'{d[\"method\"]}: best={d[\"best_fom\"]:.4f}, time={d[\"total_time\"]:.1f}s, iters={len(d[\"iterations\"])}')
"
```

- [ ] **Step 4: Commit results**

```bash
git add results/
git commit -m "results: mock objective BO vs LLMBO comparison"
```

---

### Task 10: Results visualization

**Files:**
- Create: `report/plot_results.py`

- [ ] **Step 1: Write plotting script**

Create `report/plot_results.py`:

```python
"""Generate comparison plots for the optimization experiments."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_plot(method_files: dict, output_path: str):
    """Plot FoM vs iteration for each method."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"bo": "#2196F3", "llmbo": "#FF9800", "rl": "#4CAF50"}

    for method, filepath in method_files.items():
        if not Path(filepath).exists():
            continue
        data = json.loads(Path(filepath).read_text())
        foms = [it["fom"] for it in data["iterations"]]
        iters = range(len(foms))
        cummax = np.maximum.accumulate(foms)

        color = colors.get(method, "#999999")
        ax1.plot(iters, foms, alpha=0.3, color=color)
        ax1.plot(iters, cummax, linewidth=2, label=f"{method} (best={max(foms):.3f})", color=color)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Figure of Merit")
    ax1.set_title("Optimization Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar chart of final best FoM and time
    methods = []
    best_foms = []
    times = []
    for method, filepath in method_files.items():
        if not Path(filepath).exists():
            continue
        data = json.loads(Path(filepath).read_text())
        methods.append(method.upper())
        best_foms.append(data["best_fom"])
        times.append(data.get("total_time", 0))

    x = np.arange(len(methods))
    width = 0.35
    ax2.bar(x - width/2, best_foms, width, label="Best FoM", color="#2196F3")
    ax2.set_ylabel("Best FoM")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_title("Final Performance Comparison")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    files = {
        "bo": str(RESULTS_DIR / "bo_results.json"),
        "llmbo": str(RESULTS_DIR / "llmbo_results.json"),
    }
    load_and_plot(files, str(OUTPUT_DIR / "comparison.png"))
```

- [ ] **Step 2: Run plotting**

```bash
cd /home/xen/Project/llmeda/analogopt && python report/plot_results.py
```
Expected: `Saved plot to report/figures/comparison.png`

- [ ] **Step 3: Commit**

```bash
git add report/
git commit -m "feat: add results visualization"
```

---

### Task 11: Connect to real AnalogGym Docker environment

- [ ] **Step 1: Find AnalogGym Docker setup instructions**

Read `analoggym/RGNN_RL_Docker/` directory for Dockerfile and setup.

- [ ] **Step 2: Pull AnalogGym Docker image**

```bash
# Attempt direct pull
docker pull <analoggym-image> || echo "Docker pull failed - will need proxy or manual setup"
```

If Docker pull fails through proxy, try:
```bash
docker pull <analoggym-image> --proxy http://127.0.0.1:7890 || echo "Check clash proxy settings"
```

- [ ] **Step 3: Modify adapter to call Docker-based environment**

Update `env_interface/analoggym_adapter.py` to support Docker execution mode:

```python
import subprocess
import json
import numpy as np


class DockerAnalogGymAdapter(BaseEnv):
    """Adapter that runs AnalogGym simulations inside Docker."""

    def __init__(self, config: EnvConfig, docker_image: str = "analoggym"):
        super().__init__(config)
        self.docker_image = docker_image

    def _run_simulation(self, params: np.ndarray) -> float:
        """Run one simulation via Docker and return FoM."""
        params_json = json.dumps(params.tolist())
        cmd = [
            "docker", "run", "--rm",
            self.docker_image,
            "python", "-c",
            f"import json, numpy as np; "
            f"from main_AMP import run_episode; "
            f"params = np.array(json.loads('{params_json}')); "
            f"result = run_episode(params); "
            f"print(json.dumps({{'fom': float(result)}}))"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return json.loads(result.stdout)["fom"]
```

- [ ] **Step 4: Commit**

```bash
git add env_interface/
git commit -m "feat: add Docker-based AnalogGym adapter"
```

---

### Task 12: Write comparison report

**Files:**
- Create: `report/report.md`

- [ ] **Step 1: Draft report structure**

Create `report/report.md`:

```markdown
# LLM辅助模拟电路参数优化 对比实验报告

## 1. 引言
- 模拟电路参数优化的挑战
- 传统方法（RL, BO）与 LLM 辅助方法的动机
- 本实验目标：系统对比三种方法在 NMCF 电路上的表现

## 2. 方法

### 2.1 问题定义
- NMCF 电路参数优化问题
- Figure of Merit (FoM) 定义
- 参数空间和约束

### 2.2 RL 方法（Baseline）
- RGNN_RL 模型结构
- 训练配置

### 2.3 Bayesian Optimization
- GP surrogate model
- Expected Improvement 采集函数
- BoTorch 实现

### 2.4 LLM+BO 方法
- LLM 在 BO 流程中的角色
- 候选点生成的 prompt 设计
- 与标准 BO 的混合策略

## 3. 实验设置
- 电路环境：NMCF (SKY130 PDK)
- 迭代次数、初始采样
- 硬件/软件环境
- API 配置

## 4. 结果

### 4.1 收敛速度对比
### 4.2 最终性能对比
### 4.3 样本效率
### 4.4 稳定性分析

## 5. 讨论
- LLM+BO 的优势和局限
- LLM API 成本分析
- 与传统方法的适用场景对比

## 6. 结论与未来工作
```

- [ ] **Step 2: Fill in results from experiment data**

- [ ] **Step 3: Commit**

```bash
git add report/report.md
git commit -m "docs: add experiment report"
```

---

### Task 13: Experiment with LLAMBO/LLANA (optional)

- [ ] **Step 1: Clone LLAMBO**

```bash
cd /home/xen/Project/llmeda/analogopt && git clone https://github.com/tennisonliu/LLAMBO.git llambo
```

- [ ] **Step 2: Clone LLANA**

```bash
cd /home/xen/Project/llmeda/analogopt && git clone https://github.com/dekura/LLANA.git llana
```

- [ ] **Step 3: Explore each and document findings in report**

---

### Task 14: Package and submit

- [ ] **Step 1: Create submission archive**

```bash
cd /home/xen/Project/llmeda/analogopt
tar -czf ../analogopt_submission.tar.gz \
    --exclude='analoggym' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    .
```

- [ ] **Step 2: Verify archive**

```bash
tar -tzf ../analogopt_submission.tar.gz | head -30
```
