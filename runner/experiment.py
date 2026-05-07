import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class RunConfig:
    method: str
    circuit: str
    n_iterations: int
    n_initial: int = 10
    output_path: str = ""
    seed: int = 42


class ExperimentRunner:
    """Runs an optimization experiment and records results."""

    @staticmethod
    def run(
        config: RunConfig,
        objective_fn: Callable[[np.ndarray], float],
        suggest_fn: Callable[[], np.ndarray],
    ) -> dict:
        np.random.seed(config.seed)
        results = {
            "method": config.method,
            "circuit": config.circuit,
            "n_iterations": config.n_iterations,
            "seed": config.seed,
            "iterations": [],
        }

        start_time = time.time()
        best_fom = -float("inf")
        best_params = None

        for i in range(config.n_iterations):
            params = suggest_fn()
            fom = objective_fn(params)

            results["iterations"].append({
                "iter": i,
                "params": np.asarray(params).tolist(),
                "fom": float(fom),
                "timestamp": time.time() - start_time,
            })

            if fom > best_fom:
                best_fom = fom
                best_params = np.asarray(params).tolist()

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

    @staticmethod
    def compare(results_list: List[dict]) -> str:
        lines = ["Method      | Best FoM   | Time (s) | Iterations"]
        lines.append("-" * 55)
        for r in results_list:
            lines.append(
                f"{r['method']:<12} | {r['best_fom']:>10.4f} | {r['total_time']:>8.1f} | {len(r['iterations']):>10}"
            )
        return "\n".join(lines)
