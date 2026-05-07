"""Run optimization experiments comparing BO and LLM+BO on mock/AnalogGym objectives."""
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


def mock_objective(x: np.ndarray) -> float:
    """Ackley-like landscape for testing. Maximum at origin, ~2.7 max value."""
    x = np.asarray(x)
    n = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return float(-20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e)


def run_bo_experiment(n_iterations: int, n_dims: int = 10, seed: int = 42) -> dict:
    objective_fn = mock_objective
    bounds = np.array([[-1.0, 1.0]] * n_dims)
    solver = BOSolver(param_bounds=bounds, n_initial=10, random_state=seed)

    X_init = solver.sample_initial_points()
    for x in X_init:
        solver.add_observation(x, objective_fn(x))

    iteration = 0

    def suggest():
        nonlocal iteration
        x = solver.suggest()
        y = objective_fn(x)
        solver.add_observation(x, y)
        iteration += 1
        return x

    remaining = n_iterations - len(X_init)
    config = RunConfig(
        method="bo", circuit="mock", n_iterations=remaining,
        n_initial=10, seed=seed,
        output_path=str(RESULTS_DIR / "bo_results.json"),
    )
    results = ExperimentRunner.run(config, objective_fn, suggest)
    print(f"BO  : best FoM={results['best_fom']:.4f}, time={results['total_time']:.1f}s, iters={len(results['iterations'])}")
    ExperimentRunner.save_results(results, config.output_path)
    return results


def run_llmbo_experiment(n_iterations: int, n_dims: int = 10, seed: int = 42) -> dict:
    objective_fn = mock_objective
    bounds = np.array([[-1.0, 1.0]] * n_dims)

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    model = os.environ.get("ANTHROPIC_MODEL", "")

    llm = LLMInterface(api_key=api_key, base_url=base_url, model=model)
    solver = LLMBOSolver(param_bounds=bounds, llm_interface=llm, n_initial=10, random_state=seed)

    X_init = solver.sample_initial_points()
    for x in X_init:
        solver.add_observation(x, objective_fn(x))

    iteration = 0

    def suggest():
        nonlocal iteration
        candidates = solver.suggest_candidates(n_bo=1, n_llm=2)
        best_x, best_y = None, -float("inf")
        for x in candidates:
            y = objective_fn(x)
            solver.add_observation(x, y)
            if y > best_y:
                best_y = y
                best_x = x
        iteration += len(candidates)
        return best_x

    remaining = n_iterations - len(X_init)
    config = RunConfig(
        method="llmbo", circuit="mock", n_iterations=remaining,
        n_initial=10, seed=seed,
        output_path=str(RESULTS_DIR / "llmbo_results.json"),
    )
    results = ExperimentRunner.run(config, objective_fn, suggest)
    print(f"LLMBO: best FoM={results['best_fom']:.4f}, time={results['total_time']:.1f}s, iters={len(results['iterations'])}")
    ExperimentRunner.save_results(results, config.output_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run optimization experiments")
    parser.add_argument("--method", choices=["bo", "llmbo", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--dims", type=int, default=10, help="Parameter dimensions (mock mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-llm", action="store_true", help="Skip LLMBO (no API)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.no_llm and args.method in ("llmbo", "both"):
        args.method = "bo"
        print("--no-llm set, skipping LLMBO")

    if args.method in ("bo", "both"):
        print("=== BO Baseline ===")
        run_bo_experiment(args.iterations, args.dims, args.seed)

    if args.method in ("llmbo", "both") and not args.no_llm:
        print("=== LLM+BO ===")
        try:
            run_llmbo_experiment(args.iterations, args.dims, args.seed)
        except Exception as e:
            print(f"LLMBO failed: {e}")

    print("\nDone. Results in results/")


if __name__ == "__main__":
    main()
