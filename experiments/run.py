"""Run optimization experiments on real AnalogGym NMCF circuit."""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.utils import ExperimentRunner, RunConfig
from solver.bo_solver import BOSolver
from solver.llm_interface import LLMInterface
from solver.llmbo_solver import LLMBOSolver
from solver.adapter import AnalogGymInterface

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_bo_real(n_iterations: int, n_initial: int = 10, seed: int = 42) -> dict:
    print(f"=== BO Baseline (NMCF real) ===\n  Iterations: {n_iterations}, Initial: {n_initial}")

    env = AnalogGymInterface()
    bounds = env.bounds
    objective_fn = env.evaluate

    solver = BOSolver(param_bounds=bounds, n_initial=n_initial, random_state=seed)
    X_init = solver.sample_initial_points()

    print(f"  Running {len(X_init)} initial samples...")
    for i, x in enumerate(X_init):
        y = objective_fn(x)
        solver.add_observation(x, y)
        print(f"    Init {i+1}/{len(X_init)}: FOM={y:.4f}")

    def step():
        x = solver.suggest()
        y = objective_fn(x)
        solver.add_observation(x, y)
        return x, y

    remaining = n_iterations - n_initial
    config = RunConfig(
        method="bo", circuit="NMCF", n_iterations=remaining,
        n_initial=n_initial, seed=seed,
        output_path=str(RESULTS_DIR / "bo_nmcf_results.json"),
    )
    results = ExperimentRunner.run_combined(config, step)
    env.close()

    print(f"  Best FOM: {results['best_fom']:.4f}, Time: {results['total_time']:.1f}s")
    ExperimentRunner.save_results(results, config.output_path)
    return results


def run_llmbo_real(n_iterations: int, n_initial: int = 10, seed: int = 42) -> dict:
    print(f"=== LLM+BO (NMCF real) ===\n  Iterations: {n_iterations}, Initial: {n_initial}")

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    model = os.environ.get("ANTHROPIC_MODEL", "")

    env = AnalogGymInterface()
    bounds = env.bounds
    objective_fn = env.evaluate

    llm = LLMInterface(api_key=api_key, base_url=base_url, model=model)
    solver = LLMBOSolver(param_bounds=bounds, llm_interface=llm, n_initial=n_initial, random_state=seed)
    X_init = solver.sample_initial_points()

    print(f"  Running {len(X_init)} initial samples...")
    for i, x in enumerate(X_init):
        y = objective_fn(x)
        solver.add_observation(x, y)
        print(f"    Init {i+1}/{len(X_init)}: FOM={y:.4f}")

    def step():
        candidates = solver.suggest_candidates(n_bo=1, n_llm=2)
        best_x, best_y = None, -float("inf")
        for x in candidates:
            y = objective_fn(x)
            solver.add_observation(x, y)
            if y > best_y:
                best_y = y
                best_x = x
        return best_x, best_y

    remaining = n_iterations - n_initial
    config = RunConfig(
        method="llmbo", circuit="NMCF", n_iterations=remaining,
        n_initial=n_initial, seed=seed,
        output_path=str(RESULTS_DIR / "llmbo_nmcf_results.json"),
    )
    results = ExperimentRunner.run_combined(config, step)
    env.close()

    print(f"  Best FOM: {results['best_fom']:.4f}, Time: {results['total_time']:.1f}s")
    ExperimentRunner.save_results(results, config.output_path)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bo", "llmbo", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--initial", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    if args.no_llm and args.method in ("llmbo", "both"):
        print("--no-llm: skipping LLMBO")
        args.method = "bo"

    if args.method in ("bo", "both"):
        run_bo_real(args.iterations, args.initial, args.seed)

    if args.method in ("llmbo", "both") and not args.no_llm:
        try:
            run_llmbo_real(args.iterations, args.initial, args.seed)
        except Exception as e:
            print(f"LLMBO failed: {e}")

    print("\nDone.")

    # Show comparison if both ran
    bo_file = RESULTS_DIR / "bo_nmcf_results.json"
    llmbo_file = RESULTS_DIR / "llmbo_nmcf_results.json"
    if bo_file.exists() and llmbo_file.exists():
        bo = ExperimentRunner.load_results(str(bo_file))
        llmbo = ExperimentRunner.load_results(str(llmbo_file))
        print(ExperimentRunner.compare([bo, llmbo]))


if __name__ == "__main__":
    main()
