"""Run LLANA (LLM-Enhanced BO) experiment on AnalogGym NMCF circuit.

LLANA replaces the GP surrogate and acquisition function with LLM in-context
learning, using few-shot prompting instead of mathematical models.

Based on: https://github.com/dekura/LLANA (arXiv 2406.05250)

The default parameters are conservative (n_gens=3, n_candidates=5, n_templates=1)
to avoid overwhelming the API with concurrent requests. Each trial makes ~15
API calls (down from ~200 with the original LLANA defaults of 10/10/2).

Usage:
    # Full experiment (5 initial + 15 trials, ~60 min)
    nix-shell -p ngspice --run "python experiments/run_llana.py"

    # Quick smoke test (3+3, ~15 min)
    LLANA_TRIALS=3 LLANA_INITIAL=3 nix-shell -p ngspice --run \\
        "python experiments/run_llana.py"

    # Use Volcengine Coding Plan
    LLANA_BASE_URL="https://ark.cn-beijing.volces.com/api/coding/v3" \\
    LLANA_MODEL="deepseek-v3.2" \\
    nix-shell -p ngspice --run "python experiments/run_llana.py"

Configuration via environment variables:
    LLANA_BASE_URL  — OpenAI-compatible base URL (default: https://api.deepseek.com)
    LLANA_API_KEY   — API key (default: ANTHROPIC_AUTH_TOKEN)
    LLANA_MODEL     — model name (default: deepseek-chat)
    LLANA_TRIALS    — optimization trials (default: 15)
    LLANA_INITIAL   — initial random samples (default: 5)
    LLANA_GENS      — LLM generations per candidate (default: 3)
    LLANA_CANDIDATES— candidate points per trial (default: 5)
    LLANA_TEMPLATES — prompt templates (default: 1)
Note: deepseek-chat is used instead of deepseek-v4-pro because the reasoning
model (v4-pro) returns empty content for long prompts via the OpenAI-compatible
endpoint.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "llana"))

from env_interface.analoggym_adapter import AnalogGymInterface
from runner.experiment import ExperimentRunner


def main():
    base_url = os.environ.get("LLANA_BASE_URL", "https://api.deepseek.com")
    api_key = os.environ.get("LLANA_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    model = os.environ.get("LLANA_MODEL", "deepseek-chat")
    n_trials = int(os.environ.get("LLANA_TRIALS", "15"))
    n_initial = int(os.environ.get("LLANA_INITIAL", "5"))
    # Use conservative defaults to avoid API rate limits
    n_gens = int(os.environ.get("LLANA_GENS", "3"))
    n_candidates = int(os.environ.get("LLANA_CANDIDATES", "5"))
    n_templates = int(os.environ.get("LLANA_TEMPLATES", "1"))

    # Ensure API key is available for OpenAI SDK
    os.environ.setdefault("OPENAI_API_KEY", api_key)

    print(f"=== LLANA (LLM-Enhanced BO) on NMCF ===")
    print(f"  base_url: {base_url}")
    print(f"  model: {model}")
    print(f"  n_trials: {n_trials}, n_initial: {n_initial}")
    print(f"  n_gens: {n_gens}, n_candidates: {n_candidates}, n_templates: {n_templates}")

    env = AnalogGymInterface()
    bounds = env.bounds  # shape (24, 2), normalized [-1, 1]
    n_dims = bounds.shape[0]

    # Build task_context for LLANA
    task_context = {
        "model": "NMCF_Amplifier",
        "task": "regression",
        "tot_feats": n_dims,
        "cat_feats": 0,
        "num_feats": n_dims,
        "n_classes": 1,
        "metric": "neg_mean_squared_error",
        "lower_is_better": False,   # FoM: higher is better
        "num_samples": 1,
        "hyperparameter_constraints": {
            f"x{i}": ["float", "linear", [float(bounds[i, 0]), float(bounds[i, 1])]]
            for i in range(n_dims)
        },
    }

    # Track all evaluations for result recording
    all_iterations = []

    def init_f(n_samples):
        configs = []
        for _ in range(n_samples):
            cfg = {}
            for i in range(n_dims):
                cfg[f"x{i}"] = float(np.random.uniform(bounds[i, 0], bounds[i, 1]))
            configs.append(cfg)
        return configs

    def bbox_eval_f(config):
        x = np.array([config[f"x{i}"] for i in range(n_dims)], dtype=np.float64)
        fom = float(env.evaluate(x))
        all_iterations.append({
            "params": x.tolist(),
            "fom": fom,
        })
        return config, {"score": fom, "generalization_score": fom}

    # Import LLAMBO from the cloned llana directory
    from llambo.llambo import LLAMBO

    start_time = time.time()

    llambo = LLAMBO(
        task_context=task_context,
        sm_mode="discriminative",
        n_candidates=n_candidates,
        n_templates=n_templates,
        n_gens=n_gens,
        alpha=-0.2,
        n_initial_samples=n_initial,
        n_trials=n_trials,
        init_f=init_f,
        bbox_eval_f=bbox_eval_f,
        chat_engine=model,
        provider="openai",
        base_url=base_url,
        use_input_warping=False,
        shuffle_features=False,
    )

    output_path = Path(__file__).parent.parent / "results" / "llana_nmcf_results.json"

    try:
        configs_df, fvals_df = llambo.optimize()
    except Exception as e:
        print(f"\n!!! LLANA crashed: {e}")
        print(f"  Saving partial results ({len(all_iterations)} evals) to {output_path}")
        import traceback
        traceback.print_exc()
        # Save whatever we have
        if all_iterations:
            partial = {
                "method": "llana",
                "circuit": "NMCF",
                "n_iterations": len(all_iterations),
                "seed": 42,
                "best_fom": max(it["fom"] for it in all_iterations),
                "best_params": [],
                "total_time": time.time() - start_time,
                "config": {"base_url": base_url, "model": model,
                           "n_trials": n_trials, "n_initial": n_initial,
                           "n_gens": n_gens, "n_candidates": n_candidates,
                           "n_templates": n_templates},
                "iterations": all_iterations,
                "crashed": True,
            }
            ExperimentRunner.save_results(partial, str(output_path))
            print(f"  Saved {len(all_iterations)} evals to {output_path}")
        env.close()
        raise

    total_time = time.time() - start_time
    env.close()

    # Build results in same format as existing experiments
    best_idx = fvals_df["score"].idxmax()
    best_fom = float(fvals_df["score"].max())
    best_params = [float(configs_df.iloc[best_idx][f"x{i}"]) for i in range(n_dims)]

    results = {
        "method": "llana",
        "circuit": "NMCF",
        "n_iterations": len(all_iterations),
        "seed": 42,
        "best_fom": best_fom,
        "best_params": best_params,
        "total_time": total_time,
        "config": {
            "base_url": base_url,
            "model": model,
            "n_trials": n_trials,
            "n_initial": n_initial,
            "n_gens": n_gens,
            "n_candidates": n_candidates,
            "n_templates": n_templates,
        },
        "iterations": all_iterations,
    }

    ExperimentRunner.save_results(results, str(output_path))

    print(f"\n=== LLANA Results ===")
    print(f"  Best FoM: {best_fom:.4f}")
    print(f"  Total evals: {len(all_iterations)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Saved to: {output_path}")

    # Quick comparison with existing results
    try:
        bo = ExperimentRunner.load_results(
            str(Path(__file__).parent.parent / "results" / "bo_nmcf_results.json")
        )
        llmbo = ExperimentRunner.load_results(
            str(Path(__file__).parent.parent / "results" / "llmbo_nmcf_results.json")
        )
        print(ExperimentRunner.compare([bo, llmbo, results]))
    except FileNotFoundError:
        pass

    return results


if __name__ == "__main__":
    main()
