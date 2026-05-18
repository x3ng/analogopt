"""Run LLANA v2 — improved prompt with circuit-aware feature names and domain context.

Key difference from run_llana.py:
  - Feature names: "W_M0", "L_M0", ... instead of "x0", "x1", ...
  - Model description includes circuit topology context
  - Metric described as "Figure of Merit" instead of "mean squared error"
  - Same normalized [-1, 1] bounds (required by AnalogGym env)
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
from experiments.utils import ExperimentRunner

# 24 NMCF amplifier design parameters (SKY130 PDK)
# These replace the meaningless x0..x23 with physically meaningful names
PARAM_NAMES = [
    # Bias PMOS cascade (M0-M3)
    "W_M0", "L_M0", "M_M0",
    # gm1 differential pair PMOS (M8-M9)
    "W_M8", "L_M8", "M_M8",
    # gm2 stage PMOS (M10)
    "W_M10", "L_M10", "M_M10",
    # gmf2 feedforward PMOS (M11)
    "W_M11", "L_M11", "M_M11",
    # Bias NMOS cascade (M17-M20)
    "W_M17", "L_M17", "M_M17",
    # Load NMOS (M21-M22)
    "W_M21", "L_M21", "M_M21",
    # gm3 output NMOS (M23)
    "W_M23", "L_M23", "M_M23",
    # Bias current
    "Ib",
    # Compensation capacitor multipliers
    "M_C0", "M_C1",
]

# NMCF amplifier topology description — injected into the model name field
# so LLANA includes it in the prompt prefix
CIRCUIT_DESCRIPTION = (
    "NMCF_Amplifier (SKY130, 3-stage nested Miller compensation with feedforward, "
    "24 design parameters: 7 MOSFET pairs with width/length/multiplier, "
    "bias current Ib, compensation caps M_C0/M_C1. "
    "Optimization goal: maximize Figure of Merit = sum of 11 performance scores "
    "including DC gain, GBW, phase margin, power, slew rate, settling time, "
    "CMRR, PSRR, offset, and temperature coefficient)"
)


def main():
    base_url = os.environ.get("LLANA_BASE_URL", "https://api.deepseek.com")
    api_key = os.environ.get("LLANA_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    model = os.environ.get("LLANA_MODEL", "deepseek-chat")
    n_trials = int(os.environ.get("LLANA_TRIALS", "15"))
    n_initial = int(os.environ.get("LLANA_INITIAL", "5"))
    n_gens = int(os.environ.get("LLANA_GENS", "3"))
    n_candidates = int(os.environ.get("LLANA_CANDIDATES", "5"))
    n_templates = int(os.environ.get("LLANA_TEMPLATES", "1"))

    os.environ.setdefault("OPENAI_API_KEY", api_key)

    print(f"=== LLANA v2 (Circuit-Aware Prompt) on NMCF ===")
    print(f"  base_url: {base_url}")
    print(f"  model: {model}")
    print(f"  n_trials: {n_trials}, n_initial: {n_initial}")
    print(f"  n_gens: {n_gens}, n_candidates: {n_candidates}")
    print(f"  feature names: {PARAM_NAMES[0]}, {PARAM_NAMES[1]}, ..., {PARAM_NAMES[-1]}")

    env = AnalogGymInterface()
    bounds = env.bounds  # shape (24, 2), normalized [-1, 1]
    n_dims = bounds.shape[0]

    task_context = {
        "model": CIRCUIT_DESCRIPTION,
        "task": "regression",
        "tot_feats": n_dims,
        "cat_feats": 0,
        "num_feats": n_dims,
        "n_classes": 1,
        "metric": "Figure_of_Merit",
        "lower_is_better": False,   # FoM: higher is better
        "num_samples": 1,
        "hyperparameter_constraints": {
            PARAM_NAMES[i]: ["float", "linear", [float(bounds[i, 0]), float(bounds[i, 1])]]
            for i in range(n_dims)
        },
    }

    all_iterations = []

    def init_f(n_samples):
        configs = []
        for _ in range(n_samples):
            cfg = {}
            for i in range(n_dims):
                cfg[PARAM_NAMES[i]] = float(np.random.uniform(bounds[i, 0], bounds[i, 1]))
            configs.append(cfg)
        return configs

    def bbox_eval_f(config):
        x = np.array([config[name] for name in PARAM_NAMES], dtype=np.float64)
        fom = float(env.evaluate(x))
        all_iterations.append({
            "params": x.tolist(),
            "fom": fom,
        })
        return config, {"score": fom, "generalization_score": fom}

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

    output_path = Path(__file__).parent.parent / "results" / "llana_v2_nmcf_results.json"

    try:
        configs_df, fvals_df = llambo.optimize()
    except Exception as e:
        print(f"\n!!! LLANA v2 crashed: {e}")
        print(f"  Saving partial results ({len(all_iterations)} evals) to {output_path}")
        import traceback
        traceback.print_exc()
        if all_iterations:
            partial = {
                "method": "llana_v2",
                "circuit": "NMCF",
                "n_iterations": len(all_iterations),
                "seed": 42,
                "best_fom": max(it["fom"] for it in all_iterations),
                "best_params": [],
                "total_time": time.time() - start_time,
                "config": {"base_url": base_url, "model": model,
                           "n_trials": n_trials, "n_initial": n_initial,
                           "n_gens": n_gens, "n_candidates": n_candidates,
                           "n_templates": n_templates,
                           "prompt": "circuit-aware"},
                "iterations": all_iterations,
                "crashed": True,
            }
            ExperimentRunner.save_results(partial, str(output_path))
            print(f"  Saved {len(all_iterations)} evals to {output_path}")
        env.close()
        raise

    total_time = time.time() - start_time
    env.close()

    best_idx = fvals_df["score"].idxmax()
    best_fom = float(fvals_df["score"].max())
    best_params = [float(configs_df.iloc[best_idx][name]) for name in PARAM_NAMES]

    results = {
        "method": "llana_v2",
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
            "prompt": "circuit-aware",
        },
        "iterations": all_iterations,
    }

    ExperimentRunner.save_results(results, str(output_path))

    print(f"\n=== LLANA v2 Results ===")
    print(f"  Best FoM: {best_fom:.4f}")
    print(f"  Total evals: {len(all_iterations)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Saved to: {output_path}")

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
