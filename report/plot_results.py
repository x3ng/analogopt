"""Generate comparison plots for optimization experiments (v2 with LLANA prompt ablation)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPORT_DIR = Path(__file__).parent.parent / "report"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load(method):
    return json.loads((RESULTS_DIR / f"{method}_nmcf_results.json").read_text())


def plot_convergence(bo, llmbo, rl, llana_v1, llana_v2, out):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    bo_foms = [it["fom"] for it in bo["iterations"]]
    llmbo_foms = [it["fom"] for it in llmbo["iterations"]]
    rl_foms = rl["reward_history"]
    llana1_foms = [it["fom"] for it in llana_v1["iterations"]]
    llana2_foms = [it["fom"] for it in llana_v2["iterations"]]

    colors = {
        "BO": "#2196F3", "LLM+BO": "#FF9800", "RL": "#4CAF50",
        "LLANA v1": "#E91E63", "LLANA v2": "#9C27B0",
    }

    # Panel 1: per-iteration FoM
    ax1.plot(range(len(bo_foms)), bo_foms, "o-", ms=3, alpha=0.7,
             label=f"BO (best={max(bo_foms):.3f})", color=colors["BO"])
    ax1.plot(range(len(llmbo_foms)), llmbo_foms, "o-", ms=3, alpha=0.7,
             label=f"LLM+BO (best={max(llmbo_foms):.3f})", color=colors["LLM+BO"])
    ax1.plot(range(len(rl_foms)), rl_foms, alpha=0.3, linewidth=0.5,
             label=f"RL (best={max(rl_foms):.3f})", color=colors["RL"])
    ax1.plot(range(len(llana1_foms)), llana1_foms, "s-", ms=3, alpha=0.7,
             label=f"LLANA v1 (best={max(llana1_foms):.3f})", color=colors["LLANA v1"])
    ax1.plot(range(len(llana2_foms)), llana2_foms, "D-", ms=3, alpha=0.7,
             label=f"LLANA v2 (best={max(llana2_foms):.3f})", color=colors["LLANA v2"])
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("FoM")
    ax1.set_title("Per-Iteration FoM (all evaluations)")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Panel 2: cumulative best
    def cummax(arr):
        r = []
        m = -float("inf")
        for x in arr:
            m = max(m, x)
            r.append(m)
        return r

    ax2.plot(range(len(bo_foms)), cummax(bo_foms), linewidth=2,
             label=f"BO ({cummax(bo_foms)[-1]:.3f})", color=colors["BO"])
    ax2.plot(range(len(llmbo_foms)), cummax(llmbo_foms), linewidth=2,
             label=f"LLM+BO ({cummax(llmbo_foms)[-1]:.3f})", color=colors["LLM+BO"])
    ax2.plot(range(len(rl_foms)), cummax(rl_foms), linewidth=2,
             label=f"RL ({cummax(rl_foms)[-1]:.3f})", color=colors["RL"])
    ax2.plot(range(len(llana1_foms)), cummax(llana1_foms), linewidth=2,
             label=f"LLANA v1 ({cummax(llana1_foms)[-1]:.3f})", color=colors["LLANA v1"])
    ax2.plot(range(len(llana2_foms)), cummax(llana2_foms), linewidth=2,
             label=f"LLANA v2 ({cummax(llana2_foms)[-1]:.3f})", color=colors["LLANA v2"])
    ax2.set_xlabel("Evaluation")
    ax2.set_ylabel("Cumulative Best FoM")
    ax2.set_title("Convergence (Cumulative Best)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Panel 3: bar comparison
    methods = ["BO", "LLM+BO", "RL", "LLANA v1", "LLANA v2"]
    bests = [max(bo_foms), max(llmbo_foms), max(rl_foms), max(llana1_foms), max(llana2_foms)]
    times = [bo["total_time"], llmbo["total_time"], rl["total_time"], llana_v1["total_time"], llana_v2["total_time"]]
    evals = [len(bo_foms) + 10, len(llmbo_foms) + 10, len(rl_foms), len(llana1_foms), len(llana2_foms)]
    bar_colors = [colors[m] for m in methods]

    x = np.arange(len(methods))
    w = 0.25
    ax3.bar(x - w, bests, w, label="Best FoM", color=bar_colors, edgecolor="white")
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x, times, w, label="Time (s)", color=bar_colors, alpha=0.3, edgecolor="white")
    ax3_twin.bar(x + w, evals, w, label="Evaluations", color=bar_colors, alpha=0.5, edgecolor="white")
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, fontsize=8)
    ax3.set_title("Final Comparison")
    ax3.set_ylabel("Best FoM")
    ax3_twin.set_ylabel("Time / Evals")
    bars1 = [plt.Rectangle((0, 0), 1, 1, color=c) for c in bar_colors]
    bars2 = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for c in bar_colors]
    ax3.legend(bars1 + bars2, methods + [f"{m} time/eval" for m in methods], fontsize=6, ncol=2)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    bo = load("bo")
    llmbo = load("llmbo")
    rl = load("rl")
    llana_v1 = load("llana")
    llana_v2 = load("llana_v2")
    plot_convergence(bo, llmbo, rl, llana_v1, llana_v2, str(FIG_DIR / "comparison.png"))
