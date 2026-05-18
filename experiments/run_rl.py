"""Run RL baseline (DDPG + RGCN) on NMCF circuit and record results."""
import os
import sys
import time
import numpy as np
from pathlib import Path

# Setup path
AG_DIR = Path(__file__).parent.parent / "analoggym" / "RGNN_RL"
os.chdir(str(AG_DIR))
sys.path.insert(0, str(AG_DIR))

import torch
from ckt_graphs import GraphAMPNMCF
from ddpg import DDPGAgent
from models import ActorCriticRGCN
from utils import OutputParser2
from AMP_NMCF import AMPNMCFEnv
import gymnasium as gym

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.utils import ExperimentRunner
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_rl(num_steps: int = 1000, memory_size: int = 10000, batch_size: int = 128,
           initial_random_steps: int = 100, noise_sigma: float = 2.0, seed: int = 42) -> dict:
    """Run DDPG RL training on NMCF circuit."""
    print(f"=== RL Baseline (DDPG + RGCN) ===")
    print(f"  Steps: {num_steps}, Init random: {initial_random_steps}, Batch: {batch_size}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Register environment
    env_id = "sky130AMP_NMCF-v0"
    for env_name in list(gym.envs.registration.registry.keys()):
        if env_id in str(env_name):
            del gym.envs.registration.registry[env_name]
    gym.envs.registration.register(
        id=env_id, entry_point="AMP_NMCF:AMPNMCFEnv", max_episode_steps=50
    )
    env = gym.make(env_id)

    ckt = GraphAMPNMCF()
    agent = DDPGAgent(
        env, ckt,
        ActorCriticRGCN().Actor(ckt),
        ActorCriticRGCN().Critic(ckt),
        memory_size, batch_size, noise_sigma, 0.1, 0.9995,
        initial_random_steps=initial_random_steps, noise_type="uniform",
    )

    start_time = time.time()
    print(f"  Training...")

    agent.train(num_steps)

    # Get best result
    memory = agent.memory
    rews_buf = memory.rews_buf[:num_steps]
    best_idx = int(np.argmax(rews_buf))
    best_action = memory.acts_buf[best_idx]
    best_reward = float(np.max(rews_buf))

    # Replay best action to get detailed info
    env.step(best_action)
    results_parser = OutputParser2(ckt)
    op_results = results_parser.dcop("AMP_NMCF_op")

    elapsed = time.time() - start_time

    # Build results
    result = {
        "method": "rl",
        "circuit": "NMCF",
        "config": {
            "num_steps": num_steps,
            "memory_size": memory_size,
            "batch_size": batch_size,
            "initial_random_steps": initial_random_steps,
            "noise_sigma": noise_sigma,
            "seed": seed,
        },
        "best_fom": best_reward,
        "best_params": best_action.tolist(),
        "total_time": elapsed,
        "steps_per_second": num_steps / elapsed,
        "reward_history": rews_buf.tolist(),
    }

    output_path = str(RESULTS_DIR / "rl_nmcf_results.json")
    ExperimentRunner.save_results(result, output_path)

    print(f"  Best FOM: {best_reward:.4f} (step {best_idx})")
    print(f"  Time: {elapsed:.1f}s ({num_steps / elapsed:.1f} steps/s)")
    print(f"  Saved to {output_path}")

    env.close()
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_rl(num_steps=args.steps, seed=args.seed)
