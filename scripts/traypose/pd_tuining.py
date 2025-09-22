#!/usr/bin/env python3
import os
import sys
import time
import csv
import itertools
import multiprocessing as mp
from typing import Tuple, Dict, Any, List

import numpy as np

# Ensure project root available so we can import env
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.traypose.traypose_env import TrayPoseEnv

# Tuning config
MODEL_PATH = "assets/panda_tray/panda_tray_cylinder.xml"
NUM_WORKERS = max(1, mp.cpu_count() - 1)

# Grid (or random) search params
KP_VALUES = [20.0, 30.0, 40.0, 50.0]          # proportional gains to try
KD_VALUES = [2.0, 5.0, 8.0, 12.0]             # derivative gains to try
MAX_INC_VALUES = [0.003, 0.005, 0.008, 0.012] # max_joint_increment (radians/step)

# Evaluation settings
N_EPISODES_PER_SETTING = 6
EPISODE_MAX_STEPS = 300
ACTION_SCALE = 0.8   # scale applied to env.action_space.sample() to simulate agent magnitude
DROP_Z_THRESHOLD = 0.1  # same as your env drop threshold

OUTPUT_CSV = "tuning_results.csv"


def evaluate_setting(params: Tuple[float, float, float], seed: int = 0) -> Dict[str, Any]:
    """Evaluate a single parameter setting. Returns metrics dict."""
    kp, kd, max_inc = params
    # Create environment
    env = TrayPoseEnv(model_path=MODEL_PATH)
    # Set tunable params
    env.kp = float(kp)
    env.kd = float(kd)
    env.max_joint_increment = float(max_inc)

    rng = np.random.RandomState(seed)

    total_rewards = []
    drops = 0
    mean_min_offsets = []

    for ep in range(N_EPISODES_PER_SETTING):
        obs = env.reset()
        ep_reward = 0.0
        min_offset = float("inf")
        dropped = False

        # Warmup: some zero steps to settle dynamics (optional)
        for _ in range(3):
            obs, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        for t in range(EPISODE_MAX_STEPS):
            # sample a small random action to emulate agent behavior
            a = env.action_space.sample().astype(np.float32)
            # scale action magnitude
            a = a * ACTION_SCALE
            obs, rew, done, info = env.step(a)
            ep_reward += float(rew)

            # track cylinder offset (XY)
            cyl = env._get_cylinder_xyz()
            tray = env.tray_pos.copy()
            offset = np.linalg.norm(cyl[:2] - tray[:2])
            if offset < min_offset:
                min_offset = offset

            # detect drop
            if cyl[2] < DROP_Z_THRESHOLD:
                dropped = True
                drops += 1
                break

            if done:
                break

        total_rewards.append(ep_reward)
        mean_min_offsets.append(min_offset)

    env.close()

    avg_reward = float(np.mean(total_rewards))
    drop_rate = float(drops) / float(N_EPISODES_PER_SETTING)
    avg_min_offset = float(np.mean(mean_min_offsets))

    return {
        "kp": kp,
        "kd": kd,
        "max_inc": max_inc,
        "avg_reward": avg_reward,
        "drop_rate": drop_rate,
        "avg_min_offset": avg_min_offset,
        "n_episodes": N_EPISODES_PER_SETTING
    }


def worker_task(args):
    params, seed = args
    try:
        return evaluate_setting(params, seed)
    except Exception as e:
        # Return failure record
        return {
            "kp": params[0],
            "kd": params[1],
            "max_inc": params[2],
            "avg_reward": -1e9,
            "drop_rate": 1.0,
            "avg_min_offset": 1e9,
            "error": str(e)
        }


def main():
    # Build grid of parameter combos
    param_grid = list(itertools.product(KP_VALUES, KD_VALUES, MAX_INC_VALUES))
    print(f"Evaluating {len(param_grid)} parameter combinations using {NUM_WORKERS} workers...")

    # Create seeds for reproducibility
    seeds = [int(time.time()) + i for i in range(len(param_grid))]

    tasks = [(param_grid[i], seeds[i % len(seeds)]) for i in range(len(param_grid))]

    # Parallel evaluation
    results: List[Dict[str, Any]] = []
    if NUM_WORKERS > 1:
        with mp.Pool(NUM_WORKERS) as pool:
            for res in pool.imap_unordered(worker_task, tasks):
                results.append(res)
                print(f"Finished: kp={res['kp']} kd={res['kd']} max_inc={res['max_inc']} "
                      f"drop_rate={res['drop_rate']:.3f} avg_reward={res['avg_reward']:.2f}")
    else:
        for task in tasks:
            res = worker_task(task)
            results.append(res)
            print(f"Finished: kp={res['kp']} kd={res['kd']} max_inc={res['max_inc']} "
                  f"drop_rate={res['drop_rate']:.3f} avg_reward={res['avg_reward']:.2f}")

    # Save to CSV
    keys = ["kp", "kd", "max_inc", "avg_reward", "drop_rate", "avg_min_offset", "n_episodes"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, None) for k in keys}
            writer.writerow(row)
    print(f"Saved results to {OUTPUT_CSV}")

    # Print best candidates sorted by (lowest drop rate, highest avg_reward)
    valid = [r for r in results if "error" not in r]
    sorted_by = sorted(valid, key=lambda r: (r["drop_rate"], -r["avg_reward"], r["avg_min_offset"]))
    print("\nTop 8 parameter sets (sorted by drop rate then avg_reward):")
    for r in sorted_by[:8]:
        print(r)

    print("\nDone.")


if __name__ == "__main__":
    main()