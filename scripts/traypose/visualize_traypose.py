#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

# -------------------------------
# Ensure project root is in sys.path
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now we can import from envs.traypose
from envs.traypose.traypose_env import TrayPoseEnv

def main():
    # Initialize environment
    env = TrayPoseEnv(model_path="assets/panda_tray/panda_tray_cylinder.xml")

    # Viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for episode in range(3):  # run 3 demo episodes
            print(f"\n=== Episode {episode+1} ===")

            # Reset env
            obs = env.reset()

            # Show start pose for 2 seconds but advance physics so things settle
            pause_seconds = 2.0
            n_pause_steps = int(pause_seconds / env.control_dt)
            print(f"Showing start pose for {pause_seconds}s (advancing physics {n_pause_steps} control steps)...")
            for _ in range(n_pause_steps):
                # step with zero action to let dynamics settle (also advances env.t)
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
                mujoco.mj_forward(env.model, env.data)
                viewer.sync()
                time.sleep(env.control_dt)

            # Warm-up: run a few extra zero-action steps (optional)
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
                viewer.sync()
                time.sleep(env.control_dt)

            done = False
            step_count = 0

            prev_action = np.zeros(7, dtype=np.float32)
            action_scale = 0.02   # scale on top of env.action_space (keeps actions tiny)
            smoothing_alpha = 0.2  # lower alpha = smoother (0..1)

            while not done and viewer.is_running():
                # sample a tiny random action in the allowed range then scale it further
                raw = env.action_space.sample().astype(np.float32) * action_scale

                # apply exponential smoothing to avoid abrupt jumps
                action = smoothing_alpha * raw + (1.0 - smoothing_alpha) * prev_action
                prev_action = action.copy()

                obs, reward, done, info = env.step(action)

                # debug - occasionally print max delta_q applied (approx)
                if step_count % 50 == 0:
                    print(f"Step {step_count}, reward {reward:.3f}, max|delta_q| {np.max(np.abs(action)):.6f}")

                viewer.sync()
                step_count += 1
                time.sleep(env.control_dt)  # consistent control-timestep rendering

            print(f"Episode {episode+1} finished after {step_count} steps")

    env.close()

if __name__ == "__main__":
    main()