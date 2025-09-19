import time
import numpy as np
import os
import sys
import mujoco
import mujoco.viewer

# -------------------------------
# Ensure project root is in sys.path
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

# Now we can import from envs.traypose
from envs.traypose.traypose_env import TrayPoseEnv


def main():
    # Initialize environment
    env = TrayPoseEnv(model_path="assets/panda_tray/panda_tray_cylinder.xml")
    # Reset env
    obs = env.reset()
    print("Initial observation:", obs)

    # Launch Mujoco viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for episode in range(3):  # run 3 demo episodes
            obs = env.reset()
            done = False
            step_count = 0

            while not done and viewer.is_running():
                # Sample a random action (demo only)
                action = env.action_space.sample()

                obs, reward, done, info = env.step(action)

                # Forward simulate MuJoCo (useful if modifying qpos/qvel directly)
                mujoco.mj_forward(env.model, env.data)

                # Print some debug info
                if step_count % 10 == 0:
                    print(f"Step {step_count}, Obs {obs}, Reward {reward:.3f}")

                # Step sync with viewer (render one frame)
                viewer.sync()

                step_count += 1
                time.sleep(0.02)  # ~50 FPS sim speed

            print(f"Episode {episode} finished after {step_count} steps")

    env.close()


if __name__ == "__main__":
    main()