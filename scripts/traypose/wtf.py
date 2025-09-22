#!/usr/bin/env python3
import os
import sys
import time
import mujoco
import mujoco.viewer

# Make project root importable (two levels up from scripts/traypose)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
print("Working directory set to:", os.getcwd())

# Now import your environment
from envs.traypose.traypose_env import TrayPoseEnv

def visualize_start_pose():
    env = TrayPoseEnv(model_path="assets/panda_tray/panda_tray_cylinder.xml")

    # Reset to start pose (this sets robot joints and cylinder)
    obs = env.reset()

    # Launch MuJoCo viewer to inspect start pose
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("Viewer launched. Close the window to exit.")
        # keep window open until user closes it
        while viewer.is_running():
            # keep MuJoCo state current (no simulation steps)
            mujoco.mj_forward(env.model, env.data)

            # Print tray base position each frame
            tray_pos = env.data.xpos[env.tray_body_id]
            print(f"Tray base position: {tray_pos}")
            
            viewer.sync()
            time.sleep(0.02)

    env.close()

if __name__ == "__main__":
    visualize_start_pose()